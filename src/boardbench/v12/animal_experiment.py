"""Bounded same-network PPO, frozen BC, and BC-to-PPO continuation pilot."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from .encoding import encode, index
from .evaluation import paired
from .experiment import Experiment, family_gate, fixed_policy_gate, rows


def average_rows(groups):
    if len(groups) != 3:
        raise ValueError('three training repetitions required')
    reference = [r['seed'] for r in groups[0]]
    if any([r['seed'] for r in group] != reference for group in groups):
        raise ValueError('training repetitions must use identical ordered seeds')
    return [{**groups[0][i], 'score': statistics.mean(g[i]['score'] for g in groups),
             'failed': any(g[i]['failed'] for g in groups),
             'components': {k: statistics.mean(g[i]['components'][k] for g in groups)
                            for k in groups[0][i]['components']}} for i in range(len(reference))]


def animal_gate(groups, incumbents, old_ppo, frozen_bc=None):
    gate = family_gate(groups, incumbents)
    if len(groups) != 3 or any(r['failed'] for g in groups for r in g):
        return gate
    candidate = average_rows(groups)
    fields = ('animals', 'animal_placements', 'completed_cards')
    differences = {k: statistics.mean(r['components'][k] for r in candidate) -
                      statistics.mean(r['components'][k] for r in old_ppo) for k in fields}
    gate['animal_differences_vs_old_ppo'] = differences
    gate['passed'] = gate['passed'] and all(v > 0 for v in differences.values())
    if frozen_bc is not None:
        baseline = average_rows(frozen_bc)
        contribution = paired(candidate, baseline)
        animal_change = statistics.mean(a['components']['animals'] - b['components']['animals']
                                        for a, b in zip(candidate, baseline))
        valid = not any(r['failed'] for r in baseline)
        gate['rl_contribution_vs_frozen_bc'] = {**contribution, 'animal_score_change': animal_change,
                                               'valid': valid}
        gate['passed'] = gate['passed'] and valid and contribution['normal_95_interval'][0] > 0 and animal_change >= 0
    return gate


def nominated_gate(candidate, incumbents, old_ppo, frozen_bc=None):
    compare = dict(incumbents)
    if frozen_bc is not None:
        compare['frozen_bc'] = frozen_bc
    gate = fixed_policy_gate(candidate, compare)
    gate['method'] = 'fixed nominated learning artifact'
    differences = {k: statistics.mean(r['components'][k] for r in candidate) -
                      statistics.mean(r['components'][k] for r in old_ppo)
                   for k in ('animals', 'animal_placements', 'completed_cards')}
    gate['animal_differences_vs_old_ppo'] = differences
    gate['passed'] = gate['passed'] and all(v > 0 for v in differences.values())
    if frozen_bc is not None:
        change = statistics.mean(r['components']['animals'] for r in candidate) - statistics.mean(r['components']['animals'] for r in frozen_bc)
        gate['animal_difference_vs_frozen_bc'] = change
        gate['passed'] = gate['passed'] and change >= 0
    return gate


class AnimalExperiment(Experiment):
    def __init__(self, plan, root, repo, device):
        super().__init__(plan, root, repo, device)
        if len(plan['training_seeds']) != 3 or len(set(plan['training_seeds'])) != 3:
            raise ValueError('exactly three distinct training seeds required')
        self.previous = (self.repo / plan['continuation_from']).resolve()
        self.teacher = self.root / 'teachers/dataset.json'
        self.curve = []

    def train_route(self, route, seed, name, seconds, *, resume=None, initialize=None):
        config = self.config('bc' if route == 'bc' else 'ppo', seed)
        config['target_kl'] = .02
        if route == 'bc':
            config['teacher_sha256'] = digest(self.teacher)
        if route == 'bc_ppo':
            config['learning_rate'] = self.plan['bc_ppo_learning_rate']
        path = self.root / 'training' / name
        cfg = self.root / 'configs' / (name + '.json')
        write_json(cfg, config)
        arguments = ['--config', cfg, '--out', path, '--seconds', seconds, '--device', self.device]
        if resume:
            arguments += ['--resume', resume]
        if route == 'bc':
            arguments += ['--teacher', self.teacher]
        if initialize:
            arguments += ['--initialize-bc', initialize['weights'], '--initialize-sha256', initialize['weights_sha256']]
        self.run('training', name, arguments, seconds + 60)
        weights = path / 'final.resume.weights.pt'
        spec = {'id': name, 'method': 'imitation' if route == 'bc' else 'rl',
                'label': f'{route.upper()} · {seed}', 'route': route, 'training_seed': seed,
                'solver': {'id': 'compact', 'params': {**config['network'], 'device': 'cpu'}},
                'weights': str(weights), 'weights_sha256': digest(weights),
                'training_config_sha256': digest(cfg), 'training_summary': read_json(path / 'summary.json')}
        return spec, path / 'final.resume.pt'

    def reuse_teacher(self):
        self.ledger.update('teacher_reencode')
        path = self.previous / 'teachers/semantic_trajectories.jsonl'
        allowed = set(self.splits['train'])
        dataset = []
        source_games = []
        with path.open() as stream:
            for line in stream:
                game = json.loads(line)
                if game['seed'] not in allowed or game['seed'] in source_games:
                    raise ValueError('teacher seeds must be unique and training-only')
                source_games.append(game['seed'])
                for example in game['examples']:
                    obs = example['observation']
                    data = encode(obs)
                    dataset.append({'features': data['features'], 'mask': data['action_mask'],
                                    'action': index(obs, example['action'], 'slots240_v1'), 'seed': game['seed']})
        write_json(self.teacher, dataset)
        write_json(self.root / 'teachers/provenance.json', {
            'source': str(path), 'source_sha256': digest(path), 'dataset_sha256': digest(self.teacher),
            'games': source_games, 'examples': len(dataset), 'source_id': source_id(),
            'original_generation_charge': 'included in prior allocation; reencoding included in continuation',
            'inference_teacher_calls': 0})
        return path

    def finish(self, **report):
        write_json(self.root / 'report.json', {'source_id': source_id(), 'capability_verified': False, **report})
        self.ledger.update('completed', report=str(self.root / 'report.json'), **report)

    def execute(self):
        prior_plan = read_json(self.previous / 'plan.json')
        for key in ('train_seed_start', 'train_seed_count', 'teacher_seed_start', 'teacher_games',
                    'dev_seed_start', 'dev_games', 'validation_seed_start', 'validation_games', 'test_seed_start', 'test_games'):
            if self.plan[key] != prior_plan[key]:
                raise ValueError(f'continuation split changed: {key}')
        budget = read_json(self.previous / 'online_budget.json')['primary_seconds']
        write_json(self.root / 'online_budget.json', {'primary_seconds': budget,
                   'frozen_from': str(self.previous / 'online_budget.json'), 'reprofiled': False})
        legacy_rl = max((p for p in self.legacy['harmonies'] if p['method'] == 'rl'), key=lambda p: p.get('validation_mean') or -1)
        baselines = [p for p in self.legacy['harmonies'] if p['method'] in ('random', 'heuristic', 'search')] + [legacy_rl]
        baselines += [{'id': 'time_aware_search', 'method': 'search', 'label': '限时搜索',
                       'solver': {'id': 'timed_search', 'params': {'max_simulation_steps': 384}}}]
        incumbent_rows = {}
        for spec in baselines:
            previous = self.previous / 'evaluations' / f'validation_harmonies_{spec["id"]}'
            config = read_json(previous / 'config.json')
            if config['seeds'] != self.validation or config['seconds'] != budget or config['policy']['solver'] != spec['solver']:
                raise ValueError('reused baseline protocol/config mismatch')
            if spec['method'] != 'random':
                incumbent_rows[spec['id']] = rows(previous)
        write_json(self.root / 'baseline_reuse.json', {'root': str(self.previous), 'paired_seeds': self.validation,
                   'seconds': budget, 'baseline_ids': list(incumbent_rows),
                   'boundary': 'historical validation under unchanged policy/protocol; not fresh repetitions'})
        trajectories = self.reuse_teacher()
        diag_cfg = self.root / 'configs/animal_diagnosis.json'
        write_json(diag_cfg, self.config('ppo', self.plan['training_seeds'][0]))
        self.run('animal_diagnostics', 'animal_diagnosis', ['--trajectories', trajectories, '--config', diag_cfg,
                 '--out', self.root / 'diagnosis', '--device', self.device,
                 '--prepare-seconds', self.plan['diagnosis_prepare_seconds'],
                 '--seconds-per-method', self.plan['diagnosis_method_seconds']],
                 self.plan['diagnosis_prepare_seconds'] + 6*self.plan['diagnosis_method_seconds'] + 60)
        diagnosis = read_json(self.root / 'diagnosis/report.json')
        placement = diagnosis['levels']['placement']
        if placement.get('bc', {}).get('status') != 'fit_passed':
            self.finish(status='diagnosis_incomplete', promoted_family=None, policy_unchanged=True,
                        reason='BC could not fit the immediate animal task; inspect representation before scaling', diagnosis=diagnosis)
            return
        extend_ppo = placement.get('ppo', {}).get('status') == 'fit_passed'
        candidates, resumes, best, frozen_bc, controls = {}, {}, {}, {}, []
        for seed in self.plan['training_seeds']:
            name = f'bc_{seed}_frozen'
            spec, _ = self.train_route('bc', seed, name, self.plan['bc_seconds'])
            frozen_bc[seed] = spec
            report = self.evaluate(spec, 'dev', self.dev, budget)
            self.curve.append({'route': 'bc', 'seed': seed, 'spec': spec, 'summary': report['summary']})
            control = deepcopy(spec)
            weights = self.root / 'training' / name / 'initial.resume.weights.pt'
            control.update(id=f'untrained_{seed}', method='untrained_control', label=f'未训练 · {seed}',
                           weights=str(weights), weights_sha256=digest(weights), training_summary=None)
            report = self.evaluate(control, 'dev', self.dev, budget)
            controls.append(control)
            self.curve.append({'route': 'untrained', 'seed': seed, 'spec': control, 'summary': report['summary']})
            write_json(self.root / 'learning_curve.json', self.curve)
        candidates['bc'] = list(frozen_bc.values())
        previous_seconds = 0
        for stage, cumulative in enumerate(self.plan['rl_cumulative_seconds']):
            delta = cumulative - previous_seconds
            routes = ('ppo', 'bc_ppo') if stage == 0 or extend_ppo else ('bc_ppo',)
            if self.ledger.remaining() < self.reserve + len(routes)*3*(delta + 150):
                break
            for route in routes:
                for seed in self.plan['training_seeds']:
                    name = f'{route}_{seed}_stage{stage}'
                    key = (route, seed)
                    spec, checkpoint = self.train_route(route, seed, name, delta, resume=resumes.get(key),
                            initialize=frozen_bc[seed] if route == 'bc_ppo' and key not in resumes else None)
                    resumes[key] = checkpoint
                    report = self.evaluate(spec, 'dev', self.dev, budget)
                    if key not in best or report['summary']['score_mean'] > best[key]['summary']['score_mean']:
                        best[key] = {'spec': spec, 'summary': report['summary']}
                    self.curve.append({'route': route, 'seed': seed, 'stage': stage, 'spec': spec,
                                       'cumulative_ppo_seconds': cumulative, 'summary': report['summary']})
                    write_json(self.root / 'learning_curve.json', self.curve)
                candidates[route] = [best[route, seed]['spec'] for seed in self.plan['training_seeds']]
            previous_seconds = cumulative
        # Freeze selected stages on dev; never choose a stage using final test.
        validation, grouped, gates = {}, {}, {}
        for route, specs in candidates.items():
            grouped[route] = []
            for spec in specs:
                report = self.evaluate(spec, 'validation', self.validation, budget, final=True)
                spec['summary'] = report['summary']
                validation[spec['id']] = report
                grouped[route].append(report['rows'])
            gates[route] = animal_gate(grouped[route], incumbent_rows, incumbent_rows[legacy_rl['id']],
                                      grouped.get('bc') if route == 'bc_ppo' else None)
        # Also require the actual deployed repetition, not just its family average, to pass.
        winners, nominated = [], {}
        for route, specs in candidates.items():
            chosen = max(specs, key=lambda s: s['summary']['score_mean'])
            bc_rows = validation[frozen_bc[chosen['training_seed']]['id']]['rows'] if route == 'bc_ppo' else None
            single = nominated_gate(validation[chosen['id']]['rows'], incumbent_rows,
                                    incumbent_rows[legacy_rl['id']], bc_rows)
            gates[route]['nominated_artifact_gate'] = single
            gates[route]['passed'] = gates[route]['passed'] and single['passed']
            nominated[route] = chosen
            if gates[route]['passed']:
                winners.append(route)
        winner = max(winners, key=lambda r: nominated[r]['summary']['score_mean']) if winners else None
        selected = [p for group in candidates.values() for p in group] + [legacy_rl, baselines[-1]]
        freeze = {'candidates': selected, 'gates': gates, 'promoted_family': winner,
                  'test_used_for_selection': False, 'test_seeds': self.test, 'primary_seconds': budget,
                  'source_id': source_id(), 'teacher_reuse': str(self.root / 'teachers/provenance.json')}
        write_json(self.root / 'frozen_selection.json', freeze)
        tests = {}
        for spec in selected:
            tests[spec['id']] = self.evaluate(spec, 'test', self.test, budget, final=True)['summary']
            write_json(self.root / 'test_results.json', tests)
        valid = all(not value['failures'] for value in tests.values())
        release = None
        if winner and valid:
            for task in ('micro_tiles', 'take_it_easy'):
                incumbent = max(self.legacy[task], key=lambda s: s.get('validation_mean') or -1)
                report = self.evaluate(incumbent, 'compatibility', self.test[:8], budget, task=task, final=True)
                valid = valid and not report['summary']['failures']
            if valid:
                policies = deepcopy(self.legacy)
                policies['harmonies'] += candidates[winner]
                release = self.publish(policies, {**self.defaults, 'harmonies': nominated[winner]['id']},
                                       'promoted_release', 'promotion', gates[winner])
        self.finish(status='completed', promoted_family=winner if release else None,
                    policy_unchanged=release is None, active_release=release, gates=gates,
                    test_execution_valid=valid, diagnosis=diagnosis,
                    budget_boundary='continuation; original wall deadline and all prior charges retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--root', required=True)
    parser.add_argument('--repo', required=True); parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    experiment = AnimalExperiment(read_json(args.plan), args.root, args.repo, args.device)
    try:
        experiment.execute()
    except Exception as exc:
        experiment.ledger.update('stopped', error=f'{type(exc).__name__}: {exc}', partial_results_retained=True)
        raise


if __name__ == '__main__':
    main()
