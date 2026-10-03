"""Predeclared four-hour first pilot; training/selection/testing share one ledger."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import sys

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.benchmark import percentile
from boardbench.provenance import validate_splits
from .budget import Ledger
from .deployment import legacy_policies
from .evaluation import paired


def seeds(plan, name):
    return list(range(plan[f'{name}_seed_start'], plan[f'{name}_seed_start'] + plan[f'{name}_games']))


def rows(path):
    return [json.loads(line) for line in (Path(path) / 'episodes.jsonl').read_text().splitlines()]


def family_gate(repetitions, incumbents):
    if len(repetitions) != 3 or any(r['failed'] for group in repetitions for r in group):
        return {'passed': False, 'reason': 'three valid training repetitions required'}
    if not incumbents or any(row['failed'] for group in incumbents.values() for row in group):
        return {'passed': False, 'reason': 'incumbent validation must also be valid'}
    reference = [r['seed'] for r in repetitions[0]]
    if any([r['seed'] for r in group] != reference for group in repetitions):
        raise ValueError('training repetitions must use identical ordered evaluation seeds')
    average = [{**repetitions[0][i], 'score': statistics.mean(group[i]['score'] for group in repetitions)}
               for i in range(len(reference))]
    comparisons = {name: paired(average, group) for name, group in incumbents.items()}
    return {'passed': bool(comparisons) and all(v['normal_95_interval'][0] > 0 for v in comparisons.values()),
            'comparisons': comparisons, 'training_seeds': 3, 'selection_split': 'validation',
            'test_used': False, 'uncertainty': 'conditional on these three artifacts; not population training-seed CI'}


def fixed_policy_gate(candidate, incumbents):
    if not incumbents or any(row['failed'] for group in [candidate, *incumbents.values()] for row in group):
        return {'passed': False, 'reason': 'candidate and incumbent validation must be valid'}
    comparisons = {name: paired(candidate, group) for name, group in incumbents.items()}
    return {'passed': all(value['normal_95_interval'][0] > 0 for value in comparisons.values()),
            'comparisons': comparisons, 'training_seeds': None, 'method': 'fixed search configuration',
            'selection_split': 'validation', 'test_used': False}


class Experiment:
    def __init__(self, plan, root, repo, device):
        self.plan, self.root, self.repo, self.device = plan, Path(root).resolve(), Path(repo).resolve(), device
        self.ledger = Ledger(self.root, plan['budget_seconds'], status_path=self.repo / 'outputs/v12/latest_status.json')
        self.reserve = plan['final_reserve_seconds']
        self.counter = 0
        self.teacher_seeds = seeds(plan, 'teacher')
        self.dev, self.validation, self.test = (seeds(plan, n) for n in ('dev', 'validation', 'test'))
        self.splits = {'train': list(range(plan['train_seed_start'], plan['train_seed_start'] + plan['train_seed_count'])) + self.teacher_seeds,
                       'validation': self.validation + self.dev, 'test': self.test}
        validate_splits(self.splits)
        write_json(self.root / 'splits.json', self.splits)
        self.legacy = legacy_policies(self.repo / plan['legacy_root'], self.dev + self.validation + self.test)
        self.pointer = self.repo / plan['deployment_pointer']
        self.defaults = {task: max(items, key=lambda p: p.get('validation_mean') or -1)['id']
                         for task, items in self.legacy.items()}

    def run(self, module, name, arguments, maximum, reserve=None):
        return self.ledger.run(name, [sys.executable, '-m', 'boardbench.v12.' + module, *map(str, arguments)],
                               maximum, self.reserve if reserve is None else reserve)

    def evaluate(self, spec, stage, which, seconds, task='harmonies', final=False):
        name = f'{stage}_{task}_{spec["id"]}'
        path = self.root / 'evaluations' / name
        config = self.root / 'configs' / (name + '.json')
        write_json(config, {'task': task, 'policy': spec, 'seeds': which, 'seconds': seconds,
                            'workers': self.plan['workers']})
        self.run('evaluation', name, ['--config', config, '--out', path],
                 60 + math.ceil(len(which) / self.plan['workers']) * (seconds + 30), reserve=120 if final else None)
        return {'summary': read_json(path / 'summary.json'), 'rows': rows(path), 'path': str(path)}

    def config(self, algorithm, seed, network=None):
        return {'algorithm': algorithm, 'training_seed': seed, 'splits': self.splits,
                'network': network or {'hidden': 128, 'action_encoding': 'slots240_v1',
                                       'observation_encoding': 'cards_v1', 'allow_stop': True},
                'threads': 1, 'batch_episodes': 16, 'epochs': 4, 'checkpoint_batches': 16,
                'learning_rate': .0003, 'reward_scale': 150., 'entropy_coef': .01}

    def train(self, config, name, seconds, resume=None):
        path = self.root / 'training' / name
        cfg = self.root / 'configs' / (name + '.json')
        write_json(cfg, config)
        arguments = ['--config', cfg, '--out', path, '--seconds', seconds, '--device', self.device]
        if resume:
            arguments += ['--resume', resume]
        if config['algorithm'] == 'bc':
            arguments += ['--teacher', self.root / 'teachers/dataset.json']
        self.run('training', name, arguments, seconds + 60)
        weights = path / 'final.resume.weights.pt'
        spec = {'id': name, 'method': 'rl' if config['algorithm'] == 'ppo' else 'imitation',
                'label': f"{config['algorithm'].upper()} · {config['training_seed']}",
                'solver': {'id': 'compact', 'params': {**config['network'], 'device': 'cpu'}},
                'weights': str(weights), 'weights_sha256': digest(weights),
                'training_seed': config['training_seed'], 'training_config_sha256': digest(cfg),
                'training_summary': read_json(path / 'summary.json')}
        return spec, path / 'final.resume.pt'

    def publish(self, policies, defaults, name, kind, gate=None):
        cfg = self.root / 'configs' / (name + '.json')
        release = self.root / name
        write_json(cfg, {'policies': policies, 'defaults': defaults, 'release': str(release),
                         'pointer': str(self.pointer), 'kind': kind, 'gate': gate})
        self.run('deployment', name, ['--config', cfg], 180, reserve=0)
        return str(release)

    def execute(self):
        smoke = self.config('ppo', self.plan['training_seeds'][0])
        smoke.update(batch_episodes=2, epochs=1)
        self.train(smoke, 'device_smoke', .01)
        # Explicit re-export under current source, not a source-check bypass.
        imported = self.publish(self.legacy, self.defaults, 'legacy_import', 'legacy_import')
        legacy_rl = max((p for p in self.legacy['harmonies'] if p['method'] == 'rl'),
                        key=lambda p: p.get('validation_mean') or -1)
        profile = self.evaluate(legacy_rl, 'profile', self.dev[:self.plan['profile_games']], 30.)
        if profile['summary']['failures'] or any(r['reason'] != 'natural' for r in profile['rows']):
            raise RuntimeError('incumbent profiling failed; do not choose a favorable deadline post hoc')
        budget = max(1, min(8, math.ceil(10 * percentile([r['online_seconds'] for r in profile['rows']], .95))))
        write_json(self.root / 'online_budget.json', {'primary_seconds': budget, 'rule': self.plan['primary_budget_rule'],
                   'profile_path': profile['path'], 'training_or_test_used': False, 'inference_device': 'cpu', 'threads': 1})
        baselines = [p for p in self.legacy['harmonies'] if p['method'] in ('random', 'heuristic', 'search')] + [legacy_rl]
        baselines += [{'id': 'time_aware_search', 'method': 'search', 'label': '限时搜索',
                       'solver': {'id': 'timed_search', 'params': {'max_simulation_steps': 384}}}]
        baseline_val = {p['id']: self.evaluate(p, 'validation', self.validation, budget) for p in baselines}
        # Teacher and exact animal endgame diagnosis are inside the same budget.
        teacher_cfg = self.root / 'configs/teacher.json'
        write_json(teacher_cfg, {'seeds': self.teacher_seeds, 'workers': self.plan['workers']})
        self.run('teacher', 'teachers', ['--config', teacher_cfg, '--out', self.root / 'teachers'],
                 self.plan['teacher_limit_seconds'])
        diag_cfg = self.root / 'configs/diagnosis.json'
        write_json(diag_cfg, self.config('bc', self.plan['training_seeds'][0]))
        self.run('diagnostics', 'endgame_diagnosis', ['--config', diag_cfg, '--endgames', self.root / 'teachers/endgames.json',
                  '--out', self.root / 'endgame_diagnosis.json', '--device', self.device], 240)
        diagnosis = read_json(self.root / 'endgame_diagnosis.json')
        candidates, learning_curve, ablations, controls = {}, [], [], []
        if diagnosis['status'] == 'passed':
            for i, (action, obs) in enumerate([('legacy4564_v1', 'legacy_v1'), ('no_rotation884_v1', 'legacy_v1'),
                    ('no_rotation884_v1', 'slots_v1'), ('slots240_v1', 'slots_v1'), ('slots240_v1', 'cards_v1')]):
                cfg = self.config('ppo', self.plan['training_seeds'][0],
                                  {'hidden': 128, 'action_encoding': action, 'observation_encoding': obs, 'allow_stop': True})
                spec, _ = self.train(cfg, f'ablation_{i}', self.plan['ablation_seconds'])
                report = self.evaluate(spec, 'ablation_dev', self.dev[:8], budget)
                ablations.append({'action': action, 'observation': obs, 'policy': spec, 'evaluation': report['summary']})
            for algorithm in ('ppo', 'bc'):
                current, resumes, best = {}, {}, {}
                previous_mean, plateau = None, 0
                previous_seconds = 0
                for stage, cumulative in enumerate(self.plan[f'{algorithm}_cumulative_seconds']):
                    # Complete repetitions as a group or do not start this stage.
                    delta = cumulative - previous_seconds
                    if self.ledger.remaining() < self.reserve + 3 * (delta + 140):
                        learning_curve.append({'algorithm': algorithm, 'stage': stage, 'stop': 'budget reserve'})
                        break
                    for seed in self.plan['training_seeds']:
                        name = f'{algorithm}_{seed}_stage{stage}'
                        spec, checkpoint = self.train(self.config(algorithm, seed), name, delta, resumes.get(seed))
                        if algorithm == 'ppo' and stage == 0:
                            control = deepcopy(spec)
                            weights = checkpoint.parent / 'initial.resume.weights.pt'
                            control.update(id=f'untrained_{seed}', method='untrained_control', label=f'未训练 · {seed}',
                                           weights=str(weights), weights_sha256=digest(weights), training_summary=None)
                            initial = self.evaluate(control, 'initial_dev', self.dev, budget)
                            learning_curve.append({'algorithm': algorithm, 'seed': seed, 'stage': 'initial',
                                                   'cumulative_target_seconds': 0, 'summary': initial['summary'], 'spec': control})
                            controls.append(control)
                        resumes[seed] = checkpoint
                        report = self.evaluate(spec, 'dev', self.dev, budget)
                        current[seed] = {'spec': spec, 'summary': report['summary']}
                        if seed not in best or report['summary']['score_mean'] > best[seed]['summary']['score_mean']:
                            best[seed] = current[seed]
                        learning_curve.append({'algorithm': algorithm, 'seed': seed, 'stage': stage,
                                               'cumulative_target_seconds': cumulative, **current[seed]})
                        write_json(self.root / 'learning_curve.json', learning_curve)
                    stage_mean = statistics.mean(v['summary']['score_mean'] for v in current.values())
                    plateau = plateau + 1 if previous_mean is not None and stage_mean - previous_mean < self.plan['plateau_min_improvement'] else 0
                    previous_mean, previous_seconds = stage_mean, cumulative
                    if plateau >= self.plan['plateau_stages']:
                        learning_curve.append({'algorithm': algorithm, 'stage': stage, 'stop': 'two-stage plateau; diagnosis required'})
                        break
                if len(best) == 3:
                    candidates[algorithm] = [best[s]['spec'] for s in self.plan['training_seeds']]
        write_json(self.root / 'ablations.json', ablations)
        write_json(self.root / 'learning_curve.json', learning_curve)
        # Now spend the reserved budget on validation, freeze, held-out test, restore and publication.
        validation, gates = {}, {}
        incumbent_rows = {p['id']: baseline_val[p['id']]['rows'] for p in baselines if p['method'] != 'random'}
        for algorithm, specs in candidates.items():
            groups = []
            for spec in specs:
                report = self.evaluate(spec, 'validation', self.validation, budget, final=True)
                validation[spec['id']] = report
                spec['summary'] = report['summary']
                groups.append(report['rows'])
            gates[algorithm] = family_gate(groups, incumbent_rows)
        search = next(p for p in baselines if p['id'] == 'time_aware_search')
        search['summary'] = baseline_val[search['id']]['summary']
        validation[search['id']] = baseline_val[search['id']]
        gates['search'] = fixed_policy_gate(baseline_val[search['id']]['rows'],
                                            {k: v for k, v in incumbent_rows.items() if k != search['id']})
        candidates['search'] = [search]
        winners = [a for a, gate in gates.items() if gate['passed']]
        chosen = max(winners, key=lambda a: statistics.mean(validation[p['id']]['summary']['score_mean'] for p in candidates[a])) if winners else None
        selected = list({p['id']: p for p in baselines + [p for group in candidates.values() for p in group] + controls}.values())
        freeze = {'policies': selected, 'gates': gates, 'promoted_family': chosen, 'source_id': source_id(),
                  'test_used_for_selection': False, 'test_seeds': self.test, 'primary_seconds': budget,
                  'extra_budgets': [budget / 2, budget * 2], 'human_target': None}
        write_json(self.root / 'frozen_selection.json', freeze)
        test_results = {}
        for spec in selected:
            test_results[spec['id']] = self.evaluate(spec, 'test', self.test, budget, final=True)['summary']
            write_json(self.root / 'test_results.json', test_results)
        # Fixed budget sensitivity, on development seeds only; never feed back into selection.
        sensitivity = []
        for spec in [legacy_rl] + [p for group in candidates.values() for p in group]:
            for factor in (.5, 2.):
                if self.ledger.remaining() < 240:
                    break
                report = self.evaluate(spec, f'sensitivity_{factor}', self.dev[:8], budget * factor, final=True)
                sensitivity.append({'policy': spec['id'], 'seconds': budget * factor, 'summary': report['summary']})
        write_json(self.root / 'budget_sensitivity.json', sensitivity)
        for task in ('micro_tiles', 'take_it_easy'):
            incumbent = max(self.legacy[task], key=lambda p: p.get('validation_mean') or -1)
            self.evaluate(incumbent, 'compatibility', self.test[:8], budget, task=task, final=True)
        release = imported
        # Test performance cannot change selection. Test execution/integrity failures veto release.
        valid_test = all(not report['failures'] for report in test_results.values())
        if chosen and valid_test:
            policies = deepcopy(self.legacy)
            policies['harmonies'] += candidates[chosen]
            defaults = {**self.defaults, 'harmonies': max(candidates[chosen], key=lambda p: p['summary']['score_mean'])['id']}
            release = self.publish(policies, defaults, 'promoted_release', 'promotion', gates[chosen])
        report = {'status': 'completed', 'promoted_family': chosen if chosen and valid_test else None,
                  'gates': gates, 'test_execution_valid': valid_test, 'active_release': release,
                  'diagnosis': diagnosis, 'human_target': None,
                  'capability_boundary': 'bounded first pilot; no claim of human parity or converged optimum',
                  'budget_seconds': self.plan['budget_seconds'], 'elapsed_seconds': self.plan['budget_seconds'] - self.ledger.remaining()}
        write_json(self.root / 'report.json', report)
        self.ledger.update('completed', report=str(self.root / 'report.json'), promoted_family=report['promoted_family'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--root', required=True)
    parser.add_argument('--repo', required=True); parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    experiment = Experiment(read_json(args.plan), args.root, args.repo, args.device)
    try:
        experiment.execute()
    except Exception as exc:
        experiment.ledger.update('stopped', error=f'{type(exc).__name__}: {exc}',
                                 policy_unchanged=True, partial_results_retained=True)
        raise


if __name__ == '__main__':
    main()
