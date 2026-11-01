"""Portable, immutable mixed opponents and bounded calibration matches.

Search receives the engine only inside the existing trusted public-belief bridge.
Fast policies receive public observations and legal actions. This module never
trains, changes game rules, or treats an incomplete game as a terminal loss.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import time

from .evaluate import archive_source, canonical_hash, file_hash
from .play import make_policy, policy_seed, write_json
from .reference_search import TerminalReference, ProposalPolicy, guarded_spec, with_guard
from .registry import game_class
from .simulation import step_legal
from .state_io import copy_data

FORMAT = 'boardbench_joint_pool_v1'


def inside(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if Path(relative).is_absolute() or not target.is_relative_to(root):
        raise ValueError('Artifact must stay inside the portable pool')
    return target


def freeze_spec(spec, root, artifacts, sources):
    """Copy bytes and explicitly capture recursively required prior models."""
    spec = copy_data(spec)
    if 'path' in spec:
        import torch
        source = Path(spec['path']).resolve()
        sha = file_hash(source)
        if spec.get('sha256') and spec['sha256'] != sha:
            raise ValueError('Original model hash mismatch')
        relative = f'models/{sha}/model.pt'
        target = inside(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
            target.chmod(0o444)
        if file_hash(target) != sha:
            raise ValueError('Frozen model changed')
        artifacts[relative] = sha
        sources[relative] = str(source)
        spec.update(path=relative, sha256=sha)
        if spec['kind'] == 'guided_rich':
            payload = torch.load(source, map_location='cpu', weights_only=True)
            spec['base_spec'] = freeze_spec(payload['metadata']['base_spec'], root, artifacts, sources)
            if payload.get('symbols') is None:
                vocab = source.parent / 'vocabulary.json'
                if not vocab.exists():
                    raise ValueError('Guided policy requires a frozen vocabulary')
                relative_vocab = str(Path(relative).parent / 'vocabulary.json')
                shutil.copyfile(vocab, inside(root, relative_vocab))
                artifacts[relative_vocab] = file_hash(vocab)
                inside(root, relative_vocab).chmod(0o444)
    if spec.get('kind') == 'terminal_search':
        spec['rl_spec'] = freeze_spec(spec['rl_spec'], root, artifacts, sources)
    return spec


def absolute_spec(spec, root):
    result = copy_data(spec)
    if 'path' in result:
        result['path'] = str(inside(root, result['path']))
    for key in ('base_spec', 'rl_spec'):
        if key in result:
            result[key] = absolute_spec(result[key], root)
    return result


def verify_pool(root, verify_source=False):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    expected = manifest.pop('manifest_sha256')
    if manifest['format'] != FORMAT or canonical_hash(manifest) != expected:
        raise ValueError('Pool manifest identity changed')
    manifest['manifest_sha256'] = expected
    for name, sha in manifest['artifacts'].items():
        if file_hash(inside(root, name)) != sha:
            raise ValueError('Pool artifact identity changed: ' + name)
    if verify_source:
        from .evaluate import code_identity
        if code_identity()['content_sha256'] != manifest['code']['content_sha256']:
            raise ValueError('Installed runtime differs from frozen source')
    for game, entry in manifest['games'].items():
        cls = game_class(game)
        if entry['config'] != cls.config_id or str(entry['rules_version']) != str(cls.rules_version):
            raise ValueError('Rules/config identity mismatch')
        for name, member in entry['members'].items():
            identity = dict(game=game, config=entry['config'], rules_version=entry['rules_version'], spec=member['spec'])
            if canonical_hash(identity) != member['policy_id']:
                raise ValueError('Policy identity changed: ' + name)
    return manifest


def prepare(root, references, candidates, legacy_pool):
    root = Path(root)
    if (root / 'manifest.json').exists():
        raise ValueError('Already frozen; use verify or a new directory')
    root.mkdir(parents=True, exist_ok=True)
    refs = json.loads(Path(references).read_text())['games']
    newer = json.loads(Path(candidates).read_text())['games']
    legacy = json.loads(Path(legacy_pool).read_text())
    artifacts, sources, games = {}, {}, {}
    for game, reference in sorted(refs.items()):
        if game == 'harmonies_score':
            continue  # Alternate objective of Harmonies, not a seventeenth game.
        cls = game_class(game)
        proposed = {
            'heuristic': guarded_spec(game, dict(kind='heuristic', weights=list(cls.default_weights))),
            'progress': dict(kind='proposal', progress=True, habitat=game == 'harmonies'),
            'legacy': reference['rl_spec'],
        }
        # Predeclared additions, not selected again on calibration outcomes.
        if game in ('century_spice_road', 'clever', 'patchwork', 'toy_battle'):
            proposed['candidate'] = guarded_spec(game, newer[game]['candidate'])
        for seconds in (30, 180, 900):
            proposed[f'search_{seconds}'] = dict(kind='terminal_search', seconds=float(seconds),
                rl_spec=reference['rl_spec'], weights=reference.get('heuristic_weights'),
                objective=reference['objective'], continuation=reference['continuation'],
                improvement_z=reference.get('improvement_z', 0.), width=6,
                max_simulation_steps=4000, budget_reserve=2., seed_xor=174)
        if game == 'harmonies':
            score_reference = refs['harmonies_score']
            proposed['search_score_900'] = dict(proposed['search_900'], objective='score',
                continuation=score_reference['continuation'],
                improvement_z=score_reference.get('improvement_z', 0.),
                rl_spec=score_reference['rl_spec'], weights=score_reference.get('heuristic_weights'))
        entry = dict(config=cls.config_id, rules_version=str(cls.rules_version),
                     rules_coverage=legacy['games'][game]['rules_coverage'], members={})
        for name, spec in proposed.items():
            spec = freeze_spec(spec, root, artifacts, sources)
            identity = dict(game=game, config=entry['config'], rules_version=entry['rules_version'], spec=spec)
            entry['members'][name] = dict(spec=spec, policy_id=canonical_hash(identity),
                role='challenge_candidate' if spec['kind'] == 'terminal_search' else 'fast_reference',
                origin=('v173_frozen_candidate' if name == 'candidate' else
                        'v172_search_configuration' if name.startswith('search') else 'existing_policy_code'))
        games[game] = entry
    code = archive_source(root)
    artifacts['source.tar.gz'] = code['archive_sha256']
    manifest = dict(format=FORMAT, version='1.7.4', created_unix=time.time(), games=games,
        artifacts=artifacts, code=code, excluded={'five_tribes': 'research-only unresolved rule branches'},
        protocol=dict(observation='public_only', first_player=0, paired_seats=True,
            random_streams='same initial seed; subsequent draws can diverge with actions',
            search_budget='cumulative per-game single CPU decision seconds',
            search_reproducibility='wall-clock search is hardware/load dependent; seeds do not imply identical actions',
            calibration_not_formal_test=True, incomplete_outcomes='unknown, never fabricated'))
    manifest['manifest_sha256'] = canonical_hash(manifest)
    write_json(root / 'provenance.json', dict(model_sources=sources, input_files={
        str(p): file_hash(p) for p in (references, candidates, legacy_pool)}))
    write_json(root / 'manifest.json', manifest)
    (root / 'manifest.json').chmod(0o444)
    return verify_pool(root)


def load_fast(game, spec, seed):
    cls = game_class(game)
    if spec['kind'] == 'guided_rich':
        from .guided_neural import load_model, GuidedPolicy
        model, encoder, metadata = load_model(spec['path'], game, expected_hash=spec['sha256'])
        metadata = dict(metadata, base_spec=spec['base_spec'])
        return with_guard(cls, GuidedPolicy(game, spec, seed, model, encoder, metadata))
    if spec['kind'] == 'proposal':
        habitat = None
        if spec.get('habitat'):
            from boardbench.v146.policy import Habitat
            habitat = Habitat()
        return with_guard(cls, ProposalPolicy(cls, progress=spec.get('progress', False), habitat=habitat))
    return make_policy(cls, spec, seed)


def load_opponent(root, game, name, seat, seed, manifest=None):
    """Trusted opponent loader. Fast objects expose decide(obs, legal); search
    exposes decide(engine) and must run inside the evaluator, never agent input.
    Call verify_pool once at the boundary; each model loader also checks hashes.
    """
    if seat not in (0, 1):
        raise ValueError('Expected one of two seats')
    manifest = manifest or verify_pool(root)
    spec = absolute_spec(manifest['games'][game]['members'][name]['spec'], root)
    if spec['kind'] != 'terminal_search':
        return load_fast(game, spec, policy_seed(seed, seat))
    options = {k: v for k, v in spec.items() if k not in ('kind', 'rl_spec', 'seed_xor')}
    return TerminalReference(game_class(game), spec['rl_spec'], seat=seat,
                             seed=policy_seed(seed, seat) ^ spec['seed_xor'], **options)


def match(root, job, manifest=None, trace_path=None):
    manifest = manifest or verify_pool(root)
    game, seed, names = job['game'], job['seed'], job['names']
    if len(names) != 2:
        raise ValueError('Exactly two opponents required')
    cls = game_class(game)
    env = cls(); env.reset(seed, first_player=0)
    began = time.monotonic(); cpu = time.process_time()
    policies = [load_opponent(root, game, name, seat, seed, manifest) for seat, name in enumerate(names)]
    specs = [manifest['games'][game]['members'][n]['spec'] for n in names]
    budgets = [s.get('seconds') if s['kind'] == 'terminal_search' else None for s in specs]
    cap = sum(x or 0 for x in budgets) + job.get('max_fast_seconds', 180.)
    policy_seconds, maxima, decisions = [0., 0.], [0., 0.], [0, 0]
    actions_count = [Counter(), Counter()]; trace = []; status = None
    while not env.outcome()['terminated']:
        if env.outcome().get('unresolved') or env.outcome().get('unresolved_rule'):
            status = 'unresolved_rule'; break
        if sum(decisions) >= job.get('max_decisions', 10000) or time.monotonic()-began >= cap:
            status = 'truncated'; break
        seat = env.current_player; obs = env.observe(seat); legal = env.legal_actions(seat)
        if not legal:
            status = 'unresolved_rule' if obs.get('blocked_reason') else 'no_legal_action'; break
        t = time.monotonic()
        action = policies[seat].decide(env) if budgets[seat] is not None else policies[seat].decide(obs, copy_data(legal))
        elapsed = time.monotonic()-t
        policy_seconds[seat] += elapsed; maxima[seat] = max(maxima[seat], elapsed)
        if action not in legal:
            raise ValueError('Illegal policy action')
        if trace_path:
            trace.append(dict(player=seat, observation=copy_data(obs), legal_actions=copy_data(legal),
                              action=copy_data(action)))
        actions_count[seat][str(action.get('type', 'unknown'))] += 1
        step_legal(env, legal, legal.index(action)); decisions[seat] += 1
    outcome = env.outcome(); status = status or ('complete' if outcome['terminated'] else 'incomplete')
    search = [dict(spent=p.spent, rounds=p.total_rounds, simulation_steps=p.total_steps,
                   fallbacks=p.fallbacks) if b is not None else None for p, b in zip(policies, budgets)]
    if any(b is not None and s['spent'] > b for b, s in zip(budgets, search)):
        status = 'budget_exceeded'
    result = dict(job=job, pool_sha256=manifest['manifest_sha256'], game=game, seed=seed,
        names=names, policy_ids=[manifest['games'][game]['members'][n]['policy_id'] for n in names],
        config=cls.config_id, rules_version=str(cls.rules_version), first_player=0,
        status=status, outcome=outcome, score_breakdown=env.score_breakdown(),
        seat_decisions=decisions, action_counts=[dict(x) for x in actions_count],
        policy_seconds=policy_seconds, policy_max_seconds=maxima, search=search,
        cpu_seconds=time.process_time()-cpu, wall_seconds=time.monotonic()-began,
        ended_unix=time.time())
    if trace_path:
        trace_path = Path(trace_path); trace_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = trace_path.with_suffix('.part')
        with gzip.open(temporary, 'wt') as f:
            json.dump(dict(result=result, trajectory=trace), f, allow_nan=False)
        temporary.replace(trace_path)
        result.update(trajectory=trace_path.name, trajectory_sha256=file_hash(trace_path))
    return result


def make_schedule(manifest, fast_seeds=64, search_seeds=16, seed_start=174010000):
    if any(type(x) is not int or x < 1 for x in (fast_seeds, search_seeds)):
        raise ValueError('Positive integer seed counts required')
    if type(seed_start) is not int or seed_start < 0:
        raise ValueError('Nonnegative integer starting seed required')
    jobs = []
    def add(game, a, b, count, panel):
        for k in range(count):
            for swap in (0, 1):
                names = [a, b] if not swap else [b, a]
                job = dict(game=game, names=names, seed=seed_start+k, panel=panel,
                           pair=[a, b], swap=swap, max_fast_seconds=180., max_decisions=10000)
                job['id'] = f'{game}__{a}__{b}__{k:04d}__{swap}'
                jobs.append(job)
    for game, entry in manifest['games'].items():
        fast = [n for n, v in entry['members'].items() if v['spec']['kind'] != 'terminal_search']
        for a, b in itertools.combinations(fast, 2):
            add(game, a, b, fast_seeds, 'fast_round_robin')
        add(game, 'legacy', 'legacy', fast_seeds, 'style_anchor_selfplay')
        for name in entry['members']:
            if name.startswith('search'):
                add(game, name, 'legacy', search_seeds*2 if name == 'search_30' else search_seeds,
                    'common_anchor')
        if 'candidate' in fast:
            add(game, 'search_900', 'candidate', max(1, search_seeds//2), 'candidate_challenge')
    for job in jobs:
        job['record'] = job['seed'] < seed_start+2 and job['swap'] == 0
    return dict(format='boardbench_joint_calibration_schedule_v1', pool_sha256=manifest['manifest_sha256'],
                seed_start=seed_start, fast_seeds=fast_seeds, search_seeds=search_seeds,
                jobs=jobs, created_unix=time.time())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('freeze'); f.add_argument('--output', type=Path, required=True)
    f.add_argument('--references', type=Path, required=True); f.add_argument('--candidates', type=Path, required=True)
    f.add_argument('--legacy-pool', type=Path, required=True)
    v = sub.add_parser('verify'); v.add_argument('--pool', type=Path, required=True)
    v.add_argument('--source', action='store_true')
    s = sub.add_parser('schedule'); s.add_argument('--pool', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--fast-seeds', type=int, default=64); s.add_argument('--search-seeds', type=int, default=16)
    m = sub.add_parser('match'); m.add_argument('--pool', type=Path, required=True)
    m.add_argument('--job', type=Path, required=True); m.add_argument('--output', type=Path, required=True)
    b = sub.add_parser('batch'); b.add_argument('--pool', type=Path, required=True)
    b.add_argument('--jobs', type=Path, required=True); b.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'freeze':
        manifest = prepare(a.output, a.references, a.candidates, a.legacy_pool)
        print(json.dumps(dict(games=len(manifest['games']), policies=sum(len(x['members']) for x in manifest['games'].values()), sha256=manifest['manifest_sha256'])))
    elif a.command == 'verify':
        print(verify_pool(a.pool, a.source)['manifest_sha256'])
    elif a.command == 'schedule':
        schedule = make_schedule(verify_pool(a.pool), a.fast_seeds, a.search_seeds)
        schedule['schedule_sha256'] = canonical_hash(schedule)
        if a.output.exists(): raise ValueError('Schedule is immutable')
        write_json(a.output, schedule)
        print(json.dumps(dict(jobs=len(schedule['jobs']), sha256=schedule['schedule_sha256'])))
    elif a.command == 'match':
        import torch
        torch.set_num_threads(1)
        if a.output.exists():
            raise FileExistsError('Keep completed match artifacts immutable; choose a new output')
        job = json.loads(a.job.read_text())
        result = match(a.pool, job, trace_path=a.output.with_suffix('.trajectory.json.gz') if job.get('record') else None)
        result['result_sha256'] = canonical_hash(result)
        write_json(a.output, result)
        print(json.dumps(dict(id=job.get('id'), status=result['status'], wall_seconds=result['wall_seconds'])))
    else:
        import torch
        import traceback
        torch.set_num_threads(1)
        manifest = verify_pool(a.pool)
        for job in json.loads(a.jobs.read_text()):
            output = a.output / (job['id']+'.json')
            if output.exists():
                old = json.loads(output.read_text())
                digest = old.pop('result_sha256', None)
                if (old.get('job') != job or old.get('pool_sha256') != manifest['manifest_sha256']
                        or canonical_hash(old) != digest):
                    raise ValueError('Existing result identity changed')
                continue
            began = time.monotonic(); cpu = time.process_time()
            try:
                result = match(a.pool, job, manifest, output.with_suffix('.trajectory.json.gz') if job.get('record') else None)
            except Exception as error:
                result = dict(job=job, game=job['game'], names=job['names'], seed=job['seed'],
                    pool_sha256=manifest['manifest_sha256'], status='error', error=repr(error),
                    traceback=traceback.format_exc(), wall_seconds=time.monotonic()-began,
                    cpu_seconds=time.process_time()-cpu, ended_unix=time.time())
            result['result_sha256'] = canonical_hash(result)
            write_json(output, result)
            print(json.dumps(dict(id=job['id'], status=result['status'], wall_seconds=result['wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
