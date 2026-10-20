"""Replay old train states; compare two fixed actions under two continuations.

No real hidden state is passed to simulation. Future streams are new and common
to all four branches. The old search winner is fixed BEFORE these samples.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.v121.registry import validate_manifest
from boardbench.v125.policies import GoalContinuation
from boardbench.v130.teacher import Student64


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def paired_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('at least two finite paired returns required')
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    mean = float(values.mean())
    return {'n': len(values), 'mean': mean, 'se': se,
            'lower': mean-1.96*se, 'upper': mean+1.96*se,
            'interpretation': 'descriptive normal interval, not a simultaneous guarantee'}


def select_states(runs, checkpoint, checkpoint_sha256, manifest, per_environment=16):
    validate_manifest(manifest)
    allowed = set(manifest['sets']['train'])
    student = Student64(checkpoint, checkpoint_sha256)
    grouped = defaultdict(dict); sources = []
    for run in map(Path, runs):
        if read_json(run/'plan.json')['split'] != 'train_calibration':
            raise ValueError('only train trajectories may provide states')
        status = read_json(run/'status.json')
        if status['phase'] != 'completed' or status['active_count']:
            raise ValueError('source trajectory campaign must be complete')
        sources.append({'run': str(run), 'plan_sha256': digest(run/'plan.json')})
        for result_file in sorted((run/'resolved').glob('*.json')):
            result = read_json(result_file); seed = result['environment_seed']
            if seed not in allowed or result['status'] != 'completed' or not result['natural_terminal']:
                raise ValueError('invalid source split or terminal')
            engine = Harmonies(); obs, _ = engine.reset(seed)
            path = Path(result.get('inherited_from', str(run)))/'episodes'/result['id']/'decisions.jsonl'
            log_hash = digest(path)
            for line in path.open():
                row = json.loads(line)
                if obs != row['observation_before']:
                    raise ValueError('source replay state mismatch')
                teacher = row['search']['expert_action']
                if teacher not in obs['legal_actions']:
                    raise ValueError('illegal teacher action')
                if row['search']['completed_rounds'] >= 2 and len(obs['legal_actions']) > 1:
                    action = student.decide(obs)
                    if teacher != action:
                        key = fingerprint(obs)
                        grouped[seed].setdefault(key, {
                            'id': f'e{seed}_{key[:16]}', 'environment_seed': seed,
                            'observation': obs, 'teacher_action': teacher, 'student_action': action,
                            'source_log': str(path), 'source_log_sha256': log_hash,
                            'source_decision': row['decision'],
                            'old_estimated_gap': row['search']['teacher_minus_student_estimated_return'],
                            'student_checkpoint_sha256': checkpoint_sha256})
                obs = engine.step(row['action']).observation
                if obs != row['observation']:
                    raise ValueError('source replay successor mismatch')
            if not obs['terminated'] or obs['score'] != result['score']:
                raise ValueError('source replay terminal mismatch')
    selected = []
    for seed, records in sorted(grouped.items()):
        # Round-robin across stage/action strata, never select on estimated gain.
        strata = defaultdict(list)
        for row in records.values():
            phase = min(2, (row['source_decision']-1)//25)
            strata[(phase, row['teacher_action']['type'])].append(row)
        for values in strata.values():
            values.sort(key=lambda r: fingerprint(['v131-selection', r['id']]))
        choices = []
        while strata and len(choices) < per_environment:
            for key in sorted(list(strata)):
                choices.append(strata[key].pop())
                if not strata[key]: del strata[key]
                if len(choices) == per_environment: break
        selected.extend(choices)
    return selected, sources


def rollout(obs, action, policy, future_seed, deadline):
    engine = Harmonies.from_observation(obs, future_seed)
    for step in range(256):
        if time.monotonic() >= deadline:
            raise TimeoutError('counterfactual task wall budget exhausted')
        result = engine.step(action)
        if result.terminated:
            return {'score': result.observation['score'],
                    'animals': result.observation['score_breakdown']['animals'], 'steps': step+1}
        action = policy.decide(result.observation)
    raise RuntimeError('natural terminal not reached; no leaf score substituted')


def evaluate_state(job, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.monotonic(), time.process_time()
    deadline = started+job['timeout_seconds']-2
    student = Student64(job['checkpoint'], job['checkpoint_sha256'])
    obs = job['observation']
    if fingerprint(obs) != job['observation_sha256']:
        raise ValueError('public state hash mismatch')
    if student.decide(obs) != job['student_action']:
        raise ValueError('frozen student action mismatch')
    if job['teacher_action'] not in obs['legal_actions']:
        raise ValueError('illegal teacher proposal')
    policies = {'habitat': GoalContinuation(compiled=True), 'student': student}
    samples = []; rng = random.Random(job['future_seed'])
    status, error = 'completed', None
    try:
        for repeat in range(job['repeats']):
            seed = rng.getrandbits(63)
            values = {}
            # Rotate order so a wall deadline does not privilege a branch.
            order = [(p, a) for p in policies for a in ('student', 'teacher')]
            order = order[repeat % 4:]+order[:repeat % 4]
            for name, action in order:
                values[name+'_'+action] = rollout(obs, job[action+'_action'], policies[name], seed, deadline)
            samples.append({'future_seed': seed, 'values': values})
            write_json(out/'progress.json', {'id': job['id'], 'status': 'running',
                'completed_rounds': len(samples), 'updated_unix': time.time(),
                'elapsed_seconds': time.monotonic()-started})
    except Exception as exc:
        status, error = 'failed', repr(exc)
    stats = {}
    if len(samples) >= 2:
        for name in policies:
            stats[name] = paired_stats([s['values'][name+'_teacher']['score']-
                                       s['values'][name+'_student']['score'] for s in samples])
        stats['interaction'] = paired_stats([
            (s['values']['student_teacher']['score']-s['values']['student_student']['score'])-
            (s['values']['habitat_teacher']['score']-s['values']['habitat_student']['score']) for s in samples])
    result = {k: v for k, v in job.items() if k != 'observation'}
    result.update(status=status, error=error, score=None,
        natural_terminal=status == 'completed', samples=samples, statistics=stats,
        elapsed_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
        score_semantics='paired counterfactual terminal differences, not full-game policy performance')
    write_json(out/'result.json', result)
    return result


def summarize(rows):
    complete = [r for r in rows if r.get('status') == 'completed' and r.get('natural_terminal')]
    if not complete: return {'completed': 0}
    result = {'completed': len(complete), 'environments': len({r['environment_seed'] for r in complete}),
              'aggregation': 'equal environment means; states and futures are not independent environments'}
    for name in ('habitat', 'student', 'interaction'):
        env = defaultdict(list)
        for r in complete: env[r['environment_seed']].append(r['statistics'][name]['mean'])
        values = [float(np.mean(v)) for v in env.values()]
        result[name] = paired_stats(values) if len(values) > 1 else {'n': 1, 'mean': values[0]}
    result['habitat_positive_student_negative'] = sum(
        r['statistics']['habitat']['mean'] > 0 > r['statistics']['student']['mean'] for r in complete)
    result['reliable_habitat_positive_student_negative'] = sum(
        r['statistics']['habitat']['lower'] > 1 and r['statistics']['student']['upper'] < 0 for r in complete)
    result['admitted'] = {name: sum(r['statistics'][name]['lower'] > 1 for r in complete)
                          for name in ('habitat', 'student')}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); result = evaluate_state(read_json(a.job), a.out)
    print(json.dumps({k: result[k] for k in ('id','status','statistics','elapsed_seconds')}), flush=True)
    if result['status'] != 'completed': raise SystemExit(1)


if __name__ == '__main__': main()
