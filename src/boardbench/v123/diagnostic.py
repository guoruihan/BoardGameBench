"""All-root screening, independent confirmation, fixed public-only continuations."""
import argparse
import json
from pathlib import Path
import time
import traceback

from boardbench.artifacts.store import read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.solvers.board import ranked_actions
from .bank import board_metrics, object_hash, validate_public
from .policies import Continuation, POLICIES


def sample_seeds(master, state_id, stage, count):
    if stage not in ('discovery', 'confirmation'):
        raise ValueError('invalid stage')
    # Domain-separated hashes, also checked for accidental collision in each job.
    return [int(object_hash(['v123', master, state_id, stage, i])[:16], 16) for i in range(count)]


def rollout(obs, action, policy, seed, tick=lambda: None, branch_cap=256):
    engine = Harmonies.from_observation(obs, seed)
    current = action
    for steps in range(1, branch_cap + 1):
        result = engine.step(current)
        tick()
        if result.terminated:
            return {**board_metrics(result.observation), 'steps': steps, 'natural_terminal': True,
                    'board_end_condition': sum(c['stack_id'] == 0 for c in result.observation['board']) <= 2}
        # No branch object, seed, actual hidden supply or source metadata provided.
        current = policy.decide(result.observation)
    raise RuntimeError('terminal rollout exceeded safety bound; no partial leaf substituted')


def select_roots(discovery, width):
    choices = {}
    for policy, rows in discovery.items():
        inside, outside = rows[:width], rows[width:]
        choices[policy] = {'top': max(inside, key=lambda r: r['mean'])['rank'],
                           'outside': max(outside, key=lambda r: r['mean'])['rank'] if outside else None,
                           'all': max(rows, key=lambda r: r['mean'])['rank']}
    roots = sorted({rank for row in choices.values() for rank in row.values() if rank is not None})
    return choices, roots


def run_state(job, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'job.json', job)
    started, cpu_started = time.monotonic(), time.process_time()
    count = terminal_count = 0
    last_progress = 0.
    stage, status, error = 'initializing', 'running', None
    discovery, confirmation, choices = {}, {}, {}
    def progress():
        return {k: v for k, v in job.items() if k != 'observation'} | {
            'status': status, 'error': error, 'stage': stage,
            'simulation_steps': count, 'terminal_rollouts': terminal_count,
            'elapsed_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu_started,
            'updated_unix': time.time()}
    def tick():
        nonlocal count, last_progress
        count += 1
        if time.monotonic() - last_progress >= 10:
            write_json(directory / 'progress.json', progress())
            last_progress = time.monotonic()
    write_json(directory / 'progress.json', progress())
    try:
        obs = job['observation']
        validate_public(obs)
        if object_hash(obs) != job['observation_sha256']:
            raise ValueError('job public observation digest mismatch')
        ranked = ranked_actions(obs)
        seeds = {phase: sample_seeds(job['seed'], job['id'], phase, job[phase + '_samples'])
                 for phase in ('discovery', 'confirmation')}
        if len(set(seeds['discovery'] + seeds['confirmation'])) != sum(map(len, seeds.values())):
            raise ValueError('seed collision')
        policies = {name: Continuation(name, job.get('cache_capacity', 8192)) for name in POLICIES}
        def evaluate(name, ranks, log):
            nonlocal terminal_count
            rows = []
            for rank in ranks:
                outcomes = []
                for sample, seed in enumerate(seeds[stage]):
                    outcome = rollout(obs, ranked[rank][1], policies[name], seed, tick,
                                      branch_cap=job.get('branch_cap', 256))
                    terminal_count += 1
                    outcomes.append(outcome)
                    log.write(json.dumps({'stage': stage, 'policy': name, 'rank': rank, 'sample': sample,
                                          'seed': seed, **outcome}, separators=(',', ':')) + '\n')
                log.flush()
                rows.append({'rank': rank, 'action': ranked[rank][1],
                             'mean': sum(o['score'] for o in outcomes) / len(outcomes),
                             'scores': [o['score'] for o in outcomes],
                             'components': {k: sum(o[k] for o in outcomes) / len(outcomes)
                                            for k in board_metrics(obs)},
                             'samples': len(outcomes)})
            return rows
        with (directory / 'rollouts.jsonl').open('x') as log:
            stage = 'discovery'
            for name in POLICIES:
                discovery[name] = evaluate(name, range(len(ranked)), log)
            choices, roots = select_roots(discovery, job['width'])
            # Commit selections before any confirmation sample is observed.
            write_json(directory / 'selection.json', {'choices': choices, 'roots': roots,
                       'discovery': discovery, 'seeds': seeds})
            stage = 'confirmation'
            for name in POLICIES:
                confirmation[name] = evaluate(name, roots, log)
        status, stage = 'completed', 'done'
    except Exception:
        status, error = 'failed', traceback.format_exc()
    result = {**progress(), 'discovery': discovery, 'confirmation': confirmation, 'choices': choices,
              'legal_count': len(job['observation']['legal_actions'])}
    write_json(directory / 'progress.json', progress())
    write_json(directory / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = run_state(read_json(args.job), args.out)
    print(json.dumps({k: result[k] for k in ('id', 'status', 'stage', 'simulation_steps', 'elapsed_seconds')}))
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
