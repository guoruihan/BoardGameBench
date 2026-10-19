"""One auditable full-game attempt; external supervisor enforces its wall limit."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from boardbench.artifacts.store import read_json, write_json
from boardbench.environments import TASKS
from boardbench.solvers.board import BoardSolver
from .search import TerminalRollout, short_parameters


def make_policy(method, multiplier, search_seed, **params):
    if method == 'heuristic':
        return BoardSolver(search_seed, method='heuristic')
    if method == 'short':
        return BoardSolver(search_seed, **{**short_parameters(multiplier), **params})
    if method == 'terminal':
        return TerminalRollout(search_seed, **{'rollouts': 4 * multiplier, **params})
    raise ValueError(f'unknown method {method}')


def run_episode(job, directory, *, max_decisions=256, policy_factory=make_policy,
                log_pre_observation=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'job.json', job)
    started, cpu_started = time.monotonic(), time.process_time()
    engine = TASKS['harmonies'].factory()
    obs, _ = engine.reset(job['environment_seed'])
    solver = policy_factory(job['method'], job['multiplier'], job['search_seed'], **job.get('params', {}))
    count = terminals = steps = 0
    decision_seconds = 0.
    rollouts = horizon_completed = cache_hits = cache_calls = 0
    candidate_total = legal_total = pruned_decisions = 0
    error, status = None, 'running'
    last_progress = time.monotonic()
    def tick():
        nonlocal count, last_progress
        count += 1
        if time.monotonic() - last_progress > 10:
            write_json(directory / 'progress.json', progress())
            last_progress = time.monotonic()
    # The wrapper counts natural terminals, independently of solver-reported stats.
    class Branch:
        def __init__(self, public, seed):
            self.engine = TASKS['harmonies'].factory.from_observation(public, seed)
        def step(self, action):
            nonlocal terminals
            result = self.engine.step(action)
            tick()
            terminals += int(result.terminated)
            return result
    def progress():
        breakdown = obs['score_breakdown']
        return {**job, 'status': status, 'error': error,
                'natural_terminal': bool(obs['terminated']),
                'score': obs['score'] if obs['terminated'] else None,
                'partial_score': breakdown['total'], 'components': breakdown,
                'real_steps': steps, 'simulation_steps': count, 'terminal_rollouts': terminals,
                'rollouts': rollouts, 'horizon_completed_rollouts': horizon_completed,
                'candidate_total': candidate_total, 'legal_total': legal_total,
                'pruned_decisions': pruned_decisions,
                'cache_hits': cache_hits, 'cache_calls': cache_calls,
                'decision_seconds': decision_seconds,
                'elapsed_seconds': time.monotonic() - started,
                'cpu_seconds': time.process_time() - cpu_started,
                'updated_unix': time.time()}
    write_json(directory / 'progress.json', progress())
    try:
        with (directory / 'decisions.jsonl').open('x') as log:
            while not obs['terminated']:
                if steps >= max_decisions:
                    raise RuntimeError('real game exceeded action safety bound')
                public = deepcopy(obs)
                context = SimpleNamespace(reference_simulator=SimpleNamespace(
                    fork=lambda seed: Branch(public, seed)))
                before_count, begin = count, time.monotonic()
                action = solver.decide(deepcopy(public), context)
                elapsed = time.monotonic() - begin
                decision_seconds += elapsed
                if action not in obs['legal_actions']:
                    raise RuntimeError('illegal solver action')
                stats = deepcopy(solver.last_search)
                legal = len(public['legal_actions'])
                candidates = (min(solver.width, legal, solver.max_simulation_steps // solver.rollouts)
                              if job['method'] == 'short' else stats.get('candidate_count', legal))
                candidate_total += candidates
                legal_total += legal
                pruned_decisions += int(candidates < legal)
                rollouts += stats.get('rollouts', 0)
                horizon_completed += stats.get('horizon_completed_rollouts', 0)
                cache_hits += stats.get('cache_hits', 0)
                cache_calls += stats.get('cache_calls', 0)
                obs = engine.step(action).observation
                steps += 1
                extra = ({'observation_before': public,
                          'raw_score_delta': obs['score_breakdown']['total'] - public['score_breakdown']['total']}
                         if log_pre_observation else {})
                log.write(json.dumps({'decision': steps, 'action': action, 'observation': obs,
                                      'seconds': elapsed, 'simulation_steps': count - before_count,
                                      'search': stats, **extra}, allow_nan=False) + '\n')
                # Shared cluster storage can make per-action atomic replacement
                # slower than the heuristic itself. Bound lost crash telemetry
                # to about one second without putting I/O inside solver timing.
                if time.monotonic() - last_progress > 1:
                    log.flush()
                    write_json(directory / 'progress.json', progress())
                    last_progress = time.monotonic()
        status = 'completed'
    except Exception:
        status, error = 'failed', traceback.format_exc()
    result = progress()
    write_json(directory / 'progress.json', result)
    write_json(directory / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = run_episode(read_json(args.job), args.out)
    print(json.dumps({k: result[k] for k in ('id', 'status', 'score', 'decision_seconds', 'simulation_steps')}), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
