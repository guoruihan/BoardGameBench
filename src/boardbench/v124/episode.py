"""V1.2.4 full games using the existing public-simulator / replay runner."""
import argparse
import json
from pathlib import Path
import time

from boardbench.artifacts.store import read_json, write_json
from boardbench.environments.harmonies import STACKS
from boardbench.v122.episode import run_episode as base_episode
from boardbench.v122.search import TerminalRollout
from .search import Greedy, TimedRollout


def make_policy(method, multiplier, search_seed, **params):
    mode = params.pop('mode')
    continuation = params.pop('continuation')
    if method == 'heuristic' and mode == 'greedy':
        if params:
            raise ValueError('unexpected greedy parameters')
        return Greedy(continuation)
    if method == 'terminal' and mode == 'fixed' and continuation == 'original':
        return TerminalRollout(search_seed, rollouts=4 * multiplier, **params)
    if method == 'terminal' and mode == 'timed':
        return TimedRollout(search_seed, continuation, **params)
    raise ValueError('unregistered method / continuation')


def run_episode(job, directory, **kwargs):
    started, cpu_started = time.monotonic(), time.process_time()
    result = base_episode(job, directory, policy_factory=make_policy, **kwargs)
    result.update(fallback_decisions=0, completed_rounds=0, accepted_terminal_rollouts=0,
                  discarded_terminal_rollouts=0, interrupted_rounds=0)
    final = None
    for line in (Path(directory) / 'decisions.jsonl').read_text().splitlines():
        decision = json.loads(line)
        stats = decision['search']
        for key in ('completed_rounds', 'accepted_terminal_rollouts', 'discarded_terminal_rollouts'):
            result[key] += stats.get(key, 0)
        result['fallback_decisions'] += int(stats.get('fallback', False))
        result['interrupted_rounds'] += int(stats.get('interrupted_round', False))
        final = decision['observation']
    if final is not None:
        result['terminal_metrics'] = {
            'terrain': result['partial_score'] - result['components'].get('animals', 0),
            'animals': result['components'].get('animals', 0),
            'animals_placed': sum(c['animal'] is not None for c in final['board']),
            'completed_cards': len(final['completed_cards']), 'turns': final['turn_index'],
            'tokens': sum(len(STACKS[c['stack_id']]) for c in final['board']),
        }
    cap = job.get('params', {}).get('seconds')
    result['solver_budget_seconds'] = cap
    result['solver_budget_utilization'] = result['decision_seconds'] / cap if cap else None
    result['solver_budget_overrun_seconds'] = max(0., result['decision_seconds'] - cap) if cap else 0.
    result.update(elapsed_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started)
    write_json(Path(directory) / 'result.json', result)
    write_json(Path(directory) / 'progress.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = run_episode(read_json(args.job), args.out)
    print(json.dumps({k: result[k] for k in ('id', 'status', 'score', 'decision_seconds')}), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
