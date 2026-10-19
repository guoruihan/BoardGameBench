"""Reuse the audited complete-game runner and add explicit learning alignment."""
import argparse
import json

from boardbench.artifacts.store import read_json
from boardbench.v124.episode import run_episode as base_episode
from .search import Search


def make_policy(method, multiplier, search_seed, **params):
    if method != 'terminal' or multiplier != 1 or params.pop('mode') != 'timed':
        raise ValueError('v125 requires cumulative timed terminal search')
    variant = params.pop('continuation')
    return Search(search_seed, variant, **params)


def run_episode(job, directory, **kwargs):
    return base_episode(job, directory, policy_factory=make_policy, log_pre_observation=True, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    result = run_episode(read_json(args.job), args.out)
    print(json.dumps({k: result[k] for k in ('id', 'status', 'score', 'decision_seconds')}), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
