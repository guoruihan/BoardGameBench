"""Reuse natural-terminal runner and public-state replay logging."""
import argparse
import json
from boardbench.artifacts.store import read_json
from boardbench.v124.episode import run_episode as base_episode
from boardbench.v125.search import Search


def make_policy(method, multiplier, search_seed, **params):
    if method != 'terminal' or multiplier != 1 or params.pop('mode') not in ('timed','student'):
        raise ValueError('unregistered V126 method')
    variant = params.pop('continuation')
    if variant in ('mixed_roots', 'habitat', 'original'):
        return Search(search_seed, variant, **params)
    from .policy import LearnedSearch
    return LearnedSearch(search_seed, variant, **params)


def run_episode(job, directory, **kwargs):
    return base_episode(job, directory, policy_factory=make_policy, log_pre_observation=True, **kwargs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(); result=run_episode(read_json(a.job),a.out)
    print(json.dumps({k:result[k] for k in ('id','status','score','decision_seconds')}),flush=True)
    if result['status']!='completed':
        raise SystemExit(1)


if __name__=='__main__':
    main()
