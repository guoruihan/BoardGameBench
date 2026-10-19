"""Frozen full-game experiment using the shared, guarded CPU lifecycle."""
import argparse
import math
from pathlib import Path
import sys

from boardbench.artifacts.store import read_json
from boardbench.v122 import campaign as base
from boardbench.v123.policies import POLICIES
from .report import publish

output_version = 'v124'
episode_module = 'boardbench.v124.episode'
controller_module = 'boardbench.v124.campaign'
make_jobs = base.make_jobs


def prepare_plan(raw, manifest):
    plan = base.prepare_plan(raw, manifest)
    offset = plan.get('environment_offset', 0)
    if type(offset) is not int or offset < 0:
        raise ValueError('nonnegative environment offset required')
    pool = manifest['sets']['train' if plan['split'] == 'train_calibration' else 'development']
    if offset + plan['panels'][-1] > len(pool):
        raise ValueError('offset panel exceeds split')
    plan['environment_seeds'] = pool[offset:offset + plan['panels'][-1]]
    plan['protocol'] = 'full_game_continuation_wall_budget_v124'
    for arm in plan['arms']:
        params = arm['params']
        if params.get('continuation') not in POLICIES:
            raise ValueError('unregistered continuation')
        mode = params.get('mode')
        if mode == 'greedy':
            if arm['method'] != 'heuristic' or set(params) != {'mode', 'continuation'}:
                raise ValueError('invalid greedy control')
        elif mode == 'timed':
            if (arm['method'] != 'terminal' or arm['multiplier'] != 1 or
                set(params) != {'mode', 'continuation', 'seconds'} or
                not math.isfinite(params['seconds']) or not 0 < params['seconds'] <= 3600 or
                arm['timeout_seconds'] <= params['seconds']):
                raise ValueError('invalid timed arm or insufficient watchdog margin')
        elif mode == 'fixed':
            if (arm['method'] != 'terminal' or params['continuation'] != 'original' or
                set(params) != {'mode', 'continuation'}):
                raise ValueError('fixed control must preserve original TerminalRollout')
        else:
            raise ValueError('unknown mode')
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'control', 'report'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--plan')
    parser.add_argument('--repo')
    args = parser.parse_args()
    backend = sys.modules[__name__]
    if args.mode == 'launch':
        if not args.plan or not args.repo:
            parser.error('launch needs --plan and --repo')
        base.launch(args.plan, args.root, args.repo, backend=backend)
    elif args.mode == 'control':
        base.supervise(args.root, backend=backend)
    else:
        root = Path(args.root)
        publish(root, read_json(root / 'plan.json'), read_json(root / 'jobs.json'),
                base.collect_results(root), read_json(root / 'status.json'))


if __name__ == '__main__':
    main()
