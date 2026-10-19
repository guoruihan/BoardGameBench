"""Small independent v125 tasks, reusing frozen-source workers and watchdogs."""
import argparse
import math
from pathlib import Path
import sys
import time

from boardbench.artifacts.store import read_json, write_json
from boardbench.v122 import campaign as base
from .search import VARIANTS
from .report import publish

output_version = 'v125'
episode_module = 'boardbench.v125.episode'
controller_module = 'boardbench.v125.campaign'
make_jobs = base.make_jobs


def prepare_plan(raw, manifest):
    plan = base.prepare_plan(raw, manifest)
    indices = raw.get('environment_indices', list(range(plan['panels'][-1])))
    pool = manifest['sets']['train' if plan['split'] == 'train_calibration' else 'development']
    if (len(indices) != plan['panels'][-1] or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(pool) for i in indices)):
        raise ValueError('unique in-split environment indices matching final panel required')
    plan['environment_seeds'] = [pool[i] for i in indices]
    plan['protocol'] = 'autonomous_small_cohort_wall_budget_v125'
    for a in plan['arms']:
        p = a['params']
        if (a['method'] != 'terminal' or a['multiplier'] != 1 or p.get('mode') != 'timed'
                or p.get('continuation') not in VARIANTS or set(p) != {'mode', 'continuation', 'seconds'}
                or not math.isfinite(p['seconds']) or not 0 < p['seconds'] <= 900
                or a['timeout_seconds'] <= p['seconds']):
            raise ValueError('invalid timed variant or 900-second game cap')
    return plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('launch', 'control', 'report'))
    p.add_argument('--root', required=True)
    p.add_argument('--plan')
    p.add_argument('--repo')
    p.add_argument('--preflight-core-seconds', type=float, default=0.)
    args = p.parse_args()
    backend = sys.modules[__name__]
    if args.mode == 'launch':
        if not args.plan or not args.repo:
            p.error('--plan and --repo required')
        repo, root = Path(args.repo).resolve(), Path(args.root).resolve()
        if not root.is_relative_to(repo / 'outputs/v125') or root.exists():
            p.error('new output root under project outputs/v125 required')
        raw = read_json(args.plan)
        duration = raw.pop('task_seconds', 1500)
        if not math.isfinite(duration) or not 10 < duration <= 8 * 3600:
            p.error('bounded independent task_seconds required')
        raw['deadline_unix'] = time.time() + duration
        if args.preflight_core_seconds:
            if not math.isfinite(args.preflight_core_seconds) or args.preflight_core_seconds < 0:
                p.error('finite nonnegative preflight cost required')
            receipt = repo / 'outputs/v125/receipts' / (root.name + '.json')
            if receipt.exists():
                p.error('preflight receipt already exists')
            write_json(receipt, {'charged_cpu_core_seconds': args.preflight_core_seconds,
                                 'kind': 'declared conservative implementation and verification allowance'})
            raw['preflight_cost_receipts'] = [str(receipt.relative_to(repo))]
        prepared = repo / 'outputs/v125/plans' / (root.name + '.json')
        if prepared.exists():
            p.error('task plan already exists; choose a new attempt identity')
        write_json(prepared, raw)
        base.launch(prepared, root, repo, backend=backend)
    elif args.mode == 'control':
        base.supervise(args.root, backend=backend)
    else:
        root = Path(args.root)
        publish(root, read_json(root / 'plan.json'), read_json(root / 'jobs.json'),
                base.collect_results(root), read_json(root / 'status.json'))


if __name__ == '__main__':
    main()
