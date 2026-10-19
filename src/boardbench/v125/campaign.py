"""Small independent v125 tasks, reusing frozen-source workers and watchdogs."""
import argparse
import math
import os
from pathlib import Path
import sys
import time

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v122 import campaign as base
from .search import VARIANTS
from .report import publish

output_version = 'v125'
episode_module = 'boardbench.v125.episode'
controller_module = 'boardbench.v125.campaign'
make_jobs = base.make_jobs


def cpu_core_limit(cpus, plan):
    profile = plan.get('resource_profile', 'default_half')
    if profile == 'default_half':
        return len(cpus) // 2
    if profile != 'b35_reserve16_v1' or os.uname().nodename != 'b35.cluster.helixon.com':
        raise ValueError('expanded CPU profile is explicitly authorized only on b35')
    return min(112, max(0, len(cpus) - 16))


def resume_parent(plan):
    parent = Path(plan['resume_from']).resolve()
    state = read_json(parent / 'status.json')
    if state.get('active_count', 0) or state.get('phase') == 'running' or not (parent / 'watchdog_result.json').exists():
        raise ValueError('resume requires a finalized inactive parent')
    allocation = read_json(parent / 'allocation.json')
    if (source_id(parent / 'source') != allocation['source_id'] or
            digest(parent / 'plan.json') != allocation['plan_sha256'] or
            digest(parent / 'splits.json') != allocation['splits_sha256']):
        raise ValueError('parent source/plan/split hash mismatch')
    if make_jobs(plan) != read_json(parent / 'jobs.json'):
        raise ValueError('resume jobs differ from original frozen task')
    # v0.4.7 had no recovery hook. Refuse an unsafe apparent resume that would
    # actually rerun completed jobs under its old frozen controller.
    text = (parent / 'source/src/boardbench/v125/campaign.py').read_text()
    if 'def initial_results(' not in text:
        raise ValueError('parent predates recovery support; preserve it, do not silently rerun')
    return parent


def snapshot_repository(repo, plan):
    return resume_parent(plan) / 'source' if plan.get('resume_from') else repo


def initial_results(root, plan, jobs):
    if not plan.get('resume_from'):
        return []
    parent = resume_parent(plan)
    from boardbench.v124.report import summarize
    rows = base.collect_results(parent)
    summarize(plan, jobs, rows)  # verifies identities, parameters and scores
    completed = []
    for row in rows:
        if row['status'] == 'completed' and row['natural_terminal']:
            inherited = {**row, 'inherited_from': row.get('inherited_from', str(parent)),
                         'reused_completed_attempt': True}
            write_json(Path(root) / 'resolved' / (row['id'] + '.json'), inherited)
            completed.append(inherited)
    write_json(Path(root) / 'resume.json', {'parent': str(parent), 'reused_completed': len(completed),
        'pending_or_retry': len(jobs) - len(completed), 'parent_costs_remain_at': str(parent / 'watchdog_result.json'),
        'policy_source_id': source_id(parent / 'source')})
    return completed


def prepare_plan(raw, manifest):
    plan = base.prepare_plan(raw, manifest)
    indices = raw.get('environment_indices', list(range(plan['panels'][-1])))
    pool = manifest['sets']['train' if plan['split'] == 'train_calibration' else 'development']
    if (len(indices) != plan['panels'][-1] or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < len(pool) for i in indices)):
        raise ValueError('unique in-split environment indices matching final panel required')
    plan['environment_seeds'] = [pool[i] for i in indices]
    plan['protocol'] = 'autonomous_small_cohort_wall_budget_v125'
    if 'historical_reference' in plan:
        from .historical import validate_reference
        validate_reference(plan['historical_reference'], plan)
        plan['protocol'] = 'retained_policy_expanded_wall_budget_v125'
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
    p.add_argument('mode', choices=('launch', 'resume', 'control', 'report'))
    p.add_argument('--root', required=True)
    p.add_argument('--plan')
    p.add_argument('--repo')
    p.add_argument('--preflight-core-seconds', type=float, default=0.)
    p.add_argument('--parent')
    p.add_argument('--task-seconds', type=float)
    args = p.parse_args()
    backend = sys.modules[__name__]
    if args.mode in ('launch', 'resume'):
        if not args.repo or (args.mode == 'launch' and not args.plan) or (args.mode == 'resume' and not args.parent):
            p.error('--repo plus --plan (launch) or --parent (resume) required')
        repo, root = Path(args.repo).resolve(), Path(args.root).resolve()
        if not root.is_relative_to(repo / 'outputs/v125') or root.exists():
            p.error('new output root under project outputs/v125 required')
        if args.mode == 'resume':
            parent = Path(args.parent).resolve()
            if not parent.is_relative_to(repo / 'outputs/v125'):
                p.error('resume parent must belong to this project v125')
            raw = read_json(parent / 'plan.json')
            raw['resume_from'] = str(parent)
            raw['preflight_cost_receipts'] = []  # parent keeps its own ledger
            # Validate before creating a new directory or charging a reservation.
            resume_parent(raw)
        else:
            raw = read_json(args.plan)
            if 'historical_reference_file' in raw:
                reference = (repo / raw['historical_reference_file']).resolve()
                if not reference.is_relative_to(repo / 'outputs/v125/references'):
                    p.error('historical reference must be a project v125 snapshot')
                raw['historical_reference'] = read_json(reference)
                raw['historical_reference_sha256'] = digest(reference)
        duration = args.task_seconds if args.task_seconds is not None else raw.pop('task_seconds', 1500)
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
