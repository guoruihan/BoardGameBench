"""Frozen-source CPU search campaign with an independent absolute-deadline guard."""
import argparse
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from boardbench.v12.budget import kill_tree
from boardbench.v121.registry import validate_manifest
from .report import publish


def physical_cpus():
    selected, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        identity = tuple((topology / field).read_text().strip() for field in ('physical_package_id', 'core_id'))
        if identity not in seen:
            selected.append(cpu)
            seen.add(identity)
    return selected


def prepare_plan(plan, manifest):
    validate_manifest(manifest)
    plan = dict(plan)
    split = plan['split']
    if split not in ('train_calibration', 'development'):
        raise ValueError('only training calibration or development allowed; no confirmation/test')
    panels = plan['panels']
    if not panels or any(type(n) is not int or n < 1 for n in panels) or panels != sorted(set(panels)):
        raise ValueError('positive strictly increasing fixed panel sizes required')
    seeds = manifest['sets']['train' if split == 'train_calibration' else 'development']
    if panels[-1] > len(seeds):
        raise ValueError('panel exceeds split')
    if not plan['search_seeds'] or len(set(plan['search_seeds'])) != len(plan['search_seeds']):
        raise ValueError('unique nonempty search seeds required')
    arms = plan['arms']
    if not arms or len({a['id'] for a in arms}) != len(arms):
        raise ValueError('unique nonempty arms required')
    if not any(a.get('primary', True) for a in arms):
        raise ValueError('at least one primary arm required')
    for arm in arms:
        if not arm['id'].replace('_', '').isalnum() or arm['method'] not in ('heuristic', 'short', 'terminal'):
            raise ValueError('invalid arm')
        if type(arm['multiplier']) is not int or arm['multiplier'] < 1:
            raise ValueError('invalid sample multiplier')
        if not math.isfinite(arm['timeout_seconds']) or arm['timeout_seconds'] <= 0:
            raise ValueError('finite positive per-attempt watchdog required')
    plan.update(environment_seeds=seeds[:panels[-1]], split_sha256=manifest['sha256'],
                protocol='full_game_search_diagnostic_v1', rules_version='harmonies_solo_a_v1')
    return plan


def make_jobs(plan):
    # Fixed environment-major ordering. Later panels may fill idle slots while
    # earlier expensive jobs finish; the report only promotes complete prefixes.
    jobs = []
    for index, seed in enumerate(plan['environment_seeds']):
        panel = next(n for n in plan['panels'] if index < n)
        for repeat, search_seed in enumerate(plan['search_seeds']):
            for arm in plan['arms']:
                if arm['method'] == 'heuristic' and repeat:
                    continue
                jobs.append({'id': f'{arm["id"]}_e{index:03d}_r{repeat}', 'arm': arm['id'],
                             'panel': panel, 'environment_index': index, 'environment_seed': seed,
                             'search_seed': search_seed, 'method': arm['method'],
                             'multiplier': arm['multiplier'], 'params': arm.get('params', {}),
                             'timeout_seconds': arm['timeout_seconds']})
    return jobs


def incomplete(job, directory, reason):
    path = Path(directory) / 'progress.json'
    result = read_json(path) if path.exists() else {
        **job, 'real_steps': 0, 'partial_score': None, 'components': {},
        'simulation_steps': 0, 'decision_seconds': 0., 'cpu_seconds': 0., 'elapsed_seconds': 0.}
    result.update(status=reason, score=None, natural_terminal=False,
                  error=result.get('error') or reason, recorded_counters_may_be_lower_bounds=True)
    return result


def collect_results(root):
    return [read_json(p) for p in sorted((Path(root) / 'resolved').glob('*.json'))]


def supervise(root, *, backend=None):
    backend = backend or sys.modules[__name__]
    root = Path(root)
    plan, allocation = read_json(root / 'plan.json'), read_json(root / 'allocation.json')
    if source_id() != allocation['source_id']:
        raise ValueError('controller is not running frozen source')
    if digest(root / 'plan.json') != allocation['plan_sha256']:
        raise ValueError('plan hash mismatch')
    if digest(root / 'splits.json') != allocation['splits_sha256']:
        raise ValueError('split hash mismatch')
    jobs = read_json(root / 'jobs.json')
    if jobs != backend.make_jobs(plan):
        raise ValueError('jobs differ from frozen plan')
    results = getattr(backend, 'initial_results', lambda *_: [])(root, plan, jobs)
    finished = {r['id'] for r in results}
    queue, active = deque(j for j in jobs if j['id'] not in finished), {}
    available = deque(allocation['worker_cpus'])
    deadline = time.monotonic() + max(0., allocation['deadline_unix'] - time.time())
    stopping = [False]
    def stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    phase, last_report = 'running', 0.
    def snapshot():
        status = {'phase': phase, 'updated_unix': time.time(), 'pid': os.getpid(),
                  'source_id': allocation['source_id'], 'queued': len(queue),
                  'resolved': len(results), 'active_count': len(active),
                  'remaining_seconds': max(0., deadline - time.monotonic()),
                  'active': [{'id': value['job']['id'], 'pid': pid, 'cpu': value['cpu'],
                              'elapsed_seconds': time.monotonic() - value['started']}
                             for pid, value in active.items()]}
        write_json(root / 'status.json', status)
        backend.publish(root, plan, jobs, results, status)
        print(json.dumps({k: status[k] for k in ('phase', 'resolved', 'queued', 'active_count', 'remaining_seconds')}), flush=True)
    try:
        while queue or active:
            now = time.monotonic()
            expired = now >= deadline - 8
            if expired or stopping[0]:
                phase = 'deadline' if expired else 'interrupted'
            while queue and available and phase == 'running':
                job, cpu = queue.popleft(), available.popleft()
                job_path, out = root / 'jobs' / f'{job["id"]}.json', root / 'episodes' / job['id']
                write_json(job_path, job)
                stream = (root / 'logs' / f'{job["id"]}.log').open('x')
                command = ['taskset', '-c', str(cpu), sys.executable, '-u', '-m',
                           getattr(backend, 'episode_module', 'boardbench.v122.episode'),
                           '--job', str(job_path), '--out', str(out)]
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[child.pid] = {'process': child, 'job': job, 'cpu': cpu, 'stream': stream,
                                     'started': time.monotonic(), 'out': out}
            for pid, value in list(active.items()):
                child, job = value['process'], value['job']
                elapsed = time.monotonic() - value['started']
                reason = None
                if child.poll() is None and (phase != 'running' or elapsed >= job['timeout_seconds']):
                    reason = phase if phase != 'running' else 'job_timeout'
                    kill_tree(pid)
                    child.wait(timeout=2)
                if child.poll() is not None:
                    result_path = value['out'] / 'result.json'
                    result = read_json(result_path) if result_path.exists() and reason is None else incomplete(
                        job, value['out'], reason or 'worker_exit_without_result')
                    if child.returncode and result.get('status') == 'completed':
                        result.update(status='worker_nonzero_exit', score=None, natural_terminal=False)
                    result.update(returncode=child.returncode, supervised_seconds=elapsed, cpu=value['cpu'])
                    write_json(root / 'resolved' / f'{job["id"]}.json', result)
                    results.append(result)
                    value['stream'].close()
                    available.append(value['cpu'])
                    del active[pid]
            if time.monotonic() - last_report >= 15:
                snapshot()
                last_report = time.monotonic()
            if phase != 'running' and not active:
                break
            time.sleep(.2)
        if phase == 'running':
            phase = 'completed'
    except Exception:
        phase = 'controller_failed'
        write_json(root / 'controller_error.json', {'error': traceback.format_exc()})
        raise
    finally:
        for pid, value in list(active.items()):
            if value['process'].poll() is None:
                kill_tree(pid)
                value['process'].wait(timeout=2)
            result = incomplete(value['job'], value['out'], phase)
            result.update(supervised_seconds=time.monotonic() - value['started'])
            write_json(root / 'resolved' / f'{value["job"]["id"]}.json', result)
            results.append(result)
            value['stream'].close()
            del active[pid]
        snapshot()


def launch(plan_path, root, repo, *, backend=None):
    backend = backend or sys.modules[__name__]
    version = getattr(backend, 'output_version', 'v122')
    started, started_unix = time.monotonic(), time.time()
    root, repo = Path(root).resolve(), Path(repo).resolve()
    if not root.is_relative_to(repo / 'outputs' / version) or root.exists():
        raise ValueError(f'new output directory under outputs/{version} required')
    raw = read_json(plan_path)
    if not math.isfinite(raw['deadline_unix']) or not 10 < raw['deadline_unix'] - started_unix <= 8 * 3600:
        raise ValueError('future absolute deadline within eight hours required')
    cpus = physical_cpus()
    count = raw['cpu_cores']
    if type(count) is not int or not 2 <= count <= len(cpus) // 2:
        raise ValueError('controller plus workers may use at most half the available physical CPU cores')
    if os.getloadavg()[0] > len(cpus) * .5:
        raise RuntimeError('host CPU load too high; no other workload was stopped')
    manifest_path = (repo / raw['split_manifest']).resolve()
    plan = backend.prepare_plan(raw, read_json(manifest_path))
    prior_costs = []
    for relative in plan.get('preflight_cost_receipts', []):
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo / 'outputs' / version):
            raise ValueError(f'preflight receipt must belong to project {version} artifacts')
        receipt = read_json(path)
        charge = receipt['charged_cpu_core_seconds']
        if not math.isfinite(charge) or charge < 0:
            raise ValueError('invalid prior resource charge')
        prior_costs.append({'path': str(path), 'sha256': digest(path),
                            'charged_cpu_core_seconds': charge, 'receipt': receipt})
    root.parent.mkdir(parents=True, exist_ok=True)
    # All CPU research versions share the original host reservation lock.
    lock_path = repo / 'outputs/v122' / f'{os.uname().nodename}.cpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cores = cpus[:count]
    root.mkdir()
    for directory in ('logs', 'jobs', 'episodes', 'resolved'):
        (root / directory).mkdir()
    snapshot_repo = getattr(backend, 'snapshot_repository', lambda repo, plan: repo)(repo, plan)
    for relative in source_files(snapshot_repo):
        target = root / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(snapshot_repo / relative, target)
    frozen_source = source_id(root / 'source')
    write_json(root / 'source_hashes.json', source_files(root / 'source'))
    shutil.copy2(manifest_path, root / 'splits.json')
    write_json(root / 'plan.json', plan)
    jobs = backend.make_jobs(plan)
    write_json(root / 'jobs.json', jobs)
    allocation = {'started_unix': started_unix, 'deadline_unix': plan['deadline_unix'],
                  'cpu_affinity': cores, 'worker_cpus': cores[1:], 'controller_cpu': cores[0],
                  'host_physical_core_count': len(cpus), 'host': os.uname().nodename,
                  'source_id': frozen_source, 'plan_sha256': digest(root / 'plan.json'),
                  'splits_sha256': digest(root / 'splits.json'), 'watchdog_pid': os.getpid(),
                  'git_head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                  'tracked_diff': subprocess.check_output(['git', '-C', str(repo), 'diff', '--stat'], text=True),
                  'gpu_seconds': 0, 'prior_costs': prior_costs,
                  'prior_charged_cpu_core_seconds': sum(p['charged_cpu_core_seconds'] for p in prior_costs),
                  'accounting': 'entire CPU reservation plus declared calibration/verification charges'}
    write_json(root / 'allocation.json', allocation)
    os.environ.update(PYTHONPATH=str(root / 'source/src'), PYTHONNOUSERSITE='1',
                      OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                      NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', PYTHONUNBUFFERED='1')
    os.sched_setaffinity(0, {cores[0]})
    child = subprocess.Popen([sys.executable, '-u', '-m',
                              getattr(backend, 'controller_module', 'boardbench.v122.campaign'),
                              'control', '--root', str(root)], start_new_session=True)
    deadline = started + plan['deadline_unix'] - started_unix
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    forced = False
    try:
        while child.poll() is None:
            if time.monotonic() >= deadline - 3 or stopped[0]:
                forced = True
                kill_tree(child.pid)
                break
            time.sleep(.2)
        child.wait(timeout=2)
    finally:
        if child.poll() is None:
            kill_tree(child.pid)
            child.wait(timeout=2)
        elapsed = time.monotonic() - started
        outcome = {'returncode': child.returncode, 'forced_cleanup': forced,
                   'interrupted': stopped[0], 'elapsed_seconds': elapsed, 'gpu_seconds': 0,
                   'allocated_cpu_core_seconds': count * elapsed,
                   'cleanup_tail_seconds': max(0., time.time() - plan['deadline_unix']),
                   'finished_unix': time.time()}
        write_json(root / 'watchdog_result.json', outcome)
        # Reconcile killed controller/worker attempts, rather than leaving stale
        # running status or silently losing attempts that lack a final result.
        for path in (root / 'jobs').glob('*.json'):
            target = root / 'resolved' / path.name
            if not target.exists():
                job = read_json(path)
                write_json(target, incomplete(job, root / 'episodes' / job['id'], 'watchdog_interrupted'))
        results = collect_results(root)
        prior = read_json(root / 'status.json') if (root / 'status.json').exists() else {}
        phase = ('interrupted' if stopped[0] else 'watchdog_deadline') if forced else prior.get('phase', 'controller_failed')
        if child.returncode and not forced:
            phase = 'controller_failed'
        status = {**prior, 'phase': phase, 'active': [], 'active_count': 0,
                  'resolved': len(results), 'queued': len(jobs) - len(results),
                  'updated_unix': time.time(), 'watchdog': outcome}
        write_json(root / 'status.json', status)
        backend.publish(root, plan, jobs, results, status)
        # Include final reconciliation/report writes, not just worker teardown.
        outcome.update(elapsed_seconds=time.monotonic() - started,
                       allocated_cpu_core_seconds=count * (time.monotonic() - started),
                       cleanup_tail_seconds=max(0., time.time() - plan['deadline_unix']),
                       finished_unix=time.time())
        outcome['cumulative_charged_cpu_core_seconds'] = (
            outcome['allocated_cpu_core_seconds'] + allocation['prior_charged_cpu_core_seconds'])
        write_json(root / 'watchdog_result.json', outcome)
        status['watchdog'] = outcome
        write_json(root / 'status.json', status)
        print(json.dumps({'final': status}), flush=True)
    if child.returncode and not forced:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'control', 'report'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--plan')
    parser.add_argument('--repo')
    args = parser.parse_args()
    if args.mode == 'launch':
        if not args.plan or not args.repo:
            parser.error('launch needs --plan and --repo')
        launch(args.plan, args.root, args.repo)
    elif args.mode == 'control':
        supervise(args.root)
    else:
        root = Path(args.root)
        publish(root, read_json(root / 'plan.json'), read_json(root / 'jobs.json'),
                collect_results(root), read_json(root / 'status.json'))


if __name__ == '__main__':
    main()
