"""Bounded diagnostic queue sharing the existing V1.2.2 CPU reservation lock."""
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
from boardbench.v122.campaign import physical_cpus
from .bank import validate_bank
from .report import publish


def prepare_plan(raw, bank, manifest):
    validate_bank(bank, manifest)
    plan = dict(raw)
    for key in ('discovery_samples', 'confirmation_samples', 'width', 'cpu_cores'):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(f'positive integer {key} required')
    if plan['width'] != 6:
        raise ValueError('historical top6 is the registered pruning control')
    for key in ('run_seconds', 'job_timeout_seconds'):
        if not math.isfinite(plan[key]) or not 0 < plan[key] <= 8 * 3600:
            raise ValueError(f'finite bounded {key} required')
    panels = plan['panels']
    if not panels or any(type(n) is not int or n < 1 for n in panels) or panels != sorted(set(panels)):
        raise ValueError('positive increasing panels required')
    if panels[-1] != bank['environments']:
        raise ValueError('panels must cover exactly the bank environments')
    return {**plan, 'protocol': 'local_teacher_diagnostic_v123', 'bank_sha256': bank['sha256'],
            'split': bank['split'], 'split_sha256': manifest['sha256']}


def make_jobs(plan, bank):
    return [{**state, 'seed': plan['seed'], 'width': plan['width'],
             'discovery_samples': plan['discovery_samples'], 'confirmation_samples': plan['confirmation_samples'],
             'timeout_seconds': plan['job_timeout_seconds']} for state in bank['states']]


def incomplete(job, directory, reason):
    path = Path(directory) / 'progress.json'
    result = read_json(path) if path.exists() else {k: v for k, v in job.items() if k != 'observation'}
    return {**result, 'status': reason, 'error': result.get('error') or reason,
            'recorded_counters_may_be_lower_bounds': True}


def collect(root):
    return [read_json(p) for p in sorted((Path(root) / 'resolved').glob('*.json'))]


def supervise(root):
    root = Path(root)
    plan, bank = read_json(root / 'plan.json'), read_json(root / 'bank.json')
    allocation = read_json(root / 'allocation.json')
    if source_id() != allocation['source_id']:
        raise ValueError('controller must use frozen source')
    for name in ('plan', 'bank', 'splits', 'jobs'):
        if digest(root / f'{name}.json') != allocation[f'{name}_file_sha256']:
            raise ValueError(f'{name} file hash mismatch')
    jobs = read_json(root / 'jobs.json')
    if jobs != make_jobs(plan, bank):
        raise ValueError('job matrix differs from frozen bank/plan')
    available, queue = deque(allocation['worker_cpus']), deque(jobs)
    active, results = {}, []
    deadline = time.monotonic() + max(0., allocation['deadline_unix'] - time.time())
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    phase, last_snapshot = 'running', 0.
    def snapshot():
        status = {'phase': phase, 'pid': os.getpid(), 'updated_unix': time.time(),
                  'queued': len(queue), 'resolved': len(results), 'active_count': len(active),
                  'remaining_seconds': max(0., deadline - time.monotonic()),
                  'active': [{'id': v['job']['id'], 'pid': pid, 'cpu': v['cpu'],
                              'elapsed_seconds': time.monotonic() - v['started']} for pid, v in active.items()]}
        write_json(root / 'status.json', status)
        publish(root, plan, bank, jobs, results, status)
        print(json.dumps({k: status[k] for k in ('phase', 'queued', 'resolved', 'active_count')}), flush=True)
    try:
        while queue or active:
            if time.monotonic() >= deadline - 8 or stopped[0]:
                phase = 'interrupted' if stopped[0] else 'deadline'
            while queue and available and phase == 'running':
                job, cpu = queue.popleft(), available.popleft()
                job_path, out = root / 'jobs' / f'{job["id"]}.json', root / 'episodes' / job['id']
                write_json(job_path, job)
                stream = (root / 'logs' / f'{job["id"]}.log').open('x')
                child = subprocess.Popen(['taskset', '-c', str(cpu), sys.executable, '-u', '-m',
                    'boardbench.v123.diagnostic', '--job', str(job_path), '--out', str(out)],
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[child.pid] = dict(process=child, job=job, cpu=cpu, out=out,
                                         stream=stream, started=time.monotonic())
            for pid, value in list(active.items()):
                child, job = value['process'], value['job']
                elapsed, reason = time.monotonic() - value['started'], None
                if child.poll() is None and (phase != 'running' or elapsed >= job['timeout_seconds']):
                    reason = phase if phase != 'running' else 'job_timeout'
                    kill_tree(pid)
                    child.wait(timeout=2)
                if child.poll() is not None:
                    path = value['out'] / 'result.json'
                    result = read_json(path) if path.exists() and reason is None else incomplete(
                        job, value['out'], reason or 'worker_exit_without_result')
                    if child.returncode and result['status'] == 'completed':
                        result.update(status='worker_nonzero_exit')
                    result.update(returncode=child.returncode, supervised_seconds=elapsed, cpu=value['cpu'])
                    write_json(root / 'resolved' / f'{job["id"]}.json', result)
                    results.append(result)
                    value['stream'].close()
                    available.append(value['cpu'])
                    del active[pid]
            if time.monotonic() - last_snapshot >= 15:
                snapshot()
                last_snapshot = time.monotonic()
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
            row = incomplete(value['job'], value['out'], phase)
            write_json(root / 'resolved' / f'{value["job"]["id"]}.json', row)
            results.append(row)
            value['stream'].close()
            del active[pid]
        snapshot()


def launch(plan_path, root, repo):
    created, created_cpu, created_unix = time.monotonic(), time.process_time(), time.time()
    root, repo = Path(root).resolve(), Path(repo).resolve()
    if not root.is_relative_to(repo / 'outputs/v123') or root.exists():
        raise ValueError('new output directory under outputs/v123 required')
    raw = read_json(plan_path)
    if not math.isfinite(raw['queue_until_unix']) or not 0 < raw['queue_until_unix'] - time.time() <= 3 * 3600:
        raise ValueError('queue requires a future finite deadline within three hours')
    cpus = physical_cpus()
    if type(raw['cpu_cores']) is not int or not 2 <= raw['cpu_cores'] <= len(cpus) // 2:
        raise ValueError('controller and workers limited to half the physical CPU cores')
    bank_path, splits_path = repo / raw['state_bank'], repo / raw['split_manifest']
    bank, manifest = read_json(bank_path), read_json(splits_path)
    plan = prepare_plan(raw, bank, manifest)
    costs = []
    for relative in raw.get('preflight_cost_receipts', []):
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo / 'outputs/v123'):
            raise ValueError('preflight receipt must be inside outputs/v123')
        receipt = read_json(path)
        cost = receipt['charged_cpu_core_seconds']
        if not math.isfinite(cost) or cost < 0:
            raise ValueError('invalid preflight cost')
        costs.append({'path': str(path), 'sha256': digest(path), 'receipt': receipt})
    cores = cpus[:plan['cpu_cores']]
    original_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {cores[0]})  # queue shares the old controller CPU, not an extra core
    root.mkdir(parents=True)
    for name in ('logs', 'jobs', 'episodes', 'resolved'):
        (root / name).mkdir()
    for relative in source_files(repo):
        destination = root / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, destination)
    write_json(root / 'source_hashes.json', source_files(root / 'source'))
    write_json(root / 'plan.json', plan)
    shutil.copy2(bank_path, root / 'bank.json')
    shutil.copy2(splits_path, root / 'splits.json')
    jobs = make_jobs(plan, bank)
    write_json(root / 'jobs.json', jobs)
    allocation = {'source_id': source_id(root / 'source'), 'host': os.uname().nodename,
                  'available_physical_core_count': len(cpus), 'cpu_affinity': cores, 'worker_cpus': cores[1:],
                  'controller_cpu': cores[0], 'watchdog_pid': os.getpid(), 'created_unix': created_unix,
                  'gpu_seconds': 0, 'prior_costs': costs,
                  'prior_charged_cpu_core_seconds': sum(c['receipt']['charged_cpu_core_seconds'] for c in costs),
                  'git_head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                  'tracked_diff': subprocess.check_output(['git', '-C', str(repo), 'diff', '--stat'], text=True),
                  **{f'{name}_file_sha256': digest(root / f'{name}.json') for name in ('plan', 'bank', 'splits', 'jobs')}}
    write_json(root / 'allocation.json', allocation)
    # Compatibility is intentional: NEVER run alongside the old 64-core campaign.
    lock_path = repo / 'outputs/v122' / f'{os.uname().nodename}.cpu.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    queued_status = {'phase': 'queued', 'pid': os.getpid(), 'active_count': 0, 'active': [],
                     'queued': len(jobs), 'resolved': 0, 'updated_unix': time.time()}
    publish(root, plan, bank, jobs, [], queued_status)
    with lock_path.open('a') as lock:
        acquired = False
        last_queue_update = 0.
        while time.time() < plan['queue_until_unix'] and not stopped[0]:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            if acquired and os.getloadavg()[0] <= len(cpus) * .5:
                break
            if acquired:
                fcntl.flock(lock, fcntl.LOCK_UN)
                acquired = False
            if time.monotonic() - last_queue_update >= 15:
                queued_status.update(updated_unix=time.time(), wait_reason='old_cpu_lock_or_host_load',
                                     queue_remaining_seconds=max(0., plan['queue_until_unix'] - time.time()))
                write_json(root / 'status.json', queued_status)
                print(json.dumps(queued_status), flush=True)
                last_queue_update = time.monotonic()
            time.sleep(.2)
        queue_cost = {'elapsed_seconds': time.monotonic() - created,
                      'cpu_seconds': time.process_time() - created_cpu,
                      'charged_cpu_core_seconds': time.monotonic() - created,
                      'accounting': 'one shared controller-core wall upper bound including snapshot/setup/queue',
                      'acquired': acquired, 'lock_path': str(lock_path), 'finished_unix': time.time()}
        write_json(root / 'queue_result.json', queue_cost)
        if not acquired or stopped[0]:
            queued_status.update(phase='interrupted' if stopped[0] else 'queue_expired', updated_unix=time.time())
            write_json(root / 'status.json', queued_status)
            publish(root, plan, bank, jobs, [], queued_status)
            os.sched_setaffinity(0, original_affinity)
            return
        started = time.monotonic()
        allocation.update(started_unix=time.time(), deadline_unix=time.time() + plan['run_seconds'])
        write_json(root / 'allocation.json', allocation)
        env = {**os.environ, 'PYTHONPATH': str(root / 'source/src'), 'PYTHONNOUSERSITE': '1',
               'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
               'NUMEXPR_NUM_THREADS': '1', 'CUDA_VISIBLE_DEVICES': '', 'PYTHONUNBUFFERED': '1'}
        child = subprocess.Popen([sys.executable, '-u', '-m', 'boardbench.v123.campaign', 'control',
                                  '--root', str(root)], env=env, start_new_session=True)
        forced = False
        try:
            while child.poll() is None:
                if time.monotonic() >= started + plan['run_seconds'] - 3 or stopped[0]:
                    forced = True
                    kill_tree(child.pid)
                    break
                time.sleep(.2)
            child.wait(timeout=2)
        finally:
            if child.poll() is None:
                kill_tree(child.pid)
                child.wait(timeout=2)
            for path in (root / 'jobs').glob('*.json'):
                if not (root / 'resolved' / path.name).exists():
                    job = read_json(path)
                    write_json(root / 'resolved' / path.name, incomplete(
                        job, root / 'episodes' / job['id'], 'watchdog_interrupted'))
            results = collect(root)
            status = read_json(root / 'status.json') if (root / 'status.json').exists() else {}
            phase = ('interrupted' if stopped[0] else 'watchdog_deadline') if forced else status.get('phase', 'controller_failed')
            if child.returncode and not forced:
                phase = 'controller_failed'
            status.update(phase=phase, active=[], active_count=0, resolved=len(results),
                          queued=len(jobs) - len(results), updated_unix=time.time())
            write_json(root / 'status.json', status)
            publish(root, plan, bank, jobs, results, status)
            elapsed = time.monotonic() - started
            outcome = {'returncode': child.returncode, 'forced_cleanup': forced, 'interrupted': stopped[0],
                       'elapsed_seconds': elapsed, 'allocated_cpu_core_seconds': len(cores) * elapsed,
                       'cleanup_tail_seconds': max(0., time.time() - allocation['deadline_unix']),
                       'finished_unix': time.time(), 'gpu_seconds': 0}
            outcome['cumulative_charged_cpu_core_seconds'] = (outcome['allocated_cpu_core_seconds'] +
                queue_cost['charged_cpu_core_seconds'] + allocation['prior_charged_cpu_core_seconds'])
            write_json(root / 'watchdog_result.json', outcome)
            status['watchdog'] = outcome
            write_json(root / 'status.json', status)
            os.sched_setaffinity(0, original_affinity)
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
        publish(root, read_json(root / 'plan.json'), read_json(root / 'bank.json'),
                read_json(root / 'jobs.json'), collect(root), read_json(root / 'status.json'))


if __name__ == '__main__':
    main()
