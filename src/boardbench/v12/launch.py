"""Durable watchdog: freeze source, pin resources, enforce one global deadline."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from boardbench.artifacts.store import digest, read_json, source_files, source_id, write_json
from .budget import kill_tree


def physical_cores(count):
    selected, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        identity = ((root / 'physical_package_id').read_text().strip(), (root / 'core_id').read_text().strip())
        if identity not in seen:
            selected.append(cpu); seen.add(identity)
        if len(selected) == count:
            return selected
    raise RuntimeError('not enough distinct physical CPU cores')


def idle_gpu(index):
    result = subprocess.run(['nvidia-smi', f'--id={index}',
                            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=10)
    uuid, used, utilization = [part.strip() for part in result.stdout.strip().split(',')]
    if int(used) > 500 or int(utilization) > 5:
        raise RuntimeError(f'requested GPU {index} is not idle; no workload was stopped')
    return uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True); parser.add_argument('--root', required=True)
    parser.add_argument('--repo', required=True); parser.add_argument('--cpu-smoke', action='store_true')
    args = parser.parse_args()
    # Begins before preflight, source copy, CUDA initialization and every experiment phase.
    started = time.monotonic()
    plan, root, repo = read_json(args.plan), Path(args.root).resolve(), Path(args.repo).resolve()
    if not root.is_relative_to(repo / 'outputs/v12') or root.exists():
        raise ValueError('a new run directory under outputs/v12 is required')
    if plan['cpu_cores'] != 8 or not 0 < plan['budget_seconds'] <= 14400:
        raise ValueError('pilot allocation must be eight CPU cores and at most four hours')
    root.parent.mkdir(parents=True, exist_ok=True)
    lock = (root.parent / 'experiment.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu = '' if args.cpu_smoke else idle_gpu(plan['gpu_index'])
    if os.getloadavg()[0] > len(os.sched_getaffinity(0)) * .5:
        raise RuntimeError('host CPU load is too high for this pilot')
    cores = physical_cores(plan['cpu_cores'])
    os.sched_setaffinity(0, cores)
    root.mkdir()
    os.environ.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                      OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONNOUSERSITE='1',
                      PYTHONUNBUFFERED='1', BOARDBENCH_STARTED_MONOTONIC=str(started))
    deadline = started + plan['budget_seconds']
    for relative in source_files(repo):
        target = root / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, target)
    write_json(root / 'plan.json', plan)
    os.environ['PYTHONPATH'] = str(root / 'source/src')
    now = time.time()
    write_json(root / 'allocation.json', {'started_unix': now - (time.monotonic() - started),
               'deadline_unix': now + deadline - time.monotonic(), 'budget_seconds': plan['budget_seconds'],
               'cpu_affinity': cores, 'gpu_uuid': gpu, 'gpu_index': plan['gpu_index'],
               'host': os.uname().nodename, 'source_id': source_id(root / 'source'),
               'plan_sha256': digest(root / 'plan.json'), 'watchdog_pid': os.getpid(),
               'accounting': 'entire reservation incl startup, failed attempts, diagnosis, teachers, evaluation and export',
               'original_git_head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()})
    command = [sys.executable, '-m', 'boardbench.v12.experiment', '--plan', str(root / 'plan.json'),
               '--root', str(root), '--repo', str(repo), '--device', 'cpu' if args.cpu_smoke else 'cuda']
    child = subprocess.Popen(command, start_new_session=True)
    interrupted = [False]
    def stopping(signum, frame):
        interrupted[0] = True
    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    timed_out = False
    try:
        while child.poll() is None:
            # Reserve two seconds of the same allocation for tree teardown and final logs.
            if time.monotonic() >= deadline - 2 or interrupted[0]:
                timed_out = not interrupted[0]
                kill_tree(child.pid)
                break
            time.sleep(min(.25, max(.001, deadline - 2 - time.monotonic())))
        child.wait(timeout=2)
    finally:
        if child.poll() is None:
            kill_tree(child.pid)
        elapsed = time.monotonic() - started
        value = {'returncode': child.poll(), 'timeout': timed_out, 'interrupted': interrupted[0],
                 'elapsed_seconds': elapsed, 'allocated_gpu_seconds': 0 if args.cpu_smoke else elapsed,
                 'allocated_cpu_core_seconds': 8 * elapsed, 'cleanup_tail_seconds': max(0., elapsed - plan['budget_seconds'])}
        write_json(root / 'watchdog_result.json', value)
        if timed_out or interrupted[0] or child.returncode:
            status = {'phase': 'budget_exhausted' if timed_out else 'stopped', 'experiment_root': str(root),
                      'partial_results_retained': True, **value}
            write_json(root / 'status.json', status)
            write_json(repo / 'outputs/v12/latest_status.json', status)
    print(value, flush=True)
    if child.returncode:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
