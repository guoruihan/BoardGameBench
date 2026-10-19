"""Drain a full frozen V1.2.4 dispatcher; run its unstarted jobs on more cores.

Only orchestration changes. Workers continue importing the parent's frozen
policy source. A SIGSTOP affects the old dispatcher, never its workers; it is
resumed with SIGTERM only after those workers finish. Separate watchdogs keep
the original absolute deadline. No pending job is run by both dispatchers.
"""
import argparse
from collections import deque
from copy import deepcopy
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

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v12.budget import kill_tree
from boardbench.v122.campaign import physical_cpus, incomplete
from boardbench.v124.report import publish as base_publish


def identity(pid):
    path = Path('/proc') / str(pid)
    try:
        stat = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start': stat[19], 'state': stat[0], 'ppid': int(stat[1]),
                'uid': path.stat().st_uid,
                'argv': (path / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process(ref):
    current = identity(ref['pid'])
    return current if current and all(current[k] == ref[k] for k in ('start', 'uid', 'argv')) else None


def live(ref):
    current = same_process(ref)
    return current is not None and current['state'] != 'Z'


def owns_root(ref, root):
    try:
        value = Path(ref['argv'][ref['argv'].index('--root') + 1])
        cwd = Path('/proc') / str(ref['pid']) / 'cwd'
        return (value if value.is_absolute() else cwd.resolve() / value).resolve() == root
    except (ValueError, IndexError, FileNotFoundError):
        return False


def send(ref, sig):
    if not same_process(ref) or ref['uid'] != os.getuid():
        raise RuntimeError('refuse signal: process identity changed')
    os.kill(ref['pid'], sig)


def safe_partition(jobs, started):
    ids = {j['id'] for j in jobs}
    if len(ids) != len(jobs) or not started <= ids:
        raise ValueError('invalid job partition')
    pending = [j for j in jobs if j['id'] not in started]
    touched = {j['environment_index'] for j in jobs if j['id'] in started}
    fresh = sorted({j['environment_index'] for j in pending} - touched)
    return pending, fresh


def pause_dispatcher(parent, timeout=45):
    """Pause only at a verifiably full queue frontier, otherwise immediately undo.

All worker slots must be occupied by exact live children and the job-file set
must equal resolved+active. Thus the frozen dispatcher's available deque is
empty: after TERM+CONT it cannot dispatch another job before checking stop.
"""
    allocation = read_json(parent / 'allocation.json')
    expected = len(allocation['worker_cpus'])
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        status = read_json(parent / 'status.json')
        if status['phase'] != 'running':
            raise RuntimeError('parent no longer running; do not pause stale PID')
        ref = identity(status['pid'])
        if (not ref or ref['uid'] != os.getuid() or not owns_root(ref, parent) or
            'boardbench.v124.campaign' not in ref['argv'] or 'control' not in ref['argv']):
            raise RuntimeError('parent controller identity mismatch')
        if status['active_count'] != expected or time.time() - status['updated_unix'] > 20:
            time.sleep(.25)
            continue
        send(ref, signal.SIGSTOP)
        accepted = False
        try:
            time.sleep(.03)
            if not same_process(ref) or same_process(ref)['state'] != 'T':
                raise RuntimeError('dispatcher did not stop')
            status = read_json(parent / 'status.json')
            resolved = {p.stem for p in (parent / 'resolved').glob('*.json')}
            started = {p.stem for p in (parent / 'jobs').glob('*.json')}
            active = []
            for row in status['active']:
                worker = identity(row['pid'])
                path = str(parent / 'jobs' / (row['id'] + '.json'))
                if (not worker or worker['state'] == 'Z' or worker['ppid'] != ref['pid'] or
                    worker['uid'] != os.getuid() or path not in worker['argv'] or
                    os.sched_getaffinity(worker['pid']) != {row['cpu']}):
                    break
                job = read_json(path)
                active.append({**worker, 'id': row['id'], 'cpu': row['cpu'],
                    'deadline_unix': status['updated_unix'] - row['elapsed_seconds'] + job['timeout_seconds']})
            if (len(active) != expected or len({r['cpu'] for r in active}) != expected or
                started != resolved | {r['id'] for r in active}):
                continue
            accepted = True
            return {'controller': ref, 'workers': active, 'started_ids': sorted(started),
                    'paused_unix': time.time(), 'status_at_pause': status}
        finally:
            if not accepted and same_process(ref):
                send(ref, signal.SIGCONT)
                time.sleep(.25)
        time.sleep(.25)
    raise RuntimeError('no safe full-worker frontier found; parent left running')


def finish_parent(handoff):
    if any(live(w) for w in handoff['workers']):
        return False
    controller = handoff['controller']
    if live(controller) and same_process(controller)['state'] == 'T':
        # Stopping flag is delivered before the old loop can schedule again.
        send(controller, signal.SIGTERM)
        send(controller, signal.SIGCONT)
    return not live(handoff['watchdog'])


def result_rows(root, origin, handoff):
    parent = Path(origin['parent'])
    inherited = {p.stem: read_json(p) for p in (parent / 'resolved').glob('*.json')}
    # Completed children of the paused dispatcher are not reaped yet. Read only
    # after exit, then preserve their final result in the expansion artifacts.
    for worker in handoff['workers']:
        if not live(worker) and worker['id'] not in inherited:
            job = read_json(parent / 'jobs' / (worker['id'] + '.json'))
            out = parent / 'episodes' / worker['id']
            path = out / 'result.json'
            row = read_json(path) if path.exists() else incomplete(job, out, 'parent_worker_exit_without_result')
            inherited[worker['id']] = row
    own = {p.stem: read_json(p) for p in (root / 'resolved').glob('*.json')}
    censored = root / 'inherited_censoring.json'
    for job_id, reason in (read_json(censored).items() if censored.exists() else []):
        job = read_json(parent / 'jobs' / (job_id + '.json'))
        inherited[job_id] = incomplete(job, parent / 'episodes' / job_id, reason)
    if set(own) & set(inherited):
        raise RuntimeError('duplicate execution ownership')
    return [{**r, 'execution_batch': 'parent'} for r in inherited.values()] + [
        {**r, 'execution_batch': 'expanded'} for r in own.values()]


def publish(root, origin, handoff, status, intervals):
    plan, jobs = read_json(root / 'plan.json'), read_json(root / 'jobs.json')
    rows = result_rows(root, origin, handoff)
    status.update(resolved=len(rows), updated_unix=time.time())
    write_json(root / 'status.json', status)
    summary = base_publish(root, plan, jobs, rows, status)
    now = time.time()
    cost = sum(len(i['cpus']) * max(0., i.get('end_unix', now) - i['start_unix']) for i in intervals)
    parent = Path(origin['parent'])
    a = read_json(parent / 'allocation.json')
    w = parent / 'watchdog_result.json'
    parent_cost = (read_json(w)['cumulative_charged_cpu_core_seconds'] if w.exists() else
        a['prior_charged_cpu_core_seconds'] + len(a['cpu_affinity']) * (now - a['started_unix']))
    accounting = {'expansion_reserved_cpu_core_seconds': cost, 'parent_cumulative_cpu_core_seconds': parent_cost,
                  'expansion_preflight_cpu_core_seconds': origin['preflight_charge'],
                  'total_charged_cpu_core_hours': (cost + parent_cost + origin['preflight_charge']) / 3600,
                  'gpu_hours': 0, 'intervals': intervals}
    summary.update(resource_charge=accounting, execution_batches={'parent': origin['parent'],
        'expanded': str(root), 'mixed_load_warning': True,
        'fresh_environment_indices': origin['fresh_environment_indices']})
    write_json(root / 'summary.json', summary)
    write_json(root / 'accounting.json', accounting)
    report = root / 'REPORT.md'
    report.write_text(report.read_text() + '\n## 扩容批次说明\n\n'
        '本表合并旧批次和扩容批次，CPU负载条件发生变化，不冒称固定并发条件。'
        '旧对局不重跑，新worker继续使用原冻结策略源码。\n\n'
        '完整未启动环境的独立结果见[fresh_only/REPORT.md](fresh_only/REPORT.md)。'
        '该分组按交接时的任务归属冻结，不根据分数挑选。\n\n'
        f'累计含旧运行及扩容预检：{accounting["total_charged_cpu_core_hours"]:.3f} CPU核时，GPU0。\n')
    fresh = set(origin['fresh_environment_indices'])
    if fresh:
        fresh_plan = deepcopy(plan)
        fresh_plan['environment_seeds'] = [plan['environment_seeds'][i] for i in sorted(fresh)]
        fresh_plan['panels'] = sorted({n for n in (8, 16, len(fresh)) if n <= len(fresh)})
        out = root / 'fresh_only'
        out.mkdir(exist_ok=True)
        fresh_jobs = [j for j in jobs if j['environment_index'] in fresh]
        fresh_rows = [r for r in rows if r['environment_index'] in fresh]
        base_publish(out, fresh_plan, fresh_jobs, fresh_rows, status)
    return rows


def control(root):
    origin, handoff = read_json(root / 'origin.json'), read_json(root / 'handoff.json')
    parent = Path(origin['parent'])
    if source_id() != origin['policy_source_id'] or digest(root / 'driver.py') != origin['driver_sha256']:
        raise RuntimeError('policy or driver snapshot mismatch')
    for name in ('plan', 'jobs', 'handoff'):
        if digest(root / f'{name}.json') != origin[f'{name}_sha256']:
            raise RuntimeError('frozen expansion manifest mismatch')
    jobs = read_json(root / 'jobs.json')
    pending, _ = safe_partition(jobs, set(handoff['started_ids']))
    queue, available, active = deque(pending), deque(origin['extra_cpus'][1:]), {}
    intervals = [{'start_unix': origin['started_unix'], 'cpus': origin['extra_cpus']}]
    parent_done, stopped = False, [False]
    borrowed = set()
    def stop(sig, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    phase, last = 'running', 0.
    def snapshot():
        inherited_active = sum(live(w) for w in handoff['workers'])
        status = {'phase': phase, 'pid': os.getpid(), 'active_count': len(active) + inherited_active,
            'expanded_active': len(active), 'inherited_active': inherited_active, 'queued': len(queue),
            'parent_finalized': parent_done, 'remaining_seconds': max(0., origin['deadline_unix'] - time.time()),
            'active': [{'pid': pid, 'id': r['job']['id'], 'cpu': r['cpu']} for pid, r in active.items()]}
        publish(root, origin, handoff, status, intervals)
        print(json.dumps({k: status[k] for k in ('phase', 'active_count', 'resolved', 'queued', 'parent_finalized')}), flush=True)
    try:
        while queue or active or not parent_done:
            if stopped[0] or time.time() >= origin['deadline_unix'] - 8:
                phase = 'interrupted' if stopped[0] else 'deadline'
            if not parent_done:
                # Borrow each old worker core only after its exact process exits.
                # The old reservation still accounts for it until guard handoff;
                # no duplicate resource charge and no need to idle drained cores.
                for w in handoff['workers']:
                    if not live(w) and w['cpu'] not in borrowed:
                        available.append(w['cpu'])
                        borrowed.add(w['cpu'])
                        write_json(root / 'borrowed_cores.json', {'cpus': sorted(borrowed), 'updated_unix': time.time()})
                if (root / 'resource_transfer.json').exists():
                    intervals[-1]['end_unix'] = time.time()
                    intervals.append({'start_unix': time.time(), 'cpus': origin['all_cpus']})
                    write_json(root / 'intervals.json', intervals)
                    available.extend(c for c in origin['old_cpus'] if c not in borrowed)
                    parent_done = True
                    write_json(root / 'parent_finalized.json', {'at_unix': time.time(), 'worker_cpus_added': origin['old_cpus']})
            while queue and available and phase == 'running':
                job, cpu = queue.popleft(), available.popleft()
                path, out = root / 'jobs' / (job['id'] + '.json'), root / 'episodes' / job['id']
                if (parent / 'jobs' / path.name).exists():
                    raise RuntimeError('parent dispatch resumed unexpectedly; refusing duplicate')
                write_json(path, job)
                stream = (root / 'logs' / (job['id'] + '.log')).open('x')
                child = subprocess.Popen(['taskset', '-c', str(cpu), sys.executable, '-u', '-m',
                    'boardbench.v124.episode', '--job', str(path), '--out', str(out)],
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[child.pid] = {'process': child, 'job': job, 'out': out, 'cpu': cpu,
                                     'stream': stream, 'started': time.monotonic()}
            for pid, value in list(active.items()):
                child, job = value['process'], value['job']
                elapsed, reason = time.monotonic() - value['started'], None
                if child.poll() is None and (phase != 'running' or elapsed > job['timeout_seconds']):
                    reason = phase if phase != 'running' else 'job_timeout'
                    kill_tree(pid)
                    child.wait(timeout=2)
                if child.poll() is not None:
                    path = value['out'] / 'result.json'
                    result = read_json(path) if path.exists() and reason is None else incomplete(
                        job, value['out'], reason or 'worker_exit_without_result')
                    if child.returncode and result['status'] == 'completed':
                        result.update(status='worker_nonzero_exit', score=None, natural_terminal=False)
                    result.update(returncode=child.returncode, supervised_seconds=elapsed, cpu=value['cpu'])
                    write_json(root / 'resolved' / (job['id'] + '.json'), result)
                    value['stream'].close()
                    available.append(value['cpu'])
                    del active[pid]
            if time.monotonic() - last >= 15:
                snapshot()
                last = time.monotonic()
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
        for pid, v in active.items():
            if v['process'].poll() is None:
                kill_tree(pid)
                v['process'].wait(timeout=2)
            write_json(root / 'resolved' / (v['job']['id'] + '.json'), incomplete(v['job'], v['out'], phase))
            v['stream'].close()
        active.clear()
        # Reservation remains owned by the outer guard until it finishes cleanup.
        write_json(root / 'intervals.json', intervals)
        snapshot()


def launch(parent, root, reserve, preflight_charge):
    started = time.time()
    parent, root = parent.resolve(), root.resolve()
    if root.exists() or root.parent != parent.parent or not (parent / 'allocation.json').exists():
        raise ValueError('fresh sibling output root and an existing parent required')
    cpus = physical_cpus()
    old = read_json(parent / 'allocation.json')
    if (old['host'] != os.uname().nodename or type(reserve) is not int or not 1 <= reserve < len(cpus)
        or not math.isfinite(preflight_charge) or preflight_charge < 0):
        raise ValueError('same host and positive physical reserve required')
    all_cpus = cpus[:len(cpus) - reserve]
    if not set(old['cpu_affinity']) < set(all_cpus):
        raise ValueError('expansion must keep old cores and reserve free physical cores')
    extra = [c for c in all_cpus if c not in old['cpu_affinity']]
    if len(extra) < 2 or os.getloadavg()[0] > len(cpus) - len(extra) + 4:
        raise RuntimeError('insufficient headroom for expansion')
    if not 30 < old['deadline_unix'] - time.time() <= 8 * 3600:
        raise ValueError('old deadline must remain in force')
    if source_id(parent / 'source') != old['source_id']:
        raise RuntimeError('parent frozen source modified')
    watchdog = identity(old['watchdog_pid'])
    if (not watchdog or watchdog['uid'] != os.getuid() or not owns_root(watchdog, parent)
        or 'launch' not in watchdog['argv']):
        raise RuntimeError('old watchdog identity mismatch')
    extension_lock = (root.parent / f'{os.uname().nodename}.expansion.lock').open('a')
    fcntl.flock(extension_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, {extra[0]})
    root.mkdir()
    for name in ('jobs', 'episodes', 'logs', 'resolved'):
        (root / name).mkdir()
    shutil.copy2(__file__, root / 'driver.py')
    for name in ('plan', 'jobs'):
        shutil.copy2(parent / f'{name}.json', root / f'{name}.json')
    handoff = pause_dispatcher(parent)
    handoff['watchdog'] = watchdog
    child = None
    try:
        jobs = read_json(root / 'jobs.json')
        pending, fresh = safe_partition(jobs, set(handoff['started_ids']))
        write_json(root / 'handoff.json', handoff)
        origin = {'parent': str(parent), 'started_unix': started, 'deadline_unix': old['deadline_unix'],
            'policy_source_id': old['source_id'], 'policy_source_path': str(parent / 'source/src'),
            'parent_git_head': old['git_head'], 'driver_sha256': digest(root / 'driver.py'),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'physical_core_count': len(cpus), 'reserved_physical_cores': reserve, 'all_cpus': all_cpus,
            'extra_cpus': extra, 'old_cpus': old['cpu_affinity'], 'pending_count': len(pending),
            'fresh_environment_indices': fresh, 'preflight_charge': preflight_charge,
            **{f'{name}_sha256': digest(root / f'{name}.json') for name in ('plan', 'jobs', 'handoff')}}
        write_json(root / 'origin.json', origin)
        write_json(parent / 'expansion_link.json', {'root': str(root), 'at_unix': time.time(),
            'controller_paused_for_drain': handoff['controller'], 'preserved_active_jobs': len(handoff['workers']),
            'new_physical_core_limit': len(all_cpus), 'reserved_physical_cores': reserve})
        env = {**os.environ, 'PYTHONPATH': origin['policy_source_path'], 'PYTHONNOUSERSITE': '1',
            'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
        child = subprocess.Popen([sys.executable, '-u', str(root / 'driver.py'), 'control', '--root', str(root)],
            env=env, start_new_session=True)
    except BaseException:
        if child is None and live(handoff['controller']):
            send(handoff['controller'], signal.SIGCONT)
        raise
    stopped, forced, legacy_lock = [False], False, None
    def stop(sig, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while child.poll() is None:
            # Parent drain supervision is independent of the new controller too.
            # A stalled new dispatcher must not strand finished old workers.
            for w in handoff['workers']:
                if live(w) and time.time() >= w['deadline_unix']:
                    path = root / 'inherited_censoring.json'
                    reasons = read_json(path) if path.exists() else {}
                    reasons[w['id']] = 'job_timeout'
                    write_json(path, reasons)
                    send(w, signal.SIGKILL)
            if finish_parent(handoff) and legacy_lock is None:
                legacy_lock = (parent.parent.parent / 'v122' / f'{os.uname().nodename}.cpu.lock').open('a')
                fcntl.flock(legacy_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                write_json(root / 'resource_transfer.json', {'watchdog_pid': os.getpid(), 'at_unix': time.time(),
                    'old_guard_exited': True, 'legacy_cpu_lock_held_through_cleanup': True})
            if stopped[0] or time.time() >= old['deadline_unix'] - 3:
                forced = True
                kill_tree(child.pid)
                break
            time.sleep(.2)
        child.wait(timeout=2)
    finally:
        if child.poll() is None:
            kill_tree(child.pid)
            child.wait(timeout=2)
        # Failure after claiming jobs cannot restore the old queue (duplicates).
        # Drain its workers, retaining original per-attempt and global deadlines.
        while not finish_parent(handoff):
            for w in handoff['workers']:
                if live(w) and (stopped[0] or time.time() >= min(w['deadline_unix'], old['deadline_unix'])):
                    path = root / 'inherited_censoring.json'
                    reasons = read_json(path) if path.exists() else {}
                    reasons[w['id']] = 'watchdog_interrupted'
                    write_json(path, reasons)
                    send(w, signal.SIGKILL)
            if time.time() >= old['deadline_unix'] + 10:
                break  # old independent watchdog also owns final cleanup
            time.sleep(.2)
        for path in (root / 'jobs').glob('*.json'):
            if not (root / 'resolved' / path.name).exists():
                job = read_json(path)
                write_json(root / 'resolved' / path.name, incomplete(job, root / 'episodes' / job['id'], 'watchdog_interrupted'))
        intervals = read_json(root / 'intervals.json') if (root / 'intervals.json').exists() else [
            {'start_unix': started, 'cpus': all_cpus if (root / 'parent_finalized.json').exists() else extra}]
        intervals[-1]['end_unix'] = time.time()
        status = read_json(root / 'status.json') if (root / 'status.json').exists() else {}
        phase = ('interrupted' if stopped[0] else 'watchdog_deadline') if forced else status.get('phase', 'controller_failed')
        if child.returncode and not forced:
            phase = 'controller_failed'
        status.update(phase=phase, active=[], active_count=0, expanded_active=0, inherited_active=0)
        publish(root, origin, handoff, status, intervals)
        outcome = {'returncode': child.returncode, 'forced': forced, 'finished_unix': time.time(),
                   'old_controller_still_live': live(handoff['controller']), 'old_watchdog_still_live': live(watchdog),
                   'elapsed_seconds': time.time() - started, 'gpu_seconds': 0}
        write_json(root / 'watchdog_result.json', outcome)
        print(json.dumps(outcome), flush=True)
    if child.returncode and not forced:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'control'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--reserve-physical', type=int, default=16)
    parser.add_argument('--preflight-core-seconds', type=float, default=0.)
    args = parser.parse_args()
    if args.mode == 'launch':
        if args.parent is None:
            parser.error('--parent required')
        launch(args.parent, args.root, args.reserve_physical, args.preflight_core_seconds)
    else:
        control(args.root.resolve())


if __name__ == '__main__':
    main()
