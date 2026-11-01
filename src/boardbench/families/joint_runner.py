"""Node-local CPU queue with cumulative worker-wall and deadline ceilings.

Uses single-core subprocesses and archives failures instead of inventing game
outcomes. Private host plans and launch scripts belong in ignored outputs/.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

from .evaluate import canonical_hash
from .play import write_json


def run(plan):
    root = Path(plan['output']); root.mkdir(parents=True, exist_ok=True)
    cpus = plan['cpus']; allowed = os.sched_getaffinity(0)
    if len(set(cpus)) != len(cpus) or not set(cpus) <= allowed:
        raise ValueError('Invalid affinity')
    signature = canonical_hash(plan)
    state_path = root / 'queue_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        plan_sha256=signature, finished={}, worker_seconds=0., active={}, started_unix=time.time())
    if state['plan_sha256'] != signature:
        raise ValueError('Resume must keep the same queue/budget/identity')
    if state.get('active'):
        # Abrupt failure must be inspected; do not duplicate still-running jobs.
        raise ValueError('Unclosed active processes; audit before resuming')
    lock = threading.Lock(); stop = threading.Event(); children = {}
    def on_signal(*_): stop.set()
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, on_signal)
    queue = iter([g for g in plan['groups'] if g['id'] not in state['finished']])
    def usage(now):
        return state['worker_seconds'] + sum(now-v['began_unix'] for v in state['active'].values())
    def worker(core):
        while not stop.is_set():
            with lock:
                if time.time() >= plan['deadline_unix']-5 or usage(time.time()) >= plan['worker_seconds_limit']-5*len(cpus):
                    stop.set(); break
                group = next(queue, None)
                if group is None: break
                directory = root / 'groups' / group['id']; directory.mkdir(parents=True, exist_ok=True)
                began = time.time()
                log = (directory/'run.log').open('a')
                child = subprocess.Popen(['taskset', '-c', str(core)] + group['argv'],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children[group['id']] = child
                state['active'][group['id']] = dict(pid=child.pid, core=core, began_unix=began,
                                                   timeout=group['timeout'])
                write_json(state_path, state)
            reason = None
            while child.poll() is None:
                now = time.time()
                with lock:
                    if now >= plan['deadline_unix']-5 or usage(now) >= plan['worker_seconds_limit']-5*len(cpus):
                        stop.set(); reason = 'global_budget'
                if stop.is_set() or now-began > group['timeout']:
                    reason = reason or ('stopped' if stop.is_set() else 'group_timeout')
                    try: os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError: pass
                    try: child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL); child.wait()
                    break
                stop.wait(.5)
            code = child.wait(); log.close()
            with lock:
                elapsed = time.time()-began
                state['worker_seconds'] += elapsed
                state['active'].pop(group['id']); children.pop(group['id'])
                result = dict(returncode=code, reason=reason, wall_seconds=elapsed,
                              core=core, ended_unix=time.time())
                # Completed child groups may contain explicit invalid games, preserved in results.
                if code == 0: state['finished'][group['id']] = result
                write_json(directory/'attempt.json', result)
                write_json(state_path, state)
            print(json.dumps(dict(group=group['id'], **result)), flush=True)
    with ThreadPoolExecutor(max_workers=len(cpus)) as executor:
        futures = [executor.submit(worker, core) for core in cpus]
        for f in futures: f.result()
    state.update(ended_unix=time.time(), stopped=stop.is_set(), total_groups=len(plan['groups']))
    write_json(state_path, state)
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args(); result = run(json.loads(a.plan.read_text()))
    print(json.dumps({k: v for k, v in result.items() if k not in ('finished', 'active')}))


if __name__ == '__main__': main()
