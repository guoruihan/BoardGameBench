"""One wall-clock ledger for every phase; only scoped descendant processes killed."""
import os
from pathlib import Path
import signal
import subprocess
import time

from boardbench.artifacts.store import write_json


def descendants(pid):
    entries = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            value = path.read_text().rsplit(')', 1)[1].split()
            entries[int(path.parent.name)] = (int(value[1]), value[19])
        except (OSError, ValueError, IndexError):
            continue
    result = []
    def visit(parent):
        for child, (ppid, started) in entries.items():
            if ppid == parent:
                visit(child); result.append((child, started))
    visit(pid)
    if pid in entries:
        result.append((pid, entries[pid][1]))
    return result


def kill_tree(pid):
    targets = descendants(pid)
    for child, started in targets:
        try:
            # Guard against accidentally signaling a reused unrelated PID.
            actual = Path(f'/proc/{child}/stat').read_text().rsplit(')', 1)[1].split()[19]
            if actual == started:
                os.kill(child, signal.SIGKILL)
        except (OSError, IndexError):
            pass


class Ledger:
    def __init__(self, root, seconds, *, status_path=None):
        self.root = Path(root)
        self.started = float(os.environ.get('BOARDBENCH_STARTED_MONOTONIC', time.monotonic()))
        self.deadline = self.started + seconds
        self.seconds = seconds
        self.status_path = Path(status_path) if status_path else self.root / 'status.json'
        self.phases = []
        self.child = None
        self.update('initializing')

    def remaining(self):
        return max(0., self.deadline - time.monotonic())

    def update(self, phase, **extra):
        status = {'phase': phase, 'experiment_root': str(self.root.resolve()),
                  'pid': os.getpid(), 'updated_unix': time.time(),
                  'elapsed_seconds': time.monotonic() - self.started,
                  'remaining_seconds': self.remaining(), 'budget_seconds': self.seconds,
                  'cpu_affinity': sorted(os.sched_getaffinity(0)),
                  'allocated_gpu_seconds': time.monotonic() - self.started,
                  'allocated_cpu_core_seconds': len(os.sched_getaffinity(0)) * (time.monotonic() - self.started),
                  'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'phases': self.phases, **extra}
        write_json(self.root / 'status.json', status)
        if self.status_path != self.root / 'status.json':
            write_json(self.status_path, status)

    def run(self, name, command, maximum, reserve=0.):
        allowance = min(maximum, self.remaining() - reserve)
        if allowance <= 0:
            raise TimeoutError('experiment budget reserved for final evaluation/cleanup')
        started = time.monotonic()
        self.update(name, command=command)
        log = self.root / 'logs' / f'{name}.log'
        log.parent.mkdir(parents=True, exist_ok=True)
        timeout = False
        with log.open('x') as stream:
            self.child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            tick = 0.
            try:
                while self.child.poll() is None:
                    now = time.monotonic()
                    if now - started >= allowance or now >= self.deadline:
                        timeout = True; kill_tree(self.child.pid); break
                    if now - tick > 10:
                        self.update(name, child_pid=self.child.pid, log=str(log))
                        tick = now
                    time.sleep(min(.25, max(.001, allowance - (now - started))))
                self.child.wait(timeout=2)
            finally:
                if self.child.poll() is None:
                    kill_tree(self.child.pid)
                code = self.child.poll()
                self.child = None
        row = {'name': name, 'seconds': time.monotonic() - started,
               'returncode': code, 'timeout': timeout, 'log': str(log)}
        self.phases.append(row)
        self.update(name + '_finished')
        if timeout:
            raise TimeoutError(name)
        if code:
            raise RuntimeError(f'{name} failed; inspect {log}')
        return row
