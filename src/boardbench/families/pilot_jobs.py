"""Trusted queues for jobs executed by the sandbox PID 1, never on the host."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time


class ManagedJobs:
    def __init__(self, root, admissible=lambda: True, max_jobs=64):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root/'state.json').exists():
            raise RuntimeError('Interrupted job queue cannot automatically restart')
        self.jobs = {}
        self.sealed = False
        self.sandbox_pid = None
        self.admissible = admissible
        self.max_jobs = max_jobs
        self.lock = threading.RLock()
        self._save()

    def _save(self):
        target = self.root/'state.json'
        tmp = target.with_suffix('.tmp')
        with tmp.open('w') as f:
            json.dump({'sealed': self.sealed, 'jobs': self.jobs}, f, indent=2)
            f.flush(); os.fsync(f.fileno())
        tmp.replace(target)

    def _event(self, kind, **data):
        with (self.root/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(), event=kind, **data))+'\n')
            f.flush(); os.fsync(f.fileno())

    def start(self, argv):
        if (not isinstance(argv, list) or not argv or len(argv)>128
                or any(not isinstance(x, str) or '\0' in x for x in argv)
                or sum(map(len, argv))>32768):
            raise ValueError('argv must be a bounded list of strings')
        with self.lock:
            if self.sealed or not self.admissible():
                raise RuntimeError('job_admission_closed')
            if len(self.jobs)>=self.max_jobs:
                raise RuntimeError('job_count_limit')
            job_id = f'job-{len(self.jobs)+1:04d}'
            self.jobs[job_id] = dict(job_id=job_id, argv=argv, status='queued', submitted=time.time())
            self._save(); self._event('job_submitted', job_id=job_id, argv=argv)
            return dict(self.jobs[job_id])

    def active(self):
        with self.lock:
            return any(x['status'] in ('queued','dispatched','running') for x in self.jobs.values())

    def broker_allowed(self, pid):
        # SO_PEERCRED is supplied by the kernel, not the request. Only the
        # trusted namespace PID 1 descended from this supervised sandbox may
        # send execution events; participant children cannot spoof it.
        if self.sandbox_pid is None:
            return False
        try:
            status = dict(line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
            if int(status['NSpid'].split()[-1]) != 1:
                return False
            for _ in range(16):
                if pid == self.sandbox_pid:
                    return True
                status = dict(line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
                pid = int(status['PPid'])
        except (OSError, ValueError, KeyError):
            return False
        return False

    def poll(self):
        with self.lock:
            result=[]
            for job in self.jobs.values():
                if job['status']=='queued':
                    job['status']='dispatched'; result.append(dict(job))
            if result:self._save()
            return result

    def event(self, kind, job_id=None, **data):
        with self.lock:
            if kind=='controller_exit':
                self.sealed=True;self._event(kind, **data);self._save();return
            job=self.jobs.get(job_id)
            if job is None:raise ValueError('unknown_job')
            if kind=='job_started':
                if job['status']!='dispatched':raise ValueError('job_not_dispatched')
                job.update(status='running', started=time.time(), pid=data['pid'])
            elif kind=='job_exit':
                if job['status'] not in ('running','dispatched'):raise ValueError('job_already_ended')
                job.update(status='exited', ended=time.time(), exit_code=data['exit_code'], signal=data.get('signal'))
            elif kind=='job_output':
                stream=data['stream']
                if stream not in ('stdout','stderr'):raise ValueError('invalid_stream')
                payload=data['text'].encode('utf-8',errors='replace')
                path=self.root/f'{job_id}.{stream}'
                size=path.stat().st_size if path.exists() else 0
                with path.open('ab') as f:f.write(payload[:max(0,4*1024**2-size)])
                return
            else:raise ValueError('invalid_job_event')
            self._event(kind, job_id=job_id, **data);self._save()

    def status(self, job_id=None):
        with self.lock:
            if job_id is not None:
                if job_id not in self.jobs:raise ValueError('unknown_job')
                return dict(self.jobs[job_id])
            return dict(sealed=self.sealed, jobs=[dict(x) for x in self.jobs.values()])

    def output(self, job_id, stream='stdout', offset=0):
        with self.lock:
            if job_id not in self.jobs or stream not in ('stdout','stderr') or type(offset) is not int or offset<0:
                raise ValueError('invalid_output_request')
            path=self.root/f'{job_id}.{stream}'
            data=b''
            if path.exists():
                with path.open('rb') as f:f.seek(offset);data=f.read(16384)
            return dict(text=data.decode('utf-8',errors='replace'), next_offset=offset+len(data))

    def close(self, reason):
        with self.lock:
            self.sealed=True
            for job in self.jobs.values():
                if job['status'] in ('queued','dispatched','running'):
                    job.update(status='terminated', ended=time.time(), reason=reason)
                    self._event('job_terminated', job_id=job['job_id'], reason=reason)
            self._save()
