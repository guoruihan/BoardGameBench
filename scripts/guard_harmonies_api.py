#!/usr/bin/env python3
"""Live shared API admission guard; never generates model health probes.

Existing gateways already honor the same transport file lock. Holding it blocks
new outbound attempts while existing requests and local work can finish. No
participant process, quota, request timeout or environment source is changed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import sqlite3
import time
from urllib.parse import urlsplit


def write_json(path,value):
    temporary=path.with_suffix('.part')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def transport_errors(root,seen):
    errors=[]
    for gateway in Path(root).glob('*/gateway'):
        for path in (gateway/'requests').glob('*/*_error.json'):
            if path.name not in ('input_tokens_error.json','generate_error.json'):continue
            try:
                stat=path.stat();key=(str(path),stat.st_mtime_ns,stat.st_size)
                if key in seen:continue
                data=json.loads(path.read_text())
            except (OSError,ValueError):continue
            seen.add(key)
            kind=data.get('kind');status=data.get('status')
            # A local request timeout while waiting on our lock is not another
            # upstream failure and must not keep extending the global hold.
            if (kind in ('URLError','OSError','TimeoutError','ConnectionError','RemoteDisconnected','SSLError')
                    or status in (408,429) or isinstance(status,int) and status>=500):
                errors.append(dict(source=str(path.relative_to(root)),kind=kind,http_status=status))
        database=gateway/'gateway.sqlite'
        if not database.exists():continue
        try:
            with sqlite3.connect(f'file:{database}?mode=ro',uri=True,timeout=.2) as db:
                unknown=db.execute("SELECT id,data FROM requests WHERE state='unknown'").fetchall()
            for request_id,raw in unknown:
                key=(str(database),request_id)
                if key in seen:continue
                data=json.loads(raw);seen.add(key)
                if data.get('error') in ('upstream_or_transport_failure','request_timeout'):
                    errors.append(dict(source=str(database.relative_to(root)),kind='unknown_transport_usage',request_id=request_id))
        except (sqlite3.Error,ValueError):continue
    return errors


class AdmissionHold:
    def __init__(self,path,deadline):
        self.path=Path(path);self.deadline=deadline;self.stream=None

    def hold(self):
        if self.stream is not None:return
        stream=self.path.open('a+')
        try:
            while True:
                if time.time()>=self.deadline:raise TimeoutError('study deadline reached')
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:time.sleep(.05)
            # If this guard crashes, leave a future marker so gateways fail
            # closed rather than resume a request burst after the lock releases.
            self.stream=stream
            self._mark(time.monotonic()+max(0,self.deadline-time.time())+120)
        except BaseException:
            stream.close();raise

    def _mark(self,value):
        self.stream.seek(0);self.stream.truncate();self.stream.write(str(value));self.stream.flush()

    def release(self,allow_requests):
        if self.stream is None:return
        if allow_requests:self._mark(time.monotonic())
        fcntl.flock(self.stream,fcntl.LOCK_UN);self.stream.close();self.stream=None


def study_finished(root):
    """The old coordinator may stop while an advance-dispatched worker lives."""
    root=Path(root)
    if not (root/'finished.json').exists():return False
    for task in list(root.glob('dev-*'))+list(root.glob('eval-*')):
        if not task.is_dir():continue
        path=task/'experiment/state.json'
        if path.exists():
            if json.loads(path.read_text()).get('status')=='running':return False
        elif not (task/'worker_completion.json').exists():return False
    return True


def guard(root,private_upstream,*,cooldown=60,dns_interval=60,poll=1,
          clock=time.time,sleep=time.sleep,resolve=socket.getaddrinfo):
    root=Path(root);plan=json.loads((root/'preregistered_plan.json').read_text())
    deadline=plan['deadline'];transport=root/'private/api_clock'
    config=json.loads(Path(private_upstream).read_text());endpoint=urlsplit(config['base_url'])
    state_path=root/'api_guard_state.json';events=root/'api_guard_events.jsonl'
    def event(kind,**fields):
        with events.open('a') as stream:stream.write(json.dumps(dict(event=kind,time=clock(),**fields))+'\n')
    def state(mode,**fields):
        write_json(state_path,dict(mode=mode,pid=os.getpid(),updated=clock(),deadline=deadline,**fields))
    with (root/'api_guard.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        hold=AdmissionHold(transport,deadline);seen=set();next_check=0.;heartbeat=0.
        event('guard_started',cooldown_seconds=cooldown,dns_retry_seconds=dns_interval,poll_seconds=poll)
        state('normal');clean_end=False
        try:
            while clock()<deadline and not study_finished(root):
                errors=transport_errors(root,seen)
                if errors:
                    hold.hold();next_check=max(next_check,clock()+cooldown)
                    event('all_new_requests_held',errors=errors,next_check=next_check)
                    state('waiting',next_check=next_check)
                if hold.stream is not None and clock()>=next_check:
                    try:
                        addresses=resolve(endpoint.hostname,endpoint.port or (443 if endpoint.scheme=='https' else 80),type=socket.SOCK_STREAM)
                        if not addresses:raise OSError('empty resolver response')
                    except OSError as error:
                        next_check=clock()+dns_interval
                        event('dns_still_unavailable',error_type=type(error).__name__,errno=error.errno,next_check=next_check)
                        state('waiting',next_check=next_check)
                    else:
                        # DNS success permits paced traffic; it does not prove
                        # model health. Another transport failure holds everyone.
                        hold.release(True);next_check=0.
                        event('paced_requests_resumed_after_dns',model_health_confirmed=False)
                        state('normal')
                if clock()-heartbeat>=10:
                    state('waiting' if hold.stream is not None else 'normal',next_check=next_check or None)
                    heartbeat=clock()
                sleep(min(poll,max(0,deadline-clock())))
            clean_end=True
            state('finished' if study_finished(root) else 'deadline')
            event('guard_finished')
        finally:
            if not clean_end and hold.stream is None and time.time()<deadline:hold.hold()
            hold.release(allow_requests=clean_end and study_finished(root))
            if not clean_end:
                state('guard_error');event('guard_error_fail_closed')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--private-upstream',required=True,type=Path)
    parser.add_argument('--cooldown-seconds',type=float,default=60)
    parser.add_argument('--dns-interval-seconds',type=float,default=60)
    args=parser.parse_args()
    if not 30<=args.cooldown_seconds<=300 or not 30<=args.dns_interval_seconds<=300:
        parser.error('cooldown and DNS interval must be between 30 and 300 seconds')
    guard(args.root,args.private_upstream,cooldown=args.cooldown_seconds,dns_interval=args.dns_interval_seconds)


if __name__=='__main__':main()
