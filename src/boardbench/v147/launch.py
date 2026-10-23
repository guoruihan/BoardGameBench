"""Immutable ten-hour windows, per-host half-CPU cap and detached deadline guard."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from boardbench.artifacts.store import read_json,write_json,digest,source_id,source_files
from boardbench.v122.campaign import physical_cpus
from boardbench.v12.budget import kill_tree


def guard(root,role,*,module='boardbench.v147.launch'):
    root=Path(root);c=read_json(root/'resolved_config.json');launch=read_json(root/'launch.json')
    if c['role']!=role or digest(root/'resolved_config.json')!=launch['resolved_config_sha256']:
        raise ValueError('immutable role/config/deadline required')
    if (root/'summary.json').exists() or time.time()>=c['deadline_unix']:raise ValueError('completed/expired; cannot renew')
    env={**os.environ,'PYTHONPATH':str(root/'source/src'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    child=subprocess.Popen([sys.executable,'-u','-m',module,'--out',str(root),
        '--role',role,'--execute'],env=env,start_new_session=True)
    # Child must discover the full host physical topology BEFORE its own pin;
    # inheriting the guardian's one-core mask would invalidate half-host checks.
    os.sched_setaffinity(0,{launch['cpus'][-1]})
    def stop(*_):
        if child.poll() is None:child.send_signal(signal.SIGTERM)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    signaled=False
    while child.poll() is None:
        now=time.time()
        if now>=c['deadline_unix']-60 and not signaled:stop();signaled=True
        if now>=c['deadline_unix']:
            kill_tree(child.pid);child.wait()
            write_json(root/'guard_deadline.json',dict(time=time.time(),deadline_unix=c['deadline_unix']));break
        time.sleep(min(5,max(.05,c['deadline_unix']-now)))
    if child.returncode:raise SystemExit(child.returncode)


def launch(repo,root,config,role):
    repo=Path(repo).resolve();root=Path(root).resolve();c=read_json(config);cpus=physical_cpus()
    if (c['role']!=role or os.uname().nodename!=c['host'] or c['deadline_unix']-c['started_unix']!=36000
            or time.time()>=c['deadline_unix']-3600 or not 2<=c['cpu_cores']<=len(cpus)//2):
        raise ValueError('ten-hour immutable window and authorized host/half-CPU required')
    if os.getloadavg()[0]>len(cpus)*.5:raise RuntimeError('busy host; no preemption')
    if role=='learning' and (len(c['gpu_uuids'])!=4 or len(set(c['gpu_uuids']))!=4
            or set(c['gpu_uuids'])!=set(c['authorized_gpu_uuids']) or c['cpu_cores']!=57 or c['workers']!=13):
        raise ValueError('four independent authorized GPUs and57 CPUs required')
    root.mkdir(parents=True,exist_ok=False)
    for name in ('logs','configs','rounds','decisions','jobs','episodes','selection','confirmation'):(root/name).mkdir()
    for name in source_files(repo):
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    c['repo']=str(repo);write_json(root/'config.json',c);write_json(root/'resolved_config.json',c)
    write_json(root/'launch.json',dict(source_id=source_id(repo),source_commit=subprocess.check_output(
        ['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),cpus=cpus[:c['cpu_cores']],
        config_sha256=digest(root/'config.json'),resolved_config_sha256=digest(root/'resolved_config.json'),created_unix=time.time()))
    guard(root,role)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo');p.add_argument('--config');p.add_argument('--out',required=True)
    p.add_argument('--role',choices=['search','learning'],required=True);p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.execute:
        if a.role=='search':
            from .search_campaign import execute
        else:
            from .campaign import execute
        execute(a.out)
    elif a.resume:guard(Path(a.out).resolve(),a.role)
    else:launch(a.repo,a.out,a.config,a.role)
