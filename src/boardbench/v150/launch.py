"""Freeze a new explicitly authorized window; scoped deadline guard, never renew."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from boardbench.artifacts.store import read_json,write_json,digest,source_files,source_id
from boardbench.v122.campaign import physical_cpus
from boardbench.v12.budget import kill_tree


def guard(root):
    root=Path(root).resolve();c=read_json(root/'config.json');m=read_json(root/'launch.json')
    if (digest(root/'config.json')!=m['config_sha256'] or c['host']!=os.uname().nodename
            or time.time()>=c['deadline_unix']):raise ValueError('immutable authorized host/window')
    if (root/'summary.json').exists() and read_json(root/'summary.json')['status']=='completed':raise ValueError('completed run')
    env={**os.environ,'PYTHONPATH':str(root/'source/src'),'CUDA_VISIBLE_DEVICES':'',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
        'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    process=subprocess.Popen([sys.executable,'-u','-m','boardbench.v150.campaign','--root',str(root)],env=env,start_new_session=True)
    os.sched_setaffinity(0,{c['controller_cpu']})
    def stop(*_):
        if process.poll() is None:process.send_signal(signal.SIGTERM)
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    signaled=False
    while process.poll() is None:
        now=time.time()
        if now>=c['deadline_unix']-45 and not signaled:stop();signaled=True
        if now>=c['deadline_unix']:
            kill_tree(process.pid);process.wait();write_json(root/'deadline.json',dict(time=time.time()));break
        time.sleep(1)
    if process.returncode:raise SystemExit(process.returncode)


def launch(config,root,workers=48):
    c=read_json(config);repo=Path.cwd().resolve();root=Path(root).resolve();cpus=physical_cpus()
    start=datetime.fromisoformat(c['authorized_start']).timestamp();deadline=start+8*3600
    if (os.uname().nodename!=c['host'] or not start<=time.time()<deadline-1800
            or workers<1 or workers+2>len(cpus)//2 or os.getloadavg()[0]>len(cpus)*.6):
        raise ValueError('authorized host, original8h window, half-host CPU capacity required')
    root.mkdir(parents=True,exist_ok=False)
    for name in source_files(repo):
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    for k in ('initial_value','initial_data'):c[k]=str((repo/c[k]).resolve())
    value=read_json(c['initial_value']);c['base_model']=value['model'];c['base_sha256']=value['model_sha256']
    if digest(c['base_model'])!=c['base_sha256']:raise ValueError('immutable base weights')
    for k in ('student','opponent'):
        c['job'][k]=str((repo/c['job'][k]).resolve())
        if digest(c['job'][k])!=c['job'][k+'_sha256']:raise ValueError('policy hash')
    c['job']['native_artifact']=read_json(repo/c.pop('native_manifest'))
    selected=cpus[32:32+workers+2] if len(cpus)>=32+workers+2 else cpus[:workers+2]
    c.update(repo=str(repo),source_id=source_id(repo),started_unix=start,deadline_unix=deadline,
        launched_unix=time.time(),worker_cpus=selected[:workers],trainer_cpu=selected[-2],controller_cpu=selected[-1],gpu=0)
    write_json(root/'config.json',c);write_json(root/'launch.json',dict(config_sha256=digest(root/'config.json'),
        source_id=c['source_id'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cpu_slots=selected,physical_cpu_count=len(cpus),created_unix=time.time()))
    guard(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--root',required=True)
    p.add_argument('--workers',type=int,default=48);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.resume:guard(a.root)
    else:launch(a.config,a.root,a.workers)
