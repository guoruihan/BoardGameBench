"""Frozen source, half-host CPU cap, a227 GPU check and detached deadline guard."""
import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from boardbench.artifacts.store import digest,read_json,write_json,source_files,source_id
from boardbench.v122.campaign import physical_cpus
from boardbench.v12.budget import kill_tree


def guard(root):
    root=Path(root).resolve();c=read_json(root/'config.json');launch=read_json(root/'launch.json')
    if (digest(root/'config.json')!=launch['config_sha256'] or time.time()>=c['deadline_unix']
            or (root/'summary.json').exists() or os.uname().nodename!=c['host']):
        raise ValueError('immutable host/config/window required; completed windows cannot resume')
    env={**os.environ,'PYTHONPATH':str(root/'source/src'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
         'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',
         'CUDA_VISIBLE_DEVICES':'','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    process=subprocess.Popen([sys.executable,'-u','-m','boardbench.v149.campaign','--root',str(root)],
                             env=env,start_new_session=True)
    os.sched_setaffinity(0,{c['controller_cpu']})
    def stop(*_):
        if process.poll() is None:process.send_signal(signal.SIGTERM)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    stopped=False
    while process.poll() is None:
        if time.time()>=c['deadline_unix']-45 and not stopped:stop();stopped=True
        if time.time()>=c['deadline_unix']:
            kill_tree(process.pid);process.wait();write_json(root/'deadline.json',dict(time=time.time()));break
        time.sleep(1)
    if process.returncode:raise SystemExit(process.returncode)


def launch(config,root,hours=2,workers=24,gpu=0,parent=None):
    repo=Path.cwd().resolve();root=Path(root).resolve();cpus=physical_cpus()
    if parent:
        parent=Path(parent).resolve();c=read_json(parent/'config.json')
        if (not (parent/'summary.json').exists() or time.time()>c['deadline_unix']-900):
            raise ValueError('completed parent and at least15min of original window required')
        c.update(feedback_parent=str(parent),parent_config_sha256=digest(parent/'config.json'))
    else:c=read_json(config)
    if (os.uname().nodename!=c['host'] or not 0<hours<=2 or workers+2>len(cpus)//2
            or workers<1 or os.getloadavg()[0]>len(cpus)*.5):
        raise ValueError('authorized host, bounded initial window, idle CPU and half-node cap required')
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'],text=True)
    devices={int(p[0]):list(map(int,p[1:])) for p in (line.split(',') for line in gpu_rows.splitlines())}
    if gpu not in devices or devices[gpu][0]>100 or devices[gpu][1]>10:
        raise RuntimeError('requested GPU occupied; do not preempt or silently share')
    root.mkdir(parents=True,exist_ok=False)
    for name in source_files(repo):
        target=root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
    for key in ('dataset',):c[key]=str((repo/c[key]).resolve())
    c['old_value']['path']=str((repo/c['old_value']['path']).resolve())
    c['old_value']['sha256']=digest(c['old_value']['path'])
    for key in ('student','opponent'):
        c['job'][key]=str((repo/c['job'][key]).resolve())
        if digest(c['job'][key])!=c['job'][key+'_sha256']:raise ValueError('input model hash mismatch')
    if 'native_manifest' in c:c['job']['native_artifact']=read_json(repo/c.pop('native_manifest'))
    # Leave low-numbered CPUs available; no other task's affinity is changed.
    selected=cpus[32:32+workers+2] if len(cpus)>=32+workers+2 else cpus[:workers+2]
    now=time.time();deadline=c['deadline_unix'] if parent else now+3600*hours
    c.update(repo=str(repo),source_id=source_id(repo),started_unix=now,deadline_unix=deadline,
        worker_cpus=selected[:workers],trainer_cpu=selected[-2],controller_cpu=selected[-1],gpu=gpu)
    write_json(root/'config.json',c)
    write_json(root/'launch.json',dict(config_sha256=digest(root/'config.json'),source_id=c['source_id'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        cpus=selected,physical_cpu_count=len(cpus),gpu_preflight=devices,created_unix=now))
    guard(root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--root',required=True)
    p.add_argument('--hours',type=float,default=2);p.add_argument('--workers',type=int,default=24)
    p.add_argument('--gpu',type=int,default=0);p.add_argument('--resume',action='store_true')
    p.add_argument('--parent',help='one feedback cycle within the completed parent original deadline');a=p.parse_args()
    if a.resume:guard(a.root)
    else:launch(a.config,a.root,a.hours,a.workers,a.gpu,a.parent)
