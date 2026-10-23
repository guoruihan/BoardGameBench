"""Read-only identity, all-thread affinity, GPU and first-PPO-trajectory audit."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import time
from boardbench.artifacts.store import read_json,write_json,source_id
from boardbench.v122.campaign import physical_cpus
from boardbench.v145.replay import replay


def audit(root):
    root=Path(root).resolve();c=read_json(root/'resolved_config.json');launch=read_json(root/'launch.json')
    if os.uname().nodename!=c['host'] or source_id()!=launch['source_id']:raise ValueError('actual host and frozen source required')
    physical=physical_cpus();os.sched_setaffinity(0,{launch['cpus'][-1]})
    s=read_json(root/'status.json')
    if time.time()>=c['deadline_unix'] or time.time()-s['updated_unix']>20 or not s['active']:
        raise ValueError('fresh active work within original allocation required')
    if len(launch['cpus'])>len(physical)//2:raise ValueError('half-node violation')
    children={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines():
        pid,ppid=map(int,line.split());children.setdefault(ppid,[]).append(pid)
    gpu_pids={}
    if c['role']=='learning':
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        gpu_pids={int(r.split(',')[1]):r.split(',')[0].strip() for r in rows.splitlines() if ',' in r}
    resources=[];used=set(os.sched_getaffinity(s['pid']))
    if used!={launch['cpus'][-1]}:raise ValueError('controller affinity')
    for active in s['active']:
        pid=active['pid'];allowed={active['cpu']} if c['role']=='search' else set(active['cpus'])
        proc=Path('/proc')/str(pid)
        try:command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:continue
        if 'boardbench.v147.' not in command or str(root) not in command:raise ValueError('process identity changed')
        pids=[pid]
        for child in pids:pids.extend(children.get(child,[]))
        for child in pids:
            try:threads=list((Path('/proc')/str(child)/'task').iterdir())
            except FileNotFoundError:continue
            for thread in threads:
                try:affinity=os.sched_getaffinity(int(thread.name))
                except ProcessLookupError:continue
                if not affinity<=allowed:raise ValueError('thread escaped allocated cores')
                used.update(affinity)
        if c['role']=='learning' and gpu_pids.get(pid)!=active['gpu']:raise ValueError('actual CUDA device mismatch/not yet initialized')
        resources.append(dict(pid=pid,cpus=sorted(allowed),processes=len(pids),gpu=gpu_pids.get(pid)))
    if not resources or not used<=set(launch['cpus']):raise ValueError('missing work or escaped allocation')
    proofs=[]
    for path in sorted((root/'rounds/r1').glob('*/trajectories/batch_000001.json.gz')):
        with gzip.open(path,'rt') as stream:payload=json.load(stream)
        if payload['source_id']!=launch['source_id']:raise ValueError('trajectory source')
        # Independently replay a paired-seat sample per arm; all full batches stay archived.
        games=payload['games'][:8];actions=sum(sum(1 for _ in replay(game)) for game in games)
        proofs.append(dict(trial=path.parent.parent.name,games=len(games),actions=actions))
    return dict(passed=True,source_id=launch['source_id'],checked_unix=time.time(),role=c['role'],phase=s['phase'],
        resources=resources,allocated_physical_cpus=len(launch['cpus']),active_physical_cpus=len(used),
        independent_first_batch_replay=proofs,scope='No task changed or foreign process signaled.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=audit(a.root);write_json(a.out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='resources'}),flush=True)
