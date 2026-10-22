"""Read-only live resource audit and independent first-batch replay receipt."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import time
from boardbench.artifacts.store import read_json,write_json,source_id
from boardbench.v145.replay import replay


def run(root):
    root=Path(root).resolve();started=time.monotonic();status=read_json(root/'status.json')
    launch=read_json(root/'launch.json');config=read_json(root/'resolved_config.json')
    if os.uname().nodename!=config['host'] or source_id()!=launch['source_id']:raise ValueError('matching host and frozen source required')
    if time.time()>=config['deadline_unix']:raise ValueError('expired audit allocation')
    children={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines():
        pid,ppid=map(int,line.split());children.setdefault(ppid,[]).append(pid)
    gpu_rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    gpu_pids={int(row.split(',')[1]):row.split(',')[0].strip() for row in gpu_rows.splitlines()}
    resources=[];used=set(os.sched_getaffinity(status['pid']))
    assert used=={launch['cpus'][-1]}
    for active in status['active']:
        if not active['stage'].startswith('r1_'):raise ValueError('audit is for first formal round')
        allowed=set(active['cpus']);pids=[active['pid']]
        for pid in pids:pids.extend(children.get(pid,[]))
        for pid in pids:
            affinity=set(os.sched_getaffinity(pid));used.update(affinity)
            if not affinity<=allowed:raise RuntimeError('worker escaped assigned cores')
        if gpu_pids.get(active['pid'])!=active['gpu']:raise RuntimeError('wrong actual GPU')
        resources.append(dict(stage=active['stage'],pid=active['pid'],processes=len(pids),cpus=sorted(allowed),gpu=active['gpu']))
    if len(resources)!=4 or used!=set(launch['cpus']):raise RuntimeError('four verified jobs and57 allocated cores required')
    proof=[];paired=None
    for path in sorted((root/'rounds/r1').glob('*/trajectories/batch_000001.json.gz')):
        with gzip.open(path,'rt') as f:payload=json.load(f)
        if payload['source_id']!=launch['source_id']:raise ValueError('trajectory provenance mismatch')
        games=payload['games'];seeds=[g['seed'] for g in games]
        if paired is None:paired=seeds
        if seeds!=paired:raise RuntimeError('first-batch environment seeds differ')
        count=0
        for game in games:
            for _ in replay(game):count+=1
        proof.append(dict(trial=path.parent.parent.name,games=len(games),actions=count,
            learning_trajectories=sum(len(g['learner_seats']) for g in games)))
    if len(proof)!=4 or sum(p['games'] for p in proof)!=512:raise RuntimeError('incomplete first-batch archive')
    result=dict(passed=True,source_id=launch['source_id'],checked_unix=time.time(),resources=resources,
        allocated_cpu_cores=len(used),physical_gpus=sorted({r['gpu'] for r in resources}),
        paired_first_batch=True,raw_and_potential_reward_replay=proof,wall_seconds=time.monotonic()-started,
        scope='Read-only verification of this run only; no foreign processes signaled.')
    write_json(root/'launch_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
