"""Read-only actual-process, thread-affinity, device and completed-game audit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from boardbench.artifacts.store import read_json,write_json,source_id,digest
from boardbench.v122.campaign import physical_cpus
from boardbench.v146.campaign import valid_result


def audit(root):
    root=Path(root).resolve();c=read_json(root/'resolved_config.json');launch=read_json(root/'launch.json')
    if (os.uname().nodename!=c['host'] or source_id()!=launch['source_id']
            or digest(root/'resolved_config.json')!=launch['resolved_config_sha256']):
        raise ValueError('actual host, frozen source and exact config required')
    physical=physical_cpus();os.sched_setaffinity(0,{launch['cpus'][-1]})
    s=read_json(root/'status.json')
    if time.time()>=c['deadline_unix'] or time.time()-s['updated_unix']>20:
        raise ValueError('fresh heartbeat within original window required')
    if len(launch['cpus'])>len(physical)//2:raise ValueError('half-node violation')
    used=set(os.sched_getaffinity(s['pid']))
    if used!={launch['cpus'][-1]}:raise ValueError('controller affinity')
    controller=(Path('/proc')/str(s['pid'])/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'boardbench.v148.launch' not in controller or str(root) not in controller:
        raise ValueError('controller identity')
    children={}
    for line in subprocess.check_output(['ps','-eo','pid=,ppid='],text=True).splitlines():
        pid,ppid=map(int,line.split());children.setdefault(ppid,[]).append(pid)
    gpu_pids={}
    if c['role']=='learning':
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        gpu_pids={int(r.split(',')[1]):r.split(',')[0].strip() for r in rows.splitlines() if ',' in r}
    resources=[]
    for active in s['active']:
        pid=active['pid'];allowed={active['cpu']} if c['role']=='search' else set(active['cpus'])
        proc=Path('/proc')/str(pid)
        try:command=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:continue  # Process may complete after the heartbeat.
        if 'boardbench.v148.' not in command or str(root) not in command:raise ValueError('child identity')
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
        cpu_data='boardbench.v148.data' in command
        if c['role']=='learning' and not cpu_data and gpu_pids.get(pid)!=active['gpu']:
            raise ValueError('actual GPU mismatch/not yet initialized')
        resources.append(dict(pid=pid,cpus=sorted(allowed),processes=len(pids),
                              gpu=gpu_pids.get(pid),cpu_only_data_stage=cpu_data))
    if not used<=set(launch['cpus']):raise ValueError('escaped node allocation')
    gates=[]
    if c['role']=='learning':
        for path in sorted(root.glob('preflight_gpu*/result.json')):
            result=read_json(path)
            if not result['passed'] or result['source_id']!=launch['source_id']:raise ValueError('CUDA gate provenance')
            gates.append(dict(name=path.parent.name,gpu=result['gpu_uuid'],seconds=result['wall_seconds'],
                              exact_resume=result['exact_resume'],gradient=result['first']['gradient_norm']))
    completed=[]
    for path in sorted((root/'episodes').glob('*/result.json')):
        result=read_json(path);job=read_json(root/'jobs'/(result['id']+'.json'));valid_result(result,job)
        completed.append(dict(id=result['id'],score=result['score'],decision_seconds=result['decision_seconds'],
                              natural_terminal=result['natural_terminal']))
    return dict(passed=True,checked_unix=time.time(),source_id=launch['source_id'],role=c['role'],phase=s['phase'],
        resources=resources,allocated_physical_cpus=len(launch['cpus']),active_physical_cpus=len(used),
        cuda_gates=gates,completed_games=completed,
        note='Idle peer-data waits are reported as idle; no process changed or foreign job signaled.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=audit(a.root);write_json(a.out,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('resources','completed_games')}),flush=True)
