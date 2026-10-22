"""Read-only live-worker audit; writes only the explicitly requested receipt."""
import argparse
import os
from pathlib import Path
import time

from boardbench.artifacts.store import read_json,source_id,write_json
from boardbench.v122.campaign import physical_cpus


def audit(root):
    root=Path(root).resolve();config=read_json(root/'config.json');launch=read_json(root/'launch.json')
    if os.uname().nodename!=config['host'] or source_id()!=launch['source_id']:
        raise ValueError('run on actual host with frozen source')
    physical=physical_cpus()
    os.sched_setaffinity(0,{launch['cpus'][-1]})
    if time.time()>=config['deadline_unix']:raise ValueError('live launch audit only within original window')
    state=read_json(root/'status.json')
    if not state['active'] or time.time()-state['updated_unix']>15:raise ValueError('fresh active work required')
    verified=[];used={launch['cpus'][-1]}
    if os.sched_getaffinity(state['pid'])!={launch['cpus'][-1]}:raise ValueError('controller affinity mismatch')
    for row in state['active']:
        process=Path('/proc')/str(row['pid'])
        try:command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:continue  # A completing game is harmless; re-audit later if all finish.
        if 'boardbench.v146.worker' not in command or str(root/'episodes'/row['id']) not in command:
            raise ValueError('worker identity changed')
        affinity=os.sched_getaffinity(row['pid'])
        if affinity!={row['cpu']}:raise ValueError('single-CPU worker affinity mismatch')
        for thread in (process/'task').iterdir():
            if os.sched_getaffinity(int(thread.name))!=affinity:raise ValueError('thread escaped allocated CPU')
        used.update(affinity);progress=root/'episodes'/row['id']/'progress.json'
        verified.append(dict(**row,progress=read_json(progress) if progress.exists() else None))
    if not verified or not used<=set(launch['cpus']) or len(launch['cpus'])>len(physical)//2:
        raise ValueError('missing workers or half-node CPU violation')
    completed=[read_json(p) for p in (root/'episodes').glob('*/result.json')]
    return dict(verified_unix=time.time(),phase=state['phase'],workers=verified,active_physical_cpus=sorted(used),
        maximum_allocated_physical_cpus=len(launch['cpus']),cpu_only=True,completed_games=len(completed),
        all_completed_replayed=all(r['audit']['natural_terminal'] for r in completed),
        failed_files=[str(p) for p in (root/'episodes').glob('*/failed.json')])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();result=audit(a.root);write_json(a.out,result)
    print({k:v for k,v in result.items() if k!='workers'})
