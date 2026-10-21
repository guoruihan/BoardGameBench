"""Small real own-state corpus and CUDA frozen/joint/resume preflight."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from boardbench.artifacts.store import read_json, source_id, write_json
from boardbench.v122.campaign import physical_cpus
from boardbench.v133.campaign import wait_resources
from .data import build, reader
from .training import Trainer


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--repo',required=True);p.add_argument('--root',required=True)
    a=p.parse_args();repo,root=Path(a.repo).resolve(),Path(a.root).resolve();config=read_json(root/'config.json')
    if os.uname().nodename!=config['host']:raise ValueError('a226 only')
    cpus=physical_cpus()
    if len(cpus)<16:raise ValueError('half-machine CPU cap')
    os.sched_setaffinity(0,set(cpus[:8])); started=time.monotonic()
    lock=wait_resources(root,repo,config,time.time()+config['resource_wait_seconds'],'preflight')
    waited=time.monotonic()-started
    out=root/f'preflight_attempt_{len(list(root.glob("preflight_attempt_*")))+1:02d}';out.mkdir()
    tick,extra_gpu=time.monotonic(),0.
    receipt={'source_id':source_id(),'passed':False,'checks':[],'waiting_seconds':waited}
    try:
        for i,gpu in enumerate(config['gpu_uuids']):
            t0=time.monotonic()
            r=subprocess.run([sys.executable,'-c',"import torch; assert torch.cuda.is_available(); x=torch.ones((32,32),device='cuda',dtype=torch.float64); assert (x@x).sum().item()==32768; print(torch.cuda.get_device_name())"],
                             env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu},text=True,capture_output=True,timeout=45)
            if i:extra_gpu+=time.monotonic()-t0
            receipt['checks'].append({'gpu':gpu,'returncode':r.returncode,'output':r.stdout,'error':r.stderr})
            if r.returncode:raise RuntimeError('actual CUDA failed')
        os.environ['CUDA_VISIBLE_DEVICES']=config['gpu_uuids'][0]
        with (out/'pytest.log').open('w') as log:
            r=subprocess.run([sys.executable,'-m','pytest','-q',str(repo/'tests/test_v137.py')+'::test_cuda_resume','--basetemp',str(out/'pytest')],
                             env={**os.environ,'BOARDBENCH_V137_CUDA':'1'},stdout=log,stderr=subprocess.STDOUT,timeout=300)
        if r.returncode:raise RuntimeError('CUDA fixtures failed')
        parent=read_json(root/'parents.json')['a']
        settings={**config['labels'],'environments':8,'fit_futures':2,'verify_futures':2,'offset':256,'future_seed':1379000000,'cohort':128}
        corpus=build(parent,out/'pilot',settings,'cuda')
        receipt['pilot']={k:corpus[k] for k in ('states','branches','diagnostic','elapsed_seconds')}
        write_json(out/'progress.json',receipt)
        for name,original in read_json(root/'templates.json')[::2]:
            c={**original,'goal_dataset':{k:corpus[k] for k in ('path','sha256')},
               'manager_warmup_actions':0,'trajectory_directory':str(out/name/'trajectories')}
            t=Trainer(c,'cuda')
            try:
                t.import_sampling_continuation(c['continuation'])
                if c['selection_method']=='control':
                    baseline=reader(parent,'cuda',6)
                    try:
                        if t.evaluate(t.train_seeds[:32])['games']!=baseline.evaluate(t.train_seeds[:32])['games']:
                            raise RuntimeError('parent/control mismatch')
                    finally:baseline.close()
                before=deepcopy(t.model.state_dict());row=t.batch()
                if row['initial_ratio_max_error']>1e-4:raise RuntimeError('behavior mismatch')
                if c['selection_method']=='selector':
                    if any(not torch.equal(v,t.model.state_dict()[k]) for k,v in before.items()):raise RuntimeError('frozen worker changed')
                elif row['actor_change_l2']<=0:raise RuntimeError('worker has no gradient')
                if c['selection_method']!='control' and (row['manager_change']<=0 or row['manager_ratio_error']>1e-4):
                    raise RuntimeError('manager gradient/probability failure')
                checkpoint=out/name/'latest.pt';t.save(checkpoint)
                expected=t.batch();weights=deepcopy(t.model.state_dict());high=deepcopy(t.manager.state_dict())
            finally:t.close()
            restored=Trainer(c,'cuda')
            try:
                restored.load(checkpoint);actual=restored.batch()
                if actual['games']!=expected['games']:raise RuntimeError('next trajectory differs on resume')
                for obj,saved in ((restored.model,weights),(restored.manager,high)):
                    for k,v in saved.items():torch.testing.assert_close(v,obj.state_dict()[k],rtol=1e-10,atol=1e-10)
            finally:restored.close()
            check={'trial':name,'ratio_error':row['initial_ratio_max_error'],'worker_change':row['actor_change_l2'],
                   'manager_change':row.get('manager_change',0.),'batch_seconds':expected['batch_seconds'],'exact_resume':True}
            receipt['checks'].append(check);write_json(out/'progress.json',receipt);print(check,flush=True)
        if source_id()!=receipt['source_id']:raise RuntimeError('source changed during preflight')
        receipt['passed']=True
    finally:
        elapsed=time.monotonic()-tick
        receipt.update(cpu_core_seconds=8*elapsed+waited,gpu_seconds=elapsed+extra_gpu,elapsed_seconds=elapsed,ended_unix=time.time())
        write_json(out/'receipt.json',receipt)
        receipts=[read_json(p) for p in root.glob('preflight_attempt_*/receipt.json')]
        write_json(root/'preflight_cost.json',{**receipt,'cpu_core_seconds':sum(r['cpu_core_seconds'] for r in receipts),
                   'gpu_seconds':sum(r['gpu_seconds'] for r in receipts),'attempts':len(receipts)})
        lock.close()


if __name__=='__main__':main()
