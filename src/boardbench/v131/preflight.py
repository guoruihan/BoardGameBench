"""Reserved CUDA exact-resume checks with actual admitted labels and native pool."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from boardbench.artifacts.store import read_json, write_json, source_id
from boardbench.v12.budget import kill_tree
from boardbench.v127.preflight import reserve
from .launch import build_jobs
from .training import Trainer


def check(config,out):
    assert torch.cuda.is_available()
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    t=Trainer(config,'cuda');t.import_sampling_continuation(config['continuation'])
    first=t.batch();assert first['guidance_gradient_norm_probe']>0
    p=out/'resume.pt';t.save(p);expected=t.batch()
    model={k:v.clone() for k,v in t.model.state_dict().items()};t.close()
    b=Trainer(config,'cuda');b.load(p);actual=b.batch()
    assert expected['games']==actual['games'] and expected['bc_loss']==actual['bc_loss']
    assert all(torch.equal(v,model[k]) for k,v in b.model.state_dict().items())
    evaluation=b.evaluate(config['final_seeds'][:4]);b.close()
    assert all(g['natural_terminal'] for g in evaluation['games'])
    write_json(out/'receipt.json',{'passed':True,'source_id':source_id(),
        'initial_ratio_max_error':first['initial_ratio_max_error'],'bc_loss':first['bc_loss'],
        'guidance_gradient_norm':first['guidance_gradient_norm_probe'],'games':evaluation['games'],
        'exact_next_update':True,'device':torch.cuda.get_device_name(0)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('launch','check'));p.add_argument('--repo');p.add_argument('--out',required=True)
    p.add_argument('--config',required=True);p.add_argument('--paired-dataset')
    a=p.parse_args()
    if a.mode=='check':check(read_json(a.config),a.out);return
    if os.uname().nodename!='a218.cluster.helixon.com':raise ValueError('a218 only')
    repo,out=Path(a.repo).resolve(),Path(a.out).resolve();config=read_json(a.config)
    lock=reserve(repo,config);out.mkdir(parents=True,exist_ok=False);started=time.monotonic();active=[];receipts=[]
    jobs=dict(build_jobs(config,repo,Path(a.paired_dataset).resolve()))
    try:
        for i,name in enumerate(('habitat_a','student_b')):
            path=out/(name+'.json');write_json(path,jobs[name])
            log=(out/(name+'.log')).open('w');gpu=config['gpu_uuids'][i%len(config['gpu_uuids'])]
            env={**os.environ,'PYTHONPATH':str(repo/'src'),'CUDA_VISIBLE_DEVICES':gpu,'PYTHONNOUSERSITE':'1',
                'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
            child=subprocess.Popen(['taskset','-c',f'{i*8}-{i*8+7}',sys.executable,'-u','-m','boardbench.v131.preflight',
                'check','--config',str(path),'--out',str(out/name)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append((name,gpu,child,log,time.monotonic()))
        for name,gpu,child,log,tick in active:
            rc=child.wait(timeout=max(1,240-(time.monotonic()-tick)));log.close()
            receipts.append({'trial':name,'gpu':gpu,'returncode':rc,'reserved_cpu_seconds':8*(time.monotonic()-tick),
                             'gpu_seconds':time.monotonic()-tick})
    finally:
        for name,gpu,child,log,tick in active:
            if child.poll() is None:kill_tree(child.pid);child.wait()
            log.close()
        write_json(out/'cost.json',{'attempts':receipts,'source_id':source_id(),'wall_seconds':time.monotonic()-started,
            'accounting':'covered by conservative V131 preflight allowance once; not an extra formal charge'})
        lock.close()
    if any(r['returncode'] for r in receipts):raise RuntimeError('CUDA gate failed, inspect logs')
    write_json(out/'passed.json',{'source_id':source_id(),'trials':[r['trial'] for r in receipts]})


if __name__=='__main__':main()
