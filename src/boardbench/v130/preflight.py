"""Actual b35 label/replay smoke and two-device a218 guidance/resume gates."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v126.data import build
from boardbench.v127.preflight import reserve
from boardbench.v130.pipeline import build_jobs, teacher_plan
from boardbench.v130.training import Trainer


def teacher(repo, out, config):
    if os.uname().nodename != 'b35.cluster.helixon.com':
        raise ValueError('b35 CPU-only preflight')
    start=time.monotonic();plan=teacher_plan(config,out,repo,1,None)
    plan.update(environment_indices=[3000,3001,3002],panels=[3],cpu_cores=7,task_seconds=90)
    for a in plan['arms']:
        a['timeout_seconds']=60;a['params']['seconds']=30.
    write_json(out/'plan.json',plan)
    subprocess.run([sys.executable,'-m','boardbench.v130.campaign','launch','--repo',str(repo),
        '--root',str(out/'campaign'),'--plan',str(out/'plan.json')],check=True,timeout=110)
    summary=read_json(out/'campaign/summary.json')
    if summary['completed'] != 6:
        raise RuntimeError('CPU teacher smoke incomplete')
    sets=read_json(repo/config['split_manifest'])['sets']
    dataset=out/'corrections.npz';info=build([out/'campaign'],dataset,[sets['train'][3002]],minimum_teacher_rounds=2)
    import numpy as np
    with np.load(dataset) as z:
        nonforced_fit=int(((z['mask'].sum(1)>1)&~z['validation']).sum())
    if info['records'] < 10 or nonforced_fit < 10:
        raise RuntimeError('insufficient sampled correction labels in smoke')
    elapsed=time.monotonic()-start
    write_json(out/'receipt.json',{'source_id':source_id(),'status':'passed','elapsed_seconds':elapsed,
        'cpu_core_seconds':7*elapsed,'gpu_seconds':0.,'dataset':str(dataset),'dataset_sha256':digest(dataset),
        'nonforced_fit_labels':nonforced_fit,
        'note':'CPU time conservatively reserves7cores over full preflight; included in declared allowance'})


def cuda_worker(config_path, out):
    import torch
    from boardbench.v130.training import run
    config=read_json(config_path);start=time.monotonic();t=Trainer(config,'cuda')
    t.import_sampling_continuation(config['continuation'])
    first=t.batch()
    if first['bc_weight'] <= 0 or first['bc_loss'] <= 0 or first['actor_change_l2'] <= 0:
        raise RuntimeError('persistent guidance not active')
    checkpoint=out/'resume.pt';t.save(checkpoint);expected=t.batch()
    weights={k:v.clone() for k,v in t.model.state_dict().items()};t.close()
    restored=Trainer(config,'cuda');restored.load(checkpoint);actual=restored.batch()
    if expected['games']!=actual['games'] or expected['bc_loss']!=actual['bc_loss']:
        raise RuntimeError('guidance resume trajectories/loss differ')
    if not all(torch.equal(v,weights[k]) for k,v in restored.model.state_dict().items()):
        raise RuntimeError('guidance resume parameters differ')
    restored.close()
    summary=run(config,out/'smoke','cuda')
    if summary['status']!='completed':raise RuntimeError('full guidance smoke incomplete')
    write_json(out/'receipt.json',{'source_id':source_id(),'status':'passed','elapsed_seconds':time.monotonic()-start,
        'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'exact_resume':True,'real_corrections':config['correction_datasets'],
        'initial_ratio_max_error':first['initial_ratio_max_error'],'bc_loss':first['bc_loss'],
        'actor_change_l2':first['actor_change_l2'],'ppo_actions':summary['ppo_actions']})


def cuda(repo, out, config, dataset):
    if os.uname().nodename != 'a218.cluster.helixon.com':raise ValueError('a218 only')
    lock=reserve(repo,config);start=time.monotonic();active=[];receipt=[]
    try:
        jobs=dict(build_jobs(config,repo,1,None,[{'path':str(dataset),'sha256':digest(dataset)}]))
        for i,gpu in enumerate(config['gpu_uuids']):
            trial=jobs['correct_high_'+'ab'[i]];count=read_json(Path(trial['continuation']['checkpoint']).with_suffix('.json'))['ppo_actions']
            trial.update(target_actions=count+20000,milestones=[count+10000,count+20000],max_seconds=180,
                         curve_seeds=trial['curve_seeds'][:2],final_seeds=trial['final_seeds'][:2])
            target=out/f'gpu{i}';target.mkdir();write_json(target/'config.json',trial)
            stream=(target/'log.txt').open('w');started=time.monotonic()
            env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                 'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONNOUSERSITE':'1','PYTHONPATH':str(repo/'src')}
            child=subprocess.Popen(['taskset','-c',f'{i*8}-{i*8+7}',sys.executable,'-u','-m','boardbench.v130.preflight',
                'cuda_worker','--repo',str(repo),'--out',str(target),'--config',str(target/'config.json')],
                env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            active.append((child,stream,started,gpu,target))
        for child,stream,started,gpu,target in active:
            code=child.wait(timeout=240);stream.close();elapsed=time.monotonic()-started
            receipt.append({'returncode':code,'gpu':gpu,'elapsed_seconds':elapsed,'cpu_core_seconds':8*elapsed})
        if any(r['returncode'] for r in receipt):raise RuntimeError('CUDA guidance preflight failed')
    finally:
        from boardbench.v12.budget import kill_tree
        for child,stream,started,gpu,target in active:
            if child.poll() is None:kill_tree(child.pid);child.wait()
            stream.close()
        elapsed=time.monotonic()-start
        write_json(out/'receipt.json',{'source_id':source_id(),'status':'passed' if len(receipt)==2 and all(r['returncode']==0 for r in receipt) else 'failed',
            'attempts':receipt,'elapsed_seconds':elapsed,'cpu_core_seconds':17*elapsed,'gpu_seconds':2*elapsed,
            'note':'Conservative full-preflight reservation; included in formal allowance, never added twice'})
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('teacher','cuda','cuda_worker'))
    p.add_argument('--repo',required=True);p.add_argument('--out',required=True);p.add_argument('--config')
    p.add_argument('--dataset');a=p.parse_args();repo=Path(a.repo).resolve();out=Path(a.out).resolve()
    if a.mode=='cuda_worker':cuda_worker(a.config,out)
    else:
        out.mkdir(parents=True,exist_ok=False);config=read_json(repo/'configs/v130_teacher_guided_ppo.json')
        if a.mode=='teacher':teacher(repo,out,config)
        else:cuda(repo,out,config,Path(a.dataset).resolve())
