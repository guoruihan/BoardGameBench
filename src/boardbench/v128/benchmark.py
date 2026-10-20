"""Matched full-game PPO throughput: same128 cohort, model, seeds and updates.

Only a218 GPU0/1. Holds the existing node CPU lock. All overhead/attempt costs
are recorded separately from steady-state throughput; GPU time is physical
device occupancy, not the sum of four processes sharing that device.
"""
import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time

from boardbench.artifacts.store import read_json,write_json,source_id,source_files,source_root
from boardbench.v12.budget import kill_tree
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.launch import build_jobs

GPUS=['GPU-bbb68b88-9c8d-eb24-01df-05116dafb926','GPU-e9f9586b-5ddd-4597-42b9-3a3fe7eed421']


def worker(config,out,gate):
    import numpy as np
    from boardbench.v127.training import Trainer
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    trainer=Trainer(config,'cuda');began=time.monotonic()
    try:
        for _ in range(3):trainer.batch()  #2critic warmups +1 PPO warmup
        write_json(out/'ready.json',{'pid':os.getpid(),'warmup_seconds':time.monotonic()-began})
        while not Path(gate).exists():
            if time.monotonic()-began>240:raise TimeoutError('benchmark gate')
            time.sleep(.02)
        started=time.time();rows=[]
        for _ in range(config.get('benchmark_batches',12)):
            rows.append(trainer.batch())
        ended=time.time()
        h=hashlib.sha256()
        for k,v in trainer.model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
        result={'status':'completed','source_id':source_id(),'started_unix':started,'ended_unix':ended,
                'seconds':ended-started,'actions':sum(r['actions'] for r in rows),'model_sha256':h.hexdigest(),
                'score_sequence':[r['mean_score'] for r in rows],
                'initial_ratio_max_error':max(r['initial_ratio_max_error'] for r in rows),
                'timing':{k:statistics.mean(r[k] for r in rows) for k in ('batch_seconds','environment_seconds','model_seconds','update_seconds')},
                'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'cpu_affinity':sorted(os.sched_getaffinity(0))}
        write_json(out/'summary.json',result)
    finally:trainer.close()


def campaign(repo,out):
    repo,out=Path(repo).resolve(),Path(out).resolve()
    if os.uname().nodename!='a218.cluster.helixon.com':raise ValueError('a218 only')
    cpus=physical_cpus()[:128]
    if len(cpus)!=128:raise ValueError('expected128 physical CPU representatives')
    os.sched_setaffinity(0,{cpus[-1]})
    lock=(repo/'outputs/v122'/f'{os.uname().nodename}.cpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    inv={v[0].strip():(int(v[1]),int(v[2])) for v in (s.split(',') for s in raw.splitlines())}
    if os.getloadavg()[0]>64 or any(g not in inv or inv[g][0]>256 or inv[g][1]>=5 for g in GPUS):
        raise RuntimeError('benchmark resources busy')
    if source_id()!=read_json(out/'launch.json')['source_id']:raise ValueError('frozen benchmark source required')
    start=time.monotonic()
    config=build_jobs(read_json(repo/'configs/v127_a218_online_ppo.json'),repo)[0][1]
    summary={'source_id':source_id(),'cases':[],'cpu_core_seconds':0.,'gpu_seconds':0.,'gpu_process_seconds':0.}
    write_json(out/'config.json',config)
    suite=read_json(out/'launch.json').get('suite','initial')
    cases=[('reference_w14','reference',14,1,1),('fast_w14','fast_v128',14,1,1),
           ('fast_w7','fast_v128',7,1,1),('fast_w4','fast_v128',4,1,1),('fast_w2','fast_v128',2,1,1)]
    if suite=='packed':
        cases=[('packed8_w7','fast_v128',7,8,2),('packed8_w4','fast_v128',4,8,2),
               ('reference_packed8_w14','reference',14,8,2),('fast_packed8_w14_repeat','fast_v128',14,8,2)]
    # After the worker sweep use the observed fastest worker count for packing.
    current=[]
    try:
        index=0
        while index<len(cases):
            name,backend,workers,njobs,ngpus=cases[index];index+=1
            root=out/name;root.mkdir();current=[]
            jobconfig={**config,'workers':workers,'environment_backend':backend,'benchmark_batches':12}
            write_json(root/'config.json',jobconfig)
            for i in range(njobs):
                cores=cpus[i*(workers+1):(i+1)*(workers+1)]
                if not cores or (workers+1)*njobs+1>128:raise ValueError('CPU cap')
                trial=root/f'job{i}';trial.mkdir();log=(root/f'job{i}.log').open('w')
                gpu=GPUS[i%ngpus]
                env={**os.environ,'PYTHONPATH':str(source_root()/'src'),'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1',
                     'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':gpu}
                tick=time.time()
                child=subprocess.Popen(['taskset','-c',','.join(map(str,cores)),sys.executable,'-u','-m',
                    'boardbench.v128.benchmark','worker','--config',str(root/'config.json'),'--out',str(trial),
                    '--gate',str(root/'gate.json')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                current.append({'process':child,'log':log,'start':tick,'gpu':gpu,'trial':trial,'cores':len(cores)})
            deadline=time.monotonic()+300
            while not all((x['trial']/'ready.json').exists() for x in current):
                if time.monotonic()>deadline or any(x['process'].poll() is not None for x in current):
                    raise RuntimeError(f'benchmark warmup failed: {name}')
                time.sleep(.1)
            write_json(root/'gate.json',{'go':time.time()})
            while any(x['process'].poll() is None for x in current):
                for x in current:
                    if x['process'].poll() is not None:x.setdefault('end',time.time())
                if time.monotonic()>deadline:raise TimeoutError(name)
                time.sleep(.1)
            for x in current:
                x.setdefault('end',time.time());x['log'].close()
                if x['process'].returncode:raise RuntimeError(f'benchmark child failed:{name}')
            results=[read_json(x['trial']/'summary.json') for x in current]
            elapsed=max(r['ended_unix'] for r in results)-min(r['started_unix'] for r in results)
            actions=sum(r['actions'] for r in results)
            hashes={r['model_sha256'] for r in results}
            baseline=summary['cases'][0]['model_sha256'] if summary['cases'] else results[0]['model_sha256']
            if hashes!={baseline}:raise RuntimeError('optimization changed matched PPO model/trajectory')
            charge_cpu=sum(x['cores']*(x['end']-x['start']) for x in current)
            charge_gpu=sum(max(x['end'] for x in current if x['gpu']==g)-min(x['start'] for x in current if x['gpu']==g)
                           for g in {x['gpu'] for x in current})
            case={'name':name,'backend':backend,'workers_per_job':workers,'jobs':njobs,'gpus':ngpus,
                  'actions_per_second':actions/elapsed,'seconds':elapsed,'actions':actions,
                  'cpu_cores':njobs*(workers+1)+1,'model_sha256':baseline,'exact_match':True,
                  'gpu_hours_per_million_steady':ngpus*1e6/(actions/elapsed)/3600,
                  'cpu_core_hours_per_million_steady':njobs*(workers+1)*1e6/(actions/elapsed)/3600,
                  'timing':{k:statistics.mean(r['timing'][k] for r in results) for k in results[0]['timing']},
                  'cpu_core_seconds_with_setup':charge_cpu,'gpu_seconds_with_setup':charge_gpu}
            summary['cases'].append(case);summary['cpu_core_seconds']+=charge_cpu;summary['gpu_seconds']+=charge_gpu
            summary['gpu_process_seconds']+=sum(x['end']-x['start'] for x in current)
            summary['controller_seconds']=time.monotonic()-start;write_json(out/'summary.json',summary)
            print(case,flush=True);current=[]
            if suite=='initial' and index==5:
                fast=[c for c in summary['cases'] if c['backend']=='fast_v128']
                best=max(fast,key=lambda c:c['actions_per_second'])['workers_per_job'];summary['selected_workers']=best
                cases.extend([('packed2','fast_v128',best,2,1),('packed4','fast_v128',best,4,1),
                              ('packed8_two_gpu','fast_v128',best,8,2)])
        summary.update(status='completed',controller_seconds=time.monotonic()-start)
        write_json(out/'summary.json',summary)
    finally:
        for x in current:
            if x['process'].poll() is None:kill_tree(x['process'].pid);x['process'].wait()
            x['log'].close()
            summary['cpu_core_seconds']+=x['cores']*(time.time()-x['start'])
        if current:
            summary.update(status='failed',controller_seconds=time.monotonic()-start)
            summary['gpu_seconds']+=sum(time.time()-min(x['start'] for x in current if x['gpu']==g) for g in {x['gpu'] for x in current})
            write_json(out/'summary.json',summary)
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('campaign','run','worker'))
    p.add_argument('--repo');p.add_argument('--out',required=True);p.add_argument('--config');p.add_argument('--gate')
    p.add_argument('--suite',choices=('initial','packed'),default='initial')
    a=p.parse_args()
    if a.mode=='worker':worker(read_json(a.config),a.out,a.gate)
    elif a.mode=='run':campaign(a.repo,a.out)
    else:
        repo=Path(a.repo).resolve();out=Path(a.out).resolve()
        if os.uname().nodename!='a218.cluster.helixon.com' or not out.is_relative_to(repo/'outputs/v128'):
            raise ValueError('a218 project benchmark only')
        out.mkdir(parents=True,exist_ok=False)
        for relative in source_files(repo):
            target=out/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/relative,target)
        write_json(out/'launch.json',{'source_id':source_id(out/'source'),'started_unix':time.time(),'suite':a.suite})
        os.environ['PYTHONPATH']=str(out/'source/src')
        os.execv(sys.executable,[sys.executable,'-u','-m','boardbench.v128.benchmark','run','--repo',str(repo),'--out',str(out)])
