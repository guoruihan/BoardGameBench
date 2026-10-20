"""Explicit a218-only eight-device CUDA checks and one real warm-start smoke."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from boardbench.artifacts.store import read_json,write_json,source_id
from boardbench.v127.launch import build_jobs
from boardbench.v12.budget import kill_tree


def reserve(repo,config):
    """Use the same node lock as formal runs; a preflight is not a free resource."""
    lock=(Path(repo)/'outputs/v122'/f'{os.uname().nodename}.cpu.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'],text=True)
        devices={v[0].strip():(int(v[1]),int(v[2])) for v in (row.split(',') for row in rows.splitlines())}
        if os.getloadavg()[0]>64 or any(g not in devices or devices[g][0]>=256 or devices[g][1]>=5 for g in config['gpu_uuids']):
            raise RuntimeError('preflight resources busy; no tasks preempted')
    except BaseException:
        lock.close();raise
    return lock


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();repo=Path(a.repo).resolve();out=Path(a.out).resolve()
    if os.uname().nodename!='a218.cluster.helixon.com':raise ValueError('a218 only')
    config=read_json(repo/'configs/v127_a218_online_ppo.json')
    lock=reserve(repo,config)
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    active=[];receipts=[]
    env={**os.environ,'PYTHONPATH':str(repo/'src'),'PYTHONNOUSERSITE':'1','OMP_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','BOARDBENCH_V127_CUDA':'1'}
    for i,gpu in enumerate(config['gpu_uuids']):
        log=(out/f'gpu{i}.log').open('w');tick=time.monotonic()
        child=subprocess.Popen(['taskset','-c',f'{i*3}-{i*3+2}',sys.executable,'-m','pytest',
            str(repo/'tests/test_v127.py')+'::test_cuda_collection_update_and_resume','-q',
            '--basetemp',str(out/f'pytest_gpu{i}')],env={**env,'CUDA_VISIBLE_DEVICES':gpu},stdout=log,stderr=subprocess.STDOUT)
        active.append((i,gpu,child,log,tick))
    try:
        for i,gpu,child,log,tick in active:
            rc=child.wait(timeout=240);log.close();elapsed=time.monotonic()-tick
            receipts.append({'gpu':gpu,'returncode':rc,'allocated_cpu_core_seconds':3*elapsed,'gpu_seconds':elapsed})
            print({'gpu':i,'returncode':rc,'seconds':elapsed},flush=True)
    finally:
        for _,_,child,log,_ in active:
            if child.poll() is None:kill_tree(child.pid);child.wait()
            log.close()
    write_json(out/'cuda_checks.json',receipts)
    if any(r['returncode'] for r in receipts):raise RuntimeError('CUDA test failed; inspect per-device logs')
    trial=build_jobs(config,repo)[0][1]
    trial.update(target_actions=10000,max_seconds=180,curve_seeds=trial['curve_seeds'][:2],
                 final_seeds=trial['final_seeds'][:2],milestones=[10000],calibration=True)
    write_json(out/'smoke_config.json',trial);tick=time.monotonic()
    with (out/'smoke.log').open('w') as stream:
        result=subprocess.run(['taskset','-c','0-14',sys.executable,'-u','-m','boardbench.v127.training',
            '--config',str(out/'smoke_config.json'),'--out',str(out/'smoke'),'--device','cuda'],
            env={**env,'CUDA_VISIBLE_DEVICES':config['gpu_uuids'][0]},stdout=stream,stderr=subprocess.STDOUT,timeout=270)
    seconds=time.monotonic()-tick
    write_json(out/'receipt.json',{'source_id':source_id(),'returncode':result.returncode,
        'cpu_core_seconds':sum(r['allocated_cpu_core_seconds'] for r in receipts)+15*seconds+time.monotonic()-started,
        'gpu_seconds':sum(r['gpu_seconds'] for r in receipts)+seconds,
        'note':'Measured process occupancy; covered by formal conservative preflight allowance, do not add twice'})
    if result.returncode:raise RuntimeError('actual imitation->PPO smoke failed')
    print(read_json(out/'smoke/summary.json'),flush=True)
    lock.close()


if __name__=='__main__':main()
