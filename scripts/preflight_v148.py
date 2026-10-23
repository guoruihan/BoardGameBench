"""Bounded one-GPU real preflight with the shared node lock and frozen source."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from boardbench.artifacts.store import read_json,write_json,source_files,source_id
from boardbench.v122.campaign import physical_cpus
from boardbench.v127.preflight import reserve
from boardbench.v12.budget import kill_tree


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--gpu',type=int,choices=range(4),default=0);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];out=Path(a.out).resolve()
    if os.uname().nodename!='a226.cluster.helixon.com':raise ValueError('authorized a226 only')
    cpus=physical_cpus()
    if len(cpus)//2<5:raise ValueError('five cores must fit half node')
    gpu=read_json(repo/'configs/v147_a226_learning.json')['authorized_gpu_uuids'][a.gpu]
    lock=reserve(repo,dict(gpu_uuids=[gpu]));start=time.time();child=None
    try:
        out.mkdir(parents=True,exist_ok=False)
        for name in source_files(repo):
            target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
        c=read_json(a.config);c['source_id']=source_id(out/'source')
        if c.get('evaluation_config'):c['evaluation_config']['source_id']=c['source_id']
        if c['deadline_unix']<=start:raise ValueError('preflight config expired')
        write_json(out/'config.json',c)
        env={**os.environ,'PYTHONPATH':str(out/'source/src'),'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1',
             'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
             'NUMEXPR_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        os.sched_setaffinity(0,{cpus[4]})
        with (out/'preflight.log').open('w') as stream:
            child=subprocess.Popen(['taskset','-c',','.join(map(str,cpus[:4])),sys.executable,'-u','-m',
                'boardbench.v148.preflight','--config',str(out/'config.json'),'--out',str(out/'gate')],
                stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:rc=child.wait(timeout=min(240,max(1,c['deadline_unix']-time.time())))
            except subprocess.TimeoutExpired:
                kill_tree(child.pid);child.wait();rc=-1
        elapsed=time.time()-start
        write_json(out/'receipt.json',dict(returncode=rc,wall_seconds=elapsed,gpu_seconds=elapsed,
            allocated_cpu_core_seconds=5*elapsed,gpu=gpu,cpus=cpus[:5],source_id=c['source_id'],
            note='Bounded execution check, not a new long-run authorization or strength result.'))
        if rc:raise RuntimeError('preflight failed; see preserved log')
        print(read_json(out/'receipt.json'),flush=True)
    finally:
        if child and child.poll() is None:kill_tree(child.pid);child.wait()
        lock.close()


if __name__=='__main__':main()
