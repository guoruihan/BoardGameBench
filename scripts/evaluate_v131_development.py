"""Frozen-endpoint check on remaining development[96:128], never formal test."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v12.budget import kill_tree
from boardbench.v127.preflight import reserve
from boardbench.v130.pipeline import checkpoint_spec
from boardbench.v131.training import Trainer
from boardbench.v131.counterfactual import paired_stats


def worker(root,out):
    plan=read_json(out/'plan.json')
    if source_id()!=plan['source_id']:raise ValueError('frozen evaluation source mismatch')
    results={}
    for name,spec in plan['checkpoints'].items():
        config=read_json(root/'ppo/trials'/('control_a' if name=='parent' else name)/'config.json')
        t=Trainer(config,'cuda')
        if name=='parent':t.import_sampling_continuation(spec)
        else:
            if digest(spec['checkpoint'])!=spec['sha256']:raise ValueError('checkpoint changed after panel registration')
            t.load(spec['checkpoint'])
        value=t.evaluate(plan['environment_seeds']);t.close()
        if not all(g['natural_terminal'] for g in value['games']):raise ValueError('nonterminal evaluation')
        write_json(out/(name+'.json'),value);results[name]=value
    arms={}
    for name in ('control','habitat','student'):
        rows=[results[name+'_'+suffix] for suffix in 'ab']
        differences=[]
        for i,seed in enumerate(plan['environment_seeds']):
            if any(r['games'][i]['environment_seed']!=seed for r in results.values()):
                raise ValueError('paired evaluation order mismatch')
            differences.append(np.mean([results[name+'_'+s]['games'][i]['score']-
                                        results['control_'+s]['games'][i]['score'] for s in 'ab']))
        arms[name]={'mean_score':float(np.mean([r['mean_score'] for r in rows])),
                    'animal_mean':float(np.mean([r['animal_mean'] for r in rows])),
                    'minus_control':paired_stats(differences)}
    write_json(out/'summary.json',{'arms':arms,'parent':results['parent']['mean_score'],
        'environment_count':len(plan['environment_seeds']),'policy':'greedy_model_only',
        'selection':'all six fixed final endpoints; no tuning or checkpoint promotion on this panel',
        'split':'additional development; not official test/confirmation'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--worker',action='store_true')
    a=p.parse_args();root=Path(a.root).resolve();out=root/'development_extra32'
    if a.worker:worker(root,out);return
    repo=Path(__file__).resolve().parents[1]
    if os.uname().nodename!='a218.cluster.helixon.com':raise ValueError('a218 only')
    config=read_json(root/'ppo/config.json')
    if read_json(root/'ppo/state.json')['phase']!='finished':raise ValueError('training must be quiescent')
    if out.exists():raise ValueError('refusing evaluation overwrite')
    manifest=read_json(repo/config['split_manifest']);seeds=manifest['sets']['development'][96:128]
    if len(seeds)!=32:raise ValueError('registered remaining development panel unavailable')
    lock=reserve(repo,config);os.sched_setaffinity(0,{48});out.mkdir();started=time.monotonic()
    checkpoints={'parent':checkpoint_spec(repo/config['parent'])}
    for trial in sorted((root/'ppo/trials').glob('*')):checkpoints[trial.name]=checkpoint_spec(trial/'final.pt')
    write_json(out/'plan.json',{'source_id':source_id(),'checkpoints':checkpoints,'environment_seeds':seeds,
        'split_sha256':digest(repo/config['split_manifest']),'script_sha256':digest(__file__),
        'maximum_seconds':180,'gpu':config['gpu_uuids'][0],'cpu_reservation':[47,48]})
    env={**os.environ,'PYTHONPATH':str(repo/'src'),'CUDA_VISIBLE_DEVICES':config['gpu_uuids'][0],
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    child=None;rc=None
    try:
        with (out/'worker.log').open('w') as log:
            child=subprocess.Popen(['taskset','-c','47',sys.executable,'-u',__file__,'--worker','--root',str(root)],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            rc=child.wait(timeout=180)
    finally:
        if child is not None and child.poll() is None:kill_tree(child.pid);child.wait()
        elapsed=time.monotonic()-started
        write_json(out/'cost.json',{'returncode':rc,'elapsed_seconds':elapsed,
            'cpu_core_hours':2*elapsed/3600,'gpu_hours':elapsed/3600,
            'basis':'additional 2CPU+1GPU evaluation reservation; add once to campaign cost'})
        lock.close()
    if rc:raise RuntimeError('evaluation failed; inspect preserved log')


if __name__=='__main__':main()
