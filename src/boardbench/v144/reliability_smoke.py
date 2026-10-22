"""Bounded one-GPU repair verification; not a new training campaign."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time
import torch
from boardbench.artifacts.store import read_json,write_json,source_id
from boardbench.v121.collector import WorkerTimeoutError
from boardbench.v127.preflight import reserve
from boardbench.v140.training import Trainer as OldEvaluator
from boardbench.v143.replay import replay
from .preflight import trial_config
from .training import Trainer


def run(repo,out,panels=20):
    repo=Path(repo).resolve();out=Path(out).resolve();config=read_json(repo/'configs/v144_a226_own_score.json')
    gpu=config['gpu_uuids'][0];cpus=sorted(os.sched_getaffinity(0))
    if os.uname().nodename!=config['host'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or len(cpus)!=4:
        raise ValueError('a226 GPU0 and four pinned CPU cores only')
    if not 1<=panels<=20:raise ValueError('bounded verification only')
    out.mkdir(parents=True,exist_ok=False)
    lock=reserve(repo,{**config,'gpu_uuids':[gpu]});started=time.monotonic();t=None;restored=None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('one real CUDA device required')
        c=trial_config(repo,config,dict(objective='score',mode='league',gae_lambda=.95))
        c.update(workers=3,batch_matches=8,epochs=1,minibatch=256,snapshot_interval=1,deadline_unix=time.time()+540)
        t=Trainer(c,'cuda');first=t.batch();t.save(out/'before_eval.pt');pool=t.pool;pids=[p.pid for p in pool.processes]
        seeds=c['development_seeds'][:4];old=OldEvaluator.evaluate(t,seeds);rng=t.rng.get_state().clone()
        for _ in range(panels):
            if time.monotonic()-started>480:raise TimeoutError('smoke reserve exceeded')
            panel=t.evaluate_panel(seeds)
            assert t.pool is pool and [p.pid for p in t.pool.processes]==pids
            actual=[{k:v for k,v in g.items() if k!='evaluation_opponent'} for g in panel['per_game'] if g['evaluation_opponent']=='transfer']
            assert old['per_game']==actual and torch.equal(rng,t.rng.get_state())
        expected=t.batch();weights=deepcopy(t.model.state_dict());t.close()
        restored=Trainer(c,'cuda');restored.load(out/'before_eval.pt');actual=restored.batch()
        assert expected['per_game']==actual['per_game']
        assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in weights.items())
        for game in restored.last_traces:list(replay(game,'score'))
        before=restored.evaluate_panel(seeds);failed_pool=restored.pool
        def fail(_):raise WorkerTimeoutError(dict(command='reset',expected='ok',pid=failed_pool.processes[0].pid,injected=True))
        failed_pool.reset=fail
        after=restored.evaluate_panel(seeds)
        assert before==after and failed_pool.closed and len(restored.pool_recovery_events)==1
        write_json(out/'result.json',dict(passed=True,source_id=source_id(),panels=panels,
            gpu=gpu,cpus=cpus,original_worker_pids=pids,persistent_pids=True,reference_evaluator_parity=True,
            exact_next_update=True,injected_timeout_recovered=True,
            first_learner_trajectories=first['learner_trajectories'],second_learner_trajectories=actual['learner_trajectories'],
            actor_change=actual['actor_change'],context_change=actual['context_change'],initial_ratio_max_error=actual['initial_ratio_max_error'],
            recovery_events=restored.pool_recovery_events,wall_seconds=time.monotonic()-started,
            note='Small verification updates only; no saved model promoted, no old budget resumed.'))
        print(read_json(out/'result.json'),flush=True)
    except BaseException as exc:
        write_json(out/'failed.json',dict(error=repr(exc),wall_seconds=time.monotonic()-started));raise
    finally:
        if t:t.close()
        if restored:restored.close()
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.repo,a.out)
