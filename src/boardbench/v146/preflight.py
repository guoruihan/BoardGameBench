"""Mature CPU inference/replay/branch gate and preserved V145 handoff receipts."""
import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import time

import torch

from boardbench.artifacts.store import digest,read_json,write_json,source_id
from boardbench.environments.harmonies_duo import HarmoniesDuo
from boardbench.v122.campaign import physical_cpus
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner
from boardbench.v145.policy import SelfplaySolver,load_policy
from .policy import ModelPlayer
from .search import TerminalSearch
from .worker import play


def handoff(repo,out):
    root=Path(repo)/'outputs/v145/animal_a226_20261023_b'
    status=read_json(root/'status.json')
    if status['phase']!='interrupted' or status['active']:
        raise ValueError('V145 must stop before switch')
    records=[]
    for trial in sorted((root/'rounds/r2').iterdir()):
        interrupted=read_json(trial/'interrupted.json')
        for name in ('latest','final'):
            meta=read_json(trial/(name+'.json'))
            if meta['sha256']!=digest(trial/(name+'.pt')) or meta['actions']!=interrupted['actions']:
                raise ValueError('stopped checkpoint hash/counter mismatch')
        records.append(dict(trial=trial.name,interrupted=interrupted,latest=read_json(trial/'latest.json'),
                            final=read_json(trial/'final.json'),best_policy_sha256=digest(trial/'best_policy.pt')))
    write_json(Path(out)/'handoff.json',dict(previous_run=str(root),stop_reason='user authorized switch to official-duo search teacher',
        status=status,retained_round2=records,previous_cost=read_json(root/'cost.json'),deadline_renewed=False))


def run(config,out):
    c=read_json(config);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if os.uname().nodename!=c['host']:raise ValueError('actual target CPU host required')
    repo=Path(__file__).resolve().parents[3];physical=physical_cpus()
    lock=(repo/'outputs/v122'/(c['host']+'.cpu.lock')).open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,{physical[-1]});torch.set_num_threads(1);begin=time.monotonic()
    try:
        handoff(repo,out)
        native=NativeEncoder.from_artifact(c['native_artifact']);model=load_policy(c['student'])
        public=ModelPlayer(model,native);reference=SelfplaySolver(model)
        plans=[Planner(),Planner()];game=HarmoniesDuo();obs,_=game.reset(146900010);steps=0
        before=deepcopy(model.state_dict())
        while not obs['terminated']:
            seat=obs['active_player'];a=public.decide(obs,plans[seat]);b=reference.decide(obs)
            if a!=b or plans[seat].__dict__!=reference.planners[seat].__dict__:
                raise ValueError('mature portable policy/planner parity')
            obs=game.step(a).observation;steps+=1
        if any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items()):
            raise ValueError('inference mutated model')
        receipts=[]
        for seat,method in [(0,'model'),(1,'model'),(0,'habitat'),(0,'search_habitat'),(1,'search_model')]:
            job={**{k:c[k] for k in ('student','student_sha256','opponent','opponent_sha256','native_artifact')},
                'id':f'gate_{method}_{seat}','method':method,'split':'calibration','environment_seed':146900011,
                'seat':seat,'seconds':3.,'search_seed':146,'deadline_unix':min(c['deadline_unix'],time.time()+600)}
            result,_=play(job);receipts.append(result)
        # A full early-game six-root model-continuation round, not just fallback.
        opponent=ModelPlayer(load_policy(c['opponent']),native)
        obs,_=game.reset(146900012);plans=[Planner(),Planner()]
        roots=public.rank_batch([obs],[plans[0]])[0][0][:6]
        search=TerminalSearch(public,opponent,0,'model');tick=time.monotonic()
        values,steps_sim,own=search.terminal_round(obs,roots,146,plans,time.monotonic()+120)
        if values is None or len(values)!=6:raise ValueError('complete terminal round required')
        write_json(out/'result.json',dict(passed=True,source_id=source_id(),config_sha256=digest(config),host=c['host'],cpu=physical[-1],
            mature_parity_steps=steps,complete_root_scores=values,complete_root_steps=steps_sim,
            complete_root_seconds=time.monotonic()-tick,calibration=receipts,
            elapsed_seconds=time.monotonic()-begin,allocated_cpu_core_seconds=time.monotonic()-begin,gpu_seconds=0))
    finally:lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(a.config,a.out)
