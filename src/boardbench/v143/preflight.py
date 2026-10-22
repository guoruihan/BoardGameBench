"""Actual CUDA transfer parity, both-seat update, history exclusion and exact restore."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch

from boardbench.artifacts.store import read_json,write_json,source_id
from boardbench.environments.harmonies_duo import HarmoniesDuo,player_view
from boardbench.v138.baseline import Planner
from boardbench.v140.encoding import action_map,encode_batch
from boardbench.v142.encoding import encode_batch as market_encode
from .policy import LAYOUT,own_projection
from .training import Trainer
from .replay import replay


def trial_config(repo,campaign,mode='current',replica=0):
    repo=Path(repo)
    return dict(**campaign['training'],layout=LAYOUT,rules_version=HarmoniesDuo.rules_version,
        parent=str(repo/campaign['parent']),parent_sha256=campaign['parent_sha256'],
        native_artifact=read_json(repo/campaign['native_manifest']),objective='margin',mode=mode,
        seed=14301+replica,workers=campaign['workers'],batch_matches=campaign['batch_matches'],
        max_seconds=campaign['wall_seconds'],source_id=source_id(repo),
        development_seeds=list(range(243000000,243000000+campaign['development_environments'])),
        confirmation_seeds=list(range(343000000,343000000+campaign['confirmation_environments'])))


def transfer_check(model,device='cpu',seed=143):
    """One complete official game; compare mature actor against its own-view projection."""
    env=HarmoniesDuo();obs,_=env.reset(seed);planners=[Planner(),Planner()];error=0.;count=0
    while not obs['terminated']:
        seat=obs['active_player'];own=player_view(obs);plan=planners[seat].vector(own)
        x=torch.as_tensor(encode_batch([obs],[plan]),dtype=torch.float64,device=device)
        expected=torch.as_tensor(market_encode([own],[plan]),dtype=torch.float64,device=device)
        if not torch.equal(own_projection(x),expected):raise RuntimeError('own-view feature transfer mismatch')
        mapping=action_map(obs);mask=torch.tensor([[i in mapping for i in range(244)]],device=device)
        with torch.inference_mode():
            new,value=model(x,mask);old,old_value=model.core(expected,mask)
        error=max(error,float((new-old).abs().max()),float((value-old_value).abs().max()))
        if error>1e-12 or int(new.argmax(-1))!=int(old.argmax(-1)):raise RuntimeError('initial actor changed on transfer')
        obs=env.step(mapping[int(new.argmax(-1))]).observation;count+=1
    env.validate();return dict(actions=count,max_error=error,scores=obs['scores'],turns=obs['players'][0]['completed_turns'])


def run(repo,campaign,out,mode,baseline_only=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);tick=time.monotonic()
    if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    c=trial_config(repo,campaign,mode)
    if baseline_only:
        t=Trainer(c,'cuda')
        try:
            result=t.evaluate(c['development_seeds'])
            if result['margin_mean']!=0 or result['win_credit_mean']!=.5:raise RuntimeError('initial self-battle symmetry failed')
            write_json(out/'result.json',dict(**result,source_id=source_id(repo),wall_seconds=time.monotonic()-tick))
            with t.make_pool(8) as pool:
                _,demo=t.collect(pool,c['development_seeds'][:8],[0,1]*4,['current']*8,greedy=True)
            for g in t.last_traces: list(replay(g))
            with gzip.open(out/'self_battle_examples.json.gz','wt') as f:
                json.dump(dict(layout=LAYOUT,rules_version=HarmoniesDuo.rules_version,source_id=source_id(repo),games=t.last_traces),f)
        finally:t.close()
        return
    c.update(workers=3,batch_matches=8,minibatch=256,epochs=1,snapshot_interval=1,history_size=2)
    t=Trainer(c,'cuda');parity=transfer_check(t.model,'cuda')
    frozen=deepcopy(t.frozen.state_dict());first=t.batch()
    if first['actor_change']<=0 or first['context_change']<=0 or first['learner_trajectories']!=16:
        raise RuntimeError('both seats must learn with real actor/opponent updates')
    for game in t.last_traces:list(replay(game))
    t.save(out/'resume.pt');expected=t.batch();weights=deepcopy(t.model.state_dict());t.close()
    restored=Trainer(c,'cuda');restored.load(out/'resume.pt');actual=restored.batch()
    if expected['per_game']!=actual['per_game']:raise RuntimeError('resume games differ')
    for k,v in weights.items():
        if not torch.equal(v,restored.model.state_dict()[k]):raise RuntimeError('resume update mismatch: '+k)
    for k,v in frozen.items():
        if not torch.equal(v,restored.frozen.state_dict()[k]):raise RuntimeError('evaluation opponent changed')
    if mode=='league' and actual['learner_trajectories']!=14:raise RuntimeError('history actions incorrectly train')
    for game in restored.last_traces:list(replay(game))
    rng=restored.rng.get_state().clone();evaluation=restored.evaluate(c['development_seeds'][:4])
    if not torch.equal(rng,restored.rng.get_state()):raise RuntimeError('evaluation consumes training RNG')
    restored.close();result=dict(passed=True,mode=mode,source_id=source_id(repo),gpu=torch.cuda.get_device_name(),
        wall_seconds=time.monotonic()-tick,transfer=parity,exact_next_update=True,
        first={k:v for k,v in first.items() if k!='per_game'},second={k:v for k,v in actual.items() if k!='per_game'},
        evaluation={k:v for k,v in evaluation.items() if k!='per_game'})
    write_json(out/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','config','out','mode'):p.add_argument('--'+key,required=True)
    p.add_argument('--baseline-only',action='store_true');a=p.parse_args()
    run(a.repo,read_json(a.config),a.out,a.mode,a.baseline_only)
