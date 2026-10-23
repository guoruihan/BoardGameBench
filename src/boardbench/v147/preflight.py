"""Actual expert gradient, complete-game PPO and exact next-update resume gate."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time
import torch
from boardbench.artifacts.store import read_json,write_json,source_id
from boardbench.v144.training import compact
from boardbench.v145.policy import load_policy
from boardbench.v145.replay import replay
from .training import Trainer


def run(config,out,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    if device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):raise RuntimeError('one real CUDA GPU required')
    config={**config,'batch_matches':8,'workers':3 if device=='cuda' else 0,'epochs':1,'minibatch':256,
            'expert_weight':.1,'expert_steps':1,'expert_q_weight':.5,'expert_value_weight':.1}
    t=Trainer(config,device);other=None
    try:
        initial=load_policy(config['warm_start'],device)
        if any(not torch.equal(v,t.model.state_dict()[k]) for k,v in initial.state_dict().items()):raise RuntimeError('initial parity')
        anchor=deepcopy(t.training_anchor.state_dict());before=deepcopy(t.model.state_dict())
        auxiliary=t.expert_step()
        if not any(not torch.equal(v,t.model.state_dict()[k]) for k,v in before.items()):raise RuntimeError('no expert gradients')
        first=t.batch()
        if min(first['actor_change'],first['context_change'])<=0:raise RuntimeError('no PPO gradient')
        for game in t.last_traces:list(replay(game))
        t.save(out/'resume.pt');expected=t.batch();weights=deepcopy(t.model.state_dict());traces=deepcopy(t.last_traces);t.close()
        other=Trainer(config,device);other.load(out/'resume.pt');actual=other.batch()
        if expected['per_game']!=actual['per_game'] or traces!=other.last_traces:raise RuntimeError('resume trajectories')
        if any(not torch.equal(v,other.model.state_dict()[k]) for k,v in weights.items()):raise RuntimeError('resume weights')
        if any(not torch.equal(v,other.training_anchor.state_dict()[k]) for k,v in anchor.items()):raise RuntimeError('opponent mutated')
        rng=other.rng.get_state().clone();ev=other.evaluate_panel(config['development_seeds'][:4])
        if not torch.equal(rng,other.rng.get_state()):raise RuntimeError('evaluation consumed training RNG')
        other.export_policy(other.model,out/'portable.pt',config);portable=load_policy(out/'portable.pt',device)
        if any(not torch.equal(v,portable.state_dict()[k]) for k,v in weights.items()):raise RuntimeError('portable parity')
        result=dict(passed=True,source_id=source_id(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
            device=torch.cuda.get_device_name() if device=='cuda' else 'cpu',expert=auxiliary,first=compact(first),
            evaluation=compact(ev),exact_resume=True,frozen_opponent_unchanged=True,wall_seconds=time.monotonic()-start)
        write_json(out/'result.json',result);print(result,flush=True)
    finally:
        t.close()
        if other:other.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();run(read_json(a.config),a.out,a.device)
