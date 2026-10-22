"""Mature-policy real-CUDA and independent reward/replay gates."""
import argparse
import os
from copy import deepcopy
from pathlib import Path
import time
import torch
from boardbench.artifacts.store import read_json,write_json,source_id
from boardbench.environments.harmonies_duo import HarmoniesDuo
from boardbench.v144.training import compact
from .policy import LAYOUT,load_policy,export_policy
from .training import Trainer
from .replay import replay


def trial_config(repo,campaign,arm,round_index=0):
    repo=Path(repo)
    result={**campaign['training'],**arm}
    result.update(layout=LAYOUT,rules_version=HarmoniesDuo.rules_version,
        parent=campaign['parent'],parent_sha256=campaign['parent_sha256'],
        warm_start=arm.get('warm_start',campaign['warm_start']),
        warm_start_sha256=arm.get('warm_start_sha256',campaign['warm_start_sha256']),
        training_anchor=campaign['warm_start'],training_anchor_sha256=campaign['warm_start_sha256'],
        native_artifact=read_json(campaign['native_manifest']),evaluation_opponents=campaign['evaluation_opponents'],
        seed=14501+round_index*100+arm.get('replica',0),workers=campaign['workers'],batch_matches=campaign['batch_matches'],
        source_id=source_id(repo),development_seeds=list(range(245000000,245000000+campaign['development_environments'])),
        decision_seeds=list(range(245010000+round_index*1000,245010000+round_index*1000+campaign['decision_environments'])),
        confirmation_seeds=list(range(345000000,345000000+campaign['confirmation_environments'])))
    return result


def run(config,out,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    if device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):raise RuntimeError('one visible real CUDA device required')
    c={**config,'workers':3 if device=='cuda' else 0,'batch_matches':8,'epochs':1,'minibatch':256,
        'replay_directory':str(out/'replay'),'deadline_unix':config['deadline_unix']}
    t=Trainer(c,device);restored=None
    try:
        initial=load_policy(c['warm_start'],device)
        if any(not torch.equal(v,t.model.state_dict()[k]) for k,v in initial.state_dict().items()):raise RuntimeError('weight transfer changed')
        anchor=deepcopy(t.training_anchor.state_dict());first=t.batch();traces=deepcopy(t.last_traces)
        expected_count={'current':16,'fixed':8,'mixed':12}[c['mode']]
        if first['learner_trajectories']!=expected_count or not all(first['learner_actions_by_seat']):raise RuntimeError('learner seat/role mismatch')
        if min(first['actor_change'],first['context_change'])<=0:raise RuntimeError('no actual PPO update')
        for g in traces:list(replay(g))
        t.save(out/'resume.pt');expected=t.batch();weights=deepcopy(t.model.state_dict());expected_traces=deepcopy(t.last_traces);t.close()
        restored=Trainer(c,device);restored.load(out/'resume.pt');actual=restored.batch()
        if expected_traces!=restored.last_traces or expected['per_game']!=actual['per_game']:raise RuntimeError('resume trajectories changed')
        if any(not torch.equal(v,restored.model.state_dict()[k]) for k,v in weights.items()):raise RuntimeError('resume weights changed')
        if any(not torch.equal(v,restored.training_anchor.state_dict()[k]) for k,v in anchor.items()):raise RuntimeError('fixed opponent updated')
        for g in restored.last_traces:list(replay(g))
        state=deepcopy(restored.replay.state()) if restored.replay else None;rng=restored.rng.get_state().clone();pool=restored.pool
        evaluation=restored.evaluate_panel(c['development_seeds'][:4])
        if not torch.equal(rng,restored.rng.get_state()) or restored.pool is not pool:raise RuntimeError('evaluation changed RNG/pool')
        if restored.replay and restored.replay.state()!=state:raise RuntimeError('evaluation entered replay')
        path=out/'portable.pt';export_policy(restored.model,path,c);loaded=load_policy(path,device)
        if any(not torch.equal(v,loaded.state_dict()[k]) for k,v in weights.items()):raise RuntimeError('portable mismatch')
        write_json(out/'traces.json',traces)
        result=dict(passed=True,source_id=source_id(),gpu=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
            gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),
            initial_weight_parity=True,exact_resume=True,raw_and_shaped_reward_replay=True,stationary_opponent_unchanged=True,
            first=compact(first),second=compact(actual),evaluation=compact(evaluation),wall_seconds=time.monotonic()-start)
        write_json(out/'result.json',result);print(result,flush=True)
    finally:
        t.close()
        if restored:restored.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();run(read_json(a.config),a.out,a.device)
