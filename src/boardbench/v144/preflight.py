"""Real CUDA gates for score/margin reward, both-seat learning and exact resume."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import torch
from boardbench.artifacts.store import digest,read_json,write_json,source_id
from boardbench.environments.harmonies_duo import HarmoniesDuo
from boardbench.v143.preflight import transfer_check
from boardbench.v143.replay import replay
from .policy import LAYOUT,export_policy,load_policy
from .training import Trainer,compact


def trial_config(repo,campaign,arm,round_index=0):
    repo=Path(repo)
    return dict(**{**campaign['training'],**arm},layout=LAYOUT,rules_version=HarmoniesDuo.rules_version,
        parent=str(repo/campaign['parent']),parent_sha256=campaign['parent_sha256'],
        native_artifact=read_json(repo/campaign['native_manifest']),evaluation_opponents=campaign.get('evaluation_opponents',[]),
        seed=14401+round_index*100+arm.get('replica',0),workers=campaign['workers'],batch_matches=campaign['batch_matches'],
        source_id=source_id(repo),development_seeds=list(range(244000000,244000000+campaign['development_environments'])),
        decision_seeds=list(range(244010000+round_index*1000,244010000+round_index*1000+campaign['decision_environments'])),
        confirmation_seeds=list(range(344000000,344000000+campaign['confirmation_environments'])))


def run(config,out,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);tick=time.monotonic()
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('real CUDA required')
    config={**config,'workers':3 if device=='cuda' else 0,'batch_matches':8,'epochs':1,'minibatch':256,
            'snapshot_interval':1,'history_size':2}
    t=Trainer(config,device)
    try:
        parity=transfer_check(t.model,device) if not config.get('warm_start') else None
        frozen=deepcopy(t.frozen.state_dict());first=t.batch()
        expected_first=14 if config['mode']=='diverse' else 16
        if (first['actor_change']<=0 or first['context_change']<=0
                or first['learner_trajectories']!=expected_first or not all(first['learner_actions_by_seat'])):
            raise RuntimeError('both-seat actual updates required')
        for game in t.last_traces:list(replay(game,config['objective']))
        t.save(out/'resume.pt');expected=t.batch();weights=deepcopy(t.model.state_dict());traces=deepcopy(t.last_traces);t.close()
        restored=Trainer(config,device)
        try:
            restored.load(out/'resume.pt');actual=restored.batch()
            if expected['per_game']!=actual['per_game'] or traces!=restored.last_traces:raise RuntimeError('resume trajectory mismatch')
            if any(not torch.equal(v,restored.model.state_dict()[k]) for k,v in weights.items()):raise RuntimeError('resume weight mismatch')
            if any(not torch.equal(v,restored.frozen.state_dict()[k]) for k,v in frozen.items()):raise RuntimeError('fixed opponent changed')
            expected_second={'current':16,'league':14,'diverse':12}[config['mode']]
            if actual['learner_trajectories']!=expected_second:raise RuntimeError('frozen/history side entered PPO')
            for game in restored.last_traces:list(replay(game,config['objective']))
            rng=restored.rng.get_state().clone();evaluation=restored.evaluate_panel(config['development_seeds'][:4])
            if not torch.equal(rng,restored.rng.get_state()):raise RuntimeError('evaluation consumed RNG')
            path=out/'portable.pt';export_policy(restored.model,path,config)
            portable=load_policy(path,device)
            if any(not torch.equal(v,portable.state_dict()[k]) for k,v in restored.model.state_dict().items()):
                raise RuntimeError('portable roundtrip failed')
        finally:restored.close()
    finally:t.close()
    result=dict(passed=True,source_id=source_id(),objective=config['objective'],mode=config['mode'],
        gae_lambda=config['gae_lambda'],gpu=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
        transfer=parity,exact_resume=True,first=compact(first),second=compact(actual),evaluation=compact(evaluation),
        wall_seconds=time.monotonic()-tick,portable_sha256=digest(path))
    write_json(out/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(read_json(a.config),a.out,a.device)
