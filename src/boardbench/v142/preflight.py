"""CUDA update, exact resume, masks, full trace replay, and RNG-isolated evaluation."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import torch
from boardbench.artifacts.store import read_json, write_json, source_id
from boardbench.environments.harmonies_market import HarmoniesMarket
from .encoding import LAYOUT
from .training import Trainer, replay


def trial_config(repo, campaign, initialization='solo', replica=0):
    repo=Path(repo)
    return dict(**campaign['training'], layout=LAYOUT, rules_version=HarmoniesMarket.rules_version,
        parent=str(repo/campaign['parent']), parent_sha256=campaign['parent_sha256'],
        duo_parent=str(repo/campaign['duo_parent']),duo_parent_sha256=campaign['duo_parent_sha256'],
        native_artifact=read_json(repo/campaign['native_manifest']), objective='score',
        initialization=initialization,seed=14201+replica,workers=campaign['workers'],
        batch_matches=campaign['batch_matches'],max_seconds=campaign['wall_seconds'],
        development_seeds=list(range(242000000,242000000+campaign['development_environments'])),
        confirmation_seeds=list(range(342000000,342000000+campaign['confirmation_environments'])),
        source_id=source_id(repo))


def run(repo, campaign, out, initialization, baseline_only=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);tick=time.monotonic()
    c=trial_config(repo,campaign,initialization)
    if not torch.cuda.is_available(): raise RuntimeError('real CUDA required')
    if baseline_only:
        t=Trainer(c,'cuda')
        try:
            initial=t.evaluate(c['development_seeds'])
            write_json(out/'result.json',dict(**initial,source_id=source_id(repo),initialization=initialization,
                                            wall_seconds=time.monotonic()-tick))
        finally: t.close()
        return
    c.update(workers=3,batch_matches=8,minibatch=256,epochs=1)
    t=Trainer(c,'cuda')
    first=t.batch()
    if first['actor_change']<=0 or first['context_change']<=0: raise RuntimeError('silent no-update')
    for trace in t.last_traces: replay(trace)
    t.save(out/'resume.pt');expected=t.batch();weights=deepcopy(t.model.state_dict());t.close()
    restored=Trainer(c,'cuda');restored.load(out/'resume.pt');actual=restored.batch()
    if expected['per_game']!=actual['per_game']: raise RuntimeError('resume trajectory mismatch')
    for key,value in weights.items():
        if not torch.equal(value,restored.model.state_dict()[key]): raise RuntimeError('resume weight mismatch: '+key)
    rng=restored.rng.get_state().clone()
    evaluation=restored.evaluate(c['development_seeds'][:4])
    if not torch.equal(rng,restored.rng.get_state()): raise RuntimeError('evaluation consumes training RNG')
    restored.close()
    result=dict(passed=True,source_id=source_id(repo),gpu=torch.cuda.get_device_name(),initialization=initialization,
        wall_seconds=time.monotonic()-tick,first={k:v for k,v in first.items() if k!='per_game'},
        evaluation={k:v for k,v in evaluation.items() if k!='per_game'},exact_next_update=True,replayed_games=8)
    write_json(out/'result.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('repo','config','out','initialization'): p.add_argument('--'+key,required=True)
    p.add_argument('--baseline-only',action='store_true')
    a=p.parse_args();run(a.repo,read_json(a.config),a.out,a.initialization,a.baseline_only)
