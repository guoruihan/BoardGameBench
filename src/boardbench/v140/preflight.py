"""Actual-device update/resume check using the mature exported solo parent."""
import argparse
from copy import deepcopy
from pathlib import Path
import time
import torch

from boardbench.artifacts.store import read_json, write_json, source_id
from .encoding import LAYOUT
from .training import Trainer


def trial_config(repo, campaign, objective='margin', replica=0):
    repo = Path(repo)
    return dict(**campaign['training'], layout=LAYOUT, parent=str(repo/campaign['parent']),
                parent_sha256=campaign['parent_sha256'], native_artifact=read_json(repo/campaign['native_manifest']),
                objective=objective, seed=14001+replica, workers=campaign['workers'],
                batch_matches=campaign['batch_matches'], max_seconds=campaign['wall_seconds'],
                development_seeds=list(range(200000000, 200000000+campaign['development_environments'])),
                confirmation_seeds=list(range(300000000, 300000000+campaign['confirmation_environments'])),
                source_id=source_id(repo))


def run(repo, config, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    c = trial_config(repo, config)
    c.update(workers=3, batch_matches=8, minibatch=256, epochs=1, snapshot_interval=1)
    if not torch.cuda.is_available(): raise RuntimeError('real CUDA required')
    t = Trainer(c, 'cuda')
    first = t.batch()
    if first['actor_change'] <= 0 or first['context_change'] <= 0: raise RuntimeError('silent no-update')
    t.save(out/'resume.pt')
    expected = t.batch(); weights = deepcopy(t.model.state_dict()); t.close()
    restored = Trainer(c, 'cuda'); restored.load(out/'resume.pt'); actual = restored.batch()
    if expected['per_game'] != actual['per_game']: raise RuntimeError('resume trajectory mismatch')
    for key, value in weights.items():
        if not torch.equal(value, restored.model.state_dict()[key]): raise RuntimeError('resume update mismatch: '+key)
    rng = restored.rng.get_state().clone()
    evaluation = restored.evaluate(c['development_seeds'][:4])
    if not torch.equal(rng, restored.rng.get_state()): raise RuntimeError('evaluation consumes training RNG')
    restored.close()
    result = dict(passed=True, source_id=source_id(repo), gpu=torch.cuda.get_device_name(),
                  wall_seconds=time.monotonic()-start, first={k:v for k,v in first.items() if k!='per_game'},
                  evaluation={k:v for k,v in evaluation.items() if k!='per_game'}, exact_next_update=True)
    write_json(out/'result.json', result); print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', required=True); p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    args = p.parse_args(); run(args.repo, read_json(args.config), args.out)
