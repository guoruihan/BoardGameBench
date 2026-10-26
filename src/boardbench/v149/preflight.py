"""Real-device value gradients, exact recovery and value-only forward parity."""
import argparse
from copy import deepcopy
from pathlib import Path

import torch

from boardbench.artifacts.store import read_json, write_json
from boardbench.v148.learning import Learner
from boardbench.v145.policy import load_policy
from .value import remaining_value


def run(config,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.cuda.set_per_process_memory_fraction(.1)
    learner=Learner(config,device)
    before=deepcopy(learner.model.state_dict());first=learner.step()
    if not any(not torch.equal(v,learner.model.state_dict()[k]) for k,v in before.items()):
        raise RuntimeError('no value gradients')
    learner.save(out/'resume.pt');learner.step();expected=deepcopy(learner.model.state_dict())
    learner.load(out/'resume.pt');learner.step()
    if any(not torch.equal(v,learner.model.state_dict()[k]) for k,v in expected.items()):
        raise RuntimeError('non-exact next-update recovery')
    learner.export(out/'value.pt');portable=load_policy(out/'value.pt',device)
    ids=learner.valid_ids[:8];x=learner.data['x'][ids];mask=learner.data['mask'][ids]
    with torch.inference_mode():
        ordinary=portable(x,mask)[1];fast=remaining_value(portable,x)
    error=float((ordinary-fast).abs().max())
    if error>1e-10:raise RuntimeError('value-only path changed estimates')
    if device=='cuda':torch.cuda.synchronize()
    result=dict(passed=True,device=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
                gradient_norm=first['gradient_norm'],fast_value_error=error,exact_resume=True,
                peak_gpu_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0)
    write_json(out/'result.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();print(run(read_json(a.config),a.out,a.device))
