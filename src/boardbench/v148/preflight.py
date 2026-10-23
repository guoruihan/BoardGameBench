"""Real device gradients, exact next update, portable parity and full-game audit."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import time

import torch

from boardbench.artifacts.store import read_json,write_json,source_id
from boardbench.v145.policy import load_policy
from .learning import Learner
from .evaluation import Trainer


def run(config,out,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    if device=='cuda' and (not torch.cuda.is_available() or torch.cuda.device_count()!=1):
        raise RuntimeError('one actual CUDA GPU required')
    torch.use_deterministic_algorithms(True)
    c={**config,'mode':'joint','minibatch':32,'value_weight':.25}
    t=Learner(c,device);evaluator=None
    before=deepcopy(t.model.state_dict());anchor=deepcopy(t.anchor.state_dict())
    first=t.step()
    if first['gradient_norm']<=0 or not any(not torch.equal(v,t.model.state_dict()[k]) for k,v in before.items()):
        raise RuntimeError('missing gradient')
    t.save(out/'resume.pt');expected=t.step();weights=deepcopy(t.model.state_dict())
    other=Learner(c,device);other.load(out/'resume.pt');actual=other.step()
    if expected!=actual or any(not torch.equal(v,other.model.state_dict()[k]) for k,v in weights.items()):
        raise RuntimeError('exact next-update restoration failed')
    if any(not torch.equal(v,other.anchor.state_dict()[k]) for k,v in anchor.items()):raise RuntimeError('anchor mutated')
    other.export(out/'portable.pt');portable=load_policy(out/'portable.pt',device)
    if any(not torch.equal(v,portable.state_dict()[k]) for k,v in weights.items()):raise RuntimeError('export parity')
    validation=other.validate();evaluation=None
    try:
        if c.get('evaluation_config'):
            ec={**c['evaluation_config'],'workers':3 if device=='cuda' else 0,'batch_matches':8}
            evaluator=Trainer(ec,device);frozen=deepcopy(evaluator.training_anchor.state_dict())
            evaluator.model.load_state_dict(weights)
            evaluation=evaluator.evaluate_panel(ec['development_seeds'][:4])
            if any(not g['natural_terminal'] for g in evaluation['per_game']):raise RuntimeError('partial game')
            if any(not torch.equal(v,evaluator.training_anchor.state_dict()[k]) for k,v in frozen.items()):
                raise RuntimeError('fixed opponent changed')
        result=dict(passed=True,source_id=source_id(),device=torch.cuda.get_device_name() if device=='cuda' else device,
                    gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),first=first,exact_resume=True,
                    portable_parity=True,anchor_unchanged=True,validation=validation,
                    evaluation=evaluation,wall_seconds=time.monotonic()-begin)
        if c.get('fit_smoke'):
            if evaluator:evaluator.close();evaluator=None
            from .fit import run as fit
            small={**c,'steps':4,'evaluate_interval':2,'finish_reserve_seconds':10,
                   'evaluation_config':{**ec,'development_seeds':ec['development_seeds'][:4],
                                        'decision_seeds':ec['decision_seeds'][:4]}}
            fit(small,out/'fit_smoke',device)
            result['fit_smoke']=read_json(out/'fit_smoke/done.json')
            result['wall_seconds']=time.monotonic()-begin
        write_json(out/'result.json',result)
        print({k:v for k,v in result.items() if k not in ('validation','evaluation')},flush=True)
        return result
    finally:
        if evaluator:evaluator.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');a=p.parse_args();run(read_json(a.config),a.out,a.device)
