"""Expert auxiliary updates AFTER true own-score complete-game PPO updates."""
import argparse
import math
import numpy as np
import torch
from boardbench.artifacts.store import digest,read_json
from boardbench.v144.training import run as base_run
from boardbench.v145.training import Trainer as PreviousTrainer
from .data import training_seed

LAYOUT='duo_public_expert_initialized_own_score_v147'


class Trainer(PreviousTrainer):
    VERSION=147
    LAYOUT=LAYOUT
    TRAINING_BASE=147000000
    DEVELOPMENT_BASE=247000000
    CONFIRMATION_BASE=347000000

    def __init__(self,config,device='cpu'):
        super().__init__(config,device)
        if config['progress_scale']!=0 or config['sil_weight']!=0:
            raise ValueError('V147 expert MC uses raw own returns; no potential/SIL intervention')
        # Match the teacher's actual fixed opponent, not an implicit transferred
        # solo policy. Optional extra opponents remain separately reported.
        self.references={'teacher_opponent':self.training_anchor,**{
            k:v for k,v in self.references.items() if k!='transfer'}}
        self.expert=None;self.bc_steps=0
        if config.get('expert_manifest'):
            if digest(config['expert_manifest'])!=config['expert_manifest_sha256']:raise ValueError('expert manifest changed')
            meta=read_json(config['expert_manifest'])
            if meta['format']!='v147_replayed_duo_training' or meta['split']!='train' or digest(meta['path'])!=meta['sha256']:
                raise ValueError('immutable train-only expert dataset required')
            with np.load(meta['path'],allow_pickle=False) as raw:
                data={k:raw[k] for k in raw.files}
            if any(not training_seed(int(s)) for s in data['seed']):raise ValueError('expert split leakage')
            if (not np.isfinite(data['x']).all() or not np.isfinite(data['mc']).all() or not np.isfinite(data['q']).all() or
                    not data['mask'][np.arange(len(data['action'])),data['action']].all() or
                    np.any(data['qmask'] & ~data['mask'])):raise ValueError('expert mask/target mismatch')
            self.expert={k:torch.as_tensor(v,device=device,dtype=torch.float64 if k in ('x','mc','q') else None)
                         for k,v in data.items() if k not in ('seed','seat')}
        if not 0<=config.get('expert_weight',0)<=1 or config.get('expert_decay_actions',1)<=0:
            raise ValueError('bounded expert auxiliary required')
        if (not math.isfinite(config.get('expert_temperature',4.)) or config.get('expert_temperature',4.)<=0
                or any(not 0<=config.get(k,0)<=1 for k in ('expert_q_weight','expert_value_weight'))):
            raise ValueError('bounded finite expert loss weights/temperature required')

    def checkpoint_extra_state(self):
        return dict(**super().checkpoint_extra_state(),bc_steps=self.bc_steps)

    def restore_extra_state(self,state):
        super().restore_extra_state(state);self.bc_steps=state.get('bc_steps',0)

    def expert_step(self,weight=1.):
        if self.expert is None:raise ValueError('no training experts')
        ids=torch.randint(len(self.expert['action']),(self.config.get('expert_minibatch',256),),generator=self.rng,device=self.device)
        sample={k:v[ids] for k,v in self.expert.items()}
        logits,value=self.model(sample['x'],sample['mask'])
        hard=torch.nn.functional.cross_entropy(logits,sample['action'])
        valid=sample['qmask'].sum(-1)>=2;q_loss=logits.sum()*0.
        if valid.any() and self.config.get('expert_q_weight',0):
            qm=sample['qmask'][valid]
            q=sample['q'][valid].masked_fill(~qm,-1e9)
            targets=torch.softmax(q/self.config.get('expert_temperature',4.),-1).detach()
            logp=torch.log_softmax(logits[valid].masked_fill(~qm,-1e9),-1)
            q_loss=-(targets*logp).sum(-1).mean()
        value_loss=(value-sample['mc']).square().mean()
        loss=weight*(hard+self.config.get('expert_q_weight',0)*q_loss+self.config.get('expert_value_weight',0)*value_loss)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite expert loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),.5,error_if_nonfinite=True);self.optimizer.step();self.bc_steps+=1
        return dict(expert_loss=float(loss.detach()),expert_ce=float(hard.detach()),expert_q_loss=float(q_loss.detach()),
                    expert_value_loss=float(value_loss.detach()),bc_steps=self.bc_steps,expert_weight=weight)

    def update(self,rows):
        # Parent first checks exact current-policy behavior ratios; no expert
        # update may happen between rollout collection and this check.
        result=super().update(rows)
        weight=self.config.get('expert_weight',0)*math.exp(-self.actions/self.config.get('expert_decay_actions',2000000))
        if weight>1e-5 and self.expert is not None:
            for _ in range(self.config.get('expert_steps',2)):result.update(self.expert_step(weight))
        return result


def run(config,out,device='cuda',resume=False):
    return base_run(config,out,device,resume,trainer_class=Trainer)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(read_json(a.config),a.out,a.device,a.resume)
