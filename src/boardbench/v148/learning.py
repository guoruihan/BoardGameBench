"""Search-distribution policy learning and remaining-score value regression.

No expert record is treated as an on-policy PPO sample. Validation environments
are held out as whole groups, including both seats and all teacher repetitions.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from boardbench.artifacts.store import digest, read_json
from boardbench.v145.policy import load_policy, export_policy as export_legacy
from .data import training_seed


class Learner:
    def __init__(self, config, device='cpu'):
        self.config, self.device = config, str(device)
        torch.set_num_threads(1)
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('real CUDA required')
        if config['mode'] not in ('policy','joint','value'):
            raise ValueError('explicit policy/joint/value learning mode required')
        if (not 0 < config['learning_rate'] <= .001 or not 0 <= config['anchor_weight'] <= 10
                or not 0 <= config['value_weight'] <= 10 or config['minibatch'] < 1):
            raise ValueError('invalid bounded learner parameters')
        if digest(config['model']) != config['model_sha256']:
            raise ValueError('immutable parent model')
        if digest(config['dataset']) != config['dataset_sha256']:
            raise ValueError('immutable dataset manifest')
        meta = read_json(config['dataset'])
        if (meta['format'] != 'v148_search_training' or meta['split'] != 'train'
                or meta['value_target'] != 'remaining_own_score_div150'
                or digest(meta['path']) != meta['sha256']):
            raise ValueError('training data schema/hash')
        with np.load(meta['path'], allow_pickle=False) as raw:
            arrays = {k:raw[k] for k in raw.files}
        seeds = sorted(set(arrays['seed'].tolist()))
        if len(seeds)<2 or any(not training_seed(s) for s in seeds):
            raise ValueError('at least two training environment groups required')
        # The split is fixed by data, not by training RNG or trajectory order.
        held = set(seeds[::5]); validation = np.asarray([s in held for s in arrays['seed']])
        self.train_ids = torch.as_tensor(np.flatnonzero(~validation),device=device)
        self.valid_ids = torch.as_tensor(np.flatnonzero(validation),device=device)
        if (not np.isfinite(arrays['x']).all() or not np.isfinite(arrays['mc']).all()
                or not np.isfinite(arrays['pi']).all() or np.any(arrays['pi'] < 0)
                or np.any(arrays['pi'][~arrays['mask']])
                or not np.allclose(arrays['pi'].sum(-1),arrays['label']>0)):
            raise ValueError('finite targets and legal policy support required')
        self.data = {k:torch.as_tensor(v,device=device,dtype=torch.float64 if k in ('x','pi','mc','q') else None)
                     for k,v in arrays.items() if k not in ('seed','seat')}
        self.mcts_ids = self.train_ids[self.data['label'][self.train_ids]==2]
        if not 0<=config.get('mcts_fraction',.7)<=1:raise ValueError('bounded MCTS sampling mixture')
        self.validation_seeds = sorted(held)
        self.opponent_hashes = sorted({g['opponent_sha256'] for g in meta['games']})
        self.model = load_policy(config['model'],device)
        self.anchor = deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config['learning_rate'])
        self.rng = torch.Generator(device=device).manual_seed(config['seed'])
        self.steps = 0

    def step(self):
        ids = self.train_ids[torch.randint(len(self.train_ids),(self.config['minibatch'],),
                                         generator=self.rng,device=self.device)]
        if len(self.mcts_ids) and self.config['mode']!='value':
            count=int(self.config['minibatch']*self.config.get('mcts_fraction',.7))
            ids[:count]=self.mcts_ids[torch.randint(len(self.mcts_ids),(count,),generator=self.rng,device=self.device)]
        d = {k:v[ids] for k,v in self.data.items()}
        logits,value = self.model(d['x'],d['mask'])
        logp = logits.log_softmax(-1)
        valid = d['label']>0
        policy = -(d['pi'][valid]*logp[valid]).sum(-1).mean() if valid.any() else logits.sum()*0.
        value_loss = (value-d['mc']).square().mean()
        with torch.no_grad():
            anchor_logits,_ = self.anchor(d['x'],d['mask'])
            anchor_p = anchor_logits.softmax(-1)
        anchor_kl = (anchor_p*(anchor_logits.log_softmax(-1)-logp)).sum(-1).mean()
        loss = (value_loss if self.config['mode']=='value' else
                policy+self.config['anchor_weight']*anchor_kl+
                (self.config['value_weight']*value_loss if self.config['mode']=='joint' else 0.))
        if not torch.isfinite(loss): raise RuntimeError('nonfinite search learning loss')
        self.optimizer.zero_grad(set_to_none=True); loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(),.5,error_if_nonfinite=True)
        self.optimizer.step(); self.steps+=1
        return dict(steps=self.steps,loss=float(loss.detach()),policy_loss=float(policy.detach()),
                    value_loss=float(value_loss.detach()),anchor_kl=float(anchor_kl.detach()),gradient_norm=float(gradient))

    def validate(self):
        errors, baseline, policy, accuracy, count = [], [], [], 0, 0
        with torch.inference_mode():
            for ids in self.valid_ids.split(512):
                d={k:v[ids] for k,v in self.data.items()}
                logits,value=self.model(d['x'],d['mask']);_,old=self.anchor(d['x'],d['mask'])
                errors.extend((value-d['mc']).square().cpu().tolist())
                baseline.extend((old-d['mc']).square().cpu().tolist())
                valid=d['label']>0
                policy.extend((-(d['pi'][valid]*logits[valid].log_softmax(-1)).sum(-1)).cpu().tolist())
                accuracy+=int((logits[valid].argmax(-1)==d['pi'][valid].argmax(-1)).sum())
                count+=int(valid.sum())
        return dict(value_mse=float(np.mean(errors)),baseline_value_mse=float(np.mean(baseline)),
                    policy_ce=float(np.mean(policy)) if policy else None,
                    top_action_agreement=accuracy/count if count else None,
                    rows=len(errors),environments=len(self.validation_seeds),
                    validation_seeds=self.validation_seeds,split='held_out_training_environments')

    def save(self,path):
        path=Path(path);tmp=path.with_suffix('.partial.pt')
        torch.save(dict(version=148,config=self.config,steps=self.steps,model=self.model.state_dict(),
                        optimizer=self.optimizer.state_dict(),rng=self.rng.get_state()),tmp)
        tmp.replace(path)

    def load(self,path):
        state=torch.load(path,map_location=self.device,weights_only=False)
        if state['version']!=148 or state['config']!=self.config: raise ValueError('exact training config required')
        self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
        self.rng.set_state(state['rng'].cpu());self.steps=state['steps']

    def export(self,path):
        # Preserve the existing inference ABI; extra metadata identifies the new
        # training objective rather than pretending these were PPO updates.
        c=dict(parent_sha256=self.config['model_sha256'],warm_start_sha256=self.config['model_sha256'],
               mode='fixed',progress_scale=0.,sil_weight=0.,gae_lambda=1.,source_id=self.config.get('source_id'))
        value=export_legacy(self.model,path,c)
        value['training']=dict(algorithm='search_policy_iteration',mode=self.config['mode'],
                               dataset_sha256=self.config['dataset_sha256'],steps=self.steps,
                               value_target='remaining_own_score_div150')
        path=Path(path);tmp=path.with_suffix('.partial.pt');torch.save(value,tmp);tmp.replace(path)
        return value
