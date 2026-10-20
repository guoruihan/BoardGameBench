"""Frozen-parent PPO plus bounded, independently advantage-filtered guidance."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from boardbench.artifacts.store import digest, read_json
from boardbench.v12.training import policy_distribution
from boardbench.v127.training import Trainer as BaseTrainer, run as base_run
from .data import weights


class Trainer(BaseTrainer):
    def __init__(self,config,device='cpu'):
        if config.get('guidance_version')!='1.2.11':raise ValueError('explicit V131 guidance version required')
        super().__init__(config,device)
        if config['arm'] not in ('control','habitat','student') or not 0<=config['bc_weight']<=.05:
            raise ValueError('bounded registered guidance arm required')
        spec=config['paired_dataset']
        if digest(spec['path'])!=spec['sha256']:raise ValueError('paired dataset checksum mismatch')
        with np.load(spec['path'],allow_pickle=False) as z:
            self.paired={k:z[k].copy() for k in z.files}
        d=self.paired;hold=d['validation'].astype(bool)
        fit,val=set(d['seed'][~hold]),set(d['seed'][hold])
        if not fit or not val or fit&val or not (fit|val)<=set(self.manifest['sets']['train']):
            raise ValueError('paired dataset split leakage')
        if (d['x'].shape!=(len(d['y']),1346) or d['mask'].shape!=(len(d['y']),240)
                or not np.isfinite(d['x']).all() or (d['samples']<2).any()
                or not d['mask'][np.arange(len(d['y'])),d['y']].all()
                or not d['mask'][np.arange(len(d['y'])),d['student_y']].all()
                or (d['y']==d['student_y']).any()):
            raise ValueError('invalid paired features/actions/sample counts')
        self.train_seeds=[s for s in self.train_seeds if s not in val]
        if not self.train_seeds:raise ValueError('empty online train pool')
        self.guidance={};self.validation={}
        for name in ('habitat','student'):
            weight=weights(d[name+'_mean'],d[name+'_se'],z=config['guidance_z'],
                           minimum_gain=config['guidance_minimum_gain'],scale=config['guidance_gain_scale'])
            for destination,keep in ((self.guidance,~hold),(self.validation,hold)):
                selected=keep&(weight>0)
                destination[name]={k:v[selected] for k,v in d.items()}
                destination[name]['weight']=weight[selected]
        if config['arm']!='control' and len(self.guidance[config['arm']]['y'])<config.get('minimum_guidance_rows',32):
            raise ValueError('insufficient reliable nonforced fit labels; no empty-signal run')

    def imitation_weight(self):return self.config['bc_weight']

    def guidance_loss(self,ids):
        d=self.guidance[self.config['arm']]
        logits,_=self.model(torch.as_tensor(d['x'][ids],device=self.device,dtype=torch.float64),
                            torch.as_tensor(d['mask'][ids],device=self.device))
        ce=F.cross_entropy(logits,torch.as_tensor(d['y'][ids],device=self.device),reduction='none')
        return (ce*torch.as_tensor(d['weight'][ids],device=self.device,dtype=torch.float64)).mean()

    def demonstration_loss(self):
        n=len(self.guidance[self.config['arm']]['y'])
        return self.guidance_loss(self.rng.integers(n,size=min(n,256)))

    def update(self,rows):
        diagnostics={}
        if self.config['bc_weight']:
            # Deterministic diagnostic, does not consume sampling/training RNG.
            x,mask,action,_,_,adv,_=self.tensors(rows)
            adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            logits,_=self.model(x[:256],mask[:256])
            policy=-(policy_distribution(logits).log_prob(action[:256])*adv[:256]).mean()
            params=[p for n,p in self.model.named_parameters() if not n.startswith('critic.')]
            gp=torch.autograd.grad(policy,params,allow_unused=True)
            auxiliary=self.guidance_loss(np.arange(min(256,len(self.guidance[self.config['arm']]['y']))))
            ga=torch.autograd.grad(self.config['bc_weight']*auxiliary,params,allow_unused=True)
            a=torch.cat([v.flatten() for v in gp if v is not None])
            b=torch.cat([v.flatten() for v in ga if v is not None])
            diagnostics={'policy_gradient_norm_probe':float(a.norm()),'guidance_gradient_norm_probe':float(b.norm()),
                         'guidance_policy_gradient_cosine_probe':float(a.dot(b)/(a.norm()*b.norm()).clamp_min(1e-15))}
        return {**super().update(rows),**diagnostics}

    def import_sampling_continuation(self,spec):
        if self.actions or self.updates or self.optimizer.state:raise ValueError('fresh trainer required')
        path=Path(spec['checkpoint']);meta=read_json(path.with_suffix('.json'))
        if digest(path)!=spec['sha256'] or meta['sha256']!=spec['sha256'] or meta['source_id']!=spec['source_id']:
            raise ValueError('parent hash/source mismatch')
        state=torch.load(path,map_location=self.device,weights_only=False)
        if state['version']!=127 or state['source_id']!=spec['source_id']:raise ValueError('unsupported parent')
        for key in ('hidden','reward_scale','dataset_sha256','split_sha256','gae_lambda','learning_rate',
                    'epochs','minibatch','target_kl','entropy_coef','batch_episodes'):
            if state['config'][key]!=self.config[key]:raise ValueError('protected parent field: '+key)
        if any(v.dtype!=torch.float64 or not torch.isfinite(v).all() for v in state['model'].values()):
            raise ValueError('finite FP64 parent required')
        self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        for key in ('episodes','actions','ppo_actions','batches','warm_batches','updates'):
            setattr(self,key,state['counters'][key])
        self.lineage={'kind':'continuation_aware_guidance','parent_checkpoint':str(path),'sha256':spec['sha256'],
                      'parent_source_id':spec['source_id'],'starting_ppo_actions':self.ppo_actions,
                      'preserved':'actor,critic,both Adam states and lifetime counters',
                      'rng':'new paired experiment seed; exact same-run resume separately preserves all RNG',
                      'changed':'filtered guidance; no teacher returns enter PPO/critic'}

    def evaluate(self,seeds):
        result=super().evaluate(seeds);diagnostics={}
        with torch.no_grad():
            for split,pools in (('fit',self.guidance),('validation',self.validation)):
                for name,d in pools.items():
                    n=len(d['y'])
                    if not n:diagnostics[split+'_'+name]={'n':0};continue
                    logits,_=self.model(torch.as_tensor(d['x'],device=self.device,dtype=torch.float64),
                                        torch.as_tensor(d['mask'],device=self.device))
                    y=torch.as_tensor(d['y'],device=self.device)
                    diagnostics[split+'_'+name]={'n':n,'nll':float(F.cross_entropy(logits,y)),
                        'action_agreement':float((logits.argmax(-1)==y).double().mean())}
        result['guidance_diagnostic']=diagnostics
        return result


def run(config,out,device='cuda'):return base_run(config,out,device,trainer_class=Trainer)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(read_json(a.config),a.out,a.device)
