"""Small residual heads: MC components plus a separately identified ranking task."""
import argparse
from copy import deepcopy
from pathlib import Path
import signal
import time
import json
import numpy as np
import torch
from boardbench.artifacts.store import digest,read_json,write_json
from boardbench.v145.policy import load_policy
from boardbench.v149.value import remaining_value
from .value import Heads,load_head
from .data import training_seed


def pair_loss(pred,target,group):
    # Center each sibling set: equal weighting per root/future, never per action
    # count. The small mean term fixes the otherwise unidentifiable offset.
    losses=[]
    for g in torch.unique(group):
        mask=group==g;p=pred[mask];q=target[mask]
        if len(p)<2:raise ValueError('at least two sibling candidates')
        losses.append(((p-p.mean())-(q-q.mean())).square().mean()+.05*(p.mean()-q.mean()).square())
    return torch.stack(losses).mean()


class Learner:
    def __init__(self,c,device):
        torch.set_num_threads(1);self.c=c;self.device=device;self.steps=0
        if device=='cuda':
            if not torch.cuda.is_available():raise RuntimeError('real CUDA required')
            torch.cuda.set_per_process_memory_fraction(.15)
        m=read_json(c['dataset']);pm=read_json(c['pairs'])
        if (m['format']!='v150_components' or pm['format']!='v150_habitat_pairs'
                or m['split']!='train' or pm['split']!='train'):raise ValueError('train schemas only')
        for path,sha in [(c['dataset'],c['dataset_sha256']),(c['pairs'],c['pairs_sha256']),
                (m['path'],m['sha256']),(pm['path'],pm['sha256']),(c['base_model'],c['base_sha256'])]:
            if digest(path)!=sha:raise ValueError('immutable training data/model')
        with np.load(m['path'],allow_pickle=False) as f:raw={k:f[k] for k in f.files}
        with np.load(pm['path'],allow_pickle=False) as f:pairs={k:f[k] for k in f.files}
        for arrays in (raw,pairs):
            if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('finite targets/inputs')
            if any(not training_seed(int(s)) for s in np.unique(arrays['seed'])):raise ValueError('training seeds')
        self.valid=raw['seed']%5==0;self.train_ids=np.flatnonzero(~self.valid)
        if not self.valid.any() or not len(self.train_ids):raise ValueError('grouped train/validation required')
        self.pair_groups={int(g):np.flatnonzero(pairs['group']==g) for g in np.unique(pairs['group'])}
        for ids in self.pair_groups.values():
            if len(np.unique(pairs['seed'][ids]))!=1:raise ValueError('sibling group crosses environments')
        self.train_groups=[g for g,ids in self.pair_groups.items() if pairs['seed'][ids[0]]%5!=0]
        self.valid_groups=[g for g,ids in self.pair_groups.items() if pairs['seed'][ids[0]]%5==0]
        if not self.train_groups or not self.valid_groups:raise ValueError('grouped pair validation required')
        self.base=load_policy(c['base_model'],device).requires_grad_(False)
        self.x=torch.as_tensor(raw['x'],dtype=torch.float64,device=device)
        self.y=torch.as_tensor(raw['y'],dtype=torch.float64,device=device)
        self.px=torch.as_tensor(pairs['x'],dtype=torch.float64,device=device)
        self.q=torch.as_tensor(pairs['q'],dtype=torch.float64,device=device)
        self.group=torch.as_tensor(pairs['group'],device=device)
        with torch.inference_mode():
            self.base_v=torch.cat([remaining_value(self.base,x) for x in self.x.split(256)])
            self.pbase=torch.cat([remaining_value(self.base,x) for x in self.px.split(256)])
        torch.manual_seed(c['seed']);self.rng=np.random.default_rng(c['seed'])
        if c.get('warm_head'):
            if digest(c['warm_head'])!=c['warm_sha256']:raise ValueError('warm head changed')
            self.head=load_head(c['warm_head'],device)
        else:self.head=Heads().to(device)
        self.optimizer=torch.optim.Adam(self.head.parameters(),lr=c['learning_rate'])
        self.validation_seeds=sorted(set(raw['seed'][self.valid].tolist()))
        self.opponents=sorted({g['opponent_sha256'] for g in m['games']})
        self.ranking_target=pm.get('continuation','habitat_focal_frozen_opponent')

    def step(self):
        ids=self.rng.choice(self.train_ids,self.c.get('batch',256))
        p,_=self.head(self.x[ids],self.base_v[ids]);target=self.y[ids]
        total=(p.sum(-1)-target.sum(-1)).square().mean()
        weights=torch.tensor([self.c.get('terrain_weight',1.),self.c.get('animal_weight',1.)],device=self.device)
        comp=((p-target).square()*weights).mean()
        rank=total*0
        if self.c['rank_loss']:
            groups=self.rng.choice(self.train_groups,min(16,len(self.train_groups)),replace=False)
            slots=np.concatenate([self.pair_groups[int(g)] for g in groups])
            _,r=self.head(self.px[slots],self.pbase[slots]);rank=pair_loss(r,self.q[slots],self.group[slots])
        loss=total+self.c['component_loss']*comp+self.c['rank_loss']*rank
        if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
        self.optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(self.head.parameters(),1.,error_if_nonfinite=True)
        self.optimizer.step();self.steps+=1
        return dict(steps=self.steps,total=float(total.detach()),component=float(comp.detach()),
                    ranking=float(rank.detach()),loss=float(loss.detach()),gradient_norm=float(norm))

    def validate(self):
        ids=np.flatnonzero(self.valid);errors=[]
        with torch.inference_mode():
            for chunk in np.array_split(ids,max(1,int(np.ceil(len(ids)/512)))):
                p,_=self.head(self.x[chunk],self.base_v[chunk]);errors.append((p-self.y[chunk]).cpu().numpy())
            rank=[];correct=0;count=0
            for g in self.valid_groups:
                ids=self.pair_groups[g];_,p=self.head(self.px[ids],self.pbase[ids]);q=self.q[ids]
                rank.append(float(pair_loss(p,q,self.group[ids])))
                for i in range(len(ids)):
                    for j in range(i):
                        if abs(float(q[i]-q[j]))>1/150:
                            count+=1;correct+=int(float((p[i]-p[j])*(q[i]-q[j]))>0)
        err=np.concatenate(errors);total=float(np.mean(err.sum(-1)**2));comp=float(np.mean(err**2));ranking=float(np.mean(rank))
        return dict(objective=total+self.c['component_loss']*comp+self.c['rank_loss']*ranking,
            total_rmse=150*total**.5,terrain_rmse=150*np.mean(err[:,0]**2)**.5,animal_rmse=150*np.mean(err[:,1]**2)**.5,
            total_bias=float(150*err.sum(-1).mean()),ranking_loss=ranking,pair_accuracy=correct/count if count else None,
            pairs=count,validation_seeds=self.validation_seeds)

    def state(self):
        return dict(config=self.c,steps=self.steps,head=self.head.state_dict(),optimizer=self.optimizer.state_dict(),rng=self.rng.bit_generator.state)

    def save(self,path):
        path=Path(path);tmp=path.with_suffix('.partial.pt');torch.save(self.state(),tmp);tmp.replace(path)

    def restore(self,state):
        if state['config']!=self.c:raise ValueError('immutable fit config')
        self.head.load_state_dict(state['head']);self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state=state['rng'];self.steps=state['steps']

    def export(self,path):
        path=Path(path);tmp=path.with_suffix('.partial.pt')
        torch.save(dict(format='v150_component_rank_head',hidden=128,state={k:v.detach().cpu().clone() for k,v in self.head.state_dict().items()}),tmp);tmp.replace(path)


def run(c,out,device,resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=resume)
    if resume and read_json(out/'config.json')!=c:raise ValueError('immutable fit config')
    write_json(out/'config.json',c)
    start=time.time();learner=Learner(c,device);stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    if resume:learner.restore(torch.load(out/'latest.pt',map_location=device,weights_only=False))
    else:
        initial=deepcopy(learner.state());row=learner.step();saved=deepcopy(learner.state())
        learner.step();expected=deepcopy(learner.head.state_dict());learner.restore(saved);learner.step()
        if not all(torch.equal(v,learner.head.state_dict()[k]) for k,v in expected.items()):raise ValueError('nonexact same-device resume')
        if row['gradient_norm']<=0:raise ValueError('no gradients')
        learner.export(out/'gate_head.pt');portable=load_head(out/'gate_head.pt',device)
        if not all(torch.equal(v,portable.state_dict()[k]) for k,v in learner.head.state_dict().items()):raise ValueError('export changed weights')
        write_json(out/'device_gate.json',dict(device=device,device_name=torch.cuda.get_device_name() if device=='cuda' else 'cpu',
            exact_same_device_resume=True,gradient_norm=row['gradient_norm'],portable_weights_exact=True))
        learner.restore(initial)
    selection=read_json(out/'selection.json') if (out/'selection.json').exists() else dict(best=learner.validate(),best_step=0,stale=0)
    if not (out/'head.pt').exists():learner.export(out/'head.pt')
    while learner.steps<c['steps'] and not stop[0] and time.time()<c['deadline_unix']-30:
        row=learner.step()
        if learner.steps%100==0:
            metrics=learner.validate();row['validation']=metrics
            if metrics['objective']<selection['best']['objective']:
                selection=dict(best=metrics,best_step=learner.steps,stale=0);learner.export(out/'head.pt')
            else:selection['stale']+=1
            write_json(out/'selection.json',selection)
        if learner.steps%25==0:
            learner.save(out/'latest.pt')
            with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if selection['stale']>=6:break
    learner.save(out/'latest.pt')
    manifest=dict(format='v150_component_value',target='remaining_own_terrain_animal_div150',
        ranking_target=learner.ranking_target+'_final_own_score_div150',base_model=c['base_model'],base_sha256=c['base_sha256'],
        head=str((out/'head.pt').resolve()),head_sha256=digest(out/'head.pt'),opponent_hashes=learner.opponents,
        config_sha256=digest(out/'config.json'),validation=selection['best'])
    write_json(out/'value_manifest.json',manifest)
    write_json(out/('interrupted.json' if stop[0] else 'done.json'),dict(steps=learner.steps,selection=selection,
        seconds=time.time()-start,device=device,deadline_unix=c['deadline_unix']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true');a=p.parse_args()
    run(read_json(a.config),a.out,a.device,a.resume)
