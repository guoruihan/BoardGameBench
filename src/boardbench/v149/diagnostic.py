"""Held-out train-state ranking against archived rollout candidate estimates.

Reference Q is the recorded habitat-continuation sample mean, not optimal Q or
closed-loop strong-search value. Its sample uncertainty is reported explicitly.
No labels or states from development/confirmation games enter training.
"""
from copy import deepcopy
import gzip
import json
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies_duo import player_view
from boardbench.v138.baseline import Planner
from .search import Search


def cases(manifest, count=12):
    meta=read_json(manifest)
    if meta['split']!='train':raise ValueError('training-source diagnostic only')
    held=set(sorted({g['seed'] for g in meta['games']})[::5])
    output=[];used=set()
    for g in meta['games']:
        if g['seed'] not in held or g['method']!='search_habitat':continue
        if digest(g['path'])!=g['sha256']:raise ValueError('trajectory changed')
        with gzip.open(g['path'],'rt') as f:trace=json.load(f)
        plans=[Planner(),Planner()]
        for row in trace['decisions']:
            obs=row['observation'];seat=row['seat']
            plans[seat].vector(player_view(obs))
            turn=obs['players'][seat]['turn_index']
            phase='early' if turn<=3 else 'middle' if turn<=7 else 'late'
            key=(g['seed'],phase)
            candidates=row.get('search',{}).get('candidates',[])
            valid=[c for c in candidates if c['samples']>=2 and c['mean'] is not None]
            if seat!=trace['seat'] or key in used or len(valid)<2:continue
            used.add(key);output.append((obs,deepcopy(plans),valid,phase,g['seed']))
            if len(output)>=count:return output
    return output


def evaluate(manifest, value, focal, opponent, out, count=12, samples=4):
    reports=[]
    for n,(obs,plans,candidates,phase,seed) in enumerate(cases(manifest,count)):
        roots=[c['action'] for c in candidates];predictions=[]
        search=Search(focal,opponent,obs['active_player'],'value_one',value=value)
        for rep in range(samples):
            scores,_=search.short_round(obs,roots,149900000+n*100+rep,plans,time.monotonic()+30)
            if scores is None:raise RuntimeError('diagnostic continuation timeout')
            predictions.append(scores)
        predicted=np.mean(predictions,axis=0);reference=np.array([c['mean'] for c in candidates])
        correct=confident=0
        for i in range(len(roots)):
            for j in range(i):
                noise=2*np.sqrt(sum((candidates[k]['variance'] or 0)/candidates[k]['samples'] for k in (i,j)))
                if abs(reference[i]-reference[j])>max(1.,noise):
                    confident+=1;correct+=int((reference[i]-reference[j])*(predicted[i]-predicted[j])>0)
        chosen=int(predicted.argmax())
        reports.append(dict(seed=seed,phase=phase,decision=obs['decision_index'],candidates=candidates,
            predicted=predicted.tolist(),empirical_regret=float(reference.max()-reference[chosen]),
            confident_pairs=confident,correct_pairs=correct))
    pairs=sum(r['confident_pairs'] for r in reports)
    result=dict(cases=reports,states=len(reports),confident_pairs=pairs,
                confident_pair_accuracy=sum(r['correct_pairs'] for r in reports)/pairs if pairs else None,
                empirical_regret_mean=float(np.mean([r['empirical_regret'] for r in reports])) if reports else None,
                reference='archived habitat rollout Q; noisy policy-specific estimates, not optimal Q')
    write_json(out,result);return result


def prediction_report(manifest,value,out):
    import torch
    from .value import remaining_value
    meta=read_json(manifest)
    with np.load(meta['path'],allow_pickle=False) as data:
        held=sorted(set(data['seed'].tolist()))[::5]
        selected=np.isin(data['seed'],held)
        x=data['x'][selected];target=data['mc'][selected]
    predicted=[]
    with torch.inference_mode():
        for start in range(0,len(x),256):
            predicted.extend(remaining_value(value.model,torch.as_tensor(x[start:start+256],dtype=torch.float64)).tolist())
    errors=150*(np.asarray(predicted)-target);turns=(x[:,-6]*20).round()+1
    owned=x[:,526:622].reshape(-1,32,3)
    progress=((owned[:,:,1]>0)&(owned[:,:,1]<1)).any(-1)
    groups=dict(all=np.ones(len(x),bool),early=turns<=3,middle=(turns>=4)&(turns<=7),late=turns>=8,
                unfinished_animals=progress)
    report={name:dict(rows=int(mask.sum()),rmse=float(np.sqrt(np.mean(errors[mask]**2))),
                     mae=float(np.mean(np.abs(errors[mask]))),bias=float(np.mean(errors[mask])))
            for name,mask in groups.items() if mask.any()}
    write_json(out,dict(groups=report,units='raw score',validation_seeds=held,
                       note='Single realized MC returns include future randomness; old model is only a historical reference.'))
