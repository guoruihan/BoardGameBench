"""Public sibling branches with common futures and explicit habitat continuation.

These labels belong to the separate ranking head, not on-policy MC components.
Each row group is one root, one sampled public future, all retained candidates.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import time
import numpy as np
import torch
from boardbench.artifacts.store import digest,read_json,write_json,source_id
from boardbench.environments.harmonies_duo import FastHarmoniesDuo,player_view
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner
from boardbench.v140.encoding import encode_batch
from boardbench.v145.policy import load_policy
from boardbench.v146.policy import ModelPlayer,Habitat,mixed_roots
from .data import load_trace


def select_cases(manifest,limit=256):
    meta=read_json(manifest)
    if meta['split']!='train':raise ValueError('training branch roots only')
    buckets=[[] for _ in range(3)]
    for game in meta['games']:
        trace=load_trace(game);seen=set()
        for i,row in enumerate(trace['decisions']):
            obs=row['observation'];turn=obs['players'][row['seat']]['completed_turns']
            phase=0 if turn<3 else 1 if turn<7 else 2
            if row['seat']!=trace['seat'] or phase in seen or len(obs['legal_actions'])<3:continue
            seen.add(phase);buckets[phase].append(dict(path=game['path'],sha256=game['sha256'],index=i,
                seed=trace['environment_seed'],seat=trace['seat'],phase=phase))
    # Interleave phase buckets; do not let the first few games define the whole batch.
    return [v for row in zip(*buckets) for v in row][:limit]


def label(job,out):
    torch.set_num_threads(1);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if job['source_id']!=source_id():raise ValueError('frozen labeler source')
    native=NativeEncoder.from_artifact(job['native_artifact'])
    focal=ModelPlayer(load_policy(job['student']),native);opponent=ModelPlayer(load_policy(job['opponent']),native)
    for k in ('student','opponent'):
        if digest(job[k])!=job[k+'_sha256']:raise ValueError('immutable label policy')
    habitat=Habitat();xs=[];ys=[];groups=[];seeds=[];receipts=[]
    continuation=job.get('continuation','habitat')
    if continuation not in ('habitat','actor'):raise ValueError('explicit continuation policy')
    for ci,case in enumerate(job['cases']):
        trace=load_trace(case);plans=[Planner(),Planner()]
        for row in trace['decisions'][:case['index']+1]:plans[row['seat']].vector(player_view(row['observation']))
        obs=trace['decisions'][case['index']]['observation'];seat=case['seat']
        if obs['active_player']!=seat:raise ValueError('root perspective')
        ranking=focal.rank_batch([obs],[plans[seat]])[0][0];roots=mixed_roots(obs,ranking,habitat,job['width'])
        for rep in range(job['samples']):
            future=150900000+case['seed']*100+case['index']*7+rep
            group_x=[];group_y=[]
            for action in roots:
                game=FastHarmoniesDuo.from_observation(obs,future);memory=deepcopy(plans)
                state=game.step(action).observation;turn=obs['players'][seat]['completed_turns'];leaf=None
                for _ in range(768):
                    if time.time()>=job['deadline_unix']:raise TimeoutError('label deadline, no partial group')
                    if not state['terminated'] and state['active_player']==seat and state['players'][seat]['completed_turns']>turn and leaf is None:
                        leaf=encode_batch([state],[memory[seat].vector(player_view(state))],native)[0]
                    if state['terminated']:break
                    active=state['active_player']
                    if active!=seat:action=opponent.decide(state,memory[active])
                    elif leaf is not None and continuation=='actor':action=focal.decide(state,memory[active])
                    else:action=habitat.decide(state)
                    state=game.step(action).observation
                else:raise RuntimeError('no natural branch terminal')
                if leaf is None:break  # terminal before next-own leaf: no learned-value ranking needed
                group_x.append(leaf);group_y.append(state['scores'][seat]/150)
            if len(group_x)!=len(roots):continue
            gid=len(receipts);xs.extend(group_x);ys.extend(group_y);groups.extend([gid]*len(roots));seeds.extend([case['seed']]*len(roots))
            receipts.append(dict(case=case,rep=rep,public_future_seed=future,actions=roots,final_scores=(150*np.array(group_y)).tolist()))
        write_json(out/'progress.json',dict(cases=ci+1,groups=len(receipts),rows=len(xs)))
    from boardbench.v140.encoding import FEATURES
    path=out/'pairs.npz';np.savez_compressed(path,x=np.asarray(xs,np.float32).reshape(-1,FEATURES),q=np.asarray(ys,np.float64),
        group=np.asarray(groups,np.int64),seed=np.asarray(seeds,np.int64))
    result=dict(format='v150_habitat_pairs',split='train',path=str(path),sha256=digest(path),
        rows=len(xs),groups=len(receipts),receipts=receipts,opponent_sha256=job['opponent_sha256'],
        continuation=continuation+'_focal_frozen_opponent',source_id=source_id())
    write_json(out/'result.json',result)


def merge(results,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);data={k:[] for k in ('x','q','group','seed')};offset=0
    policies={r['continuation'] for r in results}
    if len(policies)!=1:raise ValueError('do not pool different continuation labels')
    for result in results:
        if result['format']!='v150_habitat_pairs' or digest(result['path'])!=result['sha256']:raise ValueError('pair provenance')
        with np.load(result['path'],allow_pickle=False) as f:
            for k in data:data[k].append(f[k]+offset if k=='group' else f[k])
        offset+=result['groups']
    if offset<2:raise ValueError('too few complete candidate groups')
    path=out/'pairs.npz';np.savez_compressed(path,**{k:np.concatenate(v) for k,v in data.items()})
    result=dict(format='v150_habitat_pairs',split='train',path=str(path.resolve()),sha256=digest(path),groups=offset,
        shards=[dict(path=r['path'],sha256=r['sha256']) for r in results],continuation=next(iter(policies)))
    write_json(out/'manifest.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    label(read_json(a.job),a.out)
