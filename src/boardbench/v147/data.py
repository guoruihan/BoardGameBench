"""Replay actual expert actions into native duo inputs; never consume dev data."""
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np

from boardbench.artifacts.store import digest,read_json,write_json,source_id
from boardbench.environments.harmonies_duo import player_view
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner
from boardbench.v140.encoding import ACTIONS,action_map,encode_batch
from boardbench.v146.worker import audit_trace


def training_seed(seed):
    return 146000000<=seed<146900000 or 147000000<=seed<147900000


def reconstruct(trace,native=None):
    if (trace.get('split')!='train' or not training_seed(trace['environment_seed'])
            or trace['format'] not in ('v146_duo_expert_trajectory','v147_duo_expert_trajectory')
            or not trace['method'].startswith('search_')):
        raise ValueError('only actual complete V146/V147 TRAIN expert games')
    audit_trace(trace)
    planners=[Planner(),Planner()];rows=[]
    for decision in trace['decisions']:
        obs=decision['observation'];seat=decision['seat']
        vector=planners[seat].vector(player_view(obs))
        if seat!=trace['seat']:continue  # Never label the weaker opponent as expert.
        mapping=action_map(obs);action=next(i for i,a in mapping.items() if a==decision['action'])
        x=encode_batch([obs],[vector],native)[0]
        mask=np.array([i in mapping for i in range(ACTIONS)],bool)
        q=np.zeros(ACTIONS,np.float32);qmask=np.zeros(ACTIONS,bool)
        for candidate in decision.get('search',{}).get('candidates',[]):
            if candidate['samples']<2 or candidate['mean'] is None:continue
            i=next(i for i,a in mapping.items() if a==candidate['action'])
            if not np.isfinite(candidate['mean']):raise ValueError('nonfinite candidate Q')
            q[i]=candidate['mean'];qmask[i]=True
        if not mask[action] or np.any(qmask & ~mask) or not np.isfinite(x).all():raise ValueError('input/mask alignment')
        rows.append((x,mask,action,decision['mc_return'],q,qmask,trace['environment_seed'],seat))
    return rows


def build(manifests,out,native_artifact,deadline):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.time();native=NativeEncoder.from_artifact(native_artifact)
    rows=[];receipts=[];seen=set()
    for manifest in manifests:
        meta=read_json(manifest)
        if meta.get('split')!='train':raise ValueError('training manifest only')
        for result in meta['games']:
            if time.time()>=deadline:raise TimeoutError('data original deadline')
            key=(result['environment_seed'],result['seat'],result['method'],result['search_seed'])
            if key in seen:continue
            if (result['split']!='train' or not training_seed(result['environment_seed']) or
                    not result['natural_terminal'] or result['status']!='completed' or result['budget_excess_seconds']>.1):
                raise ValueError('invalid expert result')
            path=Path(result['trajectory'])
            if digest(path)!=result['trajectory_sha256']:raise ValueError('trajectory hash mismatch')
            with gzip.open(path,'rt') as stream:trace=json.load(stream)
            for name in ('environment_seed','seat','method','search_seed','source_id','student_sha256','opponent_sha256'):
                if trace[name]!=result[name]:raise ValueError('trace/result provenance mismatch '+name)
            if trace['scores'][trace['seat']]!=result['score']:raise ValueError('trace/result score mismatch')
            data=reconstruct(trace,native);rows.extend(data);seen.add(key)
            receipts.append(dict(trajectory=str(path),sha256=result['trajectory_sha256'],rows=len(data),score=result['score'],
                seed=result['environment_seed'],seat=result['seat'],method=result['method']))
            write_json(out/'progress.json',dict(games=len(receipts),rows=len(rows),updated_unix=time.time()))
    if not rows:raise ValueError('empty expert dataset')
    columns=list(zip(*rows));arrays=dict(x=np.stack(columns[0]),mask=np.stack(columns[1]),
        action=np.asarray(columns[2],np.int64),mc=np.asarray(columns[3],np.float64),q=np.stack(columns[4]),
        qmask=np.stack(columns[5]),seed=np.asarray(columns[6],np.int64),seat=np.asarray(columns[7],np.int8))
    path=out/'expert.npz';tmp=out/'expert.partial.npz';np.savez_compressed(tmp,**arrays);tmp.replace(path)
    result=dict(format='v147_replayed_duo_training',split='train',path=str(path),sha256=digest(path),source_id=source_id(),
        games=receipts,rows=len(rows),environments=len(set(arrays['seed'].tolist())),
        manifests=[dict(path=str(p),sha256=digest(p)) for p in manifests],wall_seconds=time.time()-start,
        note='Expert action/MC targets are off-policy auxiliaries, never PPO behavior samples. All focal episodes retained.')
    write_json(out/'manifest.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifests',nargs='+',required=True);p.add_argument('--out',required=True)
    p.add_argument('--native-manifest',required=True);p.add_argument('--deadline',required=True,type=float)
    a=p.parse_args();build(a.manifests,a.out,read_json(a.native_manifest),a.deadline)
