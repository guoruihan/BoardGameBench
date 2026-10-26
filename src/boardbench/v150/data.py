"""Audited train-only component labels; traces remain the source of truth."""
import argparse
import gzip
import json
from pathlib import Path
import time
import numpy as np
from boardbench.artifacts.store import digest,read_json,write_json
from boardbench.environments.harmonies_duo import player_view
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner
from boardbench.v140.encoding import encode_batch,action_map
from boardbench.v146.worker import audit_trace


def training_seed(seed):
    return any(base<=seed<base+900000 for base in (146000000,147000000,148000000,149000000,150000000))


def load_trace(receipt):
    path=receipt.get('path',receipt.get('trajectory'));sha=receipt.get('sha256',receipt.get('trajectory_sha256'))
    if digest(path)!=sha:raise ValueError('trace hash')
    with gzip.open(path,'rt') as f:trace=json.load(f)
    if trace['split']!='train' or not training_seed(trace['environment_seed']):raise ValueError('training only')
    if trace['format'] not in ('v146_duo_expert_trajectory','v147_duo_expert_trajectory',
            'v148_duo_search_trajectory','v149_value_planning_trajectory','v150_value_planning_trajectory'):
        raise ValueError('audited trace schema')
    audit_trace(trace)
    return trace


def components(trace):
    if trace['split']!='train' or not training_seed(trace['environment_seed']):raise ValueError('training only')
    # In the frozen rules natural termination is an end_turn; animal counters
    # cannot change in that operation. Total is independently replay-audited.
    if trace['decisions'][-1]['action']['type']!='end_turn':raise ValueError('unexpected terminal semantics')
    seat=trace['seat'];animal=trace['decisions'][-1]['observation']['score_breakdown'][seat]['animals']
    final=np.array([trace['scores'][seat]-animal,animal],float)
    rows=[]
    for row in trace['decisions']:
        if row['seat']!=seat:continue
        score=row['observation']['score_breakdown'][seat]
        y=(final-np.array([score['total']-score['animals'],score['animals']]))/150
        if not np.isclose(y.sum(),row['mc_return'],atol=1e-10):raise ValueError('component return conservation')
        rows.append(y)
    return rows


def build(source,out,native_artifact,deadline):
    out=Path(out)
    if (out/'manifest.json').exists():
        m=read_json(out/'manifest.json')
        if m['source_sha256']!=digest(source):raise ValueError('immutable data source')
        return m
    meta=read_json(source)
    if meta['split']!='train':raise ValueError('no development/confirmation data')
    receipts=meta.get('games',meta.get('rows'));native=NativeEncoder.from_artifact(native_artifact)
    xs=[];ys=[];seeds=[];games=[];behaviors=set();seen=set()
    for receipt in receipts:
        if time.time()>=deadline:raise TimeoutError('data deadline')
        trace=load_trace(receipt);key=(trace['environment_seed'],trace['seat'])
        if key in seen:raise ValueError('duplicate environment/seat');
        seen.add(key)
        behaviors.add((trace['method'],trace.get('value_manifest_sha256'),trace.get('rank_mix',0),trace['seconds'],trace.get('width',6),
                       trace['student_sha256'],trace['opponent_sha256']))
        plans=[Planner(),Planner()];labels=components(trace);start=len(xs)
        for row in trace['decisions']:
            obs=row['observation'];seat=row['seat'];vector=plans[seat].vector(player_view(obs))
            if seat!=trace['seat']:continue
            if row['action'] not in action_map(obs).values():raise ValueError('action encoding')
            xs.append(encode_batch([obs],[vector],native)[0]);seeds.append(trace['environment_seed'])
        ys.extend(labels)
        games.append(dict(path=receipt.get('path',receipt.get('trajectory')),sha256=receipt.get('sha256',receipt.get('trajectory_sha256')),
            seed=trace['environment_seed'],seat=trace['seat'],rows=len(xs)-start,opponent_sha256=trace['opponent_sha256']))
        write_json(out/'progress.json',dict(games=len(games),rows=len(xs)))
    if not xs or len(behaviors)!=1:raise ValueError('one behavior policy per component dataset required')
    path=out/'data.npz';tmp=out/'data.partial.npz'
    np.savez_compressed(tmp,x=np.asarray(xs,np.float32),y=np.asarray(ys,np.float64),seed=np.asarray(seeds,np.int64));tmp.replace(path)
    result=dict(format='v150_components',split='train',path=str(path.resolve()),sha256=digest(path),
        source_sha256=digest(source),games=games,rows=len(xs),environments=len(set(seeds)),
        target='remaining_own_terrain_animal_div150',behavior=list(next(iter(behaviors))))
    write_json(out/'manifest.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();j=read_json(a.job);build(j['source'],a.out,j['native_artifact'],j['deadline_unix'])
