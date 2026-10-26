"""Reuse audited old value labels; select a single teacher without re-encoding."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from boardbench.artifacts.store import digest, read_json, write_json


def value_rows(trace,native=None):
    """Realized returns from this planner, with no invented expert action labels."""
    from boardbench.v146.worker import audit_trace
    from boardbench.v138.baseline import Planner
    from boardbench.environments.harmonies_duo import player_view
    from boardbench.v140.encoding import ACTIONS, action_map, encode_batch
    if (trace.get('split')!='train' or not 149000000<=trace['environment_seed']<149900000
            or trace.get('format')!='v149_value_planning_trajectory'
            or not trace['method'].startswith('search_value_')):
        raise ValueError('new planner training games only; never evaluation trajectories')
    audit_trace(trace)
    planners=[Planner(),Planner()];rows=[]
    for row in trace['decisions']:
        obs=row['observation'];seat=row['seat'];vector=planners[seat].vector(player_view(obs))
        if seat!=trace['seat']:continue
        mapping=action_map(obs);action=next(i for i,a in mapping.items() if a==row['action'])
        mask=np.array([i in mapping for i in range(ACTIONS)],bool)
        rows.append(dict(x=encode_batch([obs],[vector],native)[0],mask=mask,
            pi=np.zeros(ACTIONS),label=0,action=action,mc=row['mc_return'],q=np.zeros(ACTIONS),
            qmask=np.zeros(ACTIONS,bool),seed=trace['environment_seed'],seat=seat))
    return rows


def build_feedback(panel,out,native):
    """One behavior policy, new training environments, original complete receipts."""
    out=Path(out)
    if (out/'manifest.json').exists():
        existing=read_json(out/'manifest.json')
        if existing['parent_panel_sha256']!=digest(panel):raise ValueError('feedback panel changed')
        return existing
    meta=read_json(panel)
    if meta['split']!='train':raise ValueError('feedback cannot use development data')
    rows=[];games=[];seen=set();behaviors=set()
    for result in meta['rows']:
        key=(result['environment_seed'],result['seat'])
        if key in seen or not result['timing_gate'] or result['status']!='completed':
            raise ValueError('duplicate, incomplete or over-budget training game')
        if digest(result['trajectory'])!=result['trajectory_sha256']:raise ValueError('trace hash mismatch')
        with gzip.open(result['trajectory'],'rt') as f:trace=json.load(f)
        for k in ('split','method','environment_seed','seat','search_seed','source_id','student_sha256','opponent_sha256'):
            if result[k]!=trace[k]:raise ValueError('trace/result provenance mismatch: '+k)
        if result['score']!=trace['scores'][trace['seat']]:raise ValueError('score mismatch')
        behaviors.add((trace['method'],trace['seconds'],trace['value_manifest_sha256'],trace['width']))
        data=value_rows(trace,native);rows.extend(data);seen.add(key)
        games.append(dict(path=result['trajectory'],sha256=result['trajectory_sha256'],rows=len(data),
            seed=result['environment_seed'],seat=result['seat'],method=result['method'],
            opponent_sha256=result['opponent_sha256']))
    if not rows or len(behaviors)!=1:raise ValueError('nonempty single-policy feedback required')
    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    arrays['x']=arrays['x'].astype(np.float32);arrays['label']=arrays['label'].astype(np.int8)
    out.mkdir(parents=True,exist_ok=True);path=out/'data.npz';tmp=out/'data.partial.npz'
    np.savez_compressed(tmp,**arrays);tmp.replace(path)
    result=dict(format='v148_search_training',split='train',path=str(path.resolve()),sha256=digest(path),
        rows=len(rows),games=games,environments=len({g['seed'] for g in games}),mcts_rows=0,legacy_rows=0,
        parent_panel_sha256=digest(panel),value_target='remaining_own_score_div150',
        policy_target='none_value_only',behavior=list(next(iter(behaviors))))
    write_json(out/'manifest.json',result);return result


def select_teacher(manifest, out, method='search_habitat'):
    out=Path(out)
    if (out/'manifest.json').exists():
        existing=read_json(out/'manifest.json')
        if existing['parent_manifest_sha256']!=digest(manifest) or existing['selected_teacher']!=method:
            raise ValueError('immutable dataset configuration changed')
        return existing
    meta=read_json(manifest)
    if meta['format']!='v148_search_training' or meta['split']!='train' or digest(meta['path'])!=meta['sha256']:
        raise ValueError('audited immutable V148 training data required')
    ids=[];games=[];cursor=0
    for game in meta['games']:
        if game['method']==method:
            ids.extend(range(cursor,cursor+game['rows']));games.append(game)
        cursor+=game['rows']
    if cursor!=meta['rows'] or not ids:raise ValueError('dataset row ordering mismatch or no teacher')
    with np.load(meta['path'],allow_pickle=False) as f:
        arrays={k:f[k][ids] for k in f.files}
    expected=[(g['seed'],g['seat']) for g in games for _ in range(g['rows'])]
    if list(zip(arrays['seed'].tolist(),arrays['seat'].tolist()))!=expected:
        raise ValueError('dataset receipt/row alignment failed')
    out.mkdir(parents=True,exist_ok=True)
    path=out/'data.npz';tmp=out/'data.partial.npz'
    np.savez_compressed(tmp,**arrays);tmp.replace(path)
    result={**meta,'path':str(path.resolve()),'sha256':digest(path),'rows':len(ids),'games':games,
            'environments':len(set(arrays['seed'].tolist())),
            'mcts_rows':int((arrays['label']==2).sum()),'legacy_rows':int((arrays['label']==1).sum()),
            'parent_manifest_sha256':digest(manifest),'selected_teacher':method}
    write_json(out/'manifest.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();select_teacher(a.manifest,a.out)
