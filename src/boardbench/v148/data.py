"""Train-only trajectories, full legal masks, visits and remaining own returns.

Legacy expert actions can bootstrap the network. They are marked hard labels,
never represented as measured MCTS visit distributions. Unsearched MCTS fallback
decisions have value labels but no policy label.
"""
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json, write_json, source_id
from boardbench.environments.harmonies_duo import player_view
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner
from boardbench.v140.encoding import ACTIONS, action_map, encode_batch
from boardbench.v146.worker import audit_trace


def training_seed(seed):
    return any(base <= seed < base+900000 for base in (146000000,147000000,148000000))


def reconstruct(trace, native=None):
    if (trace.get('split') != 'train' or not training_seed(trace['environment_seed'])
            or trace['format'] not in ('v146_duo_expert_trajectory','v147_duo_expert_trajectory',
                                       'v148_duo_search_trajectory')
            or not trace['method'].startswith('search_')):
        raise ValueError('only actual complete training search games')
    audit_trace(trace)
    planners, rows = [Planner(),Planner()], []
    for decision in trace['decisions']:
        obs, seat = decision['observation'], decision['seat']
        vector = planners[seat].vector(player_view(obs))
        if seat != trace['seat']:
            continue
        mapping = action_map(obs)
        action = next(i for i,a in mapping.items() if a == decision['action'])
        x = encode_batch([obs],[vector],native)[0]
        mask = np.asarray([i in mapping for i in range(ACTIONS)], bool)
        target, q, qmask = np.zeros(ACTIONS,np.float64), np.zeros(ACTIONS,np.float64), np.zeros(ACTIONS,bool)
        info = decision.get('search',{})
        mcts = info.get('algorithm') == 'public_outcome_mcts'
        seen = set()
        for candidate in info.get('candidates',[]):
            i = next(i for i,a in mapping.items() if a == candidate['action'])
            if i in seen: raise ValueError('duplicate candidate target')
            seen.add(i)
            n = candidate['samples']
            if not isinstance(n,int) or n < 0: raise ValueError('invalid sample count')
            if mcts:
                if candidate['visits'] != n: raise ValueError('visits/sample mismatch')
                target[i] = n
            if n >= 2 and candidate['mean'] is not None:
                q[i], qmask[i] = candidate['mean'], True
        if mcts:
            if target.sum() != info['root_visits'] or target.sum() != info['completed_rounds']:
                raise ValueError('root backup count mismatch')
            if target.sum(): target /= target.sum()
            label = 2 if target.sum() else 0
        else:
            target[action], label = 1., 1
        if (not mask[action] or np.any(target[~mask]) or np.any(qmask & ~mask)
                or not np.isfinite(x).all() or not np.isfinite(q).all()
                or not np.isfinite(decision['mc_return'])):
            raise ValueError('input/mask/target alignment')
        rows.append(dict(x=x,mask=mask,pi=target,label=label,action=action,mc=decision['mc_return'],
                         q=q,qmask=qmask,seed=trace['environment_seed'],seat=seat))
    return rows


def build(manifests, out, native_artifact, deadline):
    out = Path(out); out.mkdir(parents=True,exist_ok=False)
    native = NativeEncoder.from_artifact(native_artifact) if native_artifact else None
    rows, receipts, seen = [], [], set()
    for manifest in manifests:
        meta = read_json(manifest)
        if meta.get('split') != 'train': raise ValueError('training manifest only')
        for result in meta['games']:
            if time.time() >= deadline: raise TimeoutError('data deadline')
            key = (result['environment_seed'],result['seat'],result['method'],result['search_seed'],result['student_sha256'])
            if key in seen: continue
            if (result['split'] != 'train' or not training_seed(result['environment_seed'])
                    or not result['natural_terminal'] or result['status'] != 'completed'
                    or result['budget_excess_seconds'] > .1):
                raise ValueError('incomplete/over-budget expert game')
            if digest(result['trajectory']) != result['trajectory_sha256']: raise ValueError('trajectory hash')
            with gzip.open(result['trajectory'],'rt') as stream: trace = json.load(stream)
            for name in ('environment_seed','seat','method','search_seed','source_id','student_sha256','opponent_sha256'):
                if trace[name] != result[name]: raise ValueError('trace provenance '+name)
            if trace['scores'][trace['seat']] != result['score']: raise ValueError('score mismatch')
            data = reconstruct(trace,native); rows.extend(data); seen.add(key)
            receipts.append(dict(path=result['trajectory'],sha256=result['trajectory_sha256'],rows=len(data),
                                 seed=result['environment_seed'],seat=result['seat'],method=result['method'],
                                 opponent_sha256=result['opponent_sha256']))
            write_json(out/'progress.json',dict(games=len(receipts),rows=len(rows)))
    if not rows: raise ValueError('empty dataset')
    arrays = {k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    arrays['x'] = arrays['x'].astype(np.float32)  # Encoder's existing public ABI.
    arrays['label'] = arrays['label'].astype(np.int8)
    path=out/'data.npz'; tmp=out/'data.partial.npz'
    np.savez_compressed(tmp,**arrays); tmp.replace(path)
    result=dict(format='v148_search_training',split='train',path=str(path),sha256=digest(path),
                rows=len(rows),games=receipts,environments=len(set(arrays['seed'].tolist())),
                mcts_rows=int((arrays['label']==2).sum()),legacy_rows=int((arrays['label']==1).sum()),
                source_id=source_id(),manifests=[dict(path=str(p),sha256=digest(p)) for p in manifests],
                value_target='remaining_own_score_div150',policy_target='root_visits_or_explicit_legacy_action')
    write_json(out/'manifest.json',result); return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--manifests',nargs='+',required=True)
    p.add_argument('--out',required=True); p.add_argument('--native-manifest',required=True)
    p.add_argument('--deadline',type=float,required=True); a=p.parse_args()
    build(a.manifests,a.out,read_json(a.native_manifest),a.deadline)
