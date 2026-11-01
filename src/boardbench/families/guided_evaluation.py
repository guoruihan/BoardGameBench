"""Separate CPU evaluator for immutable rich-policy snapshots; never blocks PPO."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from .play import summarize, write_json
from .pool_evaluator import bounded_jobs, paired_interval
from .reference_search import guarded_spec
from .reward_signals import numeric_leaves
from .train import match_worker


def summarize_rows(rows, scheduled):
    summary=summarize(rows,[r['learner_seat'] for r in rows])
    complete=[r for r in rows if r['status']=='complete'];groups={}
    for r in complete:groups.setdefault(r['seed'],[]).append(.5+.5*r['rewards'][r['learner_seat']])
    summary.update(scheduled=scheduled,not_run=scheduled-len(rows),
        all_scheduled_win_rate=summary['win_points']/scheduled,
        interval=paired_interval([sum(v)/2 for v in groups.values() if len(v)==2]),
        all_complete=len(complete)==scheduled)
    for who in ('own','opponent'):
        scored=[];components={}
        for r in complete:
            seat=r['learner_seat'] if who=='own' else 1-r['learner_seat']
            score=r['outcome']['scores'][seat]
            if score is not None:scored.append(score)
            for k,v in numeric_leaves(r['score_breakdown'][seat]).items():components.setdefault(k,[]).append(v)
        summary[who+'_score']=float(np.mean(scored)) if scored else None
        summary[who+'_components']={k:float(np.mean(v)) for k,v in components.items()}
    summary['policy_seconds']=float(np.mean([r['policy_seconds'][r['learner_seat']] for r in complete])) if complete else None
    return summary


def evaluate_many(root, requests, workers, deadline, seeds=8, seed_start=173020000, split='quick'):
    jobs=[];labels=[]
    for cfg,candidate in requests:
        info=json.loads(Path(cfg['dataset']).read_text());opponent=info['base_spec']
        spec=guarded_spec(cfg['game'],dict(kind='guided_rich',path=candidate['path'],sha256=candidate['sha256']))
        for k in range(seeds):
            for seat in (0,1):
                specs=[opponent,opponent];specs[seat]=spec
                jobs.append(dict(game_id=cfg['game'],specs=specs,seed=seed_start+k,first_player=0,
                                 max_seconds=180.,measure_policy_time=True))
                labels.append((cfg,candidate,seat))
    grouped={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        results=bounded_jobs(pool,match_worker,jobs,deadline,workers)
    for row,(cfg,candidate,seat) in zip(results,labels):
        row['learner_seat']=seat
        key=(cfg['game'],cfg['arm'],candidate['name'])
        grouped.setdefault(key,[]).append(row)
    summaries=[]
    for cfg,candidate in requests:
        key=(cfg['game'],cfg['arm'],candidate['name']);rows=grouped.get(key,[])
        payload=dict(game=cfg['game'],arm=cfg['arm'],candidate=candidate,split=split,
            seed_start=seed_start,summary=summarize_rows(rows,seeds*2),rows=rows,observed_unix=time.time())
        dest=root/'evaluations'/split/cfg['game']/cfg['arm']/(candidate['name']+'.json')
        write_json(dest,payload);summaries.append(payload)
        print(json.dumps(dict(game=cfg['game'],arm=cfg['arm'],name=candidate['name'],
                             summary=payload['summary'])),flush=True)
    return summaries


def collect(root):
    requests=[]
    for path in sorted((root/'runs').glob('*/*/config.json')):
        cfg=json.loads(path.read_text());manifest=path.parent/'candidates.json'
        if not manifest.exists():continue
        candidates=json.loads(manifest.read_text())
        selected=[x for x in candidates if x['name'] in ('initial','bc_final')]
        if candidates:selected.append(candidates[-1])
        seen=set()
        for item in selected:
            if item['name'] in seen:continue
            seen.add(item['name'])
            dest=root/'evaluations'/'quick'/cfg['game']/cfg['arm']/(item['name']+'.json')
            if not dest.exists():requests.append((cfg,item))
    return requests


def best_candidates(root):
    best={}
    for p in (root/'evaluations'/'quick').glob('*/*/*.json'):
        item=json.loads(p.read_text());s=item['summary']
        if not s['all_complete']:continue
        # Official wins first, genuine score difference second; no hidden labels.
        score=(s['all_scheduled_win_rate'],(s['own_score'] or 0)-(s['opponent_score'] or 0))
        game=item['game']
        if game not in best or score>best[game][0]:best[game]=(score,item)
    return {g:x[1] for g,x in best.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--workers',type=int,default=24);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--confirm-at',type=float,required=True);p.add_argument('--confirm-seeds',type=int,default=64)
    p.add_argument('--poll-seconds',type=float,default=30.)
    a=p.parse_args();a.root=a.root.resolve()
    while time.time()<a.confirm_at:
        requests=collect(a.root)
        if requests:evaluate_many(a.root,requests,a.workers,a.confirm_at,seeds=8)
        best=best_candidates(a.root);write_json(a.root/'best_development.json',best)
        write_json(a.root/'evaluation_status.json',dict(phase='quick',observed_unix=time.time(),games=len(best)))
        time.sleep(min(a.poll_seconds,max(0.,a.confirm_at-time.time())))
    best=best_candidates(a.root);requests=[]
    write_json(a.root/'confirmation_selection.json',best)
    for game,item in best.items():
        cfg=json.loads((a.root/'runs'/game/item['arm']/'config.json').read_text())
        requests.append((cfg,item['candidate']))
    if requests:
        evaluate_many(a.root,requests,a.workers,a.deadline_unix,seeds=a.confirm_seeds,
                      seed_start=173030000,split='confirmation')
    write_json(a.root/'evaluation_status.json',dict(phase='complete',observed_unix=time.time(),games=len(best)))


if __name__=='__main__':main()
