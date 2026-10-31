"""Resumable per-game development comparison: retained RL versus terminal search."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback

from .play import make_policy,policy_seed,run_match,write_json
from .reference_search import TerminalReference,guarded_spec
from .registry import game_class


def spec_from_bundle(root,entry,name):
    path=root/entry['bots'][name]['path'];spec=json.loads(path.read_text())
    if 'path' in spec:spec['path']=str((path.parent/spec['path']).resolve())
    return spec


def select_rl(game,bundle,current,output,seeds=8):
    """Choose on a separate development split; all selection games retained."""
    path=output/'selection.json'
    if path.exists():return json.loads(path.read_text())
    manifest=json.loads((bundle/'manifest.json').read_text());entry=manifest['games'][game]
    specs={name:spec_from_bundle(bundle,entry,name) for name in entry['bots']}
    names=[n for n in entry['recommended'] if specs[n]['kind'] in ('neural','harmonies_rl')]
    names += [n for n in specs if specs[n]['kind'] in ('neural','harmonies_rl') and n not in names]
    if not names:raise ValueError('No retained RL reference')
    old_name=names[0];old=guarded_spec(game,specs[old_name])
    # best.json is selected by the previous pilot's own development gate. Do
    # not promote latest weights solely because they are newer.
    candidates={}
    for p in sorted((current/'runs'/game).glob('*/best.json')):
        spec=guarded_spec(game,json.loads(p.read_text()));key=spec['sha256']
        candidates.setdefault(key,(p.parent.name,spec))
    all_results=[];chosen=old;chosen_name='retained/'+old_name;best_rate=.5
    for digest,(name,spec) in candidates.items():
        if digest==old.get('sha256'):continue
        results=[]
        for k in range(seeds):
            for seat in (0,1):
                pair=[old,old];pair[seat]=spec
                r=run_match(game,pair,seed=172010000+k,first_player=0,max_seconds=90,record=False)
                r['candidate_seat']=seat;results.append(r)
        complete=all(r['status']=='complete' for r in results)
        rate=sum(.5+.5*r['rewards'][r['candidate_seat']] for r in results if r['status']=='complete')/len(results)
        all_results.append(dict(candidate=name,spec=spec,complete=complete,win_rate=rate,games=results))
        if complete and rate>best_rate:chosen=spec;chosen_name='common/'+name;best_rate=rate
    weights=next((s['weights'] for n,s in specs.items() if n.startswith('cem') and 'weights' in s),None)
    result=dict(game=game,chosen_name=chosen_name,rl_spec=chosen,retained_name=old_name,retained_spec=old,
        selection_seeds=list(range(172010000,172010000+seeds)),selection=all_results,
        heuristic_weights=weights,selection_note='Small development gate, not proof of globally best RL')
    write_json(path,result);return result


def match(game,selection,seed,search_seat,budget,output,objective='win',max_rounds=None,continuation='portfolio'):
    cls=game_class(game);env=cls();env.reset(seed,first_player=0)
    rl=make_policy(cls,selection['rl_spec'],policy_seed(seed,1-search_seat))
    search=TerminalReference(cls,selection['rl_spec'],seat=search_seat,seconds=budget,
        seed=policy_seed(seed,search_seat)^172,weights=selection['heuristic_weights'],
        max_rounds=max_rounds,objective=objective,continuation=continuation)
    began=time.monotonic();cpu=time.process_time();count=0;trace=[];status='incomplete';rl_seconds=0.;last_progress=began
    while not env.outcome()['terminated']:
        if count>=10000 or time.monotonic()-began>budget+180:
            status='truncated';break
        seat=env.current_player;obs=env.observe(seat);legal=env.legal_actions(seat)
        if not legal:status='no_legal_actions';break
        t=time.monotonic()
        if seat==search_seat:
            action=search.decide(env);detail=search.last
        else:
            action=rl.decide(obs,legal);detail=None;rl_seconds+=time.monotonic()-t
        trace.append(dict(player=seat,observation=obs,legal_actions=legal,action=action,search=detail))
        env.step(action);count+=1
        if time.monotonic()-last_progress>=10:
            write_json(output.with_suffix('.progress.json'),dict(game=game,seed=seed,search_seat=search_seat,
                budget=budget,decisions=count,search_seconds=search.spent,simulation_steps=search.total_steps,
                completed_rounds=search.total_rounds,fallback_decisions=search.fallbacks,observed_unix=time.time()))
            last_progress=time.monotonic()
    out=env.outcome()
    if out['terminated']:status='complete'
    result=dict(game=game,config=cls.config_id,rules_version=cls.rules_version,seed=seed,
        search_seat=search_seat,first_player=0,budget=budget,objective=objective,status=status,
        search_config=dict(width=search.width,max_rounds=max_rounds,max_simulation_steps=search.max_simulation_steps,
                           continuation=continuation,
                           continuations=['domain_heuristic','progress_or_habitat','retained_rl']),
        outcome=out,score_breakdown=env.score_breakdown(),decisions=count,
        rl=selection['chosen_name'],rl_sha256=selection['rl_spec'].get('sha256'),
        search_seconds=search.spent,rl_seconds=rl_seconds,wall_seconds=time.monotonic()-began,
        cpu_seconds=time.process_time()-cpu,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        search_decisions=search.decisions,simulation_steps=search.total_steps,
        complete_candidate_rounds=search.total_rounds,fallback_decisions=search.fallbacks,
        max_search_decision_seconds=max([r['search']['decision_seconds'] for r in trace if r['search']]+[0.]))
    # Contains player observations only; no engine snapshots or hidden futures.
    trace_path=output.with_suffix('.trajectory.json.gz');temporary=trace_path.with_suffix('.part')
    with gzip.open(temporary,'wt',encoding='utf-8') as stream:
        json.dump(dict(result=result,trajectory=trace),stream,ensure_ascii=False,allow_nan=False)
    temporary.replace(trace_path)
    result['trajectory']=trace_path.name
    result['trajectory_sha256']=hashlib.sha256(trace_path.read_bytes()).hexdigest()
    write_json(output,result);return result


def report(root):
    groups=defaultdict(list)
    expected_counts=defaultdict(int)
    if (root/'schedule.json').exists():
        for entry in json.loads((root/'schedule.json').read_text())['jobs']:
            key=(entry['game'],float(entry['budget']));expected_counts[key]+=1;groups[key]
    for path in root.glob('games/*/matches/*.json'):
        if path.name.endswith(('.error.json','.progress.json')):continue
        row=json.loads(path.read_text());groups[(row['game'],row['budget'])].append(row)
    rows=[]
    for (game,budget),games in sorted(groups.items()):
        continuations={r.get('search_config',{}).get('continuation','portfolio') for r in games}
        if len(continuations)>1:raise ValueError('Use separate experiment roots for different continuation comparisons')
        complete=[r for r in games if r['status']=='complete'];points=own=other=0.
        for r in complete:
            p=r['search_seat'];out=r['outcome']
            points+=.5 if len(out['winners'])!=1 else float(p in out['winners'])
            if out['scores'][p] is not None:
                own+=out['scores'][p];other+=out['scores'][1-p]
        n=len(complete)
        # Cluster bootstrap by environment: both seats stay paired. Four seeds
        # give only a descriptive interval, not a reliable final ranking.
        import numpy as np
        by_seed=defaultdict(list)
        for r in complete:
            p=r['search_seat'];scores=r['outcome']['scores'];winners=r['outcome']['winners']
            by_seed[r['seed']].append((.5 if len(winners)!=1 else float(p in winners),
                scores[p]-scores[1-p] if scores[p] is not None else 0.))
        clusters=[np.mean(v,axis=0) for v in by_seed.values() if len(v)==2]
        intervals=None
        if clusters:
            z=np.asarray(clusters);rng=np.random.default_rng(172)
            boot=z[rng.integers(len(z),size=(4000,len(z)))].mean(1)
            intervals=np.quantile(boot,[.025,.975],axis=0).T.tolist()
        expected=expected_counts.get((game,budget),len(games))
        statuses={s:sum(r['status']==s for r in games) for s in sorted({r['status'] for r in games})}
        statuses['no_result']=expected-len(games)
        rows.append(dict(game=game,budget=budget,scheduled=expected,finished=len(games),complete=n,
            continuation=next(iter(continuations)) if continuations else None,
            statuses=statuses,
            paired_environment_groups=len(clusters),search_win_interval=intervals[0] if intervals else None,
            score_gap_interval=intervals[1] if intervals and complete[0]['outcome']['scores'][0] is not None else None,
            search_win_rate=points/expected,rl_win_rate=(n-points)/expected,
            search_score=own/n if n and complete[0]['outcome']['scores'][0] is not None else None,
            rl_score=other/n if n and complete[0]['outcome']['scores'][0] is not None else None,
            search_seconds=sum(r['search_seconds'] for r in games)/len(games) if games else None,
            rl_seconds=sum(r['rl_seconds'] for r in games)/len(games) if games else None,
            cpu_hours=sum(r['cpu_seconds'] for r in games)/3600,
            simulation_steps=sum(r['simulation_steps'] for r in games),
            fallback_fraction=sum(r['fallback_decisions'] for r in games)/max(1,sum(r['search_decisions'] for r in games)),
            rl=games[0]['rl'] if games else None))
    write_json(root/'report.json',dict(results=rows,observed_unix=time.time(),
        error_files=[str(p.relative_to(root)) for p in root.glob('games/*/matches/*.error.json')]))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--game')
    p.add_argument('--bundle',type=Path,default=Path('outputs/v160/frozen_pool_v160'))
    p.add_argument('--current',type=Path,default=Path('outputs/v171/reward_pilot_20261101'))
    p.add_argument('--budgets',type=float,nargs='+',default=[30,180,900])
    p.add_argument('--seeds',type=int,default=4);p.add_argument('--selection-seeds',type=int,default=8)
    p.add_argument('--seed-index',type=int);p.add_argument('--seat',type=int)
    p.add_argument('--select-only',action='store_true');p.add_argument('--report',action='store_true')
    p.add_argument('--max-rounds',type=int);p.add_argument('--objective',default='win')
    p.add_argument('--continuation',choices=['portfolio','rl','expert'],default='portfolio')
    a=p.parse_args();a.root=a.root.resolve()
    if a.report:
        print(json.dumps(report(a.root),indent=2));return
    output=a.root/'games'/a.game;output.mkdir(parents=True,exist_ok=True)
    selection=select_rl(a.game,a.bundle,a.current,output,a.selection_seeds)
    if a.select_only:return
    for budget in a.budgets:
        for k in ([a.seed_index] if a.seed_index is not None else range(a.seeds)):
            for seat in ([a.seat] if a.seat is not None else (0,1)):
                suffix='' if a.continuation=='portfolio' else '_'+a.continuation
                path=output/'matches'/f'b{budget:g}_s{k}_p{seat}_{a.objective}{suffix}.json'
                if path.exists():continue
                path.parent.mkdir(parents=True,exist_ok=True)
                try:
                    row=match(a.game,selection,172020000+k,seat,budget,path,a.objective,a.max_rounds,a.continuation)
                    print(json.dumps({k:v for k,v in row.items() if k!='score_breakdown'}),flush=True)
                except Exception:
                    write_json(path.with_suffix('.error.json'),dict(game=a.game,budget=budget,seed_index=k,seat=seat,error=traceback.format_exc()))
                    raise

if __name__=='__main__':main()
