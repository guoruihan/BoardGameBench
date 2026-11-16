#!/usr/bin/env python3
"""Summarize trusted pilot results, preserving failed and incomplete matches."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics


def summarize(root, *, pair_limit=None):
    path = root/'results.jsonl'
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if pair_limit is not None:
        if type(pair_limit) is not int or pair_limit<1:raise ValueError('pair limit must be positive')
        records=[r for r in records if int(r['match_id'].split('-s')[1].split('-')[0])<pair_limit]
    rows = []
    for record in records:
        result=record['result']; gt=result.get('gt',{}); outcomes=gt.get('outcomes',[])
        last=outcomes[-1] if outcomes else {}; natural=result.get('natural',last.get('natural',False))
        usage=result.get('model',{}).get('usage',{}); compute=result.get('compute',{})
        sources=gt.get('sources',{})
        actions=sum(sources.get(k,0) for k in ('agent','program','forced','default_random'))
        rows.append(dict(match_id=record['match_id'],profile=record['profile'],control=record['control'],
                         seed=record['seed'],seat=record['seat'],status=result['status'],natural=natural,
                         own_score=last.get('own_score') if natural else None,
                         opponent_score=last.get('opponent_score') if natural else None,
                         win_points=result.get('win_points',last.get('win_points')),calls=usage.get('calls',0),
                         input_tokens=usage.get('input_tokens',0),output_tokens=usage.get('output_tokens',0),
                         cpu_seconds=compute.get('cpu_seconds',0),wall_seconds=compute.get('active_wall_seconds',0),
                         default_actions=sources.get('default_random',0),player_actions=actions,
                         first_binding_limit=result.get('model',{}).get('first_binding_limit'),
                         submission_sha256=record['submission_sha256']))
    groups=defaultdict(list)
    for row in rows: groups[('control' if row['control'] else 'developed',row['profile'])].append(row)
    aggregate={}
    for (arm,profile),values in groups.items():
        finished=[r for r in values if r['natural']]
        scored=[r for r in values if r['win_points'] is not None]
        aggregate[f'{arm}-{profile}']=dict(recorded=len(values),natural=len(finished),
            win_points_mean=statistics.mean(r['win_points'] for r in scored) if scored else None,
            win_denominator=len(scored),own_score_mean=statistics.mean(r['own_score'] for r in finished) if finished else None,
            opponent_score_mean=statistics.mean(r['opponent_score'] for r in finished) if finished else None,
            default_action_fraction=sum(r['default_actions'] for r in values)/max(1,sum(r['player_actions'] for r in values)),
            calls=sum(r['calls'] for r in values),input_tokens=sum(r['input_tokens'] for r in values),
            output_tokens=sum(r['output_tokens'] for r in values),cpu_seconds=sum(r['cpu_seconds'] for r in values),
            active_wall_seconds=sum(r['wall_seconds'] for r in values))
    # Paired differences operate on seat-averaged seeds, never treat seats as independent.
    paired={}
    indexed={(r['control'],r['profile'],r['seed'],r['seat']):r for r in rows}
    seeds=sorted({r['seed'] for r in rows})
    for other in ('L10','L40','control-L20'):
        diffs=[]
        for seed in seeds:
            pairs=[]
            for seat in (0,1):
                a=indexed.get((False,'L20',seed,seat))
                b=indexed.get((other=='control-L20','L20' if other=='control-L20' else other,seed,seat))
                if a and b and a['natural'] and b['natural']:
                    pairs.append(a['own_score']-b['own_score'])
            if len(pairs)==2: diffs.append(statistics.mean(pairs))
        paired['developed-L20_minus_'+other]={'complete_seed_pairs':len(diffs),'score_difference':statistics.mean(diffs) if diffs else None}
    return rows,{'groups':aggregate,'paired':paired,'interpretation':'calibration only; seed/seat matched when present; failures retained; input preflight approximate'}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('root',type=Path); args=parser.parse_args()
    rows,report=summarize(args.root)
    (args.root/'summary.json').write_text(json.dumps(report,indent=2))
    if rows:
        with (args.root/'results.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
