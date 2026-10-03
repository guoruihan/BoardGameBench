"""Recompute V1.1 results from unchanged Runner shards and training artifacts."""
import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from boardbench.artifacts.store import read_json, write_json


def interval(differences):
    mean=statistics.mean(differences)
    half=1.96*statistics.stdev(differences)/(len(differences)**.5) if len(differences)>1 else 0.
    return {'mean_difference':mean,'normal_95_interval':[mean-half,mean+half],
            'n_environment_pairs':len(differences),'scope':'conditional on this fixed training artifact, not training-seed uncertainty'}


def build(root):
    plan=read_json(root/'experiment_plan/plan.json')
    report={'version':'v1.1','games':{},'completed_games':0,'illegal_actions':0,
            'failures':0,'truncations':0,'training_seeds':plan['training_seeds'],
            'uncertainty':'training-repeat sample SD is separate from paired environment-seed intervals',
            'latency_boundary':'CPU two threads per worker, concurrency recorded per policy; solver and episode wall exclude different setup costs; not cross-version hardware-normalized speedup'}
    for game in plan['games']:
        base=root/game
        selection=read_json(base/'test/frozen_selection.json')
        results=read_json(base/'test/results.json')
        assert set(results)=={c['id'] for c in selection['selected']}
        groups=defaultdict(list)
        scores={}
        for candidate in selection['selected']:
            cid=candidate['id']
            episodes=[]
            for relative in results[cid]['raw_runs']:
                from json import loads
                episodes.extend(loads(line) for line in (base/'test'/relative/'episodes.jsonl').read_text().splitlines())
            assert [r['seed'] for r in episodes]==selection['splits']['test']
            scores[cid]={r['seed']:r['score'] for r in episodes if r['status']=='completed'}
            assert len(scores[cid])==plan['test_count'],cid
            assert abs(statistics.mean(scores[cid].values())-results[cid]['score_mean'])<1e-10
            for source,target in [('completed','completed_games'),('illegal_actions','illegal_actions'),
                                  ('failed','failures'),('truncated','truncations')]:
                report[target]+=results[cid][source]
            key=candidate['method']+('_untrained' if candidate.get('untrained') else '')
            groups[key].append({'policy':cid,'training_seed':candidate.get('training',{}).get('training_seed'),
                                **results[cid]})
        aggregate={}
        for method,items in groups.items():
            means=[x['score_mean'] for x in items]
            aggregate[method]={'mean_of_policy_means':statistics.mean(means),
                               'training_repeat_sample_sd':statistics.stdev(means) if len(means)>1 else None,
                               'policy_means':means,'policies':items}
            if method in ('rl','imitation'):
                assert sorted(x['training_seed'] for x in items)==plan['training_seeds']
        paired={}
        for candidate in selection['selected']:
            if candidate['method'] not in ('rl','imitation') or candidate['untrained']:
                continue
            control=next(c for c in selection['selected'] if c.get('untrained')
                         and c['origin']['run_id']==candidate['origin']['run_id'])
            trained=scores[candidate['id']]
            paired[candidate['id']]={
                'vs_matched_untrained':interval([trained[s]-scores[control['id']][s] for s in trained]),
                'vs_heuristic':interval([trained[s]-scores['heuristic'][s] for s in trained])}
        costs={}
        for algorithm in ('ppo','imitation'):
            rows=[]
            for seed in plan['training_seeds']:
                path=base/f'{algorithm}_{seed}'/'summary.json'
                if path.exists():
                    summary=read_json(path)
                    rows.append({'seed':seed,'status':summary['status'],
                                 'wall_seconds':summary['training_wall_seconds'],
                                 'last_episodes':summary['checkpoints'][-1]['episodes']})
            if rows:
                costs[algorithm]={'repetitions':rows,'sum_run_seconds':sum(x['wall_seconds'] for x in rows)}
        report['games'][game]={'methods':aggregate,'paired_comparisons':paired,'training_costs':costs,
                               'source_id':selection['source_id'],'test_seeds':selection['splits']['test']}
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='outputs/v11')
    p.add_argument('--out',default='outputs/v11/report.json')
    args=p.parse_args()
    result=build(Path(args.root))
    write_json(args.out,result)
    print({k:v for k,v in result.items() if k!='games'})
    for game,data in result['games'].items():
        print(game,{k:(round(v['mean_of_policy_means'],3),v['training_repeat_sample_sd']) for k,v in data['methods'].items()})
