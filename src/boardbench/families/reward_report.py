"""Read-only progress and paired development results for reward interventions."""
import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np

from .play import write_json
from .pool_evaluator import paired_difference, paired_interval
from .reward_signals import raw_labels


def read(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def evaluation(run, identity):
    if not identity:
        return None
    path = run/'evaluation'/f'{identity}.json.gz'
    if not path.exists():
        return None
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)


def score_difference(candidate, control):
    def groups(rows):
        return {(r['opponent_id'],r['seed'],r['learner_seat']):raw_labels(r,r['learner_seat'])
                for r in rows if r['status']=='complete'}
    a,b = groups(candidate),groups(control)
    if not a or len(a)!=len(candidate) or len(b)!=len(control) or set(a)!=set(b):
        return dict(complete=False, signals={})
    names = sorted(set.intersection(*(set(v) for v in list(a.values())+list(b.values()))))
    summary = {}
    for name in names:
        seeds = {}
        for key in a:
            seeds.setdefault(key[1],[]).append(a[key][name]-b[key][name])
        values = [float(np.mean(v)) for v in seeds.values()]
        summary[name] = dict(mean=float(np.mean(values)), interval=paired_interval(values))
    return dict(complete=True, signals=summary)


def report(directory):
    directory = Path(directory)
    jobs, evaluations = [], {}
    for path in sorted(directory.glob('runs/*/*/config.json')):
        cfg = read(path,{})
        run = path.parent
        state = read(run/'state.json',{})
        condition = cfg['condition']
        game = cfg['game']
        last_metric, validation = {}, None
        try:
            with (run/'metrics.jsonl').open() as handle:
                for line in handle:
                    try: metric=json.loads(line)
                    except ValueError: continue
                    if metric.get('event')=='update': last_metric=metric
                    elif metric.get('event')=='validation': validation=metric
        except OSError:
            pass
        row = dict(game=game,condition=condition,phase=state.get('phase','not_started'),
                   updates=state.get('next_update',0),learner_decisions=state.get('learner_decisions',0),
                   environment_decisions=state.get('environment_decisions',0),
                   decision_cap=cfg['max_decisions'],costs=state.get('costs',{}),
                   auxiliary_loss=last_metric.get('auxiliary_loss'),
                   prior_kl=last_metric.get('prior_kl'),
                   prior_disagreement=last_metric.get('greedy_disagreement_with_prior'),
                   preflight=state.get('preflight'),error=state.get('error'),
                   exit_code=read(run/'exit_code',None),validation=validation)
        if validation:
            evaluations[game,condition] = evaluation(run,validation['candidate'])
        jobs.append(row)
    comparisons = []
    for row in jobs:
        game,condition = row['game'],row['condition']
        if condition=='win': continue
        a,b = evaluations.get((game,condition)),evaluations.get((game,'win'))
        control = next(j for j in jobs if j['game']==game and j['condition']=='win')
        av,bv = row['validation'],control['validation']
        if a and b:
            comparisons.append(dict(game=game,condition=condition,
                candidate_decisions=av['learner_decisions'],control_decisions=bv['learner_decisions'],
                equal_decisions=av['learner_decisions']==bv['learner_decisions'],
                win_difference=paired_difference(a['games'],b['games']),
                score_difference=score_difference(a['games'],b['games'])))
    return dict(observed_unix=time.time(),jobs=jobs,comparisons=comparisons,
                note='Development only; same environment seeds do not imply equal achieved training decisions. No automatic promotion.')


def markdown(data):
    lines=['# Reward intervention pilot', '', data['note'], '',
           '| Game | Condition | Phase | Decisions | Validated win rate | Aux loss | Prior disagreement |',
           '|---|---|---|---:|---:|---:|---:|']
    def fmt(v): return '—' if v is None else f'{v:.4f}'
    for row in data['jobs']:
        validation=row['validation']
        quality=validation['summary']['quality'] if validation else None
        lines.append(f"| {row['game']} | {row['condition']} | {row['phase']} | {row['learner_decisions']} | "
                     f"{fmt(quality)} | {fmt(row['auxiliary_loss'])} | {fmt(row['prior_disagreement'])} |")
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--watch',action='store_true')
    args=p.parse_args()
    while True:
        data=report(args.directory)
        write_json(args.directory/'reward_report.json',data)
        (args.directory/'reward_report.md').write_text(markdown(data))
        print(json.dumps(dict(jobs=len(data['jobs']),updated=sum(j['updates']>0 for j in data['jobs']),
                              exited=sum(j['exit_code'] is not None for j in data['jobs']))),flush=True)
        deadlines=[read(p,{})['deadline_unix'] for p in args.directory.glob('runs/*/*/config.json')]
        if (not args.watch or all(j['exit_code'] is not None for j in data['jobs']) or
                deadlines and min(deadlines)>0 and time.time()>max(deadlines)+120):
            break
        time.sleep(60)


if __name__=='__main__': main()
