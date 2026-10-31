"""Paired fixed-panel evaluation and public-state behavior diagnostics."""
import hashlib
from concurrent.futures import FIRST_COMPLETED, wait
import json
import math
import time
import numpy as np

from .opponent_pool import pool_seed
from .play import make_policy, summarize
from .registry import game_class
from .train import match_worker


def bounded_jobs(executor, function, jobs, deadline, workers, stopped=lambda:False):
    """Keep a bounded rolling window busy, returning original job order.

    Slow games do not create a barrier for all other workers. At most `workers`
    are in flight; stop/deadline prevents further submissions, but current games
    finish under their per-game caps. No completion-order training randomness.
    """
    results, pending, next_index = {}, {}, 0
    def fill():
        nonlocal next_index
        while next_index<len(jobs) and len(pending)<workers:
            remaining=deadline-time.time()
            if stopped() or remaining<3:break
            job=dict(jobs[next_index])
            job['max_seconds']=min(job.get('max_seconds',120.),max(.1,remaining-2))
            pending[executor.submit(function,job)]=next_index
            next_index+=1
    fill()
    while pending:
        completed,_=wait(pending,return_when=FIRST_COMPLETED)
        for future in completed:
            index=pending.pop(future)
            results[index]=future.result()
        fill()
    return [results[i] for i in range(next_index)]


def paired_interval(values):
    """Bootstrap whole environment-seed groups, preserving seat/opponent links."""
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(170)
    means = a[rng.integers(len(a), size=(1000, len(a)))].mean(1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def panel_summary(rows, names, expected):
    by_opponent, paired = {}, {}
    for name in names:
        r = [x for x in rows if x['opponent_id'] == name]
        summary = summarize(r, [x['learner_seat'] for x in r])
        summary['scheduled'] = expected*2
        summary['not_run'] = expected*2-len(r)
        scored = [x for x in r if x['status']=='complete' and
                  len(x.get('outcome',{}).get('scores',[]))==2 and
                  all(isinstance(v,(int,float)) and math.isfinite(v) for v in x['outcome']['scores'])]
        summary['own_score'] = float(np.mean([x['outcome']['scores'][x['learner_seat']] for x in scored])) if scored else None
        summary['opponent_score'] = float(np.mean([x['outcome']['scores'][1-x['learner_seat']] for x in scored])) if scored else None
        seconds = sum(x.get('policy_seconds',[0.,0.])[x['learner_seat']] for x in r)
        decisions = sum(x.get('seat_decisions',[0,0])[x['learner_seat']] for x in r)
        summary['mean_policy_seconds'] = seconds/decisions if decisions else None
        summary['max_policy_seconds'] = max((x.get('policy_max_seconds',[0.,0.])[x['learner_seat']] for x in r),default=None)
        by_opponent[name] = summary
    for r in rows:
        if r['status'] == 'complete':
            paired.setdefault(str(r['seed']), []).append(.5+.5*r['rewards'][r['learner_seat']])
    grouped = [float(np.mean(v)) for v in paired.values() if len(v) == 2*len(names)]
    complete = (len(rows) == expected*2*len(names) and all(r['status']=='complete' for r in rows))
    quality = float(np.mean(grouped)) if grouped else None
    return dict(opponents=by_opponent, all_complete=complete, quality=quality,
                seed_groups=len(grouped), interval=paired_interval(grouped),
                worst_opponent=min((x['completed_win_rate'] for x in by_opponent.values()
                                   if x['completed_win_rate'] is not None), default=None))


def evaluate(executor, game, candidate, panel, num_seeds, split, deadline, workers,
             max_seconds=120., record=False, stopped=lambda:False):
    jobs, labels = [], []
    for name, opponent in panel.items():
        for i in range(num_seeds):
            seed = pool_seed(split, 0, index=i)
            for seat in (0, 1):
                specs = [opponent, opponent]
                specs[seat] = candidate
                jobs.append(dict(game_id=game, specs=specs, seed=seed, first_player=0,
                                 max_seconds=max_seconds, record=record, record_legal=record,
                                 measure_policy_time=True))
                labels.append((name, seat))
    began = time.time()
    rows = bounded_jobs(executor, match_worker, jobs, deadline, workers, stopped)
    for row, (name, seat) in zip(rows, labels):
        row.update(opponent_id=name, learner_seat=seat)
    summary = panel_summary(rows, list(panel), num_seeds)
    summary['seconds'] = time.time()-began
    summary['requested_seeds'] = num_seeds
    summary['split'] = split
    return summary, rows


def paired_difference(candidate, baseline):
    def groups(rows):
        return {(r['opponent_id'], r['seed'], r['learner_seat']): .5+.5*r['rewards'][r['learner_seat']]
                for r in rows if r['status']=='complete'}
    a, b = groups(candidate), groups(baseline)
    # Missing games cannot silently yield a promotion from a favorable subset.
    if not a or len(a) != len(candidate) or len(b) != len(baseline) or set(a) != set(b):
        return dict(mean=None, interval=None, complete=False)
    values = {}
    for key in a:
        values.setdefault(key[1], []).append(a[key]-b[key])
    grouped = [float(np.mean(v)) for v in values.values()]
    return dict(mean=float(np.mean(grouped)), interval=paired_interval(grouped), complete=True)


def public_fingerprint_states(rows, count=512):
    """Balance early/middle/late observed histories from multiple fixed policies."""
    bins = [[], [], []]
    seen = set()
    for row in rows:
        trace = row.get('trajectory', [])
        for i, item in enumerate(trace):
            if not item.get('legal_actions'):
                continue
            state = dict(observation=item['observation'], actions=item['legal_actions'])
            key = json.dumps(state, sort_keys=True, separators=(',', ':'))
            if key in seen:
                continue
            seen.add(key)
            bins[min(2, i*3//max(1, len(trace)))].append(state)
    rng = np.random.default_rng(170)
    for group in bins:
        rng.shuffle(group)
    result = []
    while any(bins) and len(result) < count:
        for group in bins:
            if group and len(result) < count:
                result.append(group.pop())
    return result


def fingerprint(game, spec, states):
    # Common candidate networks have no hidden recurrent state. Dedicated,
    # history-dependent evaluation bots are never fingerprinted as pool members.
    if spec['kind'] != 'neural' or spec.get('stochastic'):
        raise ValueError('Fingerprints require the deployed common deterministic network')
    policy = make_policy(game_class(game), spec)
    result = []
    for item in states:
        action = policy.decide(item['observation'], item['actions'])
        encoded = json.dumps(action, sort_keys=True, separators=(',', ':'))
        result.append(hashlib.sha256(encoded.encode()).hexdigest()[:24])
    return result
