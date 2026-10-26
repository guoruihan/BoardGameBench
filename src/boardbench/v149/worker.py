"""Full-game audit reused unchanged, with explicit per-call timing statistics."""
import argparse
import resource
import time

import numpy as np

from boardbench.artifacts.store import digest, read_json
from boardbench.v146.worker import play as base_play, run
from .search import Search
from .value import Value
from .timed import TimedSearch


def validate_job(job):
    if job['method'] not in ('model','habitat','search_rollout','search_value_one','search_value_short','search_value_turn'):
        raise ValueError('unknown V149 method')
    ranges = {'train': (149000000,149900000), 'calibration': (149900000,150000000),
              'development': (249000000,250000000), 'confirmation': (349000000,350000000)}
    low, high = ranges[job['split']]
    if (not low<=job['environment_seed']<high or job['seat'] not in (0,1)
            or not 0<job['seconds']<=1 or time.time()>=job['deadline_unix']):
        raise ValueError('invalid split/seat/per-call budget or expired campaign')
    for name in ('student','opponent'):
        if digest(job[name]) != job[name+'_sha256']:
            raise ValueError('policy hash mismatch')


def play(job, progress=None, *, validator=validate_job, value_factory=Value,
         search_factory=Search, resident_factory=TimedSearch):
    residents=[]
    children_before=resource.getrusage(resource.RUSAGE_CHILDREN)
    def factory(focal, opponent, seat, continuation, seconds, seed, **kwargs):
        value = (value_factory(job['value_manifest'],job['value_manifest_sha256'],focal.native,job['opponent_sha256'])
                 if continuation.startswith('value') else None)
        search=search_factory(focal,opponent,seat,continuation,seconds,seed,value=value,
                      width=job.get('width',6),**kwargs)
        resident=resident_factory(search);residents.append(resident);return resident
    try:
        result,trace = base_play(job,progress,validator=validator,search_factory=factory)
    finally:
        for resident in residents:resident.close()
    children_after=resource.getrusage(resource.RUSAGE_CHILDREN)
    resident_cpu=(children_after.ru_utime+children_after.ru_stime-children_before.ru_utime-children_before.ru_stime)
    result['cpu_seconds']+=resident_cpu
    result['resident_cpu_seconds']=resident_cpu
    trace['format'] = 'v149_value_planning_trajectory'
    rows = [r for r in trace['decisions'] if r['seat']==job['seat']]
    seconds = np.asarray([r['decision_seconds'] for r in rows])
    excess = np.maximum(0.,seconds-job['seconds'])
    result.update(method=job.get('label',job['method']),decisions=len(rows),
        latency_median=float(np.median(seconds)),latency_p95=float(np.percentile(seconds,95)),
        latency_max=float(seconds.max()),budget_excess_seconds=float(excess.max()),
        over_budget_calls=int((excess>0).sum()),timing_gate=bool(excess.max()<=.002),
        deadline_interruptions=sum(r['search'].get('deadline_interrupted',False) for r in rows),
        resident_timeouts=sum(r['search'].get('resident_timeout',False) for r in rows),
        fallback_rate=result['fallback_decisions']/len(rows),
        completed_cards=len(trace['decisions'][-1]['observation']['players'][job['seat']]['completed_cards']),
        budget_scope='per_actual_atomic_decision',seconds_per_decision=job['seconds'])
    return result,trace


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(read_json(a.job),a.out,play_function=play)
