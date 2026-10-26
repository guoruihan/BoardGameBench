"""V150 seeds, component heads, unchanged exact-game audit and timing receipts."""
import argparse
import time
from boardbench.artifacts.store import digest,read_json
from boardbench.v146.worker import run
from boardbench.v149.worker import play as previous
from boardbench.v149.timed import TimedSearch
from .value import make_value


def validate(job):
    ranges={'train':(150000000,150900000),'calibration':(150900000,151000000),
            'development':(250000000,251000000),'confirmation':(350000000,351000000)}
    lo,hi=ranges[job['split']]
    if (not lo<=job['environment_seed']<hi or job['seat'] not in (0,1)
            or not 0<job['seconds']<=1 or time.time()>=job['deadline_unix']):raise ValueError('split/seat/budget')
    if job['method'] not in ('model','habitat','search_rollout','search_value_one','search_value_short','search_value_turn'):
        raise ValueError('method')
    for name in ('student','opponent'):
        if digest(job[name])!=job[name+'_sha256']:raise ValueError('policy hash')


def play(job,progress=None):
    def value(*args):return make_value(*args,rank_mix=job.get('rank_mix',0.))
    result,trace=previous(job,progress,validator=validate,value_factory=value,
        resident_factory=lambda search:TimedSearch(search,reserve_seconds=min(.02,search.seconds*.15)))
    trace['format']='v150_value_planning_trajectory'
    return result,trace


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(read_json(a.job),a.out,play_function=play)
