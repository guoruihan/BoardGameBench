"""Versioned search games with the unchanged V146 full-game audit/archive."""
import argparse
import time
from boardbench.artifacts.store import digest,read_json
from boardbench.v146.worker import play as base_play,run as base_run
from .search import Search,VARIANTS


def validate_job(job):
    if job['method'] not in ('model','habitat',*('search_'+v for v in VARIANTS)) or job['seat'] not in (0,1):
        raise ValueError('invalid search variant or seat')
    ranges={'calibration':(147900000,148000000),'train':(147000000,147900000),
            'development':(247000000,248000000)}
    if job['split'] not in ranges or not ranges[job['split']][0]<=job['environment_seed']<ranges[job['split']][1]:
        raise ValueError('train/development namespace required; no confirmation search')
    if not 0<job['seconds']<=900 or job['deadline_unix']<=time.time():raise ValueError('expired/unbounded job')
    for key in ('student','opponent'):
        if digest(job[key])!=job[key+'_sha256']:raise ValueError('policy hash changed')


def play(job,progress=None):
    result,trace=base_play(job,progress,validator=validate_job,search_factory=Search)
    trace['format']='v147_duo_expert_trajectory'
    trigger=None
    for row in trace['decisions']:
        obs=row['observation'];seat=row['seat']
        if row['action']['type']!='end_turn' or obs['final_round']:continue
        if sum(c['stack_id']==0 for c in obs['players'][seat]['board'])<=2:
            trigger='self_board' if seat==job['seat'] else 'opponent_board';break
        if sum(obs['remaining_tokens'].values())<3:trigger='supply';break
    result['first_end_trigger']=trigger
    result['final_empty_cells']=sum(c['stack_id']==0 for c in trace['decisions'][-1]['observation']['players'][job['seat']]['board'])
    return result,trace


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();base_run(read_json(a.job),a.out,play_function=play)
