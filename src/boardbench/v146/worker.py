"""One full official-duo game; no hidden snapshots or partial-score successes."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.environments.harmonies_duo import FastHarmoniesDuo, HarmoniesDuo, player_view
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner
from boardbench.v145.policy import load_policy
from .policy import Habitat, ModelPlayer
from .search import TerminalSearch

METHODS = ('model', 'habitat', 'search_habitat', 'search_model')


def public_digest(obs):
    return hashlib.sha256(json.dumps(obs,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_job(job):
    if job['method'] not in METHODS or job['seat'] not in (0,1):
        raise ValueError('invalid method/seat')
    ranges = {'calibration': (146900000,147000000), 'development': (246000000,247000000),
              'train': (146000000,146900000)}
    if job['split'] not in ranges or not ranges[job['split']][0] <= job['environment_seed'] < ranges[job['split']][1]:
        raise ValueError('invalid split; formal test/confirmation forbidden')
    if not 0 < job['seconds'] <= 900 or job['deadline_unix'] <= time.time():
        raise ValueError('expired/unbounded game')
    for name in ('student','opponent'):
        if digest(job[name]) != job[name+'_sha256']:
            raise ValueError('immutable policy mismatch')


def audit_trace(trace):
    game = HarmoniesDuo(); obs,_ = game.reset(trace['environment_seed'])
    rows = trace['decisions']
    for row in rows:
        if (row['seat'] != obs['active_player'] or public_digest(obs) != row['public_sha256']
                or public_digest(row['observation']) != row['public_sha256']
                or row['scores_before'] != [s['total'] for s in obs['score_breakdown']]
                or row['action'] not in obs['legal_actions']):
            raise ValueError('public state/action replay mismatch')
        obs = game.step(row['action']).observation
        if [s['total'] for s in obs['score_breakdown']] != row['next_scores']:
            raise ValueError('score replay mismatch')
    if not obs['terminated'] or obs['scores'] != trace['scores']:
        raise ValueError('natural terminal required')
    if obs['players'][0]['completed_turns'] != obs['players'][1]['completed_turns']:
        raise ValueError('unequal completed turns')
    for seat in (0,1):
        own = [r for r in rows if r['seat'] == seat]
        for i,row in enumerate(own):
            following = own[i+1]['scores_before'][seat] if i+1 < len(own) else obs['scores'][seat]
            expected = (following-row['scores_before'][seat])/150
            if abs(row['reward_to_next_own_decision']-expected)>1e-10:
                raise ValueError('own return mismatch')
            if abs(row['mc_return']-(obs['scores'][seat]-row['scores_before'][seat])/150)>1e-10:
                raise ValueError('remaining MC target mismatch')
        if abs(sum(r['reward_to_next_own_decision'] for r in own)*150-obs['scores'][seat])>1e-7:
            raise ValueError('full episode own reward conservation failed')
    return dict(actions=len(rows),scores=obs['scores'],natural_terminal=True)


def play(job, progress=None, *, validator=validate_job, search_factory=TerminalSearch):
    validator(job); started=time.monotonic(); cpu=time.process_time()
    torch.set_num_threads(1)
    native=NativeEncoder.from_artifact(job['native_artifact']) if job.get('native_artifact') else None
    focal=ModelPlayer(load_policy(job['student']),native)
    opposing=ModelPlayer(load_policy(job['opponent']),native)
    game=FastHarmoniesDuo(); obs,_=game.reset(job['environment_seed'])
    planners=[Planner(),Planner()]; habitat=Habitat(); focus=job['seat']
    search=search_factory(focal,opposing,focus,job['method'].removeprefix('search_'),job['seconds'],
                          job['search_seed'],max_rounds=job.get('max_rounds')) if job['method'].startswith('search_') else None
    decisions=[]; decision_seconds=0.; opponent_seconds=0.
    for _ in range(768):
        if obs['terminated']:break
        if time.time()>=job['deadline_unix']:raise TimeoutError('original campaign deadline')
        seat=obs['active_player']; before=[s['total'] for s in obs['score_breakdown']]
        tick=time.monotonic(); info={}
        if seat!=focus:
            action=opposing.decide(obs,planners[seat])
        elif search:
            action=search.decide(obs,planners);info=search.last
        elif job['method']=='model':
            action=focal.decide(obs,planners[seat])
        else:
            planners[seat].vector(player_view(obs));action=habitat.decide(obs)
        seconds=time.monotonic()-tick
        if seat==focus:decision_seconds+=seconds
        else:opponent_seconds+=seconds
        result=game.step(action)
        decisions.append(dict(seat=seat,observation=obs,public_sha256=public_digest(obs),action=action,
            scores_before=before,next_scores=[s['total'] for s in result.observation['score_breakdown']],
            search=info,decision_seconds=seconds,behavior=job['method'] if seat==focus else 'frozen_opponent'))
        obs=result.observation
        if progress:progress(dict(real_actions=len(decisions),partial_scores=[s['total'] for s in obs['score_breakdown']],
            decision_seconds=decision_seconds,opponent_seconds=opponent_seconds,updated_unix=time.time()))
    if not obs['terminated']:raise RuntimeError('action safety bound; no partial score reported')
    for seat in (0,1):
        own=[r for r in decisions if r['seat']==seat]
        for i,row in enumerate(own):
            following=own[i+1]['scores_before'][seat] if i+1<len(own) else obs['scores'][seat]
            row['reward_to_next_own_decision']=(following-row['scores_before'][seat])/150
            row['mc_return']=(obs['scores'][seat]-row['scores_before'][seat])/150
    trace=dict(format='v146_duo_expert_trajectory',rules_version=HarmoniesDuo.rules_version,
        **job,scores=obs['scores'],decisions=decisions)
    audit=audit_trace(trace)
    score=obs['score_breakdown'][focus]; opposing_score=obs['scores'][1-focus]
    stats=dict(status='completed',method=job['method'],split=job['split'],environment_seed=job['environment_seed'],
        seat=focus,search_seed=job['search_seed'],score=score['total'],animal=score['animals'],
        terrain=score['total']-score['animals'],opponent_score=opposing_score,margin=score['total']-opposing_score,
        turns=obs['players'][focus]['completed_turns'],animals_placed=sum(c['animal'] is not None for c in obs['players'][focus]['board']),
        cards_taken=len(obs['players'][focus]['active_cards'])+len(obs['players'][focus]['completed_cards']),
        decision_seconds=decision_seconds,opponent_seconds=opponent_seconds,
        elapsed_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,natural_terminal=True,
        budget_excess_seconds=max(0,decision_seconds-job['seconds']) if search else 0.,
        completed_rounds=sum(r['search'].get('completed_rounds',0) for r in decisions),
        search_decisions=sum(bool(r['search']) for r in decisions),
        fallback_decisions=sum(r['search'].get('fallback',False) for r in decisions),
        simulation_steps=sum(r['search'].get('simulation_steps',0) for r in decisions),audit=audit)
    return stats,trace


def run(job,out,*,play_function=play):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if job['source_id']!=source_id():raise ValueError('worker must use frozen source')
    stopping=[False]
    def stop(*_):stopping[0]=True
    signal.signal(signal.SIGTERM,stop)
    def progress(value):
        write_json(out/'progress.json',value)
        if stopping[0]:raise InterruptedError('requested stop; incomplete game is not training data')
    try:
        result,trace=play_function(job,progress)
        path=out/'trajectory.json.gz';tmp=out/'trajectory.partial.gz'
        with gzip.open(tmp,'wt') as stream:json.dump(trace,stream,separators=(',',':'),allow_nan=False)
        tmp.replace(path);result.update(id=job['id'],source_id=source_id(),trajectory=str(path),trajectory_sha256=digest(path),
            student_sha256=job['student_sha256'],opponent_sha256=job['opponent_sha256'])
        write_json(out/'result.json',result)
    except BaseException as exc:
        write_json(out/'failed.json',dict(error=repr(exc),time=time.time()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run(read_json(a.job),a.out)
