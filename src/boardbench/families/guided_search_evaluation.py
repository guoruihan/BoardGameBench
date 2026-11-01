"""Frozen search matches or expert labels on a student's actual trajectory.

Label mode executes the student's action and stores the teacher's suggestion
separately. Its terminal return belongs to the executed student trajectory.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from .play import make_policy, policy_seed, write_json
from .reference_search import TerminalReference, guarded_spec
from .registry import game_class
from .state_io import copy_data


def run(cfg):
    game=cfg['game'];cls=game_class(game);env=cls();env.reset(cfg['seed'],first_player=0)
    reference=cfg['reference'];search_seat=cfg['search_seat'];label=cfg['mode']=='label'
    if cfg['mode'] not in ('label','match'):raise ValueError('Unknown search evaluation mode')
    student_seat=search_seat if label else 1-search_seat
    candidate=make_policy(cls,guarded_spec(game,cfg['candidate']),policy_seed(cfg['seed'],student_seat))
    opponent=make_policy(cls,guarded_spec(game,reference['rl_spec']),policy_seed(cfg['seed'],1-student_seat))
    search=TerminalReference(cls,reference['rl_spec'],seat=search_seat,seconds=cfg['budget'],
        seed=policy_seed(cfg['seed'],search_seat)^173,weights=reference.get('heuristic_weights'),
        objective=reference['objective'],continuation=reference['continuation'],
        improvement_z=reference.get('improvement_z',0.))
    began=time.monotonic();cpu=time.process_time();trace=[];count=0;student_seconds=0.;last=began
    path=Path(cfg['output']);path.parent.mkdir(parents=True,exist_ok=True)
    while not env.outcome()['terminated']:
        if count>=10000 or time.monotonic()-began>cfg['budget']+180:break
        seat=env.current_player;obs=env.observe(seat);legal=env.legal_actions(seat)
        if not legal:break
        detail=None;teacher_action=None
        if seat==search_seat:
            teacher_action=search.decide(env);detail=copy_data(search.last)
        if seat==student_seat:
            t=time.monotonic();action=candidate.decide(obs,legal);student_seconds+=time.monotonic()-t
        elif not label: action=teacher_action
        else: action=opponent.decide(obs,legal)
        trace.append(dict(player=seat,observation=copy_data(obs),legal_actions=copy_data(legal),
                          action=copy_data(action),teacher_action=copy_data(teacher_action),search=detail))
        env.step(action);count+=1
        if time.monotonic()-last>10:
            write_json(path.with_suffix('.progress.json'),dict(game=game,decisions=count,
                       search_seconds=search.spent,observed_unix=time.time()));last=time.monotonic()
    outcome=env.outcome();status='complete' if outcome['terminated'] else 'truncated'
    if search.spent>cfg['budget']:status='budget_exceeded'
    result=dict(game=game,config=cls.config_id,rules_version=cls.rules_version,seed=cfg['seed'],
        first_player=0,search_seat=search_seat,learner_seat=student_seat,budget=cfg['budget'],
        objective=reference['objective'],status=status,mode=cfg['mode'],candidate=cfg['candidate'],
        reference=reference,source_commit=cfg['source_commit'],outcome=outcome,
        score_breakdown=env.score_breakdown(),decisions=count,search_seconds=search.spent,
        student_seconds=student_seconds,cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-began,
        simulation_steps=search.total_steps,completed_rounds=search.total_rounds)
    trajectory=path.with_suffix('.trajectory.json.gz');temp=trajectory.with_suffix('.part')
    with gzip.open(temp,'wt',encoding='utf-8') as f:json.dump(dict(result=result,trajectory=trace),f,ensure_ascii=False,allow_nan=False)
    temp.replace(trajectory);result['trajectory']=trajectory.name
    result['trajectory_sha256']=hashlib.sha256(trajectory.read_bytes()).hexdigest()
    write_json(path,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if Path(cfg['output']).exists():return
    result=run(cfg);print(json.dumps({k:v for k,v in result.items() if k not in ('reference','candidate')}),flush=True)


if __name__=='__main__':main()
