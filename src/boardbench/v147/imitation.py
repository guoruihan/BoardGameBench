"""Finite BC initialization, selected by actual greedy full games, not action fit."""
import argparse
import json
from pathlib import Path
import signal
import time
import torch
from boardbench.artifacts.store import read_json,write_json
from boardbench.v144.training import compact
from .training import Trainer


def run(config,out,device='cuda',resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=resume)
    if time.time()>=config['deadline_unix'] or (out/'done.json').exists():raise ValueError('completed/expired initialization')
    if resume and read_json(out/'config.json')!=config:raise ValueError('immutable BC config')
    write_json(out/'config.json',config);t=Trainer(config,device);started=time.monotonic();stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        if resume:t.load(out/'latest.pt')
        else:
            initial=t.evaluate_panel(config['development_seeds']);write_json(out/'initial.json',initial)
            t.best_metric=t.metric(initial);t.save(out/'best.pt');t.save(out/'latest.pt')
            t.export_policy(t.model,out/'best_policy.pt',config)
        prior=t.elapsed_seconds
        baseline=read_json(out/'initial.json')
        skill_path=out/'skill_record.json'
        skill=read_json(skill_path) if skill_path.exists() else None
        while t.bc_steps<config['imitation_steps'] and time.time()<config['deadline_unix']-config['finish_reserve_seconds'] and not stop[0]:
            result=t.expert_step();t.elapsed_seconds=prior+time.monotonic()-started
            if t.bc_steps%config['imitation_evaluate_interval']==0:
                ev=t.evaluate_panel(config['development_seeds']);write_json(out/f'dev_{t.bc_steps:06d}.json',ev)
                result['development']=compact(ev)
                if t.metric(ev)>t.best_metric:
                    t.best_metric=t.metric(ev);t.save(out/'best.pt');t.export_policy(t.model,out/'best_policy.pt',config)
                # Retain a complementary animal skill even when total-score BC
                # selection rolls back. This is a PPO initialization candidate,
                # NOT a newly promoted stronger policy.
                if (ev['score_mean']>=baseline['score_mean']-20 and ev['animal_mean']>baseline['animal_mean']+3
                        and (skill is None or ev['animal_mean']>skill['animal_mean'])):
                    skill=dict(score_mean=ev['score_mean'],animal_mean=ev['animal_mean'],bc_steps=t.bc_steps)
                    t.save(out/'skill.pt');t.export_policy(t.model,out/'skill_policy.pt',config);write_json(skill_path,skill)
            if t.bc_steps%25==0:
                t.save(out/'latest.pt')
                with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
                print(json.dumps(result),flush=True)
        t.save(out/'latest.pt');t.export_policy(t.model,out/'final_policy.pt',config)
        if stop[0]:write_json(out/'interrupted.json',dict(steps=t.bc_steps));return
        chosen=torch.load(out/'best.pt',map_location=device,weights_only=False);t.model.load_state_dict(chosen['model'])
        decision=t.evaluate_panel(config['decision_seeds']);write_json(out/'decision.json',decision)
        if skill is not None:
            state=torch.load(out/'skill.pt',map_location=device,weights_only=False)
            t.model.load_state_dict(state['model']);skill_decision=t.evaluate_panel(config['decision_seeds'])
            write_json(out/'skill_decision.json',skill_decision)
        write_json(out/'done.json',dict(status='completed',bc_steps=t.bc_steps,selected_bc_steps=chosen['extra_state']['bc_steps'],
            decision=compact(decision),complementary_skill=skill,wall_seconds=t.elapsed_seconds,selection='real full-game development score'))
    finally:t.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(read_json(a.config),a.out,a.device,a.resume)
