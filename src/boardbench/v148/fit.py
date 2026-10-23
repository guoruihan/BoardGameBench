"""Bounded offline search learning, full-game model selection, exact recovery."""
import argparse
import json
from pathlib import Path
import signal
import time

from boardbench.artifacts.store import read_json,write_json
from boardbench.v144.training import compact
from .learning import Learner
from .evaluation import Trainer
from .value import calibration


def run(config,out,device='cuda',resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=resume)
    if time.time()>=config['deadline_unix'] or (out/'done.json').exists(): raise ValueError('completed/expired')
    if resume and read_json(out/'config.json')!=config: raise ValueError('immutable fit config')
    write_json(out/'config.json',config);learner=Learner(config,device)
    evaluator=None;stop=[False];start=time.monotonic()
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        if config['mode']!='value':
            evaluator=Trainer(config['evaluation_config'],device)
        if resume: learner.load(out/'latest.pt')
        if not (out/'selection.json').exists():
            metrics=learner.validate();write_json(out/'initial_validation.json',metrics)
            learner.save(out/'value_best.pt');learner.export(out/'value_policy.pt')
            score=None
            if evaluator:
                evaluator.model.load_state_dict(learner.model.state_dict())
                ev=evaluator.evaluate_panel(config['evaluation_config']['development_seeds'])
                write_json(out/'initial.json',ev);score=ev['score_mean']
            learner.save(out/'best.pt');learner.export(out/'best_policy.pt');learner.save(out/'latest.pt')
            write_json(out/'selection.json',dict(best_score=score,best_value=metrics['value_mse'],
                                               value_metrics=metrics,best_step=0,value_step=0))
        selected=read_json(out/'selection.json')
        while (learner.steps<config['steps'] and not stop[0]
               and time.time()<config['deadline_unix']-config.get('finish_reserve_seconds',120)):
            row=learner.step()
            if learner.steps%config['evaluate_interval']==0 or learner.steps==config['steps']:
                metrics=learner.validate();row['validation']=metrics
                if metrics['value_mse']<selected['best_value']:
                    selected.update(best_value=metrics['value_mse'],value_metrics=metrics,value_step=learner.steps)
                    learner.save(out/'value_best.pt');learner.export(out/'value_policy.pt')
                if evaluator:
                    evaluator.model.load_state_dict(learner.model.state_dict())
                    ev=evaluator.evaluate_panel(config['evaluation_config']['development_seeds'])
                    write_json(out/f'dev_{learner.steps:06d}.json',ev);row['development']=compact(ev)
                    if ev['score_mean']>selected['best_score']:
                        selected.update(best_score=ev['score_mean'],best_step=learner.steps)
                        learner.save(out/'best.pt');learner.export(out/'best_policy.pt')
                write_json(out/'selection.json',selected)
                write_json(out/'value_manifest.json',calibration(out/'value_policy.pt',selected['value_metrics'],learner))
            if learner.steps%25==0:
                learner.save(out/'latest.pt')
                with (out/'curve.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        completed_steps=learner.steps
        learner.save(out/'latest.pt');learner.export(out/'final_policy.pt')
        if stop[0]: write_json(out/'interrupted.json',dict(steps=learner.steps));return
        write_json(out/'value_manifest.json',calibration(out/'value_policy.pt',selected['value_metrics'],learner))
        if evaluator:
            learner.load(out/'best.pt');evaluator.model.load_state_dict(learner.model.state_dict())
            ev=evaluator.evaluate_panel(config['evaluation_config']['decision_seeds']);write_json(out/'decision.json',ev)
        write_json(out/'done.json',dict(status='completed',selection=selected,completed_steps=completed_steps,
            requested_steps=config['steps'],stop_reason='steps' if completed_steps>=config['steps'] else 'stage_budget',
            wall_seconds=time.monotonic()-start,completed_unix=time.time(),selection_uses_confirmation=False))
    finally:
        if evaluator:evaluator.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');a=p.parse_args()
    run(read_json(a.config),a.out,a.device,a.resume)
