"""Both-seat complete-game own-score PPO; immutable fixed-opponent selection."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import signal
import time
import torch

from boardbench.artifacts.store import digest,read_json,write_json
from boardbench.v121.collector import WorkerTimeoutError,WorkerExitedError
from boardbench.v140.training import summarize
from boardbench.v143.training import Trainer as PreviousTrainer
from boardbench.v143.policy import load_policy as load_previous
from .policy import LAYOUT,export_policy,load_policy
from .evaluation import summarize_panel


class Trainer(PreviousTrainer):
    VERSION=144
    LAYOUT=LAYOUT
    OBJECTIVES=('score','margin')
    MODES=('current','league','diverse')
    DEVELOPMENT_BASE=244000000
    CONFIRMATION_BASE=344000000

    def __init__(self,config,device='cpu'):
        super().__init__(config,device)
        self.pending_curve=None
        self.pool_recovery_events=[]
        if not 0<=config['gae_lambda']<=1:raise ValueError('invalid GAE lambda')
        if not config['decision_seeds'] or any(not 244000000<=s<245000000 for s in config['decision_seeds']):
            raise ValueError('decision panel must be development only')
        if set(config['decision_seeds']) & set(config['development_seeds']):raise ValueError('decision/curve panels overlap')
        if config.get('warm_start'):
            if digest(config['warm_start'])!=config['warm_start_sha256']:raise ValueError('warm-start digest mismatch')
            self.model.load_state_dict(load_policy(config['warm_start'],device).state_dict())
        # self.frozen ALWAYS remains the common V142 transfer, not this round's parent.
        self.references={'transfer':self.frozen}
        for reference in config.get('evaluation_opponents',[]):
            name=reference['name']
            if name in self.references:raise ValueError('duplicate evaluation opponent')
            if digest(reference['path'])!=reference['sha256']:raise ValueError('reference digest mismatch')
            loader=load_previous if reference['format']=='v143' else load_policy if reference['format']=='v144' else None
            if loader is None:raise ValueError('unknown opponent format')
            self.references[name]=loader(reference['path'],device).eval().requires_grad_(False)

    def checkpoint_extra_state(self):
        return dict(pending_curve=self.pending_curve,pool_recovery_events=self.pool_recovery_events)

    def restore_extra_state(self,state):
        self.pending_curve=state.get('pending_curve')
        self.pool_recovery_events=state.get('pool_recovery_events',[])

    def get_pool(self,count):
        if self.pool is not None and self.pool.count!=count:
            if self.pool.started and not self.pool.arrays['done'].all():raise ValueError('cannot replace unfinished cohort')
            self.close()
        if self.pool is None:self.pool=self.make_pool(count)
        return self.pool

    def batch(self):
        self.get_pool(self.config['batch_matches'])
        seeds=self.seed_rng.integers(144000000,145000000,self.pool.count).tolist()
        seats=[(i//4+self.batches)%2 for i in range(self.pool.count)]
        lag=f'lag{self.batches%len(self.snapshots)}' if self.snapshots else None
        opponents=[]
        for i in range(self.pool.count):
            if self.config['mode']=='diverse' and i%4==2:opponents.append('frozen')
            elif self.config['mode']!='current' and lag and i%4==3:opponents.append(lag)
            else:opponents.append('current')
        rows,result=self.collect(self.pool,seeds,seats,opponents);result.update(self.update(rows))
        self.batches+=1;self.matches+=len(seeds);self.actions+=len(rows)
        if self.config['mode']!='current' and self.batches%self.config['snapshot_interval']==0:
            self.snapshots.append(deepcopy(self.model).eval().requires_grad_(False));self.snapshot_batches.append(self.batches)
            self.snapshots=self.snapshots[-self.config['history_size']:];self.snapshot_batches=self.snapshot_batches[-self.config['history_size']:]
        return dict(**result,batches=self.batches,matches=self.matches,actions=self.actions,
            fixed_training_matches=opponents.count('frozen'),snapshot_batches=list(self.snapshot_batches),objective=self.config['objective'])

    def evaluate(self,seeds):
        """Reuse the completed training cohort; never spawn per opponent/panel.

        Evaluation is greedy, so a worker failure can replay this entire panel
        once without consuming policy or environment-training RNG. Training
        failures instead restore the last checkpoint in the outer supervisor.
        """
        specs=[(s,seat) for s in seeds for seat in range(2)]
        size=min(self.config['batch_matches'],len(specs))
        if not size or len(specs)%size:raise ValueError('paired evaluation cohort must divide panel')
        for attempt in range(2):
            try:
                pool=self.get_pool(size);games=[]
                for start in range(0,len(specs),size):
                    batch=specs[start:start+size]
                    _,result=self.collect(pool,[s for s,_ in batch],[seat for _,seat in batch],
                        ['frozen']*size,greedy=True,keep_rows=False)
                    games.extend(result['per_game'])
                return dict(**summarize(games),per_game=games)
            except (WorkerTimeoutError,WorkerExitedError,BrokenPipeError,EOFError) as exc:
                self.pool_recovery_events.append(dict(time=time.time(),attempt=attempt,error=repr(exc),
                    diagnostic=getattr(exc,'diagnostic',None),operation='greedy_evaluation'))
                self.close()
                remaining=self.config.get('evaluation_deadline_unix',self.config.get('deadline_unix',float('inf')))-time.time()
                if attempt or remaining<130:raise

    def evaluate_panel(self,seeds):
        if not seeds or len(seeds)!=len(set(seeds)) or any(not 244000000<=s<245000000 and not 344000000<=s<345000000 for s in seeds):
            raise ValueError('only fresh development/confirmation panels allowed')
        games=[];opponents={};anchor=self.frozen
        try:
            for name,model in self.references.items():
                self.frozen=model
                ev=self.evaluate(seeds)
                rows=[dict(g,evaluation_opponent=name) for g in ev['per_game']]
                opponents[name]=summarize_panel(rows);games.extend(rows)
        finally:self.frozen=anchor
        return dict(**summarize_panel(games),opponents=opponents,per_game=games)

    def metric(self,evaluation):
        return evaluation['margin_mean' if self.config['objective']=='margin' else 'score_mean']


def compact(evaluation):return {k:v for k,v in evaluation.items() if k!='per_game'}


def finish_pending(t,out,elapsed):
    """Finish a checkpointed update's evaluation/curve without training it again."""
    if t.pending_curve is None:return
    result=t.pending_curve
    if t.batches%t.config['evaluate_interval']==0:
        evaluation=t.evaluate_panel(t.config['development_seeds'])
        write_json(out/f'dev_{t.batches:06d}.json',evaluation);result['development']=compact(evaluation)
        if t.metric(evaluation)>t.best_metric:
            t.best_metric=t.metric(evaluation);t.elapsed_seconds=elapsed();t.save(out/'best.pt')
            export_policy(t.model,out/'best_policy.pt',t.config)
    t.elapsed_seconds=elapsed()
    result.update(best_development_metric=t.best_metric,selection_metric='margin' if t.config['objective']=='margin' else 'score',
        elapsed_seconds=t.elapsed_seconds,pool_recovery_events=t.pool_recovery_events)
    curve=out/'curve.jsonl';last_batch=None
    if curve.exists():
        # A crash between append and checkpoint clearing must not duplicate a row.
        with curve.open('rb') as f:
            f.seek(max(0,curve.stat().st_size-262144))
            for line in reversed(f.read().splitlines()):
                try:last_batch=json.loads(line)['batches'];break
                except (json.JSONDecodeError,KeyError):continue
    if last_batch!=t.batches:
        with curve.open('a') as f:f.write(json.dumps(result)+'\n')
        print(json.dumps(compact(result)),flush=True)
    t.pending_curve=None;t.save(out/'latest.pt')


def run(config,out,device='cuda',resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=resume)
    if resume and (out/'done.json').exists():raise ValueError('completed trial cannot be extended')
    if config['deadline_unix']<=time.time():raise ValueError('original budget expired')
    if resume and read_json(out/'config.json')!=config:raise ValueError('exact config required')
    write_json(out/'config.json',config);start=time.monotonic();t=Trainer(config,device)
    stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    try:
        if resume:
            t.load(out/'latest.pt')
            for name in ('failed','interrupted'):
                path=out/(name+'.json')
                if path.exists():path.rename(out/(name+f'.before_resume_{time.time_ns()}.json'))
        prior=t.elapsed_seconds
        if not resume:
            initial=t.evaluate_panel(config['development_seeds']);write_json(out/'initial.json',initial)
            t.best_metric=t.metric(initial);t.elapsed_seconds=time.monotonic()-start
            t.save(out/'best.pt');t.save(out/'latest.pt');export_policy(t.model,out/'initial_policy.pt',config)
            export_policy(t.model,out/'best_policy.pt',config)
            print(json.dumps(dict(event='initial',objective=config['objective'],**compact(initial))),flush=True)
        elapsed=lambda:prior+time.monotonic()-start
        finish_pending(t,out,elapsed)
        while (not stopping[0] and time.time()<config['deadline_unix']-config['finish_reserve_seconds']
               and t.batches<config.get('max_batches',float('inf'))):
            tick=time.monotonic();behavior=digest(out/'latest.pt');result=t.batch()
            directory=out/'trajectories';directory.mkdir(exist_ok=True)
            path=directory/f'batch_{t.batches:06d}.json.gz';tmp=path.with_suffix('.partial.gz')
            with gzip.open(tmp,'wt') as f:
                json.dump(dict(layout=LAYOUT,rules_version=config['rules_version'],source_id=config.get('source_id'),
                    objective=config['objective'],mode=config['mode'],batch=t.batches,behavior_checkpoint_sha256=behavior,
                    parent_sha256=config['parent_sha256'],warm_start_sha256=config.get('warm_start_sha256'),games=t.last_traces),f,separators=(',',':'))
            tmp.replace(path)
            t.elapsed_seconds=prior+time.monotonic()-start
            result.update(batch_seconds=time.monotonic()-tick)
            # Persist the complete PPO update BEFORE fallible evaluation. Resume
            # processes pending_curve first, with unchanged parameters/RNG.
            t.pending_curve=result
            t.save(out/'latest.pt')
            finish_pending(t,out,elapsed)
        t.elapsed_seconds=prior+time.monotonic()-start;t.save(out/'latest.pt');t.save(out/'final.pt')
        export_policy(t.model,out/'final_policy.pt',config)
        if stopping[0]:
            write_json(out/'interrupted.json',dict(batches=t.batches,actions=t.actions,deadline_unix=config['deadline_unix']));return
        t.close();selected=torch.load(out/'best.pt',map_location=device,weights_only=False)
        t.model.load_state_dict(selected['model']);export_policy(t.model,out/'best_policy.pt',config)
        decision=t.evaluate_panel(config['decision_seeds']);write_json(out/'decision.json',decision)
        # No confirmation is read here. Later rounds only consume DEVELOPMENT.
        write_json(out/'done.json',dict(status='completed',batches=t.batches,actions=t.actions,
            selected_batch=selected['batches'],best_development_metric=t.best_metric,
            objective=config['objective'],decision=compact(decision),wall_seconds=prior+time.monotonic()-start))
        print(json.dumps(dict(event='completed',**read_json(out/'done.json'))),flush=True)
    except BaseException as exc:
        write_json(out/'failed.json',dict(error=repr(exc),time=time.time(),diagnostic=getattr(exc,'diagnostic',None),
            in_memory_pending_update=t.pending_curve is not None,pool_recovery_events=t.pool_recovery_events));raise
    finally:t.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(read_json(a.config),a.out,a.device,a.resume)
