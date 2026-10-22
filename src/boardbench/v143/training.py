"""Competitive complete-game PPO: BOTH current-policy seats learn, history does not."""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch

from boardbench.artifacts.store import digest,read_json,write_json,source_id
from boardbench.environments.harmonies_duo import HarmoniesDuo
from boardbench.v127.training import advantages
from boardbench.v140.training import Trainer as DuoTrainer,infer_actions,summarize
from .policy import LAYOUT,initialize,export_policy


class Trainer(DuoTrainer):
    # Reuse verified native official-duo pool, PPO update, paired-seat evaluator,
    # and close(). New collection is the only changed on-policy semantics.
    VERSION=143
    LAYOUT=LAYOUT
    OBJECTIVES=('margin',)
    MODES=('current','league')
    DEVELOPMENT_BASE=243000000
    CONFIRMATION_BASE=343000000

    def __init__(self,config,device='cpu'):
        self.config=deepcopy(config);self.device=str(device)
        if (config['layout']!=self.LAYOUT or config['rules_version']!=HarmoniesDuo.rules_version
                or config['objective'] not in self.OBJECTIVES or config['mode'] not in self.MODES):
            raise ValueError('explicit official-duo self-play schema required')
        if digest(config['parent'])!=config['parent_sha256']: raise ValueError('curriculum parent digest mismatch')
        if config.get('source_id') and config['source_id']!=source_id(): raise ValueError('frozen source mismatch')
        if (not config['development_seeds'] or not config['confirmation_seeds']
                or any(not self.DEVELOPMENT_BASE<=s<self.DEVELOPMENT_BASE+1000000 for s in config['development_seeds'])
                or any(not self.CONFIRMATION_BASE<=s<self.CONFIRMATION_BASE+1000000 for s in config['confirmation_seeds'])):
            raise ValueError('fresh self-play split namespaces required')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(config['seed'])
        self.model=initialize(config['parent'],device).train()
        self.frozen=deepcopy(self.model).eval().requires_grad_(False)
        self.snapshots=[];self.snapshot_batches=[]
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config['learning_rate'])
        self.rng=torch.Generator(device=device).manual_seed(config['seed']+143000)
        self.seed_rng=np.random.default_rng(config['seed'])
        self.batches=self.matches=self.actions=0
        self.elapsed_seconds=0.;self.best_metric=-float('inf');self.pool=None

    def potential(self,stats,seat):
        return float(stats[seat]-(stats[1-seat] if self.config['objective']=='margin' else 0))

    def collect(self,pool,seeds,learner_seats,opponents,greedy=False,keep_rows=True):
        if not len(seeds)==len(learner_seats)==len(opponents)==pool.count:
            raise ValueError('one match/role specification per slot')
        tick=time.monotonic();arrays=pool.reset(seeds);env_seconds=time.monotonic()-tick;model_seconds=0.
        calls=total_actions=0;trajectories=[[[],[]] for _ in seeds];traces=[[] for _ in seeds]
        tokens=np.zeros((len(seeds),2),int);animals=np.zeros_like(tokens)
        models={'learner':self.model,'current':self.model,'frozen':self.frozen,
                **{f'lag{i}':m for i,m in enumerate(self.snapshots)}}
        while not arrays['done'].all():
            slots=np.flatnonzero(~arrays['done']);seats=arrays['stats'][slots,4].astype(int)
            xs=arrays['features'][slots].copy();masks=arrays['masks'][slots].copy()
            roles=['learner' if seat==learner_seats[slot] else opponents[slot] for slot,seat in zip(slots,seats)]
            tick=time.monotonic()
            inferred,n=infer_actions(models,roles,xs,masks,self.device,self.rng,greedy)
            actions=inferred[:,0].astype(np.int64);calls+=n;total_actions+=len(slots)
            model_seconds+=time.monotonic()-tick
            for j,slot in enumerate(slots):
                seat=int(seats[j]);action=int(actions[j]);role=roles[j]
                tokens[slot,seat]+=3<=action<141;animals[slot,seat]+=144<=action<236
                if keep_rows:
                    trainable=role in ('learner','current')
                    traces[slot].append(dict(seat=seat,action=action,logp=float(inferred[j,1]),
                        value=float(inferred[j,2]),role=role,trainable=trainable))
                    if trainable:
                        s=arrays['stats'][slot];potential=self.potential(s,seat)
                        trajectories[slot][seat].append((xs[j],masks[j],action,float(inferred[j,1]),
                            float(inferred[j,2]),potential,len(traces[slot])-1))
            tick=time.monotonic();pool.step(slots,actions);env_seconds+=time.monotonic()-tick
        games=[];rows=[];seat_counts=[0,0];learners=0;archived=[]
        for slot,(seed,focus,opponent) in enumerate(zip(seeds,learner_seats,opponents)):
            s=arrays['stats'][slot];margin=float(s[focus]-s[1-focus])
            game=dict(seed=int(seed),seat=int(focus),opponent=opponent,score=float(s[focus]),
                opponent_score=float(s[1-focus]),animal=float(s[2+focus]),opponent_animal=float(s[3-focus]),
                margin=margin,win_credit=float(s[5] if focus==0 else 1-s[5]),natural_terminal=True,
                turns=int(s[6])-1,tokens=int(tokens[slot,focus]),animals_placed=int(animals[slot,focus]))
            games.append(game)
            if keep_rows:
                learned_seats=[]
                for seat,trajectory in enumerate(trajectories[slot]):
                    if not trajectory: continue
                    learners+=1;learned_seats.append(seat);seat_counts[seat]+=len(trajectory)
                    final=self.potential(s,seat);potentials=[r[5] for r in trajectory]+[final]
                    rewards=[(b-a)/150 for a,b in zip(potentials,potentials[1:])]
                    if abs(sum(rewards)*150+potentials[0]-final)>1e-7: raise RuntimeError('seat reward mismatch')
                    for row,reward in zip(trajectory,rewards):
                        traces[slot][row[6]]['reward_to_next_own_decision']=reward
                    adv,targets=advantages(rewards,[r[4] for r in trajectory],self.config['gae_lambda'])
                    rows.extend((*r[:4],float(targets[i]),float(adv[i])) for i,r in enumerate(trajectory))
                archived.append(dict(**game,learner_seats=learned_seats,
                    opponent_snapshot_batch=self.snapshot_batches[int(opponent[3:])] if opponent.startswith('lag') else None,
                    decisions=traces[slot]))
        if keep_rows: self.last_traces=archived
        return rows,dict(**summarize(games),per_game=games,environment_seconds=env_seconds,model_seconds=model_seconds,
            total_player_actions=total_actions,learner_actions_by_seat=seat_counts,learner_trajectories=learners,
            current_selfplay_matches=sum(o=='current' for o in opponents),history_matches=sum(o.startswith('lag') for o in opponents),
            inference_calls=calls,mean_inference_batch=total_actions/max(1,calls),
            turns_mean=float(np.mean([g['turns'] for g in games])),tokens_mean=float(np.mean([g['tokens'] for g in games])))

    def batch(self):
        if self.pool is None: self.pool=self.make_pool(self.config['batch_matches'])
        seeds=self.seed_rng.integers(143000000,144000000,self.pool.count).tolist()
        seats=[(i//4+self.batches)%2 for i in range(self.pool.count)]
        lag=f'lag{self.batches%len(self.snapshots)}' if self.snapshots else None
        opponents=[lag if self.config['mode']=='league' and lag and i%4==3 else 'current' for i in range(self.pool.count)]
        rows,result=self.collect(self.pool,seeds,seats,opponents);result.update(self.update(rows))
        self.batches+=1;self.matches+=len(seeds);self.actions+=len(rows)
        if self.config['mode']=='league' and self.batches%self.config['snapshot_interval']==0:
            self.snapshots.append(deepcopy(self.model).eval().requires_grad_(False));self.snapshot_batches.append(self.batches)
            self.snapshots=self.snapshots[-self.config['history_size']:];self.snapshot_batches=self.snapshot_batches[-self.config['history_size']:]
        return dict(**result,batches=self.batches,matches=self.matches,actions=self.actions,
                    snapshot_batches=list(self.snapshot_batches))

    def save(self,path):
        path=Path(path);tmp=path.with_suffix('.partial.pt')
        torch.save(dict(version=self.VERSION,config=self.config,model=self.model.state_dict(),frozen=self.frozen.state_dict(),
            snapshots=[m.state_dict() for m in self.snapshots],snapshot_batches=self.snapshot_batches,
            optimizer=self.optimizer.state_dict(),rng=self.rng.get_state(),seed_rng=self.seed_rng.bit_generator.state,
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.device.startswith('cuda') else [],
            batches=self.batches,matches=self.matches,actions=self.actions,elapsed_seconds=self.elapsed_seconds,best_metric=self.best_metric),tmp)
        tmp.replace(path);write_json(path.with_suffix('.json'),dict(sha256=digest(path),batches=self.batches,
            actions=self.actions,source_id=self.config.get('source_id')))

    def load(self,path):
        if digest(path)!=read_json(Path(path).with_suffix('.json'))['sha256']: raise ValueError('checkpoint digest mismatch')
        s=torch.load(path,map_location=self.device,weights_only=False)
        if s['version']!=self.VERSION or s['config']!=self.config: raise ValueError('exact version/config required')
        self.model.load_state_dict(s['model']);self.frozen.load_state_dict(s['frozen']);self.snapshots=[]
        for state in s['snapshots']:
            m=deepcopy(self.frozen);m.load_state_dict(state);self.snapshots.append(m)
        self.snapshot_batches=s['snapshot_batches'];self.optimizer.load_state_dict(s['optimizer'])
        self.rng.set_state(s['rng'].cpu());self.seed_rng.bit_generator.state=s['seed_rng'];torch.set_rng_state(s['cpu_rng'].cpu())
        if s['cuda_rng']: torch.cuda.set_rng_state_all([r.cpu() for r in s['cuda_rng']])
        for k in ('batches','matches','actions','elapsed_seconds','best_metric'): setattr(self,k,s[k])


def run(config,out,device='cuda',resume=False):
    out=Path(out);out.mkdir(parents=True,exist_ok=resume)
    if resume and (out/'done.json').exists(): raise ValueError('completed self-play trial cannot be extended')
    if config.get('deadline_unix',float('inf'))<=time.time(): raise ValueError('original budget expired')
    write_json(out/'config.json',config);start=time.monotonic();t=Trainer(config,device)
    if resume:
        t.load(out/'latest.pt')
        for name in ('failed','interrupted'):
            p=out/(name+'.json')
            if p.exists(): p.rename(out/(name+f'.before_resume_{time.time_ns()}.json'))
    prior=t.elapsed_seconds;stop_at=min(start+max(0,config['max_seconds']-prior),
        time.monotonic()+config.get('deadline_unix',float('inf'))-time.time())
    stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopping.__setitem__(0,True))
    try:
        if not resume:
            initial=t.evaluate(config['development_seeds']);write_json(out/'initial.json',initial)
            if abs(initial['margin_mean'])>1e-12 or initial['win_credit_mean']!=.5:
                raise RuntimeError('identical initial policy paired-seat self-battle not symmetric')
            export_policy(t.model,out/'initial_policy.pt',config)
            t.best_metric=initial['margin_mean'];t.elapsed_seconds=time.monotonic()-start
            t.save(out/'best.pt');t.save(out/'latest.pt')
            print(json.dumps(dict(event='initial_self_battle',**{k:v for k,v in initial.items() if k!='per_game'})),flush=True)
        while not stopping[0] and time.monotonic()<stop_at-config['finish_reserve_seconds']:
            tick=time.monotonic();behavior_sha=digest(out/'latest.pt');result=t.batch()
            (out/'trajectories').mkdir(exist_ok=True)
            with gzip.open(out/'trajectories'/f'batch_{t.batches:06d}.json.gz','wt') as f:
                json.dump(dict(layout=LAYOUT,rules_version=HarmoniesDuo.rules_version,source_id=config.get('source_id'),
                    behavior_checkpoint_sha256=behavior_sha,parent_sha256=config['parent_sha256'],mode=config['mode'],
                    objective='margin',batch=t.batches,games=t.last_traces),f,separators=(',',':'))
            if t.batches%config['evaluate_interval']==0:
                evaluation=t.evaluate(config['development_seeds']);write_json(out/f'dev_{t.batches:06d}.json',evaluation)
                result['development']={k:v for k,v in evaluation.items() if k!='per_game'}
                if evaluation['margin_mean']>t.best_metric:
                    t.best_metric=evaluation['margin_mean'];t.elapsed_seconds=prior+time.monotonic()-start;t.save(out/'best.pt')
            t.elapsed_seconds=prior+time.monotonic()-start
            result.update(best_development_margin=t.best_metric,elapsed_seconds=t.elapsed_seconds,batch_seconds=time.monotonic()-tick)
            t.save(out/'latest.pt')
            with (out/'curve.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='per_game'}),flush=True)
        t.elapsed_seconds=prior+time.monotonic()-start;t.save(out/'final.pt');t.save(out/'latest.pt')
        export_policy(t.model,out/'final_policy.pt',config)
        if stopping[0]:
            write_json(out/'interrupted.json',dict(elapsed_seconds=t.elapsed_seconds,actions=t.actions));return
        t.close();selected=torch.load(out/'best.pt',map_location=device,weights_only=False);t.model.load_state_dict(selected['model'])
        export_policy(t.model,out/'best_policy.pt',config)
        confirmation=t.evaluate(config['confirmation_seeds'])
        write_json(out/'confirmation.json',dict(**confirmation,selected_checkpoint='best.pt',selection_uses_confirmation=False,
            opponent='frozen transferred V142 starting policy',opponent_parent_sha256=config['parent_sha256']))
        t.model.load_state_dict(t.frozen.state_dict());initial_confirmation=t.evaluate(config['confirmation_seeds'])
        write_json(out/'initial_confirmation.json',initial_confirmation)
        write_json(out/'done.json',dict(status='completed',wall_seconds=prior+time.monotonic()-start,
            budget_seconds=config['max_seconds'],batches=t.batches,actions=t.actions,selected_batch=selected['batches'],
            best_development_margin=t.best_metric,confirmation={k:v for k,v in confirmation.items() if k!='per_game'}))
        print(json.dumps(dict(event='completed',**read_json(out/'done.json'))),flush=True)
    except BaseException as exc:
        write_json(out/'failed.json',dict(error=repr(exc),elapsed_seconds=prior+time.monotonic()-start));raise
    finally:t.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(read_json(a.config),a.out,a.device,a.resume)
