"""Complete-episode own-score PPO; warm starts, fresh Adam, score-only selection."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from boardbench.artifacts.store import digest, read_json, write_json, source_id
from boardbench.environments.harmonies_market import HarmoniesMarket
from boardbench.v127.training import advantages
from boardbench.v138.baseline import load_model
from boardbench.v140.model import DuoPolicy
from boardbench.v140.policy import load_policy as load_duo
from boardbench.v140.training import Trainer as DuoTrainer, infer_actions
from .collector import EnvironmentPool, REASONS
from .encoding import LAYOUT
from .policy import export_policy


def summarize(games):
    out = dict(games=len(games), environments=len({g['seed'] for g in games}),
               end_reasons=dict(Counter(g['end_reason'] for g in games)))
    for key in ('score', 'animal', 'completed_cards', 'turns', 'tokens', 'animals_placed'):
        values = np.array([g[key] for g in games], np.float64)
        out[key+'_mean'] = float(values.mean())
        out[key+'_sem'] = float(values.std(ddof=1)/len(values)**.5) if len(values)>1 else None
    return out


def paired_delta(after, before):
    a, b = after['per_game'], before['per_game']
    if [g['seed'] for g in a] != [g['seed'] for g in b]: raise ValueError('unpaired evaluation')
    deltas = np.array([x['score']-y['score'] for x,y in zip(a,b)])
    return dict(environments=len(a), score_delta=float(deltas.mean()),
                score_delta_sem=float(deltas.std(ddof=1)/len(a)**.5) if len(a)>1 else None)


class Trainer(DuoTrainer):
    # Only the already verified masked PPO optimizer/update is inherited.
    def __init__(self, config, device='cpu'):
        self.config = deepcopy(config); self.device = str(device)
        if (config['layout'] != LAYOUT or config['objective'] != 'score'
                or config['rules_version'] != HarmoniesMarket.rules_version
                or config['initialization'] not in ('solo', 'duo')):
            raise ValueError('explicit own-score curriculum schema required')
        if digest(config['parent']) != config['parent_sha256'] or digest(config['duo_parent']) != config['duo_parent_sha256']:
            raise ValueError('initialization artifact digest mismatch')
        if config.get('source_id') and config['source_id'] != source_id(): raise ValueError('frozen source mismatch')
        if (not config['development_seeds'] or not config['confirmation_seeds']
                or any(not 242000000 <= s < 243000000 for s in config['development_seeds'])
                or any(not 342000000 <= s < 343000000 for s in config['confirmation_seeds'])):
            raise ValueError('fresh development/confirmation namespace required')
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.manual_seed(config['seed'])
        if config['initialization'] == 'solo':
            solo, _ = load_model(config['parent']); self.model = DuoPolicy(solo, 'score').to(device)
        else:
            self.model = load_duo(config['duo_parent'], device)
            if self.model.objective != 'score': raise ValueError('own-score duo initialization required')
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.rng = torch.Generator(device=device).manual_seed(config['seed']+142000)
        self.seed_rng = np.random.default_rng(config['seed'])
        self.batches = self.matches = self.actions = 0
        self.elapsed_seconds = 0.; self.best_metric = -float('inf'); self.pool = None

    def make_pool(self, count):
        return EnvironmentPool(count, min(self.config['workers'], count), native_artifact=self.config['native_artifact'])

    def collect(self, pool, seeds, greedy=False, keep_rows=True):
        if len(seeds) != pool.count: raise ValueError('one seed per slot')
        tick = time.monotonic(); arrays = pool.reset(seeds)
        env_seconds = time.monotonic()-tick; model_seconds = 0.; calls = actions_count = 0
        trajectories = [[] for _ in seeds]
        while not arrays['done'].all():
            slots = np.flatnonzero(~arrays['done'])
            xs = arrays['features'][slots].copy(); masks = arrays['masks'][slots].copy()
            tick = time.monotonic()
            inferred, n = infer_actions({'learner': self.model}, ['learner']*len(slots),
                                         xs, masks, self.device, self.rng, greedy)
            actions = inferred[:,0].astype(np.int64)
            calls += n; actions_count += len(slots); model_seconds += time.monotonic()-tick
            if keep_rows:
                for j, slot in enumerate(slots):
                    trajectories[slot].append((xs[j], masks[j], int(actions[j]), float(inferred[j,1]),
                                                float(inferred[j,2]), float(arrays['stats'][slot,0])))
            tick = time.monotonic(); pool.step(slots, actions); env_seconds += time.monotonic()-tick
        rows, games, traces = [], [], []
        for slot, (seed, trajectory) in enumerate(zip(seeds, trajectories)):
            s = arrays['stats'][slot]
            game = dict(seed=int(seed), score=float(s[0]), animal=float(s[1]), completed_cards=int(s[2]),
                        turns=int(s[3]), tokens=int(s[4]), end_reason={v:k for k,v in REASONS.items()}[int(s[5])],
                        animals_placed=int(s[6]), natural_terminal=True)
            games.append(game)
            if keep_rows:
                potentials = [r[5] for r in trajectory]+[game['score']]
                rewards = [(b-a)/150 for a,b in zip(potentials,potentials[1:])]
                if abs(sum(rewards)*150+potentials[0]-game['score']) > 1e-7:
                    raise RuntimeError('own-score reward accounting mismatch')
                adv, target = advantages(rewards, [r[4] for r in trajectory], self.config['gae_lambda'])
                rows.extend((*r[:4],float(target[i]),float(adv[i])) for i,r in enumerate(trajectory))
                traces.append(dict(**game, decisions=[dict(action=r[2], logp=r[3], value=r[4],
                    reward=rewards[i]) for i,r in enumerate(trajectory)]))
        if keep_rows: self.last_traces = traces
        return rows, dict(**summarize(games), per_game=games, environment_seconds=env_seconds,
                         model_seconds=model_seconds, total_player_actions=actions_count, inference_calls=calls,
                         mean_inference_batch=actions_count/max(1,calls))

    def batch(self):
        if self.pool is None: self.pool = self.make_pool(self.config['batch_matches'])
        seeds = self.seed_rng.integers(142000000,143000000,self.pool.count).tolist()
        rows, metric = self.collect(self.pool, seeds); metric.update(self.update(rows))
        self.batches += 1; self.matches += len(seeds); self.actions += len(rows)
        return dict(**metric, batches=self.batches, matches=self.matches, actions=self.actions)

    def evaluate(self, seeds):
        size = min(self.config['batch_matches'],len(seeds)); games = []
        if len(seeds) % size: raise ValueError('evaluation cohort must divide panel')
        with self.make_pool(size) as pool:
            for start in range(0,len(seeds),size):
                _, result = self.collect(pool,seeds[start:start+size],greedy=True,keep_rows=False)
                games.extend(result['per_game'])
        return dict(**summarize(games), per_game=games)

    def save(self, path):
        path = Path(path); tmp = path.with_suffix('.partial.pt')
        torch.save(dict(version=142, config=self.config, model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(), rng=self.rng.get_state(),
            seed_rng=self.seed_rng.bit_generator.state, cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if self.device.startswith('cuda') else [],
            batches=self.batches, matches=self.matches, actions=self.actions,
            elapsed_seconds=self.elapsed_seconds, best_metric=self.best_metric), tmp)
        tmp.replace(path)
        write_json(path.with_suffix('.json'), dict(sha256=digest(path),batches=self.batches,actions=self.actions,
                                                  source_id=self.config.get('source_id')))

    def load(self, path):
        if digest(path) != read_json(Path(path).with_suffix('.json'))['sha256']: raise ValueError('checkpoint digest mismatch')
        state = torch.load(path,map_location=self.device,weights_only=False)
        if state['version'] != 142 or state['config'] != self.config: raise ValueError('exact config/version required')
        self.model.load_state_dict(state['model']); self.optimizer.load_state_dict(state['optimizer'])
        self.rng.set_state(state['rng'].cpu()); self.seed_rng.bit_generator.state = state['seed_rng']
        torch.set_rng_state(state['cpu_rng'].cpu())
        if state['cuda_rng']: torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
        for key in ('batches','matches','actions','elapsed_seconds','best_metric'): setattr(self,key,state[key])


def replay_steps(trace):
    """Yield public (s,a,r,s',done) for offline learning; exhaust to verify terminal."""
    env = HarmoniesMarket(); obs,_ = env.reset(trace['seed']); rewards = 0.
    from .encoding import action_map
    for step in trace['decisions']:
        before = obs; action = action_map(obs)[step['action']]
        old = obs['score_breakdown']['total']
        obs = env.step(action).observation
        reward = (obs['score_breakdown']['total']-old)/150
        if abs(reward-step['reward'])>1e-10: raise ValueError('replay reward mismatch')
        rewards += reward
        yield dict(observation=before, action=action, action_index=step['action'], reward=reward,
                   next_observation=obs, terminated=obs['terminated'], logp=step['logp'], value=step['value'])
    if not obs['terminated'] or obs['score_breakdown']['total'] != trace['score']:
        raise ValueError('replay terminal mismatch')
    if abs(rewards*150-trace['score'])>1e-7: raise ValueError('replay return mismatch')
    env.validate()


def replay(trace):
    """Reconstruct and validate a complete trajectory, returning its terminal observation."""
    last = None
    for last in replay_steps(trace): pass
    if last is None: raise ValueError('empty trajectory')
    return last['next_observation']


def run(config, out, device='cuda', resume=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=resume)
    if resume and (out/'done.json').exists(): raise ValueError('completed trial cannot silently continue')
    if config.get('deadline_unix',float('inf')) <= time.time(): raise ValueError('trial deadline expired')
    write_json(out/'config.json',config); start=time.monotonic(); t=Trainer(config,device)
    if resume:
        t.load(out/'latest.pt')
        for name in ('failed','interrupted'):
            marker=out/(name+'.json')
            if marker.exists(): marker.rename(out/(name+f'.before_resume_{time.time_ns()}.json'))
    prior=t.elapsed_seconds
    stop_at=min(start+max(0,config['max_seconds']-prior),time.monotonic()+config.get('deadline_unix',float('inf'))-time.time())
    stopping=[False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_: stopping.__setitem__(0,True))
    try:
        if not resume:
            initial=t.evaluate(config['development_seeds']); write_json(out/'initial.json',initial)
            export_policy(t.model,out/'initial_policy.pt',config)
            t.best_metric=initial['score_mean']; t.elapsed_seconds=time.monotonic()-start
            t.save(out/'best.pt'); t.save(out/'latest.pt')
            print(json.dumps(dict(event='initial',**{k:v for k,v in initial.items() if k!='per_game'})),flush=True)
        while not stopping[0] and time.monotonic()<stop_at-config['finish_reserve_seconds']:
            tick=time.monotonic(); behavior_sha=digest(out/'latest.pt'); result=t.batch()
            (out/'trajectories').mkdir(exist_ok=True)
            with gzip.open(out/'trajectories'/f'batch_{t.batches:06d}.json.gz','wt') as f:
                json.dump(dict(layout=LAYOUT,rules_version=HarmoniesMarket.rules_version,
                    source_id=config.get('source_id'),behavior_checkpoint_sha256=behavior_sha,
                    initialization=config['initialization'],batch=t.batches,games=t.last_traces),f,separators=(',',':'))
            if t.batches % config['evaluate_interval']==0:
                evaluation=t.evaluate(config['development_seeds']); write_json(out/f'dev_{t.batches:06d}.json',evaluation)
                result['development']={k:v for k,v in evaluation.items() if k!='per_game'}
                if evaluation['score_mean']>t.best_metric:
                    t.best_metric=evaluation['score_mean']; t.elapsed_seconds=prior+time.monotonic()-start
                    t.save(out/'best.pt')
            t.elapsed_seconds=prior+time.monotonic()-start
            result.update(best_development_score=t.best_metric,elapsed_seconds=t.elapsed_seconds,batch_seconds=time.monotonic()-tick)
            t.save(out/'latest.pt')
            with (out/'curve.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='per_game'}),flush=True)
        t.elapsed_seconds=prior+time.monotonic()-start; t.save(out/'final.pt'); t.save(out/'latest.pt')
        export_policy(t.model,out/'final_policy.pt',config)
        if stopping[0]:
            write_json(out/'interrupted.json',dict(elapsed_seconds=t.elapsed_seconds,actions=t.actions)); return
        t.close()
        selected=torch.load(out/'best.pt',map_location=device,weights_only=False)
        t.model.load_state_dict(selected['model']); export_policy(t.model,out/'best_policy.pt',config)
        confirmation=t.evaluate(config['confirmation_seeds'])
        write_json(out/'confirmation.json',dict(**confirmation,selected_checkpoint='best.pt',selection_uses_confirmation=False))
        from .policy import load_policy
        t.model=load_policy(out/'initial_policy.pt',device)
        baseline=t.evaluate(config['confirmation_seeds']); write_json(out/'initial_confirmation.json',baseline)
        delta=paired_delta(confirmation,baseline)
        write_json(out/'done.json',dict(status='completed',wall_seconds=prior+time.monotonic()-start,
            budget_seconds=config['max_seconds'],batches=t.batches,actions=t.actions,
            selected_batch=selected['batches'],best_development_score=t.best_metric,paired_vs_initial=delta,
            confirmation={k:v for k,v in confirmation.items() if k!='per_game'}))
        print(json.dumps(dict(event='completed',**read_json(out/'done.json'))),flush=True)
    except BaseException as exc:
        write_json(out/'failed.json',dict(error=repr(exc),elapsed_seconds=prior+time.monotonic()-start)); raise
    finally: t.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(read_json(a.config),a.out,a.device,a.resume)
