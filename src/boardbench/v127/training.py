"""Natural-terminal PPO with explicit imitation import and on-policy critic warmup.

Reuses the V121 public-observation environment pool, V12 encoding/distribution,
and compact actor-critic. No search or expert labels enter PPO returns.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v12.neural import CompactSolver
from boardbench.v12.training import policy_distribution, returns
from boardbench.v121.collector import EnvironmentPool, REWARD, SCORE, ANIMAL, COMPLETED, NATURAL, STOPPED
from boardbench.v121.registry import validate_manifest


def advantages(rewards, values, lam=.95):
    """Gamma=1 GAE, natural-terminal bootstrap=0; lambda1 equals MC minus V."""
    out = np.zeros(len(rewards), np.float64)
    carry = 0.
    for i in reversed(range(len(rewards))):
        nxt = values[i+1] if i+1 < len(values) else 0.
        carry = rewards[i] + nxt - values[i] + lam * carry
        out[i] = carry
    return out, out + np.asarray(values)


def load_demonstrations(config, manifest):
    if digest(config['dataset']) != config['dataset_sha256']:
        raise ValueError('demonstration hash mismatch')
    with np.load(config['dataset'], allow_pickle=False) as archive:
        data = {k: archive[k].copy() for k in ('x','mask','y','seed','validation')}
    if data['x'].shape[1] != 1346 or data['mask'].shape[1] != 240:
        raise ValueError('demonstration encoding mismatch')
    hold = data['validation'].astype(bool)
    fit, val = set(data['seed'][~hold]), set(data['seed'][hold])
    if not fit or not val or fit & val or not (fit | val) <= set(manifest['sets']['train']):
        raise ValueError('demonstration split leakage')
    if not data['mask'][np.arange(len(data['y'])), data['y']].all():
        raise ValueError('illegal demonstration label')
    if not np.isfinite(data['x']).all():
        raise ValueError('nonfinite demonstration')
    return {k:v[~hold] for k,v in data.items()}, val


class Trainer:
    def __init__(self, config, device='cpu'):
        self.config, self.device = deepcopy(config), str(device)
        if config['research_version'] != '1.2.7' or config['reward_scale'] != 150.:
            raise ValueError('unsupported objective or version')
        if config['hidden'] != 256 and not config.get('calibration'):
            raise ValueError('first experiment requires matched256-wide networks')
        self.source = source_id()
        manifest = read_json(config['split_manifest'])
        validate_manifest(manifest)
        if digest(config['split_manifest']) != config['split_sha256']:
            raise ValueError('split file hash mismatch')
        self.demo, heldout = load_demonstrations(config, manifest)
        self.manifest = manifest
        # Never let online updates consume imitation validation/selection games.
        forbidden = heldout | set(config.get('excluded_train_seeds', []))
        self.train_seeds = [s for s in manifest['sets']['train'] if s not in forbidden]
        if not self.train_seeds:
            raise ValueError('empty online training pool')
        for key in ('curve_seeds','final_seeds'):
            if not config[key] or not set(config[key]) <= set(manifest['sets']['development']):
                raise ValueError('evaluation must use development only')
        if set(config['curve_seeds']) & set(config['final_seeds']):
            raise ValueError('curve/final development overlap')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        self.rng = np.random.default_rng(config['seed'])
        self.solver = CompactSolver(config['seed'], hidden=config['hidden'], allow_stop=False, device=device)
        self.model = self.solver.model.double()  # Historical mature-policy consistency fix.
        self.lineage = {'kind':'random_initialization'}
        if config.get('initial_checkpoint'):
            path = Path(config['initial_checkpoint'])
            if digest(path) != config['initial_sha256']:
                raise ValueError('initial actor hash mismatch')
            metadata = read_json(path.parent/'summary.json')
            parent = read_json(path.parent/'config.json')
            if (metadata['checkpoint_sha256'] != config['initial_sha256'] or
                parent['dataset_sha256'] != config['dataset_sha256'] or
                metadata['hidden'] != config['hidden'] or not metadata['parameter_changed']):
                raise ValueError('imitation provenance mismatch')
            with np.load(path, allow_pickle=False) as weights:
                state = {k:torch.as_tensor(weights[k].copy(), device=device) for k in weights.files}
            if any(not torch.isfinite(v).all() for v in state.values()):
                raise ValueError('nonfinite initial model')
            self.model.load_state_dict(state, strict=True)
            self.lineage = {'kind':'v126_imitation_weight_import', 'checkpoint':str(path),
                            'sha256':config['initial_sha256'], 'parent_source_id':metadata['source_id'],
                            'optimizer':'fresh Adam; no expert optimizer or RNG imported'}
        # The V126 critic never trained. Reset explicitly for ALL arms.
        torch.nn.init.zeros_(self.model.critic.weight)
        torch.nn.init.zeros_(self.model.critic.bias)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.critic_optimizer = torch.optim.Adam(self.model.critic.parameters(), lr=.001)
        self.episodes = self.actions = self.ppo_actions = self.batches = self.warm_batches = self.updates = 0
        self.wall_seconds = 0.
        self.best_score = -float('inf')
        self.evaluated = []
        self.initial_scores = None
        self.pool = None
        self.boundary = True

    def collect(self, pool, seeds, greedy=False, keep_rows=True):
        if len(seeds) != pool.count:
            raise ValueError('one environment seed per slot')
        arrays = pool.reset(seeds)
        trajectories = [[] for _ in seeds]
        calls = decisions = 0
        env_seconds = model_seconds = 0.
        while not arrays['done'].all():
            slots = np.flatnonzero(~arrays['done'])
            x = arrays['features'][slots].copy()
            mask = arrays['masks'][slots, :240].copy()  # STOP never an available policy action.
            start = time.monotonic()
            with torch.no_grad():
                logits, value = self.model(torch.as_tensor(x,device=self.device,dtype=torch.float64),
                                          torch.as_tensor(mask,device=self.device))
                dist = policy_distribution(logits)
                chosen = logits.argmax(-1) if greedy else dist.sample()
                lp = dist.log_prob(chosen).cpu().numpy()
                val = value.cpu().numpy(); action = chosen.cpu().numpy()
            model_seconds += time.monotonic()-start
            start = time.monotonic(); pool.step(slots,action); env_seconds += time.monotonic()-start
            for j,slot in enumerate(slots):
                if keep_rows:
                    trajectories[slot].append((x[j],mask[j],int(action[j]),float(lp[j]),
                                               float(arrays['stats'][slot,REWARD]),float(val[j])))
            decisions += len(slots); calls += 1
        stats = arrays['stats'].copy()
        if stats[:,STOPPED].any() or not stats[:,NATURAL].all():
            raise RuntimeError('non-natural terminal in PPO or evaluation')
        rows = []
        for trajectory in trajectories:
            rewards = [r[4] for r in trajectory]; vals = [r[5] for r in trajectory]
            adv,target = advantages(rewards,vals,self.config['gae_lambda'])
            mc = returns(rewards)
            for i,r in enumerate(trajectory):
                rows.append((*r[:4],float(target[i]),float(adv[i]),float(mc[i])))
        if keep_rows:
            for trajectory,stat in zip(trajectories,stats):
                if abs(sum(r[4] for r in trajectory)*150.-stat[SCORE]) > 1e-6:
                    raise RuntimeError('reward does not telescope to true score')
        games = [{'environment_seed':int(s),'score':float(v[SCORE]),'animal_score':float(v[ANIMAL]),
                  'completed_cards':float(v[COMPLETED]),'natural_terminal':True} for s,v in zip(seeds,stats)]
        return rows, {'games':games,'actions':decisions,'mean_score':float(stats[:,SCORE].mean()),
                      'animal_mean':float(stats[:,ANIMAL].mean()),'completed_cards':float(stats[:,COMPLETED].mean()),
                      'environment_seconds':env_seconds,'model_seconds':model_seconds,
                      'mean_inference_batch':decisions/calls}

    def tensors(self, rows):
        return (torch.as_tensor(np.stack([r[0] for r in rows]),device=self.device,dtype=torch.float64),
                torch.as_tensor(np.stack([r[1] for r in rows]),device=self.device),
                torch.tensor([r[2] for r in rows],device=self.device),
                *[torch.tensor([r[k] for r in rows],device=self.device,dtype=torch.float64) for k in range(3,7)])

    def calibrate_critic(self, rows):
        x,mask,_,_,_,_,mc = self.tensors(rows)
        with torch.no_grad():
            z = self.model.body(x).detach()
            before = float((self.model.critic(z).squeeze(-1)-mc).square().mean())
        for _ in range(self.config.get('critic_epochs',8)):
            for ids in torch.randperm(len(rows),device=self.device).split(self.config['minibatch']):
                loss = (self.model.critic(z[ids]).squeeze(-1)-mc[ids]).square().mean()
                self.critic_optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.critic.parameters(),.5,error_if_nonfinite=True)
                self.critic_optimizer.step()
        with torch.no_grad():
            after = float((self.model.critic(z).squeeze(-1)-mc).square().mean())
        return {'critic_mse_before':before,'critic_mse_after':after}

    def update(self, rows):
        x,mask,action,old,target,adv,_ = self.tensors(rows)
        if not mask.gather(1,action[:,None]).all():
            raise ValueError('action outside behavior mask')
        adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        with torch.no_grad():
            logits,values = self.model(x,mask)
            errors = (policy_distribution(logits).log_prob(action)-old).exp()-1
            ratio_error = float(errors.abs().max())
            if not torch.isfinite(errors).all() or ratio_error > 1e-4:
                raise RuntimeError(f'behavior probability mismatch before update: {ratio_error}')
            variance = float(target.var(unbiased=False))
            explained = 1-float((target-values).var(unbiased=False))/max(variance,1e-12)
        anchor = self.config.get('bc_weight',0.)*max(0.,1-self.ppo_actions/self.config.get('bc_decay_actions',500000))
        metrics = []; stop = False
        actor_before = self.model.actor.weight.detach().clone()
        critic_before = self.model.critic.weight.detach().clone()
        for _ in range(self.config['epochs']):
            for ids in torch.randperm(len(rows),device=self.device).split(self.config['minibatch']):
                logits,value = self.model(x[ids],mask[ids]); dist = policy_distribution(logits)
                log_ratio = dist.log_prob(action[ids])-old[ids]; ratio=log_ratio.exp()
                kl = ((ratio-1)-log_ratio).mean()
                if float(kl.detach()) > self.config['target_kl']:
                    stop=True;break
                policy = -torch.minimum(ratio*adv[ids],ratio.clamp(.8,1.2)*adv[ids]).mean()
                value_loss = (value-target[ids]).square().mean(); entropy=dist.entropy().mean()
                auxiliary = torch.zeros((),device=self.device,dtype=torch.float64)
                if anchor:
                    d = self.rng.integers(len(self.demo['y']),size=min(256,len(self.demo['y'])))
                    dl,_ = self.model(torch.as_tensor(self.demo['x'][d],device=self.device,dtype=torch.float64),
                                      torch.as_tensor(self.demo['mask'][d],device=self.device))
                    auxiliary = F.cross_entropy(dl,torch.as_tensor(self.demo['y'][d],device=self.device))
                loss = policy+.5*value_loss-self.config['entropy_coef']*entropy+anchor*auxiliary
                if not torch.isfinite(loss): raise RuntimeError('nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),.5,error_if_nonfinite=True)
                self.optimizer.step();self.updates+=1
                metrics.append([float(v.detach()) for v in (policy,value_loss,entropy,kl,norm,auxiliary)])
            if stop: break
        if not metrics: raise RuntimeError('PPO performed no optimizer update')
        return {**dict(zip(('policy_loss','value_loss','entropy','approx_kl','gradient_norm','bc_loss'),np.mean(metrics,axis=0).tolist())),
                'bc_weight':anchor,'initial_ratio_max_error':ratio_error,'explained_variance':explained,
                'kl_early_stop':stop,'optimizer_steps':len(metrics),
                'actor_change_l2':float((self.model.actor.weight.detach()-actor_before).norm()),
                'critic_change_l2':float((self.model.critic.weight.detach()-critic_before).norm())}

    def batch(self):
        if not self.boundary: raise RuntimeError('cannot reuse failed partial rollout')
        if self.pool is None:
            self.pool=EnvironmentPool(self.config['batch_episodes'],self.config['workers'])
        self.boundary=False
        seeds=[self.train_seeds[(4096+self.episodes+i)%len(self.train_seeds)] for i in range(self.pool.count)]
        start=time.monotonic();rows,metrics=self.collect(self.pool,seeds)
        warm=self.warm_batches<self.config['critic_batches']
        tick=time.monotonic();update=self.calibrate_critic(rows) if warm else self.update(rows)
        metrics['update_seconds']=time.monotonic()-tick
        self.episodes+=len(seeds);self.actions+=metrics['actions']
        if warm: self.warm_batches+=1
        else: self.ppo_actions+=metrics['actions'];self.batches+=1
        self.boundary=True
        return {**metrics,**update,'phase':'critic_warmup' if warm else 'ppo','episodes':self.episodes,
                'total_actions':self.actions,'ppo_actions':self.ppo_actions,'batches':self.batches,
                'updates':self.updates,'batch_seconds':time.monotonic()-start}

    def evaluate(self,seeds):
        # Evaluation is greedy and uses no sampling RNG or optimizer operations.
        with EnvironmentPool(len(seeds),workers=0) as pool:
            _,metrics=self.collect(pool,seeds,greedy=True,keep_rows=False)
        return metrics

    def save(self,path):
        if not self.boundary: raise RuntimeError('checkpoint only at complete update boundary')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        state={'version':127,'source_id':self.source,'config':self.config,'device':self.device,
               'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
               'critic_optimizer':self.critic_optimizer.state_dict(),'lineage':self.lineage,
               'rng_numpy':self.rng.bit_generator.state,'rng_torch':torch.get_rng_state(),
               'rng_cuda':torch.cuda.get_rng_state_all() if self.device.startswith('cuda') else [],
               'counters':{k:getattr(self,k) for k in ('episodes','actions','ppo_actions','batches','warm_batches','updates',
                                                     'wall_seconds','best_score','evaluated','initial_scores')}}
        tmp=path.with_suffix('.partial');torch.save(state,tmp);tmp.replace(path)
        write_json(path.with_suffix('.json'),{'sha256':digest(path),'source_id':self.source,
            'ppo_actions':self.ppo_actions,'episodes':self.episodes,'lineage':self.lineage})

    def load(self,path):
        if digest(path)!=read_json(Path(path).with_suffix('.json'))['sha256']:
            raise ValueError('resume hash mismatch')
        state=torch.load(path,map_location=self.device,weights_only=False)
        if (state['version']!=127 or state['source_id']!=self.source or state['config']!=self.config or
            state['device']!=self.device): raise ValueError('resume config/source/device mismatch')
        self.model.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer']);self.lineage=state['lineage']
        self.rng.bit_generator.state=state['rng_numpy'];torch.set_rng_state(state['rng_torch'].cpu())
        if state['rng_cuda']: torch.cuda.set_rng_state_all([s.cpu() for s in state['rng_cuda']])
        for k,v in state['counters'].items(): setattr(self,k,v)

    def close(self):
        if self.pool is not None: self.pool.close();self.pool=None


def run(config,out,device='cuda'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists() and read_json(out/'config.json')!=config:
        raise ValueError('immutable run config mismatch')
    write_json(out/'config.json',config)
    if (out/'summary.json').exists() and read_json(out/'summary.json')['status']=='completed':
        summary=read_json(out/'summary.json')
        if summary['source_id']!=source_id() or digest(out/'final.pt')!=read_json(out/'final.json')['sha256']:
            raise ValueError('completed run source/checkpoint mismatch')
        return summary
    started=time.monotonic();trainer=Trainer(config,device)
    stopped=[False]
    prior={sig:signal.signal(sig,lambda *_:stopped.__setitem__(0,True)) for sig in (signal.SIGINT,signal.SIGTERM)}
    prior_wall=0.
    if (out/'latest.pt').exists(): trainer.load(out/'latest.pt');prior_wall=trainer.wall_seconds
    def elapsed(): return prior_wall+time.monotonic()-started
    def publish(phase,**extra):
        write_json(out/'progress.json',{'phase':phase,'pid':os.getpid(),'updated_unix':time.time(),
            'device':device,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'ppo_actions':trainer.ppo_actions,
            'episodes':trainer.episodes,'updates':trainer.updates,'wall_seconds':elapsed(),**extra})
    def checkpoint(path='latest.pt'):
        trainer.wall_seconds=elapsed();trainer.save(out/path)
    def evaluation(label,seeds):
        publish('evaluating',label=label)
        metric=trainer.evaluate(seeds)
        metric.update(label=label,ppo_actions=trainer.ppo_actions,wall_seconds=elapsed(),policy='greedy_model_only')
        if trainer.initial_scores is not None and not label.startswith('final'):
            baseline={g['environment_seed']:g['score'] for g in trainer.initial_scores}
            gaps=[g['score']-baseline[g['environment_seed']] for g in metric['games']]
            metric['paired_gain_from_frozen_initial']=float(np.mean(gaps))
        write_json(out/'evaluations'/f'{label}.json',metric)
        return metric
    try:
        if trainer.initial_scores is None:
            checkpoint('initial.pt')
            trainer.initial_scores=evaluation('initial',config['curve_seeds'])['games']
            trainer.best_score=float(np.mean([g['score'] for g in trainer.initial_scores]))
            checkpoint('best.pt');checkpoint()
        with (out/'training.jsonl').open('a') as log:
            while (trainer.ppo_actions<config['target_actions'] and elapsed()<config['max_seconds'] and not stopped[0]):
                publish('collecting_and_updating')
                row=trainer.batch();row['wall_seconds']=elapsed();row['timestamp']=time.time()
                checkpoint()
                log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                publish(row['phase'],metrics={k:v for k,v in row.items() if k!='games'})
                print(json.dumps({k:v for k,v in row.items() if k!='games'}),flush=True)
                due=[n for n in config['milestones'] if n<=trainer.ppo_actions and n not in trainer.evaluated]
                if due and not stopped[0] and elapsed()<config['max_seconds']:
                    m=evaluation(f'actions_{trainer.ppo_actions}',config['curve_seeds'])
                    if m['mean_score']>trainer.best_score:
                        trainer.best_score=m['mean_score'];checkpoint('best.pt')
                    trainer.evaluated.extend(due);checkpoint(f'milestone_{max(due)}.pt');checkpoint()
        status='interrupted' if stopped[0] else 'completed' if trainer.ppo_actions>=config['target_actions'] else 'time_limit'
        final_comparison={}
        if status=='completed' and elapsed()<config['max_seconds']:
            final_comparison['latest']=evaluation('final',config['final_seeds'])
            # Freeze the best checkpoint chosen on curve games BEFORE the fresh panel.
            latest={k:v.detach().clone() for k,v in trainer.model.state_dict().items()}
            try:
                for name,checkpoint_name in [('initial','initial.pt'),('best','best.pt')]:
                    if elapsed()>=config['max_seconds'] or stopped[0]: break
                    saved=torch.load(out/checkpoint_name,map_location=device,weights_only=False)
                    trainer.model.load_state_dict(saved['model'])
                    final_comparison[name]=evaluation('final_'+name,config['final_seeds'])
                    final_comparison[name]['checkpoint_ppo_actions']=saved['counters']['ppo_actions']
            finally:
                trainer.model.load_state_dict(latest)
            if 'best' in final_comparison and 'initial' in final_comparison:
                a={g['environment_seed']:g['score'] for g in final_comparison['initial']['games']}
                gaps=np.array([g['score']-a[g['environment_seed']] for g in final_comparison['best']['games']])
                half=1.96*float(gaps.std(ddof=1))/np.sqrt(len(gaps)) if len(gaps)>1 else None
                final_comparison['paired_best_minus_initial']={'n':len(gaps),'mean':float(gaps.mean()),
                    'descriptive_normal_95_ci':[float(gaps.mean()-half),float(gaps.mean()+half)] if half is not None else None}
            write_json(out/'final_comparison.json',final_comparison)
        checkpoint('final.pt');checkpoint()
        summary={'status':status,'ppo_actions':trainer.ppo_actions,'total_actions':trainer.actions,
                 'episodes':trainer.episodes,'updates':trainer.updates,'source_id':trainer.source,
                 'wall_seconds':elapsed(),'best_curve_score':trainer.best_score,'lineage':trainer.lineage,
                 'target_overshoot_actions':max(0,trainer.ppo_actions-config['target_actions']),
                 'device':device,'note':'model alone; score optimization, no search; development selection, not formal test'}
        write_json(out/'summary.json',summary);publish(status)
        return summary
    except BaseException as exc:
        publish('failed',error=repr(exc));raise
    finally:
        trainer.close()
        for sig,handler in prior.items(): signal.signal(sig,handler)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cuda')
    a=p.parse_args();run(read_json(a.config),a.out,a.device)
