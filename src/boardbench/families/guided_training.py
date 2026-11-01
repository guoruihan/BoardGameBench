"""Bounded search-initialized public-state PPO with asynchronous snapshot output.

Only complete, same-player trajectories enter PPO. Expert data enter supervised
losses, never an on-policy ratio. Evaluation is owned by a separate process.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .guided_data import dump_packet, load_packet, targets, reward
from .guided_neural import GuidedActorCritic, GuidedPolicy, load_model, save_model
from .neural import finish_returns
from .play import make_policy, policy_seed, write_json
from .pool_evaluator import bounded_jobs
from .reference_search import with_guard, guarded_spec
from .registry import game_class
from .reward_signals import numeric_leaves
from .rich_observation import PublicTreeEncoder
from .simulation import step_legal, terminated
from .state_io import copy_data

STOP = False
CACHE = {}


def stop(*_):
    global STOP
    STOP = True


def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_learning_rate(cfg,phase):
    value=float(cfg.get(phase+'_learning_rate',cfg.get('learning_rate',2e-4)))
    if not np.isfinite(value) or value<=0:raise ValueError('Positive finite phase learning rate required')
    return value


def behavior_logits(logits,rows):
    temperature=torch.tensor([r.get('sampling_temperature',1.) for r in rows],dtype=logits.dtype,device=logits.device)
    if not torch.isfinite(temperature).all() or not (temperature>0).all():
        raise ValueError('Positive finite behavior temperatures required')
    return logits/temperature[:,None]


def prior_policy_kl(prior_logits, policy_logits):
    """Forward KL on the shared legal support, without probability-zero tests.

    Low temperatures can underflow a valid FP32 probability to zero even though
    its log-probability is finite. Categorical KL then reports infinity. Compute
    directly in log space and exclude only the explicit padded actions.
    """
    legal = prior_logits != -torch.inf
    if not torch.equal(legal, policy_logits != -torch.inf):
        raise ValueError('Prior and policy legal supports differ')
    prior_logp = torch.log_softmax(prior_logits, -1)
    policy_logp = torch.log_softmax(policy_logits, -1)
    difference = prior_logp.masked_fill(~legal, 0.)-policy_logp.masked_fill(~legal, 0.)
    return (prior_logp.exp()*difference).sum(-1)


def teacher_mode(rows, mode='soft'):
    if mode=='soft': return rows
    if mode not in ('hard','selective'):raise ValueError('Unknown expert target mode')
    result=[]
    for row in rows:
        r=dict(row);prior=row['prior'].astype(np.float64)
        probabilities=np.exp(prior-prior.max());probabilities/=probabilities.sum()
        if mode=='selective' and (row['action']==int(prior.argmax()) or row.get('teacher_rounds',0)<4):
            r['teacher']=probabilities.astype(np.float32)
        else:
            r['teacher']=np.zeros(len(prior),np.float32);r['teacher'][row['action']]=1.
        result.append(r)
    return result


def prepare_demos(dataset,cfg):
    scale=cfg.get('anchor_scale',dataset['anchor_scale'])
    if not np.isfinite(scale) or scale<=0:raise ValueError('Positive finite policy anchor scale required')
    factor=scale/dataset['anchor_scale']
    rows=dataset['rows'] if factor==1 else [dict(r,prior=r['prior']*factor) for r in dataset['rows']]
    return teacher_mode(rows,cfg.get('teacher_mode','soft'))


def rollout(job):
    torch.set_num_threads(1)
    game, path = job['game'], job['path']
    if path not in CACHE:
        CACHE.clear(); CACHE[path] = load_model(path, game)
    model, encoder, metadata = CACHE[path]
    cls = game_class(game); env = cls(); env.reset(job['seed'], first_player=0)
    seat = job['seat']
    learner = GuidedPolicy(game, dict(stochastic=True), policy_seed(job['seed'], seat),
                           model, encoder, metadata, record=True)
    policies = [None, None]; policies[seat] = with_guard(cls, learner)
    policies[1-seat] = make_policy(cls, guarded_spec(game, job['opponent']), policy_seed(job['seed'], 1-seat))
    began = time.monotonic(); cpu = time.process_time(); trace = []; count = 0
    while not terminated(env):
        if count>=4000 or time.monotonic()-began>=job['max_seconds']: break
        player = env.current_player; obs = env.observe(player); legal = env.legal_actions(player)
        if not legal: break
        action = policies[player].decide(obs, legal)
        trace.append(dict(player=player, observation=copy_data(obs), legal_actions=copy_data(legal),
                          action=copy_data(action)))
        step_legal(env, legal, legal.index(action)); count += 1
    outcome = env.outcome(); status = 'complete' if outcome['terminated'] else 'truncated'
    winners = outcome['winners']; rewards = [0.,0.] if len(winners)!=1 else [1. if p in winners else -1. for p in (0,1)]
    result = dict(game=game, seed=job['seed'], learner_seat=seat, status=status, outcome=outcome,
                  rewards=rewards if status=='complete' else None, score_breakdown=env.score_breakdown(),
                  decisions=count, seconds=time.monotonic()-began, cpu_seconds=time.process_time()-cpu,
                  model_sha256=job['model_sha256'], opponent=job['opponent'])
    records = learner.records if status=='complete' else []
    if records:
        label = targets(result, seat, metadata['schema'])
        terminal = reward(label, job['reward_mode'], game)
        for row in records: row.update(label)
        finish_returns(records, terminal, 1., job.get('gae_lambda', .95))
    packet = dict(result=result, records=records, trajectory=trace,
                  excluded_decisions=len(learner.records)-len(records))
    sha = dump_packet(job['output'], packet)
    return dict(result=result, packet=job['output'], sha256=sha, records=len(records),
                excluded_decisions=packet['excluded_decisions'])


def batches(rows, batch_size, rng, max_nodes=60000):
    selected = []; size = 0
    for i in rng.permutation(len(rows)):
        row = rows[int(i)]; nodes = len(row['state'])+len(row['actions'])
        if selected and (len(selected)>=batch_size or size+nodes>max_nodes):
            yield selected; selected = []; size = 0
        selected.append(row); size += nodes
    if selected: yield selected


def supervised_loss(model, rows, device, mode, game, *, value_coef=.25, auxiliary_coef=.1):
    if any(not np.isfinite(v) or v<0 for v in (value_coef,auxiliary_coef)):
        raise ValueError('Nonnegative finite supervision coefficients required')
    logits, values, auxiliary = model(rows, device)
    logp = torch.log_softmax(logits, -1)
    teacher = torch.zeros_like(logp)
    for i, row in enumerate(rows): teacher[i, :len(row['teacher'])] = torch.as_tensor(row['teacher'], device=device)
    weights = torch.tensor([r.get('teacher_weight', 1.) for r in rows], device=device)
    ce = (-(teacher*logp.masked_fill(~torch.isfinite(logp), 0.)).sum(-1)*weights).sum()/weights.sum().clamp_min(1.)
    returns = torch.tensor([reward(r, mode, game) for r in rows], device=device)
    value_loss = (values-returns).square().mean()
    labels = torch.as_tensor(np.stack([r['auxiliary_targets'] for r in rows]), device=device)
    mask = torch.as_tensor(np.stack([r['auxiliary_mask'] for r in rows]), device=device)
    aux = ((auxiliary-labels).square()*mask).sum()/mask.sum().clamp_min(1)
    return ce+value_coef*value_loss+auxiliary_coef*aux, dict(bc=float(ce.detach()), value=float(value_loss.detach()), auxiliary=float(aux.detach()))


def behavior_error(model, rows, device):
    errors = []
    with torch.inference_mode():
        for start in range(0, min(32, len(rows)), 4):
            batch = rows[start:start+4]; logits, _, _ = model(batch, device)
            logits=behavior_logits(logits,batch)
            indices = torch.tensor([r['action'] for r in batch], device=device)
            logp = torch.log_softmax(logits, -1).gather(1, indices[:,None]).squeeze(1)
            expected = torch.tensor([r['old_logp'] for r in batch], device=device)
            errors.append(float((logp-expected).abs().max()))
    error = max(errors, default=0.)
    if not np.isfinite(error) or error>5e-4: raise ValueError(f'Stale/misaligned behavior logp: {error}')
    return error


def ppo_update(model, optimizer, rows, demos, cfg, rng, device):
    error = behavior_error(model, rows, device)
    advantages = np.asarray([r['advantage'] for r in rows]); mean, std = advantages.mean(), max(advantages.std(), 1e-6)
    metrics = []; stopped_early = False
    for epoch in range(cfg['epochs']):
        for batch in batches(rows, cfg['batch_size'], rng):
            if STOP or time.time()>=cfg['deadline_unix']-10: stopped_early = True; break
            logits, values, auxiliary = model(batch, device)
            logits=behavior_logits(logits,batch)
            dist = torch.distributions.Categorical(logits=logits)
            chosen = torch.tensor([r['action'] for r in batch], device=device)
            old = torch.tensor([r['old_logp'] for r in batch], device=device)
            actual = dist.log_prob(chosen); ratio = (actual-old).exp()
            adv = torch.tensor([(r['advantage']-mean)/std for r in batch], dtype=torch.float32, device=device)
            policy_loss = -torch.minimum(ratio*adv, ratio.clamp(.8,1.2)*adv).mean()
            returns = torch.tensor([r['return'] for r in batch], dtype=torch.float32, device=device)
            value_loss = (values-returns).square().mean()
            labels = torch.as_tensor(np.stack([r['auxiliary_targets'] for r in batch]), device=device)
            mask = torch.as_tensor(np.stack([r['auxiliary_mask'] for r in batch]), device=device)
            aux = ((auxiliary-labels).square()*mask).sum()/mask.sum().clamp_min(1)
            entropy = dist.entropy().mean()
            loss = policy_loss+.5*value_loss+.1*aux-cfg.get('entropy_coef', .005)*entropy
            prior=torch.full_like(logits,-torch.inf)
            for i,r in enumerate(batch):prior[i,:len(r['prior'])]=torch.as_tensor(r['prior'],device=device)
            anchor_kl=prior_policy_kl(behavior_logits(prior,batch),logits).mean()
            loss=loss+cfg.get('prior_kl_coef',0.)*anchor_kl
            demo_rows = [demos[int(i)] for i in rng.integers(len(demos), size=min(4,len(batch)))]
            demo_loss, _ = supervised_loss(model, demo_rows, device, cfg['reward_mode'], cfg['game'],
                value_coef=cfg.get('replay_value_coef',.25),auxiliary_coef=cfg.get('replay_auxiliary_coef',.1))
            loss = loss+cfg.get('demo_coef', .2)*demo_loss
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite guided PPO loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            kl = float(((ratio-1)-(actual-old)).mean().detach())
            metrics.append(dict(loss=float(loss.detach()), policy=float(policy_loss.detach()),
                                value=float(value_loss.detach()), aux=float(aux.detach()),
                                kl=kl, anchor_kl=float(anchor_kl.detach()), entropy=float(entropy.detach()), grad=float(grad)))
            if kl>.04: stopped_early = True; break
        if stopped_early: break
    result = {k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]} if metrics else {}
    result.update(behavior_error=error, minibatches=len(metrics), early_stop=stopped_early)
    return result


def main():
    global STOP
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', required=True, type=Path)
    a = p.parse_args(); cfg = json.loads(a.config.read_text()); out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, stop)
    torch.set_num_threads(1); torch.manual_seed(cfg['seed']); rng = np.random.default_rng(cfg['seed'])
    device = cfg.get('device', 'cuda')
    if device.startswith('cuda'):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no implicit CPU fallback')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.ones(32, device=device).sum().item(); torch.cuda.synchronize()
    info = json.loads(Path(cfg['dataset']).read_text()); dataset = load_packet(info['path'], info['sha256'])
    vocabulary_path=Path(cfg['dataset']).parent/'vocabulary.json'
    symbols=json.loads(vocabulary_path.read_text())['symbols'] if vocabulary_path.exists() else None
    encoder = PublicTreeEncoder(cfg['game'],symbols=symbols)
    if dataset['encoding']!=encoder.metadata: raise ValueError('Dataset/encoder mismatch')
    demos = prepare_demos(dataset,cfg); metadata = dict(game=cfg['game'], base_spec=dataset['base_spec'],
        schema=dataset['schema'], anchor_scale=cfg.get('anchor_scale',dataset['anchor_scale']), source_commit=cfg['source_commit'],
        sampling_temperature=float(cfg.get('sampling_temperature',1.)))
    if not np.isfinite(metadata['sampling_temperature']) or metadata['sampling_temperature']<=0:
        raise ValueError('Positive finite sampling temperature required')
    model = GuidedActorCritic(encoder.metadata['vocabulary_size'], cfg.get('hidden',64), len(dataset['schema']['names'])).to(device)
    bc_lr,ppo_lr=phase_learning_rate(cfg,'bc'),phase_learning_rate(cfg,'ppo')
    optimizer = torch.optim.Adam(model.parameters(), lr=bc_lr)
    config_hash = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    state = dict(bc_epoch=0, update=0, learner_decisions=0, episodes=0, excluded_episodes=0,
                 phase='bc', extra_loaded=False, started_unix=time.time(), candidates=[])
    checkpoint = out/'checkpoint.pt'
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['config_hash']!=config_hash: raise ValueError('Resume config changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        rng.bit_generator.state=saved['rng']; torch.set_rng_state(saved['torch_rng'])
        state=saved['state']
        for v in optimizer.state.values():
            for k,x in v.items():
                if isinstance(x,torch.Tensor):v[k]=x.to(device)
    elif cfg.get('warm_start'):
        warm, warm_encoder, meta = load_model(cfg['warm_start'], cfg['game'])
        if (meta['base_spec']!=metadata['base_spec'] or meta['schema']!=metadata['schema']
                or warm_encoder.metadata!=encoder.metadata):
            raise ValueError('Warm-start teacher/target schema changed')
        model.load_state_dict(warm.state_dict()); state['phase']='bc' if cfg.get('warm_start_bc') else 'ppo'
    def status(**extra):
        write_json(out/'status.json',dict(**state, observed_unix=time.time(), pid=os.getpid(),
                                       device=device, deadline_unix=cfg['deadline_unix'], **extra))
    def save_state():
        payload=dict(config_hash=config_hash, model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                     optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
                     torch_rng=torch.get_rng_state(), state=state)
        temp=checkpoint.with_suffix('.part');torch.save(payload,temp);temp.replace(checkpoint)
    def publish(name):
        path=out/(name+'.pt'); save_model(path,model,encoder,dict(metadata, phase=state['phase'],
                            update=state['update'], learner_decisions=state['learner_decisions']))
        item=dict(path=str(path.resolve()),sha256=file_hash(path),name=name,observed_unix=time.time(),
                  update=state['update'],decisions=state['learner_decisions'])
        state['candidates'].append(item);write_json(out/'candidates.json',state['candidates']);save_state();status()
    def log(row):
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**row))+'\n')
    write_json(out/'config.json',cfg)
    if not state['candidates']: publish('initial')
    bc_deadline=min(cfg['deadline_unix']-30,state['started_unix']+cfg.get('bc_seconds',600))
    while state['phase']=='bc' and state['bc_epoch']<cfg.get('bc_epochs',8) and time.time()<bc_deadline and not STOP:
        metrics=[]; model.train()
        for batch in batches(demos,cfg['batch_size'],rng):
            if STOP or time.time()>=bc_deadline:break
            loss, row=supervised_loss(model,batch,device,cfg['reward_mode'],cfg['game'])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite imitation loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();metrics.append(dict(row,grad=float(grad)))
        state['bc_epoch']+=1;model.eval()
        log(dict(phase='bc',epoch=state['bc_epoch'],minibatches=len(metrics),
                 metrics={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]} if metrics else {}))
        publish(f'bc_{state["bc_epoch"]:03d}')
    for group in optimizer.param_groups:group['lr']=ppo_lr
    if state['phase']=='bc':state['phase']='ppo';publish('bc_final')
    extra_manifest=Path(cfg['dataset']).parent/'extra_dataset.json'
    def read_extra():
        item=json.loads(extra_manifest.read_text())
        extra=load_packet(item['path'],item['sha256'])
        if (extra['schema']!=dataset['schema'] or extra['encoding']!=encoder.metadata
                or extra['base_spec']!=metadata['base_spec'] or extra['anchor_scale']!=dataset['anchor_scale']):
            raise ValueError('New teacher schema or prior drift')
        return extra
    if state['extra_loaded']: demos+=prepare_demos(read_extra(),cfg)
    model.eval(); save_state()
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as executor:
        while not STOP and time.time()<cfg['deadline_unix']-30 and state['update']<cfg.get('max_updates',10000):
            if (out/'PAUSE').exists(): state['phase']='paused';break
            if not state['extra_loaded'] and extra_manifest.exists():
                extra=read_extra()
                demos+=prepare_demos(extra,cfg);state['extra_loaded']=True;log(dict(phase='data',new_decisions=len(extra['rows'])))
            current=out/f'behavior_{state["update"]:05d}.pt';save_model(current,model,encoder,metadata)
            digest=file_hash(current);jobs=[]
            for i in range(cfg['games']):
                seed=173200000+cfg['seed']*100000+state['update']*cfg['games']+i
                # Environment/seat allocation is explicit; both seats are trained.
                seat=(i+state['update'])%2
                opponent=metadata['base_spec']
                if i%4==0 and state['update']>0:
                    opponent=guarded_spec(cfg['game'],dict(kind='guided_rich',path=str(current.resolve()),sha256=digest))
                jobs.append(dict(game=cfg['game'],path=str(current.resolve()),model_sha256=digest,seed=seed,
                    seat=seat,opponent=opponent,reward_mode=cfg['reward_mode'],gae_lambda=cfg.get('gae_lambda',.95),
                    max_seconds=cfg.get('max_match_seconds',180.),
                    output=str((out/'rollouts'/f'u{state["update"]:05d}_g{i:03d}.pkl.z').resolve())))
            began=time.time();status(stage='sampling')
            completed=bounded_jobs(executor,rollout,jobs,cfg['deadline_unix']-15,cfg['workers'],lambda:STOP)
            records=[]
            for item in completed:
                records+=load_packet(item['packet'],item['sha256'])['records']
            write_json(out/'batches'/f'{state["update"]:05d}.json',completed)
            state['episodes']+=len(completed);state['excluded_episodes']+=sum(x['result']['status']!='complete' for x in completed)
            if not records:log(dict(phase='empty',update=state['update'],games=len(completed)));state['update']+=1;save_state();continue
            sampling_seconds=time.time()-began;status(stage='updating');began=time.time();model.train()
            metrics=ppo_update(model,optimizer,records,demos,cfg,rng,device);model.eval()
            state['learner_decisions']+=len(records);state['update']+=1
            complete=[x['result'] for x in completed if x['result']['status']=='complete']
            log(dict(phase='ppo',update=state['update'],decisions=state['learner_decisions'],
                batch_decisions=len(records),sampling_seconds=sampling_seconds,update_seconds=time.time()-began,
                train_win=float(np.mean([.5+.5*x['rewards'][x['learner_seat']] for x in complete])),
                metrics=metrics))
            publish(f'ppo_{state["update"]:05d}')
    state['phase']='stopped' if STOP else ('paused' if state['phase']=='paused' else 'complete')
    publish('final');status();save_state()


if __name__=='__main__':main()
