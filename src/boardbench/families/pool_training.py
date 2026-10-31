"""One common, bounded, resumable terminal-win PPO baseline for every family.

A retains the legacy heuristic/prior/current-selfplay opponent mixture; B uses
frozen historical opponents. Network, initialization, public features, reward,
optimizer settings, resource budget and fixed validation panels are identical.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import shutil
import time
import traceback

import numpy as np
import torch

from .neural import NeuralPolicy, collate, finish_returns, load_model, save_model
from .opponent_pool import OpponentPool, pool_seed
from .play import make_policy, policy_seed, run_match, write_json
from .pool_evaluator import (bounded_jobs, evaluate, fingerprint, paired_difference,
                             public_fingerprint_states)
from .ppo import update
from .registry import game_class

STOP = False
MODEL_CACHE = {}


def request_stop(*_):
    global STOP
    STOP = True


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_default(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def write_compressed(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.part')
    with gzip.open(temporary, 'wt', compresslevel=1) as handle:
        json.dump(data, handle, default=json_default, allow_nan=False, separators=(',', ':'))
    temporary.replace(path)


def cached_model(path, cls):
    if path not in MODEL_CACHE:
        if len(MODEL_CACHE) >= 16:
            MODEL_CACHE.pop(next(iter(MODEL_CACHE)))
        MODEL_CACHE[path] = load_model(path, cls)
    return MODEL_CACHE[path]


class RecordingPolicy(NeuralPolicy):
    def __init__(self, *args, identity, player, opponent_id, **kwargs):
        super().__init__(*args, record=True, **kwargs)
        self.identity, self.player, self.opponent_id = identity, player, opponent_id

    def decide(self, observation, actions):
        action = super().decide(observation, actions)
        self.records[-1].update(observation=deepcopy(observation), legal_actions=deepcopy(actions),
            legal_mask=[True]*len(actions), player_id=self.player, policy_version=self.identity,
            opponent_id=self.opponent_id, terminated=False, truncated=False, reward=0.)
        return action


def rollout(job):
    torch.set_num_threads(1)
    cls = game_class(job['game'])
    model, saved = cached_model(job['path'], cls)
    policies, specs, learners = [], [], []
    for seat in (0, 1):
        if seat == job['learner_seat'] or job['self_play']:
            spec = dict(kind='neural', path=job['path'], stochastic=True)
            policy = RecordingPolicy(cls, spec, seed=policy_seed(job['seed'], seat), model=model,
                prior_weights=saved['prior_weights'], identity=job['policy_version'],
                player=seat, opponent_id=job['opponent_id'])
            learners.append(seat)
        else:
            spec = deepcopy(job['opponent'])
            if spec['kind'] == 'neural':
                frozen, metadata = cached_model(spec['path'], cls)
                policy = NeuralPolicy(cls, spec, seed=policy_seed(job['seed'], seat), model=frozen,
                                      prior_weights=metadata['prior_weights'])
            else:
                policy = make_policy(cls, spec, policy_seed(job['seed'], seat))
        policies.append(policy)
        specs.append(spec)
    began = time.time()
    try:
        result = run_match(job['game'], specs, seed=job['seed'], policies=policies,
                           max_seconds=job['max_seconds'], validate=job.get('validate', False),
                           measure_policy_time=True)
    except Exception as e:
        result = dict(game=job['game'], seed=job['seed'], status='error', rewards=None,
                      error=repr(e), traceback=traceback.format_exc(), seconds=time.time()-began)
    records, excluded = [], []
    for seat in learners:
        chain = policies[seat].records
        if not chain:
            continue
        if chain:
            chain[-1]['terminated'] = result['status']=='complete'
            chain[-1]['truncated'] = result['status']!='complete'
        if result['status']=='complete':
            chain[-1]['reward'] = result['rewards'][seat]
            records.extend(finish_returns(chain, result['rewards'][seat], 1., job['gae_lambda']))
        else:
            # This complete-game estimator excludes incomplete chains entirely.
            # It never substitutes a zero terminal value or a fabricated loss.
            excluded.extend(chain)
    result.update(learner_seat=job['learner_seat'], opponent_id=job['opponent_id'],
                  policy_version=job['policy_version'], self_play=job['self_play'])
    return dict(result=result, records=records, excluded_records=excluded)


def audit_behavior(model, rows, device):
    if not rows:
        raise ValueError('No complete learner trajectories')
    errors = []
    for start in range(0, min(len(rows), 64), 16):
        batch = rows[start:start+16]
        with torch.no_grad():
            logits, _ = model(*collate(batch, device))
            actions = torch.tensor([r['action'] for r in batch], device=device)
            actual = torch.log_softmax(logits, -1).gather(1, actions[:, None]).squeeze(1)
            expected = torch.tensor([r['old_logp'] for r in batch], device=device)
            errors.append(float((actual-expected).abs().max()))
        for row in batch:
            if (not row['legal_mask'][row['action']] or len(row['legal_mask'])!=len(row['x'])
                    or len(row['legal_actions'])!=len(row['x'])):
                raise ValueError('Candidate/mask/action alignment failed')
    error = max(errors)
    if not np.isfinite(error) or error > 2e-4:
        raise ValueError(f'Stale or misaligned PPO behavior likelihood: {error}')
    return error


def save_checkpoint(path, model, optimizer, rng, pool, state, config_hash, config):
    payload = dict(format='boardbench_pool_training_v1', model=model.state_dict(),
        optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
        python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        pool=pool.state, state=state, config_hash=config_hash, config=config)
    temporary = Path(path).with_suffix('.part')
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_checkpoint(path, model, optimizer, rng, config_hash):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['config_hash'] != config_hash:
        raise ValueError('Resume configuration/source/parent changed')
    model.load_state_dict(payload['model'])
    optimizer.load_state_dict(payload['optimizer'])
    rng.bit_generator.state = payload['numpy_rng']
    random.setstate(payload['python_rng'])
    torch.set_rng_state(payload['torch_rng'])
    if payload['cuda_rng'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return OpponentPool.restore(payload['pool']), payload['state']


def member(path, identity, decisions=0, **kwargs):
    return dict(id=identity, spec=dict(kind='neural', path=str(path), sha256=file_hash(path)),
                learner_decisions=decisions, quality=None, panel_vector=[], fingerprint=[], **kwargs)


def main(argv=None):
    global STOP
    STOP = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    config_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    if cfg.get('gamma')!=1. or cfg['arm'] not in ('A','B'):
        raise ValueError('The common pilot requires gamma=1 and an explicit A/B arm')
    if min(cfg[k] for k in ('workers','games','max_updates','max_decisions','eval_decisions',
                            'quick_seeds','confirm_seeds','epochs','batch_size'))<1:
        raise ValueError('Counts must be positive')
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    torch.set_num_threads(1)
    torch.manual_seed(cfg['seed'])
    random.seed(cfg['seed'])
    rng = np.random.default_rng(cfg['seed'])
    device = cfg['device']
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA unavailable; no CPU fallback')
        torch.ones(32, device=device).sum().item()
        torch.cuda.synchronize()
    out = Path(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    model, saved = load_model(cfg['initial_path'], game_class(cfg['game']), device)
    if file_hash(cfg['initial_path']) != cfg['initial_sha256']:
        raise ValueError('Initial checkpoint changed')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    initial = member(Path(cfg['initial_path']), 'initial', provenance=cfg['initial_provenance'])
    history = OpponentPool(initial, capacity=12)
    began = time.time()
    state = dict(next_update=0, learner_decisions=0, environment_decisions=0, games=0,
                 deadline=min(cfg['deadline_unix'], began+cfg['hours']*3600), began_unix=began,
                 phase='preflight', best='initial', last='initial', baseline=None,
                 baseline_rows=None, fingerprint_states=[], next_evaluation=cfg['eval_decisions'],
                 costs=dict(rollout_seconds=0., update_seconds=0., evaluation_seconds=0.,
                            io_seconds=0., fingerprint_seconds=0.))
    checkpoint = out/'checkpoint.pt'
    if checkpoint.exists():
        history, state = restore_checkpoint(checkpoint, model, optimizer, rng, config_hash)
        if state['phase'] in ('complete', 'stopped', 'budget_exhausted', 'failed'):
            if state['deadline'] <= time.time() or state['phase'] in ('complete', 'failed'):
                print(json.dumps(dict(already_finished=True, state=state['phase'])), flush=True)
                return
    def persist(phase):
        state['phase'] = phase
        save_checkpoint(checkpoint, model, optimizer, rng, history, state, config_hash, cfg)
        write_json(out/'state.json', {k:v for k, v in state.items()
                   if k not in ('baseline_rows', 'fingerprint_states')})
        write_json(out/'pool.json', history.state)
        best = history.state['members'][state['best']]['spec']
        write_json(out/'best.json', best)
        latest = (dict(kind='neural', path=state['last_snapshot'], sha256=file_hash(state['last_snapshot']))
                  if state.get('last_snapshot') else history.state['members'][state['last']]['spec'])
        write_json(out/'latest.json', latest)
    def log(metric):
        with (out/'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(metric, allow_nan=False)+'\n')
        print(json.dumps(metric, allow_nan=False), flush=True)
    def assess(executor, candidate, count, split, label, deadline, record=False, panel=None):
        summary, rows = evaluate(executor, cfg['game'], candidate, panel or cfg['panel'], count,
            split, min(deadline, state['deadline']-15), cfg['workers'], cfg['max_match_seconds'],
            record=record, stopped=lambda:STOP)
        state['costs']['evaluation_seconds'] += summary['seconds']
        io = time.time()
        write_compressed(out/'evaluation'/f'{label}.json.gz', dict(summary=summary, games=rows))
        state['costs']['io_seconds'] += time.time()-io
        return summary, rows
    model.eval()
    persist(state['phase'])
    try:
        with ProcessPoolExecutor(max_workers=cfg['workers'], mp_context=mp.get_context('spawn')) as executor:
            if not state.get('preflight'):
                jobs = [dict(game=cfg['game'], path=cfg['initial_path'], self_play=False,
                    learner_seat=i%2, seed=pool_seed('smoke', cfg['seed'], index=i),
                    opponent=dict(initial['spec'], stochastic=True), opponent_id='initial',
                    policy_version='initial', max_seconds=cfg['max_match_seconds'],
                    gae_lambda=cfg['gae_lambda'], validate=True) for i in range(2)]
                samples = bounded_jobs(executor, rollout, jobs, state['deadline']-30, cfg['workers'], lambda:STOP)
                write_compressed(out/'preflight.json.gz', samples)
                if len(samples)!=2 or any(x['result']['status']!='complete' for x in samples):
                    raise RuntimeError('Preflight did not complete both seats; inspect preflight.json.gz')
                rows = [r for x in samples for r in x['records']]
                error = audit_behavior(model, rows, device)
                # Verify real gradients and exact optimizer/RNG recovery without
                # changing the common starting model or spending extra PPO steps.
                pre_weights = {k:v.detach().clone() for k,v in model.state_dict().items()}
                update(model, optimizer, rows[:64], device, rng, 1, 32,
                       cfg['entropy_coef'], cfg['prior_kl_coef'])
                changed = any(not torch.equal(pre_weights[k], v) for k,v in model.state_dict().items())
                if not changed:
                    raise RuntimeError('Preflight optimizer did not update weights')
                history, state = restore_checkpoint(checkpoint, model, optimizer, rng, config_hash)
                if any(not torch.equal(pre_weights[k], v) for k,v in model.state_dict().items()):
                    raise RuntimeError('Preflight restore changed initial weights')
                state['preflight'] = dict(passed=True, behavior_max_error=error,
                                         real_optimizer_update=changed, device=device,
                                         elapsed_seconds=time.time()-state['began_unix'])
                persist('baseline_evaluation')
            if state['baseline'] is None:
                budget = min(state['deadline']-time.time()-180, cfg['hours']*3600*.25)
                summary, rows = assess(executor, initial['spec'], cfg['quick_seeds'], 'quick',
                                       'initial', time.time()+max(1., budget), record=True)
                state['baseline'] = summary
                state['fingerprint_states'] = public_fingerprint_states(rows)
                write_compressed(out/'fingerprint_states.json.gz', state['fingerprint_states'])
                # Raw public traces are persisted once; keep only result labels in checkpoint.
                for row in rows:
                    row.pop('trajectory', None)
                state['baseline_rows'] = rows
                start = time.time()
                history.state['members']['initial'].update(
                    quality=summary['quality'] if summary['all_complete'] else None,
                    panel_vector=[x['completed_win_rate'] for x in summary['opponents'].values()
                                  if x['completed_win_rate'] is not None],
                    fingerprint=fingerprint(cfg['game'], initial['spec'], state['fingerprint_states']))
                state['costs']['fingerprint_seconds'] += time.time()-start
                log(dict(event='initial_evaluation', **summary, fingerprint_count=len(state['fingerprint_states'])))
                persist('training')
            reserve = min(900., max(120., state['baseline']['seconds']*1.3))
            for iteration in range(state['next_update'], cfg['max_updates']):
                if STOP or time.time()+reserve+10 >= state['deadline'] or state['learner_decisions']>=cfg['max_decisions']:
                    break
                path = out/'snapshots'/f'behavior_{iteration:05d}.pt'
                save_model(path, model, game_class(cfg['game']), saved['prior_weights'],
                           dict(source_commit=cfg['source_commit'], learner_decisions=state['learner_decisions']))
                ids, probabilities = history.distribution()
                self_games = int(cfg['games']*.25) if cfg['arm']=='A' else 0
                jobs = []
                for i in range(cfg['games']):
                    if cfg['arm']=='B':
                        opponent_id = str(rng.choice(ids, p=probabilities))
                        opponent = dict(history.state['members'][opponent_id]['spec'], stochastic=True)
                    else:
                        opponent_id = 'current_selfplay' if i<self_games else ('heuristic' if i%3==0 else 'prior')
                        opponent = dict(kind='heuristic') if i%3==0 else dict(kind='linear', weights=saved['prior_weights'])
                    jobs.append(dict(game=cfg['game'], path=str(path), self_play=i<self_games,
                        learner_seat=i%2, seed=pool_seed('train', cfg['seed'], iteration, i),
                        opponent=opponent, opponent_id=opponent_id, policy_version=file_hash(path),
                        max_seconds=cfg['max_match_seconds'], gae_lambda=cfg['gae_lambda']))
                start = time.time()
                played = bounded_jobs(executor, rollout, jobs, state['deadline']-reserve, cfg['workers'], lambda:STOP)
                state['costs']['rollout_seconds'] += time.time()-start
                io = time.time()
                write_compressed(out/'trajectories'/f'batch_{iteration:05d}.json.gz',
                    dict(policy_version=file_hash(path), opponent_ids=ids,
                         sampling_probabilities=probabilities.tolist(), games=played))
                state['costs']['io_seconds'] += time.time()-io
                errors = [x for x in played if x['result']['status'] in ('error', 'no_legal_action')]
                if errors:
                    raise RuntimeError('Rollout legality/runtime failure; batch preserved')
                records = [row for x in played for row in x['records']]
                if not records:
                    state['stop_reason'] = 'no_complete_learner_trajectories'
                    break
                behavior_error = audit_behavior(model, records, device)
                start = time.time()
                metric = update(model, optimizer, records, device, rng, cfg['epochs'], cfg['batch_size'],
                                cfg['entropy_coef'], cfg['prior_kl_coef'])
                state['costs']['update_seconds'] += time.time()-start
                if cfg['arm']=='B':
                    history.update_training([(x['result']['opponent_id'],
                        x['result']['rewards'][x['result']['learner_seat']]) for x in played
                        if x['result']['status']=='complete'])
                state['learner_decisions'] += len(records)
                state['environment_decisions'] += sum(x['result'].get('decisions', 0) for x in played)
                state['games'] += len(played)
                state['next_update'] = iteration+1
                identity = f'update_{iteration+1:05d}'
                candidate_path = out/'snapshots'/f'{identity}.pt'
                save_model(candidate_path, model, game_class(cfg['game']), saved['prior_weights'],
                           dict(source_commit=cfg['source_commit'], learner_decisions=state['learner_decisions']))
                # All snapshots are retained; pool entries are added only at validation boundaries.
                state['last_snapshot'] = str(candidate_path)
                state['last_snapshot_id'] = identity
                metric.update(event='update', update=iteration+1, learner_decisions=state['learner_decisions'],
                    environment_decisions=state['environment_decisions'], games=len(played),
                    completed=sum(x['result']['status']=='complete' for x in played),
                    behavior_max_error=behavior_error, active_pool=len(history.state['active']),
                    wall_seconds=time.time()-state['began_unix'], costs=deepcopy(state['costs']), unix=time.time())
                log(metric)
                persist('training')
                if state['learner_decisions'] >= state['next_evaluation']:
                    validate_candidate(executor, cfg, out, state, history, candidate_path, identity,
                                       assess, persist, log, reserve, final=False)
                    state['next_evaluation'] = state['learner_decisions']+cfg['eval_decisions']
                    persist('training')
            if not STOP and state.get('last_snapshot') and time.time()+5 < state['deadline']:
                validate_candidate(executor, cfg, out, state, history, Path(state['last_snapshot']),
                    state['last_snapshot_id'], assess, persist, log, reserve, final=True)
            persist('stopped' if STOP else 'budget_exhausted' if time.time()+reserve >= state['deadline'] else 'complete')
    except Exception as e:
        state['error'] = repr(e)
        state['traceback'] = traceback.format_exc()
        persist('failed')
        raise


def validate_candidate(executor, cfg, out, state, history, path, identity,
                       assess, persist, log, reserve, final=False):
    if identity in history.state['members']:
        return
    candidate = member(path, identity, state['learner_decisions'], source_commit=cfg['source_commit'])
    summary, rows = assess(executor, candidate['spec'], cfg['quick_seeds'], 'quick', identity,
                           state['deadline']-20 if final else min(state['deadline']-reserve, time.time()+reserve))
    start = time.time()
    candidate.update(quality=summary['quality'] if summary['all_complete'] else None,
        panel_vector=[x['completed_win_rate'] for x in summary['opponents'].values()
                      if x['completed_win_rate'] is not None],
        fingerprint=fingerprint(cfg['game'], candidate['spec'], state['fingerprint_states']),
        validation=summary)
    state['costs']['fingerprint_seconds'] += time.time()-start
    difference = paired_difference(rows, state['baseline_rows'])
    promoted = False
    # Confirmation includes re-evaluating the incumbent on the exact same 128
    # fresh dev seeds. A short pilot may lack time; do not silently promote then.
    confirmation_cost = max(30., summary['seconds'])*cfg['confirm_seeds']/cfg['quick_seeds']*2.5
    if (difference['complete'] and difference['interval'][0]>0 and
            time.time()+confirmation_cost+reserve < state['deadline']):
        old = history.state['members'][state['best']]['spec']
        cs, cr = assess(executor, candidate['spec'], cfg['confirm_seeds'], 'confirm',
                        identity+'_confirmation', state['deadline']-reserve)
        bs, br = assess(executor, old, cfg['confirm_seeds'], 'confirm',
                        identity+'_incumbent', state['deadline']-reserve)
        paired = paired_difference(cr, br)
        promoted = bool(cs['all_complete'] and bs['all_complete'] and paired['complete'] and paired['interval'][0]>0)
        candidate['confirmation'] = dict(candidate=cs, incumbent=bs, paired=paired)
    history.add(candidate, confirmed_best=promoted)
    state['last'] = identity
    if promoted:
        state['best'] = identity
    history.select()
    if not final and time.time()+reserve+60 < state['deadline']:
        for old_id in history.audit_members(2):
            panel = {old_id:history.state['members'][old_id]['spec']}
            audit, audit_rows = assess(executor, candidate['spec'], cfg['quick_seeds'], 'audit',
                identity+'_audit_'+old_id, state['deadline']-reserve, panel=panel)
            if (audit['all_complete'] and audit['interval'][1]<.5 and
                    time.time()+max(30., audit['seconds'])*5+reserve < state['deadline']):
                confirm, _ = assess(executor, candidate['spec'], cfg['confirm_seeds'], 'audit',
                    identity+'_counter_'+old_id, state['deadline']-reserve, panel=panel)
                if confirm['all_complete'] and confirm['interval'][1]<.5:
                    history.state['members'][old_id].update(confirmed_counter=True, counter_upper=confirm['interval'][1])
        history.select()
    log(dict(event='validation', candidate=identity, learner_decisions=state['learner_decisions'],
             summary=summary, paired_vs_initial=difference, confirmed_promotion=promoted,
             active_pool=history.state['active']))
    persist('training')
    archive = out/'archive'
    archive.mkdir(exist_ok=True)
    shutil.copyfile(out/'checkpoint.pt', archive/f'{identity}_recovery.pt')


if __name__ == '__main__':
    main()
