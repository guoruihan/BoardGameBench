"""Complete-game PPO from a frozen heuristic/learned prior, with paired gates.

Independent CPU actors simulate games; a GPU learner batches actor/critic
updates. Each seat's decision chain has its own return computation. Incomplete
games are recorded and excluded from gradient targets, never labeled losses.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from . import train
from .neural import CandidateActorCritic, NeuralPolicy, collate, finish_returns, load_model, save_model
from .play import make_policy, policy_seed, run_match, write_json
from .registry import game_class

STOP = False
_CACHE = {}


def request_stop(*_):
    global STOP
    STOP = True


def rollout(job):
    """Worker sees privileged engine internally; each policy gets only its view."""
    torch.set_num_threads(1)
    cls = game_class(job['game'])
    path = job['path']
    if path not in _CACHE:
        _CACHE.clear()
        _CACHE[path] = load_model(path, cls)
    model, saved = _CACHE[path]
    policies, learner_seats = [], []
    specs = []
    for seat in (0, 1):
        if job['self_play'] or seat == job['learner_seat']:
            spec = dict(kind='neural', path=path, stochastic=True)
            policies.append(NeuralPolicy(cls, spec, policy_seed(job['seed'], seat),
                            model, saved['prior_weights'], record=True))
            learner_seats.append(seat)
        else:
            spec = job['opponent']
            policies.append(make_policy(cls, spec, policy_seed(job['seed'], seat)))
        specs.append(spec)
    result = run_match(job['game'], specs, seed=job['seed'], policies=policies,
                       max_seconds=job['max_seconds'], record=job.get('record', False))
    records = []
    if result['status'] == 'complete':
        for seat in learner_seats:
            records += finish_returns(policies[seat].records, result['rewards'][seat],
                                      job.get('gamma', .995), job.get('gae_lambda', .95))
    return result, records


def minibatches(records, batch_size, rng):
    order = rng.permutation(len(records))
    i = 0
    while i < len(order):
        selected, maximum = [], 1
        while i < len(order) and len(selected) < batch_size:
            record = records[order[i]]
            size = max(maximum, len(record['x']))
            if selected and size*(len(selected)+1) > 65536:
                break
            selected.append(record)
            maximum = size
            i += 1
        yield selected


def update(model, optimizer, records, device, rng, epochs=4, batch_size=64,
           entropy_coef=.01, prior_kl_coef=.01):
    advantages = np.asarray([r['advantage'] for r in records])
    mean, std = advantages.mean(), max(advantages.std(), 1e-6)
    losses, norms, kls, entropies = [], [], [], []
    model.train()
    for _ in range(epochs):
        for batch in minibatches(records, batch_size, rng):
            x, prior, mask = collate(batch, device)
            logits, values = model(x, prior, mask)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = torch.tensor([r['action'] for r in batch], device=device)
            if not mask.gather(1, actions[:, None]).all():
                raise ValueError('Recorded action outside legal mask')
            old_logp = torch.tensor([r['old_logp'] for r in batch], device=device)
            advantage = torch.tensor([(r['advantage']-mean)/std for r in batch], dtype=torch.float32, device=device)
            targets = torch.tensor([r['return'] for r in batch], dtype=torch.float32, device=device)
            logp = distribution.log_prob(actions)
            ratio = torch.exp(logp-old_logp)
            policy_loss = -torch.minimum(ratio*advantage, torch.clamp(ratio, .8, 1.2)*advantage).mean()
            entropy = distribution.entropy().mean()
            anchor = torch.distributions.Categorical(logits=prior.masked_fill(~mask, -torch.inf))
            prior_kl = torch.distributions.kl_divergence(anchor, distribution).mean()
            loss = (policy_loss + .5*(values-targets).square().mean()
                    - entropy_coef*entropy + prior_kl_coef*prior_kl)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite PPO objective')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
            norms.append(float(norm))
            kls.append(float(((ratio-1)-(logp-old_logp)).mean().detach()))
            entropies.append(float(entropy.detach()))
    model.eval()
    return dict(loss=float(np.mean(losses)), grad_norm=float(np.mean(norms)),
                approximate_kl=float(np.mean(kls)), entropy=float(np.mean(entropies)),
                decisions=len(records), return_mean=float(np.mean([r['return'] for r in records])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--game', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--prior', type=Path)
    p.add_argument('--warm-start', type=Path)
    p.add_argument('--workers', type=int, default=7)
    p.add_argument('--games', type=int, default=32)
    p.add_argument('--updates', type=int, default=512)
    p.add_argument('--epochs', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--eval-every', type=int, default=16)
    p.add_argument('--gate-seeds', type=int, default=12)
    p.add_argument('--confirm-seeds', type=int, default=64)
    p.add_argument('--seed', type=int, default=1601)
    p.add_argument('--gamma', type=float, default=.995)
    p.add_argument('--gae-lambda', type=float, default=.95)
    p.add_argument('--self-play-rate', type=float, default=.25)
    p.add_argument('--learning-rate', type=float, default=3e-4)
    p.add_argument('--entropy-coef', type=float, default=.01)
    p.add_argument('--prior-kl-coef', type=float, default=.01)
    p.add_argument('--device', default='cuda')
    p.add_argument('--hours', type=float, default=3.)
    p.add_argument('--deadline-unix', type=float, default=float('inf'))
    p.add_argument('--max-match-seconds', type=float, default=120.)
    p.add_argument('--source-commit', default='unrecorded')
    args = p.parse_args(argv)
    if min(args.workers, args.games, args.updates, args.epochs, args.batch_size,
           args.eval_every, args.gate_seeds, args.confirm_seeds) < 1:
        raise ValueError('Training and evaluation counts must be positive')
    if not (0 <= args.gamma <= 1 and 0 <= args.gae_lambda <= 1 and 0 <= args.self_play_rate <= 1
            and args.learning_rate > 0 and args.entropy_coef >= 0 and args.prior_kl_coef >= 0):
        raise ValueError('Invalid learning parameters')
    if (args.output/'confirmation_summary.json').exists():
        print('Run already confirmed; use a new output directory for new training.', flush=True)
        return
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, request_stop)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA explicitly required but unavailable')
        # Real synchronized CUDA execution, not nvidia-smi alone.
        torch.ones(32, device=args.device).sum().item()
        torch.cuda.synchronize()
    args.output.mkdir(parents=True, exist_ok=True)
    cls = game_class(args.game)
    prior_spec = json.loads(args.prior.read_text()) if args.prior else dict(kind='heuristic', weights=cls.default_weights)
    warm_model = None
    if args.warm_start:
        warm_model, saved_warm = load_model(args.warm_start, cls)
        if args.prior and prior_spec['weights'] != saved_warm['prior_weights']:
            raise ValueError('Warm model and explicitly supplied prior disagree')
        prior_spec = dict(kind='linear', weights=saved_warm['prior_weights'])
    make_policy(cls, prior_spec)  # Validate artifact identity before generating any game.
    weights = prior_spec['weights']
    model = (warm_model or CandidateActorCritic()).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    began = time.time()
    deadline = min(began+args.hours*3600, args.deadline_unix)
    best_path = args.output/'initial.pt'
    start_update = 0
    checkpoint = args.output/'checkpoint.pt'
    hyperparameters = {k: getattr(args, k) for k in (
        'gamma', 'gae_lambda', 'self_play_rate', 'learning_rate', 'entropy_coef',
        'prior_kl_coef', 'epochs', 'batch_size', 'games', 'seed')}
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['source_commit'] != args.source_commit or saved['game'] != args.game:
            raise ValueError('Resume requires matching game/source')
        if saved.get('hyperparameters', hyperparameters) != hyperparameters:
            raise ValueError('Resume learning parameters changed; start a new run instead')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(args.device)
        rng.bit_generator.state = saved['rng']
        torch.set_rng_state(saved['torch_rng'])
        start_update, best_path = saved['next_update'], Path(saved['best_path'])
        deadline = min(deadline, saved['deadline'])
        if saved['prior_weights'] != weights:
            raise ValueError('Frozen prior changed on resume')
    else:
        save_model(best_path, model, cls, weights, dict(source_commit=args.source_commit, update=-1))
    model.eval()
    write_json(args.output/'config.json', {**vars(args), 'output':str(args.output),
               'prior':str(args.prior) if args.prior else None, 'deadline_unix':deadline,
               'warm_start':str(args.warm_start) if args.warm_start else None,
               'prior_spec':prior_spec, 'pid':os.getpid(), 'rollout_device':'cpu', 'learner_device':args.device})
    def save(next_update, phase, metric=None):
        temporary = checkpoint.with_suffix('.part')
        torch.save(dict(game=args.game, source_commit=args.source_commit,
                   model=model.state_dict(), optimizer=optimizer.state_dict(),
                   rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   next_update=next_update, best_path=str(best_path), deadline=deadline,
                   prior_weights=weights, hyperparameters=hyperparameters), temporary)
        temporary.replace(checkpoint)
        write_json(args.output/'state.json', dict(game=args.game, next_update=next_update,
                   best_path=str(best_path), phase=phase, updated_unix=time.time(),
                   wall_seconds=time.time()-began, metric=metric))
        write_json(args.output/'best.json', dict(kind='neural', path=str(best_path),
                   game=args.game, config=cls.config_id, rules_version=cls.rules_version))

    next_update = start_update
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn')) as pool:
        for iteration in range(start_update, args.updates):
            if STOP or time.time()+2*args.max_match_seconds >= deadline:
                break
            path = args.output/f'policy_{iteration:04d}.pt'
            save_model(path, model, cls, weights, dict(source_commit=args.source_commit, update=iteration-1))
            jobs = [dict(game=args.game, path=str(path), self_play=i<round(args.games*args.self_play_rate),
                         learner_seat=i%2, seed=20_000_000+iteration*1000+i,
                         opponent=prior_spec if i%3 else dict(kind='heuristic'),
                         max_seconds=args.max_match_seconds, record=i==0,
                         gamma=args.gamma, gae_lambda=args.gae_lambda) for i in range(args.games)]
            played = list(pool.map(rollout, jobs, chunksize=1))
            results = [result for result, _ in played]
            records = [row for _, rows in played for row in rows]
            write_json(args.output/f'rollout_{iteration:04d}.json', results)
            if not records:
                save(iteration, 'blocked_no_complete_games')
                raise RuntimeError('No complete games to train from; inspect rollout results')
            metric = update(model, optimizer, records, args.device, rng, args.epochs, args.batch_size,
                            args.entropy_coef, args.prior_kl_coef)
            metric.update(update=iteration, games=len(results), completed=sum(r['status']=='complete' for r in results),
                          unix=time.time())
            if (iteration+1)%args.eval_every == 0:
                candidate_path = args.output/f'candidate_{iteration:04d}.pt'
                save_model(candidate_path, model, cls, weights, dict(source_commit=args.source_commit, update=iteration))
                candidate = dict(kind='neural', path=str(candidate_path))
                seeds = [21_000_000+iteration*1000+i for i in range(args.gate_seeds)]
                gate, _ = train.evaluate(pool, train.paired_jobs(args.game, candidate,
                    dict(kind='neural', path=str(best_path)), seeds, max_seconds=args.max_match_seconds),
                    args.output, f'gate_{iteration:04d}')
                metric['gate'] = gate
                metric['promoted'] = gate['completed']==gate['games'] and gate['completed_win_rate']>=.625
                if metric['promoted']:
                    best_path = candidate_path
            next_update = iteration+1
            save(next_update, 'training', metric)
            with (args.output/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(metric)+'\n')
            print(json.dumps(metric), flush=True)
        save(next_update, 'stopping' if STOP else 'confirmation')
        if not STOP and time.time()+args.max_match_seconds < deadline and not (args.output/'confirmation.json').exists():
            seeds = [22_000_000+i for i in range(args.confirm_seeds)]
            summary, _ = train.evaluate(pool, train.paired_jobs(args.game,
                dict(kind='neural', path=str(best_path)), dict(kind='heuristic'), seeds,
                record=True, max_seconds=args.max_match_seconds), args.output, 'confirmation')
            write_json(args.output/'confirmation_summary.json', summary)
            print(json.dumps(dict(confirmation=summary)), flush=True)
        save(next_update, 'stopped' if STOP else 'complete')


if __name__ == '__main__':
    main()
