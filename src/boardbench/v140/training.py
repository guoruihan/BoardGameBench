"""Complete-game PPO against frozen/lagged/current opponents.

Rewards follow the learner's own chronological decisions. Seat changes are NOT
terminal boundaries. Opponent actions never enter the on-policy loss.
"""
import argparse
from copy import deepcopy
import gzip
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from boardbench.artifacts.store import digest, read_json, write_json, source_id
from boardbench.v127.training import advantages
from boardbench.v138.baseline import load_model
from .collector import EnvironmentPool
from .encoding import LAYOUT
from .model import DuoPolicy


def infer_actions(models, roles, xs, masks, device, rng, greedy=False, *, legacy=False):
    """Group by model identity; upload inputs and fetch results once per step.

    The legacy branch is only a paired benchmark reference. Grouping changes RNG
    draw order, not behavior probabilities; exact resume is version-local.
    """
    if legacy:
        result = np.zeros((len(roles), 3), np.float64)
        for role in sorted(set(roles)):
            ids = np.array([j for j, r in enumerate(roles) if r == role])
            with torch.no_grad():
                logits, value = models[role].forward_reference(
                    torch.as_tensor(xs[ids], dtype=torch.float64, device=device),
                    torch.as_tensor(masks[ids], device=device))
                dist = torch.distributions.Categorical(logits=logits)
                chosen = logits.argmax(-1) if greedy else torch.multinomial(dist.probs, 1, generator=rng).squeeze(-1)
                result[ids, 0] = chosen.cpu().numpy()
                result[ids, 1] = dist.log_prob(chosen).cpu().numpy()
                result[ids, 2] = value.cpu().numpy()
        return result, len(set(roles))
    groups = {}
    for j, role in enumerate(roles):
        key = id(models[role])
        if key not in groups: groups[key] = (models[role], [])
        groups[key][1].append(j)
    with torch.inference_mode():
        x = torch.as_tensor(xs, dtype=torch.float64, device=device)
        mask = torch.as_tensor(masks, device=device)
        output = torch.empty((len(roles), 3), dtype=torch.float64, device=device)
        for model, indices in groups.values():
            ids = torch.tensor(indices, device=device)
            logits, value = model(x[ids], mask[ids])
            # Model enforces nonempty legal masks. PPO separately checks finite
            # losses and exact behavior ratios; avoid Categorical's GPU->CPU
            # distribution-validation synchronization on every small subgroup.
            dist = torch.distributions.Categorical(logits=logits, validate_args=False)
            chosen = logits.argmax(-1) if greedy else torch.multinomial(dist.probs, 1, generator=rng).squeeze(-1)
            output[ids] = torch.stack((chosen.to(torch.float64), dist.log_prob(chosen), value), -1)
        return output.cpu().numpy(), len(groups)


def summarize(games):
    seeds = sorted({g['seed'] for g in games})
    # Environmental clusters, not independent Bernoulli games for paired seats.
    clusters = np.array([np.mean([g['margin'] for g in games if g['seed'] == s]) for s in seeds])
    wins = np.array([np.mean([g['win_credit'] for g in games if g['seed'] == s]) for s in seeds])
    sem = lambda x: float(x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 1 else None
    return dict(games=len(games), environments=len(seeds),
                score_mean=float(np.mean([g['score'] for g in games])),
                animal_mean=float(np.mean([g['animal'] for g in games])),
                margin_mean=float(np.mean([g['margin'] for g in games])),
                win_credit_mean=float(np.mean([g['win_credit'] for g in games])),
                paired_margin_sem=sem(clusters), paired_win_sem=sem(wins))


class Trainer:
    def __init__(self, config, device='cpu'):
        self.config = deepcopy(config); self.device = str(device)
        if config['layout'] != LAYOUT or config['objective'] not in ('score', 'margin'):
            raise ValueError('explicit duo schema/objective required')
        if digest(config['parent']) != config['parent_sha256']:
            raise ValueError('solo parent digest mismatch')
        if config.get('source_id') and config['source_id'] != source_id():
            raise ValueError('frozen training source mismatch')
        if (any(not 200000000 <= s < 300000000 for s in config['development_seeds'])
                or any(not 300000000 <= s < 400000000 for s in config['confirmation_seeds'])):
            raise ValueError('development/confirmation namespace leakage')
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.manual_seed(config['seed'])
        solo, _ = load_model(config['parent'])
        self.model = DuoPolicy(solo, config['objective']).to(self.device)
        self.frozen = deepcopy(self.model).eval().requires_grad_(False)
        self.snapshots = []
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.rng = torch.Generator(device=self.device).manual_seed(config['seed']+140000)
        self.seed_rng = np.random.default_rng(config['seed'])
        self.batches = self.matches = self.actions = 0
        self.elapsed_seconds = 0.; self.best_metric = -float('inf')
        self.pool = None

    def make_pool(self, count):
        return EnvironmentPool(count, min(self.config['workers'], count),
                               native_artifact=self.config['native_artifact'])

    def collect(self, pool, seeds, learner_seats, opponents, greedy=False, keep_rows=True, *, legacy=False):
        if len(seeds) != pool.count or len(learner_seats) != pool.count or len(opponents) != pool.count:
            raise ValueError('one match specification per slot')
        tick = time.monotonic(); arrays = pool.reset(seeds)
        env_seconds = time.monotonic()-tick; model_seconds = 0.
        inference_calls = total_actions = 0
        trajectories, traces = [[] for _ in seeds], [[] for _ in seeds]
        models = {'learner': self.model, 'frozen': self.frozen, 'current': self.model,
                  **{f'lag{i}': m for i, m in enumerate(self.snapshots)}}
        while not arrays['done'].all():
            slots = np.flatnonzero(~arrays['done']); seats = arrays['stats'][slots, 4].astype(int)
            xs = arrays['features'][slots].copy(); masks = arrays['masks'][slots].copy()
            roles = ['learner' if seat == learner_seats[slot] else opponents[slot] for slot, seat in zip(slots, seats)]
            tick = time.monotonic()
            inferred, calls = infer_actions(models, roles, xs, masks, self.device, self.rng, greedy, legacy=legacy)
            actions = inferred[:, 0].astype(np.int64); logps = inferred[:, 1]; values = inferred[:, 2]
            inference_calls += calls; total_actions += len(slots)
            model_seconds += time.monotonic()-tick
            if keep_rows:
                for j, slot in enumerate(slots):
                    traces[slot].append(dict(seat=int(seats[j]), action=int(actions[j]), logp=float(logps[j]), role=roles[j]))
                    if roles[j] == 'learner':
                        seat = int(seats[j]); stats = arrays['stats'][slot]
                        potential = float(stats[seat]-(stats[1-seat] if self.config['objective'] == 'margin' else 0))
                        trajectories[slot].append((xs[j], masks[j], int(actions[j]), float(logps[j]),
                                                   float(values[j]), potential))
            tick = time.monotonic(); pool.step(slots, actions); env_seconds += time.monotonic()-tick
        rows, games = [], []
        for slot, (seed, seat, trajectory) in enumerate(zip(seeds, learner_seats, trajectories)):
            s = arrays['stats'][slot]; margin = float(s[seat]-s[1-seat])
            game = dict(seed=int(seed), seat=int(seat), opponent=opponents[slot], score=float(s[seat]),
                        opponent_score=float(s[1-seat]), animal=float(s[2+seat]), margin=margin,
                        win_credit=float(s[5] if seat == 0 else 1-s[5]), natural_terminal=True)
            games.append(game)
            if keep_rows:
                final = margin if self.config['objective'] == 'margin' else float(s[seat])
                potentials = [r[5] for r in trajectory]+[final]
                # Include opponent changes until the learner's NEXT decision.
                # These true-score differences telescope; no animal bonus/shaping.
                rewards = [(b-a)/150. for a, b in zip(potentials, potentials[1:])]
                if abs(sum(rewards)*150+potentials[0]-final) > 1e-7:
                    raise RuntimeError('competitive reward accounting mismatch')
                for trace, reward in zip((r for r in traces[slot] if r['role'] == 'learner'), rewards):
                    trace['reward_to_next_own_decision'] = reward
                adv, targets = advantages(rewards, [r[4] for r in trajectory], self.config['gae_lambda'])
                rows.extend((*r[:4], float(targets[i]), float(adv[i])) for i, r in enumerate(trajectory))
        if keep_rows:
            self.last_traces = [dict(**g, decisions=trace) for g, trace in zip(games, traces)]
        return rows, dict(**summarize(games), per_game=games, environment_seconds=env_seconds, model_seconds=model_seconds,
                         total_player_actions=total_actions, inference_calls=inference_calls,
                         mean_inference_batch=total_actions/max(1, inference_calls))

    def update(self, rows):
        x = torch.as_tensor(np.stack([r[0] for r in rows]), dtype=torch.float64, device=self.device)
        mask = torch.as_tensor(np.stack([r[1] for r in rows]), device=self.device)
        action = torch.tensor([r[2] for r in rows], device=self.device)
        old = torch.tensor([r[3] for r in rows], dtype=torch.float64, device=self.device)
        target = torch.tensor([r[4] for r in rows], dtype=torch.float64, device=self.device)
        adv = torch.tensor([r[5] for r in rows], dtype=torch.float64, device=self.device)
        adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
        size = self.config['minibatch']
        ratio_error = 0.
        with torch.no_grad():
            for ids in torch.arange(len(rows), device=self.device).split(size):
                logits, _ = self.model(x[ids], mask[ids])
                lp = torch.distributions.Categorical(logits=logits).log_prob(action[ids])
                ratio_error = max(ratio_error, float(((lp-old[ids]).exp()-1).abs().max()))
        if ratio_error > 1e-5:
            raise RuntimeError(f'duo behavior probability mismatch: {ratio_error}')
        before = self.model.solo.actor.weight.detach().clone()
        context_before = self.model.context[-1].weight.detach().clone()
        losses, kls = [], []; stopped = False
        for _ in range(self.config['epochs']):
            for ids in torch.randperm(len(rows), generator=self.rng, device=self.device).split(size):
                logits, value = self.model(x[ids], mask[ids])
                dist = torch.distributions.Categorical(logits=logits)
                logratio = dist.log_prob(action[ids])-old[ids]; ratio = logratio.exp()
                kl = ((ratio-1)-logratio).mean()
                if float(kl.detach()) > self.config['target_kl']:
                    stopped = True; break
                loss = (-torch.minimum(ratio*adv[ids], ratio.clamp(.8, 1.2)*adv[ids]).mean()
                        + .5*(value-target[ids]).square().mean()-self.config['entropy']*dist.entropy().mean())
                if not torch.isfinite(loss): raise RuntimeError('nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5, error_if_nonfinite=True)
                self.optimizer.step(); losses.append(float(loss.detach())); kls.append(float(kl.detach()))
            if stopped: break
        if not losses: raise RuntimeError('no PPO updates completed')
        return dict(initial_ratio_max_error=ratio_error, updates=len(losses), loss=float(np.mean(losses)),
                    approx_kl=float(np.mean(kls)), kl_stopped=stopped,
                    actor_change=float((self.model.solo.actor.weight.detach()-before).norm()),
                    context_change=float((self.model.context[-1].weight.detach()-context_before).norm()))

    def batch(self):
        if self.pool is None: self.pool = self.make_pool(self.config['batch_matches'])
        seeds = self.seed_rng.integers(100000000, 200000000, self.pool.count).tolist()
        seats = [(self.matches+i) % 2 for i in range(self.pool.count)]
        opponents = []
        for i in range(self.pool.count):
            if i % 4 < 2: opponents.append('frozen')
            elif i % 4 == 2 and self.snapshots: opponents.append(f'lag{self.batches % len(self.snapshots)}')
            else: opponents.append('current')
        rows, metric = self.collect(self.pool, seeds, seats, opponents)
        metric.update(self.update(rows))
        self.batches += 1; self.matches += len(seeds); self.actions += len(rows)
        if self.batches % self.config['snapshot_interval'] == 0:
            self.snapshots.append(deepcopy(self.model).eval().requires_grad_(False))
            self.snapshots = self.snapshots[-3:]
        return dict(**metric, batches=self.batches, matches=self.matches, actions=self.actions)

    def evaluate(self, seeds):
        """Fixed frozen transferred opponent, greedy and both seats, no RNG use."""
        games = []
        size = min(self.config['batch_matches'], 2*len(seeds))
        specs = [(s, seat) for s in seeds for seat in range(2)]
        if len(specs) % size: raise ValueError('paired evaluation cohort must divide panel')
        with self.make_pool(size) as pool:
            for start in range(0, len(specs), size):
                batch = specs[start:start+size]
                _, result = self.collect(pool, [s for s, _ in batch], [seat for _, seat in batch],
                                         ['frozen']*size, greedy=True, keep_rows=False)
                games.extend(result['per_game'])
        return dict(**summarize(games), per_game=games)

    def save(self, path):
        path = Path(path); tmp = path.with_suffix('.partial.pt')
        torch.save(dict(version=140, config=self.config, model=self.model.state_dict(),
                        frozen=self.frozen.state_dict(), snapshots=[m.state_dict() for m in self.snapshots],
                        optimizer=self.optimizer.state_dict(), rng=self.rng.get_state(),
                        seed_rng=self.seed_rng.bit_generator.state, cpu_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if self.device.startswith('cuda') else [],
                        batches=self.batches, matches=self.matches, actions=self.actions,
                        elapsed_seconds=self.elapsed_seconds, best_metric=self.best_metric), tmp)
        tmp.replace(path)
        write_json(path.with_suffix('.json'), dict(sha256=digest(path), batches=self.batches,
                                                  actions=self.actions, source_id=self.config.get('source_id')))

    def load(self, path):
        if digest(path) != read_json(Path(path).with_suffix('.json'))['sha256']:
            raise ValueError('checkpoint digest mismatch')
        s = torch.load(path, map_location=self.device, weights_only=False)
        if s['version'] != 140 or s['config'] != self.config: raise ValueError('exact resume requires identical version/config')
        self.model.load_state_dict(s['model']); self.frozen.load_state_dict(s['frozen'])
        self.snapshots = []
        for state in s['snapshots']:
            model = deepcopy(self.frozen); model.load_state_dict(state); self.snapshots.append(model)
        self.optimizer.load_state_dict(s['optimizer']); self.rng.set_state(s['rng'].cpu())
        self.seed_rng.bit_generator.state = s['seed_rng']; torch.set_rng_state(s['cpu_rng'].cpu())
        if s['cuda_rng']: torch.cuda.set_rng_state_all([r.cpu() for r in s['cuda_rng']])
        for key in ('batches', 'matches', 'actions', 'elapsed_seconds', 'best_metric'): setattr(self, key, s[key])

    def close(self):
        if self.pool: self.pool.close(); self.pool = None


def run(config, out, device='cuda', resume=False):
    out = Path(out); out.mkdir(parents=True, exist_ok=resume)
    if resume and (out/'done.json').exists():
        raise ValueError('completed trial must not be silently continued')
    if config.get('deadline_unix', float('inf')) <= time.time():
        raise ValueError('original trial budget already expired')
    write_json(out/'config.json', config)
    start = time.monotonic()
    t = Trainer(config, device)
    if resume: t.load(out/'latest.pt')
    prior = t.elapsed_seconds
    stop_at = start + max(0., config['max_seconds']-prior)
    stop_at = min(stop_at, time.monotonic()+config.get('deadline_unix', float('inf'))-time.time())
    stopping = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
    try:
        if not resume:
            initial = t.evaluate(config['development_seeds'])
            write_json(out/'initial.json', initial)
            t.best_metric = initial['margin_mean']; t.elapsed_seconds = time.monotonic()-start
            t.save(out/'best.pt'); t.save(out/'latest.pt')
            print(json.dumps(dict(event='initial', **{k: v for k, v in initial.items() if k != 'per_game'})), flush=True)
        while not stopping[0] and time.monotonic() < stop_at-config.get('finish_reserve_seconds', 600):
            tick = time.monotonic(); behavior_sha = digest(out/'latest.pt'); result = t.batch()
            (out/'trajectories').mkdir(exist_ok=True)
            with gzip.open(out/'trajectories'/f'batch_{t.batches:06d}.json.gz', 'wt') as f:
                json.dump(dict(layout=LAYOUT, rules_version='harmonies_duo_a_base32_v1',
                               source_id=config.get('source_id'), behavior_checkpoint_sha256=behavior_sha,
                               objective=config['objective'], batch=t.batches,
                               games=t.last_traces), f, separators=(',', ':'))
            if t.batches % config['evaluate_interval'] == 0:
                evaluation = t.evaluate(config['development_seeds'])
                write_json(out/f'dev_{t.batches:06d}.json', evaluation)
                result['development'] = {k: v for k, v in evaluation.items() if k != 'per_game'}
                if evaluation['margin_mean'] > t.best_metric:
                    t.best_metric = evaluation['margin_mean']; t.elapsed_seconds = prior+time.monotonic()-start
                    t.save(out/'best.pt')
            t.elapsed_seconds = prior+time.monotonic()-start
            result.update(elapsed_seconds=t.elapsed_seconds, batch_seconds=time.monotonic()-tick)
            t.save(out/'latest.pt')
            with (out/'curve.jsonl').open('a') as f: f.write(json.dumps(result)+'\n')
            print(json.dumps({k: v for k, v in result.items() if k != 'per_game'}), flush=True)
        t.elapsed_seconds = prior+time.monotonic()-start; t.save(out/'final.pt')
        from .policy import export_policy
        export_policy(t.model, out/'final_policy.pt', config.get('source_id'), config['parent_sha256'])
        if stopping[0]:
            write_json(out/'interrupted.json', dict(elapsed_seconds=t.elapsed_seconds, actions=t.actions))
            return
        selected = torch.load(out/'best.pt', map_location=device, weights_only=False)
        t.model.load_state_dict(selected['model'])
        export_policy(t.model, out/'best_policy.pt', config.get('source_id'), config['parent_sha256'])
        confirmation = t.evaluate(config['confirmation_seeds'])
        write_json(out/'confirmation.json', dict(**confirmation, selected_checkpoint='best.pt',
                                                selection_uses_confirmation=False))
        write_json(out/'done.json', dict(status='completed', wall_seconds=prior+time.monotonic()-start,
                   budget_seconds=config['max_seconds'], batches=t.batches, actions=t.actions,
                   confirmation={k: v for k, v in confirmation.items() if k != 'per_game'}))
        print(json.dumps(dict(event='completed', **read_json(out/'done.json'))), flush=True)
    except BaseException as exc:
        write_json(out/'failed.json', dict(error=repr(exc), elapsed_seconds=prior+time.monotonic()-start))
        raise
    finally: t.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda'); p.add_argument('--resume', action='store_true')
    args = p.parse_args(); run(read_json(args.config), args.out, args.device, args.resume)
