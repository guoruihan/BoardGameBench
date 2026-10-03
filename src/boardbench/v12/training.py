"""Resumable compact PPO/BC; immutable configuration and update-boundary saves."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.provenance import validate_splits
from .encoding import STOP, decode
from .neural import CompactSolver


def config_id(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def returns(rewards, bootstrap=0.):
    result = []
    value = bootstrap
    for reward in reversed(rewards):
        value = reward + value
        result.append(value)
    return list(reversed(result))


def policy_distribution(logits):
    """One normalizer for sampling and updates; keep network/gradients in float32.

    On the tested CUDA 12.1/4090 stack even fixed float32 logits produced variable
    logsumexp/softmax results exceeding the PPO consistency gate. Float64 probability
    arithmetic fixes that measured failure without changing the gate or old logps.
    """
    return torch.distributions.Categorical(logits=logits.double())


class Trainer:
    def __init__(self, config, device='cpu'):
        self.config = deepcopy(config)
        self.device = device
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('requested GPU unavailable')
        self.seed = config['training_seed']
        self.splits = config['splits']
        validate_splits(self.splits)
        self.train_seeds = set(self.splits['train'])
        torch.set_num_threads(config.get('threads', 1))
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.solver = CompactSolver(self.seed, device=device, **config['network'])
        self.optimizer = torch.optim.Adam(self.solver.model.parameters(), lr=config.get('learning_rate', .0003))
        self.episodes = self.steps = self.updates = self.batches = 0
        self.elapsed = 0.
        self.source = source_id()
        self.lineage = None
        self.failure_directory = None
        self.algorithm = config.get('algorithm', 'ppo')
        if self.algorithm not in ('ppo', 'bc'):
            raise ValueError('unsupported training algorithm')
        if self.solver.params['time_conditioning'] and (self.algorithm != 'ppo' or config.get('episode_seconds', 0) <= 0):
            raise ValueError('time-conditioned training requires PPO with a positive episode_seconds budget')

    def _update(self, rows, bc=False):
        x = torch.tensor(np.stack([r[0] for r in rows]), device=self.device)
        masks = torch.tensor(np.stack([r[1] for r in rows]), device=self.device)
        actions = torch.tensor([r[2] for r in rows], device=self.device)
        old = torch.tensor([r[3] for r in rows], dtype=torch.float64, device=self.device)
        targets = torch.tensor([r[4] for r in rows], device=self.device)
        advantages = torch.tensor([r[5] for r in rows], device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        stats = []
        early_stop = False
        if not masks.gather(1, actions[:, None]).all():
            raise ValueError('training action is outside its recorded legal mask')
        with torch.no_grad():
            logits, _ = self.solver.model(x, masks)
            logps = policy_distribution(logits).log_prob(actions)
            before = (logps - old).exp()
            ratio_error = float((before - 1).abs().max()) if not bc else None
            if not bc and (not torch.isfinite(before).all() or ratio_error > 1e-4):
                witness = int((before - 1).abs().argmax())
                info = {'ratio_max_error': ratio_error, 'row': witness, 'rows': len(rows),
                        'old_logp': old[witness].item(), 'recomputed_logp': logps[witness].item(),
                        'batch': self.batches + 1, 'source_id': self.source}
                if self.failure_directory is not None:
                    path = Path(self.failure_directory) / 'probability_failure.pt'
                    torch.save({'diagnostic_only': True, 'info': info, 'rows': rows,
                                'model': self.solver.model.state_dict(), 'config': self.config}, path)
                raise RuntimeError(f'behavior/recomputed log-prob mismatch before PPO update: {info}')
        for _ in range(self.config.get('epochs', 4)):
            order = self.rng.permutation(len(rows))
            for begin in range(0, len(rows), self.config.get('minibatch_size', 256)):
                ids = torch.as_tensor(order[begin:begin + self.config.get('minibatch_size', 256)], device=self.device)
                logits, values = self.solver.model(x[ids], masks[ids])
                dist = policy_distribution(logits)
                logps = dist.log_prob(actions[ids])
                ratios = (logps - old[ids]).exp()
                log_ratio = logps - old[ids]
                approx_kl = ((ratios - 1) - log_ratio).mean()
                target_kl = self.config.get('target_kl')
                if not bc and target_kl is not None and approx_kl.item() > target_kl:
                    early_stop = True
                    break  # Do not apply this excessive update; keep fresh on-policy data next batch.
                clip = self.config.get('clip_ratio', .2)
                policy_loss = (-logps.mean() if bc else -torch.minimum(
                    ratios * advantages[ids], ratios.clamp(1 - clip, 1 + clip) * advantages[ids]).mean())
                value_loss = (values - targets[ids]).square().mean()
                entropy = dist.entropy().mean()
                loss = policy_loss if bc else policy_loss + .5 * value_loss - self.config.get('entropy_coef', .01) * entropy
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.solver.model.parameters(), .5, error_if_nonfinite=True)
                self.optimizer.step()
                self.updates += 1
                with torch.no_grad():
                    stats.append([loss.item(), value_loss.item(), entropy.item(),
                                  0. if bc else approx_kl.item(),
                                  0. if bc else ((ratios - 1).abs() > clip).float().mean().item(), float(norm)])
            if early_stop:
                break
        keys = ('loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction', 'gradient_norm')
        return {**dict(zip(keys, np.mean(stats, axis=0).tolist() if stats else [0.] * len(keys))),
                'initial_ratio_max_error': ratio_error, 'kl_early_stop': early_stop,
                'optimizer_steps_this_batch': len(stats)}

    def ppo_batch(self, cases=None):
        start = time.monotonic()
        rows, scores = [], []
        animal_scores, gains = [], []
        placements = completions = 0
        if cases is not None:
            if not cases or any(c['source_seed'] not in self.train_seeds for c in cases):
                raise ValueError('snapshot curriculum must use nonempty training-only source games')
            if self.solver.params['time_conditioning']:
                raise ValueError('action-limited diagnosis is not a timed objective')
        phase_times = dict(encoding_seconds=0., model_seconds=0., engine_seconds=0.)
        natural = 0
        for _ in range(self.config.get('batch_episodes', 16)):
            env = Harmonies()
            if cases is None:
                obs, _ = env.reset(self.splits['train'][self.episodes % len(self.splits['train'])])
                horizon = 256
            else:
                case = cases[self.episodes % len(cases)]
                obs = env.load_state(case['snapshot'])
                horizon = case['horizon']
                if not isinstance(horizon, int) or not 1 <= horizon <= 3 or obs['terminated'] or sum(c['stack_id'] == 0 for c in obs['board']) > 2:
                    raise ValueError('curriculum requires live no-refill final-turn snapshots and horizon 1..3')
            initial_score = obs['score_breakdown']['total']
            initial_completed = len(obs['completed_cards'])
            trajectory = []
            budget = self.config.get('episode_seconds') if self.solver.params['time_conditioning'] else None
            deadline = time.monotonic() + budget if budget is not None else None
            for _step in range(horizon):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                tick = time.monotonic()
                data = self.solver.encode(obs, **({'remaining_seconds': max(0., deadline - tick),
                    'time_budget_seconds': budget} if deadline is not None else {}))
                x = np.asarray(data['features'], dtype=np.float32)
                mask = np.asarray(data['action_mask'], dtype=bool)
                phase_times['encoding_seconds'] += time.monotonic() - tick
                tick = time.monotonic()
                with torch.no_grad():
                    logits, v = self.solver.model(torch.tensor(x, device=self.device)[None],
                                                  torch.tensor(mask, device=self.device)[None])
                    dist = policy_distribution(logits)
                    sampled = dist.sample()
                    logp, value = dist.log_prob(sampled).item(), v.item()
                    chosen = int(sampled.item())
                phase_times['model_seconds'] += time.monotonic() - tick
                if deadline is not None and time.monotonic() >= deadline:
                    break  # A rejected late action is not a PPO sample.
                action = decode(obs, chosen, self.solver.params['action_encoding'], self.solver.params['allow_stop'])
                tick = time.monotonic()
                private = env._private_state() if cases is not None else None
                nxt = obs if action == STOP else env.step(action).observation
                if cases is not None and env._private_state() != private:
                    raise RuntimeError('curriculum encountered a hidden future draw')
                reward = (nxt['score_breakdown']['total'] - obs['score_breakdown']['total']) / self.config.get('reward_scale', 150.)
                phase_times['engine_seconds'] += time.monotonic() - tick
                trajectory.append((x, mask, chosen, logp, value, reward))
                self.steps += action != STOP
                placements += action.get('type') == 'place_animal'
                obs = nxt
                if action == STOP or obs['terminated']:
                    break
            else:
                if cases is None:
                    raise RuntimeError('game action limit')
            targets = returns([r[-1] for r in trajectory])
            for row, target in zip(trajectory, targets):
                x, mask, action, logp, value, _ = row
                rows.append((x, mask, action, logp, target, target - value))
            scores.append(obs['score_breakdown']['total'])
            animal_scores.append(obs['score_breakdown']['animals'])
            gains.append(obs['score_breakdown']['total'] - initial_score)
            completions += len(obs['completed_cards']) - initial_completed
            natural += obs['terminated']
            self.episodes += 1
        metrics = self._update(rows)
        self.elapsed += time.monotonic() - start
        self.batches += 1
        return {**metrics, **phase_times, 'episodes': self.episodes, 'steps': self.steps,
                'updates': self.updates, 'mean_training_score': float(np.mean(scores)),
                'mean_training_animal_score': float(np.mean(animal_scores)),
                'mean_score_gain': float(np.mean(gains)), 'animal_placements_per_episode': placements / len(scores),
                'completed_cards_per_episode': completions / len(scores),
                'natural_fraction': natural / len(scores), 'training_seconds': self.elapsed,
                'objective': ('auxiliary snapshot finite-action total-score return; zero terminal bootstrap' if cases is not None else
                    'score_at_natural_STOP_or_deadline; real wall time; resumed RNG does not make clocks deterministic'
                    if self.solver.params['time_conditioning'] else 'score_at_natural_or_STOP; full-game warm start, not a wall-time surrogate')}

    def bc_batch(self, examples):
        start = time.monotonic()
        indices = self.rng.integers(len(examples), size=self.config.get('bc_batch_size', 512))
        rows = []
        for i in indices:
            data = examples[int(i)]
            rows.append((np.asarray(data['features'], dtype=np.float32),
                         np.asarray(data['mask'], dtype=bool), data['action'], 0., 0., 0.))
        metrics = self._update(rows, bc=True)
        self.batches += 1
        self.steps += len(rows)
        self.elapsed += time.monotonic() - start
        return {**metrics, 'batches': self.batches, 'updates': self.updates,
                'supervised_examples_seen': self.steps, 'training_seconds': self.elapsed,
                'objective': 'search imitation; teacher used in data generation only'}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.solver.training = dict(algorithm=self.algorithm, training_seed=self.seed, episodes=self.episodes,
                                    updates=self.updates, batches=self.batches, training_seconds=self.elapsed,
                                    config_id=config_id(self.config), source_id=self.source,
                                    splits_id=config_id(self.splits), lineage=self.lineage,
                                    probability_arithmetic='float64', teacher_sha256=self.config.get('teacher_sha256'))
        temporary = path.with_suffix(path.suffix + '.partial')
        state = dict(format_version=1, source_id=self.source, config=self.config,
                     model=self.solver.model.state_dict(), optimizer=self.optimizer.state_dict(),
                     rng_numpy=self.rng.bit_generator.state, rng_numpy_global=np.random.get_state(),
                     rng_python=random.getstate(), rng_torch=torch.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if str(self.device).startswith('cuda') else [],
                     episodes=self.episodes, steps=self.steps, updates=self.updates,
                     batches=self.batches, elapsed=self.elapsed, device=self.device,
                     lineage=self.lineage,
                     boundary='after complete rollout/update batch')
        torch.save(state, temporary)
        os.replace(temporary, path)
        self.solver.save_weights(path.with_suffix('.weights.pt'))
        write_json(path.with_suffix('.json'), {'resume_sha256': digest(path),
                    'weights_sha256': digest(path.with_suffix('.weights.pt')), **self.solver.training})

    def load(self, path):
        path = Path(path)
        if digest(path) != read_json(path.with_suffix('.json'))['resume_sha256']:
            raise ValueError('training checkpoint hash mismatch')
        # Own trusted checkpoint includes NumPy RNG state (not an untrusted model upload).
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state['format_version'] != 1 or state['config'] != self.config or state['source_id'] != self.source:
            raise ValueError('training resume config/source mismatch')
        if state['device'] != self.device:
            raise ValueError('exact continuation requires the same device type')
        self.solver.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state = state['rng_numpy']
        np.random.set_state(state['rng_numpy_global']); random.setstate(state['rng_python'])
        torch.set_rng_state(state['rng_torch'].cpu())
        if state['rng_cuda']:
            torch.cuda.set_rng_state_all([value.cpu() for value in state['rng_cuda']])
        for key in ('episodes', 'steps', 'updates', 'batches', 'elapsed'):
            setattr(self, key, state[key])
        self.lineage = state.get('lineage')

    def initialize_from_bc(self, path, *, expected_sha256):
        """Explicit weight-only phase transition; never restores a BC optimizer/RNG."""
        if self.algorithm != 'ppo' or self.batches or self.updates or self.episodes or self.optimizer.state:
            raise ValueError('BC initialization requires a fresh PPO trainer')
        path = Path(path)
        if digest(path) != expected_sha256:
            raise ValueError('pretrained weight hash mismatch')
        saved = torch.load(path, map_location=self.device, weights_only=True)
        training = saved.get('training', {})
        if training.get('algorithm') != 'bc' or not training.get('updates', 0):
            raise ValueError('pretrained weights must be a trained BC artifact')
        if training.get('source_id') != self.source:
            raise ValueError('pretrained source mismatch; explicit import required')
        if training.get('splits_id') != config_id(self.splits):
            raise ValueError('pretrained splits mismatch')
        if training.get('training_seed') != self.seed:
            raise ValueError('paired BC/PPO training seed mismatch')
        self.solver.load_weights(path)  # Also checks complete network/encoding metadata.
        self.lineage = {'kind': 'bc_to_ppo', 'weights': str(path.resolve()),
                        'weights_sha256': expected_sha256, 'parent_training': training,
                        'optimizer': 'fresh Adam', 'rollouts': 'fresh on-policy PPO',
                        'not_exact_resume': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--teacher')
    parser.add_argument('--initialize-bc')
    parser.add_argument('--initialize-sha256')
    args = parser.parse_args()
    if bool(args.initialize_bc) != bool(args.initialize_sha256):
        raise ValueError('BC initialization requires both weight path and hash')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = read_json(args.config)
    trainer = Trainer(config, args.device)
    trainer.failure_directory = out
    if args.initialize_bc:
        if args.resume or not args.initialize_sha256:
            raise ValueError('BC transition needs a hash and cannot be combined with exact resume')
        trainer.initialize_from_bc(args.initialize_bc, expected_sha256=args.initialize_sha256)
    if args.resume:
        trainer.load(args.resume)
    data = read_json(args.teacher) if args.teacher else None
    if trainer.algorithm == 'bc':
        allowed_teacher_seeds = set(config['splits']['train'])
        if not data or any(row.get('seed') not in allowed_teacher_seeds for row in data):
            raise ValueError('BC requires a nonempty training-only teacher dataset')
        if config.get('teacher_sha256') and digest(args.teacher) != config['teacher_sha256']:
            raise ValueError('teacher dataset hash mismatch')
    print(json.dumps({'event': 'trainer_ready', 'device': args.device,
                      'device_name': torch.cuda.get_device_name(0) if args.device.startswith('cuda') else 'cpu',
                      'source_id': trainer.source}), flush=True)
    trainer.save(out / 'initial.resume.pt')
    deadline = time.monotonic() + args.seconds
    with (out / 'training.jsonl').open('x') as log:
        while time.monotonic() < deadline:
            row = trainer.bc_batch(data) if trainer.algorithm == 'bc' else trainer.ppo_batch()
            log.write(json.dumps(row) + '\n'); log.flush()
            if trainer.batches == 1:
                print(json.dumps({'event': 'first_update', **row}), flush=True)
            if trainer.batches % config.get('checkpoint_batches', 16) == 0:
                trainer.save(out / f'batch_{trainer.batches:07d}.resume.pt')
        trainer.save(out / 'final.resume.pt')
    write_json(out / 'summary.json', {'status': 'completed', 'config': config, 'source_id': trainer.source,
                                     'episodes': trainer.episodes, 'steps': trainer.steps,
                                     'updates': trainer.updates, 'training_seconds': trainer.elapsed,
                                     'lineage': trainer.lineage})


if __name__ == '__main__':
    main()
