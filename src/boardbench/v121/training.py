"""Interaction-target PPO using V1.2 semantics, with separate decisions/accounting."""
import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from boardbench.artifacts.store import digest, read_json, source_id, write_json
from boardbench.v12.training import Trainer as BaseTrainer, config_id
from .neural import ResearchSolver
from .registry import AVAILABLE, BATCHED, NUMERIC, definitions, identity


class Trainer(BaseTrainer):
    def __init__(self, config, device='cpu'):
        experiment = config['experiment_id']
        if experiment not in AVAILABLE or config['research_version'] != '1.2.1':
            raise ValueError('unsupported research experiment')
        definition = definitions()[experiment]
        if config['experiment_sha256'] != identity(definition):
            raise ValueError('experiment definition hash mismatch')
        if config['network'] != {'experiment_id': experiment, **definition['architecture']}:
            raise ValueError('network differs from registered factorial')
        if config['algorithm'] != 'ppo' or config.get('reward_scale') != 150.:
            raise ValueError('first factorials require unchanged PPO/score objective')
        if experiment in BATCHED:
            collector = config.get('collector', {})
            if {k:v for k,v in collector.items() if k != 'workers'} != definition['collector']:
                raise ValueError('collector differs from registered throughput variant')
            workers = collector.get('workers')
            if type(workers) is not int or not 0 <= workers <= min(7, config.get('batch_episodes',16)):
                raise ValueError('invalid environment worker count')
        elif config.get('collector'):
            raise ValueError('serial variant cannot change collector')
        base = deepcopy(config)
        base['network'] = {'hidden': 128, 'action_encoding': 'slots240_v1',
                           'observation_encoding': 'cards_v1', 'allow_stop': True}
        super().__init__(base, device)
        self.config = deepcopy(config)
        self.solver = ResearchSolver(self.seed, device=device, **config['network'])
        self.optimizer = torch.optim.Adam(self.solver.model.parameters(), lr=config['learning_rate'])
        self.stop_decisions = 0
        self.decision_count = 0
        self._pool = None
        self._at_boundary = True

    def ppo_batch(self):
        if self.config['experiment_id'] in BATCHED:
            return self._batched_ppo_batch()
        row = super().ppo_batch()
        size = self.config.get('batch_episodes', 16)
        self.stop_decisions += size - round(row['natural_fraction'] * size)
        self.decision_count = self.steps + self.stop_decisions
        row.update(real_atomic_actions=self.steps, policy_decisions=self.decision_count,
                   stop_decisions=self.stop_decisions, simulated_actions=0,
                   experiment_id=self.config['experiment_id'], parameter_count=sum(p.numel() for p in self.solver.model.parameters()))
        return row

    def _batched_ppo_batch(self):
        from .collector import EnvironmentPool, collect
        if not self._at_boundary:
            raise RuntimeError('previous collection/update failed; cannot silently discard samples')
        started = time.monotonic()
        if self._pool is None:
            self._pool = EnvironmentPool(self.config['batch_episodes'], self.config['collector']['workers'])
        self._at_boundary = False
        rows, metrics = collect(self, self._pool)
        tick = time.monotonic()
        update = self._update(rows)
        metrics['update_seconds'] = time.monotonic() - tick
        self.steps += metrics.pop('real_actions_this_batch')
        self.stop_decisions += metrics.pop('stop_decisions_this_batch')
        self.decision_count += metrics.pop('decisions_this_batch')
        self.episodes += self.config['batch_episodes']
        self.batches += 1
        self.elapsed += time.monotonic() - started
        self._at_boundary = True
        return {**update, **metrics, 'episodes': self.episodes, 'steps': self.steps,
                'updates': self.updates, 'training_seconds': self.elapsed,
                'real_atomic_actions': self.steps, 'policy_decisions': self.decision_count,
                'stop_decisions': self.stop_decisions, 'simulated_actions': 0,
                'collector': self.config['collector'], 'experiment_id': self.config['experiment_id'],
                'parameter_count': sum(p.numel() for p in self.solver.model.parameters()),
                'objective': 'score_at_natural_or_STOP; full-game warm start, not a wall-time surrogate'}

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def save(self, path):
        if not self._at_boundary:
            raise RuntimeError('checkpoint requires complete collection and update boundary')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.solver.training = dict(algorithm='ppo', research_version='1.2.1',
            experiment_id=self.config['experiment_id'], training_seed=self.seed,
            episodes=self.episodes, real_atomic_actions=self.steps, policy_decisions=self.decision_count,
            stop_decisions=self.stop_decisions, simulated_actions=0,
            updates=self.updates, batches=self.batches, training_seconds=self.elapsed,
            config_id=config_id(self.config), source_id=self.source, split_sha256=self.config['split_sha256'])
        if self.lineage is not None:
            self.solver.training['lineage'] = self.lineage
        state = dict(format_version=121, source_id=self.source, config=self.config,
                     model=self.solver.model.state_dict(), optimizer=self.optimizer.state_dict(),
                     rng_numpy=self.rng.bit_generator.state, rng_numpy_global=np.random.get_state(),
                     rng_python=random.getstate(), rng_torch=torch.get_rng_state(),
                     rng_cuda=torch.cuda.get_rng_state_all() if str(self.device).startswith('cuda') else [],
                     episodes=self.episodes, steps=self.steps, updates=self.updates,
                     batches=self.batches, elapsed=self.elapsed, device=self.device,
                     stop_decisions=self.stop_decisions, decision_count=self.decision_count,
                     lineage=self.lineage,
                     boundary='after complete full-game rollout/update batch')
        temporary = path.with_suffix(path.suffix + '.partial')
        torch.save(state, temporary)
        os.replace(temporary, path)
        self.solver.save_weights(path.with_suffix('.weights.pt'))
        write_json(path.with_suffix('.json'), {'resume_sha256': digest(path),
                   'weights_sha256': digest(path.with_suffix('.weights.pt')), **self.solver.training})

    def load(self, path):
        if not self._at_boundary:
            raise RuntimeError('resume requires complete collection and update boundary')
        self.close()
        path = Path(path)
        if digest(path) != read_json(path.with_suffix('.json'))['resume_sha256']:
            raise ValueError('checkpoint hash mismatch')
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state['format_version'] != 121 or state['source_id'] != self.source or state['config'] != self.config:
            raise ValueError('research checkpoint config/source mismatch')
        self._restore_state(state)

    def _restore_state(self, state):
        if state['device'] != self.device:
            raise ValueError('exact continuation requires the same device type')
        if state['decision_count'] != state['steps'] + state['stop_decisions']:
            raise ValueError('checkpoint action counters inconsistent')
        if state.get('boundary') != 'after complete full-game rollout/update batch':
            raise ValueError('checkpoint is not an update boundary')
        self.solver.model.load_state_dict(state['model'], strict=True)
        self.optimizer.load_state_dict(state['optimizer'])
        self.rng.bit_generator.state = state['rng_numpy']
        np.random.set_state(state['rng_numpy_global']); random.setstate(state['rng_python'])
        torch.set_rng_state(state['rng_torch'].cpu())
        if state['rng_cuda']:
            torch.cuda.set_rng_state_all([x.cpu() for x in state['rng_cuda']])
        for key in ('episodes', 'steps', 'updates', 'batches', 'elapsed', 'stop_decisions', 'decision_count'):
            setattr(self, key, state[key])
        self.lineage = state.get('lineage')

    def import_numeric_checkpoint(self, path, expected_source, expected_sha256):
        """Explicit FP32 -> FP64 continuation, never a source-check bypass for load()."""
        from .recovery import numeric_parent_config
        if not self._at_boundary or self.batches or self.steps or self.updates:
            raise ValueError('numeric import requires a fresh trainer at a complete boundary')
        self.close()
        path = Path(path).resolve()
        metadata = read_json(path.with_suffix('.json'))
        if not expected_source or digest(path) != expected_sha256 or metadata['resume_sha256'] != expected_sha256:
            raise ValueError('numeric import checkpoint hash mismatch')
        state = torch.load(path, map_location=self.device, weights_only=False)
        parent = numeric_parent_config(self.config)
        if (state.get('format_version') != 121 or state['source_id'] != expected_source or
                metadata['source_id'] != expected_source or state['config'] != parent or
                metadata['config_id'] != config_id(parent)):
            raise ValueError('numeric import parent config/source mismatch')
        for value in state['model'].values():
            if value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()):
                raise ValueError('numeric import requires finite FP32 parameters')
        for optimizer_state in state['optimizer']['state'].values():
            for value in optimizer_state.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise ValueError('numeric import requires finite optimizer state')
        self._restore_state(state)  # load_state_dict promotes parameters and Adam moments, retains RNG.
        self.lineage = {'mode': 'explicit_fp32_to_fp64_numeric_continuation',
            'parent_checkpoint': str(path), 'parent_resume_sha256': expected_sha256,
            'parent_source_id': expected_source, 'parent_config_id': config_id(parent),
            'parent_experiment_id': NUMERIC[self.config['experiment_id']],
            'parent_actions': self.steps, 'parent_updates': self.updates,
            'parent_batches': self.batches, 'parent_lineage': state.get('lineage'),
            'preserved': ['parameter_values', 'Adam_state', 'all_RNG_states', 'counters'],
            'changed': 'FP64 network, gradients and Adam moments; fresh full-game rollout',
            'exact_cross_source_reproduction': False}


def run(config, out, target_actions, seconds, device='cpu', resume=None, numeric_import=None,
        parent_source=None, parent_sha256=None):
    if type(target_actions) is not int or target_actions < 1 or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('positive real-action target and finite phase watchdog required')
    if (resume and numeric_import) or (bool(numeric_import) != bool(parent_source and parent_sha256)):
        raise ValueError('choose exact resume OR explicit numeric import with parent source/hash')
    started = time.monotonic()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', config)
    trainer = Trainer(config, device)
    trainer.failure_directory = out
    if resume:
        trainer.load(resume)
    if numeric_import:
        trainer.import_numeric_checkpoint(numeric_import, parent_source, parent_sha256)
    trainer.save(out / 'initial.resume.pt')
    stopped = [False]
    previous_handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    begin_steps = trainer.steps
    print({'event': 'trainer_ready', 'experiment': config['experiment_id'], 'device': device,
           'initial_actions': begin_steps, 'target_actions': target_actions}, flush=True)
    try:
        with (out / 'training.jsonl').open('x') as log:
            while trainer.steps < target_actions and time.monotonic() - started < seconds and not stopped[0]:
                row = trainer.ppo_batch()
                row.update(phase_wall_seconds=time.monotonic() - started, target_actions=target_actions)
                import json
                log.write(json.dumps(row) + '\n'); log.flush()
                if trainer.batches % config.get('checkpoint_batches', 16) == 0:
                    # Only latest recovery point within each phase; retain all milestone finals.
                    trainer.save(out / 'latest.resume.pt')
                if trainer.batches == 1 or trainer.batches % 16 == 0:
                    print({'event': 'progress', **row}, flush=True)
            trainer.save(out / 'final.resume.pt')
    finally:
        trainer.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    status = 'target_reached' if trainer.steps >= target_actions else 'interrupted' if stopped[0] else 'phase_limit'
    summary = {'status': status, 'experiment_id': config['experiment_id'], 'config': config,
               'source_id': source_id(), 'target_actions': target_actions, 'initial_actions': begin_steps,
               'lineage': trainer.lineage,
               'real_atomic_actions': trainer.steps, 'target_overshoot_actions': max(0, trainer.steps-target_actions),
               'policy_decisions': trainer.decision_count, 'stop_decisions': trainer.stop_decisions,
               'simulated_actions': 0, 'episodes': trainer.episodes, 'updates': trainer.updates,
               'training_seconds': trainer.elapsed, 'phase_wall_seconds': time.monotonic()-started,
               'accounting': 'real actions exclude STOP; full-game batch boundary overshoot is explicit'}
    write_json(out / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--out', required=True)
    parser.add_argument('--target-actions', required=True, type=int)
    parser.add_argument('--seconds', required=True, type=float, help='phase watchdog, not total experiment budget')
    parser.add_argument('--device', default='cpu'); parser.add_argument('--resume')
    parser.add_argument('--import-numeric'); parser.add_argument('--parent-source'); parser.add_argument('--parent-sha256')
    args = parser.parse_args()
    run(read_json(args.config), args.out, args.target_actions, args.seconds, args.device, args.resume,
        args.import_numeric, args.parent_source, args.parent_sha256)
