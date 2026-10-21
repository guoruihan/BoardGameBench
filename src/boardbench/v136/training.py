"""Anchored layout-skill warm-start, then unchanged natural-terminal PPO."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v133.training import COUNTERS
from boardbench.v134.training import Trainer as PlanTrainer, run as plan_run
from boardbench.v135.training import Trainer as AdviceTrainer
from .data import load

METHODS = ('control', 'plain', 'skills')


class Trainer(AdviceTrainer):
    checkpoint_version = 136

    def __init__(self, config, device='cpu'):
        if (config.get('skill_version') != '1.3.6' or config.get('skill_method') not in (*METHODS, 'manager')
                or config['method'] != 'sticky' or config['arm'] != ('manager' if config['skill_method'] == 'manager' else 'persistent')):
            raise ValueError('explicit skill method and consistent sticky planner required')
        # Reuse the same mature worker; only high-level choice changes in manager mode.
        super().__init__({**config, 'arm': 'persistent'}, device)
        self.config = deepcopy(config)
        self.skills = load(config['skill_dataset'], self.train_seeds)
        ids = np.flatnonzero(~self.skills['validation'])
        self.skill_groups = [ids[self.skills['group'][ids] == group] for group in np.unique(self.skills['group'][ids])]

    def skill_batch(self, count, rng):
        groups = rng.integers(len(self.skill_groups), size=count)
        ids = np.asarray([rng.choice(self.skill_groups[g]) for g in groups])
        key = 'ordinary' if self.config['skill_method'] == 'plain' else 'goal'
        x = np.concatenate((self.skills['x'][ids], self.skills[key][ids]), -1)
        return (torch.as_tensor(x, device=self.device, dtype=torch.float64),
                torch.as_tensor(self.skills['mask'][ids], device=self.device),
                torch.as_tensor(self.skills['y'][ids], device=self.device))

    def imitation_weight(self):
        if self.config['skill_method'] == 'control':
            return 0.
        elapsed = self.ppo_actions - self.config['skill_origin_actions']
        return self.config['skill_bc_weight'] * max(0., 1 - elapsed / self.config['skill_bc_decay_actions'])

    def demonstration_loss(self):
        x, mask, y = self.skill_batch(256, self.rng)
        return F.cross_entropy(self.model(x, mask)[0], y)

    def warm_start(self):
        if self.config['skill_method'] == 'control':
            return {'steps': 0, 'reason': 'continued sticky PPO control'}
        # Adapt only goal adapter + trust gate; do not overwrite mature terrain backbone.
        reference = deepcopy(self.model).eval()
        for p in reference.parameters():
            p.requires_grad_(False)
        trainable = {name: p.requires_grad for name, p in self.model.named_parameters()}
        for name, p in self.model.named_parameters():
            p.requires_grad_(name.startswith(('adapter.', 'gate.')))
        rates = [g['lr'] for g in self.optimizer.param_groups]
        for g in self.optimizer.param_groups:
            g['lr'] = self.config['skill_warmup_lr']
        rng = np.random.default_rng(self.config['seed'] + 136000)
        metrics = []
        try:
            for _ in range(self.config['skill_warmup_steps']):
                x, mask, y = self.skill_batch(256, rng)
                with torch.no_grad():
                    original, original_value = reference(x, mask)
                    target = original.softmax(-1)
                logits, value = self.model(x, mask)
                ce = F.cross_entropy(logits, y)
                kl = F.kl_div(logits.log_softmax(-1), target, reduction='batchmean')
                loss = ce + self.config['skill_anchor_kl'] * kl + (value - original_value).square().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite skill warm-start')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5, error_if_nonfinite=True)
                self.optimizer.step()
                metrics.append((float(ce.detach()), float(kl.detach())))
        finally:
            for name, p in self.model.named_parameters():
                p.requires_grad_(trainable[name])
            for g, rate in zip(self.optimizer.param_groups, rates):
                g['lr'] = rate
        return {'steps': len(metrics), 'first_ce': metrics[0][0] if metrics else None,
                'last_ce': metrics[-1][0] if metrics else None, 'last_kl': metrics[-1][1] if metrics else None,
                'trainable': 'adapter and gate only; mature body/actor/critic unchanged',
                'target': 'expert action conditioned on retrospectively achieved public goal' if self.config['skill_method'] != 'plain'
                          else 'same expert actions, current heuristic goals only; no achieved-goal label'}

    def import_sampling_continuation(self, spec):
        path = Path(spec['checkpoint'])
        meta = read_json(path.with_suffix('.json'))
        if digest(path) != spec['sha256'] or meta['sha256'] != spec['sha256'] or meta['source_id'] != spec['source_id']:
            raise ValueError('parent hash/source mismatch')
        saved = torch.load(path, map_location=self.device, weights_only=False)
        manager_branch = saved['version'] == 136 and self.config['skill_method'] == 'manager' and saved['config']['skill_method'] == 'skills'
        if saved['version'] == 136:
            if not manager_branch:
                return PlanTrainer.import_sampling_continuation(self, spec)
            expected = deepcopy(self.config)
            # Explicit new experiment, not an exact same-algorithm continuation.
            self.config = {**self.config, 'skill_method': 'skills', 'arm': 'persistent'}
            try:
                PlanTrainer.import_sampling_continuation(self, spec)
            finally:
                self.config = expected
            from .probe import bootstrap_manager
            result = bootstrap_manager(self)
            self.lineage.update(kind='current_worker_goal_value_initialization', manager_bootstrap=result)
            if self.config.get('trajectory_directory'):
                write_json(Path(self.config['trajectory_directory']).parent / 'manager_bootstrap.json', result)
            return
        if saved['version'] != 135 or saved['source_id'] != spec['source_id'] or saved['config']['method'] != 'sticky':
            raise ValueError('mature V135 sticky parent required')
        for key in ('hidden', 'reward_scale', 'dataset_sha256', 'excluded_train_seeds', 'advice_strength', 'plan_horizon'):
            if saved['config'].get(key) != self.config.get(key):
                raise ValueError('protected parent field: ' + key)
        old = read_json(saved['config']['split_manifest'])
        if digest(saved['config']['split_manifest']) != saved['config']['split_sha256'] or old['sets']['train'] != self.manifest['sets']['train']:
            raise ValueError('parent training pool changed')
        for key in ('model', 'manager', 'optimizer', 'critic_optimizer', 'manager_optimizer'):
            getattr(self, key).load_state_dict(saved[key])
        if any(not torch.isfinite(v).all() for v in self.model.state_dict().values()):
            raise ValueError('nonfinite mature parent')
        for group in self.optimizer.param_groups:
            group['lr'] = self.config['learning_rate']
        for key in COUNTERS[:6]:
            setattr(self, key, saved['counters'][key])
        self.lineage = {'kind': 'mature_sticky_skill_import', **spec, 'starting_ppo_actions': self.ppo_actions,
                        'preserved': 'full worker/manager/Adam; fresh paired RNG; true game objective',
                        'warm_start': self.warm_start()}

    def update(self, rows):
        result = super().update(rows)
        result['expert_bc_weight'] = self.imitation_weight()
        result['skill_method'] = self.config['skill_method']
        return result

    def finalize_diagnostics(self, out, status):
        path = Path(out) / 'execution_probe.json'
        if status == 'completed' and self.config['skill_method'] in ('control', 'skills') and not path.exists():
            from .probe import execution_probe
            write_json(path, execution_probe(self))


def run(config, out, device='cuda'):
    return plan_run(config, out, device, trainer_class=Trainer)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    run(read_json(args.config), args.out, args.device)
