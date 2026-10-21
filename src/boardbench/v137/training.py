"""Current-worker goal-value initialization and frozen/joint on-policy training."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v12.training import policy_distribution
from boardbench.v133.training import COUNTERS
from boardbench.v134.training import Trainer as PlanTrainer, run as plan_run
from boardbench.v136.training import Trainer as SkillTrainer
from .data import load as load_goals

METHODS = ('control', 'selector', 'joint', 'refresh')


class Trainer(SkillTrainer):
    checkpoint_version = 137

    def __init__(self, config, device='cpu'):
        mode = config.get('selection_method')
        if (config.get('selection_version') != '1.3.7' or mode not in METHODS
                or config['arm'] != ('persistent' if mode == 'control' else 'manager')
                or config['skill_method'] != ('skills' if mode == 'control' else 'manager')
                or config['goal_fit_updates'] < 1 or config['goal_temperature'] <= 0
                or config['manager_warmup_actions'] < 0 or config['joint_skill_bc'] < 0):
            raise ValueError('explicit V137 selection mode and consistent planner required')
        super().__init__(config, device)
        self.goal_data = load_goals(config['goal_dataset'], self.train_seeds) if config.get('goal_dataset') else None
        if mode == 'selector':
            self.model.requires_grad_(False)

    def imitation_weight(self):
        return self.config['joint_skill_bc'] if self.config['selection_method'] in ('joint', 'refresh') else 0.

    def bootstrap_selector(self):
        if self.goal_data is None:
            raise ValueError('goal value data required before manager training')
        d = self.goal_data
        tensor = lambda a, dtype=torch.float64: torch.as_tensor(a, device=self.device, dtype=dtype)
        x, plans, mask = tensor(d['x']), tensor(d['plans']), tensor(d['mask'], torch.bool)
        fit = torch.as_tensor(np.flatnonzero(~d['validation']), device=self.device)
        q = tensor(d['q_fit'].mean(-1))  # Independent verification futures NEVER enter the loss.
        with torch.no_grad():
            prior, _ = self.manager(x, plans, mask)
            target = (prior + q/self.config['goal_temperature']).masked_fill(~mask, -1e9).softmax(-1)
            remaining = ((q-tensor(d['start_score'])[:, None])*target).sum(-1)/150.
        losses = []
        for _ in range(self.config['goal_fit_updates']):
            logits, value = self.manager(x[fit], plans[fit], mask[fit])
            loss = (F.kl_div(logits.log_softmax(-1), target[fit], reduction='batchmean')
                    + .5*(value-remaining[fit]).square().mean())
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite goal value initialization')
            self.manager_optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.manager.parameters(), .5, error_if_nonfinite=True)
            self.manager_optimizer.step(); losses.append(float(loss.detach()))
        with torch.no_grad():
            logits, _ = self.manager(x, plans, mask); choices = logits.argmax(-1).cpu().numpy()
        audit = d['q_verify'].mean(-1); ids = np.arange(len(choices)); baseline = np.where(d['mask'][:, 1], 1, 0)
        gains = audit[ids, choices]-audit[ids, baseline]
        from .data import interval
        result = {'updates': len(losses), 'first_loss': losses[0], 'last_loss': losses[-1],
                  'data': self.config['goal_dataset'], 'objective': 'prior reweighted by current-worker Q; remaining-score critic',
                  'verification_used_for_training': False}
        for name, selected in (('fit_states', ~d['validation']), ('heldout_states', d['validation'])):
            result[name] = interval([float(gains[(d['seed']==s) & selected].mean()) for s in sorted(set(d['seed'][selected].tolist()))])
        return result

    def import_sampling_continuation(self, spec):
        path = Path(spec['checkpoint']); meta = read_json(path.with_suffix('.json'))
        if digest(path) != spec['sha256'] or meta['sha256'] != spec['sha256'] or meta['source_id'] != spec['source_id']:
            raise ValueError('parent hash/source mismatch')
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved['version'] == 137:
            refresh = saved['config'].get('goal_dataset') != self.config.get('goal_dataset')
            if refresh and self.config['selection_method'] != 'refresh':
                raise ValueError('only refresh arm may change goal labels')
            expected = deepcopy(self.config)
            self.config['goal_dataset'] = saved['config'].get('goal_dataset')
            if refresh and saved['config']['selection_method'] == 'joint':
                # Share the bit-identical first stage; fork only when labels change.
                self.config['selection_method'] = 'joint'
            try:
                PlanTrainer.import_sampling_continuation(self, spec)
            finally:
                self.config = expected
            if refresh:
                self.lineage['goal_refresh'] = self.bootstrap_selector()
                self._publish_bootstrap('refreshed')
            return
        if saved['version'] != 136 or saved['source_id'] != spec['source_id'] or saved['config']['skill_method'] != 'skills':
            raise ValueError('completed V136 skills parent required')
        for key in ('hidden', 'reward_scale', 'dataset_sha256', 'excluded_train_seeds', 'skill_dataset', 'plan_horizon', 'advice_strength'):
            if saved['config'].get(key) != self.config.get(key):
                raise ValueError('protected parent field: '+key)
        old = read_json(saved['config']['split_manifest'])
        if digest(saved['config']['split_manifest']) != saved['config']['split_sha256'] or old['sets']['train'] != self.manifest['sets']['train']:
            raise ValueError('parent training pool changed')
        self.model.load_state_dict(saved['model'])
        if any(v.dtype != torch.float64 or not torch.isfinite(v).all() for v in self.model.state_dict().values()):
            raise ValueError('finite FP64 parent required')
        for name in ('optimizer', 'critic_optimizer'):
            getattr(self, name).load_state_dict(saved[name])
        for group in self.optimizer.param_groups:
            group['lr'] = self.config['learning_rate']
        for key in COUNTERS[:6]:
            setattr(self, key, saved['counters'][key])
        # Use the mature worker representation, not the old untrained V134 manager body.
        self.manager.body.load_state_dict(self.model.body.state_dict())
        self.manager.critic.load_state_dict(self.model.critic.state_dict())
        self.lineage = {'kind': 'v137_goal_selection_import', **spec, 'starting_ppo_actions': self.ppo_actions,
                        'preserved': 'worker and its Adam/counters; fresh paired RNG; fresh manager on worker features'}
        if self.config['selection_method'] != 'control':
            self.lineage['goal_initialization'] = self.bootstrap_selector()
            self._publish_bootstrap('initial')

    def _publish_bootstrap(self, label):
        if self.config.get('trajectory_directory'):
            write_json(Path(self.config['trajectory_directory']).parent/('goal_'+label+'.json'), self.lineage)

    def update(self, rows):
        mode = self.config['selection_method']
        freeze = mode == 'selector' or (mode in ('joint', 'refresh') and
                  self.ppo_actions-self.config['selection_origin_actions'] < self.config['manager_warmup_actions'])
        if freeze:
            x, mask, action, old, *_ = self.tensors(rows)
            with torch.no_grad():
                logits, _ = self.model(x, mask)
                error = float(((policy_distribution(logits).log_prob(action)-old).exp()-1).abs().max())
            if error > 1e-4:
                raise RuntimeError('frozen worker behavior probability mismatch')
            result = {**self.update_manager(), 'initial_ratio_max_error': error,
                      'actor_change_l2': 0., 'critic_change_l2': 0., 'advice_gate_change': 0., 'expert_bc_weight': 0.}
        else:
            result = super().update(rows)
        result.update(selection_method=mode, worker_frozen=freeze)
        return result

    def finalize_diagnostics(self, out, status):
        if status == 'completed':
            from boardbench.v136.probe import execution_probe
            path = Path(out)/'execution_probe.json'
            if not path.exists():
                write_json(path, execution_probe(self))


def run(config, out, device='cuda'):
    return plan_run(config, out, device, trainer_class=Trainer)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--out', required=True); p.add_argument('--device', default='cuda')
    a = p.parse_args(); run(read_json(a.config), a.out, a.device)
