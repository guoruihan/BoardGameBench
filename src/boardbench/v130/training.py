"""Persistent imitation on replay-verified student states; PPO returns stay on-policy."""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v127.training import Trainer as BaseTrainer, load_demonstrations, run as base_run


class Trainer(BaseTrainer):
    def __init__(self, config, device='cpu'):
        if config.get('guidance_version') != '1.2.10':
            raise ValueError('explicit guidance version required')
        super().__init__(config, device)
        self.corrections = None
        if not 0 <= config['bc_weight'] <= 1 or not 0 <= config['correction_fraction'] <= 1:
            raise ValueError('invalid persistent imitation weights')
        chunks = []
        heldout = set()
        for item in config.get('correction_datasets', []):
            data, val = load_demonstrations({'dataset': item['path'], 'dataset_sha256': item['sha256']}, self.manifest)
            chunks.append(data); heldout |= val
        if chunks:
            self.corrections = {k: np.concatenate([d[k] for d in chunks]) for k in chunks[0]}
            if set(self.corrections['seed']) & heldout:
                raise ValueError('correction validation reused for fitting')
            if not (self.corrections['mask'].sum(axis=1)>1).any():
                raise ValueError('corrections contain only forced actions; no learning signal')
        if config['correction_fraction'] and self.corrections is None:
            raise ValueError('correction loss without admitted labels')
        self.train_seeds = [s for s in self.train_seeds if s not in heldout]
        if not self.train_seeds:
            raise ValueError('no online train seeds remain')

    def imitation_weight(self):
        # Unlike V127 this does not vanish after a fixed number of actions.
        return self.config['bc_weight']

    def demonstration_loss(self):
        fraction = self.config['correction_fraction']
        result = torch.zeros((), device=self.device, dtype=torch.float64)
        for data, weight in ((self.demo, 1-fraction), (self.corrections, fraction)):
            if not weight:
                continue
            ids = self.rng.integers(len(data['y']), size=min(256, len(data['y'])))
            logits, _ = self.model(torch.as_tensor(data['x'][ids], device=self.device, dtype=torch.float64),
                                   torch.as_tensor(data['mask'][ids], device=self.device))
            result = result + weight*F.cross_entropy(logits, torch.as_tensor(data['y'][ids], device=self.device))
        return result

    def import_sampling_continuation(self, spec):
        """Explicit new learning experiment, not bitwise old-algorithm continuation.

        Keep trained actor/critic, Adam and lifetime counters. First-round seeds
        are new paired RNG streams; later rounds preserve each arm's full RNG.
        """
        if self.actions or self.updates or self.optimizer.state:
            raise ValueError('parent import requires fresh trainer')
        path = Path(spec['checkpoint']); meta = read_json(path.with_suffix('.json'))
        if digest(path) != spec['sha256'] or meta['sha256'] != spec['sha256'] or meta['source_id'] != spec['source_id']:
            raise ValueError('parent hash/source mismatch')
        state = torch.load(path, map_location=self.device, weights_only=False)
        old = state['config']
        for key in ('hidden', 'reward_scale', 'dataset_sha256', 'split_sha256', 'gae_lambda',
                    'learning_rate', 'epochs', 'minibatch', 'target_kl', 'entropy_coef', 'batch_episodes'):
            if old[key] != self.config[key]:
                raise ValueError(f'parent changes protected field {key}')
        if state['version'] != 127 or state['source_id'] != spec['source_id']:
            raise ValueError('unsupported parent format/source')
        if any(not torch.isfinite(v).all() or v.dtype != torch.float64 for v in state['model'].values()):
            raise ValueError('finite FP64 parent required')
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        for key in ('episodes', 'actions', 'ppo_actions', 'batches', 'warm_batches', 'updates'):
            setattr(self, key, state['counters'][key])
        if old.get('guidance_version') == '1.2.10':
            if old['arm'] != self.config['arm'] or old['seed'] != self.config['seed']:
                raise ValueError('cannot swap arm or RNG across rounds')
            self.rng.bit_generator.state = state['rng_numpy']
            torch.set_rng_state(state['rng_torch'].cpu())
            if state['rng_cuda']:
                if not self.device.startswith('cuda'):
                    raise ValueError('CUDA round continuation requires CUDA')
                torch.cuda.set_rng_state_all([v.cpu() for v in state['rng_cuda']])
        self.lineage = {'kind': 'persistent_search_guidance', 'parent_checkpoint': str(path),
                        'sha256': spec['sha256'], 'parent_source_id': spec['source_id'],
                        'starting_ppo_actions': self.ppo_actions,
                        'parent_lineage': state['lineage'],
                        'rng': 'new paired seed' if old.get('guidance_version') != '1.2.10' else 'preserved',
                        'changed': 'persistent BC and student-state corrections; no teacher reward/value targets'}

    def evaluate(self, seeds):
        result = super().evaluate(seeds)
        # Deterministic diagnostic on fitting rows, NOT an unseen accuracy claim.
        diagnostics = {}
        with torch.no_grad():
            for name, data in (('original_fit', self.demo), ('correction_fit', self.corrections)):
                if data is None:
                    continue
                ids = np.linspace(0, len(data['y'])-1, min(2048, len(data['y'])), dtype=int)
                logits, _ = self.model(torch.as_tensor(data['x'][ids], device=self.device, dtype=torch.float64),
                                       torch.as_tensor(data['mask'][ids], device=self.device))
                y = torch.as_tensor(data['y'][ids], device=self.device)
                diagnostics[name] = {'n': len(ids), 'nll': float(F.cross_entropy(logits, y)),
                                     'action_agreement': float((logits.argmax(-1) == y).double().mean())}
        result['teacher_fit_diagnostic'] = diagnostics
        return result


def export_checkpoint(path, destination):
    """CPU teacher roll-in uses FP64 actor weights, never the old FP32 export."""
    path, destination = Path(path), Path(destination)
    meta = read_json(path.with_suffix('.json'))
    if digest(path) != meta['sha256']:
        raise ValueError('export parent hash mismatch')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['source_id'] != meta['source_id']:
        raise ValueError('export parent source mismatch')
    if destination.exists():
        receipt = read_json(destination.with_suffix('.json'))
        if receipt['parent_sha256'] != meta['sha256'] or receipt['sha256'] != digest(destination):
            raise ValueError('refusing changed export')
        return receipt
    weights = {k: v.numpy() for k, v in state['model'].items()}
    if any(v.dtype != np.float64 or not np.isfinite(v).all() for v in weights.values()):
        raise ValueError('export requires finite FP64 weights')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.partial')
    with temporary.open('wb') as stream:
        np.savez(stream, **weights)
    temporary.replace(destination)
    receipt = {'sha256': digest(destination), 'parent': str(path), 'parent_sha256': meta['sha256'],
               'parent_source_id': meta['source_id'], 'dtype': 'float64'}
    write_json(destination.with_suffix('.json'), receipt)
    return receipt


def run(config, out, device='cuda'):
    return base_run(config, out, device, trainer_class=Trainer)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); run(read_json(a.config), a.out, a.device)
