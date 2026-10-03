"""No-chance animal endgames, value-gap fitting and held-out-state diagnosis."""
import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from boardbench.artifacts.store import read_json, write_json
from boardbench.environments.harmonies import Harmonies
from .encoding import STOP, decode, index
from .training import Trainer


def exact_values(snapshot, depth=2):
    env = Harmonies(); obs = env.load_state(snapshot)
    if sum(c['stack_id'] == 0 for c in obs['board']) > 2 or obs['terminated']:
        raise ValueError('oracle only accepts live, no-refill final-turn states')
    def value(state, remaining):
        current = state.score_breakdown()['total']
        if not remaining or state.s['terminated']:
            return current
        return max([current] + [value(after(state, a), remaining - 1) for a in state.legal_actions()])
    def after(state, action):
        other = deepcopy(state)
        private = other._private_state()
        other.step(action)
        assert other._private_state() == private, 'oracle encountered a random future event'
        return other
    result = {index(obs, STOP, 'slots240_v1', True): obs['score_breakdown']['total']}
    for action in obs['legal_actions']:
        result[index(obs, action, 'slots240_v1')] = value(after(env, action), depth - 1)
    return obs, result


def diagnose(snapshots, config, device='cpu', max_seconds=120):
    selected = []
    fingerprints = set()
    for snapshot in snapshots:
        probe = Harmonies(); observation = probe.load_state(snapshot)
        if not any(a.get('type') == 'place_animal' for a in observation['legal_actions']):
            continue  # Do not pass an animal-planning diagnosis on terrain-only cases.
        key = repr(snapshot['state']['board']) + repr(snapshot['state']['active_cards'])
        if key not in fingerprints:
            selected.append(snapshot); fingerprints.add(key)
        if len(selected) == 12:
            break
    if len(selected) < 4:
        return {'status': 'insufficient_endgames', 'count': len(selected), 'capability_verified': False}
    trainer = Trainer(config, device)
    cases = [exact_values(s) for s in selected]
    split = max(2, len(cases) // 2)
    dataset = []
    for obs, values in cases[:split]:
        data = trainer.solver.encode(obs)
        optimum = max(values.values())
        for action, score in values.items():
            if score == optimum:
                dataset.append({'features': data['features'], 'mask': data['action_mask'], 'action': action})
    def gaps(part):
        result = []
        for obs, values in part:
            data = trainer.solver.encode(obs)
            with torch.no_grad():
                logits, _ = trainer.solver.model(torch.tensor(np.asarray(data['features'], dtype=np.float32), device=device)[None],
                                                 torch.tensor(data['action_mask'], device=device)[None])
            chosen = int(logits.argmax(-1).item())
            result.append(max(values.values()) - values[chosen])
        return result
    initial = {'train': gaps(cases[:split]), 'heldout': gaps(cases[split:])}
    deadline = time.monotonic() + max_seconds
    curves = []
    for batch in range(256):
        metrics = trainer.bc_batch(dataset)
        if batch % 16 == 0:
            curves.append({'batch': batch + 1, 'loss': metrics['loss'], 'train_gap': gaps(cases[:split])})
            if max(curves[-1]['train_gap']) == 0:
                break
        if time.monotonic() >= deadline:
            break
    final = {'train': gaps(cases[:split]), 'heldout': gaps(cases[split:])}
    return {'status': 'passed' if max(final['train']) == 0 else 'fit_failed',
            'initial': initial, 'final': final, 'curve': curves, 'cases': len(cases),
            'oracle': 'exact best score within two legal actions or STOP, final turn, no random future',
            'boundary': 'auxiliary action-limited diagnosis, not an optimal full-game or wall-time claim',
            'learning_evidence': any(gap > 0 for gap in initial['train']) and max(final['train']) == 0,
            'capability_verified': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endgames', required=True); p.add_argument('--config', required=True)
    p.add_argument('--out', required=True); p.add_argument('--device', default='cpu')
    args = p.parse_args()
    write_json(args.out, diagnose(read_json(args.endgames), read_json(args.config), args.device))


if __name__ == '__main__':
    main()
