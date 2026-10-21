"""Inference artifacts do not load trainers, datasets, or ancestral checkpoints."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from boardbench.artifacts.store import digest, write_json
from boardbench.contracts import Solver
from boardbench.solvers.neural import PolicyNetwork
from boardbench.v12.encoding import action_map, feature_vector
from boardbench.v134.plans import candidates, encode_plan, refresh_goal
from boardbench.v134.training import PlanPolicy
from boardbench.v135.training import AdvicePolicy

FORMAT = 'harmonies_solo_baseline_v138'


def build_model(hidden=256, method='sticky', strength=.6):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        base = PolicyNetwork(1346, 240, hidden).double()
        return AdvicePolicy(PlanPolicy(base).double(), method, strength).double()


def load_model(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved['format'] != FORMAT or saved['rules_version'] != 'harmonies_solo_a_v1'
            or saved['plan_horizon'] != 2 or saved['planner'] != 'persistent_top_public'):
        raise ValueError('unsupported baseline artifact')
    model = build_model(saved['hidden'], saved['method'], saved['strength'])
    if any(v.dtype != torch.float64 or not torch.isfinite(v).all() for v in saved['model'].values()):
        raise ValueError('finite FP64 inference weights required')
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved


class Planner:
    """Exactly the V137 control planner, independent per episode/player."""
    def __init__(self, horizon=2):
        self.horizon, self.active, self.started = horizon, None, 0

    def vector(self, obs):
        age = obs['turn_index'] - self.started
        old = self.active or []
        if (self.active is None or any(refresh_goal(obs, g) is None for g in old)
                or age >= (self.horizon if old else 1)):
            bank = candidates(obs)
            self.active = bank[1] if len(bank) > 1 else bank[0]
            self.started = obs['turn_index']
            age = 0
        return encode_plan(obs, self.active, age)


class BaselineSolver(Solver):
    def __init__(self, seed=0, device='cpu'):
        self.device, self.model, self.planner = str(device), None, Planner()

    def load_weights(self, path):
        self.model, self.metadata = load_model(path, self.device)

    def start_episode(self, observation, context=None):
        self.planner = Planner()

    def decide(self, observation, context=None):
        if self.model is None:
            raise RuntimeError('load exported baseline weights first')
        if observation['task_id'] != 'harmonies' or observation['rules_version'] != 'harmonies_solo_a_v1':
            raise ValueError('solo baseline requires the frozen solo setting')
        mapping = action_map(observation, 'slots240_v1')
        x = np.concatenate((np.asarray(feature_vector(observation), dtype=np.float32),
                            self.planner.vector(observation)))
        mask = np.array([i in mapping for i in range(240)])
        with torch.inference_mode():
            logits, _ = self.model(torch.as_tensor(x, dtype=torch.float64, device=self.device)[None],
                                   torch.as_tensor(mask, device=self.device)[None])
        return deepcopy(mapping[int(logits.argmax(-1))])

    def save(self, directory):
        torch.save(self.metadata, Path(directory) / 'weights.pt')

    def load(self, directory):
        self.load_weights(Path(directory) / 'weights.pt')


def export(checkpoint, out):
    """Only trusted local training checkpoints may be passed to this command."""
    source = torch.load(checkpoint, map_location='cpu', weights_only=False)
    c = source['config']
    if (source['version'] != 137 or c['selection_method'] != 'control'
            or c['arm'] != 'persistent' or c['plan_horizon'] != 2):
        raise ValueError('only the frozen V137 control is released')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    saved = dict(format=FORMAT, rules_version='harmonies_solo_a_v1', hidden=c['hidden'],
                 method=c['method'], strength=c['advice_strength'], plan_horizon=2,
                 planner='persistent_top_public', encoding='cards1346_slots240_fp64',
                 source_id=source['source_id'], parent_sha256=digest(checkpoint),
                 selection='control_a, chosen by development score before fresh evaluation',
                 model={k: v.detach().cpu().clone() for k, v in source['model'].items()})
    torch.save(saved, out / 'weights.pt')
    loaded, _ = load_model(out / 'weights.pt')
    if any(not torch.equal(v, loaded.state_dict()[k]) for k, v in source['model'].items()):
        raise RuntimeError('export changed weights')
    write_json(out / 'manifest.json', {**{k: v for k, v in saved.items() if k != 'model'},
                                     'weights': 'weights.pt', 'sha256': digest(out / 'weights.pt')})
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    print(export(args.checkpoint, args.out))
