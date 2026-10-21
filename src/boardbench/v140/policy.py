"""Portable two-player inference, with separate planner memory for each seat."""
from pathlib import Path
import numpy as np
import torch

from boardbench.contracts import Solver
from boardbench.environments.harmonies_duo import HarmoniesDuo, player_view
from boardbench.v138.baseline import Planner, build_model
from .model import DuoPolicy
from .encoding import LAYOUT, action_map, encode_batch

FORMAT = 'harmonies_duo_policy_v140'


def export_policy(model, path, source_id=None, parent_sha256=None):
    value = dict(format=FORMAT, rules_version=HarmoniesDuo.rules_version, layout=LAYOUT,
                 hidden=model.solo.actor.in_features, method=model.solo.method,
                 strength=model.solo.strength, objective=model.objective,
                 source_id=source_id, parent_sha256=parent_sha256,
                 model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    torch.save(value, path)
    return value


def load_policy(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved['format'] != FORMAT or saved['rules_version'] != HarmoniesDuo.rules_version
            or saved['layout'] != LAYOUT):
        raise ValueError('duo policy artifact mismatch')
    model = DuoPolicy(build_model(saved['hidden'], saved['method'], saved['strength']), saved['objective'])
    if any(v.dtype != torch.float64 or not torch.isfinite(v).all() for v in saved['model'].values()):
        raise ValueError('finite FP64 weights required')
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval()


class DuoSolver(Solver):
    def __init__(self, model=None, device='cpu'):
        self.model, self.device = model, device
        self.planners = [Planner(), Planner()]

    def load_weights(self, path):
        self.model = load_policy(path, self.device)

    def start_episode(self, observation, context=None):
        self.planners = [Planner(), Planner()]

    def decide(self, observation, context=None):
        if observation['rules_version'] != HarmoniesDuo.rules_version or self.model is None:
            raise ValueError('duo policy and matching observation required')
        seat = observation['active_player']; mapping = action_map(observation)
        plan = self.planners[seat].vector(player_view(observation))
        x = encode_batch([observation], [plan])
        mask = np.array([[i in mapping for i in range(244)]])
        with torch.inference_mode():
            logits, _ = self.model(torch.as_tensor(x, device=self.device, dtype=torch.float64),
                                   torch.as_tensor(mask, device=self.device))
        return dict(mapping[int(logits.argmax(-1))])

    def save(self, directory): export_policy(self.model, Path(directory)/'weights.pt')
    def load(self, directory): self.load_weights(Path(directory)/'weights.pt')
