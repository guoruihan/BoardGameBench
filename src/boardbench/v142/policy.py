"""Portable curriculum policy. Not silently applicable to official solo or duo."""
from pathlib import Path
import numpy as np
import torch

from boardbench.contracts import Solver
from boardbench.environments.harmonies_market import HarmoniesMarket
from boardbench.v138.baseline import Planner, build_model
from boardbench.v140.model import DuoPolicy
from .encoding import LAYOUT, ACTIONS, action_map, encode_batch

FORMAT = 'harmonies_random_market_policy_v142'


def export_policy(model, path, config):
    value = dict(format=FORMAT, rules_version=HarmoniesMarket.rules_version, layout=LAYOUT,
        hidden=model.solo.actor.in_features, method=model.solo.method, strength=model.solo.strength,
        objective='score', source_id=config.get('source_id'), initialization=config['initialization'],
        parent_sha256=config['parent_sha256'], duo_parent_sha256=config['duo_parent_sha256'],
        model={k: v.detach().cpu().clone() for k,v in model.state_dict().items()})
    torch.save(value, path)
    return value


def load_policy(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if (saved['format'] != FORMAT or saved['rules_version'] != HarmoniesMarket.rules_version
            or saved['layout'] != LAYOUT or saved['objective'] != 'score'):
        raise ValueError('curriculum policy schema mismatch')
    model = DuoPolicy(build_model(saved['hidden'], saved['method'], saved['strength']), 'score')
    if any(v.dtype != torch.float64 or not torch.isfinite(v).all() for v in saved['model'].values()):
        raise ValueError('finite FP64 weights required')
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval()


class MarketSolver(Solver):
    def __init__(self, model=None, device='cpu'):
        self.model, self.device, self.planner = model, device, Planner()

    def start_episode(self, observation, context=None): self.planner = Planner()
    def load_weights(self, path): self.model = load_policy(path, self.device)

    def decide(self, observation, context=None):
        if observation['rules_version'] != HarmoniesMarket.rules_version or self.model is None:
            raise ValueError('matching curriculum observation required')
        mapping = action_map(observation)
        features = encode_batch([observation], [self.planner.vector(observation)])
        mask = np.array([[i in mapping for i in range(ACTIONS)]])
        with torch.inference_mode():
            logits, _ = self.model(torch.as_tensor(features, dtype=torch.float64, device=self.device),
                                   torch.as_tensor(mask, device=self.device))
        return dict(mapping[int(logits.argmax(-1))])

    def save(self, directory): raise NotImplementedError('export_policy requires initialization provenance')
    def load(self, directory): self.load_weights(Path(directory)/'weights.pt')
