"""Compact fixed-head policy with strict, independent input/output versions."""
from pathlib import Path

import numpy as np
import torch

from boardbench.contracts import Solver
from boardbench.environments.harmonies import Harmonies
from boardbench.solvers.neural import PolicyNetwork
from .encoding import COUNTS, decode, encode


class CompactSolver(Solver):
    def __init__(self, seed, hidden=128, action_encoding='slots240_v1',
                 observation_encoding='cards_v1', allow_stop=True, device='cpu', time_conditioning=False):
        self.device = str(device)
        self.params = dict(hidden=hidden, action_encoding=action_encoding,
                           observation_encoding=observation_encoding, allow_stop=allow_stop,
                           time_conditioning=time_conditioning)
        data = self.encode(Harmonies().observe())
        self.metadata = dict(format_version=2, task_id='harmonies',
                             rules_version=Harmonies.rules_version, network='compact_mlp_v1',
                             input_size=len(data['features']),
                             action_count=COUNTS[action_encoding] + int(allow_stop), **self.params)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed % (2**63))
            self.model = PolicyNetwork(self.metadata['input_size'], self.metadata['action_count'], hidden)
        self.model.to(device).eval()
        self.training = {}

    def encode(self, observation, **timing):
        if not self.params['time_conditioning']:
            timing = {}  # Full-game warm starts must not see untrained nonzero time columns at deployment.
        return encode(observation, self.params['action_encoding'], self.params['observation_encoding'],
                      self.params['allow_stop'], **timing)

    def decide(self, observation, context):
        timing = {}
        if getattr(context, 'time_budget_seconds', None) is not None:
            timing = dict(remaining_seconds=context.remaining_seconds(),
                          time_budget_seconds=context.time_budget_seconds)
        data = self.encode(observation, **timing)
        with torch.inference_mode():
            logits, _ = self.model(torch.tensor(np.asarray(data['features'], dtype=np.float32),
                                                 device=self.device)[None],
                                   torch.tensor(data['action_mask'], device=self.device)[None])
        return decode(observation, int(logits.argmax(-1).item()), self.params['action_encoding'],
                      self.params['allow_stop'])

    def save_weights(self, path):
        torch.save({**self.metadata, 'training': self.training,
                    'state_dict': {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}, path)

    def load_weights(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=True)
        if any(saved.get(k) != v for k, v in self.metadata.items()):
            raise ValueError('compact artifact metadata mismatch')
        self.model.load_state_dict(saved['state_dict'], strict=True)
        self.training = saved['training']
        self.model.eval()

    def save(self, directory):
        self.save_weights(Path(directory) / 'weights.pt')

    def load(self, directory):
        self.load_weights(Path(directory) / 'weights.pt')
