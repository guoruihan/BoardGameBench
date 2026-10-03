"""Optional PyTorch masked policy; greedy inference always uses saved weights."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from boardbench.contracts import Solver
from boardbench.environments import TASKS
from boardbench.environments.numerical import ACTION_COUNTS, ENCODING_VERSION, encode, decode, features
from boardbench.solvers.action_features import policy_features, WIDTHS, ACTION_FEATURE_VERSION


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, action_count, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_size, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, action_count)
        self.critic = nn.Linear(hidden, 1)
        nn.init.normal_(self.actor.weight, std=.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self, x, mask):
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("terminal/all-zero mask must not be sampled")
        z = self.body(x)
        return self.actor(z).masked_fill(~mask, -1e9), self.critic(z).squeeze(-1)


class ActionNetwork(nn.Module):
    def __init__(self, global_size, action_count, local_size, hidden):
        super().__init__()
        self.global_size, self.action_count, self.local_size = global_size, action_count, local_size
        self.actor = nn.Sequential(nn.Linear(local_size, hidden), nn.Tanh(),
                                   nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(nn.Linear(global_size, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.normal_(self.actor[-1].weight, std=.01)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, x, mask):
        if not torch.all(mask.any(dim=-1)):
            raise ValueError('terminal/all-zero mask must not be sampled')
        local = x[:, self.global_size:].reshape(-1, self.action_count, self.local_size)
        logits = self.actor(local).squeeze(-1).masked_fill(~mask, -1e9)
        return logits, self.critic(x[:, :self.global_size]).squeeze(-1)


class NeuralSolver(Solver):
    def __init__(self, seed, task_id, hidden=128, device="cpu", network='flat_mlp'):
        self.task_id, self.hidden, self.device = task_id, hidden, str(device)
        self.network = network
        initial = TASKS[task_id].factory().observe()
        self.input_size = len(policy_features(initial, network))
        self.encoding_version = ENCODING_VERSION if network=='flat_mlp' else ACTION_FEATURE_VERSION
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed % (2**63))
            self.model = (PolicyNetwork(self.input_size, ACTION_COUNTS[task_id], hidden) if network=='flat_mlp'
                          else ActionNetwork(len(features(initial)), ACTION_COUNTS[task_id], WIDTHS[task_id], hidden))
        self.model.to(device).eval()
        self.training = {}

    def decide(self, observation, context):
        data = encode(observation)
        x = torch.as_tensor(np.asarray(policy_features(observation, self.network), dtype=np.float32), device=self.device).unsqueeze(0)
        mask = torch.as_tensor(np.asarray(data["action_mask"], dtype=bool), device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits, _ = self.model(x, mask)
            action = int(logits.argmax(-1).item())
        return decode(observation, action)

    def save_weights(self, path):
        torch.save({"format_version": 1, "task_id": self.task_id,
                    "rules_version": TASKS[self.task_id].spec["version"],
                    "encoding_version": self.encoding_version, "hidden": self.hidden, 'network': self.network,
                    "input_size": self.input_size, "training": self.training,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}, path)

    def load_weights(self, path):
        artifact = torch.load(path, map_location=self.device, weights_only=True)
        for key, value in {"format_version": 1, "task_id": self.task_id,
                           "rules_version": TASKS[self.task_id].spec["version"],
                           "encoding_version": self.encoding_version, "hidden": self.hidden,
                           "input_size": self.input_size}.items():
            if artifact[key] != value:
                raise ValueError(f"policy artifact mismatch: {key}")
        if artifact.get('network', 'flat_mlp') != self.network:
            raise ValueError('policy artifact mismatch: network')
        self.model.load_state_dict(artifact["state_dict"], strict=True)
        self.model.eval()
        self.training = deepcopy(artifact["training"])

    def save(self, directory):
        self.save_weights(Path(directory) / "weights.pt")

    def load(self, directory):
        self.load_weights(Path(directory) / "weights.pt")
