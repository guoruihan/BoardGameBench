"""Optional PyTorch masked policy; greedy inference always uses saved weights."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from boardbench.contracts import Solver
from boardbench.environments import TASKS
from boardbench.environments.numerical import ACTION_COUNTS, ENCODING_VERSION, encode, decode, features


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


class NeuralSolver(Solver):
    def __init__(self, seed, task_id, hidden=128, device="cpu"):
        self.task_id, self.hidden, self.device = task_id, hidden, str(device)
        self.input_size = len(features(TASKS[task_id].factory().observe()))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed % (2**63))
            self.model = PolicyNetwork(self.input_size, ACTION_COUNTS[task_id], hidden)
        self.model.to(device).eval()
        self.training = {}

    def decide(self, observation, context):
        data = encode(observation)
        x = torch.as_tensor(np.asarray(data["features"], dtype=np.float32), device=self.device).unsqueeze(0)
        mask = torch.as_tensor(np.asarray(data["action_mask"], dtype=bool), device=self.device).unsqueeze(0)
        with torch.inference_mode():
            logits, _ = self.model(x, mask)
            action = int(logits.argmax(-1).item())
        return decode(observation, action)

    def save_weights(self, path):
        torch.save({"format_version": 1, "task_id": self.task_id,
                    "rules_version": TASKS[self.task_id].spec["version"],
                    "encoding_version": ENCODING_VERSION, "hidden": self.hidden,
                    "input_size": self.input_size, "training": self.training,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}}, path)

    def load_weights(self, path):
        artifact = torch.load(path, map_location=self.device, weights_only=True)
        for key, value in {"format_version": 1, "task_id": self.task_id,
                           "rules_version": TASKS[self.task_id].spec["version"],
                           "encoding_version": ENCODING_VERSION, "hidden": self.hidden,
                           "input_size": self.input_size}.items():
            if artifact[key] != value:
                raise ValueError(f"policy artifact mismatch: {key}")
        self.model.load_state_dict(artifact["state_dict"], strict=True)
        self.model.eval()
        self.training = deepcopy(artifact["training"])

    def save(self, directory):
        self.save_weights(Path(directory) / "weights.pt")

    def load(self, directory):
        self.load_weights(Path(directory) / "weights.pt")
