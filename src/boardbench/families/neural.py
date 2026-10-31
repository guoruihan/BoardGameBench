"""Public candidate actor-critic with an explicit frozen heuristic prior."""
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .base import FEATURE_DIM
from .play import action_features

FORMAT = 'boardbench_family_actor_critic_v1'


def prepare(rows, prior_weights):
    raw = np.asarray(rows, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != FEATURE_DIM or not np.isfinite(raw).all():
        raise ValueError('Invalid feature matrix')
    inputs = np.sign(raw)*np.log1p(np.abs(raw))/5.
    prior = raw @ np.asarray(prior_weights, dtype=np.float32)
    prior = (prior - prior.max()) / max(float(prior.std()), .25)
    prior = np.clip(prior*3., -30., 0.)
    return inputs, prior


class CandidateActorCritic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.encoder = nn.Sequential(nn.Linear(FEATURE_DIM, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(hidden*2, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.actor.weight)
        nn.init.zeros_(self.actor.bias)

    def forward(self, x, prior, mask):
        if x.ndim != 3 or mask.shape != x.shape[:2] or not mask.any(dim=1).all():
            raise ValueError('Each state requires a nonempty aligned legal-action mask')
        hidden = self.encoder(x)
        logits = (self.actor(hidden).squeeze(-1)+prior).masked_fill(~mask, -torch.inf)
        mean = (hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1, keepdim=True)
        maximum = hidden.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(1).values
        values = self.critic(torch.cat((mean, maximum), -1)).squeeze(-1)
        return logits, values


def collate(records, device):
    count = len(records)
    largest = max(len(r['x']) for r in records)
    x = np.zeros((count, largest, FEATURE_DIM), np.float32)
    prior = np.zeros((count, largest), np.float32)
    mask = np.zeros((count, largest), bool)
    for i, r in enumerate(records):
        n = len(r['x'])
        x[i, :n], prior[i, :n], mask[i, :n] = r['x'], r['prior'], True
    return (torch.as_tensor(x, device=device), torch.as_tensor(prior, device=device),
            torch.as_tensor(mask, device=device))


def save_model(path, model, cls, prior_weights, metadata=None):
    saved = dict(format=FORMAT, game=cls.game_id, config=cls.config_id, rules_version=cls.rules_version,
                 hidden=model.hidden, prior_weights=list(prior_weights), metadata=metadata or {},
                 model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.part')
    torch.save(saved, temporary)
    temporary.replace(path)


def load_model(path, cls, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    expected = dict(format=FORMAT, game=cls.game_id, config=cls.config_id, rules_version=cls.rules_version)
    if any(saved.get(k) != v for k, v in expected.items()):
        raise ValueError('Neural policy game/config/schema mismatch')
    if len(saved['prior_weights']) != FEATURE_DIM or not all(math.isfinite(w) for w in saved['prior_weights']):
        raise ValueError('Invalid policy prior')
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in saved['model'].values()):
        raise ValueError('Expected finite FP32 model weights')
    model = CandidateActorCritic(saved['hidden'])
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval(), saved


class NeuralPolicy:
    def __init__(self, cls, spec, seed=0, model=None, prior_weights=None, record=False):
        torch.set_num_threads(1)
        self.cls = cls
        if model is None:
            model, saved = load_model(spec['path'], cls)
            prior_weights = saved['prior_weights']
        self.model, self.prior_weights = model.eval(), prior_weights
        self.rng = np.random.default_rng(seed)
        self.stochastic = spec.get('stochastic', False)
        self.record, self.records = record, []

    def decide(self, observation, actions):
        x, prior = prepare(action_features(self.cls, observation, actions), self.prior_weights)
        row = dict(x=x, prior=prior)
        with torch.inference_mode():
            logits, value = self.model(*collate([row], 'cpu'))
            logp = torch.log_softmax(logits[0], 0).numpy()
        if self.stochastic:
            probabilities = np.exp(logp.astype(np.float64))
            probabilities /= probabilities.sum()
            index = int(self.rng.choice(len(actions), p=probabilities))
        else:
            index = int(np.argmax(logp))
        if self.record:
            self.records.append(dict(**row, action=index, old_logp=float(logp[index]),
                                     value=float(value[0])))
        return actions[index]


def finish_returns(records, terminal_reward, gamma=.995, gae_lambda=.95):
    """Only same-player decisions belong to this chain, including extra turns."""
    advantage, next_value = 0., 0.
    for i in reversed(range(len(records))):
        reward = terminal_reward if i == len(records)-1 else 0.
        delta = reward + gamma*next_value - records[i]['value']
        advantage = delta + gamma*gae_lambda*advantage
        records[i]['advantage'] = advantage
        records[i]['return'] = advantage + records[i]['value']
        next_value = records[i]['value']
    return records
