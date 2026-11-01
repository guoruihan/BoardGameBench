"""Full-public-state residual actor/critic initialized around a frozen policy.

The frozen policy and optional 24-dimensional features are advice, not a
replacement for the complete public state/action tree. No engine is accepted.
"""
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .neural import prepare, collate
from .play import action_features, make_policy
from .registry import game_class
from .rich_neural import ContextActorCritic, batch_inputs
from .rich_observation import PublicTreeEncoder

FORMAT = 'boardbench_guided_public_tree_v1'


class GuidedActorCritic(nn.Module):
    def __init__(self, vocabulary_size, hidden=64, auxiliary_count=0):
        super().__init__()
        self.hidden, self.auxiliary_count = hidden, auxiliary_count
        self.core = ContextActorCritic(vocabulary_size, hidden)
        self.feature_actor = nn.Sequential(nn.Linear(24, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.auxiliary = nn.Linear(2*hidden, auxiliary_count) if auxiliary_count else None
        # Initial greedy decisions exactly retain the frozen prior.
        for layer in (self.core.actor[-1], self.feature_actor[-1]):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, records, device='cpu'):
        sr, so, ar, ao, ast, n = batch_inputs(records, device)
        context = self.core.context_norm(self.core.state(sr, so, n))
        choices = self.core.choice_norm(self.core.action(ar, ao, len(ast)))
        own = context[ast]
        x = torch.as_tensor(np.concatenate([r['features'] for r in records]), device=device)
        prior = torch.as_tensor(np.concatenate([r['prior'] for r in records]), device=device)
        logits = (self.core.actor(torch.cat((own, choices, own*choices), -1)).squeeze(-1)
                  + self.feature_actor(x).squeeze(-1) + prior)
        counts = [len(r['prior']) for r in records]
        padded = nn.utils.rnn.pad_sequence(logits.split(counts), batch_first=True,
                                           padding_value=-torch.inf)
        values = self.core.critic(context).squeeze(-1)
        auxiliary = self.auxiliary(context) if self.auxiliary is not None else None
        return padded, values, auxiliary


def base_policy(game, spec, seed=0):
    # The actual played policy owns any repetition guard, not the advice call.
    spec = dict(spec); spec.pop('decision_guard', None)
    return make_policy(game_class(game), spec, seed)


def encode_record(game, encoder, base, observation, actions, anchor_scale=1.):
    if not actions: raise ValueError('Empty legal candidates')
    cls = game_class(game)
    features = np.asarray(action_features(cls, observation, actions), np.float32)
    normalized = np.sign(features)*np.log1p(np.abs(features))/5.
    if hasattr(base, 'prior_weights') and hasattr(base, 'model'):
        x, prior = prepare(features, base.prior_weights)
        with torch.inference_mode(): logits, _ = base.model(*collate([dict(x=x, prior=prior)], 'cpu'))
        prior = logits[0].numpy().astype(np.float32).copy()
        prior -= prior.max()
        prior = np.maximum(prior, -30.)
    else:
        action = base.decide(observation, actions)
        prior = np.full(len(actions), -3.-math.log(len(actions)), np.float32)
        prior[actions.index(action)] = 0.
    encoded, owners = encoder.encode_actions(actions)
    return dict(state=encoder.encode(observation), actions=encoded, owners=owners,
                features=normalized, prior=np.asarray(prior*anchor_scale, np.float32))


def save_model(path, model, encoder, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(format=FORMAT, encoding=encoder.metadata, hidden=model.hidden,
                   auxiliary_count=model.auxiliary_count, metadata=metadata,
                   model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    temporary = path.with_suffix('.part'); torch.save(payload, temporary); temporary.replace(path)


def load_model(path, game, device='cpu', expected_hash=None):
    path = Path(path)
    if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest()!=expected_hash:
        raise ValueError('Guided model checksum mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    encoder = PublicTreeEncoder(game)
    if payload.get('format')!=FORMAT or payload.get('encoding')!=encoder.metadata:
        raise ValueError('Guided checkpoint game/rules/encoding mismatch')
    if any(v.dtype!=torch.float32 or not torch.isfinite(v).all() for v in payload['model'].values()):
        raise ValueError('Expected finite FP32 weights')
    model = GuidedActorCritic(encoder.metadata['vocabulary_size'], payload['hidden'], payload['auxiliary_count'])
    model.load_state_dict(payload['model'], strict=True)
    return model.to(device).eval(), encoder, payload['metadata']


class GuidedPolicy:
    def __init__(self, game, spec, seed=0, model=None, encoder=None, metadata=None, record=False):
        torch.set_num_threads(1)
        if model is None:
            model, encoder, metadata = load_model(spec['path'], game, expected_hash=spec.get('sha256'))
        self.game, self.model, self.encoder, self.metadata = game, model.eval(), encoder, metadata
        self.base = base_policy(game, metadata['base_spec'], seed)
        self.rng = np.random.default_rng(seed)
        self.stochastic = spec.get('stochastic', False)
        self.record, self.records = record, []

    def decide(self, observation, actions):
        row = encode_record(self.game, self.encoder, self.base, observation, actions,
                            self.metadata.get('anchor_scale', 1.))
        with torch.inference_mode():
            logits, values, _ = self.model([row])
            logp = torch.log_softmax(logits[0], -1).numpy()
        if self.stochastic:
            probabilities = np.exp(logp.astype(np.float64)); probabilities /= probabilities.sum()
            index = int(self.rng.choice(len(actions), p=probabilities))
        else: index = int(logp.argmax())
        if self.record:
            row.update(action=index, old_logp=float(logp[index]), value=float(values[0]))
            self.records.append(row)
        return actions[index]
