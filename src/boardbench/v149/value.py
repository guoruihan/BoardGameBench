"""Fast value-only path through the existing calibrated model, without actor heads."""
import math

import numpy as np
import torch
from torch.nn import functional as F

from boardbench.artifacts.store import digest, read_json
from boardbench.environments.harmonies_duo import player_view
from boardbench.v140.encoding import BASE, encode_batch
from boardbench.v134.plans import PLAN_DIM
from boardbench.v143.policy import own_projection
from boardbench.v145.policy import load_policy


def last_output(module, x):
    """The final output of a Linear/Tanh/Linear block; identical trained weights."""
    return F.linear(module[1](module[0](x)), module[2].weight[-1:], module[2].bias[-1:]).squeeze(-1)


def remaining_value(model, x):
    core = model.core
    if core.objective != 'score':
        raise ValueError('absolute own-score value required')
    projected = own_projection(x)
    n = len(x)
    z = core.solo.body(torch.cat([projected[:, i*BASE:(i+1)*BASE] for i in range(3)], 0))
    plan = x[:, 3*BASE:3*BASE+PLAN_DIM]
    inp = torch.cat((z[:2*n], torch.cat((plan, plan), 0)), -1)
    v = core.solo.critic(z[:2*n]).squeeze(-1) + last_output(core.solo.adapter, inp)
    context = torch.cat((*z.split(n), plan, projected[:, -8:]), -1)
    return .5*(v[:n]+v[n:]) + last_output(core.context, context) + last_output(model.opponent, x)


class Value:
    def __init__(self, manifest, sha256, native, opponent_hash):
        if digest(manifest) != sha256:
            raise ValueError('value manifest hash mismatch')
        m = read_json(manifest)
        if (m['format'] != 'v148_remaining_score_value' or m['target'] != 'remaining_own_score_div150'
                or opponent_hash not in m['opponent_hashes'] or digest(m['model']) != m['model_sha256']):
            raise ValueError('value target/opponent/weights mismatch')
        # Predictive eligibility is reported, never mistaken for a gameplay gain.
        self.metadata, self.native = m, native
        self.model = load_policy(m['model']).eval()

    def batch(self, observations, memories, seat):
        values = [None] * len(observations)
        live = []
        for i, obs in enumerate(observations):
            if obs['terminated']:
                values[i] = float(obs['scores'][seat])
            elif obs['active_player'] != seat:
                raise ValueError('evaluate at focal decisions; simulate opponent first')
            else:
                live.append(i)
        if live:
            obs = [observations[i] for i in live]
            plans = [memories[i][seat].vector(player_view(observations[i])) for i in live]
            x = encode_batch(obs, plans, self.native)
            with torch.inference_mode():
                remaining = remaining_value(self.model, torch.as_tensor(x, dtype=torch.float64)).numpy()
            for j, i in enumerate(live):
                values[i] = observations[i]['score_breakdown'][seat]['total'] + 150*float(remaining[j])
        if not all(math.isfinite(v) for v in values):
            raise ValueError('nonfinite final-score estimate')
        return values
