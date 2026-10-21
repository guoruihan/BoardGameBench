"""Transferred solo skill plus learnable public-opponent/extra-market residual."""
from copy import deepcopy
import torch
from torch import nn

from boardbench.v134.plans import PLAN_DIM
from .encoding import ACTIONS, BASE, FEATURES


class DuoPolicy(nn.Module):
    def __init__(self, solo, objective='margin'):
        super().__init__()
        if objective not in ('margin', 'score'):
            raise ValueError('unknown objective')
        self.solo, self.objective = deepcopy(solo), objective
        hidden = solo.actor.in_features
        self.context = nn.Sequential(nn.Linear(3*hidden + PLAN_DIM + 8, 128), nn.Tanh(),
                                     nn.Linear(128, ACTIONS+1)).double()
        nn.init.zeros_(self.context[-1].weight); nn.init.zeros_(self.context[-1].bias)

    def forward(self, x, mask):
        if x.shape[-1] != FEATURES or mask.shape[-1] != ACTIONS or not mask.any(-1).all():
            raise ValueError('duo input/mask layout mismatch')
        own, extra, opponent = x[:, :BASE], x[:, BASE:2*BASE], x[:, 2*BASE:3*BASE]
        plan = x[:, 3*BASE:3*BASE+PLAN_DIM]
        # Explicit transfer convention: two overlapping market views. All actions
        # and both boards are represented; this is NOT exact solo-policy parity.
        all_legal = torch.ones((len(x), 240), dtype=torch.bool, device=x.device)
        first, value = self.solo(torch.cat((own, plan), -1), all_legal)
        second, v2 = self.solo(torch.cat((extra, plan), -1), all_legal)
        z = [self.solo.body(v) for v in (own, extra, opponent)]
        residual = self.context(torch.cat((*z, plan, x[:, -8:]), -1))
        common = .5*(first+second)
        # Placement/animal/end heads aggregate both views; market heads map slots.
        logits = torch.cat((first[:, :3], common[:, 3:141], first[:, 141:144],
                            common[:, 144:240], second[:, :2], second[:, 141:143]), -1)
        baseline_value = .5*(value+v2)
        if self.objective == 'margin':
            baseline_value = baseline_value-self.solo.critic(z[2]).squeeze(-1)
        return ((logits+residual[:, :ACTIONS]).masked_fill(~mask, -1e9),
                baseline_value+residual[:, ACTIONS])
