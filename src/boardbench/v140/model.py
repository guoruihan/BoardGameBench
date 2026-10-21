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
        # Keep diagnostic interventions faithful to the original per-view order.
        if self.solo.intervention != 'normal' or self.solo.advice_intervention != 'normal':
            return self.forward_reference(x, mask)
        if x.shape[-1] != FEATURES or mask.shape[-1] != ACTIONS or not mask.any(-1).all():
            raise ValueError('duo input/mask layout mismatch')
        n = len(x)
        own, extra, opponent = x[:, :BASE], x[:, BASE:2*BASE], x[:, 2*BASE:3*BASE]
        plan = x[:, 3*BASE:3*BASE+PLAN_DIM]
        # One batched encoding, with autograd sharing the SAME features among all
        # heads. No detach/cache across updates, and no change to parameter keys.
        encoded = self.solo.body(torch.cat((own, extra, opponent), 0))
        z = encoded.split(n)
        both = encoded[:2*n]
        plans = torch.cat((plan, plan), 0)
        inputs = torch.cat((both, plans), -1)
        delta = self.solo.adapter(inputs)
        heads = self.solo.actor(both) + delta[:, :240]
        values = self.solo.critic(both).squeeze(-1) + delta[:, 240]
        if self.solo.method != 'control':
            advice = self.solo.advice(torch.cat((own, extra), 0), plans)
            # Reference solo heads receive an all-legal mask before duo masking.
            advice = advice - advice.mean(-1, keepdim=True)
            strength = self.solo.strength
            if self.solo.method in ('gated', 'sticky'):
                strength = 2 * self.solo.gate(inputs).sigmoid()
            heads = heads + strength * advice
        first, second = heads.split(n)
        value, v2 = values.split(n)
        residual = self.context(torch.cat((*z, plan, x[:, -8:]), -1))
        common = .5*(first+second)
        logits = torch.cat((first[:, :3], common[:, 3:141], first[:, 141:144],
                            common[:, 144:240], second[:, :2], second[:, 141:143]), -1)
        baseline_value = .5*(value+v2)
        if self.objective == 'margin':
            baseline_value = baseline_value-self.solo.critic(z[2]).squeeze(-1)
        return ((logits+residual[:, :ACTIONS]).masked_fill(~mask, -1e9),
                baseline_value+residual[:, ACTIONS])

    def forward_reference(self, x, mask):
        """Frozen V140 arithmetic for parity gates, not the training hot path."""
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
