"""Pure public-state goal-to-action compiler. No simulator, reward or hidden state.

The coding agent's judgment is encoded as inspectable rules, NOT an online LLM:
acquire needed resources/card, build shared habitat prefixes, protect unfinished
support, and place the requested animal. PPO may override every recommendation.
"""
import numpy as np
import torch
from torch import nn

from boardbench.environments.harmonies import COLORS, STACKS
from boardbench.v134.plans import GOAL_DIM, PLAN_DIM

METHODS = ("control", "advice", "gated", "sticky")


class ActionAdvice(nn.Module):
    def __init__(self):
        super().__init__()
        # current stack, desired stack, next color; -1 = destroy a target.
        table = np.full((15, 15, 6), -1., np.float64)
        for current, stack in enumerate(STACKS):
            for target, desired in enumerate(STACKS):
                if len(stack) < len(desired) and desired[:len(stack)] == stack:
                    table[current, target, COLORS.index(desired[len(stack)])] = 1.
        self.register_buffer("next_color", torch.from_numpy(table), persistent=False)

    def forward(self, base, plans):
        if base.shape[1] != 1346 or plans.shape[1] != PLAN_DIM:
            raise ValueError("public encoding width mismatch")
        n = len(base)
        board = base[:, :391].reshape(n, 23, 17)
        current = board[:, :, :15].argmax(-1)
        market = base[:, 427:526].reshape(n, 3, 33)[:, :, 1:]
        active = base[:, 626:762].reshape(n, 4, 34)[:, :, 1:33]
        bundles = base[:, 391:409].reshape(n, 3, 6) * 3
        result = base.new_zeros(n, 240)
        for slot, weight in enumerate((1., .4)):
            g = plans[:, slot * GOAL_DIM:(slot + 1) * GOAL_DIM]
            cells = g[:, :368].reshape(n, 23, 16)
            constrained = cells[:, :, :15].sum(-1)
            target = cells[:, :, :15].argmax(-1)
            card = g[:, 368:400]
            present = card.sum(-1)
            need = g[:, 400:406] * 6
            placement = self.next_color[current, target] * constrained[:, :, None]
            # Locking is respected by the game's mask; advice cannot unlock cells.
            result[:, 3:141] += weight * placement.transpose(1, 2).reshape(n, 138) * present[:, None]
            supplied = torch.minimum(bundles, need[:, None, :]).sum(-1)
            result[:, :3] += weight * supplied / need.sum(-1).clamp_min(1)[:, None]
            matches_market = (market * card[:, None, :]).sum(-1)
            matches_active = (active * card[:, None, :]).sum(-1)
            result[:, 141:144] += weight * matches_market
            anchor = cells[:, :, 15]
            unfinished = constrained * (current != target)
            animal = (1.5 * matches_active[:, :, None] * anchor[:, None, :]
                      - .5 * unfinished[:, None, :])
            result[:, 144:236] += weight * animal.reshape(n, 92) * present[:, None]
            # end_turn discard slots refer to the MARKET, not the active cards.
            result[:, 237:240] -= weight * matches_market
        return result.clamp(-1., 1.)
