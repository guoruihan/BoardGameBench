"""Public-only blueprints; optimistic resource feasibility is not a promise.

No score, rule, transition or action-mask changes. Candidate zero is always a
free-play option. Other candidates contain one primary and at most one compatible
secondary animal placement. Sequential terrain restructuring is NOT represented.
"""
from collections import Counter

import numpy as np

from boardbench.environments.harmonies import CARDS, COLORS, STACKS
from boardbench.v123.policies import available_tokens
from .geometry import top_goals

K = 8
GOAL_DIM = 23 * 16 + 32 + 6 + 3
PLAN_DIM = 2 * GOAL_DIM + 1
ARMS = ("capacity", "reactive", "persistent", "manager")


def refresh_goal(obs, goal):
    """Revalidate against the actual public board, not predicted future draws."""
    cards = {c["card_id"]: c["placed_count"] for c in obs["active_cards"]}
    if goal["card_id"] not in cards:
        if goal["card_id"] not in obs["animal_market"] or len(cards) >= 4:
            return None
        placed = 0
    else:
        placed = cards[goal["card_id"]]
    if placed != goal["placed"] or obs["board"][goal["anchor"]]["animal"] is not None:
        return None
    need = Counter()
    for cell, target in goal["targets"].items():
        current = STACKS[obs["board"][cell]["stack_id"]]
        desired = STACKS[target]
        if desired[:len(current)] != current:
            return None
        missing = desired[len(current):]
        if missing and obs["board"][cell]["animal"] is not None:
            return None
        need.update(missing)
    if any(n > available_tokens(obs)[c] for c, n in need.items()):
        return None
    marginal = CARDS[goal["card_id"]]["score_by_placed_count"][placed + 1] - CARDS[goal["card_id"]]["score_by_placed_count"][placed]
    return {**goal, "need": dict(need), "missing": sum(need.values()),
            "value": marginal / (1 + sum(need.values()))}


def achieved(obs, goal):
    return obs["board"][goal["anchor"]]["animal"] == goal["card_id"]


def compatible(obs, left, right):
    if left["card_id"] == right["card_id"] or left["anchor"] == right["anchor"]:
        return False
    targets = dict(left["targets"])
    if any(c in targets and targets[c] != t for c, t in right["targets"].items()):
        return False
    targets.update(right["targets"])
    need = Counter(c for cell, target in targets.items()
                   for c in STACKS[target][len(STACKS[obs["board"][cell]["stack_id"]]):])
    supply = available_tokens(obs)
    return all(n <= supply[c] for c, n in need.items())


def candidates(obs):
    cards = {c["card_id"]: c["placed_count"] for c in obs["active_cards"]}
    if len(cards) < 4:
        cards.update({c: 0 for c in obs["animal_market"] if c is not None})
    goals = []
    for card, placed in cards.items():
        goals.extend({**g, "placed": placed} for g in top_goals(obs, card, placed))
    goals.sort(key=lambda g: -g["value"])
    # Interleave global best and per-card best: one animal must not consume K.
    first = {}
    for g in goals:
        first.setdefault(g["card_id"], g)
    ordered = goals[:2] + list(first.values()) + goals[2:]
    plans, seen = [[]], set()
    for g in ordered:
        identity = (g["card_id"], g["anchor"], tuple(sorted(g["targets"].items())))
        if identity in seen:
            continue
        seen.add(identity)
        partner = next((other for other in goals if compatible(obs, g, other)), None)
        plans.append([g] + ([partner] if partner is not None else []))
        if len(plans) == K:
            break
    return plans


def encode_plan(obs, plan, age=0):
    vector = np.zeros(PLAN_DIM, dtype=np.float32)
    for slot, goal in enumerate(plan):
        g = refresh_goal(obs, goal)
        if g is None:
            continue
        v = vector[slot * GOAL_DIM:(slot + 1) * GOAL_DIM]
        for cell, target in g["targets"].items():
            v[cell * 16 + target] = 1.
        v[g["anchor"] * 16 + 15] = 1.
        pos = 23 * 16
        v[pos + g["card_id"] - 1] = 1.
        pos += 32
        v[pos:pos + 6] = [g["need"].get(c, 0) / 6. for c in COLORS]
        v[pos + 6:] = [g["missing"] / 12., g["value"] / 10., float(g["card_id"] in obs["animal_market"])]
    vector[-1] = min(age / 2., 1.) if plan else 0.
    return vector
