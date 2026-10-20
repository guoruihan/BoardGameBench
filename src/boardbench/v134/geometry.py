"""Vectorized exact top-k of the existing public single-placement goal geometry."""
from itertools import product

import numpy as np

from boardbench.environments.harmonies import CARDS, COLORS, PATTERNS, STACKS
from boardbench.v123.policies import available_tokens

_COST = np.full((15, 16), -1, np.int16)
_NEED = np.zeros((15, 16, 6), np.int16)
_COST[:, 15] = 0  # Not a constrained cell.
for old, current in enumerate(STACKS):
    for new, target in enumerate(STACKS):
        if target[:len(current)] == current:
            missing = target[len(current):]
            _COST[old, new] = len(missing)
            _NEED[old, new] = [missing.count(c) for c in COLORS]

_TABLES = {}
for card_id in CARDS:
    targets, anchors, groups, meta = [], [], [], []
    group = 0
    for anchor, orientations in enumerate(PATTERNS[card_id]):
        for orientation, pattern in orientations.items():
            for assignment in sorted(product(*(allowed for _, allowed in pattern))):
                row = [15] * 23
                for (cell, _), target in zip(pattern, assignment):
                    row[cell] = target
                targets.append(row)
                anchors.append(anchor)
                groups.append(group)
                meta.append((anchor, orientation, {cell: target for (cell, _), target in zip(pattern, assignment)}))
            group += 1
    _TABLES[card_id] = (np.asarray(targets, np.int16), np.asarray(anchors),
                        np.asarray(groups), meta)


def top_goals(obs, card_id, placed_count=0, limit=12):
    target, anchors, groups, meta = _TABLES[card_id]
    board = np.asarray([c["stack_id"] for c in obs["board"]], np.int16)
    locked = np.asarray([c["animal"] is not None for c in obs["board"]])
    costs = _COST[board[None, :], target]
    needs = _NEED[board[None, :], target].sum(1)
    available = available_tokens(obs)
    supply = np.asarray([available[c] for c in COLORS])
    valid = ((costs >= 0).all(1) & ~((costs > 0) & locked).any(1) &
             ~locked[anchors] & (needs <= supply).all(1))
    ids = np.flatnonzero(valid)
    counts = costs.sum(1)
    # Group chooses cheapest assignment; ties retain lexicographic assignment.
    # Across groups, same count = same score; retain original pattern order.
    ids = ids[np.argsort(counts[ids], kind="stable")]
    goals, seen = [], set()
    track = CARDS[card_id]["score_by_placed_count"]
    marginal = track[placed_count + 1] - track[placed_count]
    for index in ids:
        group = groups[index]
        if group in seen:
            continue
        seen.add(group)
        anchor, orientation, cells = meta[index]
        goals.append({"anchor": anchor, "orientation": orientation, "targets": dict(cells),
                      "card_id": card_id, "need": {c: int(n) for c, n in zip(COLORS, needs[index]) if n},
                      "missing": int(counts[index]), "value": marginal / (1 + int(counts[index]))})
        if len(goals) == limit:
            break
    return goals
