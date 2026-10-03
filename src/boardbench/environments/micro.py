"""Micro Tiles v1: 3x3, two iid offers, orthogonal edges and full lines."""
import random

from boardbench.contracts import InvalidAction
from .board import BoardGame, integer, tuples

LINES = [[3*r+c for c in range(3)] for r in range(3)] + [[3*r+c for r in range(3)] for c in range(3)]
EDGES = [(i, j) for i in range(9) for j in range(i+1, 9)
         if abs(i//3-j//3)+abs(i%3-j%3) == 1]


class MicroTiles(BoardGame):
    task_id, rules_version = "micro_tiles", "micro_tiles_v1"

    def __init__(self):
        self.reset(0)

    @classmethod
    def task_spec(cls):
        return {"id": cls.task_id, "version": cls.rules_version, "cells": list(range(9)),
                "lines": LINES, "edges": EDGES, "max_actions": 9,
                "action_format": {"offer_index": "0..1", "cell_id": "0..8"}}

    def reset(self, seed):
        self._rng = random.Random(seed)
        self.s = {"board": [-1]*9, "offers": self._draw(), "placed_count": 0,
                  "turn_index": 0, "decision_index": 0, "terminated": False, "score": None}
        return self.observe(), self.reset_info()

    def _draw(self):
        return [self._rng.randrange(3), self._rng.randrange(3)]

    def legal_actions(self):
        return [] if self.s["terminated"] else [
            {"offer_index": offer, "cell_id": cell} for offer in range(2)
            for cell, color in enumerate(self.s["board"]) if color == -1]

    def step(self, action):
        self._require_live()
        if (not isinstance(action, dict) or set(action) != {"offer_index", "cell_id"}
                or not integer(action["offer_index"], 0, 2) or not integer(action["cell_id"], 0, 9)
                or self.s["board"][action["cell_id"]] != -1):
            raise InvalidAction("choose offer_index 0/1 and an empty cell_id 0..8")
        self.s["board"][int(action["cell_id"])] = self.s["offers"][int(action["offer_index"])]
        self.s["placed_count"] += 1
        self.s["turn_index"] += 1
        self.s["terminated"] = self.s["placed_count"] == 9
        self.s["offers"] = [] if self.s["terminated"] else self._draw()
        return self._result("board_full")

    def score_breakdown(self):
        b = self.s["board"]
        edges = sum(b[i] >= 0 and b[i] == b[j] for i, j in EDGES)
        lines = sum(b[line[0]] >= 0 and len({b[i] for i in line}) == 1 for line in LINES)*3
        return {"adjacency": edges, "lines": lines, "total": edges+lines}

    def _private_state(self):
        return {"rng": self._rng.getstate()}

    def _load_private(self, state):
        self._rng = random.Random()
        self._rng.setstate(tuples(state["rng"]))

    def _resample(self, seed):
        self._rng = random.Random(seed)
