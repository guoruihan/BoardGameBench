"""Standard solo Take It Easy!: one public tile, no rotations, 15 lines."""
import random

from boardbench.contracts import InvalidAction
from .board import BoardGame, integer

CELLS = [(q, r) for q in range(-2, 3) for r in range(-2, 3)
         if max(abs(q), abs(r), abs(q+r)) <= 2]
TILES = [(v, r, f) for v in (1, 5, 9) for r in (2, 6, 7) for f in (3, 4, 8)]
LINES = [{"direction": direction, "axis": axis, "offset": c,
          "cells": [i for i, (q, r) in enumerate(CELLS)
                    if (q, q+r, r)[axis] == c]}
         for axis, direction in enumerate(("vertical", "rising", "falling")) for c in range(-2, 3)]


class TakeItEasy(BoardGame):
    task_id, rules_version = "take_it_easy", "take_it_easy_standard_v1"

    def __init__(self):
        self.reset(0)

    @classmethod
    def task_spec(cls):
        return {"id": cls.task_id, "version": cls.rules_version, "cells": CELLS,
                "tiles": TILES, "lines": LINES, "max_actions": 19,
                "action_format": {"cell_id": "0..18"}}

    def reset(self, seed):
        self._deck = list(range(27))
        random.Random(seed).shuffle(self._deck)
        current = self._deck.pop(0)
        self.s = {"board": [-1]*19, "current_tile": current,
                  "remaining_tiles": [i in self._deck for i in range(27)],
                  "placed_count": 0, "turn_index": 0, "decision_index": 0,
                  "terminated": False, "score": None}
        return self.observe(), self.reset_info()

    def legal_actions(self):
        return [] if self.s["terminated"] else [
            {"cell_id": i} for i, tile in enumerate(self.s["board"]) if tile == -1]

    def step(self, action):
        self._require_live()
        if (not isinstance(action, dict) or set(action) != {"cell_id"}
                or not integer(action["cell_id"], 0, 19) or self.s["board"][action["cell_id"]] != -1):
            raise InvalidAction("choose an empty cell_id 0..18 (tiles cannot rotate)")
        self.s["board"][int(action["cell_id"])] = self.s["current_tile"]
        self.s["placed_count"] += 1
        self.s["turn_index"] += 1
        self.s["terminated"] = self.s["placed_count"] == 19
        self.s["current_tile"] = None if self.s["terminated"] else self._deck.pop(0)
        if not self.s["terminated"]:
            self.s["remaining_tiles"][self.s["current_tile"]] = False
        return self._result("board_full")

    def score_breakdown(self):
        lines, by_direction = [], {d: 0 for d in ("vertical", "rising", "falling")}
        for line in LINES:
            values = [TILES[self.s["board"][i]][line["axis"]] for i in line["cells"]
                      if self.s["board"][i] != -1]
            status = "broken" if len(set(values)) > 1 else "complete" if len(values) == len(line["cells"]) else "possible"
            score = values[0]*len(values) if status == "complete" else 0
            lines.append({**line, "cells": list(line["cells"]), "status": status, "score": score})
            by_direction[line["direction"]] += score
        return {**by_direction, "lines": lines, "total": sum(by_direction.values())}

    def _private_state(self):
        return {"deck": list(self._deck)}

    def _load_private(self, state):
        self._deck = list(state["deck"])
        if sorted(self._deck) != [i for i, left in enumerate(self.s["remaining_tiles"]) if left]:
            raise ValueError("snapshot remaining tiles mismatch")

    def _resample(self, seed):
        self._deck = [i for i, left in enumerate(self.s["remaining_tiles"]) if left]
        random.Random(seed).shuffle(self._deck)
