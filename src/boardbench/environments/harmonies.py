"""Harmonies solo A, base32, finite supply; no implicit strategic actions."""
from collections import Counter, deque
from copy import deepcopy
import json
from pathlib import Path
import random

from boardbench.contracts import InvalidAction, derive_seed
from .board import BoardGame, integer

COLORS = ("gray", "blue", "brown", "green", "yellow", "red")
INVENTORY = dict(zip(COLORS, (23, 23, 21, 19, 19, 15)))
CELLS = [(c, j-c//2) for c in range(5) for j in range(5 if c % 2 == 0 else 4)]
CELL_INDEX = {xy: i for i, xy in enumerate(CELLS)}
DELTAS = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))
NEIGHBORS = [[CELL_INDEX[(q+dq, r+dr)] for dq, dr in DELTAS if (q+dq, r+dr) in CELL_INDEX]
             for q, r in CELLS]
STACKS = ((), ("blue",), ("yellow",), ("gray",), ("gray",)*2, ("gray",)*3,
          ("brown",), ("brown",)*2, ("green",), ("brown", "green"),
          ("brown", "brown", "green"), ("red",), ("red",)*2,
          ("brown", "red"), ("gray", "red"))
STACK_INDEX = {s: i for i, s in enumerate(STACKS)}
CARDS = {c["card_id"]: c for c in json.loads(
    Path(__file__).with_name("data").joinpath("animal_cards.base32.v1.json").read_text())["cards"]}
if set(CARDS) != set(range(1, 33)):
    raise ValueError("expected exactly 32 base animal cards")


def rotate(q, r, orientation):
    for _ in range(orientation):
        q, r = -r, q+r
    return q, r


def compile_patterns():
    compiled = {}
    for card_id, card in CARDS.items():
        if len(card["score_by_placed_count"]) != card["cube_count"]+1:
            raise ValueError("invalid animal score track")
        anchors = []
        for q, r in CELLS:
            orientations = {}
            for orientation in range(6):
                pattern = []
                for requirement in card["pattern_cells"]:
                    dq, dr = rotate(*requirement["cell"], orientation)
                    index = CELL_INDEX.get((q+dq, r+dr))
                    if index is None:
                        break
                    allowed = ((12, 13, 14) if requirement["terrain"] == "building"
                               else (STACK_INDEX[tuple(requirement["stack_bottom_to_top"])],))
                    pattern.append((index, allowed))
                else:
                    orientations[orientation] = tuple(pattern)
            anchors.append(orientations)
        compiled[card_id] = anchors
    return compiled


PATTERNS = compile_patterns()


class Harmonies(BoardGame):
    task_id, rules_version = "harmonies", "harmonies_solo_a_v1"

    def __init__(self):
        self.reset(0)

    @classmethod
    def task_spec(cls):
        return {"id": cls.task_id, "version": cls.rules_version, "cells": CELLS,
                "neighbors": NEIGHBORS, "stacks": STACKS, "colors": COLORS,
                "inventory": INVENTORY, "cards": list(CARDS.values()), "max_actions": 256,
                "supply_ruling": "terminate when a full nine-token refill is impossible"}

    def reset(self, seed):
        self._bag = [color for color, count in INVENTORY.items() for _ in range(count)]
        self._deck = list(CARDS)
        random.Random(derive_seed(seed, "tokens")).shuffle(self._bag)
        random.Random(derive_seed(seed, "cards")).shuffle(self._deck)
        self.s = {"board": [{"stack_id": 0, "animal": None} for _ in CELLS],
                  "token_market": [None]*3, "animal_market": [None]*3,
                  "active_cards": [], "completed_cards": [], "turn_index": 1,
                  "decision_index": 0, "bundle_taken": False, "card_taken": False,
                  "pending_tokens": dict.fromkeys(COLORS, 0),
                  "discarded_tokens": dict.fromkeys(COLORS, 0), "discarded_cards": [],
                  "remaining_tokens": dict(INVENTORY), "remaining_cards": list(CARDS),
                  "terminated": False, "score": None}
        self._refill_tokens()
        self._refill_cards()
        return self.observe(), self.reset_info()

    def _refill_tokens(self):
        for slot in range(3):
            self.s["token_market"][slot], self._bag = self._bag[:3], self._bag[3:]
        self.s["remaining_tokens"] = {c: self._bag.count(c) for c in COLORS}

    def _refill_cards(self):
        for slot in range(3):
            if self.s["animal_market"][slot] is None and self._deck:
                self.s["animal_market"][slot] = self._deck.pop(0)
        self.s["remaining_cards"] = sorted(self._deck)

    def can_place_token(self, color, cell):
        b = self.s["board"][cell]
        return b["animal"] is None and STACKS[b["stack_id"]]+(color,) in STACK_INDEX

    def matches(self, card_id, anchor, orientation):
        if self.s["board"][anchor]["animal"] is not None:
            return False
        pattern = PATTERNS[card_id][anchor].get(orientation)
        return pattern is not None and all(self.s["board"][i]["stack_id"] in allowed for i, allowed in pattern)

    def legal_actions(self):
        if self.s["terminated"]:
            return []
        actions = []
        if not self.s["bundle_taken"]:
            actions += [{"type": "choose_bundle", "bundle_id": i} for i in range(3)]
        for color, count in self.s["pending_tokens"].items():
            if count:
                actions += [{"type": "place_token", "color": color, "cell_id": i}
                            for i in range(23) if self.can_place_token(color, i)]
        if not self.s["card_taken"] and len(self.s["active_cards"]) < 4:
            actions += [{"type": "take_card", "market_slot": i}
                        for i, card in enumerate(self.s["animal_market"]) if card is not None]
        for card in self.s["active_cards"]:
            for anchor in range(23):
                for orientation in range(6):
                    if self.matches(card["card_id"], anchor, orientation):
                        actions.append({"type": "place_animal", "instance_id": card["instance_id"],
                                        "anchor_cell_id": anchor, "orientation": orientation})
                        break  # Same card/anchor successor; retain one orientation witness.
        if self.s["bundle_taken"] and not any(self.s["pending_tokens"].values()):
            actions.append({"type": "end_turn"})
            if not self.s["card_taken"]:
                actions += [{"type": "end_turn", "discard_card_slot": i}
                            for i, card in enumerate(self.s["animal_market"]) if card is not None]
        return actions

    def _validate(self, action):
        if not isinstance(action, dict) or "type" not in action:
            raise InvalidAction("expected a typed Harmonies action")
        kind = action["type"]
        fields = {"choose_bundle": {"type", "bundle_id"}, "place_token": {"type", "color", "cell_id"},
                  "take_card": {"type", "market_slot"},
                  "place_animal": {"type", "instance_id", "anchor_cell_id", "orientation"},
                  "end_turn": {"type", "discard_card_slot"}}
        if not isinstance(kind, str) or kind not in fields:
            raise InvalidAction("unknown Harmonies action type")
        if kind == "end_turn":
            if set(action) - fields[kind]:
                raise InvalidAction("unexpected end_turn fields")
        elif set(action) != fields[kind]:
            raise InvalidAction("missing or unexpected action fields")
        ranges = {"bundle_id": (0, 3), "cell_id": (0, 23), "market_slot": (0, 3),
                  "instance_id": (1, 33), "anchor_cell_id": (0, 23), "orientation": (0, 6),
                  "discard_card_slot": (0, 3)}
        for key, bounds in ranges.items():
            if key in action and not (key == "discard_card_slot" and action[key] is None):
                if not integer(action[key], *bounds):
                    raise InvalidAction(f"invalid {key}")
        s = self.s
        if kind == "choose_bundle":
            valid = not s["bundle_taken"] and s["token_market"][action["bundle_id"]] is not None
        elif kind == "place_token":
            color = action["color"]
            valid = (isinstance(color, str) and color in COLORS and s["pending_tokens"][color] > 0
                     and self.can_place_token(color, action["cell_id"]))
        elif kind == "take_card":
            valid = not s["card_taken"] and len(s["active_cards"]) < 4 and s["animal_market"][action["market_slot"]] is not None
        elif kind == "place_animal":
            card = next((c for c in s["active_cards"] if c["instance_id"] == action["instance_id"]), None)
            valid = card is not None and self.matches(card["card_id"], action["anchor_cell_id"], action["orientation"])
        else:
            slot = action.get("discard_card_slot")
            valid = (s["bundle_taken"] and not any(s["pending_tokens"].values())
                     and (slot is None or (not s["card_taken"] and s["animal_market"][slot] is not None)))
        if not valid:
            raise InvalidAction(f"preconditions not met for {kind}")

    def step(self, action):
        self._require_live()
        self._validate(action)  # All checks precede every mutation and draw.
        s, kind, reason = self.s, action["type"], None
        if kind == "choose_bundle":
            slot = action["bundle_id"]
            s["pending_tokens"].update(Counter(s["token_market"][slot]))
            s["token_market"][slot] = None
            s["bundle_taken"] = True
        elif kind == "place_token":
            color, cell = action["color"], action["cell_id"]
            s["board"][cell]["stack_id"] = STACK_INDEX[STACKS[s["board"][cell]["stack_id"]]+(color,)]
            s["pending_tokens"][color] -= 1
        elif kind == "take_card":
            slot = action["market_slot"]
            card_id = s["animal_market"][slot]
            s["active_cards"].append({"instance_id": card_id, "card_id": card_id, "placed_count": 0})
            s["animal_market"][slot] = None
            s["card_taken"] = True
        elif kind == "place_animal":
            card = next(c for c in s["active_cards"] if c["instance_id"] == action["instance_id"])
            card["placed_count"] += 1
            s["board"][action["anchor_cell_id"]]["animal"] = card["instance_id"]
            if card["placed_count"] == CARDS[card["card_id"]]["cube_count"]:
                s["active_cards"].remove(card)
                s["completed_cards"].append(card)
        else:
            if sum(b["stack_id"] == 0 for b in s["board"]) <= 2:
                reason = "board_full_enough"
            else:
                for bundle in s["token_market"]:
                    for color in bundle or []:
                        s["discarded_tokens"][color] += 1
                s["token_market"] = [None]*3
                if len(self._bag) < 9:
                    reason = "supply_exhausted"
                else:
                    self._refill_tokens()
                    slot = action.get("discard_card_slot")
                    if slot is not None:
                        s["discarded_cards"].append(s["animal_market"][slot])
                        s["animal_market"][slot] = None
                    self._refill_cards()
                    s["turn_index"] += 1
                    s["bundle_taken"] = s["card_taken"] = False
            s["terminated"] = reason is not None
        return self._result(reason)

    def score_breakdown(self):
        board = self.s["board"]
        ids = [b["stack_id"] for b in board]
        trees = sum({8: 1, 9: 3, 10: 7}.get(s, 0) for s in ids)
        mountains = sum({3: 1, 4: 3, 5: 7}[s] for i, s in enumerate(ids)
                        if s in (3, 4, 5) and any(ids[j] in (3, 4, 5) for j in NEIGHBORS[i]))
        fields, seen = 0, set()
        for i, sid in enumerate(ids):
            if sid != 2 or i in seen:
                continue
            pending, component = [i], set()
            while pending:
                j = pending.pop()
                if j in component:
                    continue
                component.add(j)
                pending.extend(k for k in NEIGHBORS[j] if ids[k] == 2 and k not in component)
            seen |= component
            fields += 5 if len(component) >= 2 else 0
        buildings = sum(5 for i, sid in enumerate(ids) if sid in (12, 13, 14)
                        and len({STACKS[ids[j]][-1] for j in NEIGHBORS[i] if ids[j]}) >= 3)
        river_length = 0
        for start, sid in enumerate(ids):
            if sid != 1:
                continue
            distances, queue = {start: 0}, deque([start])
            while queue:
                i = queue.popleft()
                for j in NEIGHBORS[i]:
                    if ids[j] == 1 and j not in distances:
                        distances[j] = distances[i]+1
                        queue.append(j)
            river_length = max(river_length, max(distances.values())+1)
        river = (0, 0, 2, 5, 8, 11, 15)[river_length] if river_length <= 6 else 15+4*(river_length-6)
        animals = sum(CARDS[c["card_id"]]["score_by_placed_count"][c["placed_count"]]
                      for c in self.s["active_cards"]+self.s["completed_cards"])
        return {"trees": trees, "mountains": mountains, "fields": fields,
                "buildings": buildings, "river": river, "river_length": river_length,
                "animals": animals, "total": trees+mountains+fields+buildings+river+animals}

    def _private_state(self):
        return {"bag": list(self._bag), "deck": list(self._deck)}

    def _load_private(self, state):
        self._bag, self._deck = list(state["bag"]), list(state["deck"])
        if Counter(self._bag) != Counter(self.s["remaining_tokens"]) or sorted(self._deck) != self.s["remaining_cards"]:
            raise ValueError("snapshot remaining supply mismatch")

    def _resample(self, seed):
        self._bag = [c for c in COLORS for _ in range(self.s["remaining_tokens"][c])]
        self._deck = list(self.s["remaining_cards"])
        random.Random(derive_seed(seed, "tokens")).shuffle(self._bag)
        random.Random(derive_seed(seed, "cards")).shuffle(self._deck)
