from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import random

import pytest

from boardbench.contracts import InvalidAction
from boardbench.environments.micro import MicroTiles, EDGES, LINES as MICRO_LINES
from boardbench.environments.take_it_easy import TakeItEasy, CELLS as TIE_CELLS, TILES, LINES
from boardbench.environments.harmonies import (Harmonies, CELLS, COLORS, INVENTORY, NEIGHBORS,
                                              STACKS, STACK_INDEX, CARDS, PATTERNS, CELL_INDEX)

HANDOFF = Path(__file__).resolve().parents[1] / "docs/v1_handoff/BoardBench_V1_Agent_Handoff"
ENGINES = (MicroTiles, TakeItEasy, Harmonies)


@pytest.mark.parametrize("cls", ENGINES)
def test_public_observations_and_reset_metadata_do_not_alias_rules(cls):
    env = cls()
    obs, info = env.reset(18)
    original = env.observe()
    info["cells"].clear()
    obs["board"].clear()
    if cls is TakeItEasy:
        obs["score_breakdown"]["lines"][0]["cells"].clear()
    assert env.observe() == original
    assert cls.task_spec()["cells"]


def assert_inventory(env):
    count = Counter(env.s["remaining_tokens"])
    count.update(env.s["pending_tokens"])
    count.update(env.s["discarded_tokens"])
    for group in env.s["token_market"]:
        count.update(group or [])
    for cell in env.s["board"]:
        count.update(STACKS[cell["stack_id"]])
    assert count == INVENTORY


@pytest.mark.parametrize("cls", ENGINES)
def test_random_games_atomic_rejections_and_exact_restore(cls):
    for seed in range(5):
        env, restored = cls(), cls()
        env.reset(seed)
        rng, steps, rewards = random.Random(seed+100), 0, []
        while not env.s["terminated"]:
            before = env.save_state()
            for bad in (None, {}, {"cell_id": True}, {"type": "end_turn", "discard_card_slot": 999}):
                with pytest.raises(InvalidAction):
                    env.step(bad)
                assert env.save_state() == before
            # JSON round-trip preserves exact future, even midway through a turn.
            restored.load_state(json.loads(json.dumps(before)))
            for sim_seed in (17, 18):
                fork = env.fork(sim_seed)
                assert fork.observe() == env.observe()
                fork.step(rng.choice(fork.legal_actions()))
                assert env.save_state() == before
            actions = env.legal_actions()
            assert actions
            action = rng.choice(actions)
            result = env.step(action)
            assert restored.step(action) == result
            assert env.save_state() == restored.save_state()
            rewards.append(result.reward)
            if cls is Harmonies:
                assert_inventory(env)
            steps += 1
            assert steps < 256
        assert not env.legal_actions()
        assert all(r == 0 for r in rewards[:-1])
        assert rewards[-1] == env.score_breakdown()["total"] == result.info["score"]
        with pytest.raises(InvalidAction):
            env.step(action)


def test_micro_score_examples_and_terminal_no_draw():
    env = MicroTiles()
    assert len(EDGES) == 12 and len(MICRO_LINES) == 6
    examples = json.loads((HANDOFF / "data/scoring_examples.json").read_text())
    for example in examples["micro_tiles"]:
        env.s["board"] = example["board"]
        assert env.score_breakdown() == example["expected"]
    env.reset(5)
    env.s["offers"] = [1, 1]
    assert len(env.legal_actions()) == 18
    for _ in range(8):
        env.step(env.legal_actions()[0])
    before = env._rng.getstate()
    env.step(env.legal_actions()[0])
    assert env._rng.getstate() == before and env.s["offers"] == []


def test_tie_topology_scoring_draws_and_future_boundary():
    assert len(set(TILES)) == 27 and len(TIE_CELLS) == 19 and len(LINES) == 15
    assert all(sum(i in line["cells"] for line in LINES) == 3 for i in range(19))
    for direction in range(3):
        assert [len(l["cells"]) for l in LINES if l["axis"] == direction] == [3, 4, 5, 4, 3]
    env = TakeItEasy()
    example = json.loads((HANDOFF / "data/scoring_examples.json").read_text())["take_it_easy"][0]
    env.s["board"] = [c["tile_id"] for c in example["board"]]
    score = env.score_breakdown()
    assert {k: score[k] for k in example["expected"]} == example["expected"]
    env.s["board"][0] = 18
    assert env.score_breakdown()["vertical"] == 84
    env.reset(13)
    other = TakeItEasy()
    other.load_state(env.save_state())
    other._deck.reverse()  # Same public information, different true future.
    assert env.observe() == other.observe()
    assert env.fork(55).save_state() == other.fork(55).save_state()
    seen = []
    for _ in range(19):
        seen.append(env.s["current_tile"])
        env.step(env.legal_actions()[0])
    assert len(set(seen)) == 19 and len(env._deck) == 8 and env.s["current_tile"] is None


def test_harmonies_topology_all_stack_transitions():
    assert len(set(CELLS)) == 23
    assert [sum(q == c for q, r in CELLS) for c in range(5)] == [5, 4, 5, 4, 5]
    assert max(map(len, NEIGHBORS)) == 6
    assert all(i in NEIGHBORS[j] for i in range(23) for j in NEIGHBORS[i])
    env = Harmonies()
    for sid, stack in enumerate(STACKS):
        for color in COLORS:
            for occupied in (False, True):
                env.s["board"][0] = {"stack_id": sid, "animal": 1 if occupied else None}
                env.s["pending_tokens"] = dict.fromkeys(COLORS, 0)
                env.s["pending_tokens"][color] = 1
                before = env.save_state()
                action = {"type": "place_token", "color": color, "cell_id": 0}
                if not occupied and stack+(color,) in STACK_INDEX:
                    env.step(action)
                    assert STACKS[env.s["board"][0]["stack_id"]] == stack+(color,)
                else:
                    with pytest.raises(InvalidAction):
                        env.step(action)
                    assert before == env.save_state()


def test_harmonies_interleaving_delayed_cards_and_discard():
    for position in range(5):
        env = Harmonies()
        actions = [dict(type="choose_bundle", bundle_id=0)]
        for index in range(5):
            if index == position:
                card = env.s["animal_market"][1]
                deck = list(env._deck)
                env.step(dict(type="take_card", market_slot=1))
                assert env._deck == deck and env.s["animal_market"][1] is None
                assert env.s["active_cards"][0]["card_id"] == card
                with pytest.raises(InvalidAction):
                    env.step(dict(type="take_card", market_slot=0))
            if index == 0:
                env.step(actions[0])
            elif index < 4:
                env.step(next(a for a in env.legal_actions() if a["type"] == "place_token"))
        with pytest.raises(InvalidAction):
            env.step(dict(type="end_turn", discard_card_slot=0))
        env.step(dict(type="end_turn"))
        assert all(c is not None for c in env.s["animal_market"])
        assert env.s["turn_index"] == 2 and not env.s["card_taken"]
    env.reset(2)
    before = env.save_state()
    with pytest.raises(InvalidAction):
        env.step(dict(type="end_turn"))
    assert env.save_state() == before
    env.step(dict(type="choose_bundle", bundle_id=0))
    for _ in range(3):
        env.step(next(a for a in env.legal_actions() if a["type"] == "place_token"))
    old, next_card = env.s["animal_market"][2], env._deck[0]
    env.step(dict(type="end_turn", discard_card_slot=2))
    assert env.s["discarded_cards"] == [old] and env.s["animal_market"][2] == next_card


@pytest.mark.parametrize("card_id", range(1, 33))
def test_all_animal_cards_rotations_heights_boundaries_and_occupancy(card_id):
    for orientation in range(6):
        env = Harmonies()
        witnesses = [(a, p[orientation]) for a, p in enumerate(PATTERNS[card_id]) if orientation in p]
        assert witnesses, (card_id, orientation)
        anchor, pattern = witnesses[len(witnesses)//2]
        for cell, allowed in pattern:
            env.s["board"][cell]["stack_id"] = allowed[0]
        assert env.matches(card_id, anchor, orientation)
        for cell, allowed in pattern:
            previous = env.s["board"][cell]["stack_id"]
            env.s["board"][cell]["stack_id"] = 0
            assert not env.matches(card_id, anchor, orientation)
            env.s["board"][cell]["stack_id"] = previous
            if previous in (3, 4, 5, 8, 9, 10, 12, 13, 14):
                wrong_height = {3: 4, 4: 5, 5: 3, 8: 9, 9: 10, 10: 8, 12: 11, 13: 11, 14: 11}[previous]
                env.s["board"][cell]["stack_id"] = wrong_height
                assert not env.matches(card_id, anchor, orientation)
                env.s["board"][cell]["stack_id"] = previous
            if cell != anchor:
                env.s["board"][cell]["animal"] = 32
        assert env.matches(card_id, anchor, orientation)  # Non-anchor cubes allowed.
        env.s["active_cards"] = [{"instance_id": card_id, "card_id": card_id, "placed_count": 0}]
        env.step(dict(type="place_animal", instance_id=card_id, anchor_cell_id=anchor, orientation=orientation))
        assert not env.matches(card_id, anchor, orientation)
        assert env.s["active_cards"][0]["placed_count"] == 1
        assert env.score_breakdown()["animals"] == CARDS[card_id]["score_by_placed_count"][1]
    assert any(len(p) < 6 for p in PATTERNS[card_id]), "boundary must reject at least one placement"


def test_animal_completion_frees_slot_and_progress_persists():
    env, card_id = Harmonies(), 1
    anchor, orientation, pattern = next((a, o, pat) for a, ps in enumerate(PATTERNS[card_id]) for o, pat in ps.items())
    for cell, allowed in pattern:
        env.s["board"][cell]["stack_id"] = allowed[0]
    env.s["active_cards"] = [{"instance_id": c, "card_id": c, "placed_count": CARDS[c]["cube_count"]-1} for c in range(1, 5)]
    assert not any(a["type"] == "take_card" for a in env.legal_actions())
    env.step(dict(type="place_animal", instance_id=card_id, anchor_cell_id=anchor, orientation=orientation))
    assert len(env.s["active_cards"]) == 3 and len(env.s["completed_cards"]) == 1
    assert any(a["type"] == "take_card" for a in env.legal_actions())
    for cell, _ in pattern:
        if cell != anchor:
            env.s["board"][cell]["stack_id"] = 0
    assert env.s["completed_cards"][0]["placed_count"] == CARDS[card_id]["cube_count"]


def test_harmonies_supply_exhaustion_witness():
    witness = json.loads((HANDOFF / "reference/bag_exhaustion_witness.json").read_text())
    env = Harmonies()
    env._bag = list(witness["draw_order"])
    env._refill_tokens()
    for turn in witness["turns"]:
        assert env.s["token_market"] == turn["supply_bundles"]
        env.step(dict(type="choose_bundle", bundle_id=turn["chosen_bundle"]))
        for color in turn["stack_bottom_to_top"]:
            env.step(dict(type="place_token", color=color, cell_id=turn["place_in_new_cell"]))
            assert_inventory(env)
        deck = list(env._deck)
        result = env.step(dict(type="end_turn"))
        assert_inventory(env)
    assert result.terminated and result.info["reason"] == "supply_exhausted"
    assert sum(c["stack_id"] == 0 for c in env.s["board"]) == 10
    assert len(env._bag) == 3 and sum(env.s["discarded_tokens"].values()) == 78
    assert env.s["turn_index"] == 13 and env._deck == deck
    assert env.s["token_market"] == [None]*3


def test_harmonies_score_independent_small_boards():
    env = Harmonies()
    def board(values):
        env.s["board"] = [{"stack_id": values.get(i, 0), "animal": None} for i in range(23)]
        return env.score_breakdown()
    assert board({0: 8, 1: 9, 2: 10})["trees"] == 11
    assert board({0: 5})["mountains"] == 0
    assert board({0: 3, 1: 4, 2: 5})["mountains"] == 11
    assert board({0: 5, 1: 14})["mountains"] == 0
    assert board({0: 2, 1: 2, 3: 2, 4: 2})["fields"] == 10
    assert board({0: 2, 1: 2, 2: 2, 3: 2, 4: 2})["fields"] == 5
    center = next(i for i, ns in enumerate(NEIGHBORS) if len(ns) == 6)
    ns = NEIGHBORS[center]
    assert board({center: 13, ns[0]: 11, ns[1]: 6, ns[2]: 1})["buildings"] == 5
    assert board({center: 13, ns[0]: 14, ns[1]: 12, ns[2]: 1})["buildings"] == 0
    assert board({})["river"] == 0
    assert board({0: 1})["river"] == 0
    assert board({0: 1, 1: 1})["river"] == 2
    assert board({0: 1, 1: 1, 2: 1, 3: 1, 4: 1})["river"] == 11
    ring = board(dict.fromkeys(ns, 1))
    assert ring["river_length"] == 4 and ring["river"] == 8
    star = board({**dict.fromkeys(ns, 1), center: 1})
    assert star["river_length"] == 3 and star["river"] == 5
    path = [CELL_INDEX[p] for p in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 2), (4, 2))]
    for length, points in enumerate((0, 0, 2, 5, 8, 11, 15, 19, 23)):
        result = board(dict.fromkeys(path[:length], 1))
        assert result["river_length"] == length and result["river"] == points
    assert board({0: 1, 1: 1, 2: 1, 18: 1, 19: 1})["river"] == 5


def test_board_full_checked_only_at_end_without_revealing_next_market():
    env = Harmonies()
    # Independent near-terminal fixture: three pending blues, five empty cells.
    env.s["board"] = [{"stack_id": 1 if i < 18 else 0, "animal": None} for i in range(23)]
    env.s["bundle_taken"] = True
    env.s["pending_tokens"] = dict.fromkeys(COLORS, 0)
    env.s["pending_tokens"]["blue"] = 3
    for i in (18, 19, 20):
        result = env.step(dict(type="place_token", color="blue", cell_id=i))
        assert not result.terminated and result.reward == 0
    env.step(dict(type="take_card", market_slot=0))  # Optional action still legal.
    private = env._private_state()
    result = env.step(dict(type="end_turn"))
    assert result.terminated and result.info["reason"] == "board_full_enough"
    assert env._private_state() == private
    assert result.reward == env.score_breakdown()["total"]


def test_reflection_is_not_an_extra_orientation(monkeypatch):
    # Base32 patterns in this data are reflection-equivalent to rotations. Use
    # a synthetic three-color chiral template to exercise the geometry boundary;
    # this fixture is NOT added to the shipped deck.
    from boardbench.environments.harmonies import compile_patterns
    fixture = {"cube_count": 1, "score_by_placed_count": [0, 1], "pattern_cells": [
        {"cell": [0, 0], "terrain": "water", "stack_bottom_to_top": ["blue"]},
        {"cell": [1, 0], "terrain": "mountain", "stack_bottom_to_top": ["gray"]},
        {"cell": [0, 1], "terrain": "field", "stack_bottom_to_top": ["yellow"]}]}
    monkeypatch.setitem(CARDS, 1, fixture)
    monkeypatch.setitem(PATTERNS, 1, compile_patterns()[1])
    anchor = CELL_INDEX[(2, 1)]
    env = Harmonies()
    for (dq, dr), sid in (((0, 0), 1), ((1, 0), 3), ((0, 1), 2)):
        env.s["board"][CELL_INDEX[(2+dq, 1+dr)]]["stack_id"] = sid
    assert env.matches(1, anchor, 0)
    env.s["board"] = [{"stack_id": 0, "animal": None} for _ in CELLS]
    for (dq, dr), sid in (((0, 0), 1), ((1, 0), 3), ((0, 1), 2)):
        env.s["board"][CELL_INDEX[(2-dq-dr, 1+dr)]]["stack_id"] = sid
    assert not any(env.matches(1, anchor, orientation) for orientation in range(6))


def test_harmonies_public_fork_no_real_future():
    env, other = Harmonies(), Harmonies()
    other.load_state(env.save_state())
    other._bag.reverse()
    other._deck.reverse()
    assert env.observe() == other.observe()
    assert env.fork(123).save_state() == other.fork(123).save_state()
