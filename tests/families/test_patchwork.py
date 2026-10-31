import json
import random

import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.games.patchwork import (
    Patchwork, PATCHES, ORIENTATIONS, PLACEMENTS, MASKS, FULL, LEATHER, INCOME, WINDOWS,
)


def engine(seed=0, first=0):
    g = Patchwork()
    g.reset(seed, first)
    return g


def force_patch(g, patch, player=0, position=0, buttons=20):
    g.state['current_player'] = player
    g.state['positions'][player] = position
    g.state['buttons'][player] = buttons
    g.state['neutral'] = g.state['ring'].index(patch)
    return next(a for a in g.legal_actions() if a.get('patch') == patch)


def test_authentic_components_and_corrected_track():
    assert len(PATCHES) == 33
    assert sum(len(v['cells']) for v in PATCHES.values()) == 166
    assert PATCHES['A'] == dict(id='A', cost=2, time=1, income=0, cells=[[0, 0], [1, 0]])
    assert PATCHES['H']['cost'] == 0
    assert PATCHES['I']['income'] == 2
    assert LEATHER == (26, 32, 38, 44, 50)
    assert INCOME == (5, 11, 17, 23, 29, 35, 41, 47, 53)
    assert len(ORIENTATIONS['A']) == 2
    assert len(ORIENTATIONS['E']) == 4
    for patch, placements in PLACEMENTS.items():
        assert len({mask for o, x, y, mask in placements}) == len(placements)
        assert all(mask & FULL == mask and mask.bit_count() == len(PATCHES[patch]['cells'])
                   for o, x, y, mask in placements)


def test_setup_neutral_after_smallest_and_complete_placements():
    g = engine()
    assert g.state['buttons'] == [5, 5]
    assert g.state['neutral'] == (g.state['ring'].index('A')+1) % 33
    g.state['neutral'] = g.state['ring'].index('A')
    actions = [a for a in g.legal_actions() if a.get('patch') == 'A']
    assert len(actions) == 144  # 9*8 horizontal and vertical; adjacency not required.
    assert g.legal_actions(1) == []
    assert any(a['x'] == 7 and a['y'] == 8 for a in actions)


def test_tied_time_keeps_mover_and_neutral_advances_from_removed_patch():
    g = engine()
    a = force_patch(g, 'E', position=10)
    g.state['positions'][1] = 12
    index = g.state['ring'].index('E')
    next_tile = g.state['ring'][(index+1) % 33]
    g.step(a)
    assert g.current_player == 0
    assert g.state['positions'] == [12, 12]
    assert g.state['ring'][g.state['neutral']] == next_tile
    assert 'E' not in g.state['ring']
    assert g.state['buttons'][0] == 18  # 20 - 3 + new income 1 at 11.
    g.validate()


def test_advance_actual_distance_and_cannot_borrow_future_income():
    g = engine()
    g.state['positions'] = [10, 12]
    g.step(dict(type='advance'))
    assert g.state['buttons'][0] == 8
    assert g.state['positions'][0] == 13
    g = engine()
    g.state['neutral'] = g.state['ring'].index('I')
    g.state['positions'] = [4, 10]
    g.state['buttons'][0] = 5
    g.state['income'][0] = 100
    assert not any(a.get('patch') == 'I' for a in g.legal_actions())


def test_income_order_leather_immediate_multiple_and_one_owner():
    g = engine()
    g.state['positions'] = [25, 38]
    g.step(dict(type='advance'))
    assert g.current_player == 0
    assert g.state['phase'] == 'leather'
    assert g.state['pending_events'][0] == 26
    assert 26 not in g.state['leather']
    for i, event in enumerate((26, 32, 38)):
        assert g.state['pending_events'][0] == event
        g.step(dict(type='leather', x=i, y=0))
    assert g.current_player == 1
    assert len(g.state['placements'][0]) == 3
    assert g.state['leather'] == [44, 50]
    g.validate()


def test_bonus_uses_all_nine_windows_and_is_not_spendable():
    for window in WINDOWS:
        g = engine()
        g.state['boards'][0] = window
        g._bonus(0)
        assert g.state['bonus_owner'] == 0
        assert g.state['buttons'][0] == 5
        g.state['boards'][1] = FULL
        g._bonus(1)
        assert g.state['bonus_owner'] == 0
    g.state['buttons'][0] = 14
    g.state['boards'][0] = FULL ^ 31  # Five empty cells.
    assert g.scores()[0] == 11


def test_natural_end_clamps_and_tie_goes_to_first_arrival():
    g = engine()
    g.state['positions'] = [52, 53]
    g.state['arrivals'] = [1]
    g.state['buttons'] = [5, 6]
    result = g.step(dict(type='advance'))
    assert g.state['positions'] == [53, 53]
    assert g.state['buttons'] == [6, 6]
    assert result['terminated']
    assert result['info']['winners'] == [1]
    assert result['rewards'] == [-1., 1.]
    assert g.legal_actions() == []
    with pytest.raises(InvalidAction):
        g.step(dict(type='advance'))


def test_full_board_leather_is_removed_without_impossible_choice():
    g = engine()
    g.state['boards'][0] = FULL
    g.state['positions'] = [25, 26]
    g.step(dict(type='advance'))
    assert g.state['phase'] == 'turn'
    assert g.state['discarded_leather'] == [26]
    assert 26 not in g.state['leather']


@pytest.mark.parametrize('heuristic', [False, True])
@pytest.mark.parametrize('seed', [0, 1, 7])
def test_full_games_invariants_replay_privacy_and_features(seed, heuristic):
    g = engine(seed, seed % 2)
    h = Patchwork()
    rng = random.Random(seed)
    decisions = 0
    while not g.state['terminated']:
        g.validate()
        actions = g.legal_actions()
        assert actions
        obs = g.observe()
        assert 'rng' not in obs and 'seed' not in obs
        for a in actions[::max(1, len(actions)//8)]:
            assert len(g.features(obs, a)) == 24
            assert isinstance(g.heuristic(obs, a), float)
        a = max(actions, key=lambda a: g.heuristic(obs, a)) if heuristic else rng.choice(actions)
        if decisions % 8 == 0:
            h.restore(json.loads(json.dumps(g.snapshot())))
            assert h.observe() == obs
            assert h.step(a) == g.step(a)
            assert h.snapshot() == g.snapshot()
        else:
            g.step(a)
        decisions += 1
        assert decisions < 120  # Test guard, not an engine ending.
    g.validate()
    assert g.state['positions'] == [53, 53]
    assert g.outcome()['reason'] == 'both_time_tokens_at_end'
    assert len(g.state['winners']) == 1


def test_invalid_action_and_injected_failure_are_atomic():
    g = engine()
    before = g.snapshot()
    for action in ({'type': 'leather', 'x': 0, 'y': 0}, {'type': 'advance', 'extra': 1},
                   {'type': 'buy', 'patch': 'A', 'orientation': True, 'x': 0, 'y': 0}):
        with pytest.raises(InvalidAction):
            g.step(action)
        assert g.snapshot() == before
    original = g._step
    def fails(action):
        g.state['buttons'][0] += 1
        g.rng.random()
        raise RuntimeError('injected')
    g._step = fails
    with pytest.raises(RuntimeError):
        g.step({'type': 'advance'})
    assert g.snapshot() == before
    g._step = original


def test_restore_rejects_component_loss_atomically():
    g = engine()
    before = g.snapshot()
    bad = g.snapshot()
    bad['state']['ring'].pop()
    with pytest.raises(ValueError):
        g.restore(bad)
    assert g.snapshot() == before
