from copy import deepcopy
import json
import math
import random
import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.games.clever import (GAME_CLASS, COLORS, empty_sheet,
    mark_options, apply_mark, breakdown, color_scores)


def game():
    g = GAME_CLASS()
    g.reset(7)
    return g


def test_setup_and_blue_sum_not_discard_threshold():
    g = game()
    s = g.state
    assert s['round'] == 1 and all(p['rerolls'] == 1 for p in s['sheets'])
    s['dice'] = dict(zip(COLORS, [1, 2, 4, 5, 3, 6]))
    action = next(a for a in g.legal_actions() if a.get('die') == 'blue')
    assert action['target'] == 8
    g.step(action)
    assert s['sheets'][0]['blue'] == [8]
    assert s['platter'] == ['yellow']
    assert s['kept'] == ['blue']
    assert s['selections'] == 1
    g.validate()


def test_same_yellow_number_is_a_real_choice():
    p = empty_sheet()
    options = mark_options(p, 'yellow', 3)
    assert [m['target'] for m in options] == [0, 13]
    apply_mark(p, options[0])
    assert 0 in p['yellow'] and 13 not in p['yellow']


def test_purple_strict_then_resets_after_six():
    p = empty_sheet()
    p['purple'] = [2, 5]
    assert not mark_options(p, 'purple', 5)
    apply_mark(p, mark_options(p, 'purple', 6)[0])
    assert mark_options(p, 'purple', 1)


def test_orange_bonus_multiplier_and_new_reward():
    p = empty_sheet()
    p['orange'] = [1, 2, 3]
    rewards = apply_mark(p, mark_options(p, 'orange', 4)[0])
    assert p['orange'][-1] == 8
    assert rewards == []
    rewards = apply_mark(p, mark_options(p, 'orange', 4)[0])
    assert rewards == ['yellow_cross']


def test_blue_short_first_column_and_one_time_reward():
    p = empty_sheet()
    apply_mark(p, mark_options(p, 'blue', 5)[0])
    rewards = apply_mark(p, mark_options(p, 'blue', 9)[0])
    assert rewards == ['reroll']
    assert apply_mark(p, mark_options(p, 'blue', 2)[0]) == []


def test_bonus_chain_resolved_before_next_roll():
    g = game()
    s = g.state
    p = s['sheets'][0]
    p['yellow'] += [0, 1]
    p['blue'] = [5]
    s['dice']['yellow'] = 5
    before = deepcopy(s['dice'])
    action = next(a for a in g.legal_actions() if a.get('die') == 'yellow' and a['target'] == 2)
    g.step(action)
    assert s['bonuses'] == ['blue_cross']
    assert s['dice'] == before  # Remaining dice wait for chain resolution.
    assert all(a['type'] == 'bonus' for a in g.legal_actions())
    g.step(next(a for a in g.legal_actions() if a['target'] == 9))
    assert p['rerolls'] == 2
    assert s['bonuses'] == []
    assert s['phase'] in ('active', 'active_extra')


def test_passive_fallback_only_when_all_platter_options_unusable():
    g = game()
    s = g.state
    s.update(phase='passive', current_player=1, available=[], platter=['purple'],
             kept=['yellow', 'blue', 'green', 'orange', 'white'])
    s['dice']['purple'] = 2
    assert all(a.get('die') == 'purple' for a in g.legal_actions())
    s['sheets'][1]['purple'] = [5]
    assert all(a.get('die') != 'purple' for a in g.legal_actions())
    action = g.legal_actions()[0]
    platter_before = list(s['platter'])
    g.step(action)
    assert s['platter'] == platter_before
    assert s['phase'] == 'passive_extra'


def test_extra_reuses_regular_die_only_once_per_window():
    g = game()
    s = g.state
    s.update(phase='active_extra', available=[], platter=['yellow', 'blue', 'green'],
             kept=['white', 'orange', 'purple'])
    s['sheets'][0]['extras'] = 2
    action = next(a for a in g.legal_actions() if a.get('die') == 'white' and a.get('color') == 'orange')
    g.step(action)
    assert s['sheets'][0]['extras'] == 1
    assert not any(a.get('die') == 'white' for a in g.legal_actions())
    g.step({'type': 'end_window'})
    assert s['current_player'] == 1 and s['extra_used'] == []


def test_reroll_only_available_and_consumes_token_not_opportunity():
    g = game()
    s = g.state
    s.update(available=['green', 'orange'], platter=['yellow', 'blue'], kept=['white', 'purple'], selections=2)
    before = dict(s['dice'])
    g.step({'type': 'reroll'})
    assert s['selections'] == 2 and s['sheets'][0]['rerolls'] == 0
    assert all(s['dice'][c] == before[c] for c in ['yellow', 'blue', 'white', 'purple'])
    assert {'type': 'reroll'} not in g.legal_actions()


def test_round_four_bonus_both_seats_then_active_player():
    g = game()
    s = g.state
    s['round'] = 4
    g._start_round()
    assert s['current_player'] == 0 and s['bonuses'] == ['round_four']
    g.step(next(a for a in g.legal_actions() if a['color'] == 'orange'))
    assert s['current_player'] == 1 and s['bonuses'] == ['round_four']
    g.step(next(a for a in g.legal_actions() if a['color'] == 'purple'))
    assert s['current_player'] == 0 and s['phase'] == 'active'
    assert s['sheets'][0]['orange'] == [6] and s['sheets'][1]['purple'] == [6]


def test_full_sheet_no_forced_illegal_mark_or_infinite_roll():
    g = game()
    p = g.state['sheets'][0]
    p.update(yellow=list(range(16)), blue=list(range(2, 13)), green=11,
             orange=[1,1,1,2,1,1,2,1,2,1,3], purple=[6]*11, rerolls=0)
    for _ in range(3):
        assert g.legal_actions() == [{'type': 'unusable_roll'}]
        g.step({'type': 'unusable_roll'})
    assert g.state['phase'] == 'active_extra'
    assert g.legal_actions() == [{'type': 'end_window'}]


def test_fox_zero_and_tiebreak():
    p = empty_sheet()
    p['yellow'] = sorted(set(p['yellow']+[0, 4, 8]))
    p['blue'] = [2,3,4,5,6,7]
    p['green'] = 6
    p['orange'] = [6,6,6]
    p['foxes'] = 3
    assert color_scores(p) == [10,16,21,18,0]
    assert breakdown(p)['total'] == 65
    g = game()
    s = g.state
    s.update(round=6, turn_in_round=1, phase='passive_extra')
    s['sheets'][0]['orange'] = [3, 3]
    s['sheets'][1]['orange'] = [3]
    s['sheets'][1]['purple'] = [3]
    g.step({'type': 'end_window'})
    assert g.state['terminated'] and g.outcome()['winners'] == [0]


def test_atomic_rejection_json_snapshot_replay_public_only():
    g = game()
    snap = g.snapshot()
    with pytest.raises(InvalidAction):
        g.step({'type': 'end_window'})
    assert g.snapshot() == snap
    clone = GAME_CLASS()
    clone.restore(json.loads(json.dumps(snap)))
    rng = random.Random(19)
    for _ in range(100):
        if g.current_player is None:
            break
        action = rng.choice(g.legal_actions())
        assert clone.step(action) == g.step(action)
        assert clone.snapshot() == g.snapshot()
    assert 'rng' not in g.observe() and 'state' not in g.observe()
    obs = g.observe()
    obs['dice']['yellow'] = -100
    assert g.state['dice']['yellow'] > 0


@pytest.mark.parametrize('policy', ['random', 'heuristic'])
@pytest.mark.parametrize('seed', range(6))
def test_complete_games(policy, seed):
    g = GAME_CLASS()
    g.reset(seed, first_player=seed % 2)
    rng = random.Random(seed)
    for _ in range(1000):
        g.validate()
        if g.current_player is None:
            break
        obs, actions = g.observe(), g.legal_actions()
        action = rng.choice(actions) if policy == 'random' else max(actions, key=lambda a: g.heuristic(obs, a))
        f = g.features(obs, action)
        assert len(f) == 24 and all(math.isfinite(v) for v in f)
        g.step(action)
    assert g.state['terminated'] and g.state['round'] == 6
    assert g.outcome()['reason'] == 'six_rounds'
