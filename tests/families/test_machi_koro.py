import json
import math
import random
from copy import deepcopy
import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.games.machi_koro import GAME_CLASS, CARDS


def game():
    g = GAME_CLASS()
    g.reset(9)
    return g


def give(g, seat, card, count=1):
    g.state['players'][seat]['cards'][card] += count
    g.state['market'][card] -= count


def roll(g, dice):
    g.state['dice'] = dice
    g._resolve_income()


def test_setup_conservation():
    g = game()
    g.validate()
    assert sum(g.state['market'].values()) == 84
    assert g.scores() == [None, None]
    assert g.legal_actions() == [{'type': 'roll', 'dice': 1}]


def test_red_debt_not_recovered_from_later_income():
    g = game()
    give(g, 0, 'bakery')
    give(g, 1, 'cafe')
    g.state['players'][0]['coins'] = 0
    roll(g, [3])
    assert [p['coins'] for p in g.state['players']] == [2, 3]


def test_factory_counts_all_copies():
    g = game()
    give(g, 0, 'cheese_factory', 2)
    give(g, 0, 'ranch', 3)
    roll(g, [3, 4])
    assert g.state['players'][0]['coins'] == 21


def test_mall_per_copy_and_not_primary_industry():
    g = game()
    p = g.state['players'][0]
    p['landmarks']['shopping_mall'] = True
    give(g, 0, 'bakery')
    give(g, 0, 'ranch')
    roll(g, [2])
    assert p['coins'] == 8  # 3 + 2*2 bakery + 1 ranch


def test_reroll_resolves_only_final_result_and_once():
    g = game()
    p = g.state['players'][0]
    p['landmarks'].update(train_station=True, amusement_park=True, radio_tower=True)
    give(g, 0, 'stadium')
    g.state.update(phase='reroll', dice=[3, 3])
    # Python Random(1)'s first two results are 2, 5.
    g.rng.seed(1)
    g.step({'type': 'reroll'})
    assert g.state['dice'] == [2, 5]
    assert not g.state['extra_turn']
    assert g.state['phase'] == 'construction'
    assert p['coins'] == 3
    assert {'type': 'reroll'} not in g.legal_actions()


def test_purple_order_and_optional_nonpurple_swap():
    g = game()
    for c in ('stadium', 'tv_station', 'business_center'):
        give(g, 0, c)
    give(g, 1, 'mine')
    roll(g, [6])
    assert len(g.state['pending_purple']) == 3
    assert not any(a.get('give') in ('stadium', 'tv_station') for a in g.legal_actions())
    g.step({'type': 'business', 'give': 'wheat_field', 'take': 'mine'})
    assert g.state['players'][0]['cards']['mine'] == 1
    g.step({'type': 'purple', 'card': 'tv_station', 'target': 1})
    assert [p['coins'] for p in g.state['players']] == [6, 0]
    g.step({'type': 'purple', 'card': 'stadium', 'target': 1})
    assert g.state['phase'] == 'construction'
    g.validate()


def test_extra_turn_chain_and_provisional_no_retroactive_trigger():
    g = game()
    p = g.state['players'][0]
    p['landmarks']['train_station'] = True
    p['coins'] = 30
    roll(g, [2, 2])
    g.step({'type': 'landmark', 'landmark': 'amusement_park'})
    assert g.current_player == 1  # Documented event-based interpretation.
    g.state['current_player'] = 0
    for _ in range(2):
        roll(g, [1, 1])
        g.step({'type': 'pass'})
        assert g.current_player == 0


def test_fourth_landmark_immediate_win():
    g = game()
    p = g.state['players'][0]
    p['landmarks'].update(train_station=True, shopping_mall=True, amusement_park=True)
    p['coins'] = 22
    g.state['phase'] = 'construction'
    result = g.step({'type': 'landmark', 'landmark': 'radio_tower'})
    assert result['terminated'] and result['rewards'] == [1., -1.]
    assert g.outcome()['winners'] == [0]
    assert g.legal_actions() == []


def test_atomicity_json_replay_and_information_boundary():
    g = game()
    snap = g.snapshot()
    for action in ({'type': 'roll', 'dice': 2}, {'type': 'roll', 'dice': True}):
        with pytest.raises(InvalidAction):
            g.step(action)
        assert g.snapshot() == snap
    clone = GAME_CLASS()
    clone.restore(json.loads(json.dumps(snap)))
    for _ in range(80):
        if g.current_player is None:
            break
        action = g.legal_actions()[0]
        assert clone.step(action) == g.step(action)
        assert clone.snapshot() == g.snapshot()
    obs = g.observe(0)
    assert 'rng' not in obs and 'state' not in obs
    mutated = deepcopy(obs)
    mutated['players'][0]['coins'] = -100
    assert g.state['players'][0]['coins'] >= 0


@pytest.mark.parametrize('policy', ['random', 'heuristic'])
@pytest.mark.parametrize('seed', range(3))
def test_complete_games(policy, seed):
    g = GAME_CLASS()
    g.reset(seed, first_player=seed % 2)
    rng = random.Random(seed)
    for _ in range(5000):
        g.validate()
        if g.current_player is None:
            break
        actions = g.legal_actions()
        obs = g.observe()
        if policy == 'random':
            action = rng.choice(actions)
        else:
            action = max(actions, key=lambda a: g.heuristic(obs, a))
        f = g.features(obs, action)
        assert len(f) == 24 and all(math.isfinite(v) for v in f)
        g.step(action)
    assert g.state['terminated'], (policy, seed, g.score_breakdown())
    assert g.outcome()['reason'] == 'four_landmarks'
