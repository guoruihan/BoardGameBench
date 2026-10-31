import json
import random

import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.games.toy_battle import ToyBattle, DATA, NODES, ADJ, HQ, REGIONS, targets


def game(seed=0, first=0):
    g = ToyBattle(); g.reset(seed, first)
    return g


def fixture():
    g = game()
    s = g.state
    s['hands'] = [[], []]
    s['removed'] = [[0, 1, 3, 4], [0, 1, 3, 4]]
    s['reserves'] = [[u for u in range(24) if u not in s['removed'][p]] for p in (0, 1)]
    return g


def move_piece(g, p, troop, node=None):
    uid = next(u for u in g.state['reserves'][p] if u//3 == troop)
    g.state['reserves'][p].remove(uid)
    if node is None:
        g.state['hands'][p].append(uid)
    else:
        g.state['board'][node].append(dict(owner=p, uid=uid, troop=troop))
    return uid


def test_official_map_and_setup():
    assert len(NODES) == 17
    assert len(DATA['edges']) == 26
    assert len(REGIONS) == 8
    assert sum(r['medals'] for r in REGIONS) == 14
    assert DATA['medal_targets'] == [7, 7]
    assert 'C1' not in ADJ['C0']  # Blue river is not a troop path.
    assert 'C2' not in ADJ['C1']
    for r in REGIONS:
        b = r['boundary']
        assert all(b[(i+1)%len(b)] in ADJ[n] for i, n in enumerate(b))
    for first in (0, 1):
        g = game(first=first)
        assert len(g.state['hands'][first]) == 3
        assert len(g.state['hands'][1-first]) == 4
        assert not g.observe()['special_base_effects']
        g.validate()


def test_connection_strength_wild_own_cover_and_hook_hq():
    g = fixture()
    b = g.state['board']
    assert targets(b, 0, 7) == ['L0', 'L3']
    assert set(targets(b, 0, 4)) == set(NODES)-set(HQ)
    move_piece(g, 0, 1, 'L0')
    move_piece(g, 1, 7, 'L1')
    assert 'L1' not in targets(b, 0, 7)
    assert 'L1' in targets(b, 0, 0)
    assert 'L0' in targets(b, 0, 0)
    assert 'C0' not in targets(b, 0, 7)
    b['L1'][-1]['troop'] = 0  # Isolate wildcard legality from bookkeeping.
    assert 'L1' in targets(b, 0, 1)
    assert HQ[1] not in targets(b, 0, 4)


def test_capn_chain_and_all_optional_effects():
    g = fixture()
    for _ in range(3): move_piece(g, 0, 2)
    move_piece(g, 0, 7)
    for node in ('L0', 'L1', 'C0'):
        g.step(dict(type='place', troop=2, node=node))
        assert g.current_player == 0
        assert g.state['phase'] == 'extra_place'
        assert {'type': 'skip'} in g.legal_actions()
    g.step(dict(type='place', troop=7, node='SR0'))
    assert g.current_player == 1
    g.validate()
    g = fixture(); move_piece(g, 0, 1)
    g.step(dict(type='place', troop=1, node='L0'))
    before = len(g.state['reserves'][0])
    g.step(dict(type='skip'))
    assert len(g.state['reserves'][0]) == before
    assert g.current_player == 1


def test_jumbo_removes_top_only_any_strength_and_never_retriggers():
    g = fixture()
    move_piece(g, 0, 1, 'L0')
    move_piece(g, 0, 2, 'L1')
    enemy = move_piece(g, 1, 7, 'L1')
    move_piece(g, 0, 3)
    g.step(dict(type='place', troop=3, node='L0'))
    assert g.state['phase'] == 'remove'
    assert {'type': 'remove', 'node': 'L1'} in g.legal_actions()
    g.step(dict(type='remove', node='L1'))
    assert g.state['board']['L1'][-1]['owner'] == 0
    assert g.state['discard'][1] == [enemy]
    assert g.current_player == 1  # Uncovered Cap'n does not retrigger.
    g.validate()


def test_rack_cap_and_partial_draw():
    g = fixture()
    for troop in (2, 2, 3, 3, 4, 4, 5, 1): move_piece(g, 0, troop)
    assert dict(type='draw') not in g.legal_actions()
    g.step(dict(type='place', troop=1, node='L0'))
    assert len(g.state['hands'][0]) == 7
    g.step(dict(type='use_effect'))
    assert len(g.state['hands'][0]) == 8
    g.validate()
    g = fixture()
    g.state['discard'][0] = g.state['reserves'][0][1:]
    g.state['reserves'][0] = g.state['reserves'][0][:1]
    g.step(dict(type='draw'))
    assert len(g.state['hands'][0]) == 1
    assert g.current_player == 1
    g.validate()


def test_xb_random_discard_hidden_information_and_replay():
    g = fixture()
    move_piece(g, 0, 5)
    for t in (2, 3, 4, 7): move_piece(g, 1, t)
    g.step(dict(type='place', troop=5, node='L0'))
    h = ToyBattle(); h.restore(json.loads(json.dumps(g.snapshot())))
    assert h.step(dict(type='use_effect')) == g.step(dict(type='use_effect'))
    assert len(g.state['hands'][1]) == 3
    assert len(g.state['discard'][1]) == 1
    g.validate()
    o = g.observe(0)
    assert 'reserves' not in o and 'removed' not in o and 'hands' not in o and 'rng' not in o
    before = g.observe(0)
    g.state['hands'][1], g.state['reserves'][1] = (
        g.state['reserves'][1][:3], g.state['hands'][1]+g.state['reserves'][1][3:])
    assert g.observe(0) == before  # Counterfactual invisible rack assignment.


def test_medals_permanent_and_immediate_even_disconnected():
    g = fixture()
    for n, t in zip(('L1', 'SL0'), (4, 4)): move_piece(g, 0, t, n)
    move_piece(g, 0, 4)
    g.step(dict(type='place', troop=4, node='C0'))
    assert g.state['medals'] == [1, 0]
    assert g.state['claimed'][0] == 0
    g.validate()
    top = g.state['board']['L1'].pop()
    g.state['discard'][0].append(top['uid'])
    g._claim_regions()
    assert g.state['medals'] == [1, 0]
    g.validate()


def test_headquarters_immediate_before_troop_draw_effect():
    g = fixture()
    for n, t in zip(('L0', 'L1', 'C0', 'R1', 'R0'), (2, 2, 3, 3, 4)):
        move_piece(g, 0, t, n)
    move_piece(g, 0, 6)
    reserve = len(g.state['reserves'][0])
    result = g.step(dict(type='place', troop=6, node='H1'))
    assert result['terminated'] and result['info']['winners'] == [0]
    assert result['info']['reason'] == 'enemy_headquarters_captured'
    assert len(g.state['reserves'][0]) == reserve
    assert result['info']['scores'] == [0, 0]  # HQ victory does not invent medals.
    g.validate()


def test_medal_goal_and_no_action_tiebreak():
    g = fixture()
    # Historical medals retained, independently from present ownership.
    g.state['claimed'][2] = 0
    g.state['claimed'][5] = 0
    g.state['medals'][0] = 6
    for n, t in zip(('L1', 'SL0'), (4, 4)): move_piece(g, 0, t, n)
    move_piece(g, 0, 4)
    result = g.step(dict(type='place', troop=4, node='C0'))
    assert result['terminated'] and g.state['medals'][0] == 7
    assert result['info']['reason'] == 'medal_objective_reached'
    g.validate()
    g = fixture()
    g.state['discard'][1] = g.state['reserves'][1]
    g.state['reserves'][1] = []
    result = g.step(dict(type='draw'))
    assert result['terminated'] and result['info']['winners'] == [0]
    assert result['info']['reason'] == 'active_player_has_no_draw_or_placement'
    g.validate()


@pytest.mark.parametrize('seed', range(10))
def test_complete_games_atomicity_and_json_replay(seed):
    g = game(seed, seed%2)
    h = ToyBattle()
    rng = random.Random(seed)
    decisions = 0
    while g.current_player is not None:
        g.validate()
        before = g.snapshot()
        with pytest.raises(InvalidAction): g.step(dict(type='place', troop=99, node='H0'))
        assert g.snapshot() == before
        acts, obs = g.legal_actions(), g.observe()
        assert acts
        for a in acts:
            assert len(g.features(obs, a)) == 24
            assert isinstance(g.heuristic(obs, a), float)
        action = max(acts, key=lambda a:g.heuristic(obs,a)) if g.current_player == 0 else rng.choice(acts)
        if decisions % 5 == 0:
            h.restore(json.loads(json.dumps(g.snapshot())))
            assert h.step(action) == g.step(action)
        else: g.step(action)
        decisions += 1
        assert decisions < 160
    g.validate()
    assert len(g.state['winners']) == 1
    assert g.legal_actions() == []


def test_restore_rejects_missing_troop():
    g = game(); before = g.snapshot(); bad = g.snapshot()
    bad['state']['reserves'][0].pop()
    with pytest.raises(ValueError): g.restore(bad)
    assert g.snapshot() == before
