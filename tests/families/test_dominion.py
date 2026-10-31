from copy import deepcopy
from collections import Counter
import json
import math
import random
import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.games.dominion import GAME_CLASS, CARDS, ZONES


def game():
    g = GAME_CLASS()
    g.reset(3)
    return g


def hand(g, seat, cards):
    p = g.state['players'][seat]
    stock = [c for zone in ZONES for c in p[zone]]
    p.update(hand=[], deck=[], discard=[], in_play=[])
    for c in cards:
        if c in stock:
            stock.remove(c)
        else:
            g.state['supply'][c] -= 1
        p['hand'].append(c)
    p['deck'] = stock


def test_setup_exact_supply_and_kingdom_scope():
    g = game()
    g.validate()
    assert len(CARDS) == 17
    assert {c: g.state['supply'][c] for c in ('estate','province','curse','copper')} == dict(estate=8, province=8, curse=10, copper=46)
    assert all(len(p['hand']) == 5 and len(p['deck']) == 5 for p in g.state['players'])
    assert g.scores() == [3, 3]


def test_draw_shuffles_only_when_needed_and_keeps_old_top():
    g = game()
    p = g.state['players'][0]
    p.update(deck=['gold'], hand=[], discard=['estate','copper'])
    before = g.rng.getstate()
    g._draw(0, 1)
    assert p['hand'] == ['gold'] and p['discard'] == ['estate','copper']
    assert g.rng.getstate() == before
    g._draw(0, 2)
    assert p['hand'][0] == 'gold' and sorted(p['hand'][1:]) == ['copper','estate']
    assert not p['discard'] and not p['deck']


def test_merchant_multiple_first_silver_only():
    g = game()
    hand(g, 0, ['merchant','merchant','silver','silver'])
    g.step({'type': 'play_action', 'card': 'merchant'})
    g.step({'type': 'play_action', 'card': 'merchant'})
    assert g.state['merchants'] == 2 and g.state['actions'] == 1
    g.step({'type': 'end_actions'})
    g.step({'type': 'play_treasure', 'card': 'silver'})
    g.step({'type': 'play_treasure', 'card': 'silver'})
    assert g.state['coins'] == 6
    g.validate()


def test_cellar_discards_before_any_draw_and_can_discard_zero():
    g = game()
    hand(g, 0, ['cellar','estate','estate','copper'])
    g.step({'type': 'play_action', 'card': 'cellar'})
    before = len(g.state['players'][0]['hand'])
    g.step({'type': 'discard', 'card': 'estate'})
    g.step({'type': 'discard', 'card': 'estate'})
    assert len(g.state['players'][0]['hand']) == before-2
    g.step({'type': 'finish_cellar'})
    assert len(g.state['players'][0]['hand']) == before
    assert g.state['phase'] == 'action' and g.state['actions'] == 1
    g.validate()


def test_militia_moat_reaction_retains_card_and_attacker_income():
    g = game()
    hand(g, 0, ['militia'])
    hand(g, 1, ['moat','copper','copper','estate','estate'])
    before = list(g.state['players'][1]['hand'])
    g.step({'type': 'play_action', 'card': 'militia'})
    assert g.current_player == 1 and g.state['coins'] == 2
    g.step({'type': 'reveal_moat'})
    assert g.state['players'][1]['hand'] == before
    assert g.current_player == 0 and g.state['phase'] == 'treasure'
    g.validate()


def test_militia_defender_discards_and_can_decline_moat():
    g = game()
    hand(g, 0, ['militia'])
    hand(g, 1, ['moat','copper','copper','estate','estate'])
    g.step({'type': 'play_action', 'card': 'militia'})
    g.step({'type': 'accept_attack'})
    assert g.current_player == 1
    for _ in range(2):
        g.step({'type': 'discard', 'card': 'estate'})
    assert len(g.state['players'][1]['hand']) == 3 and g.current_player == 0
    assert g.state['players'][1]['discard'] == ['estate','estate']
    g.validate()


def test_workshop_cost_not_augmented_by_current_coins():
    g = game()
    hand(g, 0, ['workshop'])
    g.state['coins'] = 100
    g.step({'type': 'play_action', 'card': 'workshop'})
    assert all(CARDS[a['card']]['cost'] <= 4 for a in g.legal_actions())
    assert {'type': 'gain', 'card': 'province'} not in g.legal_actions()
    g.step({'type': 'gain', 'card': 'militia'})
    assert g.state['players'][0]['discard'][-1] == 'militia'
    assert g.state['coins'] == 100 and g.state['buys'] == 1
    g.validate()


def test_mine_optional_and_gains_treasure_to_hand():
    g = game()
    hand(g, 0, ['mine','silver'])
    g.step({'type': 'play_action', 'card': 'mine'})
    assert {'type': 'decline_mine'} in g.legal_actions()
    g.step({'type': 'trash', 'card': 'silver'})
    assert all('treasure' in CARDS[a['card']]['types'] for a in g.legal_actions())
    g.step({'type': 'gain', 'card': 'gold'})
    assert g.state['players'][0]['hand'] == ['gold']
    assert g.state['trash'] == ['silver']
    g.validate()


def test_remodel_empty_hand_and_trash_gain_limit():
    g = game()
    hand(g, 0, ['remodel'])
    g.step({'type': 'play_action', 'card': 'remodel'})
    assert g.state['phase'] == 'treasure'
    g = game()
    hand(g, 0, ['remodel','estate'])
    g.step({'type': 'play_action', 'card': 'remodel'})
    assert g.legal_actions() == [{'type': 'trash', 'card': 'estate'}]
    g.step(g.legal_actions()[0])
    assert g.state['gain_limit'] == 4
    g.step({'type': 'gain', 'card': 'smithy'})
    assert g.state['players'][0]['discard'] == ['smithy']
    assert g.state['trash'] == ['estate']
    g.validate()


def test_treasures_cannot_be_played_after_buy_and_cleanup_top_choice():
    g = game()
    hand(g, 0, ['copper','silver','estate'])
    g.step({'type': 'end_actions'})
    g.step({'type': 'play_treasure', 'card': 'silver'})
    g.step({'type': 'start_buy'})
    assert not any(a['type'] == 'play_treasure' for a in g.legal_actions())
    g.step({'type': 'buy', 'card': 'estate'})
    g.step({'type': 'end_turn', 'discard_top': 'copper'})
    assert g.state['players'][0]['discard'][-1] == 'copper'
    assert g.current_player == 1
    g.validate()


def test_last_province_does_not_end_mid_turn():
    g = game()
    s = g.state
    s['players'][1]['discard'].extend(['province']*7)
    s['supply']['province'] = 1
    s.update(phase='buy', coins=11, buys=2)
    g.step({'type': 'buy', 'card': 'province'})
    assert not s['terminated'] and s['buys'] == 1
    g.step({'type': 'buy', 'card': 'silver'})
    g.step(next(a for a in g.legal_actions() if a['type'] == 'end_turn'))
    assert s['terminated'] and s['players'][0]['turns'] == 1
    assert s['players'][1]['turns'] == 0
    g.validate()


def test_three_piles_empty_and_fewer_turns_tiebreak():
    g = game()
    s = g.state
    for card in ('cellar','village','workshop'):
        s['players'][0]['discard'].extend([card]*s['supply'][card])
        s['supply'][card] = 0
    s['phase'] = 'buy'
    g.step(next(a for a in g.legal_actions() if a['type'] == 'end_turn'))
    assert s['terminated'] and s['winners'] == [1]
    assert g.scores() == [3, 3]
    g.validate()


def test_hidden_zones_and_public_hand_inference_boundary():
    g = game()
    obs = g.observe(0)
    assert 'hand' not in obs['players'][1] and 'draw_count' not in obs['players'][1]
    for p in obs['players']:
        assert not ({'deck','discard','final_cards'} & set(p))
    assert 'rng' not in obs and 'final_scores' not in obs
    clone = GAME_CLASS()
    clone.restore(g.snapshot())
    # Swap opponent hand/deck cards without changing their public sizes/top.
    p = clone.state['players'][1]
    p['hand'][0], p['deck'][0] = p['deck'][0], p['hand'][0]
    assert clone.observe(0) == obs
    obs['players'][0]['hand'].append('gold')
    assert g.observe(0) != obs


def test_invalid_action_atomicity_and_json_replay():
    g = game()
    snap = g.snapshot()
    with pytest.raises(InvalidAction):
        g.step({'type': 'buy','card':'province'})
    assert g.snapshot() == snap
    clone = GAME_CLASS()
    clone.restore(json.loads(json.dumps(snap)))
    rng = random.Random(5)
    for _ in range(120):
        if g.current_player is None:
            break
        action = rng.choice(g.legal_actions())
        assert clone.step(action) == g.step(action)
        assert clone.snapshot() == g.snapshot()


@pytest.mark.parametrize('policy', ['random', 'heuristic'])
@pytest.mark.parametrize('seed', range(3))
def test_complete_games(policy, seed):
    g = GAME_CLASS()
    g.reset(seed, first_player=seed % 2)
    rng = random.Random(seed)
    for _ in range(12000):
        g.validate()
        if g.current_player is None:
            break
        obs, actions = g.observe(), g.legal_actions()
        action = rng.choice(actions) if policy == 'random' else max(actions, key=lambda a: g.heuristic(obs, a))
        f = g.features(obs, action)
        assert len(f) == 24 and all(math.isfinite(v) for v in f)
        g.step(action)
    assert g.state['terminated'], (policy, seed, g.scores())
    assert g.state['end_reason'] == 'supply_end'
