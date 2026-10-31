import json
import math
import random
from collections import Counter

import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.games.century_a_new_world import (
    GAME_CLASS, DATA, CARDS, BONUSES, EXPLORATIONS, LOCATIONS,
    bonus_value, location_cost, upgrade_results,
)


def game(seed=0, first=0):
    g = GAME_CLASS()
    g.reset(seed, first)
    return g


def give_cards(g, p, ids):
    for card in ids:
        if card in g.state['deck']:
            g.state['deck'].remove(card)
        elif card in g.state['market']:
            i = g.state['market'].index(card)
            g.state['market'][i] = g.state['deck'].pop(0)
        else:
            raise AssertionError(card)
        g.state['players'][p]['cards'].append(card)


def market_card(g, slot, card):
    if card in g.state['market']:
        i = g.state['market'].index(card)
        g.state['market'][i], g.state['market'][slot] = g.state['market'][slot], card
    else:
        g.state['deck'].remove(card)
        g.state['deck'].append(g.state['market'][slot])
        g.state['market'][slot] = card


def occupy(g, p, location, n):
    g.state['occupation'][location] = dict(owner=p, count=n)
    g.state['players'][p]['available'] -= n


def buy(g, location):
    return next(a for a in g.legal_actions() if a.get('location') == location
                and a.get('buy') and a.get('bonus_stack') is None)


def test_photographed_catalog_and_exact_two_player_setup():
    assert len(CARDS) == 22 and sum(c['vp'] for c in CARDS.values()) == 102
    assert Counter(c['ability'] for c in CARDS.values()) == dict(tool=6, help=6, settlers=6, explore=4)
    assert len(BONUSES) == 17 and len(EXPLORATIONS) == 10 and len(LOCATIONS) == 23
    assert sum(l['covered'] for l in LOCATIONS.values()) == 10
    assert DATA['bonus_stack_sizes'] == [4, 3, 2, 2, 2, 2]
    assert CARDS['card_804']['cost'] == [1, 0, 0, 2] and CARDS['card_804']['vp'] == 6
    assert CARDS['card_811']['cost'] == [1, 1, 1, 0] and CARDS['card_811']['vp'] == 2
    assert LOCATIONS['D1_r1c2']['gain'] == [6, 0, 0, 0]
    g = game(first=1)
    assert g.state['players'][1]['resources'] == [3, 0, 0, 0]
    assert g.state['players'][0]['resources'] == [4, 0, 0, 0]
    assert [p['available'] for p in g.state['players']] == [7, 7]
    assert len(g.state['deck']) == 18 and len(g.state['removed_bonuses']) == 2
    g.validate()


def test_bump_cost_replaces_printed_cost_and_returns_workers():
    g = game()
    occupy(g, 1, 'B1_r2c2', 1)  # printed 3, actually occupied by 1
    assert location_cost(LOCATIONS['B1_r2c2'], g.state['occupation']['B1_r2c2'], g.state['players'][0]) == 2
    g.step(dict(type='work', location='B1_r2c2'))
    assert g.state['players'][0]['available'] == 5
    assert g.state['players'][1]['available'] == 7
    assert g.state['occupation']['B1_r2c2'] == dict(owner=0, count=2)
    g.validate()


def test_stacking_native_help_bump_and_minimum_one():
    g = game()
    give_cards(g, 0, ['card_719', 'card_725'])
    occupy(g, 1, 'B1_r2c2', 3)
    g.step(dict(type='work', location='B1_r2c2'))
    assert g.state['occupation']['B1_r2c2']['count'] == 2
    assert g.state['players'][1]['available'] == 7
    assert location_cost(LOCATIONS['A1_r2c2'], None, g.state['players'][0]) == 1
    g.validate()


def test_cannot_reuse_own_location_or_overplace():
    g = game()
    occupy(g, 0, 'A1_r2c1', 1)
    assert not any(a.get('location') == 'A1_r2c1' for a in g.legal_actions())
    snapshot = g.snapshot()
    with pytest.raises(InvalidAction):
        g.step(dict(type='work', location='A1_r2c2', workers=2))
    assert g.snapshot() == snapshot


def test_rest_returns_only_own_workers_even_with_workers_available():
    g = game()
    occupy(g, 0, 'A1_r2c1', 2)
    occupy(g, 0, 'B1_r2c1', 1)
    occupy(g, 1, 'A1_r2c2', 1)
    g.step(dict(type='rest'))
    assert g.state['players'][0]['available'] == 7
    assert g.state['players'][0]['reserve'] == 5
    assert g.state['occupation']['A1_r2c2'] == dict(owner=1, count=1)
    g.validate()


def test_tool_stack_and_delayed_capacity_discard():
    g = game()
    give_cards(g, 0, ['card_712', 'card_714'])  # both corn -> +yellow
    g.state['players'][0]['resources'] = [9, 0, 0, 0]
    g.step(dict(type='work', location='A1_r2c1'))
    assert g.state['players'][0]['resources'] == [13, 0, 0, 0]
    assert g.current_player == 0 and g.state['phase'] == 'discard'
    assert g.state['players'][0]['turns'] == 0
    for _ in range(3):
        g.step(dict(type='discard', color=0))
    assert g.current_player == 1 and sum(g.state['players'][0]['resources']) == 10
    g.validate()


def test_upgrade_all_final_outcomes_including_optional_skips():
    g = game()
    g.state['players'][0]['resources'] = [1, 0, 1, 0]
    actual = {tuple(a['resources']) for a in g.legal_actions() if a.get('location') == 'B1_r3c1'}
    assert actual == {(1,0,1,0),(0,1,1,0),(1,0,0,1),(0,0,2,0),(0,1,0,1)}
    assert len(upgrade_results([0, 0, 0, 2], 3)) == 1


def test_full_trade_repetitions_not_arbitrary_cap():
    g = game()
    g.state['players'][0]['resources'] = [0, 0, 10, 0]
    trades = [a for a in g.legal_actions() if a.get('location') == 'D1_r1c1']
    assert [a['repeats'] for a in trades] == list(range(1, 11))
    g.step(trades[-1])
    assert g.state['players'][0]['resources'] == [0,20,0,0]
    assert g.state['phase'] == 'discard'
    g.validate()


def test_fort_card_bonus_or_both_and_market_sliding():
    g = game()
    market_card(g, 1, 'card_717')
    g.state['players'][0]['resources'] = [0, 3, 0, 0]
    before = g.state['market'][:]
    top = g.state['deck'][0]
    options = [a for a in g.legal_actions() if a.get('location') == 'A1_r1c2']
    assert {(a['buy'], a['bonus_stack']) for a in options} == {(False,1),(True,None),(True,1)}
    bonus = g.state['bonus_stacks'][1][0]
    g.step(dict(type='work', location='A1_r1c2', buy=True, bonus_stack=1))
    assert g.state['market'] == [before[0], before[2], before[3], top]
    assert g.state['players'][0]['bonuses'] == [bonus]
    assert g.state['players'][0]['reserve'] == 4
    g.validate()


def test_three_bonus_limit_keeps_buy_legal_and_disallows_replace():
    g = game()
    for i in range(3):
        g.state['players'][0]['bonuses'].append(g.state['bonus_stacks'][i].pop(0))
    market_card(g, 0, 'card_715')
    g.state['players'][0]['resources'] = [1,1,1,0]
    actions = [a for a in g.legal_actions() if a.get('location') == 'A1_r1c1']
    assert actions == [dict(type='work', location='A1_r1c1', buy=True, bonus_stack=None)]
    g.step(actions[0]); g.validate()


def test_explore_opens_for_both_without_free_location_action():
    g = game()
    market_card(g, 0, 'card_720')
    g.state['players'][0]['resources'] = [1,1,1,0]
    old_deck = g.state['deck'][:]
    g.step(buy(g, 'A1_r1c1'))
    assert g.current_player == 0 and g.state['phase'] == 'explore'
    assert g.observe()['deck_top'] == old_deck[0] and g.state['deck'] == old_deck
    location = next(k for k,v in g.state['covers'].items() if v == 'explore_1005')
    g.step(dict(type='explore', location=location))
    assert g.state['covers'][location] is None
    assert g.state['occupation'][location] is None
    assert g.state['players'][0]['resources'] == [0,1,1,0]  # tile only
    assert g.observe()['deck_top'] == old_deck[1]
    assert g.current_player == 1
    g.validate()


def test_settlers_saturate_at_twelve_and_exploration_worker_caps():
    g = game()
    p = g.state['players'][0]
    p['available'], p['reserve'] = 11, 1
    market_card(g, 0, 'card_809')
    p['resources'] = [0,2,0,1]
    g.step(buy(g, 'A1_r1c1'))
    assert p['available'] == 11 and p['reserve'] == 0  # one remains on fort
    GAME_CLASS._unlock(p, 1)
    assert p['available'] == 11 and p['reserve'] == 0
    g.validate()


def test_bonus_pairs_do_not_reuse_same_icon_and_each_tile_independent():
    p = dict(cards=['card_703','card_704','card_705','card_719','card_725'],
             explorations=['explore_1002'], reserve=3)
    assert bonus_value('bonus_900', p) == 6  # four hammers -> two pairs
    assert bonus_value('bonus_903', p) == 6  # same four can score another tile
    assert bonus_value('bonus_908', p) == 3
    assert bonus_value('bonus_913', p) == 6
    assert bonus_value('bonus_910', p) == 4
    assert bonus_value('bonus_919', p) == 3


def prepare_eighth(g, player):
    # Preserve physical card conservation while constructing a late-game state.
    desired = 'card_715'
    market_card(g, 0, desired)
    ids = [c for c in g.state['deck'] if c != desired][:7]
    give_cards(g, player, ids)
    g.state['players'][player]['resources'] = CARDS[desired]['cost'][:]


def test_first_seat_eighth_card_finishes_equal_turn_round():
    g = game()
    prepare_eighth(g, 0)
    g.step(buy(g, 'A1_r1c1'))
    assert not g.state['terminated'] and g.current_player == 1
    g.step(dict(type='rest'))
    assert g.state['terminated']
    assert [p['turns'] for p in g.state['players']] == [1,1]
    g.validate()


def test_second_seat_eighth_card_ends_immediately_and_tie_later_seat_wins():
    g = game(first=1)
    g.state['current_player'] = 0
    g.state['players'][1]['turns'] = 1
    prepare_eighth(g, 0)
    g.scores = lambda: [50, 50]
    g.step(buy(g, 'A1_r1c1'))
    assert g.state['terminated'] and g.state['winners'] == [0]
    assert [p['turns'] for p in g.state['players']] == [1,1]
    g.validate()


def test_hidden_decks_do_not_change_observations_or_features():
    g = game()
    before = g.observe()
    values = [g.features(before,a) for a in g.legal_actions()]
    g.state['deck'][1:] = reversed(g.state['deck'][1:])
    for stack in g.state['bonus_stacks']:
        stack[1:] = reversed(stack[1:])
    g.state['removed_bonuses'].reverse()
    assert g.observe() == before
    assert [g.features(g.observe(), a) for a in g.legal_actions()] == values
    assert 'deck' not in before and 'removed_bonuses' not in before
    before['players'][0]['resources'][0] = 999
    assert g.observe()['players'][0]['resources'][0] == 3


def test_json_replay_illegal_action_and_rng_rollback():
    g = game(9)
    for _ in range(12):
        obs = g.observe()
        g.step(max(g.legal_actions(), key=lambda a:g.heuristic(obs,a)))
    state = json.loads(json.dumps(g.snapshot()))
    h = GAME_CLASS(); h.restore(state)
    for _ in range(20):
        if g.current_player is None: break
        obs = g.observe()
        action = max(g.legal_actions(),key=lambda a:g.heuristic(obs,a))
        assert g.step(action) == h.step(action)
    snapshot = g.snapshot()
    with pytest.raises(InvalidAction):g.step(dict(type='work',location='fake'))
    assert g.snapshot() == snapshot
    k = game()
    original_step = k._step
    def broken(action):
        original_step(action)
        k.rng.random()
        raise RuntimeError('transaction rollback')
    k._step = broken
    before = k.snapshot()
    with pytest.raises(RuntimeError):k.step(dict(type='rest'))
    assert k.snapshot() == before


@pytest.mark.parametrize('seed', range(4))
def test_seeded_complete_heuristic_matches(seed):
    g = game(seed, seed%2)
    for _ in range(500):
        if g.current_player is None: break
        obs = g.observe()
        actions = g.legal_actions()
        for a in actions:
            f = g.features(obs, a)
            assert len(f) == 24 and all(math.isfinite(x) for x in f)
        g.step(max(actions,key=lambda a:g.heuristic(obs,a)))
        g.validate()
    assert g.state['terminated']
    assert g.outcome()['winners'] in ([0],[1])
    assert g.legal_actions() == []


def test_seeded_random_match_natural_terminal():
    g = game(81)
    rng = random.Random(81)
    for _ in range(2000):
        if g.current_player is None:break
        g.step(rng.choice(g.legal_actions()))
        g.validate()
    assert g.state['terminated']
