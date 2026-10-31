"""Rule-level tests use real printed components, not generated replacement maps."""
from copy import deepcopy
import json
import math
import random

import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.data.el_dorado import BLOCKADES, CARDS, TILE_ROWS, ROUTE
from boardbench.families.games.el_dorado import (
    ElDorado, ALL_IDS, BOARD, NEIGHBORS, SEAMS, STARTS, FINISHES, card_type,
)


def make(seed=0):
    g = ElDorado()
    g.reset(seed)
    return g


def hand(g, names):
    """Move existing physical cards to seat zero; retain conservation."""
    p = g.state['players'][0]
    p['deck'].extend(p['hand'])
    p['hand'] = []
    chosen = []
    for name in names:
        candidates = sorted(c for c in ALL_IDS if card_type(c) == name and c not in chosen)
        c = candidates[0]
        zones = [q[z] for q in g.state['players'] for z in ('hand', 'deck', 'discard', 'used')]
        zones += list(g.state['market'].values()) + [g.state['removed']]
        for zone in zones:
            if c in zone:
                zone.remove(c)
                break
        p['hand'].append(c)
        chosen.append(c)
    g.state['market_slots'] = [n if n and g.state['market'][n] else None for n in g.state['market_slots']]
    return chosen


def put(g, pos, pawn=0):
    occupied = [pos]
    g.state['players'][0]['pawns'][pawn] = pos
    for seat in (0, 1):
        for k in (0, 1):
            if seat == 0 and k == pawn:
                continue
            old = g.state['players'][seat]['pawns'][k]
            if old in occupied:
                old = next(i for i, c in enumerate(BOARD) if c['color'] != 'x' and not c['finish'] and i not in occupied)
                g.state['players'][seat]['pawns'][k] = old
            occupied.append(old)


def action(g, kind, **fields):
    return next(a for a in g.legal_actions() if a['type'] == kind and all(a.get(k) == v for k, v in fields.items()))


def edge(color, cost=None, same_tile=True):
    return next((i, j) for i, c in enumerate(BOARD) if c['color'] != 'x' and not c['finish']
                for j in NEIGHBORS[i] if BOARD[j]['color'] == color
                and (cost is None or BOARD[j]['cost'] == cost)
                and (not same_tile or BOARD[j]['tile'] == c['tile']) and not BOARD[j]['finish'])


def test_original_component_counts_and_corrected_transcription():
    assert len(BOARD) == 188
    assert len(CARDS) == 21
    assert len(ALL_IDS) == 70
    assert [len(row.split()) for row in TILE_ROWS['C']] == [4, 5, 6, 7, 6, 5, 4]
    assert TILE_ROWS['I'][3].split()[3:5] == ['g2', 'c1']
    assert TILE_ROWS['I'][4].split()[2:4] == ['xx', 'xx']
    assert TILE_ROWS['I'][5].split()[0] == 'b2'
    assert TILE_ROWS['K'][1].split()[2] == 'b3'
    assert TILE_ROWS['K'][0].split()[1] == 'g2'
    assert [r[1] for r in ROUTE] == [2, 5, 2, 2, 0]
    assert [(b['rank'], b['cost']) for b in BLOCKADES] == [(1,1),(2,1),(3,1),(4,1),(5,2),(6,2)]
    assert len(SEAMS) == 56
    for seed in range(12):
        g = make(seed)
        g.validate()
        assert sorted(b['rank'] for b in g.state['blockades']+g.state['unused_blockades']) == list(range(1,7))


def test_setup_seats_two_pawns_and_hidden_hands():
    g = ElDorado(); g.reset(7, first_player=1)
    assert g.state['players'][1]['pawns'] == [STARTS[1], STARTS[3]]
    assert g.state['players'][0]['pawns'] == [STARTS[2], STARTS[4]]
    assert len(g.observe(0)['players'][0]['hand']) == 4
    assert 'hand' not in g.observe(0)['players'][1]
    for p in g.observe()['players']:
        assert 'deck' not in p
    assert g.legal_actions(0) == []


def test_observation_does_not_leak_deck_order_or_opponent_hand():
    g = make()
    a = g.observe(0)
    p = g.state['players'][1]
    p['deck'].reverse()
    p['hand'][0], p['deck'][0] = p['deck'][0], p['hand'][0]
    g.state['players'][0]['deck'].reverse()
    assert g.observe(0) == a


def test_single_card_power_cannot_be_combined_or_split_between_pawns():
    g=make(); src, dst=edge('g',3); put(g,src)
    cards=hand(g,['explorer','explorer','scout'])
    g.step(action(g,'play_move',card=cards[2],pawn=0,color='g'))
    assert not any(a.get('to')==dst for a in g.legal_actions())
    assert all(a['type'] in ('move','stop_move','remove_blockade') for a in g.legal_actions())
    assert g.state['active']['pawn']==0


def test_power_three_enters_cost_three_once():
    g=make(); src,dst=edge('g',3);put(g,src);c=hand(g,['trailblazer'])[0]
    g.step(action(g,'play_move',card=c,pawn=0,color='g'))
    g.step(action(g,'move',to=dst))
    assert g.state['active']['power']==0
    assert all(BOARD[a['to']]['cost']==0 for a in g.legal_actions() if a['type']=='move')
    g.validate()


def test_mountains_and_own_other_explorer_block_native():
    g=make();src,dst=edge('g',1);put(g,src);hand(g,['native'])
    g.state['players'][0]['pawns'][1]=dst
    acts=g.legal_actions()
    assert not any(a['type']=='native_move' and a.get('to')==dst for a in acts)
    assert all(BOARD[a['to']]['color']!='x' for a in acts if a['type']=='native_move')


def test_rubble_discards_item_but_camp_trashes_card():
    for color in ('r','c'):
        g=make();src,dst=edge(color,1);put(g,src);c=hand(g,['compass'])[0]
        g.step(action(g,'pay_move',pawn=0,to=dst,cards=[c]))
        assert (c in g.state['removed']) == (color=='c')
        assert (c in g.state['players'][0]['used']) == (color=='r')
        g.validate()


def test_item_printed_movement_is_removed():
    g=make();src,dst=edge('g',1);put(g,src);c=hand(g,['giant_machete'])[0]
    g.step(action(g,'play_move',card=c,pawn=0,color='g'))
    assert c in g.state['removed']
    g.step(action(g,'move',to=dst));g.validate()


def test_blockade_has_separate_cost_and_remaining_power():
    g=make(); (src,dst),idx=next(((i,j),b)for (i,j),b in SEAMS.items() if BOARD[i]['color']!='x' and BOARD[j]['color']=='g' and BOARD[j]['cost']==1)
    put(g,src);c=hand(g,['trailblazer'])[0]
    # Swap real blockade rank 1 into this seam, preserving all six physical tiles.
    allb=g.state['blockades']+g.state['unused_blockades'];where=next(i for i,b in enumerate(allb)if b['rank']==1)
    allb[idx],allb[where]=allb[where],allb[idx];g.state['blockades'],g.state['unused_blockades']=allb[:4],allb[4:]
    g.step(action(g,'play_move',card=c,pawn=0,color='g'))
    assert not any(a.get('to')==dst for a in g.legal_actions())
    g.step(action(g,'remove_blockade',blockade=idx))
    assert g.state['players'][0]['pawns'][0]==src
    assert g.state['active']['power']==2
    g.step(action(g,'move',to=dst))
    assert g.state['active']['power']==1
    assert g.state['players'][0]['blockades']==[1]
    g.validate()


def test_native_removes_only_blockade_not_terrain_together():
    g=make();(src,dst),idx=next(((i,j),b)for (i,j),b in SEAMS.items()if BOARD[i]['color']!='x')
    put(g,src);c=hand(g,['native'])[0]
    g.step(action(g,'native_blockade',card=c,pawn=0,blockade=idx))
    assert g.state['players'][0]['pawns'][0]==src
    assert g.state['blockades'][idx] is None
    assert c in g.state['players'][0]['used']


def test_used_cards_not_reshuffled_by_action_draw():
    g=make(); c=hand(g,['cartographer'])[0];p=g.state['players'][0]
    # Keep the remainder in used, so the action cannot draw itself back.
    p['used'].extend(p['deck']);p['deck']=[]
    g.step(action(g,'play_action',card=c))
    assert p['hand']==[] and c in p['used']
    g.validate()


def test_scientist_draw_then_optional_trash_and_card_retained():
    g=make();c=hand(g,['scientist'])[0]
    g.step(action(g,'play_action',card=c))
    assert g.state['phase']=='trash'
    assert len(g.state['players'][0]['hand'])==1
    target=g.state['players'][0]['hand'][0]
    g.step(action(g,'trash',card=target))
    assert target in g.state['removed'] and c in g.state['players'][0]['used']
    assert g.state['phase']=='play';g.validate()


def test_travel_log_draws_two_and_can_decline_trash():
    g=make();c=hand(g,['travel_log'])[0]
    g.step(action(g,'play_action',card=c));assert len(g.state['players'][0]['hand'])==2
    assert c in g.state['removed'];g.step(action(g,'finish_effect'));assert g.state['phase']=='play'


def test_buy_phase_cannot_resume_ordinary_movement():
    g=make();hand(g,['explorer','traveler','native']);g.step(action(g,'start_buy'))
    assert not any(a['type'] in ('play_move','pay_move','pay_blockade')for a in g.legal_actions())
    # Purple action cards remain usable during this phase.
    assert any(a['type']=='native_move'for a in g.legal_actions())


def test_purchase_half_coins_yellow_and_one_normal_purchase():
    g=make();a,b=hand(g,['traveler','explorer']);g.step(action(g,'start_buy'))
    g.step(action(g,'spend',card=a,mode='coin'));g.step(action(g,'spend',card=b,mode='half'))
    assert g.state['money2']==3
    g.step(action(g,'buy',card='scout'))
    assert len(g.state['players'][0]['fresh'])==1
    assert not any(a['type']in('buy','spend')for a in g.legal_actions())
    assert g.state['money2']==1;g.validate()


def test_market_outside_stacks_require_vacancy():
    g=make();g.state['phase']='buy';g.state['money2']=10
    assert not any(a.get('card')=='native' and a['type']=='buy' for a in g.legal_actions())
    name=g.state['market_slots'][0]
    g.state['players'][1]['discard'].extend(g.state['market'][name]);g.state['market'][name]=[]
    g.state['market_slots'][0]=None
    g.step(action(g,'buy',card='native',slot=0))
    assert g.state['market_slots'][0]=='native' and len(g.state['market']['native'])==2;g.validate()


def test_transmitter_can_get_outside_without_consuming_purchase():
    g=make();c=hand(g,['transmitter'])[0]
    g.step(action(g,'play_action',card=c));g.step(action(g,'gain',card='native'))
    assert c in g.state['removed']
    assert 'native' not in g.state['market_slots']
    assert not g.state['bought'] and g.state['phase']=='play';g.validate()


def test_transmitter_empty_supply_consumes_and_returns():
    g=make();c=hand(g,['transmitter'])[0]
    for pile in g.state['market'].values():g.state['players'][1]['discard'].extend(pile);pile.clear()
    g.state['market_slots']=[None]*6
    g.step(action(g,'play_action',card=c))
    assert g.state['phase']=='play';g.validate()


def test_item_coin_function_vs_half_coin():
    for mode,coins,removed in [('coin',8,True),('half',1,False)]:
        g=make();c=hand(g,['treasure_chest'])[0];g.step(action(g,'start_buy'))
        g.step(action(g,'spend',card=c,mode=mode))
        assert g.state['money2']==coins and (c in g.state['removed'])==removed


def test_keep_and_refill_does_not_discard_retained_card():
    g=make();p=g.state['players'][0];keep=p['hand'][0]
    g.step(action(g,'end_turn'))
    for c in list(p['hand'][1:]):g.step(action(g,'discard',card=c))
    g.step(action(g,'refill'))
    assert keep in p['hand'] and len(p['hand'])==4 and g.current_player==1;g.validate()


def test_only_second_explorer_triggers_final_round_and_equal_turns():
    g=make();dest=FINISHES[0];src=next(i for i in NEIGHBORS[dest]if not BOARD[i]['finish'])
    put(g,src);c=hand(g,['native'])[0];g.step(action(g,'native_move',card=c,pawn=0,to=dest))
    assert not g.state['final_round'] and not g.state['terminated']
    # A fresh copy moved from supply lets the second pawn arrive.
    g.state['players'][0]['pawns'][1]=src;c=hand(g,['native'])[0]
    g.step(action(g,'native_move',card=c,pawn=1,to=dest))
    assert g.state['final_round'] and not g.state['terminated']
    g.step(action(g,'end_turn'));g.step(action(g,'refill'))
    assert g.current_player==1 and not g.state['terminated']
    g.step(action(g,'end_turn'));g.step(action(g,'refill'))
    assert g.state['terminated'] and g.outcome()['winners']==[0]
    assert g.state['turns']==[1,1]


def test_printed_blockade_rank_not_symbol_cost_breaks_tie():
    g=make();g.state['finish_order']=[0,1]
    g.state['players'][0]['blockades']=[5]
    g.state['players'][1]['blockades']=[6]
    g._finish_race();assert g.outcome()['winners']==[1]
    g=make();g.state['finish_order']=[1,0];g._finish_race();assert g.outcome()['winners']==[1]


def test_invalid_action_is_atomic_for_state_rng_and_strict_types():
    g=make();before=g.snapshot()
    for bad in [None,{}, {'type':'buy','card':'native','slot':0}, {'type':'play_move','card':'explorer:0','pawn':False,'color':'g'}]:
        with pytest.raises(InvalidAction):g.step(bad)
        assert g.snapshot()==before


def test_json_snapshot_replay_and_public_features():
    g=make(12);r=random.Random(8)
    for _ in range(80):g.step(r.choice(g.legal_actions()))
    h=ElDorado();h.restore(json.loads(json.dumps(g.snapshot())))
    for _ in range(100):
        assert g.legal_actions()==h.legal_actions()
        a=r.choice(g.legal_actions());o=g.observe();f=g.features(o,a)
        assert len(f)==24 and all(math.isfinite(x)for x in f)
        assert g.step(a)==h.step(a);g.validate();h.validate()
        if g.state['terminated']:break
    assert g.snapshot()==h.snapshot()


@pytest.mark.parametrize('seed',[0,1,2])
def test_complete_heuristic_games(seed):
    g=make(seed)
    for _ in range(1800):
        obs=g.observe();acts=g.legal_actions();assert acts
        g.step(max(acts,key=lambda a:g.heuristic(obs,a)));g.validate()
        if g.state['terminated']:break
    assert g.state['terminated']
    assert g.outcome()['scores']==[None,None]
    assert len(g.outcome()['winners'])==1
    assert g.legal_actions()==[]
    assert g.state['turns'][0] == g.state['turns'][1]
