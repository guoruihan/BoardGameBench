import json
import math
import random
from copy import deepcopy
from collections import Counter

import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.data.targi import TRIBES, GOODS, BORDER, OPPOSITE
from boardbench.families.games.targi import Targi, displayed, effects, intersections, payments, breakdown, rearrangements


def game(seed=0):
    g=Targi();g.reset(seed);return g


def remove_tribe(g,card):
    s=g.state
    for name in ('tribe_deck','tribe_discard'):
        if card in s[name]:s[name].remove(card);return
    for c in s['center']:
        if c['kind']=='tribe' and c['card']==card:c['card']=None;return
    for p in s['players']:
        for row in p['rows']:
            if card in row:row.remove(card);return
        if p['reserve']==card:p['reserve']=None;p['reserve_round']=None;return
    raise AssertionError(card)


def give(g,player,*cards):
    for card in cards:
        remove_tribe(g,card)
        row=next(row for row in g.state['players'][player]['rows'] if len(row)<4)
        row.append(card)


def pending(g,card,source='center'):
    remove_tribe(g,card)
    g.state.update(phase='tribe',pending=dict(card=card,source=source))


def action(g,kind,**fields):
    return next(a for a in g.legal_actions() if a['type']==kind and all(a.get(k)==v for k,v in fields.items()))


def test_catalog_original_full_box():
    assert len(TRIBES)==45 and len(GOODS)==19 and len(BORDER)==16
    assert Counter(c['kind'] for c in TRIBES.values())==dict.fromkeys(['well','camel','oasis','camp','targia'],9)
    for kind in ('well','camel','oasis','camp','targia'):
        assert Counter(c['vp'] for c in TRIBES.values() if c['kind']==kind)=={1:4,2:3,3:2}
    assert sum(c is None for c in GOODS.values())==3
    assert sum(c==(0,0,0,1,0) for c in GOODS.values())==3
    assert TRIBES[26]['costs']==[(1,0,2,0),(0,0,0,1)]
    assert TRIBES[23]['costs']==[(2,2,1,0)]
    assert sum(c['vp'] for c in TRIBES.values())==80


def test_initial_board_start_resources_and_robber():
    g=game();g.validate();s=g.state
    assert s['robber']==1 and s['round']==1
    assert [c['kind'] for c in s['center']]==['goods','tribe','goods','tribe','goods','tribe','goods','tribe','goods']
    assert all(p['resources']==[2,2,2,1] and p['vp']==4 for p in s['players'])
    assert 1 not in [a['border'] for a in g.legal_actions()]
    assert 11 in [a['border'] for a in g.legal_actions()]  # Opposite robber legal.


def test_placement_opponent_opposite_only_and_unique_intersections():
    g=game();g.step(dict(type='place',border=2))
    assert 10 not in [a['border'] for a in g.legal_actions()]
    g.step(dict(type='place',border=3))
    assert 10 in [a['border'] for a in g.legal_actions()]
    assert intersections([2,10,5])==[1]
    assert intersections([1,2,5])==[0,1]
    assert intersections([1,2,3])==[]


def test_robber_permission_does_not_override_opponent_opposite():
    g=game();give(g,0,32)
    assert dict(type='place',border=1) in g.legal_actions()
    g.state['players'][1]['pawns']=[11]
    assert dict(type='place',border=1) not in g.legal_actions()


def test_action_order_entire_first_player_then_second():
    g=game()
    for b in [2,3,5,13,10,6]:g.step(dict(type='place',border=b))
    assert g.current_player==0 and g.state['phase']=='actions'
    while g.current_player==0:
        a=next((a for a in g.legal_actions() if a['type']=='skip'),None)
        if a is None:a=action(g,'finish_turn')
        g.step(a)
    assert g.current_player==1 and g.state['phase']=='actions'


def test_skip_keeps_center_but_consuming_replaces_hidden():
    g=game();p=g.state['players'][0];p['markers']=[0];g.state['phase']='actions'
    original=deepcopy(g.state['center'][0]);snap=g.snapshot()
    g.step(dict(type='skip',zone='center',position=0))
    assert g.state['center'][0]==original
    g.restore(snap);g.step(dict(type='use',zone='center',position=0));g.validate()
    c=g.state['center'][0]
    assert c['kind']=='tribe' and not c['face_up']
    assert g.observe(0)['center'][0]['card'] is None and g.observe(1)['center'][0]['card'] is None


def test_reserve_cannot_replace_and_noble_must_be_later_round():
    g=game();pending(g,0);g.step(action(g,'reserve'));g.validate()
    assert g.observe(1)['players'][0]['reserve'] is None
    assert g.observe(1)['players'][0]['has_reserve']
    pending(g,1)
    assert not any(a['type']=='reserve' for a in g.legal_actions())
    g.step(action(g,'decline'))
    g.state.update(phase='tribe',pending=dict(card=0,source='noble'))
    g.state['players'][0]['resources']=[10,10,10,10]
    assert not any(a['type']=='build' for a in g.legal_actions())
    g.state['round']=2
    assert any(a['type']=='build' for a in g.legal_actions())
    g.step(action(g,'build',row=0));g.validate()
    assert g.state['players'][0]['reserve'] is None


def test_alternative_costs_never_mixed_and_discounts_not_gold():
    g=game();p=g.state['players'][0];p['resources']=[1,0,2,1]
    assert payments(p,26,'center')==[((0,0,0,1),0),((1,0,2,0),0)]
    give(g,0,26,30)
    p['resources']=[10,10,10,10]
    # Camp 10 requires one date, one pepper, one gold; both ordinary goods waived.
    assert payments(p,10,'noble')==[((0,0,0,1),0)]
    assert payments(p,20,'noble')==[((0,0,0,1),0)]


def test_all_ordinary_discount_choices():
    g=game();give(g,0,24);p=g.state['players'][0];p['resources']=[10]*4
    assert payments(p,3,'center')==[((0,1,0,1),0),((1,0,0,1),0)]


def test_goods_bonuses_once_per_round_not_border_and_retroactive():
    g=game();give(g,0,33);p=g.state['players'][0];p['resources']=[0]*4
    g._gain(0,[0,1,0,0]);assert p['resources'][1]==1
    g._gain(0,[0,2,0,0],goods_card=True);assert p['resources'][1]==4
    g._gain(0,[0,1,0,0],goods_card=True);assert p['resources'][1]==5
    g._gain(0,[1,0,0,0],goods_card=True)
    give(g,0,35);g._goods_bonuses(0)
    assert p['resources'][0]==2


def test_bank_gold_restricted_to_tribes_not_supply():
    g=game();p=g.state['players'][0];p['resources']=[10]*4
    pending(g,40);g.step(action(g,'build',row=0));g.validate()
    assert p['bank_gold']==1 and p['resources'][3]==9
    p['resources'][3]=0
    assert payments(p,20,'center')==[((0,0,0,0),1)]
    g.state['phase']='silversmith'
    assert not any(a.get('resource')==3 for a in g.legal_actions())
    pending(g,20);g.step(action(g,'build',row=0))
    assert p['bank_gold']==0


def test_trader_discount_keeps_three_to_one_gold_and_repeat():
    g=game();give(g,0,39);p=g.state['players'][0];p['resources']=[3,0,0,0];g.state['phase']='trader'
    assert dict(type='trade',give=0,take=3,rate=3) in g.legal_actions()
    assert dict(type='trade',give=0,take=1,rate=1) in g.legal_actions()
    g.step(dict(type='trade',give=0,take=1,rate=1));assert g.state['phase']=='trader'
    assert p['resources']==[2,1,0,0]


def test_silversmith_exactly_one_transaction():
    g=game();g.state['phase']='silversmith';p=g.state['players'][0];p['resources']=[4,0,0,2]
    g.step(dict(type='sell',resource=0,count=4,vp=3))
    assert p['vp']==7 and g.state['phase']=='actions'


def test_hand_surcharge_once_per_round_and_any_ordinary_good():
    g=game();give(g,0,38);p=g.state['players'][0];p['resources']=[1,1,1,1]
    pending(g,20);g.step(action(g,'reserve'));g.state['phase']='actions'
    builds=[a for a in g.legal_actions() if a['type']=='build']
    assert len(builds)==9
    g.step(builds[0]);assert p['hand_used'] and p['reserve'] is None
    pending(g,24);g.step(action(g,'reserve'))
    assert not any(a['type']=='build' for a in g.legal_actions())


def test_immediate_goods_and_bonus_choice_are_real_choices():
    g=game();p=g.state['players'][0];p['resources']=[20]*4
    pending(g,42);g.step(action(g,'build',row=0))
    assert g.state['bonus_count']==5 and len(g.legal_actions())==21
    g.step(dict(type='gain_goods',goods=[2,1,2]));g.validate()
    pending(g,37);g.step(action(g,'build',row=0))
    assert g.state['bonus_count']==1
    g.step(dict(type='gain_goods',goods=[1,0,0]))
    pending(g,41);g.step(action(g,'build',row=0))
    assert {a['choice'] for a in g.legal_actions()}=={'goods','gold','vp'}


def test_rearrange_two_cards_with_final_gaps_forbidden():
    rows=[[0,1,2],[3,4],[5,6,7,8]]
    moves=list(rearrangements(rows))
    assert [[7,1,2],[3,4],[5,6,0,8]] in moves  # Swap any positions.
    assert [[0,1,2],[3,4,8,7],[5,6]] in moves  # Two row ends.
    assert [[8,1,2],[3,4,0],[5,6,7]] in moves  # Temporary gap filled.
    assert [[0,1,2,8],[3,4],[5,6,7]] in moves  # Just one move.
    assert all(Counter(c for row in r for c in row)==Counter(range(9)) for r in moves)


def test_fata_morgana_moves_remaining_marker_face_up_only():
    g=game();g.state['phase']='fata_morgana';p,q=g.state['players'];p['markers']=[0,1];q['markers']=[2]
    g.state['center'][3]['face_up']=False
    moves=[a for a in g.legal_actions() if a['type']=='move_marker']
    assert all(a['target'] not in [0,1,2,3] for a in moves)
    g.step(dict(type='move_marker',source=0,target=4))
    assert p['markers']==[1,4] and g.state['phase']=='actions'


def test_penalty_applies_next_round_not_this_turn():
    g=game();p,q=g.state['players'];p['resources']=[20]*4
    pending(g,44);g.step(action(g,'build',row=0))
    assert q['penalty_round']==2 and q['quota']==3
    g.state['current_player']=1;g._end_player()
    assert g.state['round']==2 and q['quota']==2 and p['quota']==3


def test_no_goods_bonus_excludes_gold_and_border_income():
    g=game();give(g,0,36);g.state['phase']='actions';p=g.state['players'][0]
    g._gain(0,[1,0,0,0]);g._gain(0,[0,0,0,1],goods_card=True)
    g.step(action(g,'finish_turn'))
    assert g.state['phase']=='choose_goods' and g.state['bonus_count']==1
    g.step(dict(type='gain_goods',goods=[0,1,0]))
    assert g.current_player==1


def test_end_turn_limits_after_bonus_not_before():
    g=game();give(g,0,36);g.state['phase']='actions';p=g.state['players'][0];p['resources']=[4,3,3,5]
    g.step(dict(type='finish_turn'));g.step(dict(type='gain_goods',goods=[0,0,1]))
    assert g.state['phase']=='discard_excess' and p['resources'][3]==3
    assert len(g.legal_actions())==3
    g.step(dict(type='discard_goods',goods=[0,0,1]))
    assert p['resources']==[4,3,3,3] and g.current_player==1


def test_raids_mixed_goods_immunity_point_and_insolvency():
    g=game();g.state.update(phase='raid',robber=12);p=g.state['players'][0]
    give(g,0,29,31)
    assert dict(type='raid_goods',goods=[1,1,1]) in g.legal_actions()
    before=p['vp'];g.step(dict(type='raid_immune'));assert p['vp']==before
    g.state.update(phase='raid',robber=16,current_player=0,raid_done=[])
    # Remove immunity to test genuine inability.
    remove_tribe(g,29);g.state['tribe_discard'].append(29)
    p['resources'][3]=0;p['vp']=2;other=g.state['players'][1]['vp']
    assert g.legal_actions()==[dict(type='raid_insolvent',opponent_vp=3)]
    g.step(g.legal_actions()[0]);assert g.state['players'][1]['vp']==other+3 and p['vp']==2


def test_raid_reward_requires_payment():
    g=game();give(g,0,31);g.state.update(phase='raid',robber=4)
    p=g.state['players'][0];before=p['vp'];g.step(dict(type='raid_vp',vp=1));assert p['vp']==before


def test_endgame_golden_rulebook_33_points():
    # Exact official manual page 6 example: 21 card VP + 3 tokens + 3 ability
    # points + homogeneous oasis row 4 + distinct row 2.
    g=game();p,q=g.state['players'];
    p['rows']=[[28,0,15,2],[12,40,39,23],[24,35,32,8]];p['vp']=3
    detail=breakdown(p,q)
    assert detail==dict(tokens=3,cards=21,rows=6,abilities=3,total=33)


def test_diverse_row_bonus_max_two_and_majority_ten():
    g=game();p,q=g.state['players']
    p['rows']=[[22,0,3,6],[12,1,4,7],[20,21,15,17]]
    d=breakdown(p,q)
    assert d['rows']==6 and d['abilities']==8  # 4 diversity, 1 majority, 1 ten, 1 pair well, 1 pair targia.


def test_twelve_cards_finishes_full_round_and_tiebreak():
    g=game();p,q=g.state['players'];give(g,0,*range(12));g.state['phase']='actions'
    g.step(action(g,'finish_turn'));assert not g.state['terminated'] and g.current_player==1
    g.step(action(g,'finish_turn'));assert g.state['terminated'] and g.state['end_reason']=='twelve_tribes'
    g=game();p,q=g.state['players'];p['resources'][3]=2;g._finish_game('test');assert g.state['winners']==[0]
    g=game();g._finish_game('test');assert g.state['winners']==[0,1]


def test_deck_shuffle_is_immediate_at_empty():
    g=game();s=g.state
    s['goods_discard']=s['goods_deck'][1:];s['goods_deck']=s['goods_deck'][:1]
    before=set(s['goods_discard']);card=g._draw('goods')
    assert set(s['goods_deck'])==before and not s['goods_discard']
    s['goods_discard'].append(card);g.validate()


def test_invalid_action_atomic_rng_and_json_replay():
    g=game();before=json.dumps(g.snapshot(),sort_keys=True)
    for a in ({'type':'place','border':True},{'type':'place','border':1},{'type':'place','border':2,'extra':1}):
        with pytest.raises(InvalidAction):g.step(a)
        assert json.dumps(g.snapshot(),sort_keys=True)==before
    rng=random.Random(4)
    for _ in range(90):g.step(rng.choice(g.legal_actions()))
    other=Targi();other.restore(json.loads(json.dumps(g.snapshot())))
    for _ in range(80):
        assert g.observe()==other.observe() and g.legal_actions()==other.legal_actions()
        if g.state['terminated']:break
        a=rng.choice(g.legal_actions());assert g.step(a)==other.step(a)
        g.validate();other.validate()
    assert json.dumps(g.snapshot(),sort_keys=True)==json.dumps(other.snapshot(),sort_keys=True)


def test_hidden_deck_order_and_reserve_do_not_change_opponent_observation():
    g=game();pending(g,0);g.step(action(g,'reserve'));before=g.observe(1)
    g.state['tribe_deck'].reverse();g.state['goods_deck'].reverse()
    reserve=g.state['players'][0]['reserve'];replacement=g.state['tribe_deck'].pop()
    g.state['tribe_deck'].append(reserve);g.state['players'][0]['reserve']=replacement
    assert g.observe(1)==before
    before['players'][1]['resources'][0]=999
    assert g.state['players'][1]['resources'][0]!=999


@pytest.mark.parametrize('seed',range(6))
def test_natural_games_conservation_and_features(seed):
    g=game(seed);rng=random.Random(seed)
    for n in range(1500):
        g.validate()
        if g.state['terminated']:break
        obs=g.observe();actions=g.legal_actions();assert actions
        features=[g.features(obs,a) for a in actions]
        assert all(len(f)==24 and all(math.isfinite(v) for v in f) for f in features)
        a=actions[max(range(len(actions)),key=lambda i:features[i][0])] if seed<3 else rng.choice(actions)
        g.step(a)
    assert g.state['terminated'] and g.state['end_reason'] in ('fourth_raid','twelve_tribes')
    assert g.legal_actions()==[]


@pytest.mark.parametrize('card',range(45))
def test_every_original_card_builds_and_resolves_without_losing_cards(card):
    g=game();g.state['players'][0]['resources']=[30]*4
    pending(g,card);g.step(action(g,'build',row=0))
    while g.state['phase']!='actions':
        g.step(g.legal_actions()[0])
    g.validate()
    assert card in displayed(g.state['players'][0])


def test_asymmetric_two_pawn_placement_completes_and_no_stall():
    g=game();g.state['players'][1]['quota']=2
    actors=[]
    while g.state['phase']=='place':
        actors.append(g.current_player);g.step(g.legal_actions()[0])
    assert actors==[0,1,0,1,0]
    assert len(g.state['players'][0]['pawns'])==3
    assert len(g.state['players'][1]['pawns'])==2
    assert g.current_player==0
