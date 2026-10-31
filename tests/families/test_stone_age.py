from collections import Counter
from copy import deepcopy
import json
import random

import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.games.stone_age import (
    StoneAge, CARDS, BUILDINGS, DATA, RESOURCES, LOCATIONS, VALUES,
    payments, building_payments, final_breakdown, card_totals,
)


def game(seed=0,first=0):
    g=StoneAge();g.reset(seed,first);return g


def resolve_setup(g, locations, player=0):
    s=g.state;s['phase']='resolve';s['current_player']=player
    s['resolution_order']=[player,1-player];s['resolver']=0
    s['placements']={loc:[0,0] for loc in LOCATIONS}
    for loc,n in locations.items():s['placements'][loc][player]=n
    for p in (0,1):s['players'][p]['unplaced']=0


def move_card_to_market(g,cid,slot=0):
    s=g.state
    if cid in s['market']:
        i=s['market'].index(cid);s['market'][i],s['market'][slot]=s['market'][slot],s['market'][i]
    else:
        i=s['card_deck'].index(cid);s['card_deck'][i],s['market'][slot]=s['market'][slot],s['card_deck'][i]


def own_card(g,p,cid):
    s=g.state
    if cid in s['market']:
        i=s['market'].index(cid);s['market'][i]=s['card_deck'].pop()
    else:s['card_deck'].remove(cid)
    s['players'][p]['cards'].append(cid)


def stock(g,p,resources=(0,0,0,0),food=0):
    for res,n in zip(RESOURCES,resources):g._gain(p,res,n)
    if food:g._gain(p,'food',food)


def test_full_physical_card_pairs_building_multiplicities_and_setup():
    assert len(CARDS)==36 and len(BUILDINGS)==28
    assert Counter(c['bottom']['kind'] for c in CARDS.values())=={'culture':16,'multiplier':20}
    assert Counter(c['bottom']['symbol'] for c in CARDS.values() if c['bottom']['kind']=='culture')==dict.fromkeys(DATA['culture_symbols'],2)
    culture,mult=card_totals(CARDS)
    assert mult==dict(farmers=7,toolmakers=8,builders=9,shamans=7)
    assert Counter(c['effect']['type'] for c in CARDS.values())==dict(dice_draft=10,food=7,resource=5,roll=3,points=3,tool=1,agriculture=2,extra_card=1,one_use_tool=3,deferred_resources=1)
    assert len({tuple(c['face_source'].values()) for c in CARDS.values()})==36
    fixed=[b for b in BUILDINGS.values() if b['kind']=='fixed']
    assert len(fixed)==17
    assert all(sum(b['cost'])==3 and sum(n*v for n,v in zip(b['cost'],VALUES))==b['points'] for b in fixed)
    assert Counter(sum(n>0 for n in b['cost']) for b in fixed)=={2:9,3:8}
    g=game(first=1);g.validate()
    assert len(g.state['card_deck'])==32 and list(map(len,g.state['building_stacks']))==[7,7]
    assert all(p['population']==5 and p['food']==12 and not p['tools'] for p in g.state['players'])
    assert g.current_player==1 and all(a['type']=='place' for a in g.legal_actions())


def test_two_player_resource_exclusivity_no_second_placement_and_dynamic_village_block():
    g=game();g.step(dict(type='place',location='wood',count=1))
    assert all(a.get('location')!='wood' for a in g.legal_actions())
    g.step(dict(type='place',location='tool',count=1))
    assert all(a.get('location')!='wood' for a in g.legal_actions())
    g.step(dict(type='place',location='field',count=1))
    assert not any(a.get('location')=='hut' for a in g.legal_actions())
    assert not any(a['type']=='pass' for a in g.legal_actions())
    g.validate()


def test_capped_village_spaces_remain_legal_and_do_not_exceed_caps():
    g=game();p=g.state['players'][0];p['population']=10;p['unplaced']=10;p['agriculture']=10
    p['tools']=[dict(value=4,used=False) for _ in range(3)]
    actions=g.legal_actions()
    assert all(any(a['location']==loc for a in actions) for loc in ('hut','tool','field'))
    resolve_setup(g,{'hut':2,'tool':1,'field':1})
    for loc in ('hut','tool','field'):g.step(dict(type='resolve',location=loc))
    assert p['population']==10 and p['agriculture']==10 and sum(t['value'] for t in p['tools'])==12
    g.validate()


def test_roll_then_choose_tools_whole_values_once_and_new_tool_same_round(monkeypatch):
    g=game();resolve_setup(g,{'tool':1,'wood':2,'gold':1})
    g.step(dict(type='resolve',location='tool'))
    assert g.state['phase']=='resolve' and g.current_player==0
    rolls=iter([4,4,6]);monkeypatch.setattr(g.rng,'randint',lambda a,b:next(rolls))
    g.step(dict(type='resolve',location='wood'))
    assert g.state['pending']['dice']==[4,4]
    assert len(g.legal_actions())==2
    g.step(dict(type='use_tools',permanent=[0],one_use=[]))
    assert g.state['players'][0]['resources'][0]==3
    g.step(dict(type='resolve',location='gold'))
    assert g.legal_actions()==[dict(type='use_tools',permanent=[],one_use=[])]
    g.step(g.legal_actions()[0]);g.validate()


def test_upgrade_choice_keeps_used_flag_and_one_use_does_not_count_as_permanent():
    g=game();p=g.state['players'][0]
    p['tools']=[dict(value=1,used=True),dict(value=1,used=False),dict(value=1,used=False)]
    resolve_setup(g,{'tool':1,'hunt':1})
    g.step(dict(type='resolve',location='tool'))
    assert [a['index'] for a in g.legal_actions()]==[0,1,2]
    g.step(dict(type='upgrade_tool',index=0))
    assert p['tools']==[dict(value=2,used=True),dict(value=1,used=False),dict(value=1,used=False)]
    own_card(g,0,33);p['one_use']=[33]
    assert final_breakdown(p)['multipliers']['toolmakers']==4
    g.validate()


def test_all_resource_payment_choices_exact_kinds_and_variable_points():
    b=next(b for b in BUILDINGS.values() if b['kind']=='variable' and b['minimum']==4 and b['kinds']==2)
    choices=building_payments([4]*4,b)
    assert (1,0,3,0) in choices and (1,1,2,0) not in choices
    assert sum(a*v for a,v in zip((1,0,3,0),VALUES))==18
    assert len(payments((2,2,2,2),2,2))==10
    free=next(b for b in BUILDINGS.values() if b['kind']=='variable' and b['minimum']==1)
    assert len(building_payments([7]*4,free))==329
    assert (0,0,0,7) in building_payments([7]*4,free)


def test_variable_building_scores_actual_payment_and_ten_hunters_use_ten_dice(monkeypatch):
    g=game();s=g.state;bid=next(b['id'] for b in BUILDINGS.values() if b['kind']=='variable' and b['minimum']==1)
    old=s['building_stacks'][0][-1]
    for group in s['building_stacks']+[s['removed_buildings']]:
        if bid in group:
            idx=group.index(bid);group[idx]=old;break
    s['building_stacks'][0][-1]=bid
    stock(g,0,[0,0,0,7]);resolve_setup(g,{'building0':1})
    g.step(dict(type='resolve',location='building0'));g.step(dict(type='build',resources=[0,0,0,7]))
    assert s['players'][0]['points']==42 and s['players'][0]['resources']==[0]*4
    g.validate()
    g=game();g.state['players'][0]['population']=10;resolve_setup(g,{'hunt':10})
    monkeypatch.setattr(g.rng,'randint',lambda a,b:6)
    g.step(dict(type='resolve',location='hunt'))
    assert g.state['pending']['dice']==[6]*10
    g.step(dict(type='use_tools',permanent=[],one_use=[]))
    assert g.state['players'][0]['food']==42
    g.validate()


@pytest.mark.parametrize('starve', [False,True])
def test_feeding_all_food_then_exact_resources_or_single_penalty(starve):
    g=game();p=g.state['players'][0];p['population']=6;p['agriculture']=2
    p['food']=1;g.state['spent'][0]=11;stock(g,0,[3,1,0,0])
    g.state.update(phase='feed',feeding_index=0)
    a=dict(type='starve') if starve else dict(type='feed',resources=[3,0,0,0])
    g.step(a)
    assert p['food']==0 and p['population']==6 and p['points']==(-10 if starve else 0)
    assert p['resources']==([3,1,0,0] if starve else [0,1,0,0])
    g.validate()


def test_agriculture_surplus_is_kept_and_newborn_feeds_this_round():
    g=game();p=g.state['players'][0];p['food']=2;p['agriculture']=7;g.state['spent'][0]=10
    g.state.update(phase='feed',feeding_index=0)
    g.step(dict(type='feed',resources=[0]*4));assert p['food']==4
    g=game();resolve_setup(g,{'hut':2});g.step(dict(type='resolve',location='hut'))
    assert g.state['phase']=='feed' and g.state['players'][0]['population']==6
    g.step(dict(type='feed',resources=[0]*4));assert g.state['players'][0]['food']==6


def test_buy_card_waits_for_end_round_refill_and_allows_decline():
    g=game();move_card_to_market(g,1);stock(g,0,[3,0,0,0]);resolve_setup(g,{'card0':1,'card1':1})
    market=g.state['market'][:]
    g.step(dict(type='resolve',location='card0'));g.step(dict(type='buy_card',resources=[1,0,0,0]))
    assert g.state['market']==[None]+market[1:] and g.state['players'][0]['points']==3
    g.step(dict(type='resolve',location='card1'));g.step(dict(type='decline'))
    assert g.state['market'][1]==market[1]
    g.step(dict(type='feed',resources=[0]*4));g.step(dict(type='feed',resources=[0]*4))
    assert g.state['market'][:3]==market[1:]
    assert g.current_player==1 and g.state['round']==2
    g.validate()


def test_all_player_dice_draft_reacts_and_upgrade_returns_to_buyer(monkeypatch):
    g=game();move_card_to_market(g,31);stock(g,0,[1,0,0,0]);resolve_setup(g,{'card0':1,'hunt':1})
    g.state['players'][1]['tools']=[dict(value=1,used=False) for _ in range(3)]
    rolls=iter([6,5]);monkeypatch.setattr(g.rng,'randint',lambda a,b:next(rolls))
    g.step(dict(type='resolve',location='card0'));g.step(dict(type='buy_card',resources=[1,0,0,0]))
    assert {a['type'] for a in g.legal_actions()}=={'choose_die'}
    g.step(dict(type='choose_die',value=6));assert g.current_player==1
    g.step(dict(type='choose_die',value=5));assert g.state['phase']=='upgrade'
    g.step(dict(type='upgrade_tool',index=2))
    assert g.current_player==0 and g.state['phase']=='resolve'
    assert g.state['players'][0]['agriculture']==1 and g.state['players'][1]['tools'][2]['value']==2
    g.validate()


def test_private_extra_card_does_not_activate_upper_effect_and_does_not_leak():
    g=game();move_card_to_market(g,10);stock(g,0,[1,0,0,0]);resolve_setup(g,{'card0':1,'hunt':1})
    if 33 in g.state['market']:
        idx=g.state['market'].index(33);g.state['market'][idx],g.state['card_deck'][-1]=g.state['card_deck'][-1],33
    else:
        idx=g.state['card_deck'].index(33);g.state['card_deck'][idx],g.state['card_deck'][-1]=g.state['card_deck'][-1],33
    g.step(dict(type='resolve',location='card0'));g.step(dict(type='buy_card',resources=[1,0,0,0]))
    p=g.state['players'][0]
    assert 33 in p['cards'] and p['one_use']==[]
    obs=g.observe(1);assert obs['players'][0]['cards'] is None
    assert g.state['public_history'][-1]==dict(type='private_bonus_card',player=0)
    idx=p['cards'].index(33);p['cards'][idx],g.state['card_deck'][0]=g.state['card_deck'][0],p['cards'][idx]
    g.state['card_deck'].reverse()
    assert g.observe(1)==obs and g.scores()==[0,0]
    for stack in g.state['building_stacks']:stack[:-1]=reversed(stack[:-1])
    g.state['removed_buildings'].reverse()
    assert g.observe(1)==obs
    g.validate()


@pytest.mark.parametrize('cid', [0,3,4,15,18,22,26,33,34,35])
def test_each_immediate_fixed_effect_and_one_use_card(cid):
    g=game();move_card_to_market(g,cid);stock(g,0,[1,0,0,0]);resolve_setup(g,{'card0':1,'hunt':1})
    before=deepcopy(g.state['players'][0])
    g.step(dict(type='resolve',location='card0'));g.step(dict(type='buy_card',resources=[1,0,0,0]))
    p=g.state['players'][0];e=CARDS[cid]['effect']
    if e['type']=='agriculture':assert p['agriculture']==1
    elif e['type']=='resource':assert p['resources'][RESOURCES.index(e['resource'])]==e['amount']
    elif e['type']=='food':assert p['food']==before['food']+e['amount']
    elif e['type']=='tool':assert p['tools']==[dict(value=1,used=False)]
    elif e['type']=='points':assert p['points']==e['amount']
    elif e['type']=='one_use_tool':assert p['one_use']==[cid]
    g.validate()


def test_deferred_resources_can_interrupt_without_consuming_another_players_move():
    g=game();own_card(g,0,14);g.state['players'][0]['deferred']=[14]
    g.step(dict(type='place',location='field',count=1))
    assert g.state['phase']=='resource_window' and g.current_player==0
    before=g.state['players'][1]['unplaced']
    g.step(dict(type='take_two',card=14,resources=[1,0,0,1]))
    assert g.state['phase']=='place' and g.current_player==1
    assert g.state['players'][1]['unplaced']==before and g.state['players'][0]['resources']==[1,0,0,1]
    assert not g.state['players'][0]['deferred']
    g.validate()


def test_building_exhaustion_still_resolves_opponent_and_feeds_both():
    g=game();s=g.state;p=s['players'][0];s['players'][1]['food']=12
    p['buildings']+=s['building_stacks'][0][:-1];s['building_stacks'][0]=s['building_stacks'][0][-1:]
    stock(g,0,[10]*4);resolve_setup(g,{'building0':1})
    s['placements']['field'][1]=1
    g.step(dict(type='resolve',location='building0'))
    g.step(next(a for a in g.legal_actions() if a['type']=='build'))
    assert not s['terminated'] and g.current_player==1
    g.step(dict(type='resolve',location='field'))
    assert s['phase']=='feed' and not s['terminated']
    g.step(dict(type='feed',resources=[0]*4));assert not s['terminated']
    g.step(dict(type='feed',resources=[0]*4));assert s['terminated']
    assert s['players'][1]['food']==8 and s['end_reason']=='building_stack_exhausted'
    g.validate()


def test_civilization_refill_end_after_feeding_no_invented_round_cap():
    g=game();s=g.state
    s['players'][0]['cards']+=s['card_deck'];s['card_deck']=[]
    s['players'][0]['cards'].append(s['market'][0]);s['market'][0]=None
    s.update(phase='feed',feeding_index=0)
    g.step(dict(type='feed',resources=[0]*4));g.step(dict(type='feed',resources=[0]*4))
    assert s['terminated'] and s['round']==1 and s['end_reason']=='civilization_refill_exhausted'
    g.validate()


def test_culture_sets_multipliers_leftover_resources_and_tiebreak():
    p=deepcopy(game().state['players'][0]);p['cards']=[0,9,1,3,4,5]
    assert final_breakdown(p)['culture']==26
    p['cards']=[33,35,18,21,22];p['population']=8;p['agriculture']=7
    p['tools']=[dict(value=2,used=False),dict(value=2,used=False),dict(value=1,used=False)]
    p['one_use']=[33,35];p['buildings']=[0,1];p['resources']=[1,2,3,4]
    b=final_breakdown(p)
    assert b['multipliers']==dict(toolmakers=15,builders=6,farmers=7,shamans=16)
    assert b['total']==54
    g=game();a,b=g.state['players'];a.update(agriculture=4,population=7);a['tools']=[dict(value=2,used=False),dict(value=2,used=False),dict(value=1,used=False)]
    b.update(agriculture=6,population=5);b['tools']=[dict(value=2,used=False),dict(value=1,used=False),dict(value=1,used=False)]
    g._finish_game('fixture')
    assert g.outcome()['winners']==[0]


@pytest.mark.parametrize('seed',range(6))
def test_complete_games_invariants_replay_privacy_features_and_natural_end(seed):
    g=game(seed,seed%2);other=StoneAge();rng=random.Random(seed+400);decisions=0
    while g.current_player is not None:
        g.validate();o=g.observe();acts=g.legal_actions();assert acts
        assert g.legal_actions(1-g.current_player)==[]
        assert o['players'][1-g.current_player]['cards'] is None
        for a in acts[::max(1,len(acts)//5)]:assert len(g.features(o,a))==24
        a=max(acts,key=lambda a:g.heuristic(o,a)) if g.current_player==0 else rng.choice(acts)
        if decisions%29==0:
            other.restore(json.loads(json.dumps(g.snapshot())));assert other.step(a)==g.step(a)
        else:g.step(a)
        decisions+=1;assert decisions<3000
    g.validate()
    assert g.outcome()['reason'] in ('building_stack_exhausted','civilization_refill_exhausted')
    assert all(p['cards'] is not None for p in g.observe()['players'])
    assert g.scores()==[b['total'] for b in g.score_breakdown()]


def test_illegal_action_and_corrupt_snapshot_rollback():
    g=game();before=g.snapshot()
    with pytest.raises(InvalidAction):g.step(dict(type='place',location='hut',count=1))
    assert g.snapshot()==before
    with pytest.raises(InvalidAction):g.step(dict(type='place',location='wood',count=True))
    assert g.snapshot()==before
    bad=json.loads(json.dumps(before));bad['state']['players'][0]['resources'][0]+=1
    with pytest.raises(AssertionError):g.restore(bad)
    assert g.snapshot()==before
