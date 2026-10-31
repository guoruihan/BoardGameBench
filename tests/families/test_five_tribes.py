from collections import Counter
from copy import deepcopy
import json
import math
import random
import pytest

from boardbench.contracts import InvalidAction
from boardbench.families.games.five_tribes import (GAME_CLASS, COLORS, GOODS, VP, DATA,
    adjacent, legal_starts, blue_neighborhood, goods_score)


def game(seed=3):
    g = GAME_CLASS()
    g.reset(seed)
    return g


def blank(g):
    s = g.state
    s['bag'] = []
    for tile in s['board']:
        for i, n in enumerate(tile['meeples']):
            s['bag'] += [i]*n
        tile['meeples'] = [0]*5
    s.update(phase='move', current_player=0, current_marker=0, turn_queue=[2,1,3],
             next_bid_queue=[0], bid_queue=[], bids=[], end_trigger=False)


def put(g, tile, color, count):
    k = COLORS.index(color)
    for _ in range(count):
        g.state['bag'].remove(k)
        g.state['board'][tile]['meeples'][k] += 1


def white(g, seat, n):
    for _ in range(n):
        k = COLORS.index('white')
        if k in g.state['bag']:
            g.state['bag'].remove(k)
        else:
            tile = next(t for t in g.state['board'] if t['meeples'][k])
            tile['meeples'][k] -= 1
        g.state['players'][seat]['white'] += 1


def djinn(g, seat, name):
    s = g.state
    for zone in ('djinn_deck','djinn_row','djinn_discard'):
        if name in s[zone]:
            s[zone].remove(name)
            break
    else:
        raise AssertionError(name)
    s['players'][seat]['djinn'].append(name)


def resource(g, seat, name, n=1):
    s = g.state
    for _ in range(n):
        for zone in ('resource_deck','resource_row','resource_discard'):
            if name in s[zone]:
                s[zone].remove(name)
                break
        else:
            raise AssertionError(name)
        s['players'][seat]['resources'].append(name)


def activate(g, name, payment='elder', target=None):
    g.step(dict(type='activate', djinn=name, payment=payment, target=target))


def test_complete_component_table_and_official_djinn_vp():
    g = game()
    g.validate()
    assert len(VP) == 22 and sum(VP.values()) == 141
    assert VP['al_amin'] == 5 and VP['lamia'] == VP['hagis'] == 10
    assert sum(sum(t['meeples']) for t in g.state['board']) == 90
    assert all(p['camels'] == 11 and p['coins'] == 50 for p in g.state['players'])
    assert len(g.state['bid_queue']) == 4
    assert len(g.state['resource_row']) == 9 and len(g.state['djinn_row']) == 3


def test_zero_bids_lifo_positive_slots_unique_and_marker_round_order():
    g = game()
    s = g.state
    s['bid_queue'] = [0,2,1,3]
    s['current_player'] = 0
    for slot in (0,0,1):
        g.step(dict(type='bid', slot=slot, fakirs=0))
    assert not any(a['slot'] == 1 for a in g.legal_actions())
    g.step(dict(type='bid', slot=0, fakirs=0))
    assert s['current_marker'] == 1
    assert s['turn_queue'] == [3,2,0]
    assert s['next_bid_queue'] == [1]
    assert [p['coins'] for p in s['players']] == [49,50]
    g.validate()


def test_sow_no_backtrack_long_loop_and_matching_newly_dropped_color():
    g = game()
    blank(g)
    put(g,0,'red',5)
    assert 0 in legal_starts(g.state['board'])
    g.step(dict(type='start_sow',tile=0))
    g.step(dict(type='drop',tile=1,color='red'))
    snap = g.snapshot()
    with pytest.raises(InvalidAction):
        g.step(dict(type='drop',tile=0,color='red'))
    assert g.snapshot() == snap
    for tile in (7,6,0,1):
        g.step(dict(type='drop',tile=tile,color='red'))
    assert g.state['tribe_count'] == 2
    assert g.state['board'][1]['owner'] == 0
    assert g.state['board'][0]['owner'] is None
    assert g.state['phase'] == 'assassin'
    g.validate()


def test_final_match_capture_all_and_start_not_claimed():
    g = game()
    blank(g)
    put(g,0,'white',1)
    put(g,1,'white',3)
    g.step(dict(type='start_sow',tile=0))
    assert g.legal_actions() == [dict(type='drop',tile=1,color='white')]
    g.step(g.legal_actions()[0])
    assert g.state['players'][0]['white'] == 4
    assert g.state['board'][1]['owner'] == 0
    assert g.state['board'][0]['owner'] is None
    assert g.state['players'][0]['camels'] == 10
    g.validate()


def test_builder_exact_multiplier_fakirs_and_echidna():
    g = game()
    blank(g)
    s = g.state
    blue = [t for t in s['board'] if t['color'] == 'blue']
    red = [t for t in s['board'] if t['color'] == 'red']
    order = []
    for i in range(30):
        if i in (0,1,2,6):
            order.append(blue.pop())
        elif i in (7,8,12,13,14):
            order.append(red.pop())
        else:
            order.append((red or blue).pop())
    s['board'] = order
    assert blue_neighborhood(s['board'],7) == 4
    resource(g,0,'fakir',2)
    s.update(phase='builder',endpoint=7,tribe_count=3,tribe_color='blue')
    g.step(dict(type='builder',fakirs=2))
    assert s['players'][0]['coins'] == 70
    djinn(g,0,'echidna')
    white(g,0,2)
    s['phase'] = 'builder'
    activate(g,'echidna','two_elders')
    g.step(dict(type='builder',fakirs=0))
    assert s['players'][0]['coins'] == 94
    g.validate()


def test_last_camel_claim_before_assassin_second_capture_and_round_continues():
    g = game()
    blank(g)
    s = g.state
    for i in range(10,20):
        s['board'][i]['owner'] = 0
    s['players'][0]['camels'] = 1
    put(g,0,'red',1)
    put(g,1,'red',1)
    put(g,2,'yellow',1)
    g.step(dict(type='start_sow',tile=0))
    g.step(dict(type='drop',tile=1,color='red'))
    assert s['players'][0]['camels'] == 0 and s['end_trigger']
    g.step(dict(type='assassinate',tile=2,opponent=None,colors=['yellow'],fakirs=0))
    assert s['board'][1]['owner'] == 0 and s['board'][2]['owner'] is None
    s['phase'] = 'sale'
    g.step(dict(type='end_turn'))
    assert not s['terminated'] and s['current_marker'] == 2
    g.validate()


def test_market_prefix_and_no_immediate_refill():
    g = game()
    s = g.state
    endpoint = next(i for i,t in enumerate(s['board']) if t['kind'] == 'large_market')
    s.update(phase='tile',endpoint=endpoint)
    before = s['resource_row'][:]
    assert all(max(a['indices']) < 6 for a in g.legal_actions() if a['type'] == 'market')
    g.step(dict(type='market',indices=[1,5]))
    assert s['players'][0]['resources'] == [before[1],before[5]]
    assert len(s['resource_row']) == 7 and s['players'][0]['coins'] == 44
    assert s['resource_row'] == [c for i,c in enumerate(before) if i not in (1,5)]
    g.validate()


def test_goods_sets_wilds_and_fakir_not_sold():
    assert goods_score((2,1,1,0,0,0,0,0,0)) == 8
    assert goods_score((1,1,1,1,1,1,1,1,0),1) == 60
    assert goods_score((1,1,1,1,1,1,1,1,1),1) == 61
    g = game()
    resource(g,0,'fakir')
    for c in GOODS[:5]:
        resource(g,0,c)
    g.state['phase'] = 'sale'
    assert not any('fakir' in a.get('goods',[]) for a in g.legal_actions())
    g.step(dict(type='sell',goods=sorted(GOODS[:5])))
    assert g.state['players'][0]['coins'] == 71
    assert g.state['players'][0]['resources'] == ['fakir']
    g.validate()


def test_sacred_acquisition_new_ability_can_be_used_and_baal_not_self():
    g = game()
    blank(g)
    s = g.state
    white(g,0,4)
    if 'baal' not in s['djinn_row']:
        s['djinn_deck'].remove('baal')
        s['djinn_deck'].append(s['djinn_row'].pop())
        s['djinn_row'].append('baal')
    s.update(phase='tile',endpoint=next(i for i,t in enumerate(s['board']) if t['kind']=='sacred'))
    g.step(dict(type='get_djinn',card='baal',payment='two_elders'))
    assert s['players'][0]['coins'] == 50
    assert len(s['djinn_row']) == 2
    s['phase'] = 'tile'
    card = s['djinn_row'][0]
    g.step(dict(type='get_djinn',card=card,payment='two_elders'))
    assert s['players'][0]['coins'] == 51
    g.validate()



def test_bouraq_hagis_monkir_and_enki_lamia_chains():
    g = game()
    blank(g)
    s = g.state
    for d in ('bouraq','hagis','enki','lamia'):
        djinn(g,0,d)
    djinn(g,1,'monkir')
    white(g,0,4)
    activate(g,'bouraq')
    origin = next(i for i,t in enumerate(s['board']) if t['kind']=='village' and adjacent(i,True))
    dest = adjacent(origin,True)[-1]
    g.step(dict(type='place',origin=origin,tile=dest,payment='elder'))
    assert s['board'][dest]['palaces'] == 1 and s['palace_supply'] == 9
    assert s['players'][1]['coins'] == 52 and s['phase']=='move'
    assert 'hagis' in s['used'] and 'bouraq' in s['used']
    activate(g,'enki')
    origin = next(i for i,t in enumerate(s['board']) if t['kind']=='oasis')
    dest = adjacent(origin,True)[0]
    g.step(dict(type='place',origin=origin,tile=dest,payment='elder'))
    assert s['board'][dest]['trees'] == 1 and s['tree_supply'] == 11
    assert not any(a.get('djinn') in ('bouraq','enki') for a in g.legal_actions())
    g.validate()


def test_iblis_kandicha_and_nekir_one_event_not_two():
    g = game()
    blank(g)
    s = g.state
    djinn(g,0,'iblis')
    djinn(g,0,'kandicha')
    djinn(g,1,'nekir')
    white(g,0,1)
    put(g,1,'yellow',1)
    put(g,1,'white',1)
    s.update(phase='assassin',endpoint=0,tribe_count=2,tribe_color='red')
    activate(g,'iblis')
    g.step(dict(type='assassinate',tile=1,opponent=None,colors=['white','yellow'],fakirs=0))
    assert s['players'][0]['yellow'] == 1 and s['players'][0]['white'] == 1
    assert s['players'][1]['coins'] == 52
    assert s['board'][1]['owner'] == 0
    g.validate()


def test_boaz_protection_and_marid_per_drop():
    g = game()
    blank(g)
    s = g.state
    djinn(g,1,'boaz')
    djinn(g,1,'marid')
    white(g,1,1)
    put(g,0,'red',1)
    put(g,1,'red',1)
    s['board'][1]['owner']=1
    s['players'][1]['camels']-=1
    g.step(dict(type='start_sow',tile=0))
    g.step(dict(type='drop',tile=1,color='red'))
    assert s['players'][1]['coins'] == 52
    assert not any(a.get('opponent') == 1 for a in g.legal_actions())
    assert s['board'][1]['owner'] == 1
    g.validate()


def test_anun_nak_target_then_random_and_end_trigger_not_cancelled():
    g = game()
    blank(g)
    s = g.state
    djinn(g,0,'anun_nak')
    white(g,0,1)
    s['end_trigger']=True
    snap = g.snapshot()
    activate(g,'anun_nak',target=0)
    assert sum(s['board'][0]['meeples']) == 3 and s['end_trigger']
    clone = GAME_CLASS()
    clone.restore(snap)
    activate(clone,'anun_nak',target=0)
    assert clone.snapshot() == g.snapshot()
    g.validate()


def test_sibittis_private_offer_sloar_private_draw_and_leta_utug():
    g = game()
    blank(g)
    s = g.state
    for d in ('sibittis','sloar','leta','utug'):
        djinn(g,0,d)
    white(g,0,6)
    resource(g,0,'fakir')
    activate(g,'sibittis','two_elders')
    assert len(s['djinn_offer']) == 3
    assert len(g.observe(0)['djinn_offer']) == 3
    assert 'djinn_offer' not in g.observe(1)
    offer=s['djinn_offer'][:]
    g.step(dict(type='keep_djinn',card=offer[1]))
    assert offer[1] in s['players'][0]['djinn']
    assert all(x in s['djinn_discard'] for x in (offer[0],offer[2]))
    activate(g,'sloar','fakir')
    assert s['history'][-1]['event']=='gain_resource' and 'card' not in s['history'][-1]
    activate(g,'leta','two_elders',target=0)
    put(g,1,'green',1)
    activate(g,'utug','two_elders',target=1)
    assert s['board'][0]['owner'] == s['board'][1]['owner'] == 0
    assert sum(s['board'][1]['meeples']) == 1
    g.validate()


def test_kumarbi_discount_does_not_change_selected_position():
    g=game()
    djinn(g,0,'kumarbi')
    resource(g,0,'fakir',2)
    g.step(dict(type='bid',slot=8,fakirs=2))
    assert g.state['players'][0]['coins']==47
    assert g.state['bids'][0]['slot']==8
    g.validate()


def test_score_modifiers_all_djinn_and_majority_strict():
    g=game()
    blank(g)
    s=g.state
    for d in ('jafaar','shamhat','haurvatat','al_amin'):
        djinn(g,0,d)
    white(g,0,2)
    for _ in range(3):
        s['bag'].remove(COLORS.index('yellow'))
        s['players'][0]['yellow']+=1
    for c in GOODS[:8]:
        resource(g,0,c)
    resource(g,0,'fakir',2)
    s['board'][0]['owner']=0
    s['players'][0]['camels']-=1
    s['board'][0]['trees']=2
    s['tree_supply']-=2
    score=g.score_breakdown()[0]
    assert score['yellow']==9 and score['yellow_majority']==10 and score['white']==8
    assert score['trees']==10 and score['goods']==60
    assert score['djinn']==sum(VP[d] for d in s['players'][0]['djinn'])
    g.validate()


def test_private_coin_resource_deck_bag_and_info_masking():
    g=game()
    obs=g.observe(0)
    assert 'coins' not in obs['players'][1] and 'resources' not in obs['players'][1]
    assert not ({'bag','resource_deck','djinn_deck','rng'} & set(obs))
    clone=GAME_CLASS()
    clone.restore(g.snapshot())
    clone.state['players'][1]['coins']+=20
    clone.state['resource_deck'].reverse()
    clone.state['djinn_deck'].reverse()
    assert clone.observe(0)==obs
    assert g.step(g.legal_actions()[0])['info']['scores']==[None,None]


def test_unresolved_bidding_never_invents_credit_or_a_winner():
    g=game()
    s=g.state
    s.update(bid_queue=[0,2,1,3],current_player=0)
    s['players'][1]['coins']=0
    for _ in range(3):
        result=g.step(dict(type='bid',slot=0,fakirs=0))
    assert s['phase']=='unresolved' and not s['terminated']
    assert g.legal_actions()==[] and result['info']['unresolved_rule']
    assert s['winners']==[] and s['players'][1]['coins']==0
    g.validate()


def test_author_emergency_sale_pays_fourth_bid_and_only_triggers_when_needed():
    g=game()
    s=g.state
    s.update(bid_queue=[0,2,1,3],current_player=0)
    s['players'][1]['coins']=0
    for c in GOODS[:3]:
        resource(g,1,c)
    resource(g,1,'fakir')
    g.step(dict(type='bid',slot=0,fakirs=0))
    assert not any(a['type']=='emergency_sell' for a in g.legal_actions())
    g.step(dict(type='bid',slot=0,fakirs=0))
    g.step(dict(type='bid',slot=0,fakirs=0))
    assert s['phase']=='bid' and s['current_player']==1
    assert all(a['type']=='emergency_sell' for a in g.legal_actions())
    assert len(g.legal_actions())==7
    assert not any('fakir' in a['goods'] for a in g.legal_actions())
    snap=g.snapshot()
    clone=GAME_CLASS()
    clone.restore(json.loads(json.dumps(snap)))
    action=dict(type='emergency_sell',goods=[sorted(GOODS[:3])[0]])
    assert clone.step(action)==g.step(action)
    assert clone.snapshot()==g.snapshot()
    assert s['players'][1]['coins']==1 and s['bid_queue']==[3]
    assert g.legal_actions()==[dict(type='bid',slot=1,fakirs=0)]
    g.step(g.legal_actions()[0])
    assert s['current_marker']==3 and s['players'][1]['coins']==0
    g.validate()


def test_emergency_sale_not_available_with_kumarbi_affordable_bid():
    g=game()
    s=g.state
    s.update(bid_queue=[0,2,1,3],current_player=0)
    s['players'][1]['coins']=0
    resource(g,1,GOODS[0])
    resource(g,1,'fakir')
    djinn(g,1,'kumarbi')
    for _ in range(3):
        g.step(dict(type='bid',slot=0,fakirs=0))
    assert g.legal_actions()==[dict(type='bid',slot=1,fakirs=1)]
    before=g.snapshot()
    with pytest.raises(InvalidAction):
        g.step(dict(type='emergency_sell',goods=[GOODS[0]]))
    assert g.snapshot()==before


def test_rules_v2_rejects_pre_ruling_snapshot():
    g=game()
    snap=g.snapshot()
    snap['rules_version']='1'
    with pytest.raises(ValueError,match='snapshot schema/game/config mismatch'):
        GAME_CLASS().restore(snap)


def test_final_move_triggers_end_before_same_turn_anun_restoration():
    g=game()
    blank(g)
    s=g.state
    put(g,0,'white',1)
    put(g,1,'white',1)
    put(g,10,'yellow',1)  # An isolated meeple does not itself create a move.
    djinn(g,0,'anun_nak')
    g.step(dict(type='start_sow',tile=0))
    assert not s['end_trigger']  # A partial sow is never a completion check.
    g.step(dict(type='drop',tile=1,color='white'))
    assert s['end_trigger'] and not legal_starts(s['board'])
    activate(g,'anun_nak',target=9)
    assert legal_starts(s['board'])  # Seed 3 restores a real legal move here.
    assert s['end_trigger'] and not s['terminated']
    g.validate()


def test_unresolved_anun_bag_shortage_retains_state_without_fake_draw():
    g=game()
    s=g.state
    first=s['board'][0]
    for k,n in enumerate(first['meeples']):
        s['board'][1]['meeples'][k]+=n
    first['meeples']=[0]*5
    white(g,0,1)
    djinn(g,0,'anun_nak')
    s['phase']='move'
    before=g.rng.getstate()
    activate(g,'anun_nak',target=0)
    assert s['phase']=='unresolved' and s['players'][0]['white']==1
    assert sum(first['meeples'])==0 and g.rng.getstate()==before
    g.validate()


def test_anun_shortage_is_reachable_from_natural_seed_without_state_editing():
    g=game(22)
    for slot in (1,3,0,0):
        g.step(dict(type='bid',slot=slot,fakirs=0))
    g.step(dict(type='start_sow',tile=3))
    for tile,color in ((2,'blue'),(8,'blue'),(9,'white')):
        g.step(dict(type='drop',tile=tile,color=color))
    g.step(dict(type='market',indices=[1]))
    g.step(dict(type='end_turn'))
    g.step(dict(type='start_sow',tile=2))
    for tile,color in ((1,'blue'),(0,'blue'),(6,'green'),(7,'white')):
        g.step(dict(type='drop',tile=tile,color=color))
    g.step(dict(type='get_djinn',card='anun_nak',payment='two_elders'))
    assert g.state['bag']==[COLORS.index('white')]*2
    before=g.snapshot()
    result=g.step(dict(type='activate',djinn='anun_nak',payment='fakir',target=2))
    assert result['info']['unresolved_rule']
    assert g.state['blocked_reason']=='anun_nak_bag_fewer_than_three_requires_ruling'
    assert g.state['players']==before['state']['players']
    assert g.state['bag']==before['state']['bag']
    assert g.rng.getstate()==before['rng']
    g.validate()


def test_snapshot_in_mid_sow_and_invalid_rng_atomicity():
    g=game()
    while g.state['phase']=='bid':
        g.step(g.legal_actions()[0])
    g.step(next(a for a in g.legal_actions() if a['type']=='start_sow'))
    snap=g.snapshot()
    clone=GAME_CLASS()
    clone.restore(json.loads(json.dumps(snap)))
    with pytest.raises(InvalidAction):
        g.step(dict(type='end_turn'))
    assert g.snapshot()==snap
    rng=random.Random(8)
    for _ in range(75):
        if g.current_player is None or not g.legal_actions():
            break
        a=rng.choice(g.legal_actions())
        assert clone.step(a)==g.step(a)
        assert clone.snapshot()==g.snapshot()


@pytest.mark.parametrize('policy',['random','heuristic'])
@pytest.mark.parametrize('seed',range(4))
def test_complete_games_and_recorded_rule_boundary(policy,seed):
    g=GAME_CLASS()
    g.reset(seed,first_player=seed%2)
    rng=random.Random(seed)
    for step in range(3000):
        g.validate()
        if g.current_player is None or g.state['blocked_reason']:
            break
        actions=g.legal_actions()
        assert actions,(seed,g.state['phase'])
        obs=g.observe()
        a=rng.choice(actions) if policy=='random' else max(actions,key=lambda x:g.heuristic(obs,x))
        f=g.features(obs,a)
        assert len(f)==24 and all(math.isfinite(v) for v in f)
        g.step(a)
    if policy == 'random' and seed == 0:
        # Retain this failed-to-complete seed in the denominator. Do not invent
        # a free bid or erase its unresolved outcome to make a completion claim.
        assert g.state['blocked_reason'] == 'no_affordable_bid_and_zero_slots_full'
        assert not g.state['terminated'] and not g.state['winners']
        return
    assert g.state['terminated'],(policy,seed,step,g.state['phase'],g.state['blocked_reason'])
    assert g.state['end_reason']=='end_of_triggered_round'
    assert all(isinstance(x,int) for x in g.scores())


def test_green_prefix_no_refill_and_oasis_forced_on_opponents_tile():
    g=game()
    blank(g)
    s=g.state
    oasis=next(i for i,t in enumerate(s['board']) if t['kind']=='oasis')
    start=adjacent(oasis)[0]
    put(g,start,'green',1)
    put(g,oasis,'green',2)
    s['board'][oasis]['owner']=1
    s['players'][1]['camels']-=1
    before=s['resource_row'][:]
    g.step(dict(type='start_sow',tile=start))
    g.step(dict(type='drop',tile=oasis,color='green'))
    assert s['players'][0]['resources']==before[:3]
    assert s['resource_row']==before[3:]
    assert not any(a['type']=='skip_tile' for a in g.legal_actions())
    g.step(dict(type='tile_effect'))
    g.step(dict(type='place',origin=oasis,tile=oasis,payment=None))
    assert s['board'][oasis]['trees']==1 and s['board'][oasis]['owner']==1
    g.validate()


def test_kandicha_builder_scope_and_green_hidden_draw():
    g=game()
    blank(g)
    s=g.state
    djinn(g,0,'kandicha')
    djinn(g,0,'echidna')
    put(g,1,'blue',1)
    put(g,2,'green',1)
    white(g,0,2)
    s.update(phase='assassin',endpoint=0,tribe_color='red',tribe_count=3)
    activate(g,'echidna','two_elders')
    expected=blue_neighborhood(s['board'],1)
    g.step(dict(type='assassinate',tile=1,opponent=None,colors=['blue'],fakirs=0))
    assert s['players'][0]['coins']==50+expected  # Explicit scoped interpretation.
    s['phase']='assassin'
    before=len(s['players'][0]['resources'])
    g.step(dict(type='assassinate',tile=2,opponent=None,colors=['green'],fakirs=0))
    assert len(s['players'][0]['resources'])==before+1
    assert any(e['event']=='gain_resource' and e.get('private_draw')==1 for e in s['history'])
    g.validate()
