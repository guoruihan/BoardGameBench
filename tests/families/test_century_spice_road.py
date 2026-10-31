import json
import random
from collections import Counter
import pytest
from boardbench.contracts import InvalidAction
from boardbench.families.games.century_spice_road import (
    CenturySpiceRoad, DATA, MERCHANTS, ORDERS, play_actions, after_play,
)


def game(seed=0, first=0):
    g=CenturySpiceRoad();g.reset(seed,first);return g


def give_merchant(g,c,p=0):
    s=g.state
    for key in ('hands','played'):
        for row in s[key]:
            if c in row:row.remove(c)
    if c in s['merchant_deck']:s['merchant_deck'].remove(c)
    s['merchant_market']=[e for e in s['merchant_market'] if e['card']!=c]
    s['hands'][p].append(c)


def place_order(g,c,pos=0):
    s=g.state
    if c in s['order_deck']:s['order_deck'].remove(c)
    if c in s['order_market']:s['order_market'].remove(c)
    s['order_market'].insert(pos,c)
    if len(s['order_market'])>5:s['order_deck'].append(s['order_market'].pop())


def test_catalog_setup_and_fixed_counts():
    assert len(DATA['merchants'])==43 and len(ORDERS)==36
    assert Counter(c['kind'] for c in DATA['merchants'])=={'produce':8,'trade':34,'upgrade':1}
    assert sum(c['points'] for c in ORDERS.values())==468
    assert [c['gain'] for c in DATA['merchants'] if c['kind']=='produce' and c['gain'][3]]==[[0,0,0,1]]
    # Complete trade input/output colors are disjoint, so repeat bounds are exact.
    assert all(not any(a and b for a,b in zip(c['input'],c['output'])) for c in DATA['merchants'] if c['kind']=='trade')
    g=game(first=1);g.validate()
    assert g.state['resources']==[[4,0,0,0],[3,0,0,0]]
    assert len(g.state['merchant_market'])==6 and len(g.state['order_market'])==5


def test_upgrade_exact_steps_and_optional_skip():
    c='start-0-upgrade'
    actions=play_actions(c,[2,0,0,0])
    outcomes={tuple(after_play([2,0,0,0],a)) for a in actions}
    assert (0,2,0,0) in outcomes and (1,0,1,0) in outcomes
    assert (2,0,0,0) in outcomes and (1,1,0,0) in outcomes
    assert (0,1,1,0) not in outcomes
    assert max(len(a['upgrades']) for a in actions)==2


def test_repeated_trade_is_one_card_and_one_turn():
    g=game();c=next(c for c,v in MERCHANTS.items() if v.get('input')==[2,0,0,0] and v.get('output')==[0,0,1,0])
    give_merchant(g,c);g.state['resources'][0]=[6,0,0,0]
    assert dict(type='play',card=c,repeats=3) in g.legal_actions()
    g.step(dict(type='play',card=c,repeats=3));g.validate()
    assert g.state['resources'][0]==[0,0,3,0] and g.current_player==1
    assert c in g.state['played'][0] and c not in g.state['hands'][0]
    assert dict(type='play',card=c,repeats=0) not in play_actions(c,[6,0,0,0])


def test_purchase_pays_before_collecting_and_preserves_each_attachment():
    g=game();g.state['resources'][0]=[2,0,0,0]
    target=g.state['merchant_market'][3]
    target['resources']=[5,0,0,0]
    assert not any(a.get('card')==target['card'] and a['type']=='acquire' for a in g.legal_actions())
    g.state['resources'][0]=[1,1,1,0]
    left=[e['card'] for e in g.state['merchant_market'][:3]]
    g.step(dict(type='acquire',card=target['card'],payment=[0,1,2]))
    assert g.state['resources'][0]==[5,0,0,0]
    assert [e['resources'] for e in g.state['merchant_market'][:3]]==[[1,0,0,0],[0,1,0,0],[0,0,1,0]]
    assert [e['card'] for e in g.state['merchant_market'][:3]]==left
    assert g.current_player==1
    g.validate()


def test_overflow_requires_choice_then_turn_finishes():
    g=game();g.state['resources'][0]=[9,0,0,0]
    entry=g.state['merchant_market'][0];entry['resources']=[0,3,0,0]
    g.step(dict(type='acquire',card=entry['card'],payment=[]))
    assert g.state['phase']=='discard' and g.current_player==0
    assert all(a['type']=='discard' for a in g.legal_actions())
    g.step(dict(type='discard',resources=[0,2,0,0]))
    assert g.state['resources'][0]==[9,1,0,0] and g.current_player==1
    g.validate()


def test_last_gold_does_not_also_give_silver():
    g=game();g.state['gold']=1;g.state['coins'][1][0]=3
    c=g.state['order_market'][0];g.state['resources'][0]=list(ORDERS[c]['cost'])
    g.step(dict(type='claim',card=c))
    assert g.state['coins'][0]==[1,0] and g.state['silver']==4
    c=g.state['order_market'][0];g.state['resources'][1]=list(ORDERS[c]['cost'])
    g.step(dict(type='claim',card=c))
    assert g.state['coins'][1]==[3,1] and g.state['silver']==3
    g.validate()


def test_exact_colors_empty_rest_and_infinite_supply():
    g=game();g.state['resources'][0]=[0,0,0,10]
    assert not any(a['type']=='claim' and ORDERS[a['card']]['cost'][0] for a in g.legal_actions())
    g.step(dict(type='rest'))
    assert g.current_player==1 and g.state['resources'][0]==[0,0,0,10]
    g.state['resources'][1]=[0,0,0,0]
    g.step(dict(type='play',card='start-1-produce'))
    assert g.state['resources'][1]==[2,0,0,0]


def test_six_orders_equal_turns_last_seat_tiebreak():
    g=game(first=1)
    g.state['final_round']=True;g.scores=lambda:[61,61]
    g._end_turn()
    assert not g.state['terminated'] and g.current_player==0
    g._end_turn()
    assert g.state['winners']==[0] and g.state['turns']==[1,1]


def test_hidden_hands_deck_and_json_atomic_replay():
    g=game();obs=g.observe(0)
    g.state['merchant_deck'].reverse();g.state['order_deck'].reverse()
    g.state['hands'][1].reverse()
    assert g.observe(0)==obs
    before=g.snapshot()
    with pytest.raises(InvalidAction):g.step(dict(type='play',card='start-0-upgrade',upgrades=[True]))
    assert g.snapshot()==before
    clone=CenturySpiceRoad();clone.restore(json.loads(json.dumps(before)))
    for _ in range(12):
        obs=g.observe();a=max(g.legal_actions(),key=lambda a:g.heuristic(obs,a))
        assert g.step(a)==clone.step(a)
        assert g.snapshot()==clone.snapshot()
        g.validate()


@pytest.mark.parametrize('seed',range(6))
def test_complete_seeded_matches(seed):
    g=game(seed,seed%2);rng=random.Random(seed)
    for step in range(600):
        obs=g.observe();actions=g.legal_actions()
        a=rng.choice(actions) if seed%2 and g.current_player==seed%2 else max(actions,key=lambda a:g.heuristic(obs,a))
        assert len(g.features(obs,a))==24
        g.step(a);g.validate()
        if g.state['terminated']:break
    assert g.state['terminated'],(seed,step,g.scores())
