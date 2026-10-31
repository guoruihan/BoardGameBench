import json
import random
from collections import Counter
import pytest
from boardbench.contracts import InvalidAction
from boardbench.families.games.gizmos import Gizmos, CARDS, DATA, payment_options


def game(seed=0,first=0):
    g=Gizmos();g.reset(seed,first);return g


def find(**criteria):
    return next(cid for cid,c in CARDS.items() if all(c.get(k)==v for k,v in criteria.items()))


def move(g,cid,destination='built',p=0):
    s=g.state
    for key in ('decks','market','built','archives'):
        for row in s[key]:
            if cid in row:row.remove(cid)
    if cid in s['removed']:s['removed'].remove(cid)
    if s['research'] and cid in s['research']['cards']:s['research']['cards'].remove(cid)
    s[destination][p].append(cid)


def give_energy(g,p,values):
    s=g.state
    for i,(old,new) in enumerate(zip(s['energy'][p],values)):
        s['internal'][i]+=old-new
    s['energy'][p]=list(values)


def test_original_photographed_catalog_and_setup():
    assert Counter(c['level'] for c in CARDS.values())=={0:4,1:36,2:36,3:36}
    assert all('source_locator' in c for c in CARDS.values())
    assert Counter(c['kind'] for c in CARDS.values() if c['level']==3)=={'upgrade':16,'converter':4,'trigger':12+4}
    g=game();g.validate()
    assert [len(d) for d in g.state['decks']]==[32,33,14]
    assert [len(row) for row in g.state['market']]==[4,3,2]
    assert len(g.state['removed'])==22 and len(g.state['energy_row'])==6
    assert g.scores()==[0,0] and g._limits(0)['storage']==5


def test_file_starter_random_is_not_a_pick():
    g=game()
    c=find(level=1,trigger='pick',colors=[0])
    move(g,c)
    file_action=next(a for a in g.legal_actions() if a['type']=='file')
    g.step(file_action)
    assert g.state['pending']==['g0-0']
    assert sum(g.state['energy'][0])==0
    g.step(dict(type='activate',card='g0-0'))
    assert sum(g.state['energy'][0])==1 and g.current_player==1
    assert c not in g.state['pending']
    g.validate()


def test_storage_full_pick_and_random_no_draw():
    g=game();give_energy(g,0,[5,0,0,0])
    assert not any(a['type']=='pick' for a in g.legal_actions())
    g.step(next(a for a in g.legal_actions() if a['type']=='file'))
    rng=g.rng.getstate();internal=list(g.state['internal'])
    g.step(dict(type='activate',card='g0-0'))
    assert g.rng.getstate()==rng and g.state['internal']==internal
    assert sum(g.state['energy'][0])==5
    g.validate()


def test_single_doubler_cannot_be_used_repeatedly():
    d=find(level=2,kind='converter',converters=[dict(type='double',source=3)])
    assert ((0,0,0,2),(d,)) in payment_options((0,0,0,2),(d,),3,3)
    assert not payment_options((0,0,0,2),(d,),3,4)
    # An already-used converter is absent from legal payment resources.
    g=game();move(g,d);give_energy(g,0,[0,0,0,2])
    c=find(level=2,color=3,cost=3)
    assert any(a['pay']==[0,0,0,2] for a in g._payments(c,'market'))
    g.state['used']=[d]
    assert not list(g._payments(c,'market'))


def test_converter_chain_matches_official_effect_sheet_example():
    to_heat=find(level=1,kind='converter',converters=[dict(type='wild',source=1)])
    double_heat=find(level=2,kind='converter',converters=[dict(type='double',source=0)])
    two_any=find(level=2,kind='converter',converters=[dict(type='wild',source=0)]*2)
    conv=tuple(sorted([to_heat,double_heat,two_any]))
    options=payment_options((0,1,0,0),conv,2,2)
    assert ((0,1,0,0),conv) in options
    assert not payment_options((0,1,0,0),(to_heat,double_heat),2,2)


def test_new_gizmo_not_triggered_by_own_build_but_later_build_can():
    g=game();c=find(level=2,trigger='build',source='archive',effect='pick',color=0)
    move(g,c,'archives');give_energy(g,0,[3,0,0,0])
    a=next(a for a in g.legal_actions() if a['type']=='build' and a['card']==c and not a['converters'])
    g.step(a)
    assert c not in g.state['pending'] and g.current_player==1
    g.state['current_player']=0
    g._event('build',list(g.state['built'][0]),card=find(level=1),color=0,source='archive')
    assert c in g.state['pending']


def test_generic_build_triggers_all_colors_once_and_order_is_free():
    g=game()
    first=find(level=1,trigger='build',effect='vp',colors=[0])
    second=find(level=1,trigger='build',effect='vp',colors=[2])
    double=find(level=2,trigger='build',effect='vp',colors=[0,2])
    for c in [first,second,double]:move(g,c)
    rainbow=find(level=3,color=None)
    g._event('build',list(g.state['built'][0]),card=rainbow,color=None,source='market')
    assert set(g.state['pending'])=={first,second,double}
    g.state['phase']='effects'
    g.step(dict(type='activate',card=second))
    assert g.state['vp'][0]==1 and first in g.state['pending']
    g._event('build',list(g.state['built'][0]),card=rainbow,color=None,source='market')
    assert second not in g.state['pending'] and g.state['pending'].count(double)==1


def test_research_draws_full_private_set_and_chosen_bottom_order():
    g=game();deck_before=list(g.state['decks'][0])
    g.step(dict(type='research',level=1))
    assert g.state['research']['cards']==list(reversed(deck_before[-3:]))
    assert 'cards' in g.observe(0)['research'] and 'cards' not in g.observe(1)['research']
    before=g.observe(1)
    g.state['research']['cards'][0],g.state['decks'][0][0]=g.state['decks'][0][0],g.state['research']['cards'][0]
    assert g.observe(1)==before
    g.step(dict(type='research_pass'))
    order=list(reversed(g.state['research']['cards']))
    for c in order:g.step(dict(type='return_research',card=c))
    assert g.current_player==1
    assert g.state['decks'][0][:3]==list(reversed(order))
    assert all('card' not in event['action'] for event in g.state['history'] if event['action']['type']=='return_research')
    g.validate()


def test_research_short_deck_and_no_file_no_research_modifiers():
    g=game();removed=g.state['decks'][0][2:];g.state['decks'][0]=g.state['decks'][0][:2];g.state['removed']+=removed
    g.step(dict(type='research',level=1))
    assert len(g.state['research']['cards'])==2
    g=game()
    for c in [find(no_file=True),find(no_research=True)]:move(g,c)
    assert not any(a['type'] in ('file','research') for a in g.legal_actions())
    g.validate()


def test_discount_and_generic_payment_choices():
    g=game();d=find(discount_archive=1);move(g,d)
    c=find(level=1,color=0,cost=1);move(g,c,'archives');give_energy(g,0,[1,0,0,0])
    actions=list(g._payments(c,'archive'))
    assert any(a['pay']==[0]*4 and a['discount']==1 for a in actions)
    assert any(a['pay']==[1,0,0,0] and a['discount']==0 for a in actions)
    assert set(pay for pay,_ in payment_options((1,1,1,1),(),None,2))=={
        (1,1,0,0),(1,0,1,0),(1,0,0,1),(0,1,1,0),(0,1,0,1),(0,0,1,1)}


def test_chain_must_finish_after_sixteenth_build_and_equal_rounds():
    g=game()
    trigger=find(level=1,trigger='build',effect='vp',colors=[0])
    target=find(level=1,color=0,trigger='file')
    others=[cid for cid,c in CARDS.items() if c['level']==1 and cid not in [trigger,target]][:13]
    for cid in others+[trigger]:move(g,cid)
    move(g,target,'archives');give_energy(g,0,[1,0,0,0])
    a=next(a for a in g.legal_actions() if a['type']=='build' and a['card']==target and not a['converters'])
    g.step(a)
    assert g.state['final_round'] and not g.state['terminated'] and g.current_player==0
    assert trigger in g.state['pending']
    g.step(dict(type='activate',card=trigger))
    assert g.state['vp'][0]==1
    if g.current_player==0:g.step(dict(type='end_turn'))
    assert g.current_player==1 and not g.state['terminated']
    g.step(next(a for a in g.legal_actions() if a['type']=='pick'))
    assert g.state['terminated'] and g.state['turns']==[1,1]
    g.validate()


def test_variable_score_and_official_tiebreaks():
    g=game();give_energy(g,0,[1,1,1,1]);g.state['vp'][0]=7
    move(g,find(award='energy'));move(g,find(award='vp'))
    assert g.scores()[0]==18  # 4 energy award +7 token award +7 tokens.
    g=game(first=1);g.scores=lambda:[20,20]
    g.state['final_round']=True;g.state['current_player']=0;g.state['turns']=[0,1]
    g._end_turn();assert g.state['winners']==[0]  # Last seat is final tiebreak.


def test_atomic_rejection_and_json_replay():
    g=game();before=g.snapshot()
    with pytest.raises(InvalidAction):g.step(dict(type='research',level=True))
    assert g.snapshot()==before
    clone=Gizmos();clone.restore(json.loads(json.dumps(before)))
    for _ in range(25):
        obs=g.observe();a=max(g.legal_actions(),key=lambda a:g.heuristic(obs,a))
        assert g.step(a)==clone.step(a)
        assert g.snapshot()==clone.snapshot()
        g.validate()


@pytest.mark.parametrize('seed',range(8))
def test_complete_seeded_matches(seed):
    g=game(seed,seed%2);rng=random.Random(seed)
    for step in range(800):
        obs=g.observe();actions=g.legal_actions();assert actions
        a=rng.choice(actions) if seed%2 and g.current_player==seed%2 else max(actions,key=lambda a:g.heuristic(obs,a))
        assert len(g.features(obs,a))==24
        g.step(a);g.validate()
        if g.state['terminated']:break
    assert g.state['terminated'],(seed,step,g.scores())
