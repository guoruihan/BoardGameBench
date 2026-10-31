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


# Frozen dev regression: cem_r1/g011_gate.json row 18 (rules 1). No model
# checkpoint is needed; these public linear weights reproduce the original path.
CEM_DEADLOCK_WEIGHTS = [
    [
        1.0339469492693878, 0.06337439132822442, 0.1390475500509063,
        0.012857256650012451, 0.20926113159657367, 0.013741093014958766,
        -0.12748214094084587, -0.017744878069935386, 0.012163363447312451,
        -0.010928349693898637, 0.05671507099039895, -0.16836332943044427,
        0.13761782365633823, 0.11034552848530466, -0.012804960265985793,
        0.14616212931619357, 0.16163133034868968, -0.23253057962381046,
        -0.02765646842750674, -0.05951751051379024, 0.0886777414421023,
        -0.10769978965170418, 0.20319023767280614, 0.1632056228892336,
    ],
    [
        1.0506906618509777, -0.19094630660362394, 0.1300256060838756,
        0.09698852712865746, 0.0982597873945096, 0.09367286267546396,
        -0.18298701727190025, 0.002219230320013409, 0.026630068146046786,
        0.031628172768096284, -0.09452896959696665, -0.19678054466427777,
        0.2724167980747631, -0.05844121721614342, -0.031880492231250945,
        0.2475805138851797, 0.2476858446678074, 0.03643516654187692,
        0.07657427662560204, 0.06127125654697824, 0.35570697995362405,
        0.09627364461713958, 0.2800385747008483, 0.048441289191692494,
    ],
]


def deadlock_regression():
    from boardbench.families.play import make_policy, policy_seed
    g=game(11011009)
    policies=[make_policy(Gizmos, dict(kind='linear', weights=w), policy_seed(11011009,p))
              for p,w in enumerate(CEM_DEADLOCK_WEIGHTS)]
    for _ in range(106):
        p=g.current_player
        g.step(policies[p].decide(g.observe(p),g.legal_actions()))
    assert g.scores()==[24,16] and g.current_player==0
    assert g.state['phase']=='main' and not g.state['terminated']
    assert g.state['energy'][0]==[0,0,0,5]
    assert g.state['archives'][0]==['g3-31'] and g._limits(0)['no_research']
    assert g._base_actions()==[]
    g.validate()
    return g,policies


def test_forced_pass_reproduces_real_full_storage_full_archive_no_research_deadlock():
    g,_=deadlock_regression()
    assert g.legal_actions()==[dict(type='forced_pass')]
    features=g.features(g.observe(),dict(type='forced_pass'))
    assert len(features)==24 and features==g.features(g.observe(),dict(type='end_turn'))
    before=g.snapshot()
    clone=Gizmos();clone.restore(json.loads(json.dumps(before)))
    result=g.step(dict(type='forced_pass'))
    assert clone.step(dict(type='forced_pass'))==result
    assert clone.snapshot()==g.snapshot()
    assert g.current_player==1 and not result['terminated']
    assert g.state['turns']==[before['state']['turns'][0]+1,before['state']['turns'][1]]
    assert g.rng.getstate()==before['rng']
    for field in ('energy','energy_row','internal','market','decks','built','archives','vp'):
        assert g.state[field]==before['state'][field]
    assert g.state['pending']==[] and g.state['tasks']==[]
    assert g.state['history'][-1]['action']==dict(type='forced_pass')
    g.validate()


def test_no_forced_pass_when_a_build_or_research_remains_available():
    g,_=deadlock_regression()
    old=g.state['archives'][0].pop()
    g.state['decks'][CARDS[old]['level']-1].append(old)
    target=next(c for c,d in CARDS.items() if d['level']==1 and d['color']==3
                and d['cost']==1 and c not in sum(g.state['built'],[]))
    move(g,target,'archives')
    actions=g.legal_actions()
    assert actions and all(a['type']=='build' for a in actions)
    before=g.snapshot()
    with pytest.raises(InvalidAction):g.step(dict(type='forced_pass'))
    assert g.snapshot()==before
    g.validate()
    # A legal Research can always decline its cards; it must not become a pass.
    for cid in list(g.state['built'][0]):
        if CARDS[cid].get('no_research'):move(g,cid,'decks',CARDS[cid]['level']-1)
    assert any(a['type']=='research' for a in g.legal_actions())
    assert not any(a['type']=='forced_pass' for a in g.legal_actions())
    g.validate()


def test_forced_pass_finishes_the_last_seats_natural_final_turn():
    g,_=deadlock_regression()
    g.state['first_player']=1
    g.state['turns']=[10,11]
    g.state['final_round']=True
    result=g.step(dict(type='forced_pass'))
    assert result['terminated'] and g.state['turns']==[11,11]
    assert g.state['end_reason']=='equal_turns_after_build_threshold'
    g.validate()


def test_original_cem_deadlock_game_now_reaches_natural_ending():
    g,policies=deadlock_regression()
    passes=0
    for _ in range(500):
        p=g.current_player
        a=policies[p].decide(g.observe(p),g.legal_actions())
        passes+=a['type']=='forced_pass'
        g.step(a);g.validate()
        if g.state['terminated']:break
    assert g.state['terminated'] and passes>=1
    assert g.state['end_reason']=='equal_turns_after_build_threshold'
    old=g.snapshot();old['rules_version']='1'
    with pytest.raises(ValueError,match='mismatch'):Gizmos().restore(old)


def test_both_players_blocked_remains_external_truncation_not_a_draw(monkeypatch):
    from boardbench.families import play
    g,_=deadlock_regression()
    # Both public engines have full black storage, full unaffordable archives,
    # no Research and no converters; preserve every physical card and sphere.
    for cid in list(g.state['built'][1]):
        if CARDS[cid]['level']:
            move(g,cid,'decks',CARDS[cid]['level']-1)
    move(g,'g3-00',p=1)
    for cid in list(g.state['archives'][1]):
        move(g,cid,'decks',CARDS[cid]['level']-1)
    target=next(c for c in g.state['decks'][0] if CARDS[c]['color']==0)
    move(g,target,'archives',1)
    give_energy(g,1,[0,0,0,5])
    before=g.snapshot()
    g.validate()
    for _ in range(4):
        assert g.legal_actions()==[dict(type='forced_pass')]
        result=g.step(dict(type='forced_pass'));g.validate()
        assert not result['terminated'] and result['info']['winners']==[]
        assert g.rng.getstate()==before['rng']
    class BlockedGizmos(Gizmos):
        def reset(self,seed=0,first_player=0):
            return self.restore(before)
    monkeypatch.setattr(play,'game_class',lambda game_id:BlockedGizmos)
    result=play.run_match('gizmos',[dict(kind='heuristic')]*2,max_decisions=6)
    assert result['status']=='truncated' and result['decisions']==6
    assert result['rewards'] is None and result['outcome']['winners']==[]
    assert not result['outcome']['terminated']
