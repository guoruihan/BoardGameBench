import pytest
from boardbench.v151.play import Session, summarize


def test_explicit_actions_recovery_and_no_private_view(tmp_path):
    s=Session.create(tmp_path/'solo.json','solo',151099999,0,None)
    obs=s.env.observe();saved=s.env.save_state()
    with pytest.raises(ValueError,match='stale'):s.submit(['b0'],'choose bundle',99)
    assert s.env.save_state()==saved
    bundle=obs['token_market'][0]
    from boardbench.v151.play import SHORT
    commands=['b0']+[SHORT[c]+str(i) for i,c in enumerate(bundle)]+['e']
    s.submit(commands,'explicit fixture positions',0)
    t=Session(s.path)
    assert s.env.save_state()==t.env.save_state()
    assert t.audit()['assistant_actions']==5
    assert not {'snapshot','bag','deck','private'} & t.view().keys()
    assert all(r['actor']=='assistant' for r in t.d['records'])
    rev=t.env.s['decision_index'];n=len(t.d['records'])
    with pytest.raises(ValueError):t.submit(['a1@0'],'illegal animal must not mutate board',rev)
    assert len(t.d['records'])==n
    assert Session(t.path).env.save_state()==t.env.save_state()


def test_batch_boundary_rejected_before_any_mutation(tmp_path):
    s=Session.create(tmp_path/'solo.json','solo',151099999,0,None)
    saved=s.env.save_state()
    for commands in (['b0','e','e'],['b0','e','g0'],[''],[]):
        with pytest.raises(ValueError):s.submit(commands,'invalid batch',0)
        assert s.env.save_state()==saved
        assert not s.d['records'] and not s.d['submissions']


def test_accepted_prefix_survives_and_audit_detects_tampering(tmp_path):
    s=Session.create(tmp_path/'solo.json','solo',151099999,0,None)
    with pytest.raises(ValueError,match='animal'):
        s.submit(['b0','a1@0'],'preserve the legal prefix',0)
    t=Session(s.path)
    assert t.env.s['bundle_taken'] and len(t.d['records'])==1
    assert t.audit()['rejected_submissions']==1
    t.d['records'][0]['observation_sha256']='tampered'
    with pytest.raises(AssertionError):t.audit()


def test_own_turn_and_saved_opponent_plans(tmp_path,monkeypatch):
    class Planner:
        def __init__(self):self.horizon=2;self.active=None;self.started=0
    class AI:
        def __init__(self):self.planners=[Planner(),Planner()]
        def decide(self,obs):return obs['legal_actions'][0]
    monkeypatch.setattr('boardbench.v151.play.solver',lambda c:AI())
    s=Session.create(tmp_path/'duo.json','duo',151099999,1,{})
    assert s.env.s['active_player']==1
    assert s.d['records'] and all(r['actor']=='rl_opponent' for r in s.d['records'])
    t=Session(s.path);assert t.audit()['actions']==len(s.d['records'])
    t.env.s['active_player']=0
    with pytest.raises(ValueError,match='not assistant'):t.step(t.env.legal_actions()[0],'assistant')


def test_summary_uses_engine_tie_break_and_requires_all_games():
    report=dict(natural_terminal=True,actions=4,assistant_actions=2,assistant_turns=1,
                animal_cubes=8,opponent_animal_cubes=5,completed_cards=1,rejected_submissions=0,
                session_sha256='fixture',final=dict(score=dict(total=83,animals=33),
                opponent_score=dict(total=83,animals=20),winners=[1]))
    manifest=dict(games=[dict(id='duo0',mode='duo',seed=7,seat=1)])
    result=summarize(manifest,{'duo0':report})
    assert result['games'][0]['outcome']=='win'
    assert result['aggregates']['duo']['mean']['terrain']==50
    report['natural_terminal']=False
    with pytest.raises(ValueError,match='every'):summarize(manifest,{'duo0':report})
