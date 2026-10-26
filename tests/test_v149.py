from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from boardbench.artifacts.store import read_json
from boardbench.environments.harmonies_duo import FastHarmoniesDuo, player_view
from boardbench.v138.baseline import Planner, build_model
from boardbench.v140.encoding import ACTIONS, FEATURES, encode_batch
from boardbench.v140.model import DuoPolicy
from boardbench.v143.policy import SelfplayPolicy
from boardbench.v146.policy import ModelPlayer
from boardbench.v149.search import Search
from boardbench.v149.value import Value, remaining_value


@pytest.fixture
def model():
    torch.set_num_threads(1)
    return SelfplayPolicy(DuoPolicy(build_model(16),'score')).eval()


def test_value_path_matches_both_seats_and_never_changes_weights(model):
    game=FastHarmoniesDuo();obs,_=game.reset(42)
    x=torch.randn(5,FEATURES,dtype=torch.float64);mask=torch.ones(5,ACTIONS,dtype=torch.bool)
    before=deepcopy(model.state_dict())
    with torch.inference_mode():
        assert torch.allclose(remaining_value(model,x),model(x,mask)[1],rtol=0,atol=1e-12)
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
    leaf=Value.__new__(Value);leaf.model=model;leaf.native=None
    plans=[Planner(),Planner()]
    for seat in (0,1):
        obs,_=game.reset(42,first_player=seat)
        p=deepcopy(plans)
        x=encode_batch([obs],[p[seat].vector(player_view(obs))])
        with torch.inference_mode():expected=150*model(torch.as_tensor(x,dtype=torch.float64),mask[:1])[1].item()
        assert leaf.batch([obs],[deepcopy(plans)],seat)[0]==pytest.approx(expected,abs=1e-10)
        with pytest.raises(ValueError,match='focal'):leaf.batch([obs],[plans],1-seat)
    terminal={**obs,'terminated':True,'scores':[108,71]}
    assert leaf.batch([terminal],[plans],0)==[108.]
    assert leaf.batch([terminal],[plans],1)==[71.]


@pytest.mark.parametrize('mode',('value_one','value_short','value_turn','rollout'))
def test_public_fork_only_and_complete_rounds(model,mode,monkeypatch):
    env=FastHarmoniesDuo();obs,_=env.reset(149900001);snapshot=env.save_state()
    leaf=Value.__new__(Value);leaf.model=model;leaf.native=None
    player=ModelPlayer(model);plans=[Planner(),Planner()]
    search=Search(player,player,0,mode,seconds=1,value=leaf,max_rounds=1,width=2)
    called=[];original=FastHarmoniesDuo.from_observation
    def fork(cls,public,seed):
        assert 'private' not in public and 'bag' not in public and 'deck' not in public
        called.append(seed);return original(public,seed)
    monkeypatch.setattr(FastHarmoniesDuo,'from_observation',classmethod(fork))
    action=search.decide(obs,plans)
    assert action in obs['legal_actions']
    assert env.save_state()==snapshot
    assert search.last['budget_scope']=='per_actual_atomic_decision'
    assert search.last['completed_rounds']<=1
    if called:assert len(set(called))==1  # Common future for every root in one round.


def test_opponent_is_simulated_before_value(model):
    env=FastHarmoniesDuo();obs,_=env.reset(149900002)
    while {'type':'end_turn'} not in obs['legal_actions']:
        obs=env.step(obs['legal_actions'][0]).observation
    calls=[]
    class Leaf:
        def batch(self,states,memories,seat):
            assert seat==0
            for s in states:
                assert s['terminated'] or s['active_player']==seat
                assert s['players'][1]['completed_turns']==1
            calls.extend(states);return [77.]*len(states)
    player=ModelPlayer(model)
    search=Search(player,player,0,'value_one',1,value=Leaf())
    values,steps=search.short_round(obs,[{'type':'end_turn'}],14,[Planner(),Planner()],time.monotonic()+10)
    assert values==[77.] and steps>1 and calls


def test_timeout_never_backs_up_partial_candidate_round(model):
    env=FastHarmoniesDuo();obs,_=env.reset(149900003)
    now=[0.]
    class Slow:
        def batch(self,states,plans,seat):
            now[0]=2.;return [1234.]*len(states)
    player=ModelPlayer(model)
    search=Search(player,player,0,'value_one',1,value=Slow(),clock=lambda:now[0])
    a=search.decide(obs,[Planner(),Planner()])
    assert a in obs['legal_actions']
    assert search.last['completed_rounds']==0 and search.last['fallback']
    assert search.last['budget_excess_seconds']==1.  # Never hide an uninterruptible overrun.


def test_budget_is_renewed_per_actual_call(model):
    env=FastHarmoniesDuo();obs,_=env.reset(149900004)
    leaf=Value.__new__(Value);leaf.model=model;leaf.native=None
    player=ModelPlayer(model);search=Search(player,player,0,'value_one',1,value=leaf,max_rounds=1,width=2)
    search.spent=10000  # Previous calls do not consume this call's allowance.
    search.decide(obs,[Planner(),Planner()])
    assert search.last['completed_rounds']==1
    assert search.spent>10000


def test_wall_timer_interrupts_candidate_work_and_returns_legal_fallback(model,monkeypatch):
    env=FastHarmoniesDuo();obs,_=env.reset(149900005)
    player=ModelPlayer(model)
    def slow(*_):
        time.sleep(1)
        raise AssertionError('deadline should interrupt')
    monkeypatch.setattr(player,'rank_batch',slow)
    search=Search(player,player,0,'rollout',.1)
    start=time.monotonic();a=search.decide(obs,[Planner(),Planner()]);elapsed=time.monotonic()-start
    assert a in obs['legal_actions'] and elapsed<.2
    assert search.last['deadline_interrupted'] and search.last['fallback']


def test_parent_deadline_suspends_even_uncooperative_search():
    from boardbench.v149.timed import TimedSearch
    class Slow:
        seconds=.1
        def decide(self,*_):
            time.sleep(5)
            raise AssertionError('parent should suspend before this')
    env=FastHarmoniesDuo();obs,_=env.reset(149900006)
    search=TimedSearch(Slow())
    try:
        for _ in range(2):
            start=time.monotonic();action=search.decide(obs,[Planner(),Planner()])
            assert time.monotonic()-start<.2 and action in obs['legal_actions']
            assert search.last['resident_timeout']
        time.sleep(.02)
        assert Path(f'/proc/{search.process.pid}/stat').read_text().rsplit(')',1)[1].split()[0]=='T'
    finally:search.close()
    assert not search.process.is_alive()


def test_ipc_keeps_partial_frames_across_deadlines():
    import socket
    from threading import Thread
    from boardbench.v149.timed import Channel
    a,b=socket.socketpair();left,right=Channel(a),Channel(b)
    large={'id':1,'data':'x'*1000000}
    try:
        with pytest.raises(TimeoutError):left.send(large,time.monotonic()+.01)
        assert left.outgoing
        thread=Thread(target=lambda:left.send({'id':2},time.monotonic()+5));thread.start()
        assert right.receive(time.monotonic()+5)==large
        assert right.receive(time.monotonic()+5)=={'id':2}
        thread.join(timeout=5);assert not thread.is_alive()
    finally:a.close();b.close()
