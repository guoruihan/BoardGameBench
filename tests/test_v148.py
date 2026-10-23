from copy import deepcopy
import time

import numpy as np
import pytest
import torch

from boardbench.environments.harmonies_duo import HarmoniesDuo
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner
from boardbench.v146.policy import ModelPlayer
from boardbench.v146.worker import audit_trace
from boardbench.v148.search import Node, Search, public_key
from boardbench.v148.worker import play, validate_job
from boardbench.v148.data import reconstruct
from test_v146 import model, job
from test_v129 import artifact


def vjob(job, **kwargs):
    return {**job,'environment_seed':148900001,**kwargs}


def test_progressive_widening_has_no_permanent_six_action_pruning():
    actions=[{'type':'choose_bundle','bundle_id':i} for i in range(20)]
    from boardbench.v125.policies import action_key
    node=Node(actions,{action_key(a):1/20 for a in actions})
    for _ in range(500):
        edge=node.select(2,1.5,.6)
        edge.visits+=1;edge.total+=90;node.visits+=1
    assert len(node.edges)==20
    assert sum(e.visits for e in node.edges)==node.visits


def test_tree_discovers_coordinated_actions(monkeypatch):
    from boardbench.v148 import search
    from boardbench.v125.policies import action_key
    def obs(path):
        return dict(path=path,active_player=0,terminated=len(path)==2,
                    players=[dict(completed_turns=0),dict(completed_turns=0)],
                    scores=[120 if path==[0,1] else 30 if path and path[0]==1 else 0,0])
    class Game:
        @classmethod
        def from_observation(cls,observation,seed):
            game=cls();game.path=list(observation['path']);return game
        def step(self,action):
            self.path.append(action['a'])
            return type('Result',(),{'observation':obs(self.path)})()
    monkeypatch.setattr(search,'FastHarmoniesDuo',Game)
    class DummyPlanner:
        def vector(self,_):pass
    class Toy(Search):
        def node(self,observation,plans):
            actions=[dict(a=0),dict(a=1)]
            return Node(actions,{action_key(a):.5 for a in actions})
        def leaf(self,game,observation,plans,depth,deadline):
            # Fixed continuation always picks action0, missing the good pair.
            return float(obs(game.path+[0])['scores'][0]) if len(game.path)==1 else float(observation['scores'][0])
    monkeypatch.setattr(search,'player_view',lambda x:x)
    s=Toy(None,None,0,seconds=900,max_depth=2,initial_width=2,widening=.5,exploration=2.)
    root=s.node(obs([]),[])
    for i in range(100):assert s.simulate(root,obs([]),[DummyPlanner(),DummyPlanner()],time.monotonic()+5,i)
    best=max(root.edges,key=lambda e:e.visits)
    assert best.action=={'a':0} and best.mean>90
    assert any(child.visits>0 for child in best.outcomes.values())
    assert root.visits==100  # Same focal player throughout: no sign inversion.


def test_real_tree_public_future_isolation_and_timeout(model,artifact):
    player=ModelPlayer(model,NativeEncoder.from_artifact(artifact))
    game=HarmoniesDuo();obs,_=game.reset(148900008);plans=[Planner(),Planner()]
    while obs['players'][0]['completed_turns']<8 and not obs['terminated']:
        obs=game.step(player.decide(obs,plans[obs['active_player']])).observation
    assert not obs['terminated']
    seat=obs['active_player'];saved=deepcopy(obs);saved_plans=deepcopy(plans)
    a=Search(player,player,seat,seconds=900,max_rounds=12,seed=17)
    b=Search(player,player,seat,seconds=900,max_rounds=12,seed=17)
    root=a.node(obs,deepcopy(plans))
    assert not a.simulate(root,obs,plans,0,19) and root.visits==0
    first=a.decide(obs,deepcopy(plans))
    game._bag.reverse();game._deck.reverse()
    second=b.decide(game.observe(),deepcopy(plans))
    assert obs==game.observe()==saved
    assert first==second and a.last['candidates']==b.last['candidates']
    assert a.last['root_visits']==12 and a.last['max_tree_depth']>=2
    assert all(p.__dict__==q.__dict__ for p,q in zip(plans,saved_plans))
    assert public_key(obs)==public_key(game.observe())


def test_tree_value_leaves_back_up_absolute_own_scores(model,artifact):
    player=ModelPlayer(model,NativeEncoder.from_artifact(artifact));obs,_=HarmoniesDuo().reset(148900012)
    calls=[]
    def controlled_value(observation,plans):
        assert observation['active_player']==0
        calls.append(observation['decision_index']);return 77.
    s=Search(player,player,0,'mcts_value',seconds=900,max_rounds=20,
             value_evaluator=controlled_value,value_fraction=1.,min_value_depth=1)
    action=s.decide(obs,[Planner(),Planner()])
    assert action in obs['legal_actions'] and s.last['root_visits']==20
    assert len(calls)==s.last['value_leaves']==20 and s.last['max_tree_depth']>=2
    assert all(c['mean']==77 for c in s.last['candidates'] if c['visits'])


@pytest.mark.parametrize('method',['search_mcts','search_mcts_wide'])
def test_complete_small_budget_games_audit_without_fake_visit_labels(job,method):
    result,trace=play(vjob(job,method=method))
    assert result['natural_terminal'] and audit_trace(trace)['scores']==trace['scores']
    trace['split']='train';trace['environment_seed']=148000001
    # Changing a seed cannot turn a calibration trajectory into train data.
    with pytest.raises(ValueError,match='replay'):reconstruct(trace)


def test_training_visit_targets_and_masks(job):
    result,trace=play(vjob(job,method='search_mcts',split='train',environment_seed=148000001))
    data=reconstruct(trace)
    assert data and all(r['label']==0 for r in data)
    assert all(not r['pi'].any() and r['mask'][r['action']] for r in data)
    broken=deepcopy(trace)
    row=next(r for r in broken['decisions'] if r['seat']==broken['seat'])
    row['search'].update(root_visits=2,completed_rounds=2,
        candidates=[dict(action=row['action'],samples=2,visits=2,mean=90.)])
    target=reconstruct(broken)[0]
    assert target['label']==2 and target['pi'].sum()==1 and target['pi'][target['action']]==1
    row['search']['root_visits']=3
    with pytest.raises(ValueError,match='backup'):reconstruct(broken)


def test_guards_reject_old_budget_split_and_uncalibrated_value(job):
    with pytest.raises(ValueError,match='split'):validate_job(job)
    with pytest.raises(ValueError,match='expired'):validate_job(vjob(job,deadline_unix=0))
    with pytest.raises(ValueError,match='validated'):Search(None,None,0,'mcts_value')
    with pytest.raises(ValueError,match='bounded'):Search(None,None,0,seconds=901)


@pytest.fixture
def learning_config(job,tmp_path):
    from boardbench.artifacts.store import digest,write_json
    _,trace=play(vjob(job,method='search_mcts',split='train',environment_seed=148000001))
    rows=reconstruct(trace)
    # Synthetic training tensor fixture only; actual production data always goes
    # through full replay. Include genuine soft labels to exercise that loss.
    arrays={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    arrays['seed']=np.asarray([148000000+i%5 for i in range(len(rows))])
    arrays['label'][:]=2
    arrays['pi']=arrays['mask']/arrays['mask'].sum(-1,keepdims=True)
    path=tmp_path/'data.npz';np.savez_compressed(path,**arrays)
    manifest=tmp_path/'manifest.json'
    write_json(manifest,dict(format='v148_search_training',split='train',path=str(path),sha256=digest(path),
        value_target='remaining_own_score_div150',games=[dict(opponent_sha256=job['opponent_sha256'])]))
    return dict(model=job['student'],model_sha256=job['student_sha256'],dataset=str(manifest),
        dataset_sha256=digest(manifest),mode='joint',learning_rate=.0001,anchor_weight=.1,
        value_weight=.25,minibatch=16,seed=148)


def test_search_policy_gradient_exact_resume_and_portable_parity(learning_config,tmp_path):
    from boardbench.v148.learning import Learner
    from boardbench.v145.policy import load_policy
    c=learning_config;t=Learner(c);before=deepcopy(t.model.state_dict());anchor=deepcopy(t.anchor.state_dict())
    result=t.step()
    assert result['gradient_norm']>0 and all(np.isfinite(v) for v in result.values())
    assert any(not torch.equal(v,t.model.state_dict()[k]) for k,v in before.items())
    t.save(tmp_path/'resume.pt');expected=t.step();weights=deepcopy(t.model.state_dict())
    other=Learner(c);other.load(tmp_path/'resume.pt');actual=other.step()
    assert expected==actual and all(torch.equal(v,other.model.state_dict()[k]) for k,v in weights.items())
    assert all(torch.equal(v,other.anchor.state_dict()[k]) for k,v in anchor.items())
    rng=other.rng.get_state().clone();metrics=other.validate()
    assert torch.equal(rng,other.rng.get_state()) and metrics['rows']>0
    other.export(tmp_path/'portable.pt');loaded=load_policy(tmp_path/'portable.pt')
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in weights.items())


def test_value_units_and_calibration_guard(monkeypatch,job):
    from boardbench.v148 import value
    leaf=value.LeafValue.__new__(value.LeafValue);leaf.player=None
    monkeypatch.setattr(value,'policy_evaluation',lambda *args:([],{},.2))
    assert leaf(dict(active_player=1,score_breakdown=[dict(total=100),dict(total=45)]),[None,None])==75
    class L:
        opponent_hashes=[job['opponent_sha256']]
    m=value.calibration(job['student'],dict(environments=8,value_mse=.01,baseline_value_mse=.02),L())
    assert m['eligible_for_search']
    assert not value.calibration(job['student'],dict(environments=7,value_mse=.01,baseline_value_mse=.02),L())['eligible_for_search']


def test_fit_persists_actual_step_count_and_value_gate(learning_config,tmp_path):
    from boardbench.v148.fit import run
    from boardbench.artifacts.store import read_json
    config={**learning_config,'mode':'value','steps':3,'evaluate_interval':1,
            'deadline_unix':time.time()+60,'finish_reserve_seconds':1}
    target=tmp_path/'fit';run(config,target,'cpu')
    done=read_json(target/'done.json')
    assert done['completed_steps']==3 and done['stop_reason']=='steps'
    assert not read_json(target/'value_manifest.json')['eligible_for_search']
    with pytest.raises(ValueError,match='completed'):run(config,target,'cpu',True)


def test_launch_needs_explicit_renewal_and_half_cpu_cap():
    import os
    from boardbench.v148.launch import validate
    now=time.time();c=dict(role='search',host=os.uname().nodename,started_unix=now-60,
        deadline_unix=now+3540,authorized_wall_seconds=3600,cpu_cores=49,seconds=900,cycles=2)
    validate(c,'search',list(range(128)),now)
    for changed in ({'deadline_unix':None},{'cpu_cores':65},{'authorized_wall_seconds':7200}):
        with pytest.raises(ValueError,match='explicit'):validate({**c,**changed},'search',list(range(128)),now)


def test_peer_resume_preserves_old_failure_records(monkeypatch,tmp_path):
    from boardbench.artifacts.store import write_json
    from boardbench.v148 import search_campaign, campaign
    peer=tmp_path/'peer';peer.mkdir();write_json(peer/'failed.json',dict(error='old retained failure'))
    write_json(peer/'status.json',dict(phase='running'))
    class R:
        stop=False;stopping=False;deadline=time.time()+3600
        config=dict(deadline_unix=deadline,learning_root=str(peer))
        c=dict(search_root=str(peer))
        def report(self):write_json(peer/'ready.json',dict(complete=True))
    monkeypatch.setattr(search_campaign.time,'sleep',lambda _:None)
    assert search_campaign.wait_for(R(),peer/'ready.json','waiting')['complete']
    (peer/'ready.json').unlink()
    assert campaign.wait_for(R(),peer/'ready.json')['complete']
