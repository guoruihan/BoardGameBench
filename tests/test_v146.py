from copy import deepcopy
import time

import numpy as np
import pytest
import torch

from boardbench.artifacts.store import digest
from boardbench.environments.harmonies_duo import HarmoniesDuo, FastHarmoniesDuo, player_view
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner, build_model
from boardbench.v140.model import DuoPolicy
from boardbench.v143.policy import SelfplayPolicy
from boardbench.v144.policy import export_policy,SelfplaySolver
from boardbench.v146.policy import Habitat,ModelPlayer,personal_view,terrain_rank,mixed_roots
from boardbench.v146.search import TerminalSearch
from boardbench.v146.worker import play,audit_trace,validate_job
from boardbench.v146.report import summarize,paired,pilot_choices,teacher_choice
from test_v129 import artifact


@pytest.fixture
def model():
    torch.set_num_threads(1);torch.manual_seed(146)
    return SelfplayPolicy(DuoPolicy(build_model(16),'score')).eval()


@pytest.fixture
def job(model,tmp_path,artifact):
    path=tmp_path/'policy.pt';export_policy(model,path,dict(objective='score',parent_sha256='test'))
    return dict(id='unit',method='model',split='calibration',environment_seed=146900001,seat=0,
        student=str(path),student_sha256=digest(path),opponent=str(path),opponent_sha256=digest(path),
        native_artifact=artifact,seconds=.1,search_seed=146,deadline_unix=time.time()+300)


def test_full_market_rank_and_public_state_not_mutated():
    env=HarmoniesDuo();obs,_=env.reset(146);before=deepcopy(obs)
    ranks=Habitat().rank(obs)
    assert len(ranks)==len(obs['legal_actions'])
    assert {a['bundle_id'] for _,a in ranks if a['type']=='choose_bundle'}==set(range(5))
    assert {a['market_slot'] for _,a in ranks if a['type']=='take_card'}==set(range(5))
    assert obs==before
    root=mixed_roots(obs,[a for _,a in terrain_rank(personal_view(obs))],Habitat())
    assert len(root)==6 and all(a in obs['legal_actions'] for a in root)
    env.step(dict(type='choose_bundle',bundle_id=4))
    assert all('discard_card_slot' not in a for _,a in Habitat().rank(env.observe()))


def test_native_reference_policy_and_planner_parity(model,artifact):
    native=ModelPlayer(model,NativeEncoder.from_artifact(artifact));slow=SelfplaySolver(model)
    env=HarmoniesDuo();obs,_=env.reset(146);plans=[Planner(),Planner()]
    while not obs['terminated']:
        seat=obs['active_player'];a=native.decide(obs,plans[seat]);b=slow.decide(obs)
        assert a==b
        assert plans[seat].__dict__==slow.planners[seat].__dict__
        obs=env.step(a).observation


@pytest.mark.parametrize('seat',[0,1])
def test_full_game_and_returns_replayed(job,seat):
    stats,trace=play({**job,'seat':seat})
    assert stats['natural_terminal'] and stats['score']==trace['scores'][seat]
    assert audit_trace(trace)['actions']==len(trace['decisions'])
    broken=deepcopy(trace);broken['decisions'][0]['reward_to_next_own_decision']+=1
    with pytest.raises(ValueError,match='return'):audit_trace(broken)
    broken=deepcopy(trace);broken['decisions'][0]['observation']['first_player']=1
    with pytest.raises(ValueError,match='public'):audit_trace(broken)


@pytest.mark.parametrize('method',['habitat','search_habitat','search_model'])
def test_small_budget_no_truncation_or_illegal_action(job,method):
    stats,trace=play({**job,'method':method})
    assert stats['natural_terminal']
    for row in trace['decisions']:
        assert row['action'] in row['observation']['legal_actions']
        if row['search']:
            assert all(c['samples']==row['search']['completed_rounds'] for c in row['search']['candidates'])
            if row['search']['fallback']:assert all(c['mean'] is None for c in row['search']['candidates'])


def test_public_future_and_branch_planner_isolation(model,artifact):
    player=ModelPlayer(model,NativeEncoder.from_artifact(artifact));env=HarmoniesDuo();obs,_=env.reset(19)
    # Move to a legitimate late state so complete branch tests stay cheap.
    plans=[Planner(),Planner()]
    while obs['players'][0]['completed_turns']<7 and not obs['terminated']:
        obs=env.step(player.decide(obs,plans[obs['active_player']])).observation
    assert not obs['terminated']
    seat=obs['active_player'];rank=player.rank_batch([obs],[plans[seat]])[0][0]
    search=TerminalSearch(player,player,seat,'model',seconds=10,seed=12)
    saved=deepcopy(plans);public=deepcopy(obs)
    first=search.terminal_round(obs,rank[:2],41,plans,time.monotonic()+20)
    second=search.terminal_round(obs,rank[:2],41,plans,time.monotonic()+20)
    assert first==second and first[0] is not None
    assert obs==public and all(a.__dict__==b.__dict__ for a,b in zip(plans,saved))
    hidden=env.save_state();env._bag.reverse();env._deck.reverse()
    assert env.observe()==obs
    third=search.terminal_round(env.observe(),rank[:2],41,plans,time.monotonic()+20)
    assert first==third
    branches=[FastHarmoniesDuo.from_observation(obs,41) for _ in range(2)]
    assert branches[0].save_state()==branches[1].save_state()


def test_expired_round_has_no_q_and_budget_is_cumulative(model,artifact):
    p=ModelPlayer(model,NativeEncoder.from_artifact(artifact));env=HarmoniesDuo();obs,_=env.reset(31)
    s=TerminalSearch(p,p,0,'model',seconds=.01);plans=[Planner(),Planner()]
    assert s.terminal_round(obs,obs['legal_actions'][:2],4,plans,0)[0] is None
    a=s.decide(obs,plans);first=s.spent;obs=env.step(a).observation
    s.decide(obs,plans);assert s.spent>first
    assert s.last['completed_rounds']==0 and all(c['mean'] is None for c in s.last['candidates'])


def test_split_and_model_hash_guards(job):
    with pytest.raises(ValueError,match='split'):validate_job({**job,'split':'train','environment_seed':246000000})
    with pytest.raises(ValueError,match='expired'):validate_job({**job,'deadline_unix':0})
    with pytest.raises(ValueError,match='policy'):validate_job({**job,'student_sha256':'wrong'})


def test_paired_clusters_and_teacher_gate():
    rows=[]
    for seed in range(16):
        for seat in (0,1):
            for method,gain in [('model',0),('search_habitat',3),('search_model',-1)]:
                rows.append(dict(method=method,environment_seed=seed,seat=seat,search_seed=0,score=80+seed+gain,
                    animal=10+gain,terrain=70+seed,turns=11,animals_placed=3,decision_seconds=900,
                    completed_rounds=20,fallback_decisions=0))
    report=summarize(rows)
    assert report['search_habitat']['paired_score']['environments']==16
    assert pilot_choices(report)==['search_habitat'] and teacher_choice(report)=='search_habitat'
    assert teacher_choice({k:v for k,v in report.items() if k!='search_habitat'}) is None
    base=[r for r in rows if r['method']=='model']
    with pytest.raises(ValueError,match='duplicate'):paired(base+base,base)
    with pytest.raises(ValueError,match='unpaired'):paired(base[1:],base)


def test_actual_worker_completion_resume_and_shared_lock(job,tmp_path):
    import fcntl
    import os
    from pathlib import Path
    from boardbench.artifacts.store import read_json,write_json,source_id
    from boardbench.v122.campaign import physical_cpus
    from boardbench.v146.campaign import Runner,valid_result
    cpus=physical_cpus()
    if len(cpus)<4:pytest.skip('four physical CPUs required for half-node cap check')
    affinity=os.sched_getaffinity(0);root=tmp_path/'run';root.mkdir()
    for name in ('jobs','logs','episodes','source'):(root/name).mkdir()
    (root/'source/src').symlink_to(Path(__file__).resolve().parents[1]/'src',target_is_directory=True)
    c=dict(host=os.uname().nodename,deadline_unix=time.time()+200,repo=str(tmp_path),previous_run='test',
           job_timeout_seconds=5,seconds=.1,search_seed=146,
           **{k:job[k] for k in ('student','student_sha256','opponent','opponent_sha256','native_artifact')})
    write_json(root/'config.json',c)
    write_json(root/'launch.json',dict(source_id=source_id(),config_sha256=digest(root/'config.json'),cpus=cpus[:2]))
    runner=None
    try:
        runner=Runner(root)
        lock=(tmp_path/'outputs/v122'/(c['host']+'.cpu.lock')).open('a')
        with pytest.raises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock.close()
        jobs=runner.jobs(['model'],[246000010],'pilot')
        rows=runner.run(jobs,'pilot');assert len(rows)==2
        receipts=len(runner.receipts);again=runner.run(jobs,'pilot')
        assert again==rows and len(runner.receipts)==receipts
        with pytest.raises(ValueError,match='metadata'):valid_result({**rows[0],'seat':1-rows[0]['seat']},jobs[0])
        with pytest.raises(ValueError,match='over-budget'):valid_result({**rows[0],'budget_excess_seconds':1.},jobs[0])
        runner.close();runner=None
        assert read_json(root/'status.json')['active']==[]
        assert read_json(root/'cost.json')['gpu_seconds']==0
    finally:
        if runner:runner.close()
        os.sched_setaffinity(0,affinity)
