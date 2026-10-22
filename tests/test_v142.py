from copy import deepcopy
import numpy as np
import pytest
import torch
from boardbench.artifacts.store import digest
from boardbench.environments.harmonies_market import HarmoniesMarket
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner
from boardbench.v140.training import Trainer as DuoTrainer
from boardbench.v140.policy import export_policy as export_duo
from boardbench.v142.encoding import ACTIONS, BASE, FEATURES, LAYOUT, action_map, encode_batch
from boardbench.v142.collector import EnvironmentPool
from boardbench.v142.training import Trainer, replay, replay_steps, paired_delta, run
from boardbench.v142.policy import export_policy,load_policy,MarketSolver
from test_v129 import artifact
from test_v140 import duo_config


@pytest.fixture
def config(duo_config,tmp_path):
    t=DuoTrainer({**duo_config,'objective':'score'});path=tmp_path/'duo.pt';export_duo(t.model,path);t.close()
    return {**duo_config,'layout':LAYOUT,'rules_version':HarmoniesMarket.rules_version,'objective':'score',
        'duo_parent':str(path),'duo_parent_sha256':digest(path),'initialization':'solo',
        'development_seeds':[242000001,242000002],'confirmation_seeds':[342000001,342000002],
        'finish_reserve_seconds':60}


def test_native_and_no_opponent(artifact):
    obs,_=HarmoniesMarket().reset(44);plan=Planner().vector(obs)
    slow=encode_batch([obs],[plan]);fast=encode_batch([obs],[plan],NativeEncoder.from_artifact(artifact))
    assert slow.shape==(1,FEATURES);np.testing.assert_array_equal(slow,fast)
    assert not slow[:,2*BASE:3*BASE].any() and not slow[:,-8:][:,[3,5,7]].any()
    mapping=action_map(obs)
    assert set(mapping)=={0,1,2,240,241,141,142,143,242,243}
    altered=deepcopy(obs);altered['animal_market'][4]=31 if obs['animal_market'][4]!=31 else 32
    assert not np.array_equal(slow,encode_batch([altered],[plan]))
    altered=deepcopy(obs);altered['private_seed']=876
    np.testing.assert_array_equal(slow,encode_batch([altered],[plan]))


@pytest.mark.parametrize('workers',[0,2])
def test_native_pool_reference(artifact,workers):
    seeds=[94,95];envs=[HarmoniesMarket(),HarmoniesMarket()];obs=[e.reset(s)[0] for e,s in zip(envs,seeds)]
    planners=[Planner(),Planner()];rng=np.random.default_rng(142)
    with EnvironmentPool(2,workers,native_artifact=artifact) as pool:
        arrays=pool.reset(seeds)
        while not arrays['done'].all():
            slots=np.flatnonzero(~arrays['done']);actions=[]
            for i in slots:
                mapping=action_map(obs[i]);np.testing.assert_array_equal(arrays['features'][i],encode_batch([obs[i]],[planners[i].vector(obs[i])])[0])
                assert set(np.flatnonzero(arrays['masks'][i]))==set(mapping)
                a=int(rng.choice(list(mapping)));actions.append(a);obs[i]=envs[i].step(mapping[a]).observation
            pool.step(slots,actions)
        for i,o in enumerate(obs): assert arrays['stats'][i,0]==o['score_breakdown']['total']


@pytest.mark.parametrize('initialization',['solo','duo'])
def test_update_resume_replay(config,tmp_path,initialization):
    config={**config,'initialization':initialization};t=Trainer(config);first=t.batch()
    assert first['actor_change']>0 and first['context_change']>0 and first['initial_ratio_max_error']<1e-5
    for trace in t.last_traces:
        replay(trace)
        transitions=list(replay_steps(trace))
        assert len(transitions)==len(trace['decisions']) and transitions[-1]['terminated']
        assert all(not r['terminated'] for r in transitions[:-1])
    path=tmp_path/'resume.pt';t.save(path);expected=t.batch();weights=deepcopy(t.model.state_dict());t.close()
    restored=Trainer(config);restored.load(path);actual=restored.batch()
    assert actual['per_game']==expected['per_game']
    assert all(torch.equal(v,restored.model.state_dict()[k]) for k,v in weights.items())
    rng=restored.rng.get_state().clone();metrics=restored.evaluate(config['development_seeds'])
    assert metrics['games']==2 and torch.equal(rng,restored.rng.get_state())
    assert paired_delta(metrics,metrics)['score_delta']==0
    policy=tmp_path/'policy.pt';export_policy(restored.model,policy,config);loaded=load_policy(policy)
    obs,_=HarmoniesMarket().reset(16)
    assert MarketSolver(loaded).decide(obs)==MarketSolver(restored.model).decide(obs)
    restored.close()


def test_guard_config_and_artifacts(config):
    for change in ({'layout':'wrong'},{'objective':'margin'},{'development_seeds':[200000001]},
                   {'confirmation_seeds':[242000001]},{'duo_parent_sha256':'wrong'},{'source_id':'wrong'}):
        with pytest.raises(ValueError): Trainer({**config,**change})


def test_initial_is_eligible_and_final_confirmation(config,tmp_path):
    import json
    # No training allowance: frozen initial model must remain a valid selected
    # checkpoint. In particular, a margin-based selection would fail this test.
    out=tmp_path/'complete'
    run({**config,'max_seconds':300,'finish_reserve_seconds':9999},out,'cpu')
    done=json.loads((out/'done.json').read_text())
    assert done['selected_batch']==0 and done['paired_vs_initial']['score_delta']==0
    assert done['confirmation']['games']==2
    assert (out/'best_policy.pt').exists() and (out/'final_policy.pt').exists()
    with pytest.raises(ValueError): run(config,out,'cpu',resume=True)
