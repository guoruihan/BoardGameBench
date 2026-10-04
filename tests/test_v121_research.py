from copy import deepcopy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from boardbench.artifacts.store import digest, read_json, source_id
from boardbench.environments.harmonies import Harmonies, CELLS, CELL_INDEX, DELTAS
from boardbench.training import parameter_hash
from boardbench.v12.encoding import encode, action_map, decode, index, STOP
from boardbench.v12.neural import CompactSolver
from boardbench.v121.registry import definitions, make_splits, validate_manifest, training_config, identity
from boardbench.v121.neural import ResearchSolver, raw_action_ids
from boardbench.v121.training import Trainer, run
from boardbench.v121.evaluation import seeds_for


def config(name='E00'):
    value = training_config(name, 911, make_splits(train_count=8))
    value.update(batch_episodes=2, epochs=1, minibatch_size=64, purpose='interface_check')
    return value


def solver(name='E00', seed=911):
    return ResearchSolver(seed, **config(name)['network'])


def test_registry_planned_hashes_and_fresh_disjoint_splits():
    a, b = make_splits(train_count=8), make_splits(train_count=8)
    assert a == b
    assert all(x['status'] == 'planned' for x in definitions().values())
    sets = a['sets']
    assert [len(sets[x]) for x in ['development_health', 'development', 'confirmation', 'test']] == [32,128,256,512]
    assert sets['development_health'] == sets['development'][:32]
    bad = deepcopy(a); bad['sets']['test'][0] = sets['train'][0]
    bad['sha256'] = identity({k:v for k,v in bad.items() if k!='sha256'})
    with pytest.raises(ValueError, match='overlapping'):
        validate_manifest(bad)
    with pytest.raises(ValueError, match='not available'):
        training_config('Q0', 911, a)


def test_e00_exact_original_initialization_encoding_and_predictions():
    old, new = CompactSolver(911), solver()
    assert parameter_hash(old.model) == parameter_hash(new.model)
    env = Harmonies(); obs, _ = env.reset(3)
    assert old.encode(obs) == new.encode(obs) == encode(obs)
    x = torch.tensor([old.encode(obs)['features']]); mask = torch.tensor([old.encode(obs)['action_mask']])
    for a,b in zip(old.model(x,mask),new.model(x,mask)):
        assert torch.equal(a,b)
    assert old.decide(obs,None) == new.decide(obs,None)


@pytest.mark.parametrize('name', ['E00','E01','E10','E11'])
def test_shapes_finite_gradients_hidden_boundary_and_weight_roundtrip(name,tmp_path):
    torch.set_num_threads(1)
    policy = solver(name)
    env = Harmonies(); obs,_=env.reset(4)
    data=policy.encode(obs)
    env._bag.reverse();env._deck.reverse()
    assert policy.encode(env.observe()) == data
    x=torch.tensor([data['features']]);mask=torch.tensor([data['action_mask']])
    logits,values=policy.model(x,mask)
    assert logits.shape==(1,241) and values.shape==(1,)
    assert (logits[~mask]==-1e9).all()
    assert policy.decide(obs,None) in obs['legal_actions']+[STOP]
    loss=values.square().mean()-torch.distributions.Categorical(logits=logits.double()).entropy().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in policy.model.parameters())
    path=tmp_path/'weights.pt';policy.save_weights(path)
    restored=solver(name,912);restored.load_weights(path)
    assert parameter_hash(policy.model)==parameter_hash(restored.model)
    incompatible=solver('E11' if name!='E11' else 'E10')
    with pytest.raises(ValueError,match='metadata'):
        incompatible.load_weights(path)
    with pytest.raises(ValueError):
        policy.model(x,torch.zeros_like(mask))


def test_directed_geometry_and_shared_candidate_identifiers():
    policy=solver('E11');encoder=policy.model.body
    for cell,(q,r) in enumerate(CELLS):
        for direction,(dq,dr) in enumerate(DELTAS):
            target=CELL_INDEX.get((q+dq,r+dr))
            assert bool(encoder.neighbor_valid[cell,direction]) == (target is not None)
            if target is not None:
                assert encoder.neighbor_indices[cell,direction].item()==target
    ids=raw_action_ids()
    assert len(set(map(tuple,ids.tolist())))==241
    assert ids[3].tolist()==[1,0,4,0]
    assert ids[144].tolist()==[3,6,0,0]
    assert ids[-1].tolist()==[5,6,4,23]
    actor=solver('E01').model.actor
    scores=actor(torch.zeros(1,128))
    assert scores[0,3]!=scores[0,4]  # identical empty targets remain distinguishable.


def test_slot_reordering_and_completion_keep_semantic_actions():
    path=Path(__file__).parents[1]/'scripts/animal_browser_fixture.py'
    spec=importlib.util.spec_from_file_location('v121_fixture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    trace=module.fixture();env=Harmonies();env.load_state(trace['snapshot'])
    for semantic in trace['actions']:
        obs=env.observe()
        mapped=action_map(obs,'slots240_v1',True)
        assert len(mapped)==len(obs['legal_actions'])+1
        for model in ('E00','E01','E10','E11'):
            assert solver(model).encode(obs)['action_mask']==encode(obs)['action_mask']
        restored=decode(obs,index(obs,semantic,'slots240_v1',True),'slots240_v1',True)
        assert restored==semantic
        env.step(restored)
    assert env.observe()==trace['final_observation']


@pytest.mark.parametrize('name',['E00','E01','E10','E11'])
def test_real_update_and_exact_cpu_resume(name,tmp_path):
    a=Trainer(config(name));before=parameter_hash(a.solver.model)
    row=a.ppo_batch()
    assert row['initial_ratio_max_error']<=1e-4
    assert parameter_hash(a.solver.model)!=before
    assert row['policy_decisions']==row['real_atomic_actions']+row['stop_decisions']
    path=tmp_path/'state.pt';a.save(path)
    a.ppo_batch();expected=parameter_hash(a.solver.model)
    b=Trainer(config(name));b.load(path);b.ppo_batch()
    assert parameter_hash(b.solver.model)==expected
    assert (a.steps,a.decision_count,a.stop_decisions,a.updates)==(b.steps,b.decision_count,b.stop_decisions,b.updates)
    for key, state in a.optimizer.state_dict()['state'].items():
        for field,value in state.items():
            assert torch.equal(value,b.optimizer.state_dict()['state'][key][field])
    wrong=config(name);wrong['learning_rate']=.001
    with pytest.raises(ValueError,match='config/source'):
        Trainer(wrong).load(path)


def test_stop_is_terminal_not_a_real_atomic_action():
    trainer=Trainer(config())
    with torch.no_grad():
        trainer.solver.model.actor.weight.zero_()
        trainer.solver.model.actor.bias.fill_(-100)
        trainer.solver.model.actor.bias[-1]=100
    result=trainer.ppo_batch()
    assert trainer.steps==0 and trainer.decision_count==trainer.stop_decisions==2
    assert result['mean_training_score']==0 and result['natural_fraction']==0


def test_step_target_and_explicit_overshoot(tmp_path):
    result=run(config(),tmp_path/'run',1,20.)
    assert result['status']=='target_reached' and result['real_atomic_actions']>=1
    assert result['target_overshoot_actions']==result['real_atomic_actions']-1
    restored=Trainer(config());restored.load(tmp_path/'run/final.resume.pt')
    assert restored.steps==result['real_atomic_actions']


def test_final_test_requires_exact_frozen_policy():
    manifest=make_splits(train_count=8);spec={'id':'x','weights_sha256':'a'}
    assert len(seeds_for(manifest,'development',spec))==128
    with pytest.raises(ValueError,match='frozen'):
        seeds_for(manifest,'test',spec)
    frozen={'split_sha256':manifest['sha256'],'source_id':source_id(),'selection_split':'confirmation',
            'selection_rule':'predeclared highest confirmation mean; all seeds reported',
            'policy_hashes':{'x':identity(spec)}}
    assert len(seeds_for(manifest,'test',spec,frozen))==512
    with pytest.raises(ValueError,match='not in frozen'):
        seeds_for(manifest,'test',{**spec,'weights_sha256':'b'},frozen)


def test_terminal_regret_does_not_label_nonterminal_animal_opportunities():
    from boardbench.v121.diagnostics import terminal_animal_gains
    from boardbench.environments.harmonies import PATTERNS
    env=Harmonies();env.reset(3)
    card=5
    anchor,pattern=next((i,orientations[0]) for i,orientations in enumerate(PATTERNS[card]) if 0 in orientations)
    for cell,allowed in pattern:
        env.s['board'][cell]['stack_id']=allowed[0]
    env.s['active_cards']=[{'card_id':card,'instance_id':card,'placed_count':0}]
    before=deepcopy(env.save_state())
    opportunities=terminal_animal_gains(env,STOP)
    assert opportunities and max(x['total_score_gain'] for x in opportunities)>0
    assert env.save_state()==before
    ordinary=next(a for a in env.observe()['legal_actions'] if a['type']=='choose_bundle')
    assert terminal_animal_gains(env,ordinary)==[]
