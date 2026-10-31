import numpy as np
import pytest
import torch

from boardbench.families.registry import GAME_IDS,game_class
from boardbench.families.play import make_policy
from boardbench.families.rich_observation import PublicTreeEncoder
from boardbench.families.rich_neural import ContextActorCritic,batch_inputs

GAMES=[g for g in GAME_IDS if g!='five_tribes']


@pytest.mark.parametrize('game',GAMES)
def test_all_families_reference_and_cached_tree_equivalence(game):
    env=game_class(game)();env.reset(172032,first_player=1)
    fast,reference=PublicTreeEncoder(game),PublicTreeEncoder(game,cache=False)
    spec={'kind':'heuristic'}
    if game=='targi':spec['decision_guard']='targi_trader_acyclic_v1'
    if game=='century_a_new_world':spec['decision_guard']='century_new_world_public_repeat_v1'
    policies=[make_policy(type(env),spec,p) for p in (0,1)]
    for i in range(180):
        if env.outcome()['terminated'] or not env.legal_actions():break
        obs=env.observe();actions=env.legal_actions()
        if i%13==0:
            before=env.snapshot()
            np.testing.assert_array_equal(fast.encode(obs),reference.encode(obs))
            for a in actions[:12]:
                np.testing.assert_array_equal(fast.encode_action(a),reference.encode_action(a))
            assert env.snapshot()==before
        env.step(policies[env.current_player].decide(obs,actions))


def test_harmonies_spatial_and_card_information_survives():
    env=game_class('harmonies')();obs=env.reset(817230001)
    encoder=PublicTreeEncoder('harmonies')
    for _ in range(6):
        actions=env.legal_actions()
        chosen=next((a for a in actions if a['type']=='take_card'),None)
        if chosen is None:chosen=next(a for a in actions if a['type']=='choose_bundle')
        env.step(chosen);obs=env.observe()
        if obs['players'][0]['active_cards'] and any(a['type']=='place_token' for a in env.legal_actions()):break
    actions=[a for a in env.legal_actions() if a['type']=='place_token']
    old=[tuple(env.features(obs,a)) for a in actions]
    pair=next((i,j) for i in range(len(actions)) for j in range(i+1,len(actions)) if old[i]==old[j])
    assert not np.array_equal(encoder.encode_action(actions[pair[0]]),encoder.encode_action(actions[pair[1]]))
    initial=encoder.encode(obs)
    obs['players'][0]['active_cards'][0]['card_id']=1
    assert not np.array_equal(initial,encoder.encode(obs))


def test_context_actor_gradient_candidate_reordering_and_ragged_batch():
    torch.set_num_threads(1);torch.manual_seed(1702)
    env=game_class('splendor')();obs=env.reset(17201);enc=PublicTreeEncoder('splendor')
    actions=env.legal_actions()[:4];a,owners=enc.encode_actions(actions)
    row=dict(state=enc.encode(obs),actions=a,owners=owners)
    model=ContextActorCritic(enc.metadata['vocabulary_size'],hidden=24)
    logits,values=model(*batch_inputs([row]))
    perm=[3,0,2,1];a2,o2=enc.encode_actions([actions[i] for i in perm])
    l2,v2=model(*batch_inputs([dict(row,actions=a2,owners=o2)]))
    torch.testing.assert_close(logits[perm],l2);torch.testing.assert_close(values,v2)
    # A change of visible context can change action preferences, not just add
    # an action-independent scalar that cancels in softmax.
    obs['tokens'][0]=[7,0,1,0,2,1]
    changed=dict(row,state=enc.encode(obs))
    l3,_=model(*batch_inputs([changed]))
    assert not torch.allclose(torch.softmax(logits,0),torch.softmax(l3,0),atol=1e-8,rtol=0)
    batch,bs=model(*batch_inputs([row,changed]))
    torch.testing.assert_close(batch[:4],logits)
    loss=-torch.log_softmax(logits,0)[0]+values.square().sum()
    loss.backward()
    assert model.state.node[0].weight.grad.abs().sum()>0
    assert model.action.node[0].weight.grad.abs().sum()>0


def test_reject_privileged_fields_and_preserve_unregistered_text():
    env=game_class('dominion')();obs=env.reset(17);enc=PublicTreeEncoder('dominion')
    with pytest.raises(ValueError):enc.encode(dict(obs,environment_seed=17))
    x=enc.encode_action({'type':'unseen-symbol-A'})
    y=enc.encode_action({'type':'unseen-symbol-B'})
    assert not np.array_equal(x,y)


def test_new_model_roundtrip_explicit_schema_and_numeric_types(tmp_path):
    from boardbench.families.rich_neural import save_model,load_model,RichPolicy
    env=game_class('splendor')();obs=env.reset(1731);enc=PublicTreeEncoder('splendor')
    assert not np.array_equal(enc.encode_action({'type':'take_same','color':1}),enc.encode_action({'type':'take_same','color':1.0}))
    model=ContextActorCritic(enc.metadata['vocabulary_size'],hidden=24)
    path=tmp_path/'rich.pt';save_model(path,model,enc)
    restored,encoding,_=load_model(path,'splendor')
    assert RichPolicy(model,enc).decide(obs,env.legal_actions())==RichPolicy(restored,encoding).decide(obs,env.legal_actions())
    with pytest.raises(ValueError):load_model(path,'dominion')
