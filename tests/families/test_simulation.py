import pytest
from boardbench.families.registry import GAME_IDS,game_class
from boardbench.families.play import make_policy
from boardbench.families.simulation import clone,step_legal,terminated

@pytest.mark.parametrize('game',[g for g in GAME_IDS if g!='five_tribes'])
def test_fast_simulation_matches_public_transition_and_random_stream(game):
    env=game_class(game)();env.reset(172391,first_player=1)
    fast=clone(env)
    spec={'kind':'heuristic'}
    if game=='targi':spec['decision_guard']='targi_trader_acyclic_v1'
    if game=='century_a_new_world':spec['decision_guard']='century_new_world_public_repeat_v1'
    policies=[make_policy(type(env),spec,p) for p in (0,1)]
    for i in range(500):
        assert env.snapshot()==fast.snapshot()
        assert env.outcome()==fast.outcome()
        assert env.score_breakdown()==fast.score_breakdown()
        if terminated(env):break
        obs=env.observe();actions=env.legal_actions()
        assert obs==fast.observe() and actions==fast.legal_actions()
        if not actions:break
        selected=actions.index(policies[env.current_player].decide(obs,actions))
        env.step(actions[selected]);step_legal(fast,actions,selected)
        fast.validate()
    assert env.snapshot()==fast.snapshot()
