"""Public policy-level loop prevention; original game rules remain unchanged."""
from copy import deepcopy
import json

import pytest

from boardbench.families.games.targi import Targi
from boardbench.families.targi_policy import trader_acyclic_actions, TargiTraderAcyclicPolicy


# Actual generation-zero CEM candidate 7, preserved as a behavioral regression.
SLOW_WEIGHTS = [
    1.1637646586803827, -.01005291940031674, -1.136993422175991,
    -.6117692607173935, .6223909789108188, .0015781479571845,
    -.9512907781029556, .35990695142186485, -.12221428064082654,
    -1.2203442996539504, .13458356490278361, .4262337124388514,
    .29634448869841357, .9703599433924848, .8436531502859316,
    .13859828980815073, 1.2763335054982334, 1.446234027448758,
    .49566721505065314, -.30825893053307957, .5494625927498344,
    .08157060633371513, -.4772068888098694, -1.0776663149011563,
]


class Scorer:
    def __init__(self, weights):
        self.weights = weights

    def decide(self, observation, actions):
        return max(actions, key=lambda a: sum(
            x*w for x,w in zip(Targi.features(observation,a),self.weights)))


def trader_game(resources=(2,0,1,2)):
    g=Targi();g.reset(10100000)
    # Move the actual discount card from its original zone, conserving cards.
    card=39
    if card in g.state['tribe_deck']:
        g.state['tribe_deck'].remove(card)
    else:
        cell=next(c for c in g.state['center'] if c['kind']=='tribe' and c['card']==card)
        cell['card']=g.state['tribe_deck'].pop()
    g.state['players'][0]['rows'][0].append(card)
    g.state['players'][0]['resources']=list(resources)
    g.state.update(phase='trader',round=11)
    return g


def trade(give,take,rate=1):
    return dict(type='trade',give=give,take=take,rate=rate)


def test_original_legal_loop_still_permitted_and_rng_unchanged():
    g=trader_game();before=g.snapshot()
    g.step(trade(0,1));g.step(trade(1,0));g.validate()
    after=g.snapshot()
    assert after['rng']==before['rng']
    for snapshot in (before,after):
        snapshot['state'].pop('history');snapshot['state'].pop('decision_index')
    assert after==before


def test_guard_removes_return_but_keeps_other_trades_and_done():
    g=trader_game();g.step(trade(0,1))
    observation=g.observe();actions=g.legal_actions()
    before=deepcopy((observation,actions,g.snapshot()))
    filtered=trader_acyclic_actions(observation,actions)
    assert trade(1,0) in actions and trade(1,0) not in filtered
    assert trade(2,0) in filtered and dict(type='done') in filtered
    assert (observation,actions,g.snapshot())==before


def test_three_step_cycle_and_snapshot_public_history_reconstruction():
    g=trader_game();g.step(trade(0,1));g.step(trade(1,2))
    assert trade(2,0) not in trader_acyclic_actions(g.observe(),g.legal_actions())
    restored=Targi();restored.restore(json.loads(json.dumps(g.snapshot())))
    assert trader_acyclic_actions(restored.observe(),restored.legal_actions()) == trader_acyclic_actions(g.observe(),g.legal_actions())


def test_history_scope_stops_at_new_trader_activation():
    g=trader_game();g.state['history']=[
        dict(player=0,round=11,action=trade(1,0)),
        dict(player=0,round=11,action=dict(type='use',zone='border',position=5)),
    ]
    assert trade(0,1) in trader_acyclic_actions(g.observe(),g.legal_actions())
    g.state['history']=[dict(player=1,round=11,action=trade(1,0))]
    assert trade(0,1) in trader_acyclic_actions(g.observe(),g.legal_actions())
    g.state['history']=[dict(player=0,round=10,action=trade(1,0))]
    assert trade(0,1) in trader_acyclic_actions(g.observe(),g.legal_actions())


def test_normal_two_to_one_and_gold_trades_retained():
    g=trader_game((5,2,1,0))
    # A prior gold trade lowers ordinary resources; reverse cannot be available.
    g.step(trade(0,3,3))
    assert trader_acyclic_actions(g.observe(),g.legal_actions())==g.legal_actions()
    g.state['players'][0]['rows'][0].remove(39);g.state['tribe_discard'].append(39)
    assert trader_acyclic_actions(g.observe(),g.legal_actions())==g.legal_actions()


def test_guard_has_no_effect_outside_trader_or_on_randomness():
    g=Targi();g.reset(7)
    before=g.snapshot();actions=g.legal_actions()
    assert trader_acyclic_actions(g.observe(),actions)==actions
    assert g.snapshot()==before
    bad=g.observe();bad['game_id']='jaipur'
    with pytest.raises(ValueError):trader_acyclic_actions(bad,actions)


def test_slow_candidate_exits_finite_trade_state_space():
    g=trader_game();bot=TargiTraderAcyclicPolicy(Scorer(SLOW_WEIGHTS));visited=set()
    for _ in range(40):
        if g.state['phase']!='trader':break
        resources=tuple(g.state['players'][0]['resources'])
        assert resources not in visited
        visited.add(resources)
        g.step(bot.decide(g.observe(),g.legal_actions()));g.validate()
    assert g.state['phase']=='actions'


def test_guard_is_only_action_selection_not_a_new_ending():
    g=trader_game();bot=TargiTraderAcyclicPolicy(Scorer(SLOW_WEIGHTS))
    for _ in range(40):
        if g.state['phase']!='trader':break
        g.step(bot.decide(g.observe(),g.legal_actions()))
    assert not g.state['terminated'] and not g.state['winners']


def test_factory_and_training_artifact_preserve_explicit_guard():
    from boardbench.families.play import FeaturePolicy, make_policy
    from boardbench.families.train import artifact
    from boardbench.families.games.jaipur import Jaipur
    from boardbench.families.targi_policy import GUARD_ID
    plain = artifact(Targi, SLOW_WEIGHTS, 0, 'test')
    guarded = artifact(Targi, SLOW_WEIGHTS, 0, 'test', GUARD_ID)
    assert 'decision_guard' not in plain
    assert guarded['decision_guard'] == GUARD_ID
    assert isinstance(make_policy(Targi, plain), FeaturePolicy)
    assert isinstance(make_policy(Targi, guarded), TargiTraderAcyclicPolicy)
    for cls, spec in ((Jaipur, guarded), (Targi, dict(kind='heuristic', decision_guard='unknown'))):
        with pytest.raises(ValueError, match='guard'):
            make_policy(cls, spec)
