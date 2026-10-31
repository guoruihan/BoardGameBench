"""Optional public-only New World guard: behavior, portability and old guards."""
from copy import deepcopy
import json

import pytest

from boardbench.families import bundle, evaluate
from boardbench.families.century_new_world_policy import (
    CenturyNewWorldRepeatPolicy, GUARD_ID, public_turn_fingerprint)
from boardbench.families.games.century_a_new_world import CenturyANewWorld as Game
from boardbench.families.games.targi import Targi
from boardbench.families.play import FeaturePolicy, make_policy, write_json
from boardbench.families.targi_policy import TargiTraderAcyclicPolicy


class FirstPolicy:
    def __init__(self):
        self.last_candidates = None

    def decide(self, observation, actions):
        self.last_candidates = actions
        return actions[0]


def fixture():
    env = Game(); env.reset(33000036)
    return env, env.observe(), env.legal_actions()


def test_first_visit_preserves_input_identity_and_randomness():
    env, obs, actions = fixture()
    direct = FeaturePolicy(Game, {'kind': 'random'}, 10)
    guarded = CenturyNewWorldRepeatPolicy(FeaturePolicy(Game, {'kind': 'random'}, 10))
    before = deepcopy((obs, actions, env.snapshot()))
    assert guarded.decide(obs, actions) == direct.decide(obs, actions)
    assert guarded.policy.rng.getstate() == direct.rng.getstate()
    assert (obs, actions, env.snapshot()) == before
    spy = FirstPolicy(); first = CenturyNewWorldRepeatPolicy(spy)
    assert first.decide(obs, actions) == actions[0]
    assert spy.last_candidates is actions
    assert first.restricted_decisions == 0


def test_repeated_state_uses_least_tried_legal_candidates_in_original_order():
    _, obs, actions = fixture(); spy = FirstPolicy(); guard = CenturyNewWorldRepeatPolicy(spy)
    for action in actions:
        assert guard.decide(obs, actions) == action
        assert action in spy.last_candidates
    # Every move tried once: all moves are available again, not a forced end.
    assert guard.decide(obs, actions) == actions[0]
    assert spy.last_candidates is actions
    counts = guard.action_counts[public_turn_fingerprint(obs)]
    assert sorted(counts.values()) == [1]*(len(actions)-1)+[2]


def test_fingerprint_ignores_only_history_and_pure_clocks():
    _, obs, _ = fixture(); changed = deepcopy(obs)
    changed['decision_index'] += 100
    changed['history'].append({'player': 0, 'action': {'type': 'rest'}})
    for player in changed['players']: player['turns'] += 50
    assert public_turn_fingerprint(obs) == public_turn_fingerprint(changed)
    changed['final_round'] = True
    assert public_turn_fingerprint(obs) != public_turn_fingerprint(changed)
    changed = deepcopy(obs); changed['players'][0]['resources'][0] += 1
    assert public_turn_fingerprint(obs) != public_turn_fingerprint(changed)
    changed = deepcopy(obs); changed['market'].reverse()
    assert public_turn_fingerprint(obs) != public_turn_fingerprint(changed)
    changed = deepcopy(obs); changed['current_player'] = 1
    assert public_turn_fingerprint(obs) != public_turn_fingerprint(changed)


@pytest.mark.parametrize('phase', ['discard', 'explore'])
def test_other_phases_forward_unchanged_and_do_not_track(phase):
    _, obs, actions = fixture(); obs['phase'] = phase
    spy = FirstPolicy(); guard = CenturyNewWorldRepeatPolicy(spy)
    for _ in range(3):
        assert guard.decide(obs, actions) == actions[0]
        assert spy.last_candidates is actions
    assert not guard.action_counts and guard.restricted_decisions == 0


def test_guard_uses_public_view_and_not_hidden_deck_or_stack_order():
    env, obs, _ = fixture()
    env.state['deck'][1:] = reversed(env.state['deck'][1:])
    for stack in env.state['bonus_stacks']:
        stack[1:] = reversed(stack[1:])
    env.state['removed_bonuses'].reverse()
    env.validate()
    assert env.observe() == obs
    assert public_turn_fingerprint(env.observe()) == public_turn_fingerprint(obs)


def test_json_view_replay_and_fresh_episode_reset():
    _, obs, actions = fixture(); guard = CenturyNewWorldRepeatPolicy(FirstPolicy())
    guard.decide(obs, actions)
    restored_obs = json.loads(json.dumps(obs))
    assert guard.decide(restored_obs, json.loads(json.dumps(actions))) == actions[1]
    guard.reset()
    assert guard.decide(obs, actions) == actions[0]
    assert guard.restricted_decisions == 0


def test_choice_is_delegated_to_scorer_inside_subset():
    class LastPolicy(FirstPolicy):
        def decide(self, observation, actions):
            self.last_candidates = actions
            return actions[-1]
    _, obs, actions = fixture(); spy = LastPolicy(); guard = CenturyNewWorldRepeatPolicy(spy)
    assert guard.decide(obs, actions) == actions[-1]
    assert guard.decide(obs, actions) == actions[-2]
    assert spy.last_candidates == actions[:-1]


def test_bad_policy_action_and_wrong_game_are_rejected():
    class BadPolicy:
        def decide(self, observation, actions): return {'type': 'not_legal'}
    _, obs, actions = fixture()
    with pytest.raises(ValueError, match='outside'):
        CenturyNewWorldRepeatPolicy(BadPolicy()).decide(obs, actions)
    guard = CenturyNewWorldRepeatPolicy(FirstPolicy())
    bad = dict(obs, game_id='targi')
    with pytest.raises(ValueError, match='New World'):
        guard.decide(bad, actions)
    with pytest.raises(ValueError, match='legal actions'):
        guard.decide(obs, [])


@pytest.mark.parametrize('cls,guard', [
    (Game, 'unknown'), (Game, 'targi_trader_acyclic_v1'),
    (Targi, GUARD_ID), (Game, ''),
])
def test_factory_and_bundle_reject_unknown_or_wrong_game(cls, guard):
    spec = {'kind': 'heuristic', 'game': cls.game_id, 'decision_guard': guard}
    with pytest.raises(ValueError, match='guard'): make_policy(cls, spec)
    with pytest.raises(ValueError, match='decision_guard'): bundle._check_guard(spec)


def test_optional_factory_keeps_old_artifacts_and_targi_behavior():
    assert isinstance(make_policy(Game, {'kind': 'heuristic'}), FeaturePolicy)
    spec = {'kind': 'heuristic', 'game': Game.game_id, 'decision_guard': GUARD_ID}
    bundle._check_guard(spec)
    assert isinstance(make_policy(Game, spec), CenturyNewWorldRepeatPolicy)
    spec = {'kind': 'heuristic', 'game': Targi.game_id, 'decision_guard': 'targi_trader_acyclic_v1'}
    bundle._check_guard(spec)
    assert isinstance(make_policy(Targi, spec), TargiTraderAcyclicPolicy)


@pytest.mark.parametrize('kind', ['heuristic', 'neural'])
def test_frozen_evaluation_and_bundle_preserve_new_guard(tmp_path, kind):
    spec = {'kind': kind}
    if kind == 'neural':
        from boardbench.families.neural import CandidateActorCritic, save_model
        path = tmp_path/'model.pt'
        save_model(path, CandidateActorCritic(hidden=8), Game, Game.default_weights)
        spec['path'] = str(path)
    bots = tmp_path/'bots.json'
    write_json(bots, {'plain': spec, 'guarded': dict(spec, decision_guard=GUARD_ID)})
    evaluate.prepare_evaluation(Game.game_id, bots, tmp_path/'dev', [31], 'dev')
    plan = tmp_path/'plan.json'
    write_json(plan, {'games': {Game.game_id: {
        'bots': str(bots), 'dev_manifest': str(tmp_path/'dev/manifest.json'),
        'status': 'selected', 'recommended': ['guarded']}}})
    manifest = bundle.export_bundle(plan, tmp_path/'pool')
    info = manifest['games'][Game.game_id]['bots']['guarded']
    assert evaluate.read_json(tmp_path/'pool'/info['path'])['decision_guard'] == GUARD_ID
    assert bundle.load_bundle(tmp_path/'pool') == manifest
    policy = bundle.load_bot(tmp_path/'pool', Game.game_id, 'guarded', seed=2)
    assert isinstance(policy, CenturyNewWorldRepeatPolicy)
    _, obs, actions = fixture()
    plain = bundle.load_bot(tmp_path/'pool', Game.game_id, 'plain', seed=2)
    assert policy.decide(obs, actions) == plain.decide(obs, actions)
