from copy import deepcopy
import random
import shutil

import pytest

from boardbench.contracts import InvalidAction
from boardbench.environments.harmonies import Harmonies as Solo, CARDS, PATTERNS
from boardbench.environments.harmonies_duo import FastHarmoniesDuo
from boardbench.families.games.harmonies import Harmonies, copy_public
from boardbench.families import harmonies_kernel as kernel
from boardbench.solvers.board import habitat_potential
from boardbench.v146.policy import personal_view, terrain_rank


@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    if not shutil.which('g++'):pytest.skip('explicit C++ compiler unavailable')
    return kernel.build(tmp_path_factory.mktemp('kernel')/'build')


@pytest.fixture(params=['python', 'native'])
def backend(request):
    native = request.getfixturevalue('artifact') if request.param == 'native' else None
    kernel.configure(native)
    yield request.param
    kernel.configure()


def old_features(obs, actions):
    view = personal_view(obs);view['legal_actions'] = actions
    ranked = {tuple(sorted(a.items())): value for value, a in terrain_rank(view)}
    result = []
    for action in actions:
        row = [ranked[tuple(sorted(action.items()))]] + [float(action['type'] == k) for k in kernel.KINDS]
        row += [len(view['active_cards']), len(view['completed_cards']), view['turn_index']]
        result.append(row+[0.]*(24-len(row)))
    return result


def test_terrain_on_random_boards_matches_reference(backend):
    rng = random.Random(1703)
    for _ in range(600):
        # Include dense rivers, singletons, cycles, disconnected fields/buildings.
        palette = rng.choice([list(range(15)), [0, 1], [1, 2, 3, 12, 13, 14]])
        player = dict(board=[dict(stack_id=rng.choice(palette), animal=None) for _ in range(23)],
                      active_cards=[dict(card_id=c, placed_count=rng.randrange(CARDS[c]['cube_count']+1))
                                    for c in (1, 7, 13, 32)], completed_cards=[])
        reference = Solo.__new__(Solo);reference.s = player
        assert kernel.board_score(player) == reference.score_breakdown()


def test_all_card_patterns_and_occupied_anchors(backend):
    rng = random.Random(109)
    for card, anchors in PATTERNS.items():
        for anchor, orientations in enumerate(anchors):
            if not orientations:continue
            pattern = next(iter(orientations.values()))
            ids = [rng.randrange(15) for _ in range(23)]
            for cell, allowed in pattern:ids[cell] = allowed[0]
            for occupied in (0, 1 << anchor, (1 << 23)-1):
                board = [dict(stack_id=sid, animal=card if occupied & (1 << i) else None)
                         for i, sid in enumerate(ids)]
                expected = habitat_potential(board, card)
                assert kernel.potential(tuple(ids), occupied, card) == expected
                if backend == 'native':
                    assert kernel._NATIVE.values(ids, occupied, 0, [(1, card, 0)]) == [.2+expected]


def test_complete_games_features_masks_snapshots_and_observation_isolation(backend):
    for seed in range(4):
        old, new = FastHarmoniesDuo(), Harmonies()
        obs, _ = old.reset(seed+490, seed % 2);new_obs = new.reset(seed+490, seed % 2)
        rng = random.Random(seed)
        for step in range(768):
            decorated = dict(obs, game_id=new.game_id, config_id=new.config_id,
                             player=obs['active_player'], current_player=None if obs['terminated'] else obs['active_player'])
            assert new_obs == decorated
            assert old.save_state() == new.engine.save_state()
            assert new.legal_actions() == obs['legal_actions']
            assert kernel.features(new_obs, new.legal_actions()) == old_features(new_obs, new.legal_actions())
            assert kernel.features(new_obs, []) == []
            if obs['terminated']:break
            if step == 35:
                snapshot = new.snapshot();new.restore(snapshot)
                assert snapshot == new.snapshot()
            saved = deepcopy(new_obs)
            action = rng.choice(obs['legal_actions'])
            obs = old.step(action).observation;new_obs_next = new.step(action)['observation']
            assert new_obs == saved  # Later transitions must not mutate prior data.
            new_obs = new_obs_next
        assert obs['terminated']


def test_invalid_actions_and_unexpected_engine_error_are_atomic(monkeypatch, backend):
    env = Harmonies();env.reset(61)
    before = env.snapshot()
    for action in ({'type': 'choose_bundle', 'bundle_id': True},
                   {'type': 'choose_bundle', 'bundle_id': 1.0},
                   {'type': 'choose_bundle', 'bundle_id': []},
                   {'type': 'end_turn'}, {'type': 'choose_bundle', 'bundle_id': 9},
                   {'type': 'choose_bundle', 'bundle_id': 0, 'extra': 1}):
        with pytest.raises(InvalidAction):env.step(action)
        assert env.snapshot() == before
    def fail(action):
        env.engine.s['players'][0]['board'][0]['stack_id'] = 8
        env.engine._bag.pop()
        raise RuntimeError('injected transition failure')
    monkeypatch.setattr(env.engine, 'step', fail)
    with pytest.raises(RuntimeError):env.step({'type': 'choose_bundle', 'bundle_id': 0})
    assert env.snapshot() == before


def test_copied_observation_owns_mutable_containers():
    env = Harmonies();obs = env.reset(8);copied = copy_public(obs)
    assert copied == obs
    copied['players'][0]['board'][0]['stack_id'] = 14
    copied['legal_actions'][0]['bundle_id'] = 8
    copied['score_breakdown'][0]['total'] = -1
    copied['token_market'][0].clear()
    assert env.observe() == obs
    with pytest.raises(ValueError, match='official duo'):
        kernel.features(dict(obs, rules_version='different'), obs['legal_actions'])


def test_native_rejects_changed_artifacts_and_bad_inputs(artifact):
    with pytest.raises(ValueError, match='mismatch'):
        kernel.NativeKernel(dict(artifact, sha256='changed'))
    native = kernel.NativeKernel(artifact)
    for ids in ([0]*22, [0]*22+[15], [True]*23):
        with pytest.raises(ValueError):native.terrain(ids)
    for actions in ([(2, 23, 1)], [(1, 33, 0)], [(9, 0, 0)], [(2, 0, -1)]):
        with pytest.raises(ValueError):native.values([0]*23, 0, 0, actions)
    assert native.values([0]*23, 0, 0, []) == []
