from copy import deepcopy
import json
import random

import pytest

from boardbench.families.state_io import copy_data
from boardbench.families.registry import available_games, game_class
from boardbench.families.base import DuoGame
from boardbench.contracts import InvalidAction


def test_builtin_copies_preserve_types_cycles_aliases_and_isolation():
    shared = [1, 2, {3: (4, 5), 's': {1, 9}}]
    value = {'left': shared, 'right': shared, 'large': 1 << 100, 'rng': random.Random(3).getstate()}
    copied = copy_data(value)
    assert copied == deepcopy(value)
    assert copied['left'] is copied['right'] and copied['left'] is not shared
    copied['left'][2][3] = (0,)
    assert value['left'][2][3] == (4, 5)
    cycle = [];cycle.append(cycle)
    result = copy_data(cycle);assert result[0] is result and result is not cycle


@pytest.mark.parametrize('game', [g for g in available_games() if g != 'five_tribes'])
def test_game_copies_and_snapshots_equal_deepcopy(game):
    env = game_class(game)();env.reset(193)
    rng = random.Random(23)
    for _ in range(20):
        obs = env.observe();snapshot = env.snapshot()
        assert copy_data(obs) == deepcopy(obs)
        assert copy_data(snapshot) == deepcopy(snapshot)
        public = json.dumps(obs, sort_keys=True)
        original = deepcopy(snapshot)
        actions = env.legal_actions()
        if not actions:break
        env.step(rng.choice(actions))
        assert json.dumps(obs, sort_keys=True) == public
        assert snapshot == original


def test_legal_fast_path_preserves_json_guard_and_fallback():
    class Fixture(DuoGame):
        def _reset(self, first_player):pass
        def _observe(self, player):return {}
        def scores(self):return [0, 0]
        def _legal_actions(self):return [{'type': 'choose', 'x': [1, 2]}]
        def _step(self, action):self.finish([0])
    for x in ([True, 2], [1., 2], [1, float('nan')]):
        env = Fixture();env.reset(1);before = env.snapshot()
        with pytest.raises(InvalidAction):env.step({'type': 'choose', 'x': x})
        assert env.snapshot() == before
    env = Fixture();env.reset(1)
    assert env.step({'type': 'choose', 'x': (1, 2)})['terminated']


def test_cascadia_observation_local_batch_cache_matches_single_features():
    env = game_class('cascadia')();env.reset(17);rng = random.Random(90)
    for _ in range(150):
        actions = env.legal_actions()
        if not actions:break
        obs = env.observe()
        assert env.features_batch(obs, actions) == [env.features(obs, a) for a in actions]
        env.step(rng.choice(actions))
    assert env.outcome()['terminated']
