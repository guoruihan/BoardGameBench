from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

from boardbench.environments.harmonies import Harmonies, CARDS, PATTERNS
from boardbench.contracts import InvalidAction
from boardbench.v12.encoding import COUNTS, STOP, action_map, decode, encode, index


def fixture():
    spec = importlib.util.spec_from_file_location('animal_fixture', Path(__file__).parents[1] / 'scripts/animal_browser_fixture.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.fixture()


@pytest.mark.parametrize('version', COUNTS)
def test_shadow_trace_slot_shift_and_stop(version):
    trace = fixture()
    env = Harmonies()
    env.load_state(trace['snapshot'])
    for action in trace['actions']:
        obs = env.observe()
        data = encode(obs, version)
        mapping = action_map(obs, version, True)
        assert len(mapping) == len(obs['legal_actions']) + 1
        assert len(data['action_mask']) == COUNTS[version] + 1
        assert decode(obs, COUNTS[version], version, True) == STOP
        before = env.save_state()
        decoded = decode(obs, index(obs, action, version), version, True)
        copy = Harmonies()
        copy.load_state(before)
        copy.step(action)
        env.step(decoded)
        assert copy.save_state() == env.save_state()
    assert env.observe() == trace['final_observation']
    assert not any(encode(env.observe(), version, episode_finished=True)['action_mask'])
    with pytest.raises(InvalidAction):
        decode(env.observe(), True, version, True)


def test_all_card_rotation_successors_and_no_information_leak():
    comparisons = 0
    for card_id in CARDS:
        for orientation in range(6):
            env = Harmonies()
            anchor, pattern = next((a, p[orientation]) for a, p in enumerate(PATTERNS[card_id]) if orientation in p)
            for cell, allowed in pattern:
                env.s['board'][cell]['stack_id'] = allowed[0]
            env.s['active_cards'] = [dict(instance_id=card_id, card_id=card_id, placed_count=0)]
            obs = env.observe()
            for version in COUNTS:
                semantic = dict(type='place_animal', instance_id=card_id, anchor_cell_id=anchor, orientation=orientation)
                if version == 'legacy4564_v1':
                    semantic = next(a for a in obs['legal_actions'] if a['type'] == 'place_animal' and a['anchor_cell_id'] == anchor)
                restored = decode(obs, index(obs, semantic, version), version)
                a, b = deepcopy(env), deepcopy(env)
                a.step(semantic); b.step(restored)
                assert a.save_state() == b.save_state()
                comparisons += 1
            encoded = encode(obs)
            env._bag.reverse(); env._deck.reverse()
            assert encoded == encode(env.observe())
    assert comparisons == 32 * 6 * 3


def test_observation_ablations_and_time_conditions():
    obs = Harmonies().observe()
    assert encode(obs, 'legacy4564_v1', 'legacy_v1')['features'] == encode(obs, 'no_rotation884_v1', 'legacy_v1')['features']
    assert encode(obs, 'no_rotation884_v1', 'slots_v1')['features'] == encode(obs, 'slots240_v1', 'slots_v1')['features']
    with pytest.raises(ValueError):
        encode(obs, 'slots240_v1', 'legacy_v1')
    data = encode(obs, remaining_seconds=.5, time_budget_seconds=1.)
    assert data['features'][-1] == .5
    with pytest.raises(ValueError):
        encode(obs, remaining_seconds=float('nan'), time_budget_seconds=1.)
