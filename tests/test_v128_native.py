"""Optional compiler tests for the standalone prototype, never a training switch."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import subprocess

import pytest
np = pytest.importorskip('numpy')

from boardbench.environments.harmonies import CARDS, COLORS
from boardbench.v12.encoding import feature_vector
from boardbench.v128.environment import FastHarmonies


@pytest.fixture(scope='module')
def native(tmp_path_factory):
    compiler = shutil.which('g++')
    if not compiler:
        pytest.skip('optional C++ prototype requires g++')
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    spec = importlib.util.spec_from_file_location('native_probe', scripts/'benchmark_harmonies_native.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    library = tmp_path_factory.mktemp('native')/'encoder.so'
    subprocess.run([compiler, '-O3', '-std=c++17', '-shared', '-fPIC',
                    str(scripts/'native/harmonies_encode.cpp'), '-o', str(library)], check=True, timeout=120)
    return module, module.NativeEncoder(library)


def test_native_encoding_complete_random_and_habitat_traces(native):
    module, encoder = native
    observations, _ = module.collect(list(range(8)))
    expected = np.asarray([feature_vector(o) for o in observations], dtype=np.float32)
    np.testing.assert_array_equal(encoder.encode(module.pack(observations)), expected)


def test_native_all_card_ids_counts_and_public_fields(native):
    module, encoder = native
    base, _ = FastHarmonies().reset(127)
    observations = []
    for card_id, card in CARDS.items():
        for placed in range(card['cube_count']+1):
            obs = deepcopy(base)
            obs['active_cards'] = [{'card_id': card_id, 'instance_id': card_id, 'placed_count': placed}]
            obs['animal_market'] = [card_id, None, 33-card_id]
            for i, cell in enumerate(obs['board']):
                cell.update(stack_id=(i+card_id)%15, animal=None if i%3 else card_id)
            obs['remaining_cards'] = list(range(card_id, 33))
            obs['turn_index'] = card_id%13+1
            obs['bundle_taken'] = card_id%2 == 0
            obs['card_taken'] = placed%2 == 0
            obs['terminated'] = placed == card['cube_count']
            obs['token_market'] = [list(COLORS[:3]), None, [COLORS[-1]]*3]
            observations.append(obs)
    np.testing.assert_array_equal(encoder.encode(module.pack(observations)),
                                  np.asarray([feature_vector(o) for o in observations], dtype=np.float32))


def test_native_empty_layout_bounds_and_output_isolation(native):
    module, encoder = native
    obs, _ = FastHarmonies().reset(2)
    packed = module.pack([obs])
    original = encoder.encode(packed)
    assert encoder.encode(packed[:0]).shape == (0, 1346)
    for column, invalid in [(0, 15), (23, 33), (82, -1), (85, 33)]:
        bad = packed.copy(); bad[0, column] = invalid
        with pytest.raises(ValueError): encoder.encode(bad)
    with pytest.raises(ValueError): encoder.encode(packed.astype(np.float32))
    with pytest.raises(ValueError): encoder.encode(packed[:, :-1])
    changed = encoder.encode(packed); changed.fill(99)
    np.testing.assert_array_equal(original, np.asarray([feature_vector(obs)], dtype=np.float32))
