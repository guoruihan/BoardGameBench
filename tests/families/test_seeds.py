import pytest
from boardbench.families.seeds import PARTITIONS, experiment_seed


def test_long_training_cannot_reach_gate_or_confirmation():
    values = {split: {experiment_seed(split, it, j)
              for it in (0, 511, 999, 1000, 2047, 100000, 2**32-1)
              for j in (0, 31, 65535)} for split in PARTITIONS}
    flattened = [x for group in values.values() for x in group]
    assert len(set(flattened)) == len(flattened)
    assert all(int(float(x)) == x and 0 <= x < 2**53 for x in flattened)
    for it, index in [(-1, 0), (2**32, 0), (0, -1), (0, 2**16)]:
        with pytest.raises(ValueError):
            experiment_seed('ppo_train', it, index)
