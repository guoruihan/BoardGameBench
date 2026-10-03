import pytest

torch = pytest.importorskip('torch')
from boardbench.artifacts.store import read_json
from boardbench.training import train


def test_fixed_training_set_and_checkpoint_cadence(tmp_path):
    result = train(dict(task_id='micro_tiles', training_seed=31, episodes=8,
                        batch_episodes=2, train_seed_count=2, checkpoint_every=2,
                        epochs=1, threads=1, hidden=32), tmp_path/'train')
    assert read_json(tmp_path/'train/splits.json')['train'] == [100000, 100001]
    assert [c['episodes'] for c in result['checkpoints']] == [0, 2, 4, 6, 8]
    assert result['parameter_changed']


@pytest.mark.parametrize('changes', ({'train_seed_count': 0}, {'train_seed_count': 9},
                                   {'checkpoint_every': 3}))
def test_bad_diagnostic_configuration_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        train(dict(task_id='micro_tiles', training_seed=31, episodes=8,
                   batch_episodes=2, epochs=1, threads=1, **changes), tmp_path/'train')
