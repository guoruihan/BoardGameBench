from copy import deepcopy

import pytest
import torch

from boardbench.environments.harmonies import Harmonies
from boardbench.training import parameter_hash
from boardbench.v12.neural import CompactSolver
from boardbench.v12.training import Trainer, returns


def config():
    return {'training_seed': 191, 'network': {'hidden': 16, 'action_encoding': 'slots240_v1',
             'observation_encoding': 'slots_v1', 'allow_stop': True},
            'splits': {'train': [10, 11, 12, 13], 'validation': [21], 'test': [31]},
            'batch_episodes': 2, 'epochs': 1, 'threads': 1, 'minibatch_size': 64}


def test_returns_and_initial_ratio_and_exact_resume(tmp_path):
    assert returns([2., -3., 5.]) == [4., 2., 5.]
    assert returns([2., -3.], 5.) == [4., 2.]
    a = Trainer(config())
    initial = parameter_hash(a.solver.model)
    first = a.ppo_batch()
    assert first['initial_ratio_max_error'] < 1e-4
    assert parameter_hash(a.solver.model) != initial
    path = tmp_path / 'state.pt'
    a.save(path)
    a.ppo_batch()
    expected = parameter_hash(a.solver.model)
    b = Trainer(config())
    b.load(path)
    b.ppo_batch()
    assert expected == parameter_hash(b.solver.model)
    assert (a.episodes, a.steps, a.updates) == (b.episodes, b.steps, b.updates)
    for i, state in a.optimizer.state_dict()['state'].items():
        for key, value in state.items():
            assert torch.equal(value, b.optimizer.state_dict()['state'][i][key])
    changed = config(); changed['learning_rate'] = .01
    with pytest.raises(ValueError, match='config/source'):
        Trainer(changed).load(path)


def test_compact_artifact_metadata_round_trip(tmp_path):
    a = CompactSolver(1, hidden=16)
    path = tmp_path / 'weights.pt'; a.save_weights(path)
    b = CompactSolver(2, hidden=16); b.load_weights(path)
    assert parameter_hash(a.model) == parameter_hash(b.model)
    with pytest.raises(ValueError, match='metadata'):
        CompactSolver(2, hidden=32).load_weights(path)


def test_warm_start_time_columns_stay_zero_and_timed_ppo_is_explicit():
    solver = CompactSolver(1, hidden=16)
    obs, _ = Harmonies().reset(2)
    assert solver.encode(obs) == solver.encode(obs, remaining_seconds=5., time_budget_seconds=10.)
    cfg = config(); cfg['network']['time_conditioning'] = True
    with pytest.raises(ValueError, match='episode_seconds'):
        Trainer(cfg)
    cfg['episode_seconds'] = .1
    trainer = Trainer(cfg)
    assert trainer.solver.encode(obs, remaining_seconds=.1, time_budget_seconds=.1)['features'][-1] == 1.
    metrics = trainer.ppo_batch()
    assert metrics['initial_ratio_max_error'] < 1e-4 and 'wall time' in metrics['objective']
