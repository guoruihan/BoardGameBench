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


def test_probability_normalizer_is_double_and_masked_gradients_are_finite():
    from boardbench.v12.training import policy_distribution
    logits = torch.tensor([[.2, -.4, -1e9]], requires_grad=True)
    dist = policy_distribution(logits)
    assert dist.logits.dtype == torch.float64
    assert dist.probs[0, 2] == 0
    assert torch.allclose(dist.probs.sum(-1), torch.ones(1, dtype=torch.float64))
    (-dist.log_prob(torch.tensor([0])).mean()).backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 2] == 0


def test_probability_gate_still_rejects_corrupted_behavior_before_update():
    from boardbench.v12.training import policy_distribution
    import numpy as np
    trainer = Trainer(config())
    obs, _ = Harmonies().reset(10)
    data = trainer.solver.encode(obs)
    x = np.asarray(data['features'], dtype=np.float32)
    mask = np.asarray(data['action_mask'], dtype=bool)
    action = int(np.flatnonzero(mask)[0])
    with torch.no_grad():
        logits, _ = trainer.solver.model(torch.tensor(x)[None], torch.tensor(mask)[None])
        lp = policy_distribution(logits).log_prob(torch.tensor([action])).item()
    before = parameter_hash(trainer.solver.model)
    with pytest.raises(RuntimeError, match='log-prob mismatch'):
        trainer._update([(x, mask, action, lp + .01, 0., 1.)])
    assert trainer.updates == 0 and parameter_hash(trainer.solver.model) == before


def test_bc_to_ppo_is_a_fresh_provenance_checked_phase(tmp_path):
    from boardbench.artifacts.store import digest
    cfg = config(); cfg['algorithm'] = 'bc'
    bc = Trainer(cfg)
    obs, _ = Harmonies().reset(10)
    data = bc.solver.encode(obs)
    example = {'features': data['features'], 'mask': data['action_mask'],
               'action': data['action_mask'].index(True), 'seed': 10}
    bc.bc_batch([example])
    saved = tmp_path / 'bc.pt'; bc.save(saved)
    weights = saved.with_suffix('.weights.pt')
    ppo = Trainer(config())
    ppo.initialize_from_bc(weights, expected_sha256=digest(weights))
    assert parameter_hash(ppo.solver.model) == parameter_hash(bc.solver.model)
    assert not ppo.optimizer.state and ppo.episodes == ppo.updates == ppo.batches == 0
    assert ppo.lineage['kind'] == 'bc_to_ppo' and ppo.lineage['weights_sha256'] == digest(weights)
    ppo.ppo_batch()
    after = tmp_path / 'ppo.pt'; ppo.save(after)
    resumed = Trainer(config()); resumed.load(after)
    assert resumed.lineage == ppo.lineage
    with pytest.raises(ValueError, match='fresh'):
        ppo.initialize_from_bc(weights, expected_sha256=digest(weights))
    with pytest.raises(ValueError, match='hash'):
        Trainer(config()).initialize_from_bc(weights, expected_sha256='bad')
    bad = config(); bad['splits']['test'] = [10]
    bad['splits']['train'] = [40]
    with pytest.raises(ValueError, match='splits'):
        Trainer(bad).initialize_from_bc(weights, expected_sha256=digest(weights))
