from copy import deepcopy

import numpy as np
import pytest
import torch

from boardbench.artifacts.store import digest, read_json, source_id
from boardbench.environments.harmonies import Harmonies
from boardbench.v12.training import policy_distribution
from boardbench.v121.neural import ResearchSolver
from boardbench.v121.registry import NUMERIC, definitions, identity, make_splits, training_config
from boardbench.v121.recovery import numeric_parent_config
from boardbench.v121.training import Trainer


def config(name):
    cfg = training_config(name, 911, make_splits(train_count=32))
    cfg.update(batch_episodes=2, epochs=1, minibatch_size=64, purpose='numeric_interface_check')
    return cfg


@pytest.mark.parametrize('name', list(NUMERIC))
def test_numeric_initialization_shapes_gradients_and_artifact(name, tmp_path):
    torch.set_num_threads(1)
    cfg = config(name)
    parent = numeric_parent_config(cfg)
    assert parent == config(NUMERIC[name])
    a = ResearchSolver(911, **cfg['network'])
    b = ResearchSolver(911, **parent['network'])
    for key, value in a.model.state_dict().items():
        torch.testing.assert_close(value, b.model.state_dict()[key].to(value.dtype), rtol=0, atol=0)
    env = Harmonies(); obs, _ = env.reset(3)
    data = a.encode(obs)
    x = torch.tensor([data['features']] * 19)
    mask = torch.tensor([data['action_mask']] * 19)
    logits, value = a.model(x, mask)
    assert logits.dtype == value.dtype == torch.float64
    assert torch.isfinite(logits).all() and (logits[~mask] == -1e9).all()
    one = a.model(x[:1], mask[:1])[0]
    torch.testing.assert_close(logits[:1], one, atol=1e-12, rtol=1e-12)
    loss = value.square().mean() - policy_distribution(logits).entropy().mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
    path = tmp_path / 'weights.pt'; a.save_weights(path)
    restored = ResearchSolver(912, **cfg['network']); restored.load_weights(path)
    assert a.decide(obs, None) == restored.decide(obs, None)
    with pytest.raises(ValueError, match='metadata'):
        b.load_weights(path)
    with pytest.raises(ValueError, match='dtype'):
        ResearchSolver(911, **{**cfg['network'], 'numeric_dtype':'float32'})


@pytest.mark.parametrize('name', list(NUMERIC))
def test_explicit_import_preserves_state_then_exact_numeric_resume(name, tmp_path):
    cfg = config(name)
    old = Trainer(numeric_parent_config(cfg)); old.ppo_batch()
    path = tmp_path / 'old.pt'; old.save(path)
    old_hash = digest(path)
    a = Trainer(cfg)
    with pytest.raises(ValueError, match='config/source'):
        a.load(path)  # Numeric migration must not masquerade as exact resume.
    a.import_numeric_checkpoint(path, old.source, old_hash)
    assert a.lineage['parent_resume_sha256'] == old_hash
    assert not a.lineage['exact_cross_source_reproduction']
    assert (a.steps, a.batches, a.updates, a.episodes, a.stop_decisions, a.decision_count) == (
        old.steps, old.batches, old.updates, old.episodes, old.stop_decisions, old.decision_count)
    state = torch.load(path, weights_only=False)
    assert torch.equal(torch.get_rng_state(), state['rng_torch'])
    assert a.rng.bit_generator.state == state['rng_numpy']
    for key, value in a.solver.model.state_dict().items():
        assert torch.equal(value, state['model'][key].to(value.dtype))
    for key, value in a.optimizer.state_dict()['state'].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, state['optimizer']['state'][key][field].to(tensor.dtype))
            if field in ('exp_avg', 'exp_avg_sq'):
                assert tensor.dtype == torch.float64
    row = a.ppo_batch()
    assert row['initial_ratio_max_error'] < 1e-10 and np.isfinite(row['loss'])
    save = tmp_path / 'new.pt'; a.save(save)
    a.ppo_batch()
    b = Trainer(cfg); b.load(save); b.ppo_batch()
    assert b.lineage == a.lineage
    for key, value in a.solver.model.state_dict().items():
        assert torch.equal(value, b.solver.model.state_dict()[key])
    for key, value in a.optimizer.state_dict()['state'].items():
        for field, tensor in value.items():
            assert torch.equal(tensor, b.optimizer.state_dict()['state'][key][field])
    assert digest(path) == old_hash
    for obj in (a,b,old):
        obj.close()


def test_numeric_import_rejects_unregistered_changes_and_wrong_provenance(tmp_path):
    cfg = config('E11_T16_F64')
    old = Trainer(numeric_parent_config(cfg)); old.ppo_batch()
    path = tmp_path / 'old.pt'; old.save(path)
    for source, sha in ((source_id(), 'bad'), ('bad', digest(path))):
        with pytest.raises(ValueError, match='hash|source'):
            Trainer(cfg).import_numeric_checkpoint(path, source, sha)
    wrong = deepcopy(cfg); wrong['learning_rate'] = .0001
    with pytest.raises(ValueError, match='config/source'):
        Trainer(wrong).import_numeric_checkpoint(path, source_id(), digest(path))
    with pytest.raises(ValueError, match='registered FP64'):
        Trainer(numeric_parent_config(cfg)).import_numeric_checkpoint(path, source_id(), digest(path))
    tampered = deepcopy(cfg); tampered['network']['numeric_dtype'] = 'float32'
    with pytest.raises(ValueError, match='network'):
        Trainer(tampered)
    # Gate must still reject corrupted recorded behavior probabilities in FP64.
    from boardbench.v121.collector import EnvironmentPool, collect
    new = Trainer(cfg)
    with EnvironmentPool(2, 0) as pool:
        rows, _ = collect(new, pool)
    rows = [(x,m,a,p+.001,r,v) for x,m,a,p,r,v in rows]
    with pytest.raises(RuntimeError, match='log-prob mismatch'):
        new._update(rows)
    assert new.updates == 0
    old.close(); new.close()
