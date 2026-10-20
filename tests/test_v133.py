from copy import deepcopy
import os

import numpy as np
import pytest
import torch

from test_v127 import config
from test_v129 import artifact
from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.v121.registry import validate_manifest
from boardbench.v127.training import Trainer as OldTrainer
from boardbench.v130.pipeline import checkpoint_spec
from boardbench.v133.campaign import new_manifest
from boardbench.v133.training import ARMS, Replay, Trainer, run


@pytest.fixture
def continuous(config, artifact, tmp_path):
    old = {**config, "environment_backend": "native_v129", "native_artifact": artifact}
    parent = OldTrainer(old)
    parent.batch()
    parent.batch()
    path = tmp_path / "parent.pt"
    parent.save(path)
    parent.close()
    manifest = new_manifest(read_json(config["split_manifest"]), config["split_sha256"], [20000001], count=32)
    split = tmp_path / "new_splits.json"
    write_json(split, manifest)
    return {**old, "continuous_version": "1.3.3", "arm": "ppo", "critic_batches": 0,
            "split_manifest": str(split), "split_sha256": digest(split),
            "curve_seeds": manifest["sets"]["development"][:2],
            "final_seeds": manifest["sets"]["development"][2:4],
            "continuation": checkpoint_spec(path), "self_imitation_weight": .02,
            "replay_batches": 2, "replay_minibatch": 16, "replay_advantage_cap": 1.,
            "replay_directory": str(tmp_path / "replay"), "record_games_per_batch": 1}


def test_new_panels_and_training_pool(continuous):
    m = read_json(continuous["split_manifest"])
    validate_manifest(m)
    parent = torch.load(continuous["continuation"]["checkpoint"], weights_only=False)
    old = read_json(parent["config"]["split_manifest"])
    assert m["sets"]["train"] == old["sets"]["train"]
    excluded = set().union(*(set(v) for v in old["sets"].values()), {20000001})
    assert not excluded & set(m["sets"]["development"] + m["sets"]["confirmation"])
    assert len(set(m["sets"]["development"] + m["sets"]["confirmation"])) == 96


@pytest.mark.parametrize("arm", ARMS)
def test_parent_logits_values_optimizer_alignment(continuous, arm):
    c = {**continuous, "arm": arm}
    trainer = Trainer(c)
    trainer.import_sampling_continuation(c["continuation"])
    saved = torch.load(c["continuation"]["checkpoint"], weights_only=False)
    original = OldTrainer(saved["config"])
    original.model.load_state_dict(saved["model"])
    x = torch.randn(3, 1346, dtype=torch.float64)
    mask = torch.ones(3, 240, dtype=torch.bool)
    for a, b in zip(trainer.model(x, mask), original.model(x, mask)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for index, state in saved["optimizer"]["state"].items():
        for key, value in state.items():
            assert torch.equal(value, trainer.optimizer.state_dict()["state"][index][key])
    assert trainer.ppo_actions == saved["counters"]["ppo_actions"]
    trainer.close()
    original.close()


def test_value_gradient_cannot_reach_actor_body(continuous):
    trainer = Trainer({**continuous, "arm": "separate_critic"})
    trainer.import_sampling_continuation(continuous["continuation"])
    _, value = trainer.model(torch.randn(4, 1346, dtype=torch.float64), torch.ones(4, 240, dtype=torch.bool))
    (value - 1).square().mean().backward()
    assert all(p.grad is None for p in trainer.model.body.parameters())
    assert all(p.grad is None for p in trainer.model.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainer.model.critic_body.parameters())
    trainer.close()


@pytest.mark.parametrize("arm", ARMS)
def test_updates_and_exact_resume(continuous, arm, tmp_path):
    c = {**continuous, "arm": arm}
    trainer = Trainer(c)
    trainer.import_sampling_continuation(c["continuation"])
    row = trainer.batch()
    assert row["actor_change_l2"] > 0 and row["critic_change_l2"] > 0
    assert row["initial_ratio_max_error"] < 1e-4
    assert np.isfinite(row["mc_explained_variance"])
    assert row["bc_weight"] == (.02 if arm == "self_imitation" else 0)
    checkpoint = tmp_path / "resume.pt"
    trainer.save(checkpoint)
    expected = trainer.batch()
    weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.close()
    other = Trainer(c)
    other.load(checkpoint)
    actual = other.batch()
    assert expected["games"] == actual["games"]
    for key, value in weights.items():
        assert torch.equal(value, other.model.state_dict()[key])
    rng = torch.get_rng_state().clone()
    replay_before = other.replay.state() if other.replay else None
    other.evaluate(c["curve_seeds"])
    assert torch.equal(rng, torch.get_rng_state())
    assert replay_before == (other.replay.state() if other.replay else None)
    other.close()


def test_replay_integrity_and_all_return_signs(tmp_path):
    replay = Replay(tmp_path / "chunks", 2, 42)
    mask = np.array([True, False])
    rows = [(np.array([float(i)], np.float32), mask, 0, -.1, 0., 0., float(i)) for i in (-1, 0, 1)]
    replay.add(rows, 1, [7])
    assert set(replay.blocks[0]["mc"]) == {-1, 0, 1}
    state = replay.state()
    expected = replay.sample(100)
    other = Replay(tmp_path / "chunks", 2, 99)
    other.restore(state, {7})
    actual = other.sample(100)
    for k in expected:
        np.testing.assert_array_equal(expected[k], actual[k])
    with pytest.raises(ValueError, match="leakage"):
        other.restore(state, set())
    bad = deepcopy(state)
    bad["receipts"][0]["sha256"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        other.restore(bad, {7})


def test_self_imitation_active_and_zero_for_negative_advantages(continuous):
    trainer = Trainer({**continuous, "arm": "self_imitation"})
    trainer.import_sampling_continuation(continuous["continuation"])
    row = trainer.batch()
    assert row["self_imitation_loss"] > 0 and row["expert_bc_weight"] == 0
    for block in trainer.replay.blocks:
        block["mc"].fill(-100.)
    trainer.optimizer.zero_grad(set_to_none=True)
    loss = trainer.demonstration_loss()
    assert loss.item() == 0
    loss.backward()
    assert trainer.model.actor.weight.grad.abs().sum() == 0
    trainer.close()


def test_controller_termination_runs_cleanup_and_restores_handler(monkeypatch):
    import signal
    from boardbench.v133 import campaign
    prior = signal.getsignal(signal.SIGTERM)
    cleaned = []
    def interrupted(*_):
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            cleaned.append(True)
    monkeypatch.setattr(campaign, "_run", interrupted)
    with pytest.raises(InterruptedError):
        campaign.run(None, None)
    assert cleaned == [True]
    assert signal.getsignal(signal.SIGTERM) == prior


def test_full_runner_and_finished_noop(continuous, tmp_path):
    c = {**continuous, "target_actions": 1000, "milestones": [1000], "max_seconds": 90}
    out = tmp_path / "run"
    result = run(c, out, "cpu")
    assert result["status"] == "completed"
    assert (out / "final_comparison.json").exists()
    assert run(c, out, "cpu") == result


@pytest.mark.skipif(os.environ.get("BOARDBENCH_V133_CUDA") != "1", reason="explicit authorized GPU only")
@pytest.mark.parametrize("arm", ARMS)
def test_cuda_exact_resume(continuous, arm, tmp_path):
    c = {**continuous, "arm": arm, "batch_episodes": 8, "workers": 2}
    trainer = Trainer(c, "cuda")
    trainer.import_sampling_continuation(c["continuation"])
    row = trainer.batch()
    assert row["initial_ratio_max_error"] < 1e-4 and row["actor_change_l2"] > 0
    path = tmp_path / "gpu.pt"
    trainer.save(path)
    expected = trainer.batch()
    weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.close()
    other = Trainer(c, "cuda")
    other.load(path)
    actual = other.batch()
    assert actual["games"] == expected["games"]
    for k, value in weights.items():
        torch.testing.assert_close(value, other.model.state_dict()[k], rtol=1e-10, atol=1e-10)
    other.close()
