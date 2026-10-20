from copy import deepcopy
import os
import time

import numpy as np
import pytest
import torch

from test_v127 import config
from test_v129 import artifact
from boardbench.artifacts.store import digest, read_json, write_json
from boardbench.environments.harmonies import Harmonies
from boardbench.v127.training import Trainer as OldTrainer
from boardbench.v129.collector import EnvironmentPool as NativePool
from boardbench.v130.pipeline import checkpoint_spec
from boardbench.v133.campaign import new_manifest
from boardbench.v134.collector import EnvironmentPool
from boardbench.v134.plans import ARMS, K, PLAN_DIM, candidates, encode_plan, refresh_goal, compatible
from boardbench.v134.training import Trainer, manager_targets, run
from boardbench.v134.geometry import top_goals
from boardbench.v125.compiled import card_goals_compiled
from boardbench.v134.campaign import new_manifest as plan_manifest


@pytest.fixture
def planning(config, artifact, tmp_path):
    old = {**config, "environment_backend": "native_v129", "native_artifact": artifact}
    trainer = OldTrainer(old)
    trainer.batch()
    trainer.batch()
    path = tmp_path / "parent.pt"
    trainer.save(path)
    trainer.close()
    manifest = new_manifest(read_json(config["split_manifest"]), config["split_sha256"], [], count=8)
    split = tmp_path / "plansplit.json"
    write_json(split, manifest)
    return {**old, "planning_version": "1.3.4", "arm": "capacity", "critic_batches": 0,
            "split_manifest": str(split), "split_sha256": digest(split),
            "curve_seeds": manifest["sets"]["development"][:2], "final_seeds": manifest["sets"]["development"][2:4],
            "continuation": checkpoint_spec(path), "plan_horizon": 2, "manager_lr": .0001}


def test_public_plan_feasibility_and_encoding():
    obs, _ = Harmonies().reset(140)
    bank = candidates(obs)
    assert bank[0] == [] and 1 < len(bank) <= K
    for plan in bank:
        x = encode_plan(obs, plan)
        assert x.shape == (PLAN_DIM,) and np.isfinite(x).all()
        for goal in plan:
            assert refresh_goal(obs, goal) is not None
        if len(plan) == 2:
            assert compatible(obs, *plan)
    other = deepcopy(obs)
    other["hidden_seed_must_be_ignored"] = 123456
    assert candidates(other) == bank
    goal = bank[1][0]
    broken = deepcopy(obs)
    broken["board"][goal["anchor"]]["animal"] = 32
    assert refresh_goal(broken, goal) is None


def test_vector_geometry_matches_existing_goal_generator():
    rng = np.random.default_rng(134)
    for seed in [1, 32, 144]:
        game = Harmonies()
        obs, _ = game.reset(seed)
        for step in range(60):
            if step % 10 == 0:
                for card in range(1, 33):
                    assert top_goals(obs, card) == card_goals_compiled(obs, card)[:12]
            if obs["terminated"]:
                break
            obs = game.step(obs["legal_actions"][int(rng.integers(len(obs["legal_actions"])))]).observation


@pytest.mark.parametrize("workers", [0, 2])
def test_plan_pool_does_not_change_engine(artifact, workers):
    rng = np.random.default_rng(134)
    with NativePool(4, workers, native_artifact=artifact) as a, EnvironmentPool(
            4, workers, native_artifact=artifact, arm="persistent") as b:
        for seeds in ([12, 34, 56, 78], [111, 222, 333, 444]):
            x, y = a.reset(seeds), b.reset(seeds)
            while True:
                for key in ("features", "masks", "done", "stats"):
                    np.testing.assert_array_equal(x[key], y[key])
                if x["done"].all():
                    break
                slots = np.flatnonzero(~x["done"])
                b.extra["choice"][slots] = 0
                chosen = [int(rng.choice(np.flatnonzero(x["masks"][s, :240]))) for s in slots]
                a.step(slots, chosen)
                b.step(slots, chosen)


def test_manager_temporal_returns():
    trajectory = [(None, None, None, None, r, None) for r in [.1, -.2, .3, .4, .5]]
    decisions = [{"start": 0, "value": .8}, {"start": 3, "value": .6}]
    rows = manager_targets(trajectory, decisions, 1.)
    np.testing.assert_allclose([r["mc"] for r in rows], [1.1, .9])
    np.testing.assert_allclose([r["target"] for r in rows], [1.1, .9])
    assert [r["duration"] for r in rows] == [3, 2]


def test_fresh_panels_leave_training_and_test_unchanged(config):
    old = read_json(config["split_manifest"])
    new = plan_manifest(old, config["split_sha256"], count=32)
    assert new["sets"]["train"] == old["sets"]["train"]
    assert new["sets"]["test"] == old["sets"]["test"]
    fresh = new["sets"]["development"] + new["sets"]["confirmation"]
    assert len(fresh) == len(set(fresh)) == 96
    assert not set(fresh) & set().union(*(set(s) for s in old["sets"].values()))


@pytest.mark.parametrize("arm", ARMS)
def test_initial_alignment_and_real_updates(planning, arm):
    c = {**planning, "arm": arm}
    trainer = Trainer(c)
    trainer.import_sampling_continuation(c["continuation"])
    parent = torch.load(c["continuation"]["checkpoint"], weights_only=False)
    old = OldTrainer(parent["config"])
    old.model.load_state_dict(parent["model"])
    x = torch.randn(4, 1346 + PLAN_DIM, dtype=torch.float64)
    mask = torch.ones(4, 240, dtype=torch.bool)
    for a, b in zip(trainer.model(x, mask), old.model(x[:, :1346], mask)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    initial = trainer.evaluate(c["curve_seeds"])
    assert initial["games"] == old.evaluate(c["curve_seeds"])["games"]
    row = trainer.batch()
    assert row["initial_ratio_max_error"] < 1e-4
    assert row["actor_change_l2"] > 0 and row["plan_adapter_change"] > 0
    if arm == "manager":
        assert row["manager_change"] > 0 and row["manager_ratio_error"] < 1e-4
        assert row["mean_option_actions"] > 1
    old.close()
    trainer.close()


@pytest.mark.parametrize("arm", ARMS)
def test_exact_resume_and_evaluation_rng(planning, arm, tmp_path):
    c = {**planning, "arm": arm}
    trainer = Trainer(c)
    trainer.import_sampling_continuation(c["continuation"])
    trainer.batch()
    path = tmp_path / "checkpoint.pt"
    trainer.save(path)
    expected = trainer.batch()
    weights = {key: {k: v.clone() for k, v in getattr(trainer, key).state_dict().items()}
               for key in ("model", "manager")}
    trainer.close()
    other = Trainer(c)
    other.load(path)
    actual = other.batch()
    assert expected["games"] == actual["games"]
    for key, state in weights.items():
        for name, value in state.items():
            assert torch.equal(value, getattr(other, key).state_dict()[name])
    rng, manager_rng = torch.get_rng_state().clone(), other.manager_rng.get_state().clone()
    first, second = other.evaluate(c["curve_seeds"]), other.evaluate(c["curve_seeds"])
    assert first["games"] == second["games"]
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(manager_rng, other.manager_rng.get_state())
    other.close()


def test_complete_runner_keeps_both_policies(planning, tmp_path):
    c = {**planning, "arm": "manager", "target_actions": 500, "max_seconds": 90, "milestones": [500]}
    out = tmp_path / "run"
    result = run(c, out, "cpu")
    assert result["status"] == "completed"
    saved = torch.load(out / "final.pt", weights_only=False)
    assert "manager" in saved and "manager_optimizer" in saved
    assert (out / "evaluations/final.json").exists()


def test_stage_continuation_preserves_manager_worker_and_rng(planning, tmp_path):
    c = {**planning, "arm": "manager"}
    first = Trainer(c)
    first.import_sampling_continuation(c["continuation"])
    first.batch()
    path = tmp_path / "stage1.pt"
    first.save(path)
    expected = first.batch()
    weights = {name: {k: v.clone() for k, v in getattr(first, name).state_dict().items()}
               for name in ("model", "manager")}
    first.close()
    new = {**c, "continuation": checkpoint_spec(path), "target_actions": 10000, "milestones": [10000]}
    second = Trainer(new)
    second.import_sampling_continuation(new["continuation"])
    actual = second.batch()
    assert expected["games"] == actual["games"]
    for name, state in weights.items():
        for k, v in state.items():
            assert torch.equal(v, getattr(second, name).state_dict()[k])
    second.close()
    changed = Trainer({**new, "plan_horizon": 3})
    with pytest.raises(ValueError, match="preserve"):
        changed.import_sampling_continuation(new["continuation"])
    changed.close()


def test_zero_plan_intervention_is_real_and_capacity_control_is_invariant(planning):
    trainer = Trainer({**planning, "arm": "persistent"})
    trainer.import_sampling_continuation(planning["continuation"])
    trainer.batch()
    obs, _ = Harmonies().reset(140)
    from boardbench.v12.encoding import encode
    encoded = encode(obs, allow_stop=False)
    plan = encode_plan(obs, candidates(obs)[1])
    x = torch.tensor(np.r_[encoded["features"], plan][None], dtype=torch.float64)
    mask = torch.tensor(np.asarray(encoded["action_mask"])[None])
    normal, _ = trainer.model(x, mask)
    trainer.model.intervention = "zero"
    removed, _ = trainer.model(x, mask)
    assert (normal[mask] - removed[mask]).abs().max() > 0
    x[:, 1346:] = 0
    a, _ = trainer.model(x, mask)
    trainer.model.intervention = "normal"
    b, _ = trainer.model(x, mask)
    assert torch.equal(a, b)
    trainer.close()


@pytest.mark.skipif(os.environ.get("BOARDBENCH_V134_CUDA") != "1", reason="authorized GPU only")
@pytest.mark.parametrize("arm", ARMS)
def test_cuda_resume(planning, arm, tmp_path):
    c = {**planning, "arm": arm, "batch_episodes": 8, "workers": 2}
    trainer = Trainer(c, "cuda")
    trainer.import_sampling_continuation(c["continuation"])
    row = trainer.batch()
    assert row["initial_ratio_max_error"] < 1e-4 and row["plan_adapter_change"] > 0
    path = tmp_path / "cuda.pt"
    trainer.save(path)
    expected = trainer.batch()
    weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    high = {k: v.clone() for k, v in trainer.manager.state_dict().items()}
    trainer.close()
    other = Trainer(c, "cuda")
    other.load(path)
    actual = other.batch()
    assert actual["games"] == expected["games"]
    for name, value in weights.items():
        torch.testing.assert_close(other.model.state_dict()[name], value, atol=1e-10, rtol=1e-10)
    for name, value in high.items():
        torch.testing.assert_close(other.manager.state_dict()[name], value, atol=1e-10, rtol=1e-10)
    other.close()
