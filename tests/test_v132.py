from copy import deepcopy
import gzip
import json
import os

import numpy as np
import pytest
import torch

from test_v127 import config
from test_v129 import artifact
from boardbench.artifacts.store import digest, read_json
from boardbench.environments.harmonies import Harmonies
from boardbench.v12.encoding import encode
from boardbench.v129.collector import EnvironmentPool as NativePool
from boardbench.v132.collector import EnvironmentPool
from boardbench.v132.data import FORMAT, load, encoded_sequences
from boardbench.v132.training import Trainer, remaining_window


@pytest.fixture
def curriculum(config, artifact, tmp_path):
    manifest = read_json(config["split_manifest"])
    seed = manifest["sets"]["train"][4]
    env = Harmonies()
    obs, _ = env.reset(seed)
    observations, actions, rewards = [], [], []
    rng = np.random.default_rng(132)
    while not obs["terminated"]:
        action = obs["legal_actions"][int(rng.integers(len(obs["legal_actions"])))]
        observations.append(obs)
        actions.append(action)
        old = obs["score_breakdown"]["total"]
        obs = env.step(action).observation
        rewards.append(obs["score_breakdown"]["total"] - old)
    corpus = {"format": FORMAT, "rules_version": Harmonies.rules_version,
              "split_sha256": manifest["sha256"], "episodes": [{
                  "id": "fixture", "environment_seed": seed, "observations": observations,
                  "actions": actions, "reward_deltas": rewards,
                  "terminal_score": obs["score_breakdown"]["total"]}]}
    path = tmp_path / "corpus.json.gz"
    path.write_bytes(gzip.compress(json.dumps(corpus).encode()))
    return {**config, "curriculum_version": "1.2.12", "reset_mode": "backward",
            "reset_fraction": .5, "curriculum_actions": 10000, "environment_backend": "native_v129",
            "native_artifact": artifact, "corpus": {"path": str(path), "sha256": digest(path)}}


def test_data_admission_and_sequences(curriculum, tmp_path):
    trainer = Trainer(curriculum)
    corpus = trainer.corpus
    sequences = encoded_sequences(corpus)
    assert sequences[0]["x"].shape[1] == 1346
    assert sequences[0]["mask"][np.arange(len(sequences[0]["y"])), sequences[0]["y"]].all()
    with pytest.raises(ValueError, match="leakage"):
        load(curriculum["corpus"], set())
    with pytest.raises(ValueError, match="checksum"):
        load({**curriculum["corpus"], "sha256": "wrong"}, set(trainer.train_seeds))
    for key, value in (("private", {}), ("rules_version", "wrong")):
        bad = deepcopy(corpus)
        bad["episodes"][0]["observations"][0][key] = value
        path = tmp_path / ("bad_" + key + ".gz")
        path.write_bytes(gzip.compress(json.dumps(bad).encode()))
        with pytest.raises(ValueError, match="invalid"):
            load({"path": str(path), "sha256": digest(path)}, set(trainer.train_seeds))
    trainer.close()


@pytest.mark.parametrize("workers", [0, 2])
def test_normal_reset_exact_and_public_reset(curriculum, workers):
    artifact = curriculum["native_artifact"]
    with NativePool(2, workers, native_artifact=artifact) as a, EnvironmentPool(
            2, workers, native_artifact=artifact) as b:
        x, y = a.reset([45, 123]), b.reset([45, 123])
        while not x["done"].all():
            for key in ("features", "masks", "done", "stats"):
                np.testing.assert_array_equal(x[key], y[key])
            slots = np.flatnonzero(~x["done"])
            actions = [np.flatnonzero(x["masks"][i, :240])[0] for i in slots]
            x, y = a.step(slots, actions), b.step(slots, actions)
        trainer = Trainer(curriculum)
        ep = trainer.corpus["episodes"][0]
        obs = ep["observations"][len(ep["actions"]) // 2]
        specs = [{"observation": obs, "future_seed": s, "environment_seed": ep["environment_seed"],
                  "source": {"episode": ep["id"], "decision": 1}} for s in [7, 19]]
        arrays = b.reset(specs)
        np.testing.assert_array_equal(arrays["features"][0], arrays["features"][1])
        np.testing.assert_array_equal(arrays["features"][0],
                                      np.asarray(encode(obs, allow_stop=False)["features"], np.float32))
        assert arrays["stats"][0, 1] == obs["score_breakdown"]["total"]
        # Different unseen futures, identical publicly visible state/features.
        ea, eb = [Harmonies.from_observation(obs, s) for s in [7, 19]]
        assert ea.observe() == eb.observe() and ea.save_state()["private"] != eb.save_state()["private"]
        trainer.close()


def test_curriculum_true_rewards_update_and_resume(curriculum, tmp_path):
    trainer = Trainer(curriculum)
    trainer.batch()
    row = trainer.batch()
    assert row["reset_count"] == 1 and row["actor_change_l2"] > 0
    assert row["initial_ratio_max_error"] < 1e-4
    for game in row["games"]:
        assert game["score"] == game["initial_score"] + game["earned_score"]
    checkpoint = tmp_path / "resume.pt"
    trainer.save(checkpoint)
    expected = trainer.batch()
    weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.close()
    other = Trainer(curriculum)
    other.load(checkpoint)
    actual = other.batch()
    assert actual["games"] == expected["games"]
    assert all(torch.equal(v, weights[k]) for k, v in other.model.state_dict().items())
    rng = deepcopy(other.rng.bit_generator.state)
    result = other.evaluate(curriculum["curve_seeds"])
    assert all(not g["curriculum_reset"] and g["initial_score"] == 0 for g in result["games"])
    assert rng == other.rng.bit_generator.state
    other.close()


def test_adaptive_branch_depends_on_result():
    from boardbench.v132.pipeline import choose_plan
    initial = choose_plan(1, [], expanded=False, stagnation=0)
    settings = dict(initial["arms"])
    def results(gain):
        return [{"arm": name, "last_curve_score": 100 + (gain if name == "backward" else 0),
                 "settings": cfg} for name, cfg in settings.items() for _ in range(2)]
    improved = choose_plan(2, results(3), expanded=False, stagnation=0)
    flat = choose_plan(2, results(0), expanded=False, stagnation=1)
    assert "continue_winner" in dict(improved["arms"])
    assert "mc_normal" in dict(flat["arms"])
    assert dict(improved["arms"])["continue_winner"]["reset_mode"] == "backward"
    assert len(improved["arms"]) == len(flat["arms"]) == 4


def test_sequence_auxiliary_and_credit(curriculum):
    trainer = Trainer({**curriculum, "sequence_bc_weight": .01, "gae_lambda": 1.})
    trainer.batch()
    row = trainer.batch()
    assert row["bc_weight"] == .01 and row["bc_loss"] > 0
    assert remaining_window(curriculum, 0)[0] == .75
    assert remaining_window(curriculum, 100000)[0] == .05
    trainer.close()


@pytest.mark.skipif(os.environ.get("BOARDBENCH_V132_CUDA") != "1",
                    reason="explicit authorized CUDA preflight only")
def test_cuda_actual_resume(curriculum, tmp_path):
    c = {**curriculum, "hidden": 256, "batch_episodes": 8, "workers": 2}
    trainer = Trainer(c, "cuda")
    trainer.batch()
    assert trainer.batch()["actor_change_l2"] > 0
    p = tmp_path / "gpu.pt"
    trainer.save(p)
    expected = trainer.batch()
    weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.close()
    other = Trainer(c, "cuda")
    other.load(p)
    actual = other.batch()
    assert expected["games"] == actual["games"]
    for k, v in other.model.state_dict().items():
        torch.testing.assert_close(v, weights[k], rtol=1e-10, atol=1e-10)
    other.close()
