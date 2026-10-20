from copy import deepcopy
import os

import numpy as np
import pytest
import torch

from test_v127 import config
from test_v129 import artifact
from test_v134 import planning
from boardbench.artifacts.store import read_json
from boardbench.environments.harmonies import Harmonies, COLORS, STACKS
from boardbench.v12.encoding import encode
from boardbench.v130.pipeline import checkpoint_spec
from boardbench.v134.plans import candidates, encode_plan
from boardbench.v134.training import Trainer as PlanTrainer
from boardbench.v135.advice import ActionAdvice, METHODS
from boardbench.v135.training import Trainer, run
from boardbench.v135.campaign import new_manifest, promotion


@pytest.fixture
def advice_config(planning, tmp_path):
    parent = PlanTrainer({**planning, "arm": "reactive"})
    parent.import_sampling_continuation(planning["continuation"])
    parent.batch()
    path = tmp_path / "plans_parent.pt"
    parent.save(path)
    parent.close()
    return {**planning, "arm": "reactive", "advice_version": "1.3.5", "method": "control",
            "advice_strength": .6, "continuation": checkpoint_spec(path)}


def configured(c, method):
    return {**c, "method": method, "arm": "persistent" if method == "sticky" else "reactive"}


def test_public_advice_exact_codec():
    compiler = ActionAdvice()
    for seed in range(12):
        obs, _ = Harmonies().reset(seed)
        plan = candidates(obs)[1]
        encoded = encode(obs, allow_stop=False)
        actual = compiler(torch.tensor([encoded["features"]], dtype=torch.float64),
                          torch.tensor(encode_plan(obs, plan)[None], dtype=torch.float64))[0].numpy()
        expected = np.zeros(240)
        for goal, weight in zip(plan, (1., .4)):
            need = goal["need"]
            for i, bundle in enumerate(obs["token_market"]):
                expected[i] += weight * sum(min(bundle.count(c), need.get(c, 0)) for c in COLORS) / max(1, sum(need.values()))
            for cell, target in goal["targets"].items():
                stack, desired = STACKS[obs["board"][cell]["stack_id"]], STACKS[target]
                for ci, color in enumerate(COLORS):
                    good = len(stack) < len(desired) and desired[:len(stack)] == stack and desired[len(stack)] == color
                    expected[3 + ci * 23 + cell] += weight * (1 if good else -1)
                if stack != desired:
                    for slot in range(4):
                        expected[144 + slot * 23 + cell] -= .5 * weight
            for slot, card in enumerate(obs["animal_market"]):
                if card == goal["card_id"]:
                    expected[141 + slot] += weight
                    expected[237 + slot] -= weight
            for slot, card in enumerate(obs["active_cards"]):
                if card["card_id"] == goal["card_id"]:
                    expected[144 + slot * 23 + goal["anchor"]] += 1.5 * weight
        np.testing.assert_allclose(actual, expected.clip(-1, 1), atol=1e-7)


def test_parent_and_zero_advice_alignment(advice_config):
    t = Trainer(advice_config)
    t.import_sampling_continuation(advice_config["continuation"])
    saved = torch.load(advice_config["continuation"]["checkpoint"], weights_only=False)
    parent = PlanTrainer(saved["config"])
    parent.model.load_state_dict(saved["model"])
    assert t.evaluate(advice_config["curve_seeds"])["games"] == parent.evaluate(advice_config["curve_seeds"])["games"]
    g = Trainer(configured(advice_config, "gated"))
    g.import_sampling_continuation(advice_config["continuation"])
    obs, _ = Harmonies().reset(140)
    e = encode(obs, allow_stop=False)
    vectors = [encode_plan(obs, p) for p in candidates(obs)[1:]]
    x = torch.tensor(np.stack([np.r_[e["features"], p] for p in vectors]), dtype=torch.float64)
    mask = torch.tensor([e["action_mask"]] * len(vectors))
    normal, _ = g.model(x, mask)
    g.model.advice_intervention = "zero"
    removed, _ = g.model(x, mask)
    torch.testing.assert_close(removed, t.model(x, mask)[0], rtol=0, atol=0)
    assert (normal[mask] - removed[mask]).abs().max() > .1
    residual = normal - removed
    assert (residual[:, mask[0]] - residual[0, mask[0]]).abs().max() > .1
    assert normal[~mask].eq(-1e9).all()
    g.model.advice_intervention = "normal"
    fixed = Trainer(configured(advice_config, "advice"))
    fixed.import_sampling_continuation(advice_config["continuation"])
    torch.testing.assert_close(g.model(x, mask)[0], fixed.model(x, mask)[0], atol=1e-14, rtol=1e-14)
    for trainer in (t, parent, g, fixed):
        trainer.close()


def check_resume(advice_config, method, tmp_path, device):
    c = configured(advice_config, method)
    t = Trainer(c, device)
    t.import_sampling_continuation(c["continuation"])
    row = t.batch()
    assert row["initial_ratio_max_error"] < 1e-4
    assert row["actor_change_l2"] > 0
    if method in ("gated", "sticky"):
        assert row["advice_gate_change"] > 0
    checkpoint = tmp_path / "checkpoint.pt"
    t.save(checkpoint)
    expected = t.batch()
    weights = deepcopy(t.model.state_dict())
    t.close()
    restored = Trainer(c, device)
    restored.load(checkpoint)
    actual = restored.batch()
    assert actual["games"] == expected["games"]
    for k, v in weights.items():
        torch.testing.assert_close(v, restored.model.state_dict()[k], rtol=0, atol=0)
    rng = torch.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if device == "cuda" else []
    restored.evaluate(c["curve_seeds"])
    assert torch.equal(rng, torch.get_rng_state())
    if cuda:
        assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    restored.close()


@pytest.mark.parametrize("method", METHODS)
def test_update_resume(advice_config, method, tmp_path):
    check_resume(advice_config, method, tmp_path, "cpu")


@pytest.mark.skipif(os.environ.get("BOARDBENCH_V135_CUDA") != "1", reason="authorized GPU only")
@pytest.mark.parametrize("method", METHODS)
def test_cuda_resume(advice_config, method, tmp_path):
    check_resume({**advice_config, "batch_episodes": 8, "workers": 2}, method, tmp_path, "cuda")


def test_stage_continuation_and_runner(advice_config, tmp_path):
    c = configured(advice_config, "gated")
    t = Trainer(c)
    t.import_sampling_continuation(c["continuation"])
    t.batch()
    path = tmp_path / "stage1.pt"
    t.save(path)
    expected = t.batch()
    weights = deepcopy(t.model.state_dict())
    t.close()
    new = {**c, "continuation": checkpoint_spec(path), "target_actions": 1000, "milestones": [1000], "max_seconds": 90}
    resumed = Trainer(new)
    resumed.import_sampling_continuation(new["continuation"])
    assert resumed.batch()["games"] == expected["games"]
    for k, v in weights.items():
        assert torch.equal(v, resumed.model.state_dict()[k])
    resumed.close()
    invalid = Trainer({**new, "advice_strength": .7})
    with pytest.raises(ValueError, match="preserve"):
        invalid.import_sampling_continuation(new["continuation"])
    invalid.close()
    result = run(new, tmp_path / "run", "cpu")
    assert result["status"] == "completed"
    assert read_json(tmp_path / "run/evaluations/final.json")["games"]


def test_manifest_and_promotion(config, tmp_path):
    from boardbench.artifacts.store import write_json
    old = read_json(config["split_manifest"])
    manifest = new_manifest(old, config["split_sha256"], 32)
    assert manifest["sets"]["train"] == old["sets"]["train"]
    assert manifest["sets"]["test"] == old["sets"]["test"]
    fresh = manifest["sets"]["development"] + manifest["sets"]["confirmation"]
    assert len(set(fresh)) == 96
    assert all(50000000 <= s < 60000000 for s in fresh)
    assert not set(fresh) & set().union(*(set(v) for v in old["sets"].values()))
    jobs = [[m + "_" + s, {"method": m}] for m in METHODS for s in "ab"]
    write_json(tmp_path / "stage1/jobs.json", jobs)
    for name, c in jobs:
        score = {"control": 101, "advice": 99, "gated": 100, "sticky": 98}[c["method"]]
        write_json(tmp_path / "stage1/trials" / name / "evaluations/final.json", {"mean_score": score})
    decision = promotion(tmp_path)
    assert decision["selected"] == ["control", "gated"]
    assert decision["winner_minus_control"] == -1
