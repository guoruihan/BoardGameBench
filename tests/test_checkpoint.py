from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from boardbench.artifacts.store import (file_hashes, read_json, source_id, validate_checkpoint, write_json)
from boardbench.runner.config import evaluation_config
from boardbench.runner.engine import evaluate, run


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


@pytest.fixture
def checkpoint(tmp_path):
    out = tmp_path / "adaptation"
    s = run({"solver": {"id": "collaboration_demo"}, "episodes": 2}, out)
    assert s["status"] == "completed"
    return out / s["checkpoints"][-1]


def test_new_process_new_cwd_repeatable_and_original_immutable(tmp_path, checkpoint):
    original = file_hashes(checkpoint.parent.parent)
    copied = tmp_path / "relocated" / "checkpoint"
    shutil.copytree(checkpoint, copied)
    config = tmp_path / "eval.json"
    write_json(config, {"episode_seeds": [1001, 1002, 1001],
                        "per_episode_limits": {"max_model_calls": 1}})
    elsewhere = tmp_path / "unrelated_cwd"
    elsewhere.mkdir()
    outputs = []
    for i in range(2):
        out = tmp_path / f"eval_{i}"
        cmd = [sys.executable, "-m", "boardbench", "evaluate", "--checkpoint", str(copied),
               "--config", str(config), "--out", str(out)]
        p = subprocess.run(cmd, cwd=elsewhere, capture_output=True, text=True, timeout=30)
        assert p.returncode == 0, p.stderr + p.stdout
        events = records(out / "events.jsonl")
        steps = [e for e in events if e["kind"] == "transition"]
        outputs.append([(e["episode"], e["action"], e["result"]) for e in steps])
        assert [(e["action"], e["result"]) for e in steps if e["episode"] == 1] == [
            (e["action"], e["result"]) for e in steps if e["episode"] == 3]
        starts = [e for e in events if e["kind"] == "episode_start"]
        assert all(e["observation"]["pot"] == 0 and e["observation"]["steps"] == 0 for e in starts)
        assert not any(e["kind"] == "job_start" for e in events)
        assert not any(e.get("metrics", {}).get("event") == "module_update" for e in events)
        summary = read_json(out / "summary.json")
        assert summary["resources"]["model_calls"] == 3
        assert summary["historical_adaptation"]["resources"]["model_calls"] == 1
        histories = [e["metrics"]["history_size"] for e in events if "history_size" in e.get("metrics", {})]
        assert histories == [2, 2, 2]
        assert list((out / "workspace").iterdir()) == []
    assert outputs[0] == outputs[1]
    assert file_hashes(checkpoint.parent.parent) == original
    assert file_hashes(copied) == file_hashes(checkpoint)


def test_integrity_and_source_version_mismatch(checkpoint, monkeypatch):
    validate_checkpoint(checkpoint)
    monkeypatch.setattr("boardbench.artifacts.store.source_id", lambda root=None: "changed" if root is None else source_id(root))
    with pytest.raises(ValueError, match="source version mismatch"):
        validate_checkpoint(checkpoint)
    monkeypatch.undo()
    state = read_json(checkpoint / "solver/state.json")
    state["module"]["mean_score"] = 999
    write_json(checkpoint / "solver/state.json", state)
    with pytest.raises(ValueError, match="hashes"):
        validate_checkpoint(checkpoint)


def test_evaluation_limits_and_restricted_overrides(tmp_path, checkpoint):
    manifest = validate_checkpoint(checkpoint)
    for bad in ({"task": {}}, {"allow_learning": True}, {"episode_seeds": []},
                {"episode_seeds": [True]}, {"solver": {"id": "random"}}):
        with pytest.raises(ValueError):
            evaluation_config({"episode_seeds": [1], **bad}, manifest)
    out = tmp_path / "short_eval"
    s = evaluate(checkpoint, {"episode_seeds": [1, 1],
                 "per_episode_limits": {"max_action_requests": 1}}, out)
    assert s["started_episodes"] == 2
    assert s["resources"]["action_requests"] == 2
    assert s["resources"]["model_calls"] == 0


def test_resume_is_new_run_with_saved_learning(tmp_path, checkpoint):
    manifest = validate_checkpoint(checkpoint)
    c = deepcopy(manifest["config"])
    c["episodes"] = 2
    out = tmp_path / "resumed"
    s = run(c, out, checkpoint=checkpoint)
    assert s["status"] == "completed"
    loaded = read_json(out / s["checkpoints"][0] / "solver/state.json")
    assert len(loaded["history"]) == 4
    assert s["historical_adaptation"]["progress"]["completed_episodes"] == 2
    resumed = read_json(out / s["checkpoints"][0] / "manifest.json")
    assert resumed["cumulative_adaptation_resources"]["model_calls"] == 2


def test_evaluation_initialization_timeout_does_not_count_as_started(tmp_path, checkpoint):
    s = evaluate(checkpoint, {"episode_seeds": [1, 2],
                 "per_episode_limits": {"max_wall_seconds": 0}}, tmp_path / "no_time")
    assert s["attempted_episodes"] == 2
    assert s["started_episodes"] == 0 and s["unstarted_episodes"] == 2
    assert s["resources"]["environment_steps"] == 0


def test_new_process_rejects_actual_different_source(tmp_path, checkpoint):
    altered = tmp_path / "altered_source"
    shutil.copytree(checkpoint / "source", altered)
    init = altered / "src/boardbench/__init__.py"
    init.write_text(init.read_text() + "\n# Deliberately different source version.\n")
    eval_config = tmp_path / "eval_config.json"
    write_json(eval_config, {"episode_seeds": [1]})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(altered / "src")
    p = subprocess.run([sys.executable, "-m", "boardbench", "evaluate", "--checkpoint", str(checkpoint),
                        "--config", str(eval_config), "--out", str(tmp_path / "must_not_start")],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert p.returncode == 2
    assert "source version mismatch" in p.stderr
    assert not (tmp_path / "must_not_start").exists()


def test_random_solver_checkpoint_restores_exact_rng(tmp_path):
    from boardbench.solvers.examples import RandomSolver
    a = RandomSolver(seed=7)
    a.actions = ("DRAW", "BANK")
    for _ in range(13):
        a.decide({}, None)
    a.save(tmp_path)
    expected = [a.decide({}, None) for _ in range(50)]
    b = RandomSolver(seed=999)
    b.actions = a.actions
    b.load(tmp_path)
    assert [b.decide({}, None) for _ in range(50)] == expected
