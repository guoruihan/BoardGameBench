"""Behavioral reproductions of R1–R3 from BoardBench_V0_Review.md."""
from array import array
from dataclasses import replace
from enum import IntEnum
import json
import math
from pathlib import Path

import pytest

from boardbench.artifacts.store import file_hashes, read_json, validate_checkpoint
from boardbench.environments import TASKS
from boardbench.environments.adapters import RiskRL
from boardbench.runner.engine import evaluate, run
from boardbench.solvers.examples import CollaborationDemo


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def checkpoint(tmp_path):
    out = tmp_path / "parent"
    result = run({"solver": {"id": "collaboration_demo"}, "episodes": 2}, out)
    assert result["status"] == "completed"
    return out / result["checkpoints"][-1]


@pytest.mark.parametrize("mode", ["evaluation", "adaptation"])
def test_r1_load_cache_is_private_and_alive_until_episode_end(tmp_path, checkpoint, monkeypatch, mode):
    before = file_hashes(checkpoint.parent.parent)
    seen, paths, decisions, endings = [], [], [], []
    original_load = CollaborationDemo.load
    original_decide = CollaborationDemo.decide
    original_end = CollaborationDemo.end_episode

    def load(self, directory):
        directory = Path(directory)
        cache = directory / "compiled_cache.bin"
        seen.append(cache.exists())
        paths.append(directory)
        original_load(self, directory)
        self.disk_model = directory / "state.json"
        self.cache = cache
        cache.write_bytes(b"local compiled model")

    def decide(self, obs, context):
        assert self.disk_model.is_file() and self.cache.read_bytes() == b"local compiled model"
        decisions.append(context.episode)
        return original_decide(self, obs, context)

    def end_episode(self, result, context):
        assert self.disk_model.is_file() and self.cache.is_file()
        endings.append(context.episode)
        original_end(self, result, context)

    monkeypatch.setattr(CollaborationDemo, "load", load)
    monkeypatch.setattr(CollaborationDemo, "decide", decide)
    monkeypatch.setattr(CollaborationDemo, "end_episode", end_episode)
    out = tmp_path / mode
    if mode == "evaluation":
        result = evaluate(checkpoint, {"episode_seeds": [123, 123]}, out)
        assert seen == [False, False]
        assert paths[0] != paths[1]
        assert all(not path.exists() for path in paths)
        assert list((out / "workspace").iterdir()) == []
    else:
        result = run(read_json(checkpoint / "manifest.json")["config"], out, checkpoint=checkpoint)
        assert seen == [False]
        assert paths[0].is_relative_to(out) and paths[0].is_dir()
    assert result["status"] == "completed", result
    assert set(decisions) == {1, 2} and endings == [1, 2]
    assert all(path != checkpoint / "solver" for path in paths)
    assert before == file_hashes(checkpoint.parent.parent)
    validate_checkpoint(checkpoint)


@pytest.mark.parametrize("mode", ["adaptation", "evaluation"])
def test_r2_reset_failure_is_an_unstarted_failed_attempt(tmp_path, checkpoint, monkeypatch, mode):
    callbacks = []

    class ResetFailure:
        def reset(self, seed):
            raise RuntimeError("reset failed")

    monkeypatch.setitem(TASKS, "risk_collect", replace(TASKS["risk_collect"], factory=ResetFailure))
    monkeypatch.setattr(CollaborationDemo, "start_episode", lambda *args: callbacks.append("start"))
    monkeypatch.setattr(CollaborationDemo, "end_episode", lambda *args: callbacks.append("end"))
    out = tmp_path / mode
    if mode == "adaptation":
        summary = run({"solver": {"id": "collaboration_demo"}, "episodes": 2}, out)
        attempted = 1  # stop the adaptation run after a failed attempt
    else:
        summary = evaluate(checkpoint, {"episode_seeds": [1, 2]}, out)
        attempted = 2  # independent evaluation attempts retain independent results
    assert summary["status"] == "failed"
    assert summary["attempted_episodes"] == summary["failed_episodes"] == attempted
    assert summary["started_episodes"] == 0 and summary["unstarted_episodes"] == 2
    assert summary["failure_reasons"] == {"reset_error": attempted}
    episodes = records(out / "episodes.jsonl")
    assert len(episodes) == attempted
    assert all(e["started"] is False and e["score"] is None and e["reason"] == "reset_error" for e in episodes)
    assert all("RuntimeError: reset failed" == e["error"] for e in episodes)
    assert all(e["resources"]["environment_steps"] == e["resources"]["action_requests"] == 0 for e in episodes)
    assert callbacks == []
    events = records(out / "events.jsonl")
    assert not any(e["kind"] in {"episode_start", "transition"} for e in events)
    assert sum(e["kind"] == "episode_initialization_failed" for e in events) == attempted


def test_r3_rl_vector_is_finite_and_terminal_flag_masks_score():
    for seed in range(10):
        rl = RiskRL()
        obs, _ = rl.reset(seed=seed)
        assert len(obs) == 8 and all(math.isfinite(x) for x in array("f", obs))
        assert obs[-2:] == (0, 0)
        for action in (0, 0, 1):
            obs, _, terminated, _, info = rl.step(action)
            assert len(obs) == 8 and all(math.isfinite(x) for x in array("f", obs))
            if terminated:
                assert obs[-2:] == (1, info["score"])
                break
            assert obs[-2:] == (0, 0)


def test_r3_integral_actions_supported_without_accepting_bool_or_float():
    from boardbench.contracts import InvalidAction
    class Action(IntEnum):
        DRAW = 0
        BANK = 1
    rl = RiskRL()
    rl.reset(seed=123)
    assert not rl.step(Action.DRAW)[2]
    assert rl.step(Action.BANK)[2]
    for action in (True, False, 0.0, 1.0, -1, 2, "0"):
        rl.reset(seed=123)
        before, rng = rl.engine.observe(), rl.engine._rng.getstate()
        with pytest.raises(InvalidAction):
            rl.step(action)
        assert rl.engine.observe() == before and rl.engine._rng.getstate() == rng
