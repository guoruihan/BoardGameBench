from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest

from boardbench.artifacts.store import ArtifactStore, read_json
from boardbench.contracts import ConsultResult, Solver
from boardbench.environments import TASKS, Task
from boardbench.environments.risk import RiskCollect
from boardbench.runner.config import normalize
from boardbench.runner.context import Context
from boardbench.runner.engine import run, run_episode
from boardbench.solvers.examples import CollaborationDemo


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def config(**changes):
    return normalize({"checkpoint": {"every_episodes": 0}, **changes})


class BankSolver(Solver):
    def __init__(self):
        self.events = []

    def start_task(self, ctx):
        self.events.append("task")

    def start_episode(self, obs, ctx):
        self.events.append("start")
        assert obs["steps"] == 0

    def decide(self, obs, ctx):
        obs["pot"] = 999  # no mutable observation alias into real state/logs
        return "BANK"

    def on_transition(self, t, ctx):
        assert t.result.terminated
        self.events.append("transition")

    def end_episode(self, result, ctx):
        self.events.append("end")


def test_resident_lifecycle_and_terminal_feedback(tmp_path):
    solver = BankSolver()
    s = run(config(episodes=3), tmp_path / "run", solver_factory=lambda *_: solver)
    assert s["completed_episodes"] == 3
    assert solver.events == ["task"] + ["start", "transition", "end"] * 3
    assert s["requested_episode_score_sum"] == 0
    transitions = [e for e in records(tmp_path / "run/events.jsonl") if e["kind"] == "transition"]
    assert all(e["observation"]["pot"] == 0 for e in transitions)


@pytest.mark.parametrize("solver", ["random", "reference_search", "collaboration_demo"])
def test_all_solvers_same_loop_and_reproducible(tmp_path, solver):
    c = config(solver={"id": solver}, capabilities={"reference_simulator": solver == "reference_search"})
    outputs = []
    for i in range(2):
        out = tmp_path / str(i)
        summary = run(c, out)
        assert summary["status"] == "completed"
        events = records(out / "events.jsonl")
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        steps = [e for e in events if e["kind"] == "transition"]
        assert len(steps) == summary["resources"]["environment_steps"]
        assert sum(e["result"]["reward"] for e in steps) == summary["requested_episode_score_sum"]
        assert len([e for e in events if e["kind"] == "action_request"]) == summary["resources"]["action_requests"]
        assert sum(e["elapsed_seconds"] for e in events if e["kind"] == "decision_end") == summary["resources"]["decision_seconds"]
        outputs.append([(e["episode"], e["action"], e["result"]) for e in steps])
        if solver != "collaboration_demo":
            assert summary["resources"]["model_calls"] == 0
        if solver == "reference_search":
            assert summary["resources"]["simulation_steps"] > len(steps)
    assert outputs[0] == outputs[1]


def test_collaboration_updates_consumed_and_budget_exhaustion_continues(tmp_path):
    c = config(solver={"id": "collaboration_demo"}, checkpoint={"every_episodes": 2})
    result = run(c, tmp_path / "run")
    assert result["completed_episodes"] == 6
    assert result["resources"]["model_calls"] == 1
    assert result["resources"]["model_rejections"] == 5
    assert result["resources"]["jobs"] == 3
    assert result["resources"]["input_tokens"] is None
    assert len(result["checkpoints"]) == 3
    events = records(tmp_path / "run/events.jsonl")
    reports = [e for e in events if e["kind"] == "solver_report"]
    updates = [e for e in reports if e["metrics"].get("event") == "module_update"]
    assert updates[0]["metrics"]["before"]["mean_score"] != updates[0]["metrics"]["after"]["mean_score"]
    assert any(e["seq"] > updates[0]["seq"] and e["metrics"].get("module_version") == "module-1"
               and "prediction" in e["metrics"] for e in reports)
    starts = [e["metrics"] for e in reports if "history_size" in e["metrics"]]
    assert [s["history_size"] for s in starts] == list(range(6))
    assert all(s["episode_steps"] == 0 for s in starts)
    solver = CollaborationDemo()
    old = solver.predict({"pot": 2})
    solver.load(tmp_path / "run" / result["checkpoints"][0] / "solver")
    assert solver.predict({"pot": 2}) != old


class InvalidSolver(BankSolver):
    def decide(self, obs, ctx):
        return "INVALID"


def test_invalid_actions_count_and_do_not_fake_transitions(tmp_path):
    c = config(episodes=4, limits={"max_action_requests": 3, "max_actions_per_episode": 2})
    summary = run(c, tmp_path / "run", solver_factory=lambda *_: InvalidSolver())
    assert summary["resources"]["action_requests"] == 3
    assert summary["resources"]["environment_steps"] == 0
    assert summary["started_episodes"] == 2
    assert summary["unstarted_episodes"] == 2
    assert summary["truncated_episodes"] == 2
    assert summary["requested_episode_score_mean"] == 0
    episodes = records(tmp_path / "run/episodes.jsonl")
    assert all(e["score"] is None for e in episodes)
    assert episodes[0]["reason"] == "episode_action_limit"
    assert episodes[1]["reason"] == "run_action_limit"


def test_existing_output_and_bad_configs_rejected(tmp_path):
    tmp_path.joinpath("existing").mkdir()
    with pytest.raises(FileExistsError):
        run(config(), tmp_path / "existing")
    for bad in ({"mode": "evaluation"}, {"allow_learning": False}, {"episodes": -1},
                {"episodes": True}, {"limits": {"max_wall_seconds": float("nan")}},
                {"model": {"max_calls": -1}}, {"typo": 3}):
        with pytest.raises(ValueError):
            normalize(bad)


def test_runner_task_independence(tmp_path, monkeypatch):
    class OtherEnv:
        def reset(self, seed):
            return {"color": "blue"}, {}
        def step(self, action):
            from boardbench.contracts import StepResult
            assert action == {"choose": "blue"}
            return StepResult({"color": "green"}, 7, True, info={"score": 7})
    class OtherSolver(Solver):
        def decide(self, obs, context):
            return {"choose": obs["color"]}
    task = Task(OtherEnv, {"id": "other", "version": "1"},
                lambda results, requested: {"custom_aggregate": sum(r["score"] for r in results)})
    monkeypatch.setitem(TASKS, "other", task)
    s = run(config(task={"id": "other"}, episodes=2), tmp_path / "other",
            solver_factory=lambda *_: OtherSolver())
    assert s["custom_aggregate"] == 14
    assert "requested_episode_score_sum" not in s


class Clock:
    value = 0.0
    def __call__(self):
        return self.value


@pytest.fixture
def context(tmp_path):
    c = config()
    store = ArtifactStore(tmp_path / "context", c)
    ctx = Context(store, c, deepcopy(TASKS["risk_collect"].spec), store.root / "workspace")
    yield ctx
    store.close()


def test_model_failure_consumes_quota_and_timeout_clamped(context):
    class FailureClient:
        provider = "test_failure"
        def consult(self, request, timeout):
            assert 0 < timeout <= 0.5
            raise RuntimeError("provider failed")
    clock = Clock()
    context.clock, context.deadline, context.client = clock, 0.5, FailureClient()
    first = context.consult({})
    assert first.status == "failed"
    second = context.consult({})
    assert second.status == "budget_exhausted"
    assert context.resources["model_calls"] == 1
    assert context.resources["model_failures"] == 1
    assert context.resources["model_rejections"] == 1


def test_jobs_success_failure_timeout_and_no_new_jobs_after_deadline(context):
    ok = context.run_job([sys.executable, "-c", "print('training done')"])
    assert ok.status == "success" and ok.returncode == 0
    assert "training done" in (context.store.root / ok.log_path).read_text()
    fail = context.run_job([sys.executable, "-c", "raise SystemExit(3)"])
    assert fail.status == "failed" and fail.returncode == 3
    timeout = context.run_job([sys.executable, "-c", "import time; time.sleep(10)"],
                              {"timeout_seconds": 0.03})
    assert timeout.status == "timeout" and timeout.elapsed_seconds > 0
    missing = context.run_job(["/nonexistent/boardbench-command"])
    assert missing.status == "failed" and missing.error
    count = context.resources["jobs"]
    context.deadline = context.clock() - 1
    assert context.run_job([sys.executable, "-c", "print('forbidden')"]).status == "budget_exhausted"
    assert context.consult({}).status == "budget_exhausted"
    assert context.resources["jobs"] == count


def test_slow_decision_is_not_executed_and_cleanup_once(context):
    clock = Clock()
    context.clock, context.deadline = clock, 1
    class Slow(BankSolver):
        def decide(self, obs, ctx):
            clock.value = 2
            return "BANK"
    solver = Slow()
    r = run_episode(RiskCollect(), solver, context, 1, 0)
    assert r["status"] == "truncated" and r["score"] is None
    assert r["environment_steps"] == 0
    assert r["resources"]["wall_overrun_seconds"] == 1
    assert solver.events == ["start", "end"]


def test_transition_feedback_and_natural_score_survive_time_expiry(context):
    clock = Clock()
    context.clock, context.deadline = clock, 1
    class SlowEnv(RiskCollect):
        def step(self, action):
            result = super().step(action)
            clock.value = 2
            return result
    class Cleanup(BankSolver):
        def on_transition(self, t, ctx):
            super().on_transition(t, ctx)
            assert ctx.consult({}).status == "budget_exhausted"
    solver = Cleanup()
    r = run_episode(SlowEnv(), solver, context, 1, 0)
    assert r["status"] == "completed" and r["score"] == 0
    assert solver.events == ["start", "transition", "end"]


def test_exception_records_partial_run_and_finalizes_once(tmp_path):
    class Broken(BankSolver):
        def decide(self, obs, ctx):
            raise RuntimeError("broken policy")
    solver = Broken()
    s = run(config(), tmp_path / "broken", solver_factory=lambda *_: solver)
    assert s["status"] == "failed" and s["failed_episodes"] == 1
    assert s["unstarted_episodes"] == 5
    assert s["requested_episode_score_mean"] == 0
    assert solver.events == ["task", "start", "end"]


def test_zero_time_budget_preserves_unstarted_count(tmp_path):
    s = run(config(limits={"max_wall_seconds": 0}), tmp_path / "zero")
    assert s["started_episodes"] == 0 and s["unstarted_episodes"] == 6
    assert s["stop_reason"] == "wall_time"


def test_end_episode_failure_keeps_true_score_but_does_not_save(tmp_path):
    class BrokenCleanup(BankSolver):
        def end_episode(self, result, ctx):
            raise RuntimeError("cleanup failed")
    s = run(config(checkpoint={"every_episodes": 1}), tmp_path / "failed_cleanup",
            solver_factory=lambda *_: BrokenCleanup())
    assert s["status"] == "failed"
    assert s["completed_episodes"] == 1
    assert s["checkpoints"] == []
    result = records(tmp_path / "failed_cleanup/episodes.jsonl")[0]
    assert result["score"] == 0
    assert "cleanup failed" in result["error"]


def test_partial_pot_is_not_final_score(tmp_path):
    class Draw(Solver):
        def decide(self, obs, ctx):
            return "DRAW"
    out = tmp_path / "partial"
    s = run(config(episodes=1, seed=123, limits={"max_actions_per_episode": 1}), out,
            solver_factory=lambda *_: Draw())
    assert s["truncated_episodes"] == 1
    assert records(out / "episodes.jsonl")[0]["score"] is None
    step = next(e for e in records(out / "events.jsonl") if e["kind"] == "transition")
    assert step["result"]["observation"]["pot"] > 0
    assert s["requested_episode_score_sum"] == 0


def test_checkpoint_save_error_keeps_results_and_stops(tmp_path):
    class SaveFailure(BankSolver):
        def save(self, path):
            raise OSError("cannot save model")
    s = run(config(checkpoint={"every_episodes": 1}), tmp_path / "save_failed",
            solver_factory=lambda *_: SaveFailure())
    assert s["status"] == "failed" and "cannot save model" in s["error"]
    assert s["completed_episodes"] == 1 and s["unstarted_episodes"] == 5
    assert s["checkpoints"] == []


def test_slow_start_episode_never_decides(context):
    clock = Clock()
    context.clock, context.deadline = clock, 1
    class SlowStart(BankSolver):
        def start_episode(self, obs, ctx):
            self.events.append("start")
            clock.value = 2
        def decide(self, obs, ctx):
            raise AssertionError("must not decide after deadline")
    solver = SlowStart()
    r = run_episode(RiskCollect(), solver, context, 1, 1)
    assert r["status"] == "truncated" and r["reason"] == "wall_time"
    assert solver.events == ["start", "end"]


def test_reference_simulator_requires_explicit_capability(tmp_path):
    s = run(config(solver={"id": "reference_search"}), tmp_path / "not_authorized")
    assert s["status"] == "failed"
    assert "requires reference_simulator" in s["error"]
    assert s["started_episodes"] == 0
