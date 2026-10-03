from array import array
import json
import math
import random
from types import SimpleNamespace

import pytest

from boardbench.contracts import InvalidAction
from boardbench.environments import TASKS
from boardbench.environments.adapters import SearchAccess
from boardbench.environments.numerical import BoardRL, action_id, decode, encode
from boardbench.runner.engine import run, evaluate
from boardbench.solvers.board import BoardSolver


@pytest.mark.parametrize("task_id", ("micro_tiles", "take_it_easy", "harmonies"))
def test_numeric_masks_direct_rl_consistency(task_id):
    direct, rl = TASKS[task_id].factory(), BoardRL(task_id)
    obs, _ = direct.reset(123)
    assert rl.reset(seed=123)[0] == encode(obs)
    rng = random.Random(42)
    while not obs["terminated"]:
        encoded = encode(obs)
        assert len(encoded["features"]) == rl.observation_shape[0]
        assert all(math.isfinite(x) for x in array('f', encoded["features"]))
        assert sum(encoded["action_mask"]) == len(obs["legal_actions"])
        for action in obs["legal_actions"]:
            assert decode(obs, action_id(task_id, action)) == action
        for bad in (-1, True, 1.5, rl.action_count):
            with pytest.raises(InvalidAction):
                rl.step(bad)
        action = rng.choice(obs["legal_actions"])
        result = direct.step(action)
        got = rl.step(action_id(task_id, action))
        assert got == (encode(result.observation), result.reward, result.terminated, False, result.info)
        obs = result.observation
    assert not any(encode(obs)["action_mask"])
    assert all(math.isfinite(x) for x in encode(obs)["features"])


@pytest.mark.parametrize("task_id", ("micro_tiles", "take_it_easy", "harmonies"))
def test_board_methods_runner_checkpoint_and_frozen_evaluation(task_id, tmp_path):
    for method in ("random", "heuristic", "search"):
        config = {"task": {"id": task_id, "version": TASKS[task_id].spec["version"]},
                  "solver": {"id": "board", "params": {"method": method, "width": 2, "rollouts": 1, "depth": 2}},
                  "episodes": 1, "limits": {"max_action_requests": 256, "max_actions_per_episode": 256, "max_wall_seconds": 120},
                  "capabilities": {"reference_simulator": method == "search"}, "checkpoint": {"every_episodes": 1}}
        out = tmp_path / method
        summary = run(config, out)
        assert summary["status"] == "completed", summary
        evaluation = {"episode_seeds": [711], "per_episode_limits": {"max_action_requests": 256, "max_wall_seconds": 120}}
        ev = evaluate(out / summary["checkpoints"][0], evaluation, tmp_path / (method+"_eval"))
        assert ev["completed_episodes"] == 1 and ev["episode_errors"] == 0
        if method == "search":
            assert ev["resources"]["simulation_steps"] > 0


def test_seed_not_in_policy_api_but_present_in_runner_logs(tmp_path):
    class Inspect(BoardSolver):
        def start_task(self, ctx):
            assert "seed" not in ctx.config
        def end_episode(self, result, ctx):
            assert result.seed is None
    summary = run({"task": {"id": "micro_tiles", "version": "micro_tiles_v1"},
                   "episodes": 1, "checkpoint": {"every_episodes": 0}}, tmp_path / "run",
                  solver_factory=lambda *_: Inspect(123))
    assert summary["status"] == "completed"
    result = json.loads((tmp_path / "run/episodes.jsonl").read_text())
    assert isinstance(result["seed"], int)


def test_search_exact_local_micro_last_move_and_budget():
    env = TASKS["micro_tiles"].factory()
    for _ in range(8):
        env.step(env.legal_actions()[0])
    obs, before = env.observe(), env.save_state()
    used = []
    ctx = SimpleNamespace(reference_simulator=SearchAccess(env.fork, lambda: used.append(1)))
    solver = BoardSolver(3, method="search", width=18, rollouts=2, depth=3, max_simulation_steps=20)
    action = solver.decide(obs, ctx)
    scores = [(env.fork(0).step(a).reward, a) for a in obs["legal_actions"]]
    assert env.fork(0).step(action).reward == max(s for s, _ in scores)
    assert before == env.save_state() and len(used) <= 20
