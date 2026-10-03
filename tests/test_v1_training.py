import json
from pathlib import Path
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
from boardbench.environments import TASKS
from boardbench.environments.numerical import ACTION_COUNTS
from boardbench.solvers.neural import NeuralSolver
from boardbench.training import train, parameter_hash


@pytest.mark.parametrize("task_id", ("micro_tiles", "take_it_easy", "harmonies"))
def test_real_ppo_updates_and_new_process_weight_restore(task_id, tmp_path):
    config = {"task_id": task_id, "training_seed": 12, "episodes": 4, "batch_episodes": 2,
              "hidden": 32, "epochs": 1, "minibatch_size": 128, "threads": 1, "device": "cpu",
              "max_wall_seconds": 120, "shaping": "score_delta"}
    out = tmp_path / task_id
    summary = train(config, out)
    assert summary["status"] == "completed" and summary["parameter_changed"]
    assert len(summary["checkpoints"]) == 3
    assert len({c["parameter_sha256"] for c in summary["checkpoints"]}) == 3
    artifact = out / summary["checkpoints"][-1]["path"]
    solver = NeuralSolver(0, task_id, hidden=32)
    solver.load_weights(artifact)
    assert parameter_hash(solver.model) == summary["checkpoints"][-1]["parameter_sha256"]
    script = """
import json, sys, torch
from boardbench.environments import TASKS
from boardbench.solvers.neural import NeuralSolver
torch.set_num_threads(1)
solver=NeuralSolver(0, sys.argv[1], hidden=32)
solver.load_weights(sys.argv[2])
env=TASKS[sys.argv[1]].factory()
obs,_=env.reset(941)
actions=[]
while not obs['terminated']:
    action=solver.decide(obs,None)
    actions.append(action)
    obs=env.step(action).observation
print(json.dumps({'score':obs['score'],'actions':actions}))
"""
    a = subprocess.check_output([sys.executable, "-c", script, task_id, str(artifact)], text=True)
    b = subprocess.check_output([sys.executable, "-c", script, task_id, str(artifact)], text=True)
    assert json.loads(a) == json.loads(b)
    with pytest.raises(ValueError):
        solver.model(torch.zeros(1, solver.input_size), torch.zeros(1, ACTION_COUNTS[task_id], dtype=torch.bool))


def test_ppo_rejects_overlapping_splits(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        train({"task_id": "micro_tiles", "training_seed": 12, "episodes": 4, "batch_episodes": 2,
               "validation_seeds": [100000], "test_seeds": [300000]}, tmp_path / "overlap")
