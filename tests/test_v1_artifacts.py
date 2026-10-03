import json
import shutil

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
from boardbench.artifacts.store import file_hashes, read_json
from boardbench.benchmark import validate, frozen_test
from boardbench.training import train
from boardbench.ui.server import PlaySession


def test_validation_export_common_runner_and_moved_ui_policy(tmp_path):
    task = "micro_tiles"
    training = tmp_path / "train"
    train({"task_id": task, "training_seed": 17, "episodes": 4, "batch_episodes": 2,
           "epochs": 1, "threads": 1, "validation_seeds": [222, 223], "test_seeds": [333, 334]}, training)
    selection = validate(task, training, tmp_path / "validation")
    assert not selection["test_used"] and selection["selection_split"] == "validation"
    assert len(selection["selected"]) == 5
    policies = tmp_path / "policies"
    result = frozen_test(tmp_path / "validation/selection.json", tmp_path / "test", policies)
    assert all(r["completed"] == 2 and r["illegal_actions"] == 0 for r in result.values())
    assert all(r["decision_p95_seconds"] >= 0 for r in result.values())
    copied = tmp_path / "moved" / "policies"
    shutil.copytree(policies, copied)
    before = file_hashes(copied)
    session = PlaySession({"task": {"id": task, "version": "micro_tiles_v1"},
                           "policy_directory": str(copied)})
    trained_id = next(c["id"] for c in selection["selected"] if c["id"].startswith("rl_trained"))
    while not session.observation["terminated"]:
        session.step(session.hint(trained_id)["action"])
    assert session.observation["score"] is not None
    assert session.registry.get(trained_id)[1] == 0.  # Model stays resident.
    session.registry.close()
    assert before == file_hashes(copied)
