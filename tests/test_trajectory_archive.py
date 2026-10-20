from copy import deepcopy
import gzip
import json
import shutil

import pytest

from boardbench.artifacts.store import read_json, write_json
from boardbench.artifacts.trajectories import (
    archive, read_train_episode, sha, verify_archive,
)
from boardbench.v121.registry import make_splits
from boardbench.v125.episode import run_episode


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    root = tmp_path_factory.mktemp("episode_source")
    splits = make_splits(train_count=16)
    job = {"id": "expert", "method": "terminal", "multiplier": 1,
           "environment_seed": splits["sets"]["train"][0], "search_seed": 11,
           "params": {"mode": "timed", "continuation": "habitat_fast", "seconds": .005}}
    result = run_episode(job, root / "episodes" / job["id"])
    assert result["status"] == "completed"
    write_json(root / "plan.json", {"split": "train_calibration", "split_sha256": splits["sha256"]})
    write_json(root / "splits.json", splits)
    write_json(root / "status.json", {"phase": "completed", "active_count": 0})
    write_json(root / "allocation.json", {"source_id": "test-source"})
    return root


def copy_run(example, tmp_path):
    run = tmp_path / "source"
    shutil.copytree(example, run)
    return run


def only_entry(root):
    return next((root / "entries").glob("*.json")).stem


def update_job_and_result(run, **changes):
    for name in ("job", "result"):
        path = run / "episodes" / "expert" / f"{name}.json"
        row = read_json(path)
        row.update(deepcopy(changes))
        write_json(path, row)


def change_rows(run, transform):
    path = run / "episodes" / "expert" / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    transform(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_lossless_resumable_and_replay(example, tmp_path):
    out = tmp_path / "archive"
    summary = archive(example, out)
    assert summary["entries"] == 1 and summary["train_candidates_pending_replay"] == 1
    assert summary["skipped"] == []
    entry_id = only_entry(out)
    entry = read_json(out / "entries" / f"{entry_id}.json")
    assert gzip.decompress((out / "blobs" / f"{entry['sha256']}.jsonl.gz").read_bytes()) == (
        example / "episodes/expert/decisions.jsonl").read_bytes()
    assert archive(example, out)["new_entries"] == 0
    assert verify_archive(out)["verified_entries"] == 1
    rows = read_train_episode(out, entry_id)
    assert len(rows) == entry["result"]["real_steps"]
    assert rows[-1]["terminated"] and not any(row["truncated"] for row in rows)
    assert sum(row["environment_reward"] for row in rows) == entry["result"]["score"]
    assert sum(row["raw_score_delta"] for row in rows) == entry["result"]["score"]
    assert rows[0]["score_delta_return_to_go"] == entry["result"]["score"]
    assert all(row["behavior_log_probability"] is None for row in rows)
    assert all(row["next_observation"] == rows[i + 1]["observation"] for i, row in enumerate(rows[:-1]))
    with pytest.raises(ValueError, match="validation"):
        read_train_episode(out, entry_id, excluded_seeds=[entry["job"]["environment_seed"]])


@pytest.mark.parametrize("split", ["development", "confirmation", "test", "unknown"])
def test_nontrain_archive_never_trains(example, tmp_path, split):
    run = copy_run(example, tmp_path)
    plan = read_json(run / "plan.json")
    plan["split"] = split
    write_json(run / "plan.json", plan)
    if split != "unknown":
        seed = read_json(run / "splits.json")["sets"][split][0]
        update_job_and_result(run, environment_seed=seed)
    out = tmp_path / "archive"
    summary = archive(run, out)
    assert summary["entries"] == 1 and summary["train_candidates_pending_replay"] == 0
    with pytest.raises(ValueError, match="train-split"):
        read_train_episode(out, only_entry(out))


def test_forged_split_and_active_campaign(example, tmp_path):
    run = copy_run(example, tmp_path)
    splits = read_json(run / "splits.json")
    splits["sets"]["test"].append(splits["sets"]["train"][0])
    write_json(run / "splits.json", splits)
    out = tmp_path / "archive"
    assert archive(run, out)["train_candidates_pending_replay"] == 0
    with pytest.raises(ValueError, match="train-split"):
        read_train_episode(out, only_entry(out))
    write_json(run / "status.json", {"phase": "running", "active_count": 1})
    summary = archive(run, tmp_path / "active_archive")
    assert summary["entries"] == 0 and summary["skipped"]


def test_student_actions_are_not_teacher_transitions(example, tmp_path):
    run = copy_run(example, tmp_path)
    params = read_json(run / "episodes/expert/job.json")["params"]
    update_job_and_result(run, params={**params, "continuation": "dagger"})
    def labels(rows):
        for row in rows:
            alternatives = [a for a in row["observation_before"]["legal_actions"] if a != row["action"]]
            row["search"]["expert_action"] = alternatives[0] if alternatives else row["action"]
            row["search"]["labels_only"] = True
    change_rows(run, labels)
    out = tmp_path / "archive"
    assert archive(run, out)["by_behavior"] == {"student_with_expert_labels": 1}
    with pytest.raises(ValueError, match="not an expert"):
        read_train_episode(out, only_entry(out))
    rows = read_train_episode(out, only_entry(out), expert_only=False)
    assert any(row["action"] != row["expert_action"] for row in rows)
    assert all(row["action"] in row["observation"]["legal_actions"] for row in rows)


@pytest.mark.parametrize("damage", ["missing_last", "bad_next", "bad_reward", "bad_number", "failed"])
def test_invalid_episodes_preserved_but_rejected(example, tmp_path, damage):
    run = copy_run(example, tmp_path)
    def corrupt(rows):
        if damage == "missing_last":
            rows.pop()
        elif damage == "bad_next":
            rows[0]["observation"]["score_breakdown"]["total"] += 1
        elif damage == "bad_reward":
            rows[0]["reward"] += 1
        elif damage == "bad_number":
            rows[0]["decision"] = 4
    change_rows(run, corrupt)
    if damage == "failed":
        path = run / "episodes/expert/result.json"
        write_json(path, {**read_json(path), "status": "failed", "natural_terminal": False})
    out = tmp_path / "archive"
    assert archive(run, out)["entries"] == 1
    with pytest.raises(ValueError):
        read_train_episode(out, only_entry(out))
    assert verify_archive(out)["verified_entries"] == 1  # Byte verification is NOT replay validation.


def test_old_missing_pre_observation_recovered_only_via_replay(example, tmp_path):
    run = copy_run(example, tmp_path)
    def old_format(rows):
        for row in rows:
            for key in ("observation_before", "raw_score_delta", "reward", "terminated", "truncated"):
                row.pop(key)
    change_rows(run, old_format)
    out = tmp_path / "archive"
    archive(run, out)
    rows = read_train_episode(out, only_entry(out))
    assert rows[-1]["terminated"]


def test_byte_corruption_and_metadata_corruption(example, tmp_path):
    out = tmp_path / "archive"
    archive(example, out)
    entry_id = only_entry(out)
    entry = read_json(out / "entries" / f"{entry_id}.json")
    blob = out / "blobs" / f"{entry['sha256']}.jsonl.gz"
    blob.write_bytes(gzip.compress(b"{}\n"))
    with pytest.raises(ValueError, match="hash"):
        read_train_episode(out, entry_id)
    with pytest.raises(ValueError, match="hash"):
        verify_archive(out)
    with pytest.raises(ValueError, match="corrupt existing"):
        archive(example, out)
    write_json(out / "entries" / f"{entry_id}.json", {**entry, "split": "development"})
    with pytest.raises(ValueError, match="invalid episode"):
        read_train_episode(out, entry_id)


def test_legacy_raw_only_preserved_without_private_state_admission(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    raw = b'{"endgames": [{"private_state": "evidence only"}]}\n'
    (source / "semantic_trajectories.jsonl").write_bytes(raw)
    out = tmp_path / "archive"
    summary = archive(source, out)
    assert summary["entries"] == 1 and summary["train_candidates_pending_replay"] == 0
    entry = read_json(out / "entries" / f"{only_entry(out)}.json")
    assert entry["legacy_warning"] and entry["sha256"] == sha(raw)
    with pytest.raises(ValueError, match="invalid episode"):
        read_train_episode(out, only_entry(out))


def test_later_revision_retains_failed_attempt(example, tmp_path):
    run = copy_run(example, tmp_path)
    result_path = run / "episodes/expert/result.json"
    original = read_json(result_path)
    write_json(result_path, {**original, "status": "failed", "natural_terminal": False})
    out = tmp_path / "archive"
    assert archive(run, out)["entries"] == 1
    write_json(result_path, original)
    summary = archive(run, out)
    assert summary["entries"] == 2 and summary["unique_blobs"] == 1
    assert summary["train_candidates_pending_replay"] == 1


def test_parallel_duplicate_blobs_and_snapshot_exclusion(example, tmp_path):
    source = tmp_path / "source"
    for i in range(8):
        shutil.copytree(example, source / f"run_{i}")
    shutil.copytree(example, source / "source" / "old_snapshot")
    out = tmp_path / "archive"
    summary = archive(source, out, workers=8)
    assert summary["entries"] == 8 and summary["unique_blobs"] == 1
    assert summary["skipped"] == []
    assert verify_archive(out)["verified_entries"] == 8
    assert archive(source, out, workers=8)["new_entries"] == 0
