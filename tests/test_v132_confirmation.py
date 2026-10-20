import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "confirmation", Path(__file__).resolve().parents[1] / "scripts/confirm_v132_models.py")
confirmation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(confirmation)


def result():
    return {"games": [{"environment_seed": s, "score": 100., "earned_score": 100.,
                       "initial_score": 0., "natural_terminal": True, "curriculum_reset": False}
                      for s in (1, 2)], "mean_score": 100.}


def test_matched_normal_start_games():
    confirmation.validate_games(result(), [1, 2])


@pytest.mark.parametrize("field,value", [
    ("environment_seed", 2), ("natural_terminal", False), ("curriculum_reset", True),
    ("initial_score", 20.), ("earned_score", 80.),
])
def test_reject_invalid_games(field, value):
    data = result()
    data["games"][0][field] = value
    with pytest.raises(ValueError):
        confirmation.validate_games(data, [1, 2])


def test_reject_wrong_aggregation():
    data = result()
    data["mean_score"] = 101.
    with pytest.raises(ValueError):
        confirmation.validate_games(data, [1, 2])


def test_export_sidecar_cannot_overwrite_games(tmp_path):
    checkpoint = tmp_path / "parent.pt"
    confirmation.torch.save({"source_id": "fixture",
                             "model": {"weight": confirmation.torch.ones(1, dtype=confirmation.torch.float64)}},
                            checkpoint)
    confirmation.write_json(checkpoint.with_suffix(".json"),
                            {"source_id": "fixture", "sha256": confirmation.digest(checkpoint)})
    metrics = tmp_path / "primary_round08_final.json"
    confirmation.write_json(metrics, result())
    confirmation.export_primary(tmp_path, {"checkpoint": str(checkpoint)})
    assert confirmation.read_json(metrics) == result()
    assert (tmp_path / "model/primary_round08_final.json").exists()


def test_fresh_panel_deterministic_unique_and_disjoint():
    excluded = confirmation.fresh_panel({}, count=20)
    sets = {"train": excluded[:5], "development": excluded[5:10],
            "confirmation": excluded[10:15], "test": excluded[15:]}
    values = confirmation.fresh_panel(sets)
    assert values == confirmation.fresh_panel(sets)
    assert len(values) == len(set(values)) == 512
    assert not set(values) & set(excluded)
    assert all(20000000 <= v < 30000000 for v in values)


@pytest.mark.parametrize("count", [0, -1, 10000001])
def test_reject_invalid_fresh_panel_size(count):
    with pytest.raises(ValueError):
        confirmation.fresh_panel({}, count=count)


def test_panel_aggregation_weights_games_not_batches():
    class Trainer:
        def evaluate(self, seeds):
            return {
                "games": [{"environment_seed": s, "score": float(s), "earned_score": float(s),
                           "animal_score": 0., "completed_cards": 0., "initial_score": 0.,
                           "natural_terminal": True, "curriculum_reset": False} for s in seeds],
                "mean_score": sum(seeds) / len(seeds), "actions": len(seeds),
                "environment_seconds": 1., "model_seconds": .1,
            }
    metrics = confirmation.evaluate_panel(Trainer(), [1, 2, 9], batch_size=2)
    assert metrics["mean_score"] == 4.
    assert metrics["actions"] == 3
    assert metrics["environment_seconds"] == 2.
    assert [g["environment_seed"] for g in metrics["games"]] == [1, 2, 9]
    confirmation.validate_games(metrics, [1, 2, 9])
