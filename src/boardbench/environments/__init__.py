"""Task definitions and shared real engines."""
from dataclasses import dataclass
from typing import Callable
import statistics

from .risk import RiskCollect
from .micro import MicroTiles
from .take_it_easy import TakeItEasy
from .harmonies import Harmonies


@dataclass(frozen=True)
class Task:
    factory: Callable
    spec: dict
    aggregate: Callable


def risk_aggregate(results: list[dict], requested: int) -> dict:
    total = sum(r["score"] or 0 for r in results)
    return {"requested_episode_score_sum": total,
            "requested_episode_score_mean": total / requested if requested else None,
            "incomplete_score_policy": "risk_collect: unstarted/incomplete count as zero"}


TASKS = {"risk_collect": Task(RiskCollect, {
    "id": "risk_collect", "version": "1", "actions": ["DRAW", "BANK"],
    "initial_counts": {"1": 2, "2": 2, "3": 2, "fail": 2},
    "reward": "terminal BANK pot; otherwise zero",
}, risk_aggregate)}


def board_aggregate(results, requested):
    completed = sum(r["status"] == "completed" for r in results)
    scores = [r["score"] for r in results if r["status"] == "completed"]
    return {"completion_rate": completed / requested if requested else None,
            "score_mean": statistics.mean(scores) if scores else None,
            "score_std": statistics.pstdev(scores) if scores else None,
            "incomplete_score_policy": "incomplete scores are null; score mean uses completed episodes only"}


for engine in (MicroTiles, TakeItEasy, Harmonies):
    TASKS[engine.task_id] = Task(engine, engine.task_spec(), board_aggregate)


def get_task(config: dict) -> Task:
    task = TASKS[config["id"]]
    if config["version"] != task.spec["version"]:
        raise ValueError("unsupported task rules version")
    return task
