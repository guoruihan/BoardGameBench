"""Public contracts. Observations/actions are task-owned JSON-compatible values."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
import hashlib

Observation = Any
Action = Any


def derive_seed(seed: int, *labels: object) -> int:
    payload = ":".join(map(str, (seed, *labels))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class InvalidAction(ValueError):
    """Rejected action: environment state and RNG remain unchanged."""


@dataclass
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool = False
    info: dict = field(default_factory=dict)


@dataclass
class Transition:
    observation: Observation
    action: Action
    result: StepResult


@dataclass
class EpisodeResult:
    episode: int
    seed: int | None
    status: str
    reason: str
    score: float | None
    action_requests: int
    environment_steps: int
    resources: dict = field(default_factory=dict)
    error: str | None = None
    started: bool = True


@dataclass
class ConsultResult:
    status: str
    text: str | None = None
    usage: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    error: str | None = None


@dataclass
class JobResult:
    status: str
    returncode: int | None
    log_path: str | None
    elapsed_seconds: float
    error: str | None = None


class Environment(Protocol):
    def reset(self, seed: int) -> tuple[Observation, dict]: ...
    def step(self, action: Action) -> StepResult: ...


class Solver:
    """Optional lifecycle hooks; parameters and task memory belong to this object."""

    @property
    def version(self) -> str:
        return "1"

    def start_task(self, context) -> None:
        pass

    def start_episode(self, observation, context) -> None:
        pass

    def decide(self, observation, context) -> Action:
        raise NotImplementedError

    def on_transition(self, transition: Transition, context) -> None:
        pass

    def end_episode(self, result: EpisodeResult, context) -> None:
        pass

    def save(self, directory: Path) -> None:
        raise NotImplementedError

    def load(self, directory: Path) -> None:
        raise NotImplementedError
