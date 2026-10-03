"""Risk collection rules, independent of solvers, UI and global RNG."""
from __future__ import annotations

import random

from boardbench.contracts import InvalidAction, StepResult


class RiskCollect:
    def __init__(self):
        self.reset(0)

    def reset(self, seed: int) -> tuple[dict, dict]:
        self._rng = random.Random(seed)
        self._counts = {"1": 2, "2": 2, "3": 2, "fail": 2}
        self._pot = 0
        self._steps = 0
        self._terminated = False
        self._score = None
        return self.observe(), {"task": "risk_collect", "version": "1"}

    def observe(self) -> dict:
        return {"pot": self._pot, "remaining": dict(self._counts),
                "steps": self._steps, "terminated": self._terminated,
                "score": self._score}

    def step(self, action: str) -> StepResult:
        if self._terminated:
            raise InvalidAction("episode already terminated; explicitly reset")
        if action not in ("DRAW", "BANK"):
            raise InvalidAction(f"invalid action: {action!r}")
        self._steps += 1
        info = {"action": action}
        reward = 0
        if action == "BANK":
            self._score = self._pot
            reward = self._pot
            self._terminated = True
            info["reason"] = "bank"
        else:
            index = self._rng.randrange(sum(self._counts.values()))
            for card, count in self._counts.items():
                if index < count:
                    break
                index -= count
            self._counts[card] -= 1
            info["card"] = card
            if card == "fail":
                self._terminated = True
                self._score = 0
                info["reason"] = "failure_card"
            else:
                self._pot += int(card)
        if self._terminated:
            info["score"] = self._score
        return StepResult(self.observe(), reward, self._terminated, info=info)

    def fork(self, sim_seed: int) -> RiskCollect:
        branch = RiskCollect()
        branch._counts = dict(self._counts)
        branch._pot = self._pot
        branch._steps = self._steps
        branch._terminated = self._terminated
        branch._score = self._score
        branch._rng = random.Random(sim_seed)
        return branch
