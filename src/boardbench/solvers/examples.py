from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random
import sys

from boardbench.artifacts.store import read_json, write_json
from boardbench.contracts import Solver


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


class RandomSolver(Solver):
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.episode_steps = 0

    def start_task(self, context):
        self.actions = tuple(context.task_spec["actions"])

    def start_episode(self, observation, context):
        self.episode_steps = 0

    def decide(self, observation, context):
        return self.rng.choice(self.actions)

    def on_transition(self, transition, context):
        self.episode_steps += 1

    def save(self, directory):
        write_json(Path(directory) / "state.json", {"rng": self.rng.getstate()})

    def load(self, directory):
        self.rng.setstate(tuples(read_json(Path(directory) / "state.json")["rng"]))


class ReferenceSearch(RandomSolver):
    def __init__(self, seed=0, rollouts=24, depth=3):
        super().__init__(seed)
        if type(rollouts) is not int or rollouts < 1 or type(depth) is not int or depth < 1:
            raise ValueError("rollouts and depth must be positive integers")
        self.rollouts, self.depth = rollouts, depth

    def start_task(self, context):
        super().start_task(context)
        if not context.config["capabilities"]["reference_simulator"]:
            raise ValueError("reference_search requires reference_simulator capability")

    def decide(self, observation, context):
        access = context.reference_simulator
        if access is None:
            raise ValueError("no reference simulator for current episode")
        score = 0.0
        for _ in range(self.rollouts):
            branch = access.fork(self.rng.getrandbits(64))
            for _ in range(self.depth):
                result = branch.step("DRAW")
                if result.terminated:
                    break
            if not result.terminated:
                result = branch.step("BANK")
            score += result.info["score"]
        draw_value = score / self.rollouts
        context.emit({"draw_estimate": draw_value, "bank_value": observation["pot"]})
        return "DRAW" if draw_value > observation["pot"] else "BANK"


class CollaborationDemo(RandomSolver):
    """Hand-written integration fixture, not an autonomous developer or strong policy."""
    def __init__(self, seed=0, update_every=2):
        super().__init__(seed)
        if type(update_every) is not int or update_every < 1:
            raise ValueError("update_every must be a positive integer")
        self.update_every = update_every
        self.module = {"version": 0, "mean_score": 0.5, "samples": 0}
        self.history = []

    @property
    def version(self):
        return f"module-{self.module['version']}"

    def predict(self, observation):
        return observation["pot"] - (1.0 + self.module["mean_score"])

    def start_episode(self, observation, context):
        super().start_episode(observation, context)
        self.consulted_this_episode = False
        context.emit({"module": deepcopy(self.module), "history_size": len(self.history),
                      "episode_steps": self.episode_steps})

    def decide(self, observation, context):
        margin = self.predict(observation)
        action = "BANK" if margin >= 0 else "DRAW"
        path = "local"
        if not self.consulted_this_episode:
            self.consulted_this_episode = True
            result = context.consult({"purpose": "action", "observation": observation})
            if result.status == "success" and result.text in self.actions:
                action = result.text
                path = "mock_consult"
        context.emit({"path": path, "module_version": self.version,
                      "prediction": margin, "module_value": self.module["mean_score"]})
        return action

    def end_episode(self, result, context):
        if not context.allow_learning or context.closing or context.expired() or result.score is None:
            return
        self.history.append(result.score)
        if len(self.history) % self.update_every:
            return
        samples = context.workspace / "observations.json"
        output = context.workspace / "module.json"
        write_json(samples, {"scores": self.history, "version": self.module["version"] + 1})
        job = context.run_job([sys.executable, "-m", "boardbench.solvers.train_module",
                               str(samples), str(output)])
        if job.status == "success":
            before = deepcopy(self.module)
            self.module = read_json(output)
            context.emit({"event": "module_update", "before": before,
                          "after": deepcopy(self.module), "module_version": self.version})

    def save(self, directory):
        write_json(Path(directory) / "state.json", {
            "rng": self.rng.getstate(), "module": self.module, "history": self.history})

    def load(self, directory):
        state = read_json(Path(directory) / "state.json")
        self.rng.setstate(tuples(state["rng"]))
        self.module, self.history = state["module"], state["history"]
