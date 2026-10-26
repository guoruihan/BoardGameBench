"""Small human-versus-model session using the unchanged two-player engine."""
import hashlib
from pathlib import Path
import secrets

from boardbench.contracts import InvalidAction
from boardbench.environments.harmonies_duo import FastHarmoniesDuo


class DuoPlaySession:
    def __init__(self, config):
        # Keep torch optional for the existing solo browser and benchmark CLI.
        import torch
        from boardbench.v145.policy import SelfplaySolver

        self.task = {"id": FastHarmoniesDuo.task_id, "version": FastHarmoniesDuo.rules_version}
        if config["task"] != self.task:
            raise ValueError("official two-player task/version required")
        checkpoint = Path(config["model_path"])
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != config["model_sha256"]:
            raise ValueError("duo model SHA256 mismatch")
        torch.set_num_threads(1)
        self.solver = SelfplaySolver(device="cpu")
        self.solver.load_weights(checkpoint)
        self.model_label = config.get("model_label", "双人学习策略")
        self.env = FastHarmoniesDuo()
        self.human_seat = 0
        self.new_game(seed=config.get("seed"), human_seat=config.get("human_seat", 0))

    def new_game(self, seed=None, human_seat=None):
        if seed is not None and (type(seed) is not int or not 0 <= seed < 2**64):
            raise ValueError("seed must be a nonnegative 64-bit integer")
        seat = self.human_seat if human_seat is None else human_seat
        if type(seat) is not int or seat not in (0, 1):
            raise ValueError("human_seat must be 0 or 1")
        self.human_seat = seat
        self.seed = secrets.randbits(64) if seed is None else seed
        self.observation, _ = self.env.reset(self.seed, first_player=0)
        self.solver.start_episode(self.observation)
        self._advance_ai()

    def _advance_ai(self):
        for _ in range(self.env.task_spec()["max_actions"]):
            if self.observation["terminated"] or self.observation["active_player"] == self.human_seat:
                return
            action = self.solver.decide(self.observation)
            if action not in self.observation["legal_actions"]:
                raise ValueError("model returned an illegal action")
            self.observation = self.env.step(action).observation
        raise ValueError("model did not finish its turn")

    def step(self, action):
        if self.observation["terminated"]:
            raise InvalidAction("game finished; start a new game")
        if self.observation["active_player"] != self.human_seat:
            raise InvalidAction("wait for the computer's turn")
        if action not in self.observation["legal_actions"]:
            raise InvalidAction("choose a legal action")
        self.observation = self.env.step(action).observation
        self._advance_ai()

    def state(self):
        return {"mode": "duo", "task": self.task, "task_spec": self.env.task_spec(),
                "observation": self.observation, "human_seat": self.human_seat,
                "model_label": self.model_label, "seed": str(self.seed)}
