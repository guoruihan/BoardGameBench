"""Thin interfaces; all transitions run through the same task engine."""
from numbers import Integral

from .risk import RiskCollect


class RiskRL:
    """Gym-style tuples without a dependency on Gym or an RL algorithm library."""
    actions = ("DRAW", "BANK")

    def __init__(self, engine=None):
        self.engine = engine if engine is not None else RiskCollect()

    @staticmethod
    def encode(obs):
        # The existing terminal flag is also the validity mask for the score slot.
        return (obs["pot"], *(obs["remaining"][k] for k in ("1", "2", "3", "fail")),
                obs["steps"], int(obs["terminated"]), obs["score"] if obs["terminated"] else 0)

    def reset(self, *, seed=0):
        obs, info = self.engine.reset(seed)
        return self.encode(obs), info

    def step(self, action):
        if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < len(self.actions):
            from boardbench.contracts import InvalidAction
            raise InvalidAction("RL action must be integer 0 (DRAW) or 1 (BANK)")
        result = self.engine.step(self.actions[int(action)])
        return (self.encode(result.observation), result.reward, result.terminated,
                result.truncated, result.info)


class Simulation:
    def __init__(self, engine, count_step):
        self._engine = engine
        self._count_step = count_step

    def step(self, action):
        result = self._engine.step(action)
        self._count_step()
        return result

    def observe(self):
        return self._engine.observe()

    def legal_actions(self):
        return self._engine.legal_actions()

    def fork(self, seed):
        return Simulation(self._engine.fork(seed), self._count_step)


class SearchAccess:
    def __init__(self, fork, count_step):
        self._fork = fork
        self._count_step = count_step

    def fork(self, sim_seed):
        return Simulation(self._fork(sim_seed), self._count_step)
