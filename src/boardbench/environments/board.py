"""Small shared mechanics; snapshots are private, observations are public."""
from copy import deepcopy
from numbers import Integral

from boardbench.contracts import InvalidAction, StepResult


def integer(value, low, high):
    return isinstance(value, Integral) and not isinstance(value, bool) and low <= value < high


def tuples(value):
    return tuple(map(tuples, value)) if isinstance(value, (tuple, list)) else value


class BoardGame:
    task_id = ""
    rules_version = ""

    def reset_info(self):
        return deepcopy(self.task_spec())

    def observe(self):
        return {**deepcopy(self.s), "task_id": self.task_id,
                "rules_version": self.rules_version, "legal_actions": self.legal_actions(),
                "score_breakdown": self.score_breakdown()}

    def _result(self, reason=None):
        self.s["decision_index"] += 1
        score = self.score_breakdown()
        info = {"turn_index": self.s["turn_index"], "decision_index": self.s["decision_index"]}
        if self.s["terminated"]:
            self.s["score"] = score["total"]
            info.update(score=score["total"], score_breakdown=score, reason=reason)
        return StepResult(self.observe(), score["total"] if self.s["terminated"] else 0,
                          self.s["terminated"], info=info)

    def save_state(self):
        return deepcopy({"format_version": 1, "task_id": self.task_id,
                         "rules_version": self.rules_version, "state": self.s,
                         "private": self._private_state()})

    def load_state(self, snapshot):
        if (snapshot.get("format_version") != 1 or snapshot.get("task_id") != self.task_id
                or snapshot.get("rules_version") != self.rules_version):
            raise ValueError("snapshot task/rules/format mismatch")
        # Parse on a separate instance so a malformed snapshot cannot partially load.
        other = type(self)()
        other.s = deepcopy(snapshot["state"])
        other._load_private(deepcopy(snapshot["private"]))
        other.observe()
        self.__dict__.update(other.__dict__)
        return self.observe()

    @classmethod
    def from_observation(cls, observation, sim_seed):
        if observation["task_id"] != cls.task_id or observation["rules_version"] != cls.rules_version:
            raise ValueError("observation task mismatch")
        other = cls.__new__(cls)
        other.s = {k: deepcopy(v) for k, v in observation.items()
                   if k not in ("task_id", "rules_version", "legal_actions", "score_breakdown")}
        other._resample(sim_seed)
        return other

    def fork(self, sim_seed):
        return type(self).from_observation(self.observe(), sim_seed)

    def _require_live(self):
        if self.s["terminated"]:
            raise InvalidAction("episode already terminated; explicitly reset")
