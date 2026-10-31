"""Small two-player contract: private snapshots never become player inputs.

Subclasses implement _reset, _legal_actions, _step, _observe and scores.
All chance events use self.rng. An action is one actual player choice, including
reactions and bonus choices; it need not alternate seats.
"""
from copy import deepcopy
import math
import random
import json

from boardbench.contracts import InvalidAction


FEATURE_DIM = 24


def json_tuples(value):
    return tuple(json_tuples(v) for v in value) if isinstance(value, (tuple, list)) else value


class DuoGame:
    game_id = "abstract"
    rules_version = "1"
    config_id = "base_duo"
    default_weights = [1.0] + [0.0] * (FEATURE_DIM - 1)

    def __init__(self):
        self.state = {}
        self.rng = random.Random()

    @classmethod
    def describe(cls):
        return dict(game_id=cls.game_id, rules_version=cls.rules_version,
                    config_id=cls.config_id, players=2, feature_dim=FEATURE_DIM)

    def reset(self, seed=0, first_player=0):
        if type(first_player) is not int or first_player not in (0, 1):
            raise ValueError("first_player must be 0 or 1")
        self.rng = random.Random(seed)
        self.state = dict(current_player=first_player, first_player=first_player,
                          terminated=False, decision_index=0, winners=[], end_reason=None)
        self._reset(first_player)
        return self.observe()

    @property
    def current_player(self):
        return None if self.state.get("terminated") else self.state["current_player"]

    def current_actor(self):
        return self.current_player

    def legal_actions(self, player=None):
        if self.state.get("terminated"):
            return []
        if player is not None and player != self.current_player:
            return []
        return self._legal_actions()

    def observe(self, player=None):
        player = self.state.get("current_player", 0) if player is None else player
        if type(player) is not int or player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        observation = deepcopy(self._observe(player))
        observation.update(game_id=self.game_id, rules_version=self.rules_version,
                           config_id=self.config_id, player=player,
                           current_player=self.current_player,
                           terminated=bool(self.state.get("terminated")),
                           decision_index=self.state["decision_index"])
        return observation

    def step(self, action):
        if self.state.get("terminated"):
            raise InvalidAction("game has ended")
        try:
            encoded = json.dumps(action, sort_keys=True, allow_nan=False)
            legal = isinstance(action, dict) and any(
                encoded == json.dumps(a, sort_keys=True, allow_nan=False)
                for a in self.legal_actions())
        except (TypeError, ValueError):
            legal = False
        if not legal:
            raise InvalidAction("action is not legal in this state")
        before = self.snapshot()
        try:
            self._step(deepcopy(action))
            self.state["decision_index"] += 1
            result = self.outcome()
            rewards = [0., 0.]
            if result["terminated"] and len(result["winners"]) == 1:
                rewards = [1. if p in result["winners"] else -1. for p in (0, 1)]
            return dict(observation=self.observe(), rewards=rewards,
                        terminated=result["terminated"], truncated=False, info=result)
        except Exception:
            self.state = before["state"]
            self.rng.setstate(json_tuples(before["rng"]))
            raise

    def snapshot(self):
        return deepcopy(dict(format_version=1, game_id=self.game_id,
                             rules_version=self.rules_version, config_id=self.config_id,
                             state=self.state, rng=self.rng.getstate()))

    serialize = snapshot

    def restore(self, snapshot):
        if (snapshot.get("format_version") != 1 or snapshot.get("game_id") != self.game_id
                or snapshot.get("rules_version") != self.rules_version
                or snapshot.get("config_id") != self.config_id):
            raise ValueError("snapshot schema/game/config mismatch")
        rng = random.Random()
        rng.setstate(json_tuples(snapshot["rng"]))
        state = deepcopy(snapshot["state"])
        if state.get("current_player") not in (0, 1):
            raise ValueError("invalid snapshot actor")
        old_state, old_rng = self.state, self.rng
        try:
            self.state, self.rng = state, rng
            self.validate()
            observation = self.observe()
        except Exception:
            self.state, self.rng = old_state, old_rng
            raise
        return observation

    def score_breakdown(self):
        return [dict(total=s) for s in self.scores()]

    def outcome(self):
        return dict(terminated=bool(self.state.get("terminated")),
                    scores=list(self.scores()), winners=list(self.state.get("winners", [])),
                    reason=self.state.get("end_reason"))

    def finish(self, winners=None, reason="natural_terminal"):
        if winners is None:
            scores = self.scores()
            winners = [p for p in (0, 1) if scores[p] == max(scores)]
        self.state.update(terminated=True, winners=list(winners), end_reason=reason)

    def validate(self):
        if self.state.get("current_player") not in (0, 1):
            raise ValueError("invalid actor")
        if self.state.get("terminated") and not self.state.get("winners"):
            raise ValueError("terminal game needs winner(s)")

    @staticmethod
    def features(observation, action):
        """24 public-only action features; default is an unbiased policy.

        Concrete engines should use feature 0 for their scalar heuristic and
        additional interpretable features for learned refinements. Never read
        a private engine state here. This method is intentionally static.
        """
        return [0.0] * FEATURE_DIM

    @classmethod
    def heuristic(cls, observation, action):
        features = cls.features(observation, action)
        if len(features) != FEATURE_DIM or not all(math.isfinite(float(x)) for x in features):
            raise ValueError("features must be 24 finite public-only numbers")
        return sum(float(x) * w for x, w in zip(features, cls.default_weights))

    def _reset(self, first_player): raise NotImplementedError
    def _observe(self, player): raise NotImplementedError
    def _legal_actions(self): raise NotImplementedError
    def _step(self, action): raise NotImplementedError
    def scores(self): raise NotImplementedError
