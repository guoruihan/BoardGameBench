"""Family adapter over the existing accelerated official Harmonies duo engine."""
from copy import deepcopy
import json

from boardbench.contracts import InvalidAction
from boardbench.environments.harmonies_duo import FastHarmoniesDuo
from boardbench.families.base import DuoGame


class Harmonies(DuoGame):
    game_id = 'harmonies'
    config_id = 'side_a_base32_no_spirits_duo'
    rules_version = FastHarmoniesDuo.rules_version

    def __init__(self):
        self.engine = FastHarmoniesDuo()

    def reset(self, seed=0, first_player=0):
        self.engine.reset(seed, first_player)
        return self.observe()

    @property
    def current_player(self):
        return None if self.engine.s['terminated'] else self.engine.s['active_player']

    def legal_actions(self, player=None):
        if player is not None and player != self.current_player:
            return []
        return self.engine.legal_actions()

    def observe(self, player=None):
        player = self.engine.s['active_player'] if player is None else player
        if type(player) is not int or player not in (0, 1):
            raise ValueError('invalid player')
        obs = self.engine.observe()
        obs.update(game_id=self.game_id, config_id=self.config_id,
                   player=player, current_player=self.current_player)
        return obs

    def step(self, action):
        try:
            key = json.dumps(action, sort_keys=True, allow_nan=False)
            legal = any(key == json.dumps(a, sort_keys=True) for a in self.legal_actions())
        except (TypeError, ValueError):
            legal = False
        if not legal:
            raise InvalidAction('action is not legal in this state')
        before = self.engine.save_state()
        try:
            self.engine.step(deepcopy(action))
        except Exception:
            self.engine.load_state(before)
            raise
        outcome = self.outcome()
        rewards = [0., 0.]
        if outcome['terminated'] and len(outcome['winners']) == 1:
            rewards = [1. if p in outcome['winners'] else -1. for p in (0, 1)]
        return dict(observation=self.observe(), rewards=rewards, terminated=outcome['terminated'],
                    truncated=False, info=outcome)

    def snapshot(self):
        return dict(format_version=1, game_id=self.game_id, config_id=self.config_id,
                    rules_version=self.rules_version, engine=self.engine.save_state())

    serialize = snapshot

    def restore(self, snapshot):
        if any(snapshot.get(k) != v for k, v in dict(format_version=1, game_id=self.game_id,
                config_id=self.config_id, rules_version=self.rules_version).items()):
            raise ValueError('snapshot game/config mismatch')
        self.engine.load_state(snapshot['engine'])
        return self.observe()

    def scores(self):
        return [s['total'] for s in self.engine.score_breakdown()]

    def score_breakdown(self):
        return self.engine.score_breakdown()

    def outcome(self):
        s = self.engine.s
        return dict(terminated=s['terminated'], scores=self.scores(),
                    winners=list(s['winners'] or []), reason=s['end_reason'])

    def validate(self):
        self.engine.validate()

    @staticmethod
    def features_batch(observation, actions):
        from boardbench.v146.policy import personal_view, terrain_rank
        from boardbench.v125.policies import action_key
        view = personal_view(observation)
        view['legal_actions'] = actions
        values = {action_key(a): value for value, a in terrain_rank(view)}
        kinds = ('choose_bundle', 'take_card', 'place_token', 'place_animal', 'end_turn')
        rows = []
        for action in actions:
            row = [values[action_key(action)]] + [float(action['type'] == k) for k in kinds]
            row += [len(view['active_cards']), len(view['completed_cards']), view['turn_index']]
            rows.append(row+[0.]*(24-len(row)))
        return rows

    @staticmethod
    def features(observation, action):
        return Harmonies.features_batch(observation, [action])[0]


GAME_CLASS = Harmonies
