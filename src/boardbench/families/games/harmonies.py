"""Family adapter over the existing accelerated official Harmonies duo engine."""
from boardbench.contracts import InvalidAction
from boardbench.environments.harmonies_duo import FastHarmoniesDuo
from boardbench.families.base import DuoGame
from boardbench.families import harmonies_kernel
from boardbench.v128.environment import public_copy


def copy_public(value):
    result = public_copy({k: v for k, v in value.items()
                          if k not in ('players', 'legal_actions', 'score_breakdown')})
    if 'players' in value:result['players'] = [public_copy(p) for p in value['players']]
    for field in ('legal_actions', 'score_breakdown'):
        if field in value:result[field] = [row.copy() for row in value[field]]
    return result


class _FamilyEngine(FastHarmoniesDuo):
    def reset(self, seed, first_player=0):
        # Bounded per-game caches: repeated benchmark seeds must not benefit
        # from a previous episode's completed trajectory.
        harmonies_kernel.terrain.cache_clear();harmonies_kernel.potential.cache_clear()
        return super().reset(seed, first_player)

    def reset_info(self):
        return {}  # The family reset contract only returns an observation.

    def score_breakdown(self):
        return [harmonies_kernel.board_score(p) for p in self.s['players']]

    def save_state(self):
        return dict(format_version=1, task_id=self.task_id, rules_version=self.rules_version,
                    state=copy_public(self.s), private=dict(bag=list(self._bag), deck=list(self._deck)))


def action_key(action):
    # The original JSON guard distinguishes 1, 1.0 and True. Keep that contract
    # while avoiding serialization of every candidate on every atomic action.
    if not isinstance(action, dict) or any(type(k) is not str or type(v) not in (str, int)
                                          for k, v in action.items()):
        raise ValueError('invalid Harmonies action fields')
    return tuple(sorted((k, type(v), v) for k, v in action.items()))


class Harmonies(DuoGame):
    game_id = 'harmonies'
    config_id = 'side_a_base32_no_spirits_duo'
    rules_version = FastHarmoniesDuo.rules_version

    def __init__(self):
        self.engine = _FamilyEngine()

    def reset(self, seed=0, first_player=0):
        observation, _ = self.engine.reset(seed, first_player)
        return self._decorate(observation)

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
        return self._decorate(self.engine.observe(), player)

    def _decorate(self, obs, player=None):
        player = self.engine.s['active_player'] if player is None else player
        obs.update(game_id=self.game_id, config_id=self.config_id,
                   player=player, current_player=self.current_player)
        return obs

    def step(self, action):
        try:
            key = action_key(action)
            legal = any(key == action_key(a) for a in self.legal_actions())
        except (TypeError, ValueError):
            legal = False
        if not legal:
            raise InvalidAction('action is not legal in this state')
        before = self.engine.save_state()
        try:
            result = self.engine.step(action.copy())
        except Exception:
            self.engine.load_state(before)
            raise
        outcome = self.outcome()
        rewards = [0., 0.]
        if outcome['terminated'] and len(outcome['winners']) == 1:
            rewards = [1. if p in outcome['winners'] else -1. for p in (0, 1)]
        return dict(observation=self._decorate(result.observation), rewards=rewards, terminated=outcome['terminated'],
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

    copy_observation = staticmethod(copy_public)

    @staticmethod
    def features_batch(observation, actions):
        return harmonies_kernel.features(observation, actions)

    @staticmethod
    def features(observation, action):
        return Harmonies.features_batch(observation, [action])[0]


GAME_CLASS = Harmonies
