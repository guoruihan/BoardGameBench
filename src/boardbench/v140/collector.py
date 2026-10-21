"""Reuse the solo shared-memory pool, native batch codec and bitset engine.

Each match has TWO persistent planners. Statistics: score0, score1, animal0,
animal1, active seat, terminal seat0 win credit, current round. No private future
or reset seed crosses the policy feature boundary.
"""
from functools import partial
import numpy as np

from boardbench.environments.harmonies_duo import FastHarmoniesDuo, player_view
from boardbench.v121.collector import EnvironmentPool as BasePool
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner
from .encoding import ACTIONS, FEATURES, action_map, encode_batch


class _SamplingDuo(FastHarmoniesDuo):
    def reset_info(self):
        return {}


class Shard:
    def __init__(self, slots, arrays, native_artifact):
        self.slots, self.arrays = slots, arrays
        self.native = NativeEncoder.from_artifact(native_artifact)
        self.games, self.obs, self.planners, self.maps = {}, {}, {}, {}

    def reset(self, seeds):
        for slot in self.slots:
            env = self.games.get(slot) or _SamplingDuo.__new__(_SamplingDuo)
            obs, _ = env.reset(seeds[slot])
            self.games[slot], self.obs[slot] = env, obs
            self.planners[slot] = [Planner(), Planner()]
        self._publish()

    def _publish(self):
        live, observations, plans = [], [], []
        for slot in self.slots:
            obs = self.obs[slot]; seat = obs['active_player']; s = obs['score_breakdown']
            done = obs['terminated']; self.arrays['done'][slot] = done
            win = .5 if not done or len(obs['winners']) == 2 else float(obs['winners'][0] == 0)
            self.arrays['stats'][slot] = (s[0]['total'], s[1]['total'], s[0]['animals'], s[1]['animals'],
                                         seat, win, obs['players'][seat]['turn_index'])
            mapping = action_map(obs); self.maps[slot] = mapping
            self.arrays['masks'][slot].fill(False)
            self.arrays['features'][slot].fill(0)
            if not done:
                self.arrays['masks'][slot, list(mapping)] = True
                live.append(slot); observations.append(obs)
                plans.append(self.planners[slot][seat].vector(player_view(obs)))
        if live:
            self.arrays['features'][live] = encode_batch(observations, plans, self.native)

    def step(self):
        for slot in self.slots:
            if self.arrays['done'][slot]:
                continue
            self.obs[slot] = self.games[slot].step(self.maps[slot][int(self.arrays['actions'][slot])]).observation
            if self.games[slot].s['decision_index'] >= 768 and not self.obs[slot]['terminated']:
                raise RuntimeError('duo action limit')
        self._publish()


class EnvironmentPool(BasePool):
    def __init__(self, count=64, workers=0, timeout=120., *, native_artifact):
        super().__init__(count, workers, timeout,
                         shard_class=partial(Shard, native_artifact=native_artifact),
                         features=FEATURES, actions=ACTIONS)
