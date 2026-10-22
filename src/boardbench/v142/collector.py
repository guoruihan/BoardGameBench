"""Accelerated own-player-only shared-memory complete-game sampler."""
from functools import partial
from boardbench.environments.harmonies import STACKS
from boardbench.environments.harmonies_market import FastHarmoniesMarket
from boardbench.v121.collector import EnvironmentPool as BasePool
from boardbench.v129.encoding import NativeEncoder
from boardbench.v138.baseline import Planner
from .encoding import ACTIONS, FEATURES, action_map, encode_batch

REASONS = {None: 0, 'board_full_enough': 1, 'turn_limit': 2, 'supply_exhausted': 3}


class _SamplingMarket(FastHarmoniesMarket):
    def reset_info(self): return {}


class Shard:
    def __init__(self, slots, arrays, native_artifact):
        self.slots, self.arrays = slots, arrays
        self.native = NativeEncoder.from_artifact(native_artifact)
        self.games, self.obs, self.planners, self.maps = {}, {}, {}, {}

    def reset(self, seeds):
        for slot in self.slots:
            env = self.games.get(slot) or _SamplingMarket.__new__(_SamplingMarket)
            obs, _ = env.reset(seeds[slot])
            self.games[slot], self.obs[slot], self.planners[slot] = env, obs, Planner()
        self._publish()

    def _publish(self):
        live, observations, plans = [], [], []
        for slot in self.slots:
            obs = self.obs[slot]; score = obs['score_breakdown']; done = obs['terminated']
            self.arrays['done'][slot] = done
            self.arrays['stats'][slot] = (score['total'], score['animals'], len(obs['completed_cards']),
                obs['completed_turns'], sum(len(STACKS[c['stack_id']]) for c in obs['board']),
                REASONS[obs['end_reason']], sum(c['animal'] is not None for c in obs['board']))
            mapping = action_map(obs); self.maps[slot] = mapping
            self.arrays['masks'][slot].fill(False); self.arrays['features'][slot].fill(0)
            if not done:
                if not mapping: raise RuntimeError('nonterminal state has no legal actions')
                self.arrays['masks'][slot, list(mapping)] = True
                live.append(slot); observations.append(obs); plans.append(self.planners[slot].vector(obs))
        if live: self.arrays['features'][live] = encode_batch(observations, plans, self.native)

    def step(self):
        for slot in self.slots:
            if self.arrays['done'][slot]: continue
            self.obs[slot] = self.games[slot].step(self.maps[slot][int(self.arrays['actions'][slot])]).observation
            if self.games[slot].s['decision_index'] >= 256 and not self.obs[slot]['terminated']:
                raise RuntimeError('curriculum action bound exceeded')
        self._publish()


class EnvironmentPool(BasePool):
    def __init__(self, count=128, workers=0, timeout=120., *, native_artifact):
        super().__init__(count, workers, timeout,
            shard_class=partial(Shard, native_artifact=native_artifact), features=FEATURES, actions=ACTIONS)
