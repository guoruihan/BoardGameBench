"""Batched native features with unchanged synchronized cohort and action order."""
from functools import partial

from boardbench.v121.collector import EnvironmentPool as BasePool
from boardbench.v128.collector import FastShard
from boardbench.v128.environment import FastHarmonies
from boardbench.v12.encoding import action_map
from .encoding import NativeEncoder, pack


class _SamplingHarmonies(FastHarmonies):
    def reset_info(self):
        # Internal-only pool never consumes static rule metadata. Public engine
        # still returns its full reset_info; no state/RNG/observation changes.
        return {}


class NativeShard(FastShard):
    def __init__(self, slots, arrays, native_artifact):
        super().__init__(slots, arrays)
        self.encoder = NativeEncoder.from_artifact(native_artifact)

    def _publish(self, slot):
        if self.arrays['done'][slot]:
            self.arrays['features'][slot].fill(0); self.arrays['masks'][slot].fill(False); self.maps[slot] = {}
        else:
            mapping = action_map(self.observations[slot], 'slots240_v1', True); self.maps[slot] = mapping
            self.arrays['masks'][slot].fill(False); self.arrays['masks'][slot, list(mapping)] = True

    def _encode_live(self):
        slots = [s for s in self.slots if not self.arrays['done'][s]]
        if slots:
            self.arrays['features'][slots] = self.encoder.encode(pack([self.observations[s] for s in slots]))

    def reset(self, seeds):
        for slot in self.slots:
            env = self.games.get(slot)
            if env is None: env = _SamplingHarmonies.__new__(_SamplingHarmonies)
            obs, _ = env.reset(seeds[slot])
            self.games[slot], self.observations[slot], self.steps[slot] = env, obs, 0
            self.arrays['done'][slot] = False; self.arrays['stats'][slot].fill(0); self._publish(slot)
        self._encode_live()

    def step(self):
        super().step()
        self._encode_live()


class EnvironmentPool(BasePool):
    def __init__(self, count=16, workers=0, timeout=60., *, native_artifact):
        super().__init__(count, workers, timeout, shard_class=partial(NativeShard, native_artifact=native_artifact))
