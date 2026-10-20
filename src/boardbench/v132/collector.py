"""Native encoding with public-state resets; private futures are resampled."""
from functools import partial

from boardbench.v121.collector import EnvironmentPool as BasePool
from boardbench.v129.collector import NativeShard, _SamplingHarmonies
from .data import PUBLIC_KEYS


class CurriculumShard(NativeShard):
    def reset(self, specifications):
        for slot in self.slots:
            specification = specifications[slot]
            if isinstance(specification, dict):
                if set(specification) != {"observation", "future_seed", "environment_seed", "source"}:
                    raise ValueError("only public curriculum reset specifications allowed")
                public = specification["observation"]
                if set(public) != PUBLIC_KEYS or public["terminated"]:
                    raise ValueError("curriculum must start before terminal")
                env = _SamplingHarmonies.from_observation(public, specification["future_seed"])
                obs = env.observe()
                if obs != public:
                    raise ValueError("invalid public curriculum state")
            else:
                env = self.games.get(slot)
                if env is None:
                    env = _SamplingHarmonies.__new__(_SamplingHarmonies)
                obs, _ = env.reset(specification)
            self.games[slot], self.observations[slot], self.steps[slot] = env, obs, 0
            self.arrays["done"][slot] = False
            score = obs["score_breakdown"]
            self.arrays["stats"][slot] = (0, score["total"], score["animals"],
                                         len(obs["completed_cards"]), False, False, 0)
            self._publish(slot)
        self._encode_live()


class EnvironmentPool(BasePool):
    def __init__(self, count=16, workers=0, timeout=60., *, native_artifact):
        super().__init__(count, workers, timeout,
                         shard_class=partial(CurriculumShard, native_artifact=native_artifact))
