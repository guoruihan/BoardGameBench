"""Native environment plus shared, public plan channels. No engine fork."""
import ctypes
from functools import partial
import multiprocessing as mp

import numpy as np

from boardbench.v121.collector import EnvironmentPool as BasePool
from boardbench.v129.collector import NativeShard
from .plans import ARMS, K, PLAN_DIM, achieved, candidates, encode_plan, refresh_goal


def views(buffers, count):
    shapes = {"plans": (count, K, PLAN_DIM), "plan_mask": (count, K), "plan": (count, PLAN_DIM),
              "request": (count,), "choice": (count,), "success": (count,), "turn": (count,)}
    types = {"plan_mask": np.bool_, "request": np.bool_, "choice": np.int64, "turn": np.int64}
    return {key: np.frombuffer(buffer, types.get(key, np.float32)).reshape(shapes[key])
            for key, buffer in buffers.items()}


class PlanShard(NativeShard):
    def __init__(self, slots, arrays, native_artifact, plan_buffers, count, arm, horizon):
        super().__init__(slots, arrays, native_artifact)
        self.extra = views(plan_buffers, count)
        self.arm, self.horizon = arm, horizon
        self.active, self.started_turn, self.bank = {}, {}, {}

    def reset(self, seeds):
        self.active.clear()
        self.started_turn.clear()
        self.bank.clear()
        super().reset(seeds)

    def step(self):
        # Choices were made from the previous observation's published candidates.
        for slot in self.slots:
            if self.arrays["done"][slot] or not self.extra["request"][slot]:
                continue
            choice = int(self.extra["choice"][slot])
            if not self.extra["plan_mask"][slot, choice]:
                raise ValueError("invalid high-level choice")
            self.active[slot] = self.bank[slot][choice]
            self.started_turn[slot] = self.observations[slot]["turn_index"]
        super().step()

    def _encode_live(self):
        super()._encode_live()
        for slot in self.slots:
            obs = self.observations[slot]
            turn = obs["turn_index"]
            old = self.active.get(slot, [])
            self.extra["success"][slot] = float(bool(old) and achieved(obs, old[0]))
            age = turn - self.started_turn.get(slot, turn)
            invalid = any(refresh_goal(obs, g) is None for g in old)
            request = (slot not in self.active or invalid or
                       age >= (self.horizon if old else 1) or self.arm in ("capacity", "reactive"))
            self.extra["turn"][slot] = turn
            self.extra["request"][slot] = request and not self.arrays["done"][slot]
            self.extra["plans"][slot].fill(0)
            self.extra["plan_mask"][slot].fill(False)
            self.extra["plan"][slot] = encode_plan(obs, old, age)
            if self.arrays["done"][slot]:
                self.extra["plan"][slot].fill(0)
            elif request:
                self.bank[slot] = candidates(obs)
                for i, plan in enumerate(self.bank[slot]):
                    self.extra["plans"][slot, i] = encode_plan(obs, plan)
                    self.extra["plan_mask"][slot, i] = True


class EnvironmentPool(BasePool):
    def __init__(self, count=128, workers=0, timeout=120., *, native_artifact, arm, horizon=2,
                 shard_class=PlanShard):
        if arm not in ARMS or horizon < 1:
            raise ValueError("explicit plan mode/horizon required")
        ctx = mp.get_context("spawn")
        specs = {"plans": (ctypes.c_float, count * K * PLAN_DIM),
                 "plan_mask": (ctypes.c_bool, count * K), "plan": (ctypes.c_float, count * PLAN_DIM),
                 "request": (ctypes.c_bool, count), "choice": (ctypes.c_int64, count),
                 "success": (ctypes.c_float, count), "turn": (ctypes.c_int64, count)}
        self.plan_buffers = {k: ctx.RawArray(t, n) for k, (t, n) in specs.items()}
        self.extra = views(self.plan_buffers, count)
        factory = partial(shard_class, native_artifact=native_artifact, plan_buffers=self.plan_buffers,
                          count=count, arm=arm, horizon=horizon)
        super().__init__(count, workers, timeout, shard_class=factory)
