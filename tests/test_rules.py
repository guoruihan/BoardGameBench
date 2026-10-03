import random
from dataclasses import asdict

import pytest

from boardbench.contracts import InvalidAction
from boardbench.environments.risk import RiskCollect
from boardbench.environments.adapters import RiskRL


class FirstCard:
    def randrange(self, n):
        return 0


def test_all_rewards_without_replacement_then_bank():
    env = RiskCollect()
    env._rng = FirstCard()
    rewards = []
    for pot in [1, 2, 4, 6, 9, 12]:
        result = env.step("DRAW")
        assert result.observation["pot"] == pot
        assert not result.terminated
        rewards.append(result.reward)
    assert env.observe()["remaining"] == {"1": 0, "2": 0, "3": 0, "fail": 2}
    result = env.step("BANK")
    assert result.terminated and result.reward == 12
    assert sum(rewards) + result.reward == result.info["score"] == 12


def test_failure_after_collecting_all_rewards_and_initial_bank():
    env = RiskCollect()
    assert env.step("BANK").info["score"] == 0
    env.reset(0)
    env._rng = FirstCard()
    results = [env.step("DRAW") for _ in range(7)]
    assert results[-1].terminated
    assert sum(r.reward for r in results) == results[-1].info["score"] == 0
    assert results[-1].observation["remaining"]["fail"] == 1


def test_rejection_preserves_state_and_rng():
    env = RiskCollect()
    for terminal in (False, True):
        if terminal:
            env.step("BANK")
        state, rng = env.observe(), env._rng.getstate()
        with pytest.raises(InvalidAction):
            env.step("DRAW" if terminal else {"unknown": 1})
        assert env.observe() == state
        assert env._rng.getstate() == rng


def test_seeds_and_simulation_do_not_touch_real_rng():
    for seed in range(20):
        env, control = RiskCollect(), RiskCollect()
        env.reset(seed)
        control.reset(seed)
        while not env.observe()["terminated"]:
            for sim_seed in range(100):
                branch = env.fork(sim_seed)
                while not branch.observe()["terminated"]:
                    branch.step("DRAW")
            random.seed(999)
            assert asdict(env.step("DRAW")) == asdict(control.step("DRAW"))


def test_rl_matches_engine():
    for seed in range(20):
        env, rl = RiskCollect(), RiskRL()
        obs, info = env.reset(seed)
        assert rl.reset(seed=seed) == (rl.encode(obs), info)
        for action in ("DRAW", "DRAW", "BANK"):
            r = env.step(action)
            assert rl.step(rl.actions.index(action)) == (
                rl.encode(r.observation), r.reward, r.terminated, r.truncated, r.info)
            if r.terminated:
                break
    with pytest.raises(InvalidAction):
        rl.step(-1)
