"""Run after editable installation: python examples/rl_rollout.py."""
from boardbench.environments.adapters import RiskRL

env = RiskRL()
observation, info = env.reset(seed=123)
total_reward = 0
for action in (0, 0, 1):  # DRAW, DRAW, BANK
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(observation, reward, terminated, truncated, info)
    if terminated or truncated:
        break
assert total_reward == info["score"]
