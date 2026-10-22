# Findings
- V142 parent selected by highest development mean: duo_a116.9766, exported best_policy.pt SHA b1510da0bfda83c2c137b36c85fe0e079fb30adee47c3629d36c03d7e9f947b8. Its separate course confirmation116.6953 does not establish official-duo ability.
- Directly restoring full opponent inputs into the existing V142 context changes its behavior before any learning. A new zero-output residual can expose official public-opponent information while preserving the old own-view actor.
- V140 pool already executes official duo with per-seat persistent planners, native codec and bitsets. Its PPO optimizer and paired-seat evaluator are reusable; collection must now retain BOTH current-policy seats.
- Fresh a22614:32 probe:GPU0-3 idle14MiB;GPU4-7 busy. CPU load14.38; no active old boardgame tmux experiment.
- Preserve unrelated untracked docs/BoardBench_V1_2_Design.md. No relevant global memory hits used.
- Formal initial self-battle128 environments paired seats:79.48828125 own score,8.74609375 animals,10.9140625 turns,32.7421875 resources. Course116.6953 is NOT preserved as formal score despite exact actor transfer. Attribution to market/termination differences remains a hypothesis, not an isolated finding.
- Both methods intentionally share an identical policy/seed prefix until the first history snapshot atbatch50; actual first20-batch metrics match for paired replicas. No reliable early learning gain yet.
- Proof source6a73a2cdd556b81567e14dc120f7d7be8c06dcb0e5b32227baef4f37501aa216, campaign outputs/v143/selfplay_a226_20261022_a. User can inspect initial_self_battle/result.json and finalconfirmation/crossplay later.
