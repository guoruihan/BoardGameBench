# Evidence and design decisions

- V134 completed: outputs/v134/plans_a226_20261021_a/completion.json.
- Existing native public encoder:1346fields, worker240actions, goal819fields.
  No new simulator or hidden information needed for advice.
- Reuse V134 runner, checkpointing, native collector and bounded launch.
  Extend checkpoint version and dependency injection while retaining defaults.
- Agent judgment is a bounded logit residual. Fixed trust0.6; learned trust
  2*sigmoid(gate) initialized0.6. Original legal moves remain available.
  PPO remains natural-terminal true-score learning.
- Advice: acquire needed colors/card; advance target stack prefixes; avoid
  destroying support; place the requested animal; protect its market card.
- Limits: blueprints cover one/two compatible animal placements, no long-horizon
  restructuring, language model invocation or optimality claim.
