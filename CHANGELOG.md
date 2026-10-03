# Research milestones

## 0.3.0 — 2026-10-03

- Freeze training seeds 811/812/813, 64 validation and 128 new test seeds per
  policy. Select each run's checkpoint on validation, retain all repetitions and
  matched initial controls. No tuning from the new test outcomes.
- Micro fixed-set flat-MLP diagnostic shows learning but limited performance.
  Shared action scoring with explicit relational features makes PPO competitive
  with the heuristic (14.83 vs 14.64); strong untrained controls prevent attributing
  the entire representation gain to learning. No significant superiority claim.
- TIE PPO reaches 144.24 vs heuristic 135.13 and search 147.34. Separate teacher
  imitation reaches 136.90; it is supervised learning, not pure RL.
- Harmonies turn-budget search reaches 85.20, including 29.23 animal points, at
  13.60 seconds of solver work per game. PPO averages 66.73 with 7.16 animal points;
  its gains remain terrain-dominated. Report score-cost tradeoffs, not SOTA.
- Separate across-training-seed sample SD from fixed-model paired environment
  intervals, solver work from Runner/load costs, and selected checkpoint training
  time from total experimental investment. All rules/data remain unchanged.

## 0.2.0 — 2026-10-03

- Establish three fixed game/rules versions and leakage-free train/validation/test
  seed splits. Terminal true-score evaluation; explicit training-only score-delta
  shaping with gamma=lambda=1 on atomic actions.
- Complete one-seed PPO pilot with one bounded 4x budget extension. Validation
  selects checkpoints and search budget; 128 frozen test episodes per policy.
- Harmonies learned policy outperforms untrained/random and the delivered
  heuristic in this pilot, but not search. Micro learning benefit is unverified;
  Take It Easy! RL remains far below heuristic/search. No SOTA or multi-seed
  stability claim and no fabricated easy/medium/hard labels.
- Runtime package 0.2.0 continues V0.1 behavior; old source-bound checkpoints
  require their own source snapshots, not automatic cross-version migration.
