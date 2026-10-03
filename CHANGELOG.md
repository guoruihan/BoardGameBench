# Research milestones

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
