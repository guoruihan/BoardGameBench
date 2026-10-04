# Research milestones

## 0.4.2 / research V1.2.1 — 2026-10-04

- Completed10M-action development128 evaluation for all four T16 models x three
  seeds. E00_T16 mean81.3281 vs historicalINC72.8672; confirmation and final test
  remain unexecuted. Capture a later E11_T16 probability mismatch and stop safely;
  this pre-fix research snapshot does not claim uninterrupted long-run reliability.
- Register fourteen research families with planned status; implement only the
  first E00/E01/E10/E11 state-encoder x action-sharing factorial. Keep the same
  raw public inputs, true-score MC PPO and STOP semantics; report parameter counts.
- Generate fresh disjoint train/dev128/confirmation256/test512 splits; old test
  is historical audit only. Gate final test behind frozen source/split/policy selection.
- Track real actions separately from STOP/decisions, full-batch target overshoot,
  exact optimizer/RNG continuation and isolated train/evaluation resource intervals.
- Add continuous user-authorized background scheduling without a total GPU-hour
  cap, but finite phase watchdogs and no automatic production promotion.
- Reproduce historical12/144terminalanimalmisses (0.381944mean immediate loss).
  This is an implementation/design milestone, not evidence of stronger policies.
- Add separately identified E00_T16/E01_T16/E10_T16/E11_T16 matched reruns with
  synchronous sixteen-game batched sampling and shared-memory CPU workers. Preserve
  full-game returns, on-policy boundaries, STOP and PPO hyperparameters. Pipeline
  immutable-checkpoint development evaluations with backlog backpressure.
- Same-b35 4090 end-to-end checks (two repetitions, seven workers/eight CPU cores)
  measured 3.42x/3.46x/3.65x/3.72x throughput respectively. Warmup and total GPU
  reservations separately recorded; this is engineering throughput, not policy quality.

## 0.4.1 — 2026-10-04

- Separate one-step oracle-label BC fitting from genuine on-policy short-horizon
  PPO animal learning; group diagnostic holdouts by source game and board layout.
- Predeclare same-network PPO / frozen BC / BC-to-PPO routes and a conservative
  KL-limited fine-tuning regime. Count teacher assistance explicitly; require PPO
  improvement beyond frozen BC before crediting the fine-tuning stage.
- Keep true total score, original fixed online budget and original final-test seeds.
  Animal scores are diagnosis/promotion requirements, not a modified reward target.
  This is an implementation/experiment-design milestone, not a new capability result.

## 0.4.0 — 2026-10-04

- Introduce a separate timed-current-state score protocol; historical completed-game
  means are not interchangeable with the new all-attempt deadline statistics.
- Predeclare equal-input action compression and separate public card-feature checks;
  fixed three-seed staged compact PPO/search distillation, initial controls, independent
  development/validation/test seeds and a shared four-hour resource allocation.
- Record conditional paired intervals as deployment gates, not a training-population
  significance claim. No new human-parity or optimal-policy result is claimed.

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
