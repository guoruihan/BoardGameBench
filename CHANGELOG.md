# Research milestones

## 0.4.28 / research V1.4.4 — 2026-10-22

- Prioritize own absolute score in official duo self-play; keep margin control.
- Contrast current/current, own-history and full-return credit assignment from
  identical curriculum weights; subsequent parents/variants follow development results.
- Fix transfer and archived V143 opponents across all rounds, use environment-
  clustered paired comparisons, then select on common development256 and confirm512.
- Eight hours includes setup and final evaluation; no confirmation-driven iteration,
  no claim that a moving self-play mean proves general competitive improvement.

## 0.4.27 / research V1.4.3 — 2026-10-22

- Move development-selected V142 own-layout skills back to official duo;
  preserve inherited actor behavior and learn public-opponent residuals.
- Compare both-seat current-policy self-play versus75%current/25%self-history,
  two replicas each. Optimize true score margin; no old weak training opponent.
- Select on development128 paired seats versus frozen transfer; confirm on
  fresh512 paired seats and crossplay selected current/league policies.
- Self-play's own50% symmetry is not a progress metric; require independent
  opponent comparisons and distinguish course scores from official-duo scores.


## 0.4.26 / research V1.4.2 — 2026-10-22

- Separate own-layout learning from competition: stationary random public-market
  depletion, no opponent board/score/end condition,13 own turns maximum.
- Compare mature solo versus own-score duo initialization, two paired replicas,
  complete-game PPO with true own-score reward AND own-score checkpoint selection.
- Fresh development128/confirmation512; retain initial checkpoints and compare
  each selected model with its own initializer on paired confirmation seeds.
- This is a labeled curriculum, not official solo/duo or a cross-protocol gain.
  Follow with frozen official-duo transfer assessment before adversarial learning.


## 0.4.24 / research V1.4.0 — 2026-10-22

- Move from solo score maximization to official two-player shared-market play.
  Preserve solo results; multiplayer scores are a different distribution.
- Initialize from frozen V138 skills and compare own-score versus competitive
  margin PPO, two replicas each, with identical fixed opponents and both seats.
- Observe all public opponent/market information; no future supply leakage.
  True-score differences telescope on each learner's own decision sequence.
- Select only on development64 paired seats; independent confirmation512.
  Four-hour guarded first campaign includes preflight/evaluation/checkpointing;
  improvement is a measured outcome, not an implementation claim.

## 0.4.23 / delivery V1.3.8 — 2026-10-22

- Freeze V137 control_a by development score, without further training.
- Fresh512 validation mean112.1797; first32 CPU3-second attempts mean113.8438,
  all natural, mean online0.454seconds. Different panels are not a gain claim.
- Keep the original formal test sealed and separate model-only, startup and
  strict online costs. Next authorized scope is two-player Harmonies withinV1.

## 0.4.22 / research V1.3.7 — 2026-10-22

- Close out the completed campaign: paired confirmation512 means112.0967
  (continued PPO),111.3633 (frozen worker),112.0000 (joint),111.7529 (refresh).
  No new goal-selection method reliably beats continued PPO. Retain negative
  results and all checkpoints; model-only GPU evaluation is not3-second CPU
  deployment acceptance or proof of a human/optimality ceiling.

- Test whether V136 goal-execution skills improve complete games through learned
  goal selection. Compare continued PPO, frozen-worker selector PPO, joint PPO,
  and joint PPO with refreshed own-state goal labels, two replicas each.
- Label public planning states with the current worker and selector, not expert
  continuation. Separate fitting futures, verification futures and environment
  holdouts; preserve true terminal rewards and sealed formal tests.
- Share identical joint/refresh first-stage training, then branch on refreshed
  data. Retain every mechanism through fixed endpoints and fresh confirmation512;
  do not eliminate a skill branch based on a tiny development difference.
- Bound formal work to four hours on a226 GPU0–3, at most57 physical CPUs. Retain
  all artifacts and validation costs. Pilot values do not establish score gains.

## 0.4.19 / research V1.3.4 — 2026-10-21

- Add public spatial blueprints, optional two-turn commitment, and a separate
  semi-Markov PPO manager over a bounded candidate set. Reuse the unchanged
  native game, legal masks and true-score full-game PPO worker.
- Compare identical worker-adapter capacity with zero goals, reactive goals,
  persistent goals, and learned plan selection, each with two training seeds.
  Zero-initialized residual preserves the predesignated mature PPO-a policy.
- Freeze new512 development,512 audit and512 confirmation environments. First
  compare4M actions per arm; automatically continue the development-selected
  non-control and capacity control for8M more, preserving both policy levels,
  optimizers and RNG. Final confirmation and plan-removal audits do not tune.
- Bound the campaign to3hours, GPU0-3 on a226 and57 physical CPUs maximum.
  Preserve V133 and all unselected policies. No online LLM/API, new search,
  rule change, reward shaping, or performance gain claimed before results.

## 0.4.14 / research V1.2.10 — 2026-10-20

- Change the learning experiment from weak imitation initialization followed by
  mostly unconstrained PPO to persistent imitation plus recurrent correction on
  states visited by the current student. Keep the true-score PPO objective.
- Register four two-seed arms starting from the identical10M-action mature model:
  PPO-only, static BC, and correction BC at0.05/0.20. Three rounds add2M actions
  each; each round gathers64teacher-labeled student games at900seconds/game.
- Search remains a training teacher, not inference fallback. Its candidate
  estimates are habitat-continuation returns, not optimal Q or unbiased student
  values. Preserve paired development-only evaluation and untouched formal test.
- No score improvement or teacher parity claimed at implementation time; retain
  previous best strategies and separate teacher computation from PPO-action cost.


## 0.4.6 / research V1.2.4 — 2026-10-20

- Move from local Q diagnostics to complete-game original/space/habitat
  continuation controls. Freeze32 development environments, three search seeds,
  common top6 and60/300/900-second cumulative solver-wall caps.
- Separate equal-cap from equal-actual-time evidence. Record measured time,
  discarded incomplete common-sample rounds, fallback and animal components;
  compare complete environment-level prefixes and preserve all failed attempts.
- Retain greedy controls and the exact original64-sample teacher. No new RL,
  held-out test, optimality claim or production promotion. Bound CPU execution
  to half of b35, frozen source and an independent absolute-deadline watchdog.

## 0.4.5 / research V1.2.3 — 2026-10-19

- Register local candidate-coverage and continuation-policy diagnostics before
  macro planning or more training. Treat space utilization as a competing factor
  to animal-pattern planning, not assume the105-point plateau is optimality.
- Separate8-sample discovery from64-sample independent confirmation; group the
  two source states per environment and retain the original development/test boundary.
  These are local Q contrasts, not full-game or equal-time performance claims.
- Bound the first pilot to32 physical cores and two hours after the existing
  campaign releases its CPU lock. Preserve prior sources/results and production.

## 0.4.4 / research V1.2.2 — 2026-10-19

- Register a Harmonies-only teacher-strength diagnostic before additional RL or
  imitation training: unchanged short-horizon search versus heuristic terminal
  rollout, four sample budgets, three fixed search repetitions and matched
  development panels. Search sampling repetitions are not training repetitions.
- Separate full-game high-compute reference quality from the unchanged3-second
  deployment protocol; a flat curve cannot establish optimality. Approximate
  terminal rollout is not guaranteed to improve every game or with more samples.
- Bound the initial background campaign by the user's eight-hour return window
  and half of b35's physical CPU cores. No new GPU training, test evaluation,
  cross-game claim or automatic production promotion is part of this release.

## 0.4.3 / research V1.2.1 — 2026-10-04

- Register explicitly identified FP64 arithmetic continuations of all four T16
  models after a mature FP32 network exposed batch-size-dependent probability
  drift. Architecture, objective, rollout cohort and PPO hyperparameters stay fixed.
- Preserve the FP32 source/results; new curves inherit parent training progress
  and costs with auditable numeric-conversion lineage, not from-scratch FP64 labels
  or a claim of bitwise cross-source reproduction. Keep the original1e-4 gate.
- Demonstrate saved-witness repair, finite updates and12 mature-policy imports.
  This milestone concerns numerical reliability/recoverability, not stronger scores.
  Development/confirmation/final-test boundaries and production pointers remain unchanged.

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
