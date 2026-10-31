# Research milestones

## V1.6.0 New World policy-cycle diagnosis — 2026-10-31

- Reproduce 12 development truncations as legal policy cycles, including learned
  candidates. Resource production and context-poor discard preferences can undo
  each other; this is not an environment deadlock or a new terminal rule.
- An explicit public-history guard finishes four of six affected diagnostic
  games within the original cap. A discard-stage variant does not improve that
  coverage and is not retained. New development comparisons are still required
  before selecting guarded opponents; diagnostic scores are not strength claims.

## V1.6.0 long-run rule-boundary correction — 2026-10-31

- Reopen the initial runtime freeze after historical training audits reveal rare
  no-action states in Splendor and Gizmos. Retain old results as rules-1 evidence;
  rules-2 selection and confirmation must be rerun, with explicit weight reuse.
- Other games retain their development-frozen recommendations and independently
  recorded confirmation source identities. No confirmation result is used to
  replace a selected policy, and no incomplete game receives a fabricated reward.

## V1.6.0 paired development and Targi cycle diagnosis — 2026-10-31

- Freeze 12 initial same-seed leagues: 5,760 scheduled games, 5,758 complete and
  two unresolved Jaipur round ties. Report paired intervals and preserve all
  denominators; these development measurements do not select on final test data.
- Diagnose a legal 1:1 Targi trade cycle in a CEM candidate. An explicit new bot
  guard uses public action history to avoid revisiting resource states within one
  Trader activation; original game legality and older bot behavior are unchanged.
- Preserve learned-prior and Monte Carlo continuation variants in separate runs.
  Frozen pool export records explicit development selections and multiple retained
  opponents; no automatic promotion from confirmation outcomes.

## V1.6.0 long-run seed partition correction — 2026-10-31

- Detect a potential train/gate/confirmation overlap above 1,000 PPO updates in
  the initial arithmetic scheme. Stop the three longer continuations before that
  boundary (613, 757 and 681 updates), preserving models, optimizer states and data.
- New experiments encode disjoint algorithm/split, iteration and game-index bits,
  staying exactly representable as JSON integers. The completed original 512-update
  runs are unaffected. Later pool confirmation uses fresh common environment seeds.

## V1.6.0 continued opponent development — 2026-10-31

- Add full-game Monte Carlo return and reduced-prior continuation variants, with
  explicit warm starts and frozen initial policy artifacts. The original PPO/GAE
  runs remain intact; variants use new output directories and configuration files.
- Balance non-self-play training seats within each batch. Promote only through
  paired development games; final cross-method comparison requires common seeds.

## V1.6.0 opponent-learning pipeline — 2026-10-31

- Introduce heuristic-prior candidate PPO alongside evolutionary policy search.
  Each player's complete decision chain gets its own terminal outcome and GAE;
  opponents are a frozen prior, base heuristic and a self-play mixture.
- Use CPU simulation and a GPU actor/critic optimizer. Preserve independent
  checkpoint gates and final confirmation, and retain heuristic opponents when
  learned candidates do not improve reliably. CUDA smoke is engineering evidence,
  not a learned-strength result.

## V1.6.0 initial development / 0.5.0.dev1 — 2026-10-31

- Extend the target to 17 original two-player games, one documented official base
  setting each; permit explicit blocking when original component data is missing.
- Start rolling opponent development with public candidate features and paired
  population policy search. Retain random and heuristic baselines, preserve
  incomplete/unresolved runs, and reserve independent confirmation seeds.
- No cross-game learning result or superiority claim yet. Existing Harmonies
  research artifacts remain separate from newly trained family opponents.

## V1.5.1 stage closure — 2026-10-31

- Close the stage with a playable solo/duo Harmonies task, reproducible strategy
  experiments and a separate direct-assistant play evaluation. Distinguish Codex
  as the user-guided experiment developer from Codex as the in-game player.
- Retain the animal/terrain tradeoff and negative total-score results; no claim
  of optimal play, new baseline promotion or additional training in this snapshot.
- Research tag `v1.5.1` preserves package 0.4.39 and links the stage summary.

## 0.4.39 / research V1.5.1 — 2026-10-27

- Current conversational assistant explicitly plays 8 solo and 8 official duo
  games with public rule/matching tools. Solo mean88.625, best101; duo mean84.75
  versus frozen V147r2_continue86.00, with3wins/5losses including an animal-cube
  tie-break victory at83:83. All fixed seeds and bad decisions remain included.
- Duo animal mean34.50 versus17.125, terrain50.25 versus68.875: stronger animal
  execution in this sample does not establish a stronger complete-game policy.
- This is sequential, tool-assisted interactive play with cross-game learning,
  not an independent frozen-policy estimate or a time-matched search/RL benchmark.
  Preserve audited trajectories for future analysis; do not promote a baseline.

## 0.4.38 / research V1.5.0 — 2026-10-26

- Complete23 adaptive rounds. Frozen128-environment/both-seat confirmation gives
  retained V15081.61 (animal27.63, terrain53.98), V14982.37, pureRL85.64,
  matched1s rollout87.05. Animal gain+3.57 over V149 is offset by terrain-4.33;
  total delta-0.76 has95% paired interval[-2.58,1.06]. No baseline promotion.
- The main1280-game panel is complete; secondary0.1s panel stops at380/384 games
  after a deadline-signal/IPC failure. The exception drops main results from the
  top-level auto-report; docs/Harmonies_V150_Results.md records the intact evidence.

- Investigate the V149 animal/terrain tradeoff with scalar-control, component-value
  and separately labeled candidate-ranking heads. Keep original total-score objective.
- Use one behavior policy per MC dataset; distinguish habitat-continuation sibling
  targets from on-policy returns. Evaluate prediction quality and actual play separately.
- New authorized8h a226GPU0 window includes implementation and preflights, reserves
  final confirmation time and adapts subsequent data/search settings to observed results.

## 0.4.37 / V149 feedback experiment — 2026-10-26

- First pilot exposes weak successor ranking and high short-budget fallback;
  add a policy-evaluation loop using fresh games from one value-guided planner.
- Fit realized returns for that planner only, avoiding pooled continuation-policy
  targets. New paired development games test whether the resulting planner improves.
- This is one measured feedback iteration, not an unbounded self-improvement claim.
- Complete736games and new128-game feedback training. On new16environment pairs,
  feedback turn-value84.44 vs parent79.66 (deltaCI includes zero), RL86.22 and
  matched1s rollout88.63. Animal28.41 is offset by terrain56.03; no baseline promotion.
- Held feedback-return RMSE drops34.79→10.14, demonstrating that prediction
  improvement alone is insufficient evidence of a stronger decision policy.

## 0.4.36 / research V1.4.9 — 2026-10-26

- Test learned remaining-score values as substitutes for long continuation rollouts
  under0.1/1s per-atomic-call budgets, separately from the old900s/game reference.
- Keep fixed-opponent own-score semantics; compare identical candidate generation
  with one-step, two-step, turn-boundary and terminal cutoffs. No pure-policy replacement claim.
- Separate raw-return prediction from noisy candidate-Q ranking and paired full-game
  strength. Reinitialize new value models before grouped validation; preserve old values.
- Use CPU inference and independent single-GPU fitting on authorized a227; the first
  bounded campaign does not renew old training budgets or automatically promote a model.

## 0.4.34 / research V1.4.8 — 2026-10-24

- V147 final512x2 best model86.17285 versus plainPPO86.12012: no reliable extra
  expert-method advantage. Fresh16x2 search98.6875 versus stack98.78125: no reliable
  replacement teacher. Preserve both negative findings and prior artifacts.
- Move from fixed continuation at six roots to multi-decision public stochastic
  MCTS with growing candidate sets, then learn root visits and real terminal returns.
- Separate pure terminal-rollout MCTS from prediction-gated value leaves; preserve
  raw own-score objective, frozen opponent, exact rules and hidden-future boundary.
- Adopt search-policy/value iteration with full-game parent fallback instead of
  assuming more BC-plus-PPO must preserve animal skill. Learned priors return to search.
- This release establishes implementation and execution gates only. New900s paired
  comparisons and512-environment confirmation require the next authorized run window.

## 0.4.33 / research V1.4.7.1 — 2026-10-23

- First24-environment expert corpus does not yield a stronger free-playing BC
  policy: all four initializations revert to the total-score parent on development.
- Preserve a complementary animal-rich checkpoint for complete-game PPO repair
  instead of discarding every non-champion initialization; retain independent
  total-score control and explicit non-duplicate fallback exploration.
- This is a result-driven change within the original ten-hour experiment, not
  a positive RL result, a new rule setting, or a renewed resource budget.

## 0.4.32 / research V1.4.7 — 2026-10-23

- V146 fresh16 paired development: search98.75 versus model82.875, animal44.906
  versus12.188;48 actual training demonstrations ready. These are teacher, not new RL results.
- Diagnose low duo search terrain and own early board fill; explore space,
  stacking, terrain balance and opponent-horizon continuations at900s/game.
- Bridge audited expert actions/relative candidate Q/MC returns to initialization,
  then own-score complete-game PPO with decaying off-policy auxiliary supervision.
- Three development-driven learning rounds, immutable train/dev/confirmation
  namespaces, retained controls and independent512-environment pure-model confirmation.
- New user-authorized ten-hour window; rules, scoring, information permissions
  and historical artifacts unchanged. No promise of reaching solo score or optimality.

## 0.4.31 / research V1.4.6 — 2026-10-23

- Switch from own-replay/plan-credit PPO variants to an official-duo teacher
  strength diagnostic before generating additional expert data.
- Fixed retained V145 learner and V144 opponent, own absolute score, both seats;
  compare direct model/habitat and900-second mixed-root terminal search with
  habitat versus student continuation. No solo transitions or hidden futures.
- Screen4 development environments, expand promising search on16 NEW environments;
  require paired score improvement plus animal gain before48 fresh training games.
- Preserve candidate Q separately from actual MC returns and coherent trajectories
  for subsequent imitation/RL. This release does not claim a newly trained model.
- Old V145 stopped by user-approved task switch, not completed confirmation;
  remaining experiments retain its original absolute09:16 deadline.

## 0.4.30 / research V1.4.5 — 2026-10-23

- Test nonstationary opponents, animal-plan credit and reuse of good own
  experience separately, starting from the same retained V144 own-score policy.
- Keep all training/evaluation official duo; no forced-refresh curriculum score
  is used for selection. Potential differences conserve each true total return;
  positive-advantage replay is a separate off-policy auxiliary, not stale PPO.
- Three adaptive rounds retain control/winner and test combinations, opponent
  mix, return horizon or exploration based on development score/animal evidence.
- Fresh common development256 selection, then independent512 with both seats
  and two fixed opponents, paired against V144 parent and V142 transfer.
- New eight-hour allocation includes setup and final evaluation; no outcome or
  animal-skill improvement claimed before the corresponding evidence exists.

## 0.4.29 / diagnostic V1.4.4.1 — 2026-10-23

- Retain own-score as the official-duo objective: common development128 yields
  84.34 versus76.01 for the margin control; final512 confirmation was not reached.
- Same-parent16-environment audit exposes different animal-card availability
  between forced-refresh curriculum and real two-player market transitions.
- On15 common environments reaching own-turn11, curriculum94.73/animals22.80
  versus official self-play79.67/animals8.07. Prefix observations, not altered
  rules, a causal decomposition, an independent confirmation or an optimality bound.
- Prioritize testing slow-market and early animal-plan execution transfer;
  no new algorithm result, long campaign or expired-budget renewal in this repair.

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
