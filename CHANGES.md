# Engineering changes

## 2026-10-20 — V1.2.5 round3, driven by round2 results

- Round2 completes24/24 games. Mixed roots116.50 beats original93.50 on this
  four-environment development cohort; portfolio110.75 and terrain finish105.00
  do not beat mixed roots. Exact compilation adds about9.9% simulated steps but
  has no clear score improvement. Small adaptive cohorts are not final test.
- Test broader10-root coverage and a paired-standard-error selection penalty,
  keeping raw terminal-Q estimates separate from heuristic selection targets.
- Use four new development environments104–107,20games/21cores and unchanged
  per-game900-second caps. No low-budget quality pruning or GPU use.
- Replay all44 completed first/second-round games from environment seeds:
  all3173 before/action/after observations and terminal scores match exactly.

## 2026-10-20 — V1.2.5 round2, driven by round1 results

- Round1 completes20/20 games: mixed roots114.00 vs original113.00 and habitat
  103.25 on four paired environments. Animal gains largely trade off terrain;
  simple goal caching adds only0.8% simulated steps. Retain all controls and do
  not claim a reliable advantage from this small selected cohort.
- Add exactly equivalent stack-extension goal compilation, a common-round
  original/habitat continuation portfolio, and a public late-game terrain switch.
- Freeze four fresh development environments100–103, one shared search seed,
  six900-second methods;24workers plus control, no large fixed expansion.
- Verify compiled goals and action rankings against original implementation,
  common-policy-per-round sampling and legal/budgeted play;40 scoped tests pass.

## 2026-10-20 — 0.4.8 / V1.2.5 recovery and aligned demonstrations

- Fix late binding of the optional goal function so existing diagnostic
  monkeypatches retain their original meaning; no policy change in production.
- Add explicit opt-in lifecycle hooks for inheriting completed attempts and
  freezing the exact parent source on recovery. Reject live/corrupt parents,
  changed task identities and pre-recovery drivers; do not silently rerun jobs.
- Export public before/action/after demonstrations with terminal return,
  score-delta return-to-go and candidate uncertainty, retaining source/split IDs.
- Consolidate registered tasks, per-environment results, research decisions and
  deduplicated resource receipts. This is not a fixed strategy-selection script.
- Round1 remains on its immutable0.4.7 snapshot; no hot changes to live workers.

## 2026-10-20 — 0.4.7 / V1.2.5 small-cohort autonomous iteration

- Add isolated 900-second cumulative-game search variants, exact goal-cache
  continuation, interleaved terrain/habitat roots, and resource/terrain-aware goals.
- Retain original/habitat controls and public-only terminal simulation; preserve
  balanced completed rounds, candidate means/variance and per-decision random streams.
- Reuse frozen-source campaign/watchdog and matched reports; each small task gets
  its own deadline, no inherited08:00 or stage-wide compute cap. First screen is
  four environments times one shared search seed, not a large evaluation matrix.
- Add opt-in before/action/after/raw-score-delta traces without changing existing
  episode defaults. Existing policy/episode hooks preserve old callers.
- Research selection and follow-up variants are decided from completed experiments;
  no production promotion, rule change, confirmation/test tuning, or GPU allocation.

## 2026-10-20 — V1.2.4 execution batch b, leave16physicalcores

- Honor the user's expanded CPU allowance:112of128physicalcores including
  controllers, no GPU, unchanged08:00deadline and original1056job identities.
- Drain the old dispatcher at a verified full-worker frontier without killing
  games. Freeze disjoint pending ownership; add48cores, borrow old worker cores
  only after exact-process exit, then transfer the old control core and CPU lock.
- Keep every worker on the existing frozen0.4.6policy source. Freeze/hash the
  independent expansion driver and record process identities, resource intervals,
  inherited costs and all failures. Do not mutate old source/plan/results.
- Mark merged results as mixed-load and report a separately frozen cohort of
  wholly unstarted environments. No new strategy tuning or production promotion.
- Verify4expansion checks, including real worker drain/no duplicate ownership
  and independent cleanup after a deliberately stopped new controller; rerun
  the existing10scoped policy/report checks. Frozen policy hash unchanged.

## 2026-10-20 — 0.4.6 full-game continuation and wall-cost controls

- Reuse the guarded CPU campaign and public-simulator episode runner through
  backward-compatible backend/factory hooks; new V1.2.4 code stays separate.
- Add common-root timed terminal search with complete-round-only evaluation,
  per-decision future streams, identical fallback and explicit partial-round costs.
- Add natural-terminal component/trace accounting, complete paired panels,
  measured-time comparison gates, training-only calibration and bounded tmux runs.
- Reject an overly front-loaded clock allocator on training-only fallback
  telemetry; replace it with public-rollout remaining-length allocation. Preserve
  both calibration snapshots/costs. Final-source calibration14/14 completes,
  with approximately30.01s actual times for all three30s controls.
- CPU regression269passed/9explicit-CUDA-skipped; rerun10scoped checks after
  the allocator revision and25final-source search/lifecycle checks, including
  real old/new cleanup paths and a deliberately stopped controller.
- Preserve old artifacts and user design changes; no old live processes remained
  at the authorized stop check. Publish source/config/tests separately from outputs.

## 2026-10-19 — 0.4.5 public-state teacher bottleneck diagnosis

- Isolate V1.2.3 from the running V1.2.2 and production: freeze64 public states
  from32 registered development environments, two producers, stage/action strata.
- Screen every legal root under original/space/habitat continuations; freeze
  discoveries before independent common-sample confirmation of the same root set.
  Preserve negative results, components, action types, traces and incomplete attempts.
- Add explicit stack/anchor/resource feasibility and compatible shared-terrain
  goals as diagnostic heuristics; retain true terminal score and original control.
- Cluster uncertainty by source environment, require complete fixed prefixes, and
  distinguish continuation gains from root-preference interactions.
- Queue behind the existing CPU lock; freeze source/bank/plan and use single-thread
  workers, independent deadlines, receipts and automatic reports. No new RL run,
  old-task interruption or deployment promotion.
- Verify256 distinct CPU cases across full regression and admission-safe lifecycle
  reruns;9 explicit-CUDA checks remain unexecuted. Final-source train calibration
  completes2/2 states with1296 terminal samples. Fix unreachable-card slot filling
  and label training calibration separately from development diagnostics.

## 2026-10-19 — 0.4.4 public terminal-rollout teacher and compute diagnosis

- Preserve the historical width6/rollouts4/depth2turn/cap384 short-search algorithm;
  scale actual sample counts and proportional caps at1/4/16/64x. Add a separate
  heuristic-to-natural-terminal teacher with public conditional future sampling,
  common candidate samples and an exact public-action cache, not learned dynamics.
- Freeze development panels and three search seeds; report components, actual
  solver/CPU/wall costs, root coverage, horizon/terminal fractions and every failed
  or censored attempt. Average search repetitions before environment-level pairing;
  only fully covered predeclared panels enter the main curve.
- Add frozen-source, single-thread CPU workers, a half-physical-core resource guard,
  process-level timeouts, independent absolute-deadline watchdog, durable heartbeats,
  and automatically refreshed JSON/CSV/SVG/Chinese reports with preflight accounting.
- Test public-information equivalence, legal terminal trajectories, sample-budget
  consumption, cache/restore equivalence, non-pseudoreplicated aggregation, real
  worker timeout/global deadline, and cleanup after a deliberately stopped controller.
- Final b35 CPU regression:238passed,9explicit-CUDA checks skipped; all three
  actual timeout/cleanup lifecycle checks pass. Training-seed calibration completes8/8
  games; retain both the1x gains and4x regressions without development-based tuning.
- Keep old PPO artifacts, test boundaries and production policy pointers unchanged.
  Calibration uses training seeds only; no policy-quality or optimality claim.

## 2026-10-04 — 0.4.3 mature-policy numerical repair and audited recovery

- First publish the pre-fix snapshot32d0c66, then reproduce the saved E11_T16_913
  failure on its original b35GPU5: deterministic FP32 batch-shape forward drift,
  1.763686e-4 at the unchanged1e-4 pre-update gate; identical-shape repeats agree.
- Add explicit *_T16_F64 variants for all four matched models. Keep old variants,
  initialization values, public inputs, full-game collector, objective and PPO
  hyperparameters intact; use FP64 forward/gradients/Adam, not a relaxed check.
- Add new-root numeric continuation with exact parent source/config/hash checks,
  preserved parameters/Adam/RNG/counters, complete-boundary progress-only recovery
  selection, inherited costs and explicit lineage. Ordinary resume remains strict.
- Verify226 CPU regression tests, plus9 actual CUDA tests including the retained
  failing-before/passing-after witness. All12 mature checkpoints pass76 rollout
  batches/777 optimizer updates with maximum probability ratio error4.09e-14.
- Preserve old sources/results/failure witnesses and deployment; resume four-card
  research independently after gates. Diagnostics are not formal curve points;
  no confirmation/final-test selection or policy-quality improvement claim.

## 2026-10-04 — V1.2.1 / 0.4.2 research snapshot before numerical repair

- Publish isolated E00/E01/E10/E11 factorial implementations, fresh disjoint splits,
  immutable research registry, source-bound checkpoints and guarded final-test entry.
- Add separately identified T16 batched collectors, shared-memory CPU environments,
  unchanged full-game PPO/STOP semantics, continuous GPU scheduling and pipelined
  immutable-checkpoint development evaluation with backlog backpressure.
- Include semantic, update, resume, subprocess and CUDA opt-in tests; measured b35
  end-to-end speedup3.42–3.72x with seven workers under eight-core allocations.
- Record all12runs reaching10M actions; E00_T16 development128 three-seed mean81.33
  versus frozenINC72.87. No confirmation/final-test or production promotion claim.
- Explicitly retain known late-training probability-consistency failure at batch12409,
  stopped-state provenance and the original1e-4 gate. Repair is not part of this snapshot.
- Pre-publication full regression:215passed,4opt-inCUDA checks skipped in the ordinary
  CPU suite; the separately reserved CUDA continuation checks passed in prior verification.
- Publish source/tests/research docs only; preserve generated weights, logs, failed
  witnesses and archived source locally, outside Git.

## 2026-10-04 — Source-preserving parallel V1.2 scheduling

- Add an independently hashed orchestration driver for concurrent method/seed jobs
  on idle b35 GPUs, with one job per card and the original shared eight-core affinity.
- Hand off only after the active child writes complete output; verify process identity,
  restore old scheduling on failed pre-handoff checks, and preserve/reuse completed work.
- Keep training source/checkpoints/hyperparameters intact. Serialize timed evaluation
  outside GPU training stages; retain original selection and publication gates.
- Sum per-device job reservations, include prior costs, and enforce both original
  14,400 GPU-second ceiling and original absolute deadline with an independent watchdog.
- Test actual concurrent CPU training subprocesses, checksum/config-checked reuse,
  evaluation exclusion, graceful handoff/fallback, PID reuse and overlapping GPU accounting.
- Isolate device-specific instability via identical captured inputs on five GPUs;
  exclude GPU1, retain strict probability checks, and verify real BC-to-PPO updates
  on GPUs4/5. Automatically reuse a stopped prior run without signaling unrelated PIDs.

## 2026-10-04 — V1.2 animal RL reliability and continuation

- Reproduce a real CUDA PPO log-prob failure; use float64 probability arithmetic
  without changing float32 networks or relaxing the 1e-4 consistency gate. Capture
  failing batches and keep a KL early-stop guard on optimizer updates.
- Add real on-policy finite-horizon animal diagnostics, exact value gaps, legal
  teacher-prefix replay, source-game/layout holdouts and explicit missing coverage.
- Add hash/source/split/seed-checked BC-to-PPO weight initialization with fresh Adam
  and rollouts; retain strict exact-resume behavior and inherited training lineage.
- Wire shared-network pure PPO, frozen BC and BC-to-PPO comparisons with independent
  controls, true-score/animal gates, explicit RL contribution and nominated-artifact checks.
- Continue the original allocation with all prior charges and its original absolute
  deadline. Preserve failed runs and deployed strategies until verified improvement.
- Verify 171 regression tests, including real BC-to-PPO CLI initialization, fresh
  rollouts, exact resume, held-out rejection and positive/negative promotion gates.
- No trained-policy performance improvement is claimed by this code release.

## 2026-10-04 — V1.2 bounded learning pilot and timed deployment

- Add versioned 4564/884/240 Harmonies heads with uniform STOP, slot/card public
  observations, strict metadata and exact optimizer/RNG training continuation.
- Add parent-owned timed_score_v1 settlement, killable policy lifecycle and framed
  deadline-aware IPC. Aggregate every attempt, preserve partial scores and timing tails.
- Add search demonstrations, no-chance endgame diagnostics, staged three-seed
  PPO/BC and fixed holdouts; distinguish warm-start and actual-wall-time objectives.
- Add source-frozen one-GPU/eight-core four-hour watchdog and full cost ledger;
  validation-only automatic promotion, verified legacy imports and rollback pointers.
- Add human UI countdown/STOP/expired-save handling and new-game policy refresh.
  Preserve all V1.1 artifacts; human setting-matched target remains unavailable.
- Publish code/plan/runbook, not an unmeasured performance improvement. Generated
  run status, weights, logs and scores remain in versioned outputs/v12 directories.

## 2026-10-03 — V1.1 review fixes and stronger baselines

- Reject cross-run train/holdout contamination before candidate evaluation; retain
  run-qualified identities and full provenance. Load each candidate's actual
  hidden width, network and encoding metadata rather than assuming 128.
- Add action-feature PPO and explicitly supervised heuristic imitation for Micro
  and TIE; retain Harmonies flat PPO. Add fixed-set diagnostic and three-seed plans.
- Add turn-aware, hard-budget search and animal-potential leaf evaluation, public
  information safeguards, component metrics and deterministic Runner seed shards.
- Verify 132 tests, 4992 completed frozen test games, 39 policy hashes and 15
  trained-policy isolated restores. Preserve V1 source-bound models and archive.
- Add real animal placement/completion/save browser scenario. Full three-game
  browser regression passed after compacting display-only policy labels; preserve
  the initial overflow screenshots and all substantive model metadata.
- Publish package 0.3.0 source and reproducible launch/report/audit/package scripts;
  generated training models, raw logs and screenshots remain in the local bundle.

## 2026-10-03 — ZIP restore regression

- Independent restoration from the first V1 ZIP exposed missing empty workspace
  directories. Preserve required empty directories explicitly in the ZIP manifest
  and entries, and add a real export/extract/Runner-restore regression test.
- Keep the failed candidate archive for diagnosis; the final archive uses a new
  filename. No game, policy weights, frozen evaluation or runtime source changed.
- Final full suite after the two packaging regressions: 107 passed.

## 2026-10-03 — V1 initial repository delivery

- Preserve V0.1 and add Micro Tiles, Take It Easy!, Harmonies solo A engines,
  public-information search, finite masked RL encodings and exact game saves.
- Add random/heuristic/bounded-search/PPO, real GPU training, validation-only
  selection, portable policy manifests and frozen common-Runner evaluation.
- Add three-game human/watch/hint/takeover/challenge UI and explicit-task replay.
- Verify 105 tests, actual browser interactions, three moved-policy restores,
  and 1920 completed frozen test games with no illegal actions or failures.
- Source is published separately from generated runs/weights; local review bundle
  includes trained/untrained artifacts, provenance, reports and raw evaluation logs.
