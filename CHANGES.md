# Engineering changes

## 2026-10-24 — V148 authorized ten-hour launch

- Start new immutable02:03:40–12:03:40CST window on b35/a226, frozen at published
  f9ea6d2/sourcea072bfcb; preserve prior windows and all earlier artifacts.
- Recheck half-node CPU limits and authorized a226GPU0–3; all four realA100
  gradient/exact-resume/portable/full-game gates pass on the launch snapshot.
- Complete two1200-step value initializations and hand off the eligible leaf
  evaluator to the first40 paired matches, including32 bounded900s searches.
- Add read-only live process/all-thread/device/provenance audit and publish
  repeatable run/session/log/deadline pointers. No completed search gain claimed.

## 2026-10-24 — 0.4.34 / V148 public MCTS and search-policy learning

- Add exact-duo public-outcome MCTS, progressively widened legal candidates,
  fixed-policy opponent handling, cumulative900s decision accounting and complete-only backups.
- Archive real root visits separately from legacy expert actions and unsearched
  fallbacks; replay masks, public planners and true remaining own-score returns.
- Add standalone policy/value learning, parent-KL variants, exact optimizer/RNG
  recovery, portable policies and separately gated score-value leaf artifacts.
- Reuse existing CPU/GPU schedulers for paired search, train-data handoff, learned
  priors returning to search and separate512-environment pure-model confirmation.
- New windows require explicit start/duration; no expired V147 budget renewal.
  Restrict mixed jobs to a226GPU0–3 and <=half physical CPUs on every host.
- Related53 CPU checks pass; actualA100 gradient/restore/export/full-game and
  four-update fit gates pass. Fix the small-gate inherited13-worker configuration.
  Six bounded b35 calibration games pass; no900s strength result or long launch claimed.
- Final13 V148 checks additionally cover value-leaf backup and resuming with
  retained old failure records; preserve cumulative controller costs on recovery.

## 2026-10-23 — V147.1 verified independent PPO and continuing search

- Publish4f9fac3 and verify upstream SHA; launch learning_b with frozen source
  3dba189b and the original19:08 deadline, while original b35 source726d6f76 continues.
- Re-pass all four actualA100 expert/PPO/exact-resume gates, then retain the
  cautious600-step animal-skill checkpoint separately from the total-score parent.
- Observe four distinct PPO arms beyond4 complete batches with nonzero updates;
  audit all-thread57-core ownership and actual authorizedGPU0–3 assignment.
- Independently replay32 first-batch matches/3636 actions. Both active campaigns
  have no failed jobs at09:47; no partial search panel or final-model gain claimed.
- Preserve first attempt15-batch checkpoints and costs; publish repeatable launch,
  recovery, source and deadline evidence without committing generated artifacts.

## 2026-10-23 — 0.4.33 / V147.1 retain complementary imitation skills

- Actual first-run four-A100 gates passed: expert gradients, whole-game PPO,
  exact next-update restoration, frozen-opponent and portable weight parity.
- First four BC initializations all selected the original parent on total-score
  development, creating a duplicate pure-PPO start. Preserve this negative result.
- Retain an animal-skill checkpoint under a bounded total-score deficit for PPO
  recovery, separately from the overall champion; use a distinct exploration arm
  if no complementary skill exists. Never call the weaker BC skill an improvement.
- Safely stop only the verified learning controller, preserve all checkpoints,
  and prepare learning_b inside the SAME19:08 deadline. Frozen b35 search continues.
- Combined learning cost references the stopped first attempt; no old data or
  weights deleted, no other GPU jobs touched, no new budget window.
- Targeted16 tests passed in30.03s; two focused complementary-checkpoint/choice
  tests passed in13.86s (overlapping coverage, not an additional18-test claim).

## 2026-10-23 — 0.4.32 / V147 search-to-RL exploration

- Add public-only space/stack/terrain/horizon search continuations, paired
  result-driven panels, retained-teacher fallback and train-only trajectory generation.
- Reconstruct real V146/V147 expert games with independent replay, two planners,
  native feature/mask alignment and explicit candidate-Q versus actual-MC targets.
- Add full-game-selected BC initialization and separate expert auxiliary updates
  after complete-game PPO; reuse existing native sampler and safe resume lifecycle.
- Parameterize the old trainer's training-seed namespace and worker/search hooks
  without changing defaults; keep V145 portable inference ABI and old frozen runs.
- Add a new immutable09:08–19:08 window, separate CPU/GPU campaigns, half-node
  CPU limits, four authorized independent GPUs, deadline guardians and512 confirmation.
- Implementation and protocol only at this entry; actual validation/launch evidence
  is recorded separately after checks, without predicting research gains.
- Related CPU regression111 passed/one old CUDA-only skip in169.74s; explicit
  on-device expert-gradient/PPO/exact-resume gates remain mandatory at launch.

## 2026-10-23 — V146 frozen launch and CPU ownership audit

- Publish implementation019d547 and verify GitHub branch SHA, then start the
  b35 frozen campaign with the exact mature-model preflight source/config.
- Verify16 complete/replayed model/habitat controls and16 progressing900-second
  search games, no failed games at the snapshot; do not report partial scores.
- Add a read-only live-worker/thread-affinity audit, using the controller CPU;
  verify17 currently active physical cores within the33-core allocation/half-node cap.
- Record run/session/source/deadline and remaining teacher/data gates. No new
  GPU training or teacher improvement is claimed at launch.

## 2026-10-23 — 0.4.31 / V1.4.6 official-duo search teacher

- Safely stop V145 through its verified controller; retain and hash-check all
  interrupted round2 latest/final/best checkpoints and the completed first round.
- Adapt terrain/habitat ranking to five-slot official duo; branch only through
  the real duo simulator, with public future resampling and isolated planner state.
- Batch CPU model continuations, compare habitat/model terminal rollouts under
  cumulative900-second focal budgets, and archive/replay all real actions/returns.
- Add frozen-source paired4/new16 gating, conditional48 train-only demonstrations,
  half-node CPU/shared-lock checks, per-game/global watchdogs and complete-game resume.
- Preserve original09:16 deadline; b35 CPU only, no new GPU allocation. Strong
  teacher and new learned-model gains remain empirical questions, not release claims.
- Initial related regression109 passed/one CUDA-only skip; mature b35 preflight
  verifies126-action policy parity and a six-candidate730-step terminal round.
- Final targeted49 passed, including actual child games, shared-lock exclusion,
  no-new-work resume and rejection of corrupt public observations/result metadata.
- Synchronize public package version with the research release; old frozen runs
  remain self-contained and unchanged.

## 2026-10-23 — V145 actual CUDA, replay and autonomous launch verification

- All four logical slots pass real-A100 gates on authorized GPU3: exact warm
  start, both-seat/current-only roles, raw/shaped reward replay, stationary
  opponent integrity, portable export and exact next-update SIL restoration.
- Verify four formal jobs beyond30 batches and identical new development
  baseline86.5586/animals14.0547; early fixed-opponent signal is not confirmation.
- Independently replay512 archived matches/63,050 actions/640 learning paths;
  audit57-core affinity and actual GPU3 ownership with a reusable read-only script.
- Record frozen source, stage deadlines and original09:16 stop. Three-round
  adaptive learning and final512 confirmation remain ongoing background work.

## 2026-10-23 — V145 safe shared-GPU resource adaptation

- Original four-device startup safely refused newly busy GPUs0–2; no other
  process was stopped and no V145 CUDA/training work ran in that attempt.
- Explicitly map four independent experiment slots onto idle authorized GPU3,
  retain57 CPU cores and the same09:16 deadline in a new preserved run directory.
- Account physical GPU time by union of occupied intervals per UUID, report
  logical slot time separately, and validate shared-slot authorization explicitly.
- Re-run all14 V145 tests including shared-device accounting: passed in55.47s.

## 2026-10-23 — 0.4.30 / V1.4.5 official-duo animal-skill learning

- Reuse V144.1 persistent evaluation/recovery lifecycle through explicit trainer
  hooks, keeping old schemas/default behavior and portable models intact.
- Add stationary official-duo opponent variants, public owned-animal potential
  differences with exact raw-return conservation, and bounded own-replay SIL.
- Archive raw/shaped rewards and MC targets, independently reconstruct plans
  during replay, and retain replay chunks/RNG for exact same-trial restoration.
- Add development-driven combine/ablate/credit/exploration rounds, prefix/card
  diagnostics, fixed-parent paired confirmation and bounded four-GPU campaign.
- Preserve all previous sources/artifacts; no rule change or improvement claim
  before actual evaluation. New01:16–09:16 Beijing window includes verification.
- Relevant CPU regression100 passed/one legacy CUDA skip, including backward
  V144 behavior, potential/replay recovery and small real round-to-confirmation.

## 2026-10-23 — 0.4.29 / V1.4.4.1 evaluation and recovery repair

- Reuse completed environment pools for greedy evaluation; preserve reference
  trajectories/RNG, add typed worker diagnostics and bounded group cleanup.
- Checkpoint complete PPO updates before evaluation, resume pending evaluation
  without duplicate updates/curve rows, and promptly export best portable policies.
- Retry failed jobs immediately on their original resource slots while healthy
  peers continue, retaining attempt limits and original absolute deadlines.
- Add worker/controller/checkpoint failure injection, bounded real-A100 checks
  and same-parent per-turn/card-visibility diagnostics; preserve old failed runs.
- Verify87 relevant tests, one legacy CUDA skip, plus real-A10020-panel pool
  reuse/parity, injected timeout recovery and exact next-update restoration.
- Close the stale V144 launch status honestly: r2 incomplete, no r3/final512,
  no silent budget renewal or new best-model claim.

## 2026-10-22 — V144 real-CUDA and background iteration launch evidence

- Four actual A100 gates pass: score/margin rewards, full-return and diverse-role
  collection, actor/context updates, portable policies and exact next-update restore.
- Verify common initial score80.25/animals9.6406 against two fixed opponents;
  replay512 matches/62,036 actions with1,024 learner trajectories and aligned seeds.
- Audit actual57-core and first-four-GPU ownership; all four formal trials exceed
  21 PPO batches. Record frozen source and unchanged22:59 Beijing absolute deadline.
- Preserve outcome boundaries: three-round adaptive selection and independent512
  confirmation remain background work, not an already achieved score improvement.

## 2026-10-22 — 0.4.28 / V1.4.4 own-score autonomous duo exploration

- Version score rewards AND score checkpoint selection together; retain V143
  margin-only defaults and reuse its accelerated both-seat collection/PPO.
- Add immutable two-opponent, paired-seat development/confirmation evaluation,
  own-score/history/full-return variants and result-driven three-round choices.
- Keep all policies/traces, exact same-trial resume, weight-only warm starts,
  fixed eight-hour deadline, bounded retry and four-GPU/57-core ownership.
- Preserve stopped V143 checkpoints and export its development-selected frozen
  reference; no game rules, raw scoring, information permissions or sealed tests changed.

## 2026-10-22 — V143 real-CUDA self-battle and launch evidence

- All four mature-weight A100 gates pass: full-game transfer error0, both-seat
  learning, historical-role exclusion, frozen-opponent integrity and exact resume.
- Record official initial self-battle256 matches:79.4883 score/8.7461 animals;
  explicitly retain the curriculum-to-duo gap instead of claiming116.70 transfers.
- Replay512 real archived matches/62,202 actions with1,024 learner trajectories;
  audit57-core/first-four-GPU bounds and actual updates beyond21 batches each.
- Record immutable source, original18:45:23 Beijing deadline and recovery paths;
  independent confirmation and cross-method battles remain pending in background.


## 2026-10-22 — 0.4.27 / V1.4.3 official-duo transfer and self-play

- Preserve V142 own-view logits at initialization; add zero-output trainable
  public-opponent residual without changing official duo rules or information.
- Reuse accelerated official-duo pool and PPO; train both current-policy seats
  on separate complete-game trajectories and exclude historical-policy actions.
- Add current/current and current/history replicas, exact recovery, replayable
  full traces, paired-seat evaluation and post-selection cross-method battles.
- Add CPU/real-CUDA gates and bounded four-GPU/57-core campaign with original
  absolute-deadline resume. Old environments, models and sealed tests unchanged.


## 2026-10-22 — V142 actual-device launch verification

- Four A100 actor/context-update and exact-next-update gates passed; behavior
  ratios match within7.11e-15. Replay512 archived games/36,746 actions exactly.
- Record common initial128 score/animal/turn/token comparison before formal PPO;
  four independent jobs each exceed23 real batches with ratio errors below4e-14.
- Audit57 physical CPU cores and GPU0–3 only; frozen source and new absolute
  12:16:44 Beijing deadline preserved. Publish recovery/status pointers and
  distinguish initial/early development observations from pending confirmation.


## 2026-10-22 — 0.4.26 / V1.4.2 own-score random-market curriculum

- Add separate five-slot,13-own-turn random-market environment with no opponent
  board/score; retain original solo/duo rules and reuse terrain/animal transitions.
- Reuse bitsets, native features, shared-memory collection and optimized PPO;
  preserve weight compatibility under explicit zero-opponent/new-layout encoding.
- Compare solo/duo warm starts, two replicas each, with fresh Adam and own-score
  development selection. Archive full replayable trajectories and exact resumes.
- Add rule/codec/pool/update/replay tests and four actual-CUDA gates, paired
  initial comparison, fresh confirmation512 and bounded57-core/four-GPU campaign.


## 2026-10-22 — V141 measured speedup and verified continuation

- Four A100 PPO/exact-resume gates passed. Equal-resource128-match/13-worker
  paired collections:9.0466s to5.3057s (1.705x), identical complete greedy paths
  across three panels; mature-weight logits error4.00e-15/gradients2.78e-17.
- Gracefully preserve all four V140 trials atbatch490, migrate full state and
  verify actual continued PPO updates on GPUs0–3 with57 physical CPUs.
- Verify11 new batches per trial (about1.60–1.62x live training throughput),
  behavior-ratio error below4e-14; replay128 new matches/15,578 actions.
- Keep original07:27:07 deadline and all parent artifacts; new runs do not reset
  learning, best-checkpoint selection or the budget. This is a speed result,
  not evidence of improved game scores.

## 2026-10-22 — 0.4.25 / V1.4.1 duo sampling acceleration

- Reuse each public view's representation across actor, gate and context heads;
  batch three encodings and two advice computations without changing weights.
- Group collection by model identity (learner/current share one call), transfer
  features/masks once and fetch action/logp/value together. Preserve role labels.
- Add FP64 forward/gradient and complete-trajectory parity, identity-grouping,
  exact-resume and full-state migration tests; actual A100 paired speed gate.
- Continue in new frozen-source runs with preserved Adam, opponent pool, RNG,
  best checkpoint, counters and original deadline. Sampling random draw order
  changes across versions; only same-version resume is claimed bit-exact.

## 2026-10-22 — V140 launch verification and artifact handoff

- Record four actual A100 mature-parent CUDA update/exact-resume gates, matching
  behavior probabilities, real PPO progress and57-core affinity audit.
- Replay128 archived matches/15,280 real actions independently. Record the frozen
  initial paired-seat baseline and absolute07:27:07 campaign deadline.
- Publish source+solo-weight bundle pointer and digest; outputs remain local,
  and ongoing training is not reported as a completed confirmation result.

## 2026-10-22 — 0.4.24 / V1.4.0 two-player environment and baseline

- Implement separate official duo-A/base32 rules: shared five-slot markets,
  equal-turn final round, animal-count ties; preserve all solo mechanics.
- Reuse bitset geometry, native batch features and shared-memory workers.
  Explicit244-action/4865-feature layout represents both boards and all markets.
- Add solo skill transfer, per-seat public planning, on-policy competitive PPO,
  lagged/frozen opponents, paired-seat evaluation, portable duo inference and
  replayable complete trajectories. Add actual CUDA/resume gates and bounded
  four-job a226 campaign;57 physical CPUs and only the authorized first4 GPUs.

## 2026-10-22 — 0.4.23 / V1.3.8 portable solo delivery

- Export the development-selected V137 control_a into inference-only FP64
  weights; strict artifact validation, no ancestral data/checkpoint loading.
- Add standard Solver and single-thread timed-runner integration, exact public
  planner/native feature parity tests, committed-source+weight packaging.
- Validate fresh512 games and32 paired single-thread3-second attempts; all32
  finish naturally, zero failures. Preserve old UI pointers and sealed tests.

## 2026-10-22 — 0.4.22 / V1.3.7 closeout and publication

- Publish the accumulated trajectory, own-policy learning and goal-selection
  work through V137; preserve original sources, checkpoints and results locally.
- Update the stale README and implementation status with final paired results,
  incremental costs and explicit portable-export/deployment acceptance gaps.
  No runtime, rule, policy or experiment configuration changes in this closeout.
- Record the user's standing repository instruction to push every completed,
  verified update and check the remote branch SHA. Generated weights and run
  directories are not included in the source publication.

## 2026-10-20 — 0.4.16 trajectory preservation

- Preserve existing actual-game logs in a resumable content-addressed gzip archive
  with per-campaign provenance, split metadata, behavior identity and all outcomes.
  Keep original artifacts, failed attempts and legacy raw evidence intact.
- Add a fail-closed train-only reader: replay the complete episode before returning
  transitions, reject held-out environments and never substitute teacher suggestions
  for student behavior actions or fabricate unknown behavior probabilities.
- Default the shared search episode logger to public pre-observations and record
  exact environment rewards and termination/truncation flags. No rule, scoring,
  policy, RNG, search-budget or training changes.
- Byte/hash verification is separate from game replay. Search-internal rollouts
  are not claimed to be archived; this is a local copy, not an offsite backup.

## 2026-10-20 — 0.4.14 / V1.2.10 persistent teacher-guided PPO

- Add three restartable rounds of student-only roll-in, public-search correction
  and native-collector PPO from the same mature V129 parent. Preserve trained
  critic/Adam and lifetime counters; use paired new RNG streams for the first round.
- Keep PPO-only and static-demonstration controls alongside two persistent
  correction strengths, two seeds each. Teacher returns never become PPO rewards
  or current-policy value targets; all headline evaluation stays model-only.
- Use FP64 NumPy roll-in matching the actual actor, replay-validate every label,
  reject insufficient search samples and forced-only correction datasets.
- Reuse existing frozen CPU watchdogs, GPU packing, resource locks, checkpoint
  and exact-next-update recovery. b35 supplies CPU-only labels; a218 usesGPU0/1,
  four independent jobs/card,66aggregate CPU cores including outer control.
- Preserve all sources/checkpoints/failed preflights and report per-environment
  scores, paired controls, candidate estimates, costs and retained model pointers.
- Verify363regression tests, actual exact-next-update CUDA recovery onGPU0/1,
  and a full three-round cross-host smoke with192labeled games and24completed
  PPO trials. Low-budget smoke scores are not method-selection evidence.


## 2026-10-20 — 0.4.13 / V1.2.9 native sampling continuation

- Package and hash-pin the verified C++ public encoder; batch each worker's live
  observations and skip unused internal reset metadata, preserving original
  transitions, legal actions, scoring, RNG and synchronous PPO semantics.
- Keep reference/fast backends intact. Require explicit native artifact identity;
  no automatic compiler install, silent fallback or fast-math.
- Add guarded interrupted-parent import and remaining stage-time carryover.
  Preserve all eight optimizer/RNG checkpoints and the cumulative10M action goal.
- Extend the frozen two-GPU benchmark with mature-checkpoint imports, exact CUDA
  next-update resume checks and fast/native/native/fast eight-job comparisons.
- Add pool/reset/STOP/equivalence/provenance/PPO/budget regression tests and
  recoverable migration runbook; no game/learning/formal-test changes.
- Verification:355passed/12explicitGPU skips;32actual CUDA benchmark processes
  pass exact-next-update recovery and share identical final model hash. Native
  PPO throughput+25.9%, environment-stage time-41.0% on matched two-card repeats.
- Launch all eight native continuations from6.19M-6.47M actions; verify65-core
  union, four CUDA jobs/card onGPU0/1 and64processes mapping the pinned library.
## 2026-10-20 — V1.2.8 isolated native encoding feasibility

- Keep the launched two-GPU PPO snapshot untouched. Add a standalone single-CPU
  C++ public-feature encoder and train-only matched replay benchmark, not a new
  production environment or learning method. Preserve all1346 float32 features.
- Include Python packing and FFI in timing: encoding4.6–6.6x, fixed-trace full
  environment replay1.65x locally; do not claim an end-to-end PPO speedup.
- Verify all32 cards/counts, complete trajectories, input bounds and isolation;
  record the independent CPU cost, reproducible source copies and compiler flags.
- Document residual reset metadata/packing/synchronization costs and staged
  native-batch integration gates. Full C++ transitions remain unimplemented.

## 2026-10-20 — 0.4.12 / V1.2.8 exact environment acceleration and GPU packing

- Add opt-in fast Harmonies observations, compiled animal predicates, cached
  action maps and environment reuse; preserve original transitions, scoring,
  action order, public information and synchronous PPO sampling semantics.
- Benchmark real matched PPO updates on a218 GPU0/1: identical final model hashes
  across reference/fast, worker counts and packing. Single-job speed improves
  57%; packed eight-job throughput improves31% at121cores or15% at65cores.
- Continue all eight completed V127 trials with model, optimizers, RNG and counters
  preserved; explicit guarded cross-source import, unchanged learning settings,
  new bounded stage and immutable old artifacts. Two GPUs/four trials per GPU.
- Account shared physical GPU occupancy by interval union, preserve separate
  process hours, and include benchmark/preflight/failed-attempt costs once.
- Add engine/encoding/trajectory/PPO/resume/resource checks; verify mature old
  checkpoint continuation and exact-next-update recovery on both target GPUs.

## 2026-10-20 — 0.4.11 / V1.2.7 knowledge-initialized online PPO

- Stop the authorized V126 imitation/search pipeline, preserving teachers, models,
  pilot results and interrupted attempts with their original cost ledger.
- Reuse the masked compact policy and batched public environment pool for genuine
  reward-driven PPO. Import hashed imitation weights; calibrate only the critic
  on own-policy returns, then optimize actor and critic with gamma1/GAE/PPO.
- Add four matched two-seed arms: BC initialization, soft-label initialization,
  fading demonstration auxiliary, and random initialization. Frozen actors are
  explicit controls; all headline evaluation is model-only natural-terminal score.
- Add exact update-boundary optimizer/RNG recovery, finite/legality/ratio gates,
  per-game traces, development curves and separate final paired evaluation.
- Add frozen-source a218 orchestration: eight independent GPU jobs,121 physical
  CPU cores aggregate, shared node lock, bounded task watchdogs and all-attempt costs.

## 2026-10-20 — 0.4.10 / V1.2.6 a218 eight-GPU handoff

- Add host-scoped a218 resource profile,128-core physical allocation ceiling,
  reserved orchestrator core, and enforce b35 CPU-only for new pipeline launches.
- Import completed b35 teachers only after exact cohort, source/plan/split hash,
  natural-terminal and replay checks; carry original costs once without rerunning games.
- Expand first training batch to BC versus conditional-Q ×128/256 hidden ×two
  matched seeds on eight independently validated GPU UUIDs. CUDA-required runs
  fail explicitly if GPUs become unavailable instead of silently running on CPU.
- Preserve frozen original runs; new host/run identity and checkpoint lineage.
  Add host/resource/provenance regression tests and opt-in CUDA training/export/recovery check.

## 2026-10-20 — V1.2.6 immutable child-source inheritance

- Force every later CPU stage to inherit the executing outer pipeline snapshot,
  never a changed worktree; retries still use the validated parent snapshot.
- Add direct regression coverage. Freeze-only full suite321passed/9 CUDA skips;
  final scoped55checks pass. Verify actual calibration child source equals outer.
- Stop only the first waiting V126 orchestrator before any stage began, preserve
  its record, and carry its98.086core-second wait into replacement preflight once.
  V125 experiments remain untouched; replacement run identity ends in `_b`.

## 2026-10-20 — 0.4.9 / V1.2.6 search-guided learning

- Add a frozen, independently bounded teacher -> BC/Q-soft-target students ->
  900-second root/continuation ablations -> student-state relabeling -> retraining
  and fresh-development comparison pipeline. Reuse existing encodings, MLP,
  public simulator, natural-terminal runner, campaign locks/watchdogs and reports.
- Generate new train-only teacher data, replay every labeled transition, split
  by environment, keep unestimated candidate Q masked and continuation-specific.
  Export FP32 NumPy inference and verify agreement with training-time Torch.
- Retain all students and controls; select correction direction by full-budget
  search, never low-budget student scores. No reward/rule changes, formal tests,
  production promotion, value truncation or from-scratch PPO sweep.
- Add resumable model/optimizer/RNG epochs, immutable hashes, per-stage attempts,
  shared CPU reservation guards, conservative resource ledger and live reports.
- b35 GPU authorization received, but every visible idle GPU fails actual CUDA;
  direct system-driver cuInit returns999. Probe again before training and fall
  back to CPU without resetting the driver or interrupting V125 experiments.
- Verification: three complete29-game train-only calibrations, each with3 real
  small-model trainings; final frozen-source calibration has0 failures and
  completed-task resume launches0 jobs. Initial full suite318passed/9 CUDA skips;
  latest scoped47tests and11 focused recovery/GPU-failure/provenance checks pass.
  Concurrent source editing correctly tripped one old provenance gate in a
  second full sweep; its exact frozen-source rerun passes, guard unchanged.

## 2026-10-20 — V1.2.5 retained-policy expanded evaluation

- Freeze five retained policies on the exact historical32-environment/3-search-
  seed development matrix:480900-second games, including rerun original/habitat
  controls. No strategy, game-rule, deployment or formal-test changes.
- Add an explicit b35-only reserve16-physical-cores profile, capped112 and
  requested101cores; preserve the default half-machine guard for every old run.
- Freeze288 historical900-second records from finalized parent/expanded runs,
  verifying ownership, source, rules, job identity and all seed repetitions.
  Embed the snapshot in the new plan; report historical and same-batch contrasts
  separately without treating search repetitions as independent environments.
- Retain bounded task/worker watchdogs, frozen-source recovery and resource
  accounting; validate100-worker concurrency on train, never prune by2s scores.
- Validation:299CPU tests pass/21resource-skipped;26 b35 lifecycle/guard checks
  pass;100/100 train calibration games finish. Launch480 frozen jobs at05:24CST,
  verify100live single-thread workers with exact core bindings; results pending.

## 2026-10-20 — V1.2.5 stage1 completed

- Complete three genuinely adaptive rounds:64/64 natural-terminal games on12
  small-cohort development environments; no confirmation/test or GPU usage.
- Retain mixed roots:117.417 vs original106.833 on the same12 environments,
  9wins/3losses; vs habitat+8.125 on eight shared environments. These are adaptive
  development results, not an optimality or independent-test claim.
- Archive negative continuation/width experiments, retain paired-uncertainty
  selection for follow-up, and preserve separate original/habitat/compiled controls.
- Export4663 aligned decisions; replay all64 trajectories and terminal scores
  exactly. Final CPU regression284passed/21resource-skipped. Record21.343
  conservativeCPUcore-hours including calibration/recovery/local verification.
- Publish concise results and exact recovery/data pointers; all experiment
  processes exited. No production strategy or game-rule changes.

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
