# Engineering changes

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
