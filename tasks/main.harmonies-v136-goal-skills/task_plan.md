# Task Plan: harmonies-v136-goal-skills

## Stable Anchor

- Scientific question: Does teaching public animal-layout execution from expert segments improve mature model-only total score, and make goal selection learnable?
- Target claim or outcome: Reproducible V136 skills versus matched ordinary imitation and continued sticky PPO; conditional manager experiment only after execution evidence.
- Success criterion: Fresh paired environment total score gain, not animal-only gains; report two training seeds and independent confirmation512. Training-state probes diagnose execution separately.
- Constraints and budget: a226 GPUs 1-3 (GPU0 occupied by unrelated gmx); <= half physical CPUs, normally 43 cores. New bounded task: stage1 +4M actions/trial, stage2 +8M, <=4h guard plus explicitly accounted preflight. No renewal of V135 deadline.
- Non-goals: No rule/reward/information changes, online paid LLM, test tuning, or search at deployment. No claim of optimality or expert intent from hindsight labels.

## Current Cycle

- Working hypothesis: Correctly specified animal targets are under-executed; public goal-conditioned expert segments can teach coherent setup without sacrificing terrain skill.
- Main uncertainty: Existing candidate bank may omit expert targets; increased animal completion may reduce total score.
- Next decisive experiment or implementation: Train-only segment extraction, paired fresh-future goal execution probe, anchored adapter warm-start, then real-reward PPO.
- Expected pass/fail signal: Goal completion responds to target conditioning; total score does not materially collapse. Manager branch gated on training-only execution probe, not confirmation.
- Fallback: If gate fails, retain continued PPO and best imitation variant; report failed execution bottleneck rather than run another unguided manager PPO.

## Success Criteria

- [x] Immutable train-only data and fresh curve/audit/confirmation panels.
- [x] Probability, masks, no-leakage, real gradient, exact resume, actual CUDA checks pass.
- [x] Detached bounded training running with saved checkpoint and real completed PPO updates.

## Parallel Tracks

| track | owner | mode | worktree / branch | dependency | deliverable | status |
|---|---|---|---|---|---|---|
| primary | primary | integrate | current | none | final synthesis | active |

## Phases

### Phase 1: Discover and specify

- [x] Confirm live state and relevant owners.
- [x] Record the smallest implementation and verification contract.
- **Status:** complete
- **Evidence:** V135 raw evaluation artifacts complete but final JSON failed numpy.bool_; a226 GPU0 gmx, GPUs1-3 idle at 15:32 CST.

### Phase 2: Implement or run the decisive test

- [x] Make the smallest coherent change or launch.
- [x] Record only decision-relevant evidence and material checkpoints.
- **Status:** complete
- **Evidence:** commit aa9c213; source219401d9dd401b861144878ec6ad39be4809530bcc5c7f12bf4206018c9387c2; real CUDA+resume passed. live_handoff.json verifies6 trials,8–9 batches each, saved checkpoints and actual gradients.

### Phase 3: Interpret and hand off

- [ ] Compare the evidence against the stable anchor and expected signal.
- [ ] Close remaining phases as complete, superseded_by_evidence, or abandoned_with_evidence.
- [ ] Commit/push coherent work and archive or hand off the task.
- **Status:** in_progress
- **Evidence:** Live handoff verified; final experimental comparison remains pending. Guard15:50:18–19:50:18CST,2026-10-21.

## Decisions And Blockers

| Item | Decision or blocker | Evidence / owner |
|---|---|---|
| | | |

## Verification Contract

- Command or probe: `bash tasks/main.harmonies-v136-goal-skills/run_a226.sh preflight` on a226; local pytest six version modules plus targeted branch tests.
- Expected signal: source-matched passed receipts, real CUDA, ratio error<1e-4, exact next batch, actor/gate changes.
- Experiment/run pointer, if any: `outputs/v136/skills_a226_20261021_a`

## Next Action

Read stage1 fixed-endpoint audit and execution probes when complete; existing daemon selects stage2 and runs fresh confirmation automatically within original deadline.
