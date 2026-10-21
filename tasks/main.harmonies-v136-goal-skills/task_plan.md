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

- [ ] Immutable train-only data and fresh curve/audit/confirmation panels.
- [ ] Probability, masks, no-leakage, real gradient, exact resume, actual CUDA checks pass.
- [ ] Detached bounded training running with saved checkpoint and real completed PPO updates.

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

- [ ] Make the smallest coherent change or launch.
- [ ] Record only decision-relevant evidence and material checkpoints.
- **Status:** pending
- **Evidence:** none

### Phase 3: Interpret and hand off

- [ ] Compare the evidence against the stable anchor and expected signal.
- [ ] Close remaining phases as complete, superseded_by_evidence, or abandoned_with_evidence.
- [ ] Commit/push coherent work and archive or hand off the task.
- **Status:** pending
- **Evidence:** none

## Decisions And Blockers

| Item | Decision or blocker | Evidence / owner |
|---|---|---|
| | | |

## Verification Contract

- Command or probe: `[pending]`
- Expected signal: `[pending]`
- Experiment/run pointer, if any: `[none]`

## Next Action

Implement train-only goal segment extraction and anchored warm-start, retaining mature sticky checkpoint and PPO objective.
