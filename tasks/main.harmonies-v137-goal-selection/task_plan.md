# Task Plan: harmonies-v137-goal-selection

## Stable Anchor

- Scientific question: Can learned goal selection convert V136 execution skills into full-game true-score gains on the model's own states?
- Target claim or outcome: Own-state Q labels, frozen-worker manager and joint PPO, with independent confirmation512 and recoverable source.
- Success criterion: Paired fresh-environment total-score gain; two training replicas; no claim from animal completion alone.
- Constraints and budget: a226 GPU0–3 idle00:08CST Oct22; <=64 physical CPUs, planned57. New4h guard includes labeling and experiments; preflight accounted separately, no renewal.
- Non-goals: No rules/reward/hidden-information changes, formal-test tuning, paid LLM, or inference-time search.

## Current Cycle

- Working hypothesis: A learned selector can exploit the improved execution skills more effectively than current heuristic selection.
- Main uncertainty: Candidate gaps may be simulation noise; hindsight expert states differ from student states.
- Next decisive experiment or implementation: 64 own-policy training games, up to4 planning states each; <=8 candidates,4 fit plus4 independent verification futures.
- Expected pass/fail signal: Selected goal gains reproduce under new futures; frozen-worker selector improves new full games.
- Fallback: Retain heuristic and continued PPO; report failed selection rather than call noisy candidate maxima an upper bound.

## Success Criteria

- [x] Verified train-only data and independent futures/panels; environment-level aggregation.
- [x] Actual CUDA, frozen-worker, PPO ratios, joint gradient and exact resume checks pass.
- [ ] Frozen, committed, detached bounded experiment with real updates and saved checkpoints.

## Parallel Tracks

| track | owner | mode | worktree / branch | dependency | deliverable | status |
|---|---|---|---|---|---|---|
| primary | primary | integrate | current | none | final synthesis | active |

## Phases

### Phase 1: Discover and specify

- [x] Confirm live state and relevant owners.
- [x] Record the smallest implementation and verification contract.
- **Status:** complete
- **Evidence:** V136 completed; a226 GPU0–3 idle,4–7 busy; local GPU unavailable. No delegated writers.

### Phase 2: Implement or run the decisive test

- [x] Make the smallest coherent implementation.
- [x] Record decision-relevant evidence and material checkpoints.
- **Status:** in_progress
- **Evidence:** source-matched regression77 passed/20 explicit CUDA skips, additional shared-stage test1 passed. Actual CUDA4 fixtures and mature control/selector/joint ratio, gradient and next-batch resume checks passed on the final source.

### Phase 3: Interpret and hand off

- [ ] Compare the evidence against the stable anchor and expected signal.
- [ ] Close remaining phases as complete, superseded_by_evidence, or abandoned_with_evidence.
- [ ] Commit/push coherent work and archive or hand off the task.
- **Status:** pending
- **Evidence:** none

## Decisions And Blockers

| Item | Decision or blocker | Evidence / owner |
|---|---|---|
| Retain mechanisms | All four arms continue through matched8M actions; no tiny-gap elimination. | User-approved V137 question |
| Resource availability | GPU0–3 idle at probe; use57 physical CPUs maximum, do not touch4–7. | a22600:08CST |
| Shared stage | joint and refresh share identical stage1, branch at new-label initialization; 6 then8 jobs. | Identical preflight updates and seed/config |
| Pilot | Sample-fit gain does not survive independent futures; many candidate ties. Do not claim gain. | Old draft preflight_attempt_01/pilot |

## Verification Contract

- Command or probe: pytest V127/V129/V133–V137; actual CUDA run_a226.sh preflight.
- Expected signal: masked/paired data, no verification targets in loss, frozen worker exact, joint gradients, exact next batch resume.
- Experiment/run pointer, if any: outputs/v137/selection_a226_20261022_b; prior draft_a retained with cost provenance.

## Next Action

Commit/freeze/launch after source-matched gates; observe initial labels and real training checkpoints.
