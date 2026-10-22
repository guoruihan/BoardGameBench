# Task Plan: V143 official-duo transfer and self-play

## Stable Anchor
- Question: can V142 own-layout skills improve through genuine official-duo self-play?
- Outcome: behavior-preserving transfer, initial self-battle evidence, verified four-job background PPO launch with reproducible checkpoints and traces.
- Constraints: official duo rules/information unchanged; a226 GPU0-3 only,57 physical CPUs<=half; NEW four-hour wall cap includes gates/baseline/train/evaluation/save. No renewal of expired V142 budget.
- Non-goals: modifying rules, claiming course116.70 is official-duo performance, training against old weak opponents, sealed-test tuning.

## Current Cycle
- Hypothesis: preserve own-board policy initially, learn a zero-initialized public-opponent residual during competitive self-play; compare current/current with current/history mixture.
- Uncertainty: value of adapting opponent-aware behavior without losing existing layout skills.
- Decisive experiment: two replicas per method; both current-policy seats contribute separate on-policy trajectories; historical seats excluded from PPO.
- Success signal: fresh paired-seat confirmation512 improves margin/win rate against the frozen transferred starting policy, with own score/animal diagnostics.
- Fallback: fail closed if transfer parity, masks/rewards, real CUDA updates, or exact resume fail.

## Success Criteria
- [x] Transfer parity, official-game traces, two-seat reward accounting and history exclusion verified.
- [x] Four real-CUDA update/exact-resume gates pass; initial self-battle saved before formal training.
- [x] Four bounded jobs genuinely update; version/source/config/checkpoints/data preserved.
- [x] Implementation6765a30 committed/pushed, remote SHA verified; live launch documented.

## Parallel Tracks
- Primary only; no delegated agents. Four GPU experiment processes are independent replicas, not distributed training.

## Phases
### Phase 1: Transfer and self-play implementation
- **Status:** complete
- **Evidence:** separate V143 policy/collector semantics/checkpoints/runner; all official engines and accelerated pool reused unchanged.
### Phase 2: Tests and actual-device gates
- **Status:** complete
- **Evidence:**46 local passes/1 old CUDA skip, four actual A100 transfer/update/exact-resume gates.512 formal matches replay with62,202 actions and1,024 learner trajectories.
### Phase 3: Initial battle, bounded launch and publication
- **Status:** complete
- **Evidence:** initial256 official matches79.4883/animals8.7461; four jobs>=21 batches, both seats updating,57-core audit;6765a30 published. Launch scope complete; background training and final evaluation ongoing until18:45:23 Beijing.

## Verification Contract
- Shared weights remain fixed for each complete collection. Separate learner trajectories per seat, no turn-switch terminals, no historical actions in loss.
- True score-margin differences telescope; development selection uses paired margin, independent confirmation and fresh seed namespaces.
- Initial residual exactly zero; V142 own-view logits preserved before learning, opponent public inputs available only to new branch.
- Exact within-version recovery includes snapshots, optimizer, all RNGs, counters and unchanged deadline. Every actual action archived.

## Next Action
none - ready to archive
