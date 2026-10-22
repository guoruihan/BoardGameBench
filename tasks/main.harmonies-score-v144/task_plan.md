# V144 own-score official-duo autonomous exploration

## Stable Anchor
- Question: can true own-score self-play recover layout/animal skill in official duo better than score-margin PPO?
- Success: runnable/recoverable three-round campaign, verified both-seat rewards/updates, fixed-opponent paired development selection and independent 512-environment confirmation. A positive gain is an empirical outcome, not a prerequisite to report honestly.
- Authorization: user approves own-score main line, margin control and autonomous variants for eight hours. Window starts 2026-10-22 14:59 Asia/Shanghai and ends 22:59; setup, old-run handoff, gates, training and final evaluation all included. Never extend silently.
- Resources: a226 GPUs0-3; 57 physical CPU cores including controller, no more than half host physical cores; no other jobs touched.
- Constraints: official duo rules and public information unchanged; sealed tests excluded; all trajectories retained; checkpoint/config/source provenance; commit/push verified updates.
- Non-goals: changing game scores, pretending course and official-duo scores are comparable, optimizing both players' joint score, an unconstrained new platform.

## Current Cycle
- Hypothesis: own-score objective AND own-score checkpoint selection improve layout; full-return credit assignment and opponent diversity may help.
- Uncertainty: short official games/market distribution may dominate the objective effect.
- Decisive experiment: paired four-arm first round from identical V142 transfer; next rounds choose parents and variants using only development evidence. Keep a margin control.
- Expected signal: higher own and animal scores against two fixed opponents, not only a higher moving-self-play score. Track margin, wins, turns, tokens and paired uncertainty.
- Fallback: preserve incumbent weights and all failed/flat variants; report no reliable gain rather than use confirmation for further tuning.

## Phases
1. Inspect active run and preserve handoff: complete.
2. Implement reusable own-score collector, multi-opponent evaluation and bounded adaptive campaign: complete.
3. CPU regression and four real-CUDA reward/update/resume gates: complete (56 CPU checks pass, one old CUDA-only skipped; actual four-device gates all pass).
4. Publish, launch three rounds, verify actual progress and adaptive transitions: complete for launch (four arms21+ batches; controller choices and real round handoff tested; long-run transitions remain scheduled).
5. Independent confirmation and final summary: pending (background campaign).

## Active Parallel Tracks
- Primary writer owns implementation, tests, integration and synthesis; no delegated agents.
- Four independent GPU experiments after gates; CPU allocation remains fixed across phases.

## Verification Contract
- Exact official-engine trace replay, per-seat score/margin telescoping, historical/frozen actions excluded from PPO.
- Legal-mask/finite FP64/behavior-ratio checks, both actor and opponent branch updates; evaluation does not consume training RNG.
- Exact same-config checkpoint next-update recovery; warm-start is explicitly fresh optimizer, not exact resume.
- Source/parent/opponent digests, split disjointness and paired environmental-cluster confidence intervals.
- Hard original deadline, scoped graceful stop, bounded resources, failure artifacts and recoverable state.

## Next Action
Background controller executes remaining rounds and final confirmation before22:59. On next user status request read outputs/v144/own_score_a226_20261022_a/status.json and decisions/; after summary.json exists, analyze paired confirmation without retuning on it.
