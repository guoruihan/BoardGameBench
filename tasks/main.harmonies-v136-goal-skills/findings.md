# V136 findings — 2026-10-21

- V135 finished all training/evaluations but final summary publication failed on numpy.bool_. Narrow pure aggregation repair produced confirmation.json without recomputation; original failure retained in guard_result.json and summary_recovery.json.
- Current user authorizes implementation and new experiments. Earlier V135 deadline is not reused.
- a226 GPU0 occupied by unrelated gmx; GPUs1–3 available. GPUs4–7 unrelated busy workloads. Use only1–3, stage1 43 physical CPU cores, cap64.
- Train-only verified expert corpus: 240 episodes; goal extraction produces1,687 feasible segments and15,257 states. Fit210 environments, skill-probe30 (not held out from old/new PPO).
- Existing candidate primary exactly covers84.35% of segment-start goals; card+anchor87.67%. This is a selected-segment diagnostic, not global action coverage or proof of strategy ceiling.
- Code path: v136.data -> anchored adapter/gate warm-start -> inherited natural-terminal PPO. Three arms control/plain/skills. Conditional stage2 manager uses current-worker public-counterfactual Q labels, not teacher continuation values.
- Local core regression:64 passed,16 explicitly CUDA-skipped; one earlier small-manifest health subset failure fixed before this result. Extra manager-branch/label/probe-failure tests run separately. Real CUDA preflight in named tmux underway.
- Local package __version__ was stale at0.4.14 while pyproject0.4.20; both now0.4.21.
