# Progress

Append material checkpoints only: phase changes, decision-relevant probes, experiment launches,
failures and replans, verification, commits, pushes, and handoffs. Do not log every command.

## 2026-10-22 — task initialized

- State: implementation and launch validation
- Implemented own-policy planning-state capture, paired independent-future labels, environment-held-out manager initialization, frozen-worker and joint PPO, explicit refresh continuation, fixed-endpoint confirmation and deadline guard.
- Initial regression:77 passed,20 CUDA skipped; real a226 CUDA4 fixture tests and mature checks passed. Old draft_a preflight153.89 seconds,161.47 GPU seconds,1231.25 CPU-core seconds; all costs retained.
- Pilot did not show independently verified gain; no algorithm success claimed.
- Optimized bit-identical joint/refresh first stage into shared checkpoint. Old draft retained immutable; formal run_b links old preflight receipt and imports old local-test costs. Source-matched local/remote checks restarted.
- Next: commit coherent source after gates, detached launch, verify actual progress/checkpoints and document exact deadline.

## 2026-10-22 00:36 CST — final-source gates passed

- Final-source local regression77 passed/20 explicit CUDA skips; shared-stage construction test1 passed. Actual CUDA4 tests passed in28.94seconds.
- Mature checks: frozen-selector worker change exactly0; control/joint worker changes nonzero; manager gradients nonzero for learned selector. Initial ratio errors<6e-14; exact resumed trajectories and weights.
- Independent-future pilot remains negative; launch is a bounded hypothesis test, not a claimed improved policy.

## 2026-10-22 00:36:47 CST — detached campaign launched

- Commit98ec9a5; frozen source f39c9b9d07a7b648bfa8da6f5b6834b9224f88b9f0c9c6be1c53f7c449d7d946.
- a226 tmux boardbench_v137_goal_selection; deadline04:36:47 CST, same deadline on resume. GPU0–3, <=57 physical CPUs.
- Root outputs/v137/selection_a226_20261022_b. Two active CUDA label workers onGPU0/1; first check1280/15384 and1280/15496 branches, fresh heartbeat and no log errors. GPU4–7 unrelated workloads untouched.
- Prior and current validation costs preserved:306.675 GPU seconds and3240.147 CPU-core seconds. No strategy-gain claim yet.
- Next: observe real PPO progress/checkpoints, then background stage2 refresh and independent confirmation512.

## 2026-10-22 00:40 CST — initial labels complete

- Two64-environment corpora contain255/253 planning states and15384/15496 candidate futures; elapsed169.69/170.74seconds, natural terminal and independent fit/verification samples preserved.
- Independent-future gain: replica_a+0.0137 interval[−0.2631,+0.2904]; replica_b+0.0550 interval[−0.1158,+0.2259]. No robust goal-reselection gain. Current-policy candidate ties remain common.
- Stage1 jobs generated automatically; all prescribed methods retained to test actual on-policy learning rather than select from noisy local maxima.

## 2026-10-22 00:42 CST — live PPO handoff verified

- All6 trials actively collecting/updating, each>170k additional actions, heartbeats<3seconds and saved latest checkpoints. Control actor gradients nonzero, selector/joint manager gradients nonzero; joint is in planned first1M frozen-worker warmup.
- Both saved selector latest workers are tensor-exact to their initial worker. First training batch environment-seed lists match across control/selector/joint within each replica.
- Initial development512 scores112.63–112.87 are pre-PPO baselines, not improvement claims. Final comparisons pending.
- Background state is deliberately not archived/marked research-complete. Next: stage1 endpoint, new labels,8stage2 trials, frozen confirmation512, cost/retained-policy handoff. No user action needed.
