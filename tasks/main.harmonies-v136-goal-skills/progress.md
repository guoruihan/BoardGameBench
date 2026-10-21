# V136 progress

2026-10-21: User said “来做吧”. Scope: implement goal-execution learning, retain sticky PPO and ordinary imitation controls; condition goal-selection branch on measured execution.

- Read planning skill and server rules; no relevant global memory hits; preserved unrelated docs/BoardBench_V1_2_Design.md.
- Live a226 probe found GPU0 gmx; selected GPUs1–3, not0 or4–7.
- Implemented data extraction, warm-start/PPO integration, paired fresh-future execution probe, manager bootstrap, bounded two-stage orchestration and V135 summary recovery.
- Prepared outputs/v136/skills_a226_20261021_a and immutable data/jobs/splits. Preparation47.82s on4 local reserved cores, charged191.27CPU-core-s.
- Core local tests64 passed/16 CUDA-skipped; source-matched receipt local_checks.json. Actual CUDA preflight started detached in tmux boardbench_v136_preflight.
- Actual CUDA completed:4 CUDA fixture tests passed; mature three-arm next-batch resume passed, maxratio error3.91e-14. Preflight160.13s, accounted1281.18CPU-core-s and166.57GPU-s. Extra local tests3 passed.
- Initial skill execution probe:24 states×2 futures, target completion12.5%→31.25%, target-branch total+3.4375; same skills with no initial target16.67%. Training-state diagnostic only, not full-game policy gain.
- Scoped version committed as aa9c213. Launch requested in named tmux boardbench_v136_skills, fixed4h guard.
- Live handoff verified: all6 trials8–9 complete batches,75.8k–85.8k additional real actions each, saved latest.pt, real actor/gate gradients, ratio errors<=6.04e-14. Initial environment seeds matched across arms; frozen source matches receipt/workspace.
- Initial development512 after warm-start: control112.6992, plain111.3516/111.0254, skills111.1836/111.1836. Skill execution improvement has NOT yet produced an overall policy gain; do not present the small-state completion signal as a score win.
- Run started15:50:18CST, deadline19:50:18CST2026-10-21; tmux boardbench_v136_skills. Stage1 curves/results and live_handoff.json are durable monitoring paths.
- Next action: let bounded stage1/selection/stage2/confirmation run; inspect endpoint total scores and execution gates on the next user check. No deadline extension or unrelated preemption.
