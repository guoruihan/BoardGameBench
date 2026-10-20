# Progress

## 2026-10-21

- User authorized autonomous implementation/exploration without per-round approval.
- Read planning skill and server rules; no applicable global memory hit.
- Confirmed parent/protocol and probed a226 GPU0–3 available.
- Implementing V135; no training launched yet.
- Preserve unrelated untracked docs/BoardBench_V1_2_Design.md.
- Implemented grounded advice compiler, fixed/learned trust, persistent target arm,
  two-stage automatic promotion and confirmation, frozen-source/resume safeguards.
- Local regression:54 passed,12 CUDA-only skipped in53.64s (58.62s wall).
  Initial fixture assumption corrected: distinct geometry can require identical
  currently available choices; compare all alternative plans, not just first two.
- a226 named tmux `bb-v135-preflight` running real CUDA and mature policy checks.
- Preflight finished successfully:4 CUDA fixture tests, four real GPU tensor
  executions, four mature-model next-update resumes. Maximum probability error
  3.91e-14; learned trust gates both update.
- Training64 initial probe:control111.3125;fixed/gated111.890625;sticky111.375.
  Diagnostic only, not independent evidence of improvement. Fixed/gated animal
  mean26.078125 vs control25.765625. Initial advice removal returns111.3125.
- Throughput actions/s:control1919.6,fixed1796.2,gated1787.4,sticky2914.0.
- Code version0.4.20, commit de3787e. Frozen source
  efda95242f427062eca39c334e83cd632b2bcc13974dc1e770511cb0e46ce806.
- Launched named tmux `bb-v135-advice`, root outputs/v135/advice_a226_20261021_a.
  Launch1792529677.7846768, deadline1792540477.7846773. Guard PID50960,
  campaign PID51408.57 physical CPU cores; authorized GPUs0–3 only.
- Preflight accounting1310.44 CPU-core-seconds+171.47 GPU-seconds; local checks
  conservatively450 CPU-core-seconds. These are added to stage1 accounting.
- Live verification:all8 learners completed6–12 PPO batches,586264 total newly
  collected/updated actions, heartbeats<5s. Learned gates update; ratio errors
  <=5.33e-14. This confirms actual learning, not merely a detached launch.
- Initial development512 (before any V135 training):control109.361328125,
  fixed/gated109.73828125,sticky109.837890625. These are new panel scores and
  must NOT be compared directly with V134's old confirmation111.08.
- Next:autonomous stage1→promotion→stage2→fresh512 confirmation. No user approval
  needed between stages; all source/config/checkpoints retained.
