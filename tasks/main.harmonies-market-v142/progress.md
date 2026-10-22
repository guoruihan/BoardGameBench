# Progress
## 2026-10-22
- Read planning skill/server rules/project workflow and inspected current implementations.
- Announced explicit13-turn random-market protocol, two initialization arms x two replicas, new4-hour wall cap and57 CPU/4GPU bounds.
- No changes to old environments or artifacts; old V141 campaign already finished.
- Implemented isolated13-turn curriculum, native/public encoding, own-only complete-game pool, reused PPO update, fresh warm-start optimizer, own-score best selection, portable policy and offline transition replay.
- First integrated regression39 passed/1 CUDA skipped; extra final-lifecycle test added. Initial turn-cap witness test needed a stacking-oriented bundle selector; no rule change was needed.
- 08:11 Beijing fresh a226 check: GPUs0-3 each14MiB/0%, GPUs4-7 busy, load13.22; no other tasks modified.
- Bounded controller, actual-CUDA gates, common initial128 evaluation and final paired512 evaluation implemented; source freeze/publication and actual launch pending.
- Final local regression40 passed/1 explicit CUDA skipped in31.58s; shell syntax and diff whitespace checks pass. Ready for frozen-source actual-device gates.
