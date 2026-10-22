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
- Source2452aea pushed and remote SHA verified; frozen run source28c30c476c4e1b8b8c5afe51dfbecd74862abe4c9820ae714c6d9224ec4ba0a6.
- 08:16:44 Beijing campaign started; fixed12:16:44 deadline. Four realCUDA gates passed, common128 initial comparison saved before training.
- Four jobs each>=23 batches and>215k own actions, changing actor/context, ratio error<4e-14. CPU0–55 pluscontroller56 verified; other GPUs untouched.
- Replayed first full archived batch per arm:512 games/36,746 actions in22.98s oncore56, every reward/terminal matches. All data remain replayable, no old experiments interrupted.
- Verified-launch scope complete; training/selection/confirmation512 proceed automatically in tmux. New results not yet final and no reliable gain claimed.
