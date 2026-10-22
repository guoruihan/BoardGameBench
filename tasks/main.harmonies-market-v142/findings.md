# Findings
- Current V140/V141 score arms optimize own score but select best by margin; new trainer must select score_mean explicitly.
- Solo caps supply at13 turns (120 tokens, nine per round); duo market has5 slots and replenishes vacancies, not equivalent supply. V142 will cap own turns at13 explicitly and label itself a curriculum variant.
- V141 optimized DuoPolicy consumes4865 features and244 actions; preserve model keys, use two own market views and a zero absent-opponent block under a NEW layout identifier.
- V141 collector/PPO, V138 persistent public planner, V129 native encoder and V128 bitset geometry are reusable. Fresh Adam for both initialization arms; exact resume only within V142.
- Initial git worktree has unrelated untracked docs/BoardBench_V1_2_Design.md; preserve. No global memory matches used.
- Same new development128: solo initialization112.2031/animal28.8047/12.4766turns versus duo initialization106.9063/animal26.3281/11.9063turns. Paired total difference-5.2969 SEM1.0797. New input/protocol prevents interpreting this as proof of old-model forgetting.
- Actual own-only collection128 takes about2.2–2.4s (environment1.5–1.6s, model0.7s); full training batch about3.1–3.6s before periodic development overhead. No equal-protocol solo/duo speed claim.
- Atbatch20, paired own-init development deltas small/uncertain: solo_a-.2891 SEM.9683, solo_b+.0938 SEM1.0685, duo_a+1.2031 SEM.8638, duo_b+1.8516 SEM1.0617. Final independent512 pending.
