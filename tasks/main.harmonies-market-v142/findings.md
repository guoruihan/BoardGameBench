# Findings
- Current V140/V141 score arms optimize own score but select best by margin; new trainer must select score_mean explicitly.
- Solo caps supply at13 turns (120 tokens, nine per round); duo market has5 slots and replenishes vacancies, not equivalent supply. V142 will cap own turns at13 explicitly and label itself a curriculum variant.
- V141 optimized DuoPolicy consumes4865 features and244 actions; preserve model keys, use two own market views and a zero absent-opponent block under a NEW layout identifier.
- V141 collector/PPO, V138 persistent public planner, V129 native encoder and V128 bitset geometry are reusable. Fresh Adam for both initialization arms; exact resume only within V142.
- Initial git worktree has unrelated untracked docs/BoardBench_V1_2_Design.md; preserve. No global memory matches used.
