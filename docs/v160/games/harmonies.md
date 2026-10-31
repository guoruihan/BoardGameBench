# Harmonies: existing official two-player setting

`side_a_base32_no_spirits_duo` wraps the existing accelerated
`FastHarmoniesDuo` without changing transitions, information permissions, scores
or ending rules. Both players use board side A, the 32 original animal cards and
no Nature Spirits. The original `harmonies_duo_a_base32_v1` rules identifier is
preserved for compatibility with existing frozen RL and search opponents.

The family adapter changes the API and supplies zero-sum winner rewards to the
generic runner. True terrain/animal scores remain in `score_breakdown`. Its
generic feature policy is a terrain heuristic, **not** the retained best
Harmonies RL/search baseline. Existing retained weights and UI configurations
continue to be the reference opponents; the adapter does not overwrite them.

Tests compare original/adapter private transitions and JSON save/restore, and
check batch/single public feature agreement. Existing duo rule tests remain the
authority for detailed animal, supply, score and equal-turn semantics.
