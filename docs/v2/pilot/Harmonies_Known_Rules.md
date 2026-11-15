# Harmonies Known: fixed executable rules

Configuration: `side_a_base32_no_spirits_duo`, two players, side A, 32 base animal cards, no spirits. This document describes the actual benchmark mechanism. The accompanying `catalog.json` contains all printed components; it is data, not a simulator. The SDK legal-action list is authoritative for current choices.

## Board and materials

Each player has their own 23-cell hex board. `CELLS` gives axial coordinates in cell-ID order; `NEIGHBORS` gives adjacent cell IDs. `STACKS[stack_id]` lists tokens from bottom to top, with stack 0 empty. The six token colors are gray, blue, brown, green, yellow and red. The initial shared inventory is respectively 23, 23, 21, 19, 19 and 15. Five public bundles of three tokens and five public animal cards form the shared markets. The bag and deck are shuffled; their hidden order is unavailable. Remaining token counts and remaining card identities are public.

The only allowed stacks are those in `STACKS`. A token can be added exactly when appending its color to the existing stack produces another listed stack, and the cell has no animal. Tokens and animals are never moved or removed.

## Turn and legal operations

Players alternate complete turns, starting with seat 0. A turn has multiple atomic operations, whose order may be interleaved subject to their prerequisites:

- `choose_bundle`: exactly one available three-token bundle per turn. Its tokens become pending and the market slot becomes empty.
- `place_token`: put one pending token on a legal cell, consuming it.
- `take_card`: optional, at most one available animal card per turn, only when fewer than four active cards are held. It starts with zero placed cubes. The market slot becomes empty.
- `place_animal`: use an active card's matching pattern, placing a cube at the specified empty-animal anchor. This may be repeated for any active cards while legal. Completing a card's cube count moves it to completed cards and frees an active-card slot.
- `end_turn`: allowed after taking a bundle and placing all pending tokens. No optional market discard occurs in this two-player configuration.

Markets refill their empty slots only at turn end. Other bundles and cards stay. Up to three bag tokens fill each empty bundle slot; if fewer than three remain, it stays empty and the final round is triggered. Empty card slots refill if cards remain. Pending resources cannot simply be discarded in this implementation.

## Animal patterns

`CARDS` maps IDs to the complete patterns, cube counts and score tracks. Pattern coordinates are relative to the cube anchor; rotation applies `(q,r) -> (-r,q+r)` zero through five times. There is no reflection. Every pattern cell must be on the board and match its required stack exactly. A requirement with terrain `building` matches stack IDs 12, 13 or 14. The anchor must have no animal. Other pattern cells may already hold animals. Existing cubes do not need to be removed if later allowed changes to other pattern cells destroy an old pattern. A cell occupied by an animal cannot receive more terrain tokens. Different animals may share terrain in their patterns, but at most one cube occupies each board cell.

A card contributes `score_by_placed_count[placed_count]`, not the sum of successive entries. Both active and completed cards contribute. `instance_id` equals the unique card ID. The action list retains only one orientation witness for identical card/anchor successors.

## Scoring

Total score is the sum of these independent categories:

- Trees: a single green token scores 1; brown-green scores 3; brown-brown-green scores 7.
- Mountains: gray stacks of height 1, 2 or 3 score 1, 3 or 7, respectively, only if adjacent to another gray stack of any height.
- Fields: each connected component of yellow cells of size at least two scores 5; isolated yellow cells score zero.
- Buildings: red-red, brown-red and gray-red stacks score 5 each if adjacent occupied cells show at least three distinct top-token colors. A single red token scores zero.
- River: consider connected blue cells. The implemented river length is the maximum shortest-path distance between any two connected blue cells, plus one (graph diameter in vertices), maximized over components. It is **not** a longest simple path on branched or cyclic water. Lengths 0 through 6 score 0, 0, 2, 5, 8, 11, 15; each length beyond 6 adds 4. Only this one maximum length scores.
- Animals: sum each owned card's score-track value at its current placed count.

## Ending and information

At turn end, having at most two empty cells on the acting player's board, or failing to refill a three-token bundle, triggers the final round. Both players finish the same number of turns: the game ends after the non-starting player's turn in that round. Highest total score wins; tied scores compare total placed animal cubes, then share victory.

Both boards, held cards and scores are public. Observe the current state before every action; action indices belong to a particular observation version. Use paginated legal actions and history to avoid dumping unnecessary data. The SDK advances the frozen opponent and forced single-choice actions. Only the current actual trajectory is available: no GT branch, rollback, hidden-state access or unlimited reset. The separate task configuration defines game, model and computation quotas. A 3000-accepted-action protection is an abnormal cutoff, not a normal scoring rule.
