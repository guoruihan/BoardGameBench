# Public card-data independent audit

This audit checks the 32 base animal cards in `miles2542/harmonies-bot/docs/cards_database.json`. It does not run the repository engine or establish the correctness of its game rules.

## Findings

- The source has 42 definitions: IDs 1–32 animals, 33–42 Nature’s Spirits. Every animal has one anchor, a nonempty increasing cumulative score track, and a 2–4-cell connected pattern.
- The 32 base cards contain 96 printed animal-cube slots in total. This is the sum over the entire deck, not the number of physical cubes needed simultaneously.
- All 32 animal sprite cards were visually compared with the dataset for terrain, height, anchor, shape and score progression. No discrepancy was identified in this pass. The sprite and JSON are co-located in one repository; this is not an independently sourced second edition.
- The independent normalization expands chain directions into axial coordinates, shifts the animal location to `(0, 0)`, reverses top-to-bottom token lists, and represents buildings by an explicit height-2 rule.
- The code parser and JSON use the same field names; `position` is a chain direction relative to the previous cell, not an absolute cell index. Repeated `position` values are expected.
- `allowCube` marks the one animal destination; it does not forbid existing cubes on non-anchor cells.
- The repository matcher `stack_matches_colors` special-cases `[6, 7]` by checking only for a red top. It would also match a lone red token. This implementation should not be adopted as the environment oracle. The normalized JSON instead requires two layers and bottom color red/brown/gray. A building need not satisfy the landscape's 3-adjacent-colors scoring condition to satisfy an animal pattern.
- Repository LICENSE is Apache-2.0. This audit reports the license of the repository; the card artwork itself is a game publisher asset and no separate artwork permission is asserted.

## Card-by-card structure checks

`Edges` counts immediate pairwise hex adjacency. Score tracks include the zero-cube score.

| Card ID | Cells | Edges | Cubes | Cumulative scores | Image comparison |
|---|---:|---:|---:|---|---|
| 1 | 3 | 2 | 3 | 0, 4, 9, 15 | pass |
| 2 | 3 | 3 | 3 | 0, 4, 10, 16 | pass |
| 3 | 2 | 1 | 4 | 0, 3, 6, 10, 16 | pass |
| 4 | 3 | 2 | 3 | 0, 5, 10, 16 | pass |
| 5 | 2 | 1 | 5 | 0, 2, 4, 6, 10, 15 | pass |
| 6 | 2 | 1 | 4 | 0, 2, 4, 8, 13 | pass |
| 7 | 3 | 3 | 3 | 0, 4, 10, 16 | pass |
| 8 | 3 | 2 | 3 | 0, 5, 10, 16 | pass |
| 9 | 3 | 2 | 3 | 0, 5, 10, 17 | pass |
| 10 | 3 | 2 | 3 | 0, 5, 10, 17 | pass |
| 11 | 2 | 1 | 3 | 0, 4, 9, 15 | pass |
| 12 | 3 | 3 | 2 | 0, 5, 12 | pass |
| 13 | 4 | 5 | 2 | 0, 8, 18 | pass |
| 14 | 3 | 3 | 2 | 0, 5, 11 | pass |
| 15 | 3 | 2 | 3 | 0, 5, 10, 17 | pass |
| 16 | 3 | 3 | 3 | 0, 4, 9, 14 | pass |
| 17 | 2 | 1 | 3 | 0, 4, 8, 13 | pass |
| 18 | 2 | 1 | 4 | 0, 3, 6, 10, 15 | pass |
| 19 | 3 | 3 | 3 | 0, 4, 10, 16 | pass |
| 20 | 3 | 2 | 3 | 0, 5, 11, 18 | pass |
| 21 | 3 | 2 | 3 | 0, 4, 10, 16 | pass |
| 22 | 2 | 1 | 4 | 0, 3, 6, 10, 15 | pass |
| 23 | 3 | 2 | 3 | 0, 4, 9, 16 | pass |
| 24 | 3 | 3 | 2 | 0, 5, 11 | pass |
| 25 | 2 | 1 | 2 | 0, 5, 11 | pass |
| 26 | 2 | 1 | 4 | 0, 2, 5, 9, 14 | pass |
| 27 | 3 | 2 | 2 | 0, 4, 9 | pass |
| 28 | 3 | 2 | 2 | 0, 5, 12 | pass |
| 29 | 3 | 2 | 3 | 0, 5, 10, 17 | pass |
| 30 | 4 | 5 | 2 | 0, 6, 12 | pass |
| 31 | 2 | 1 | 5 | 0, 2, 5, 8, 12, 17 | pass |
| 32 | 3 | 2 | 2 | 0, 5, 11 | pass |
