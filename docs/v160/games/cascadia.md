# Cascadia: standard two-player game, all A scoring cards

Status: implemented and validated. Config `standard_all_a_duo` follows the
official first-game recommendation: all five A cards, all 85 base habitat tiles,
all five authentic starter boards, and all 100 animal tokens. Each match selects
43 normal tiles and two different starters as in the standard two-player setup.
Other scoring cards, expansions, scenarios, solo and family rules are outside
this fixed setting. No action family is removed from the selected setting.

## Sources and component audit

- [AEG rulebook](https://www.alderac.com/wp-content/uploads/2025/04/Cascadia_BaseGame_Rulebook_Optimized.pdf),
  pp.3–11; the downloaded [publisher syndication PDF](https://images.salsify.com/image/upload/s--gtkZ1nWe--/fug01gvez5kb8ximzdlj.pdf)
  supplies the same 16-page English base rules and component images. Page 4
  recommends the five A cards. Page 11 states optimal, disjoint grouping of elk.
- [Public Cascadia demo component table](https://github.com/cascadiagame/cascadiagame.github.io/blob/d7c93f56995746643eda7a0651b49bad0591b724/js/data.js),
  pinned commit `d7c93f56995746643eda7a0651b49bad0591b724`. Only component facts
  were transcribed; no third-party game-engine code or artwork is included.
  The table contains 15 starter cells and 85 normal tiles, with explicit repeated
  physical copies. IDs 1–100 are stable data IDs, not numbers printed on the game.
- All five starter triples' habitats, icons, relative positions and rotations
  were independently checked against the rulebook p.3 image. Canonical dual
  tiles have habitat 0 on the left half and habitat 1 on the right half. Clockwise
  axial directions are NE, E, SE, SW, W, NW; rotation units are 60 degrees.
- The normal tile table has five keystones per habitat, six tiles per unordered
  habitat pair, 29 tiles supporting each habitat and 32 supporting each animal.
  Every one of the 25 animal-by-habitat counts also matches this independent
  [physical-box census](https://www.reddit.com/r/boardgames/comments/1b8qk3n/some_stats_about_cascadia/).
  This is a complete component-table recovery with aggregate independent checks,
  not a claim that all 85 physical faces were individually re-photographed.
  File hashes and exact source roles are recorded in `cascadia.json`.

The historic demo's scoring code is not used: current card values and designer
clarifications determine the new scorer. In particular, bear A caps at 27 points
for four or more pairs, and elk A uses the highest-scoring legal partition.

## Directly verified designer clarifications

The following original posts were retrieved through BGG's public article API,
`https://api.geekdo.com/api/articles/<article_id>`. The author is user 874081,
whose public user record identifies Randy Flynn (`randyflynn`). These are direct
primary posts, not just secondary FAQ summaries:

- [2021-10-04, article 38581606](https://boardgamegeek.com/thread/2730496/article/38581606#38581606):
  after drawing each replacement batch, return that batch's wiped tokens to the
  bag immediately, before checking another overpopulation. This corrects the
  rulebook's ambiguous instruction to wait until all overpopulations finish.
  It applies to both free wipes and nature-token wipes.
- [2023-06-28, article 42529716](https://boardgamegeek.com/thread/3107728/article/42529716#42529716):
  nature tokens are unlimited; substitute markers may be used if the 25 run out.
- [2023-12-18, article 43475693](https://boardgamegeek.com/thread/2512694/article/43475693#43475693),
  confirming the [2021-08-05 ruling](https://boardgamegeek.com/thread/2512694/article/38224493#38224493):
  a salmon with at least three salmon neighbors is excluded from run scoring.
  Apply the exclusion once using the original board, then score remaining runs.
  Excluded salmon still count for other animals such as foxes.

## Interface and natural game flow

The API phases are optional three-token wipe, selection, and placement. Mandatory
four-token wipes happen automatically with visible draw history. A free triple
wipe is allowed once per turn, including after nature-token wipes if unused.
Every nonempty subset of market animals can be wiped with a nature token. Paid
selection enumerates all 16 tile/token combinations, while the four paired choices
are free. Spending happens before selection; newly earned nature cannot be spent
in that same placement phase.

Tile and animal may be placed in either order. All connected empty hexes and all
distinct terrain rotations are legal, regardless of matching terrain. An animal
may occupy any matching empty tile, including a starter keystone or the newly
placed tile; voluntary discard is always available. A single-habitat tile has
one canonical rotation because its six rotations have identical game state.
The two independently emptied market slots are refilled without sliding other
components. There are exactly 20 turns and 23 total habitat hexes per player.
The last turn ends after both placements resolve when no tile remains to refill.

Both environments and markets, pending components, nature, scores and full
public action/draw history are observable. Future deck order, excluded tile
identities and RNG are private. Animal bag size is public; its exact counts are
inferable from the public initial supply and visible history but are not directly
exposed. All state and chance data survive private JSON snapshot/restore.

Official scoring includes edge-matched largest habitat corridors, two-player
majority bonuses, all five A wildlife objectives, and leftover nature. Connected
elk are partitioned exactly using a memoized disjoint-group optimizer. Ties use
leftover nature, then share victory. The 24 action features add public-only
preparation terms to the exact score deltas; these are bot heuristics, not changes
to scoring or privileged predictions of hidden draws.

## Validation

`PYTHONPATH=src .venv-train/bin/python -m pytest tests/families/test_cascadia.py -q`

Tests check authentic component totals and physical-census marginals, edge
rotation and corridor connectivity, bear exact groups/cap, elk optimization
against an independent exhaustive partition oracle, salmon triangles/violations,
hawk isolation, fox uniqueness, the corrected wipe return timing, one-free-wipe
state through nature spending, split-pair refill, animal-before-tile order,
keystone income, voluntary discard, hidden-data counterfactual invariance,
official tie-breaks, illegal-action rollback and corrupt snapshot rollback.
Six full games check component conservation and deterministic JSON replay at
intermediate phases and confirm the natural 40-turn ending.
