# Calico: standard two-player setup, teal/purple boards, three size cats

Status: implemented and validated. Config `standard_teal_purple_sizecats_duo`
uses the authentic teal and purple boards, Millie/Tibbit/Coconut, random two-pattern
assignments per cat, all 108 cloth tiles, and the standard choice of three ordered
goals from four revealed goals. It covers all six goal types and the full normal
turn, reward, scoring and tie rules. Other boards and the seven other cats are
outside this fixed setting; family, lower-variance and digital open-hand variants
are not enabled.

## Sources and component audit

- [AEG English rulebook, licensed Tabletopia mirror](https://c.tabletopia.com/games/calico/rules/calico-rules-s/en),
  2020, pp.2–9; [publisher original URL](https://www.alderac.com/wp-content/uploads/2020/09/Calico-Rules.pdf).
  Page 5 explicitly allows choosing which three cats to use under standard setup.
  Page 4 provides the teal and purple boards' printed border images; pp.6–9
  specify all rewards and goal multiplicities, plus tie-breaks.
- [Public coordinate table used for discovery](https://github.com/ZackWebb/calico_engine/blob/edcf62e9343101614931b3bae2dfd477a643f6a6/source/board_configurations.py).
  Both 22-cell borders were independently checked against the primary rulebook
  images. The teal board's axial `(-2,-2)` cell is **green flowers**, whereas the
  discovered table says leaves; this implementation follows the primary image.
  No third-party game-engine code or copyrighted board artwork is included.
- The preserved layout has 22 fillable cells, three goal slots and 22 adjoining
  preprinted border cells per board. Full and partial printed cells both count.
  Corner artwork that touches no playable hex is not a game cell. The JSON lists
  every coordinate, color and pattern and pins the source version.
- Rulebook p.7 permits substitute tokens if cat tokens run out. Rewards are
  therefore recorded as an uncapped earned-token ledger, not stopped by the
  supplied cardboard count.

The source cube coordinates were re-expressed as axial `(q,r)` using the first
and third cube components. Neighbors are `(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)`.
This is a coordinate convention, not a change in board geometry. Both boards use
the same 25 interior/22 border geometry and distinct printed colors/patterns.

## Interface and game state

Setup goal choices are serialized in first-player order before hands or market
are dealt. Both four-goal offers are public, matching their faceup reveal. Each
choice enumerates all 24 ordered selections; no goal is preselected by the bot.
Normal turns have a placement phase followed by a three-slot market draft, even
after the 22nd placement. Both hands stay at two after every draft. There are 90
API decisions: two setup choices plus 44 placement/draft pairs. No artificial
turn cap defines the ending.

Observers see their own two tiles, public board cells and goals, the market,
rewards, deck size, opponent hand size and the history of public selections and
placements. Initial opposing hand tiles and future deck order stay private.
Publicly drafted tiles remain inferable through the action history. RNG appears
only in private snapshots.

Cats require a connected group of **one** supported pattern. Color and pattern
components use separate histories. A rewarded component cannot earn again by
expanding; merging two already-rewarded components retains both tokens. The
anchor records make these histories survive serialization. Six colors give one
rainbow button. Goals test the six-neighbor multiset independently for color
and pattern and award the higher combined value, never add both values.
Final ties compare number of cats, then buttons including rainbow, then share
victory.

The 24 public action features include immediate rewards, uncompleted component
progress and a uniform-future approximation to goal completion probabilities.
Those probabilities are an action heuristic, not a model of the actual hidden
deck or a change to official scoring. Market selection evaluates the acquired
tile's best next placement. Search/learned weights can improve this baseline.

## Validation

`PYTHONPATH=src .venv-train/bin/python -m pytest tests/families/test_calico.py -q`

Golden cases cover all six goals, 7-versus-11 dual scoring, border rewards,
nonmixing of two cat patterns, retention of separately earned cats/buttons after
merging, rainbow scoring, all final tie-break levels and final-turn drafting.
Tests also check the fixed board geometry, 24 setup choices, all 44 first-turn
placements, exact 108-tile conservation, hidden-hand/deck counterfactual invariance,
public market history, illegal-action rollback, corrupt snapshot rollback,
JSON replay and eight complete matches. The heuristic probability calculation
is checked against exhaustive enumeration of small remaining goal neighborhoods.
