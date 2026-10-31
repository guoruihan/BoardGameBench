# Century: A New World — original two-player D1 setting

Status: implemented and verified for **Plan B Games 2019 standalone**, using
A1/B1 across the top and C1/D1 below, player board side A. This is one official
fixed base configuration. E1/F1, three/four players, the combined trilogy modes,
and Big Box G1/H1 are outside this environment.

## Original component recovery

The supplied specification was missing printed component data. The engine now
uses 23 authentic locations, 22 eligible two-player point cards, 17 standalone
bonus tiles and 10 exploration tiles. There are six bonus stacks of lengths
4, 3, 2, 2, 2, 2; two shuffled bonus tiles are removed. Ten locations begin covered.
Point cards comprise six Native Help, six Tool, six New Settler and four
Exploration cards; their printed points sum to 102. These checks describe the
catalog, not the attainable game score.

Evidence, inspected 2026-10-31:

- [Publisher's 2019 rules and Overview Sheet](https://cdn.svc.asmodee.net/production-nextmove/uploads/sites/4/2024/06/EN-C3-Rules-2019-03-12_compressed.pdf): setup, worker placement, card effects and scoring.
- [BoardSheep original-box photographs, 2019-08-12](https://usnoopy.blogspot.com/2019/08/century-new-world.html): printed boards, cards and both tile types. In particular, the basket-help 6-point card costs yellow/brown/brown, and trade-help 2-point card costs yellow/red/green.
- [Misut Meeple original-box photographs, 2019-12-13](https://misutmeeple.com/2019/12/resena-century-un-nuevo-mundo/): unobscured complete A1/B1/C1/D1 board, exploration types, bonus examples and point-card examples. This independently confirms all 23 selected locations, their worker costs and initial covers.
- [Public photographed component sheets in Workshop item 3776599714](https://steamcommunity.com/sharedfiles/filedetails/?id=3776599714): all 56 original cards and all tiles; used to recover the complete selected numerical catalog. The public metadata reported `visibility=0`, `banned=0` and update timestamp 1785724830. Data provenance pins the exact payload and image SHA-256 values, sheet URLs, grids and card-cell identifiers.

Card eligibility was checked on the card faces: retain the 16 unmarked cards
without a player-count restriction and six black-star cards without a player-count
restriction. Exclude all 12 white-star cards and the remaining 22 cards marked
3+ or =4. The selected 17 bonus faces have neither I nor III. The engine does not
assume that black stars are excluded from the standalone game.

Only manually transcribed numerical facts are distributed in
`families/data/century_a_new_world.json`; no scanned art, Workshop scripts or
third-party engine code is copied. Full excluded component catalogs are not
claimed. Independent photographs confirm the board and overlapping card/tile
examples; not every selected card has a second independently photographed copy.

## Behavior and information boundary

- Two players start with seven available and five locked workers; first seat has
  three yellow cubes, second seat four. Resource supply is unlimited.
- Work uses printed worker cost on empty locations, or opposing occupancy plus
  one on occupied locations. Matching Native Help discounts stack, minimum one;
  all displaced workers return to their owner's available pool. Own occupied
  locations cannot be reused until vacated.
- Existing Tool cards award resources before the location action. Production,
  every legal trade repetition, all distinct upgrade outcomes (including skipped
  upgrades), and every legal fort card/bonus combination are available.
- A fort can grant its corresponding card, one top bonus tile, or both. Personal
  bonus tiles cannot be replaced and are capped at three. The market shifts left
  following a purchase. Two-stack forts expose both choices.
- An Exploration card causes an explicit choice among all covered locations.
  The chosen tile is gained and that location opens for both players; its printed
  action is not performed for free. Immediate card choices resolve before market
  refill exposes another previously hidden deck top. New Settler gains saturate
  at twelve total workers.
- Storage overflow is resolved at turn end by legal individual cube discards;
  no premature clipping changes trade or production. Rest recalls own workers
  even when some workers remain available.
- Eighth-card acquisition starts the finish-current-round condition. Players
  receive equal turns; no extra full round is added. Scoring includes printed
  card points, independently scored bonus tiles, exploration points and remaining
  non-yellow cubes. Same-icon pairs use floor division; equal total scores favor
  the later seat in the fixed turn order.

Public observations contain both resource inventories, worker allocations,
acquired cards/tiles, four market cards, the face-up point-deck top, all exploration
covers, and each bonus stack's top and count. Future deck order, buried bonus
tiles and removed bonus tiles remain private. Public history retains previously
revealed information. Snapshots contain private state and must not be player
inputs. Static features consume only observations and legal actions.

Rest and optional skipped upgrades can produce non-progressing play under an
arbitrary policy. External step caps remain truncations, never official endings.

## Verification and baseline

Run from the repository root:

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/families/test_century_a_new_world.py
```

The 22 tests cover authentic catalog counts; bumping instead of adding printed
cost; stacked discounts; no overplacement; own-location exclusion; rest;
Tool stacking; delayed capacity discards; complete upgrade outcomes and trade
repetitions; every fort option; market shifting; bonus capacity; exploration
information timing and effects; worker saturation; repeated/same-icon bonus
scoring; equal-turn ending and tie resolution; hidden-order invariance; deep-copy
observations; JSON replay; illegal moves and transactional RNG rollback; finite
24-dimensional features; and complete heuristic/random seeded matches.

The public heuristic values visible-order progress, efficient cube trades,
printed points and ability/bonus synergies. Initial smoke checks on seeds 0–3
completed naturally in 59–65 decisions and approximately 0.12–0.13 seconds per
game on the development process. These are implementation smoke measurements,
not a trained-bot strength claim; shared bot training and confirmation are handled
separately.

A subsequent invariant sweep used fresh development seeds 200–249 (heuristic)
and 200–209 (random): all 50 heuristic and 10 random matches ended naturally,
with 53–75 and 194–624 decisions respectively. Every transition checked component
conservation. These seeds are development checks and must not become confirmation
or formal test seeds.
