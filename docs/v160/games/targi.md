# Targi: original two-player base configuration

`game_id=targi`, `config_id=original_2012_duo`, `GAME_CLASS=Targi`.

Implemented: the 16 fixed borders, all 45 original tribe cards, all 19 goods
cards, three figures and up to two intersection markers per player, all four
raids, full three-by-four tableau scoring, and both natural ending conditions.
No expansion, solo rules, action tokens, alternative borders, or reduced-start
variant is included. First player is supplied by `reset(seed, first_player)`.

## Evidence and component recovery

The package's missing catalog was recovered from the original component faces
published by the licensed online implementation [Yucata](https://www.yucata.de/en/Rules/Targi).
The 45 consecutive tribe images, IDs 0–44, include every cost, point value, and
ability. Goods IDs 1–19 have 12 distinct faces; duplicate image IDs are omitted
by the website. Multiplicities were checked against the physical-box catalog
and photos in [this firsthand component review](https://homeofmark.com/targi-card-game-review/).
Each tribe type has nine cards, with four 1-point, three 2-point and two 3-point
cards; printed tribe points sum to 80.

Implementation was written independently from rules and component facts. No
Yucata source code, copyrighted card prose, art, or downloaded PDF is included.
`families/data/targi_sources.json` records per-image SHA256 hashes and URLs for
independent verification. Research downloads stay in ignored `tmp/`.

The [official 2018 English manual](https://www.thamesandkosmos.co.uk/wp-content/uploads/2020/05/691479_Targi-Manual.pdf),
identifier `691479-02-070318`, supplies the general rules and an independently
reproduced 33-point scoring fixture. Earlier original card wording governs the
hand-play ability: one **arbitrary ordinary good** extra, once per round.
This is explicit in the original component and the author's FAQ. It differs
from wording in some later English physical cards that specifies a required
good type; those later-printing semantics are not claimed by this configuration.

The author's 2012 FAQ is reproduced, attributed, in the 2012-06-02 comment on
[Hall9000](https://www.hall9000.de/html/spiel/targi). BGG's public file metadata
independently identifies STOCKBROT as the uploader of
[FAQ English](https://boardgamegeek.com/filepage/92247/faq-english), but its later
2016 PDF could not be downloaded. We do not claim to have verified that revision.

## Decisions and information

Actions expose placement, arbitrary order of retrieving figures/markers,
optional skipping, each payment alternative and discounted resource choice,
reserve/display/discard, tableau row, repeated trader exchanges, one
silversmith exchange, Fata Morgana, bonus goods, raids and excess-goods discard.
Rearrangement enumerates every legal final layout changing at most two cards,
including swaps and temporarily leaving a gap which the second move fills.
Legal actions are not pruned by the heuristic.

Reservations and shuffled deck order are hidden. Replacement center cards stay
face down until the next round. Publicly revealed cards, resources, displays,
figures, scores, discards, counts and actual public action history are exposed.
The observation copies nested data. Internal snapshots retain all state and RNG
and support JSON round trips. All random operations use the environment RNG.

The optional scoring feature vector has exactly 24 finite public-only entries.
Feature zero is a practical initial heuristic; the remaining entries represent
resources, tableau development, immediate card value, intersection potential,
row compatibility, trades and engine abilities. These are bot inputs, not
changes to the game's score.

## Boundary rulings

These are explicit and testable, rather than silently represented as verified
publisher clarifications:

- Fata Morgana may move a remaining marker only onto an unoccupied **face-up**
  center card. The official manual's delayed reveal and the physical review
  support this reading; a direct designer ruling was not located.
- If the tribe draw pile runs out, its discard pile is shuffled. The official
  manual explicitly states this for goods, but is silent about the much rarer
  tribe case. This matches the component review; no new cards are introduced.
- When a player cannot pay either raid alternative, the opponent gains the
  specified VP cost. The author explicitly confirms this for the fourth raid;
  applying it to earlier raids is a defensive extension (normal initial four
  VP suffice for the first three raids absent intentionally altered fixtures).
- The hand surcharge can play a newly reserved card during the same action
  turn; the ordinary Noble action requires a reservation from an earlier round.
  The manual explicitly marks the hand ability as an exception to Noble use.
- A new per-round resource bonus sees goods already obtained earlier that
  round, consistent with the manual's same-round activation wording. Border
  resources never trigger these bonuses. An immunity waiver does not trigger
  the raid-payment point.

## Checks and reproduction

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest tests/families/test_targi.py -q
```

All **80 tests pass**. The test module covers the complete component census, every card's acquisition
and immediate resolution, the manual's exact 33-point tableau, payment/discount
boundaries, all special abilities, reserves and hidden observations, mixed
raid payments, limits, asymmetric figure counts, marker geometry, conservation,
JSON replay, invalid-action RNG identity, and six complete heuristic/random
games with finite features at every decision.

Initial heuristic self-play on seeds 0, 1 and 2 naturally reached the fourth
raid in 279–285 decisions, scoring 20:38, 33:27 and 34:25. These are smoke checks,
not a trained-bot strength claim. Runtime was 1.7–1.8 seconds per game including
full conservation validation on every step, on the development host.

An additional 24-game smoke run (seeds 100–123, alternating first seat; eight
heuristic and sixteen random games) passed conservation at every decision and
finished naturally in 222–283 decisions. Both ending conditions were exercised;
total time was 32.04 seconds.
