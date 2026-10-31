# El Dorado: official first route, two players

Status: implemented; original five-tile first route and all 18 market types.
The module is `boardbench.families.games.el_dorado`, `GAME_CLASS = ElDorado`;
configuration ID `2017_first_route_duo_no_caves`. This is a race, so official
scores are `null`; winners and two-explorer arrival counts are separate fields.

## Coverage and evidence

The selected rules are Ravensburger's 2017 English base game, product 26754,
without caves or expansions. Each player controls **two** explorers and must
bring both to the city. All 54 market cards and both eight-card starting decks
are present. The first player uses spaces 1 and 3; the second uses 2 and 4.

- [2017 publisher rulebook, unchanged mirror](https://cdn.1j1ju.com/medias/fc/6a/fa-the-quest-for-el-dorado-rulebook.pdf):
  pp. 2–5 turn, payments, blockades, ending and two-player rules; p. 8 all 18
  market types and action-card timing. Downloaded and parsed.
- [2017 publisher setup sheet, image mirror](https://img.ludex.dev/_astro/setup-route.npWe0n4z_8RkIR.webp):
  original `26 754 5` sheet, first route B/C/N/I/K, starting slots, ending gate,
  six initial market piles, and shuffle six blockades/use four. Visually checked.
- [Original component-face catalogue](https://ludex.dev/quest-for-el-dorado/cards/):
  complete individual B/C/N/I/K faces and the six original blockades. Individual
  image URLs, byte sizes and SHA256 values are pinned in
  `families/data/el_dorado_sources.json`. These are publisher component images
  hosted by a third party, not new card designs.
- [Later publisher rulebook mirror](https://bghub.org/r/thequestforeldorado.pdf):
  p. 8 resolves a residual tie with no blockades by earliest arrival. Only this
  clarification is used; later art/expansion components are not imported.

No original illustrations/PDFs or third-party implementation code are shipped.
The Python engine is newly written; the data records numerical gameplay facts.
A public implementation (`donle/el-dorado` commit
`095002dfc2b142479a6dbef74b7be3b11f09916e`, no declared license) was used only as a
lead, then rejected as an authoritative data source: the original faces exposed
several transcription errors. In particular I's centre is green 2, its camp is
one cell to the right, two adjacent mountain cells must remain mountains, its
lower-left water cell costs 2, and K's water cell costs 3 rather than 4. K also
contains several green-2 cells that that implementation records incorrectly.

The independently transcribed face rows are 4/5/6/7/6/5/4 hexes. The official
route rotates B/C/N/I/K by 2/5/2/2/0 clockwise sixth-turns relative to the pinned
face images. The resulting 185 terrain hexes plus three blue-1 finish entries
were rendered and compared with the original setup sheet. Four seams contain
seven adjacent crossing edges each. Six blockade **ranks** are 1–6; their costs
are green 1, yellow 1, rubble 1, blue 1, green 2, rubble 2. Rank and cost are
separate: the printed rank, not the symbol count, is the final blockade tie break.

## Interface and choices

All state and chance are under the shared `DuoGame` snapshot/RNG contract.
`legal_actions()` enumerates legal card commitments and decisions in six phases:
play, committed movement, purchasing, post-draw optional trashing, Transmitter
selection, and end-turn discarding/keeping. Card power cannot be combined to
enter one landscape hex or split between two explorers. Mountains and every
occupied explorer hex block movement. Removing a blockade leaves the explorer
where it was; remaining matching card power can subsequently pay for entry.

Physical cards have deterministic instance IDs such as `scout:2`. Their copy
numbers carry no deck-order information. Observations show one's own hand,
public discarded/played cards, public market/board, and public event history.
They omit both deck orders, the opponent's hand, unused hidden blockade tiles
and RNG. Reactions are not invented; all purple-card choices stay with their
owner. Legal movement can revisit earlier cells; a runtime decision cap is a
truncation, never a natural game ending.

The baseline uses public terrain distances, card/deck composition inferred from
public acquisitions, and individual action features. Its 24 numbers are policy
features, never a fabricated official score. It is intended as a trainable
starting opponent and has no claim of optimality.

## Explicit boundary interpretations

These small rulings are recorded so a future publisher clarification can version
rather than silently change the environment:

- Numbered starting spaces have no machete symbols. Re-entering one uses green
  movement with cost zero, counting printed requirement symbols rather than the
  unrelated player-placement number. This follows the generic terrain rule;
  the rulebook does not give a separate re-entry example.
- Transmitter selects one card from any pile. It does not move an outside pile
  into the market; the normal-purchase rule explicitly moves the remaining pile.
  Exhausting a market pile still creates a vacancy. It does not consume the
  normal purchase. Empty supply makes the effect empty, not a deadlock.
- The 2017 purchase text says a newly acquired card may not be used during that
  same turn. Such cards go to the discard pile and may be shuffled/drawn normally,
  but remain unplayable until the next own turn. The same conservative reading
  is applied to Transmitter acquisitions. This rare same-turn redraw interaction
  has not been independently confirmed by a designer FAQ.
- Purple actions may be used during buying, as p. 8 states; Native can therefore
  move an explorer then. This does not resume ordinary landscape-card movement.
- A blockade may be removed while standing beside its seam without also entering
  the next hex, matching the separate-payment rule and Native clarification.

## Verification

Run from the worktree/project root:

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest tests/families/test_el_dorado.py -q
```

The rule suite checks real component counts/corrections, board topology, original
setup, hidden observations, non-combinable movement, two-pawn occupancy, item
consumption exceptions, camp vs rubble, separate blockade payment, Native,
shuffle timing, Scientist/Travel Log, buying phases, market vacancy, Transmitter,
coin half units, kept cards, equal-turn ending, blockade rank ties, strict invalid
action/RNG identity, JSON replay, finite features and complete seeded games.

Smoke observation: twelve heuristic-vs-heuristic seeds (0–11) all reached a
natural ending in 487–654 decisions, 2.5–3.8 seconds per match on the development
CPU. This is a functionality/speed check, not an opponent-strength evaluation.
