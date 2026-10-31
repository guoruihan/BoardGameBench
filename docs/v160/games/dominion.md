# Dominion — second-edition official First Game, two players

Status: **ready**. `config_id=second_edition_first_game_duo`, rules schema 1.
The fixed official kingdom is Cellar, Market, Merchant, Militia, Mine, Moat,
Remodel, Smithy, Village and Workshop, with Copper, Silver, Gold, Estate, Duchy,
Province and Curse. This implements **one complete official recommended setup**,
not all 26 second-edition kingdom cards. None of the unselected sixteen cards,
first-edition removed cards, expansions or multiplayer variants is advertised.
The user approved a fixed official setup in place of the supplied package's
broader original all-26 implementation target.

## Rules data and provenance

The numerical table is in `families/data/dominion_first_game.json`, transcribed
from the supplied 2026-10-31 rules package and the publisher's second-edition
2021-printing rules. [Rio Grande official rulebook](https://www.riograndegames.com/wp-content/uploads/2016/09/Dominion2nd.pdf)
pp. 2–3 give the setup and counts, pp. 4–9 phases and information permissions,
pp. 12–14 card clarifications, and p. 15 the official First Game selection.
[Official product page](https://www.riograndegames.com/games/dominion/) identifies
the edition. This environment starts with 46 market Coppers and eight Estates
in addition to both players' starting seven Coppers and three Estates.

## Engine and privacy

Action, treasure and purchase phases remain separate. Cellar's chosen discards
finish before drawing; the choice is serialized one card at a time without
intermediate draws. Militia hands control to the defender for an optional Moat
reaction, then any required discards. Mine is optional and gains a treasure to
hand; Remodel must trash when possible and gains to discard. Workshop cannot
supplement its gain limit with coins. Merchant bonuses trigger on the first
Silver only. Actual choices are never heuristic-pruned.

Deck exhaustion shuffles only the discard, only when another card is needed.
Cleanup offers a choice of the visible top card when putting the old hand and
played cards into discard. The order below that top is immaterial in this
kingdom; its next legal access shuffles it. A last Province or third empty pile
ends play only after cleanup and the current turn count increments. Ties use
fewer completed turns, then a shared win.

Observations contain the viewer's hand and draw count; an opponent's hand and
draw count are absent. Only discard tops are exposed, never full discards or
deck composition/order. Public gain/trash/play/discard/reveal events support
legal memory. Cleanup does not expose the discarded private hand. Final card
ownership is revealed only after the natural end for scoring. Snapshots are
privileged state, never observations. The public-only baseline reconstructs
ownership counts from known starting cards and public gain/trash history.

## Verification and baseline

`tests/families/test_dominion.py`: 20 tests, with exact supply, shuffle timing,
Merchant, Cellar, Militia/Moat, Workshop, Mine, Remodel, treasure ordering,
cleanup information, delayed ending, tie-breaks, information equivalence,
illegal-action atomicity and JSON replay. Three random and three heuristic
matches reach natural endings, with conservation checked after every action.
The baseline combines a small action engine with treasure/Province purchasing;
learned opponents should preserve the same phase/visibility boundaries.
