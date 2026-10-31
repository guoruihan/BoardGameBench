# That's Pretty Clever — original sheet, two players

Status: **ready**. `config_id=original_sheet_duo_six_rounds`, rules schema 1.
The fixed setting is Wolfgang Warsch's first game, original Schmidt sheet, two
players and six rounds. No Challenge sheet, sequel or solo variant is enabled.
All twelve active turns and their corresponding passive opportunities, including
both final extra-die windows, are played before scoring.

## Evidence and constants

The supplied `10_clever_sheet.json` (checked 2026-10-31) is retained as structured
numerical data. It includes every yellow/blue cell, printed cross, progressive
score, green threshold, orange multiplier and all track/round bonuses.

- [Schmidt English original rules](https://www.schmidtspiele.de/files/Retail/72dpi_PNG/88198_That%27s_pretty_clever_GB.pdf): original sheet p. 1, active/passive flow pp. 2–3, colors pp. 4–5, actions/rewards pp. 6–7 and ties p. 8.
- [Schmidt German rules](https://www.schmidtspiele.de/files/Produkte/4/49340%20-%20Ganz%20sch%C3%B6n%20clever/49340_Ganz_schoen_Clever_DE.pdf): original diagonal bonus and sheet corroboration.
- [Original product](https://www.schmidtspiele.de/details/produkt/ganz-schon-clever.html).

Internal yellow positions use zero-based row-major indices; blue targets are
printed sums 2–12. This is an API coordinate convention only.

## Mechanics and information

A blue mark always uses the current blue-plus-white sum, regardless of where the
other die sits. Moving dice to the platter compares the selected physical die's
face, not the blue sum. A passive player gets kept-die fallback only when every
platter option is illegal. Platter dice are not consumed by passive selection.

Immediate reward chains finish before dice are rerolled or the phase advances.
All simultaneously available colored reward choices remain explicit; reward
nodes fire once. Filled target regions lose unusable immediate rewards. Round
four's reward is applied to both sheets before either player's active turn.
Rerolls only affect remaining dice and consume a token without adding a normal
selection opportunity. Extra dice may reuse a regularly selected die but each
physical die can be used at most once per extra-action window.

All sheet/dice information is public. Future RNG is private and snapshot-only.
No unsupported artificial turn cap or score normalization is part of the game.

Validation: `tests/families/test_clever.py` contains 25 tests, including golden
sheet/bonus/phase cases, RNG-preserving illegal action rejection, JSON snapshot
replay and six random plus six heuristic complete games with both first seats.
The baseline balances immediate scoring, printed rewards and future lines;
it is a baseline to tune, not a claim of optimal play.
