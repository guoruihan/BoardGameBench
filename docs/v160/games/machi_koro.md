# Machi Koro — classic 2014 English base, two players

Status: **ready_with_rulings**. `config_id=classic_2014_base_duo`, rules schema 1.
All 15 establishment types, the complete 84-card market, four landmarks, one
starting Wheat Field and Bakery and three coins per player. This does not cover
Harbor, later editions, or the optional random market. The first fourth landmark
ends the game immediately. Official scores are null: this is a race, not a
landmark-count victory-point game.

## Evidence and one explicit interpretation

Constants are transcribed from the supplied `09_machi_koro_cards.json` dated
2026-10-31, cross-checked against the classic publisher rulebook and its licensed
online rules implementation:

- [Pandasaurus original product and rulebook link](https://pandasaurusgames.com/products/machi-koro).
- [Classic English rulebook](https://www.dropbox.com/scl/fi/v2hw5o0zu9wefj8bjfhrb/Machi-Koro-Rulebook.pdf?rlkey=h10b4dxeeeo43wugb7cfvt493&dl=0): components/setup pp. 3–4; activation pp. 5–6; construction and ending p. 7.
- [Yucata licensed rules](https://www.yucata.de/en/Rules/MachiKoro): establishment
  counts, effect order, and landmark explanations. Its continuation after one
  player wins and optional variable supply are **not** used here.

Ruling `new_amusement_park_no_retroactive_trigger`: **provisional event-based
interpretation**, not a claimed publisher FAQ ruling. An amusement park must
exist when the accepted dice result is resolved to trigger an extra turn. Buying
it after rolling doubles does not retroactively trigger the roll. Yucata states
landmarks operate from the next turn, supporting this interpretation, but that
summary is not definitive evidence for every printing of the 2014 English rules.
This can affect an ordinary game when a player buys the park immediately after
rolling doubles; its regression test is named explicitly. A future official
clarification requires a rules-version bump if it changes behavior.

## Implemented choices and validation

Train Station retains one-die and two-dice choices. Radio Tower rerolls all dice
once before any income. Red payments precede bank income; unaffordable debt is
forgiven. Multiplicities and mall icon bonuses apply per copy. Purple effects
are ordered by the active player; Business Center can decline or exchange any
owned non-purple establishment, including a starting card. New acquisitions do
not retrigger income. Consecutive double-triggered extra turns are uncapped.

Everything except future random outcomes is public. Snapshots include the RNG
and are privileged; observations never include it. No artificial turn ending.

`tests/families/test_machi_koro.py` checks component conservation, income ordering,
factory/mall multipliers, reroll final-result semantics, purple order/exchanges,
extra turns, immediate victory, invalid-action RNG atomicity, JSON replay and
complete random and heuristic games on three seeds each. Baseline features are
public coin-flow/landmark signals, not a claim of strong play.
