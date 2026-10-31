# Century: Spice Road — original standalone two-player game

The fixed configuration uses the 2017 base game's 43 market merchants and 36
point cards, with two distinct starting cards per seat. It excludes Golem,
trilogy combinations, promos and Golden Deals. First player starts with three
yellow cubes; second player with four. The official unlimited supply is retained.

Implemented choices include repeated complete trades, all optional upgrade
sequences, independently chosen payment colors on every skipped merchant,
collecting attached cubes only after payment, resting even with no played cards,
exact-color point-card purchases, and player-chosen overflow discards. Last gold
moves the silver position only for later purchases. The sixth claimed order
triggers equal turns; a point tie goes to the later seat. Returned cards never
become a random draw pile. Market deck exhaustion leaves shorter rows.

## Data verification

[Publisher rules](https://cdn.svc.asmodee.net/production-nextmove/uploads/sites/4/2024/06/EN-Century-Spice-Road-Rules_2024_compressed.pdf)
and [official component list](https://www.nextmove-games.com/en/century/trilogy/)
anchor the setting. Static numerical records are pinned in
`families/data/century_spice_road.json`:

- [JavaScript numerical inventory](https://github.com/yohanlaunay/spice-trader/blob/05c5f138e4fa49a88832b53851ea0277b50fee22/src/games/spices/data.js).
- [Independent Java CSV inventories](https://github.com/ShaPhi7/CenturySpiceRoad/tree/ec638ce92e6ced847de105aff6de1e8b0c1e0b5e/src/main/resources).
- [Firsthand base-box component analysis](https://boardsandbees.wordpress.com/2019/01/15/0823/).

All 36 orders agree exactly between implementations. Their full point histogram
and 468-point total agree with the physical analysis. All 34 trade effects agree,
as do seven production cards and the three-step upgrade. The Java CSV's remaining
production entry is four brown; it conflicts with the JavaScript table's one
brown and the firsthand inventory's statement that all production cards produce
spice value three or four (yellow=1 through brown=4). We use **one brown**, not
four. Counts are therefore 34 trade, eight production, one upgrade. Only numerical
game facts were transcribed; no third-party engine code or art was copied.
Checked on 2026-10-31.

## Information, boundaries and tests

Own hand and claimed point cards are private observations. Opponent resources,
played cards, coins and card counts are public. Market acquisition, card play and
order claims remain in public history, preserving information that a player can
remember. Hidden deck order never enters policy features.

As in the supplied original-rule specification, a trade play requires at least
one affordable repetition; optional zero-step upgrades remain legal. The printed
rules do not explicitly clarify a zero-trade no-op. No no-op trade is silently
introduced as a verified rule. An empty rest already provides the printed
no-effect action if desired. This rule-text boundary remains documented.

Tests exercise component/coin conservation, upgrades, repeated trades, payment
ordering, cube attachments and overflow, gold/silver timing, exact colors,
terminal order and scoring, hidden information, invalid-action rollback, JSON
replay and complete seeded games. Search/training cutoffs are external
truncations, not additional game-ending rules.
