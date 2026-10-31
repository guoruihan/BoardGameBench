# Splendor: classic base game, two players

This is classic Splendor (2014), not Splendor Duel. All 90 development cards and
10 nobles are present. Two players use four tokens of each ordinary gem, five
gold, three randomly selected nobles and four visible cards of each level.

The engine includes all legal payment allocations (gold may replace an available
ordinary gem), blind reservations, the three-reserved-card limit, complete gem
return choices after exceeding ten, mandatory player choice among eligible
nobles, at most one noble each turn, equal turns after reaching 15 and the
fewer-purchased-cards tiebreak. A final equal score and equal card count is shared.
There is no arbitrary turn-limit ending or zero-token pass inserted into rules.

## Component provenance and cross-check

`families/data/splendor.json` pins source commits and SHA256 values. All 90 card
records agree exactly after color normalization between:

- [Physical-card photo transcription](https://github.com/anicolao/splendor/blob/5b3efd26f2e460b1bef3839a058270b536a13648/data/verified_card_properties.csv).
- [Independent numerical deck](https://github.com/seal256/splendor/blob/263abc066c563a1c89dba4bdc408446a20ad9d1d/assets/cards.csv).

Only numerical game facts are transcribed; no third-party engine, recognition
code, artwork or photographs are included. Noble requirements come from the same
[ten-tile numerical inventory](https://github.com/seal256/splendor/blob/263abc066c563a1c89dba4bdc408446a20ad9d1d/pysplendor/splendor.py):
nine match the older bouk/splendimax noble table, whose white-four/blue-four tile
is missing; that tenth tile is also present in the boardgamers inventory.

The proposed boardgamers/splendor card transcription was **rejected** because
its normalized costs disagree with physical photographs and seal256, despite its
README claiming agreement. For example, the zero-point white card costs three
blue, not three red. One of its noble triples also conflicts. This provenance
check is tested with a specific photographed-card regression.

Rules were checked against the supplied 2014 rules specification and
[publisher-verified Dized rules](https://rules.dized.com/game/vdDSzuu4RsC8F45PsW0TKw/splendor),
including [FAQ](https://rules.dized.com/game/vdDSzuu4RsC8F45PsW0TKw/faq), on
2026-10-31. Future data updates must not silently replace this fixed catalog.

## Information and verification

Observations expose public resources, acquired cards, nobles and prestige;
reserved identities are visible only to their holder. Publicly reserved cards
remain inferable from action history; a blind deck reservation never records its
identity publicly. Deck order and unused nobles remain private. Feature vectors
use only observation, action and the static printed component catalog, never
unknown replacement cards or blind reservation outcomes.

Tests cover resource/card conservation, voluntary gold payment, gem availability,
return and noble phases, private reservation identities, final-round/tiebreak
rules, seeded complete matches, strict illegal-action atomicity and deterministic
JSON snapshot replay. Evaluation should retain externally truncated runs rather
than invent a score-based winner for games that do not reach the official end.
