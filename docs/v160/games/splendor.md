# Splendor: classic base game, two players

This is classic Splendor (2014), not Splendor Duel. All 90 development cards and
10 nobles are present. Two players use four tokens of each ordinary gem, five
gold, three randomly selected nobles and four visible cards of each level.

The engine includes all legal payment allocations (gold may replace an available
ordinary gem), blind reservations, the three-reserved-card limit, complete gem
return choices after exceeding ten, mandatory player choice among eligible
nobles, at most one noble each turn, equal turns after reaching 15 and the
fewer-purchased-cards tiebreak. A final equal score and equal card count is shared.
Rules version 2 adds a forced pass only when none of the four ordinary action
types is legal, following the attributed publisher ruling described below.
There is no arbitrary turn-limit ending or voluntary zero-token take action.

## Forced-pass ruling and remaining boundary

The [2014 publisher rulebook](https://cdn.svc.asmodee.net/production-unboxnowcom/uploads/2022/02/Splendor-EN.pdf)
and the [2024 publisher rulebook](https://cdn.svc.asmodee.net/production-spacecowboys/uploads/2025/10/SCSPL01EN_SPLENDOR_RULES_LIGHT.pdf)
do not explicitly cover the case where ordinary gems are exhausted, a player
already holds three reserved cards, and no visible or reserved card is affordable.
This is reachable with legal play and occurred in retained training/evaluation.

We adopt a **publisher ruling available through attributed secondary evidence**:
pass only when there is no other legal action, showing the reserved hand as proof.
The evidence checked on 2026-10-31 is:

- A [2022-07-29 answer by Arcanist Lupus](https://boardgames.stackexchange.com/questions/57737/what-to-do-when-you-dont-have-any-move-available/57738)
  quotes this ruling, attributes it to an Asmodee representative, and links the
  [specific original BGG article](https://boardgamegeek.com/thread/1192043/article/16190546).
  The original article could not be fetched; this is not claimed as direct
  verification of the original publisher post.
- An independently dated [community player-aid discussion](https://boardgamegeek.com/filepage/112709/ultimate-splendor-player-aid)
  states the same restricted-pass and reveal-hand rule on 2015-03-01; its author
  confirms adding the clarification in version 1.2 on 2016-01-21. This corroborates
  the rule but is not itself publisher authority.
- A [secondary FAQ index](https://www.rulespal.com/splendor/faq) attributes the
  original answer to `S_Rapp_Asmodee`, dated 2014-07-01. The account/date attribution
  remains indirect. Sebastian Rapp's professional connection to Asmodee is
  independently supported by his [2016 interview](https://www.brettspiegel.de/kopie-von-ralph-querfurth).

The [Dized FAQ](https://rules.dized.com/game/vdDSzuu4RsC8F45PsW0TKw/faq) and
[BGA help revision 29147](https://en.doc.boardgamearena.com/index.php?title=Gamehelpsplendor&oldid=29147)
did not add an explicit forced-pass or mutual-block termination rule.

Passing reveals all reserved cards then held, leaves resources, score and RNG
unchanged, and completes the ordinary turn including the existing noble check.
Revealed identities remain public while held and in historical events after a
purchase. A later blind reservation remains private until separately revealed.
Pass cannot be selected when taking gems, reserving or buying is possible, or
during a return/noble choice.

**Remaining boundary:** both players can be blocked at once (a legal 18-decision
fixture demonstrates this). No verified ruling for ending such a game was found.
Both players therefore continue forced passes; the existing external evaluation
cap records `truncated`, with no fabricated draw, winner or training reward.
The environment explicitly advertises `ready_with_rulings`, and
these incomplete games remain in scheduled-game denominators.

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
unrevealed reserved identities are visible only to their holder. Publicly reserved cards
remain inferable from action history; a blind deck reservation never records its
identity publicly unless a subsequent forced pass reveals the hand. The
`revealed_reserved` observation lists currently held cards disclosed by passing.
Deck order and unused nobles remain private. Feature vectors
use only observation, action and the static printed component catalog, never
unknown replacement cards or blind reservation outcomes.

Tests cover resource/card conservation, voluntary gold payment, gem availability,
return and noble phases, private reservation identities, final-round/tiebreak
rules, seeded complete matches, strict illegal-action atomicity and deterministic
JSON snapshot replay. Evaluation should retain externally truncated runs rather
than invent a score-based winner for games that do not reach the official end.

## Version and policy compatibility

Rules version 1 results and checkpoints remain unchanged historical artifacts.
Version 2 changes both the legal-action boundary and public information after
passing; snapshots/specs/checkpoints declaring version 1 are rejected by strict
version checks rather than silently reused. Any copied policy must receive an
explicit, audited version migration and fresh evaluation; old results are not
version 2 results.

The 24 numerical features and their ordering are unchanged for every existing
action, and `pass` produces a finite vector using the unchanged feature function.
Since pass is the only candidate at that boundary, old linear/neural policy
parameters require no new output head. The feature policy currently does not
exploit the newly disclosed opponent hand. Local migration checks preserve every
model tensor and prior weight bitwise, verify the source checkpoint checksum is
unchanged, and load only a separately written version 2 copy.

The retained audit directory is
`outputs/v160/bots/splendor/no_legal_action_audit_20261031/` (outside Git): original
failed-game snapshots/actions, source receipts, version 2 replay regressions,
feature comparisons, and the separate dev2 truncation audit. Regression
continuations explicitly identify when both seats use the heuristic; they are
engine correctness checks, not claimed counterfactual scores for the old bots.
