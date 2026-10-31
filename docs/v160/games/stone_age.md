# Stone Age: complete classic base two-player setting

Status: implemented and validated. Config `classic_base_duo` uses the genuine
36 civilization cards and 28 building tiles from the classic base game. It has
two randomly selected seven-building stacks, all 36 civilization cards, the
official two-player occupancy restrictions and the normal three-phase round.
There are no expansion, anniversary winter, junior-game or artificial fixed-round
rules. Starting player is explicitly supplied to `reset` and subsequently rotates
at the end of each round.

## Primary rules and complete component evidence

- [Hans im Glück English rules](https://www.hans-im-glueck.de/wp-content/uploads/2023/06/stone_age_rules.pdf),
  copyright 2008–2014 / F2Z 2013, pp.2–8. These specify placement, order of
  resolution, dice, tools, feeding, end triggers, final scoring and ties.
- [Z-Man civilization card summary](https://cdn.svc.asmodee.net/production-zman/uploads/2024/10/zm7260_stone_age_summary.pdf),
  both pages: every upper-effect category, exact category counts, one-use tools,
  later resource use, hidden extra card and end-game icon interpretation.
- Physical card-face photographs posted by `garygarison` on BGG in December 2008:
  [cards 1](https://boardgamegeek.com/image/408812/stone-age),
  [cards 2](https://boardgamegeek.com/image/408814/stone-age),
  [cards 3](https://boardgamegeek.com/image/408817/stone-age),
  [cards 4](https://boardgamegeek.com/image/408856/stone-age).
  These four nine-card sheets show **all 36 cards**. Every top/bottom pairing,
  icon multiplicity and physical duplicate was checked. Each JSON card records
  the source image ID and row/column, so the audit can be repeated.
- Physical building photographs:
  [16 tiles](https://boardgamegeek.com/image/408843/stone-age) and
  [12 tiles](https://boardgamegeek.com/image/408847/stone-age), independently
  corroborated by this [complete 28-tile arrangement](https://boardgamegeek.com/image/361106/stone-age).
  Every cost, printed score, variable-count constraint and duplicate was checked.
- The initial numerical transcription was recovered from
  [this 2019 university implementation](https://github.com/L3-Info-Miage-Universite-Cote-D-Azur/projetinfo-2019-stoneage-sah/tree/e7f17b4ae26f4128bc784977fee68cec5b647ffa)
  (`CarteCivilisationManager.java`, `Settings.java`) and then checked against
  the complete physical photos. Its engine code was not imported. An independent
  [36-card list](https://boardgameswithtea.wordpress.com/2025/05/10/stone-age-card-guide/)
  agrees with every pairing. The licensed [Yucata rule appendix](https://www.yucata.de/en/Rules/StoneAge2)
  also agrees with the building-type distribution.

The component file includes 16 green cards (eight symbols twice), 20 sand cards
(five per profession, with totals of seven farmers, eight toolmakers, nine
builders and seven shamans), 17 fixed buildings, eight exact-count/exact-kind
buildings and three 1–7-resource buildings. Artwork, third-party implementation
code and downloaded rulebook PDFs are not redistributed. Source URLs and PDF
hashes are retained in `stone_age.json`.

## Rules and phase decisions

Placement includes every legal worker count. A player cannot return to the same
location during the round, resource locations are exclusive to one player in a
duel, and the third village location becomes unavailable after two have been
occupied. Hunting remains shared. Players cannot pass while legal placement
remains; an exhausted or blocked player is skipped automatically.

The first player resolves **all** their placements in an order they select,
before the other player's resolution begins. Resource dice are rolled publicly
before the separate tool-selection decision. Every available subset of permanent
and one-use tools is legal; full values are applied, and each permanent tile is
used at most once per round. A tool upgrade retains that tile's used flag. When
several minimum-value tiles can be upgraded, the player chooses the tile; this
matters when one has already been used. A newly acquired tool can improve a later
roll that round. Ten hunters roll ten independent dice despite seven dice in
the physical box.

Population, agriculture and permanent tools cap at 10, 10 and `[4,4,4]`.
Capped village spaces remain legal blocking placements under the rulebook's
ordinary occupancy rules and grant no further increase. The licensed digital
game developer explicitly confirms the capped-hut case in
[this 2025 reply](https://steamcommunity.com/app/2177570/discussions/0/598533368165056199/).

Card and building payment enumerate all affordable resource multisets and permit
declining the purchase. Building constraints simultaneously enforce exact total
count and exact number of resource types. New building tops reveal immediately;
civilization cards only slide/refill between rounds. The dice-selection card
switches the actor to the other player for their reward choice, and resumes the
buyer's resolution afterward. These dice cannot be changed with tools.

One-use tool cards retain their bottom scoring section after use. The hidden
extra civilization card executes **no** upper effect, including a one-use or
deferred effect. The two-resource card can be saved and redeemed once. Its
holder has a reaction window between another player's decisions, and may also
redeem it during their own phases. Declining the reaction restores the suspended
actor and phase without using a placement or resolution action.

Feeding first produces agriculture food, then spends food for population,
retaining any surplus. A shortfall can be met by exactly that many resources or
accepted as one 10-point penalty after all food is spent. New population feeds
immediately in that round. A depleted building stack ends the game only after
both players finish resolution and feeding. Inadequate civilization refill ends
the game before starting another round. There is no artificial round limit;
policies that refuse all purchases can in principle continue indefinitely, so
the external runner must classify its runtime cap as truncation.

## Information boundary and state

Workers, resources, food, agriculture, tools, buildings, immediate points, market
and visible deferred cards are public. The actor sees their own acquired cards.
Opposing face-down cards are represented by a count, with public acquisition
history preserved for legitimate memory; the extra private draw's identity is
never placed in that history. Future card/building order, excluded building
identities and RNG stay in private snapshots. All cards reveal at final scoring.

During play, `scores()` reports the official immediate score track. The hidden
card contribution is not leaked via `step.info.scores`. At termination, scores
include culture sets, printed profession multiplicities, leftover resources and
all immediate points. Food and unused one-use tools are not independently worth
points. A tie compares agriculture + permanent tool value + population, then
remains shared. An observer's own projected final score is available privately.

Every mutable game value, pending reaction, dice result and effect continuation
lives in `state`; all randomness uses the saved RNG. Resource supplies are
unlimited as the rules require, with produced/spent ledgers checking conservation.
The 24 public-only bot features use the actor's own knowledge and public board
to value economic development, cards, buildings, food, resource payments and
tool efficiency. They do not inspect future components or opposing hidden cards.

## Validation

From the repository root:

`PYTHONPATH=src .venv-train/bin/python -m pytest tests/families/test_stone_age.py -q`

Tests verify physical component counts and source locations, two-player blocking,
capped placement, full dice timing and ten-die hunting, whole/once-only tools,
upgrade choice and used flags, all fixed upper effects, hidden extra-card privacy,
deferred-resource interrupts, opposing dice-choice reactions, all resource
payments, actual 42-point seven-gold buildings, exact-kind restrictions, feeding
shortfalls/surpluses/newborns, both end triggers, culture/multiplier scoring and
the official tie-break. Six complete games check every-step conservation and
legal actions, JSON replay, feature validity, information boundaries and natural
termination. Illegal actions and corrupt snapshots must roll back atomically.

Validation result: **34 tests passed**. An additional 20 complete games over ten
seeds with swapped seats ended naturally after 13–18 rounds: the initial
heuristic averaged 437.95 versus the random policy's 8.60, winning all 20.
These 7,897 decisions took 45.63 seconds on the development host. This is a
legality and throughput smoke check against a weak random policy, not a trained
bot evaluation or a claim of strong-play performance.
