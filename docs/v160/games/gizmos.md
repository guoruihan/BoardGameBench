# Gizmos: CMON 2018 original two-player configuration

The engine includes all 108 level cards and four distinct starting-card instances.
Each game uses the original 36/36/16 level-deck setup, a 4/3/2 public display,
52 spheres (13 per color), six public spheres, and initial storage/file/research
limits 5/1/3. Two starters and twenty level-III cards are outside the game.

## Authentic component data

The complete catalog was transcribed from the author's photographs of a physical
base set in [Andy Tockman's inventory](https://tck.mn/blog/gizmos-boardgame/):
[tier I](https://tck.mn/blog/gizmos-boardgame/tier1.jpg),
[tier II](https://tck.mn/blog/gizmos-boardgame/tier2.jpg),
[tier III](https://tck.mn/blog/gizmos-boardgame/tier3.jpg).
`families/data/gizmos.json` records all 112 stable instance IDs, exact costs,
colors, points, effects, source locators and source-image SHA256 values. Each
photograph contains exactly 36 cards. Photographs/artwork are not redistributed.
The article later discusses house rules; **none** were adopted.

An independent partial [gizmosbot numerical table](https://github.com/krawthekrow/gizmosbot/blob/802a28cf46912a7235200fa2c23fef5663f82ff7/card_data.txt)
contains 28 tier-I non-converters and all 36 tier-II cards. All 64 normalized
records match the photograph transcription. Its missing tier-I converters and
entire tier III were filled directly from the photographs, not generated from
symmetry. Another implementation's `TestCards` was explicitly rejected as
incomplete and partly invented. No external engine code was copied.

Rules and symbols were checked against the
[official 2018 rulebook](https://www.cmon.com/wp-content/uploads/2023/06/Rulebook_Gizmos-1.pdf)
and [official effects reference](https://www.cmon.com/wp-content/uploads/2023/06/GIZ_ListOfEffects_-_Web_Version.pdf)
on 2026-10-31. Source pages plus pinned numerical data permit independent review
without downloading any private service configuration.

## Choice and event model

A turn starts with one File/Pick/Build/Research action. Triggered Gizmos enter a
pending set; the player chooses activation order and may decline remaining
benefits by ending the turn. Additional actions are tracked individually, so new
triggers can interleave with earlier pending effects. Every Gizmo is usable at
most once per turn. The set eligible for an event is captured before the event;
a new Gizmo cannot trigger on the action that built it. Generic builds count as
all colors, but each matching Gizmo still triggers at most once.

Build actions specify the physical energy vector, used converters and optional
discount amount. A backwards reachability solver enumerates distinct legal
payment outcomes, including converter chains and advanced two-part converters.
Equivalent internal conversion orders are one action because their physical
payment and remaining abilities are identical. Advanced sub-effects may
interleave with other converters during that payment; using any part marks the
whole Gizmo used for subsequent builds that turn. Generated energy is virtual
and never survives payment. Exact payment enumeration can be expensive in
artificial states containing many converters; no candidates are pruned from the
environment. The pure memoization cache does not affect rules, RNG or snapshots.

Research draws the full permitted count, privately offers build/file/pass, then
lets the player choose the order of every remaining card returned to the deck
bottom. This sequential choice avoids factorial action lists and never reveals
unselected cards to the opponent. Builds and files from Research generate normal
triggers, resolved after completing the Research action. Archive identities,
built cards and stored spheres are public.

The physical dispenser is modeled as uniform sampling without replacement from
its internal color multiset. Paid spheres reenter that pool; the public row is
separate. Random draws are not Picks, respect capacity before each draw and do
not activate Pick triggers. The sixteenth built card or fourth level-III card
sets the final-round flag; all remaining chains and seats finish. Original
score and tiebreak order are retained. No runtime step limit changes the rules.

## Verification

Tests cover 112-card and 52-sphere conservation, initial setup, starter File
trigger, full-capacity behavior and RNG, the official converter-chain example,
no repeated doubling, new-card exclusion, generic/dual-color triggers and order,
private Research and arbitrary bottom order, short decks, negators, voluntary
discounts, variable scoring, complete-chain terminal timing, original tiebreaks,
illegal-action atomicity, JSON replay and complete seeded matches. The initial
public-only heuristic completes the sampled games; its score is a starting bot,
not an optimal-play claim.
