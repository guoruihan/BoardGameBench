# Gizmos: CMON 2018 original two-player configuration

Status: **ready_with_rulings**, `rules_version=2`,
`config_id=cmon_2018_duo`. The forced-pass ruling and remaining double-block
edge are specified below; this is not an unconditional complete-rules claim.

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

## Rules 2: forced pass at a genuinely blocked main action

Ruling ID: `forced_pass_when_all_four_actions_unavailable`.
Evidence tier: **licensed digital adaptation**, not a publisher/designer FAQ.
The printed rulebooks require a main action but do not specify this rare edge.
The [licensed BGA game page](https://boardgamearena.com/gamepanel?game=gizmos)
credits developer Fletcheese and CMON. Fletcheese's
[2023-05-20 original fix](https://github.com/Fletcheese/gizmos/commit/bc8e07a68762388499ceaf3044824f75ec7254ec)
explicitly describes passing when no actions are legal. Its frontend offers
Pass when Pick/File/Research are unavailable, leaving Build affordability to the
player; its server advances without a Pick event. Verified 2026-10-31 13:49 UTC.

Our narrower implementation offers `forced_pass` **only when all four complete
action sets, including every converter/discount payment, are empty**. It ends
the current turn normally, without taking/discarding energy, triggering a
Gizmo, drawing randomness, changing scores or conceding. Available Builds or
Research prevent passing. Pending effects retain their separate `end_turn`.
No broader optional pass is adopted. This interpretation affects rare ordinary
games, so it remains an explicit ruling requiring future source review.

Observed development regression: `cem_r1/g011_gate.json`, row 18, seed
`11011009`, first player 0, generation-11 versus generation-6 linear policies.
Rules 1 stops at decision 106, scores 24:16: player 0 has five black spheres,
archive `g3-31`, the no-Research `g3-01`, no usable black converter and no
affordable public/archive card. The frozen linear coefficients in
`tests/families/test_gizmos.py` reproduce this state independently of run files.
Rules 2 passes twice and the same policies reach a natural build-threshold end
at decision 116, scores 24:18. This is a bug regression, not a strength estimate.

Remaining edge ID: `all_players_blocked_no_stalemate_ruling`. If both players
are permanently unable to act, alternating forced passes leave the position
unchanged. No authoritative draw/stalemate rule was found. The external runner
therefore reports `truncated` at its declared limit, with no winner or reward;
it never invents a draw. Such games must remain in the attempted-game denominator.

## Checkpoint and snapshot migration

Rules 1 artifacts remain intact. The action features are still 24 dimensions;
all previous feature values, candidate order and transitions are unchanged
until the formerly empty main-action set. `forced_pass` uses the existing
end-turn feature representation and is necessarily the sole candidate. Linear
weights, neural encoder/actor/critic tensors and prior weights are structurally
compatible. This does not establish unchanged win rates under the new rules.

Default policy/checkpoint/snapshot loaders correctly reject a rules-1 identity.
For an explicitly audited warm start, copy the artifact into a new run, preserve
all tensors/weights, change the copied `rules_version` to `2`, and record the
source file SHA256, source rule version and this migration reason in metadata.
Recompute the new SHA256 and update its copied bot spec. Do not relabel original
evaluation logs or copy confirmation conclusions. Start PPO with `--warm-start`
and a new output directory; do not resume the old optimizer/rollout directory.
Frozen-source bundles likewise need a new runtime identity and new evaluation.

An interrupted rules-1 game can be continued only by an explicit copy migration
of its snapshot identity; its state and RNG bytes need no conversion. Keep the
original snapshot alongside the migrated copy. The regression checks restored
state/RNG equality before continuing; loaders do not silently migrate snapshots.

Migration smoke on the actual rules-1 `ppo_r1/candidate_0479.pt` confirmed exact
tensor/prior equality in the migrated copy, rejection of the original identity,
selection of the sole forced-pass action, and a natural 101-decision match on
seed 91 against the heuristic. The source checkpoint SHA remained unchanged.
This is a compatibility check, not a fresh performance evaluation.

## Verification

Tests cover 112-card and 52-sphere conservation, initial setup, starter File
trigger, full-capacity behavior and RNG, the official converter-chain example,
no repeated doubling, new-card exclusion, generic/dual-color triggers and order,
private Research and arbitrary bottom order, short decks, negators, voluntary
discounts, variable scoring, complete-chain terminal timing, original tiebreaks,
illegal-action atomicity, JSON replay and complete seeded matches. The initial
public-only heuristic completes the sampled games; its score is a starting bot,
not an optimal-play claim.

Rules-2 tests additionally cover the deterministic development deadlock,
forced-pass RNG/resource/event invariance, continued snapshot replay, available
Build/Research excluding pass, illegal-pass atomicity, last-seat final-round
completion, unchanged 24-dimensional feature representation and simultaneous
blocking without a fabricated winner. Gizmos and runner regression suites:
32 tests passed on 2026-10-31.
