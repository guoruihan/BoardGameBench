# Toy Battle: Castle Field official introductory configuration

Status: implemented and validated **for the explicitly official first-game
configuration**, `castle_field_intro_duo`. All eight troop types and their
strengths, optional powers, stacking, connectivity, region scoring and win rules
are present. Printed special bases are ordinary bases in this fixed setting.
This does not claim the standard Castle Field base effects or all eight terrains.

## Why this configuration is official

The publisher's [English rulebook](https://cdn.svc.asmodee.net/production-rprod/storage/games/toy_battle/rules/toy-en01-rules-1744030094a4kQo.pdf),
page 3, says: “For your first games, we suggest playing without special base
effects. Treat all special bases like bases with no effect.” Page 2 recommends
Castle Field for the first game. This fixed configuration retains the complete
printed board and all troops. The user's approved scope is one official
configuration per game; this introductory coverage is deliberately explicit.

The alternative standard configuration is not silently approximated. Its Cap'n
chains can defer several Castle Field returns until after troop effects; whether
an earlier trigger survives covering or returning its source troop lacks a
verified final publisher ruling in the sources we recovered. That ambiguity
has no effect in the selected official setting.

## Authentic component data

- Rulebook pp.2–6: 24 troops per side, three of each type; remove four unseen,
  first player starts with three and second with four; hand cap eight; natural
  immediate HQ/medal wins; if no draw/placement, compare medals and the blocked
  player loses a tie.
- [Official player aid](https://cdn.svc.asmodee.net/production-rprod/storage/games/toy_battle/helpsheet/toy-en01-player-aid-mkt-1750679915Pm3l4.pdf),
  page 1: all eight troop powers and ordering. Kwak is a wildcard, Hook bypasses
  connection only for bases (not HQ), Jumbo removes only an adjacent visible
  enemy, and XB-42 discards uniformly from the opponent's hidden hand.
- [Publisher board photo](https://cdn.svc.asmodee.net/production-rprod/storage/games/toy_battle/elements/toy-graphics-boxbottom-game-1744636033ecoem.png)
  and the rulebook p.2 setup image were inspected together. The map has 17 slots,
  26 paths, eight medal-bearing regions worth 1,1,3,1,1,3,2,2, and a target of
  **seven** for each player. The two central regions have four surrounding bases.
  The blue river between central bases is not a path: C0–C1 and C1–C2 are absent.
  No copyrighted board artwork is distributed; JSON records numerical facts.
- Some secondary online summaries list a target of five for Castle Field. This
  contradicts the clearly visible seven on the publisher's photographed board
  and is not used here.

The JSON defines node IDs, drawing coordinates, headquarters, every edge and
region boundary, medal targets and troop multiplicities. Coordinates are just
an implementation drawing convention; the edges determine adjacency. H0 is the
left HQ and H1 the right HQ in the normalized landscape orientation. L/R are
left/right territories, C0/C1/C2 are central upper/middle/lower bases, and S marks
printed special-base artwork, disabled by the selected official configuration.

## Decisions, information and evaluation

Normal turns choose draw or a full legal troop placement. Cap'n creates an
optional extra-placement phase that can chain more Cap'ns. Other optional powers
have explicit skip/activate or target choices; normal turns have no pass.
Placement is never pruned by the heuristic. Regions are claimed immediately and
medals remain after loss of control. HQ capture stops immediately before any
optional draw or removal. The game has no artificial turn-limit ending.

Both public board stacks and discards are visible. The player sees their own
hand, the opponent's hand size and both reserve sizes. Reserve order, removed
identities and the opponent's hand are hidden. Public draw counts, placements,
removals and claims are retained in history. Snapshots, including RNG, are
private; XB-42 sampling reproduces exactly after JSON restore.

Official scores are medal counts. An HQ winner can have fewer medals: terminal
winner/reason must be used for win-rate training. Feature zero is a public
heuristic combining region completion, supply-line progress, enemy disruption,
hand replenishment and immediate victory; 23 further features support learned
refinements. It is not a solved-game policy.

## Validation

`PYTHONPATH=src .venv-train/bin/python -m pytest tests/families/test_toy_battle.py -q`

Tests cover exact map counts and regions, initial 3/4 hands, all strength and
connection restrictions, Hook's HQ exception, wildcard interactions, Cap'n
chains and skips, Jumbo top-only removal and non-retriggering, partial draws and
rack cap, deterministic random discard, hidden-hand counterfactual invariance,
permanent medals, disconnected region control, both immediate victory types,
no-action tie-break, component-loss restore rejection and ten full matches with
per-decision conservation, illegal-action atomicity and JSON replay.
