# Patchwork: classic corrected two-player configuration

Status: implemented and validated. `GAME_CLASS` is
`boardbench.families.games.patchwork.Patchwork`; config is
`classic_corrected_duo`. This covers the complete original 33-patch base game,
9×9 boards, five leather patches and the seven-point 7×7 bonus. It excludes
Express, Doodle, promos and Automa. First player is supplied by the benchmark;
all remaining setup randomness is a seeded shuffle of the complete tile ring.

## Sources and audit

- [Official publisher page and leather correction FAQ](https://www.lookout-spiele.de/en/games/patchwork.html):
  confirms 33 ordinary patches, five leather patches, and that current prints
  move the first leather patch later on the track.
- [Official English rulebook](https://www.lookout-spiele.de/upload/en_patchwork.html_Rules_Patchwork_EN.pdf),
  two pages: legal turns, placement, income, terminal scoring and arrival tie-break.
- [Official correction image](https://www.lookout-spiele.de/upload/en_patchwork.html_UpdateSticker_Patchwork_De-EN.pdf):
  corrected leather track is used. Numbering starts at position zero, income is
  at 5,11,17,23,29,35,41,47,53, and leather is at 26,32,38,44,50. The obsolete
  position 20 is replaced by 38. The two printed sides differ in appearance.
- [Pinned component facts](https://github.com/CACppuccino/PatchworkGame/blob/3876fee77064ae9f8444385e4bf39bed07fff4de/src/comp1140/ass2/PatchworkGame.java)
  and [corresponding images](https://github.com/CACppuccino/PatchworkGame/tree/3876fee77064ae9f8444385e4bf39bed07fff4de/src/comp1140/ass2/gui/assets):
  33 ordinary shapes, printed costs, time and income were checked against the
  individual tile images. They total 166 unit squares. Labels A–Z,a–g are
  implementation identifiers, not extra game content. Source component tables
  are Apache-2.0; the license accompanies `patchwork.json`. No third-party game
  engine code is used. In particular the old source's leather positions and
  omission of the 7×7 bonus are **not** inherited.
- The base rulebook does not describe the rare already-full-board leather
  case. We remove the unplaceable leather from play, matching the digital app
  and the English translator's clarification quoted in
  [this discussion](https://www.reddit.com/r/boardgames/comments/lftlg8).
  This secondary quotation is a narrower evidence limit than the primary
  sources above; the case is explicit in the engine and regression test.

## State, actions and information

The entire game is public after setup. The observation includes both boards,
buttons, incomes, time positions and top marker, the full ring and neutral gap,
placed patches, unclaimed leather, pending events, bonus owner and arrival order.
There is no future hidden deck. RNG state and the setup seed are not player inputs.
Snapshots retain all state and the RNG for JSON replay.

`buy` contains patch ID, canonical orientation and an (x,y) anchor. All unique
rotations/reflections and every nonoverlapping in-bounds placement are legal;
there is no adjacency restriction or heuristic pruning. Symmetric transforms
with exactly the same occupied cells are represented once. `advance` is always
available during a normal turn. Leather placement is an immediate phase for the
moving player, including when crossing multiple events. Events resolve in time
order, with newly purchased income active before any income event. End-of-track
clamping, mover-on-top ties and first-arrival final ties follow the rulebook.

Boards and placement legality use integer bitmasks; precomputed legal shapes
are immutable. The 24 features are public-only and include patch net economic
value, remaining income events, time efficiency, filled area, contact, isolated
gaps, progress toward 7×7, and available leather. Feature zero is a packing and
income heuristic, not a claimed optimal policy. Generic search or learned
weights can refine it without changing legal actions.

## Validation

`PYTHONPATH=src .venv-train/bin/python -m pytest tests/families/test_patchwork.py -q`

The suite includes authentic component/count assertions; all 144 placements of
the empty-board 1×2; neutral-gap removal; same-time repeated turns; advance
payment; no borrowing future income; purchase income before collection; multiple
immediate leather choices; all nine 7×7 windows; bonus versus spendable buttons;
terminal clamping and tie-breaks; full-board leather; invalid-action and injected
exception rollback; invalid snapshot rollback; JSON replay; and six complete
random/heuristic matches with conservation validation after every decision.
