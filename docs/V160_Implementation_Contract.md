# V1.6.0: fixed original two-player games

This phase targets the 17 games in the supplied original-rule package. Each game
uses one documented official base-game setting. Harmonies is retained. A game
with unverified component data remains blocked; invented cards/maps are forbidden.
The objective is verified playable environments and reproducible opponent bots,
not a claim that every game has been solved or fully covers its base box.

## Implementation contract

New modules live in `boardbench.families.games.<slug>` and expose `GAME_CLASS`.
Inherit `boardbench.families.base.DuoGame`. Own all mutable game data in `state`
and use only `self.rng` for game randomness. The shared base owns atomic action
rejection, snapshots, terminal rewards, and metadata. Override:

- `_reset(first_player)`: initialize the documented setting; base already sets
  current_player, first_player, terminated, decision_index, winners, end_reason.
- `_legal_actions()`: enumerate complete legal choices for the current phase.
  Split genuinely sequential choices into phases. Do not prune game legality to
  make the bot easier. Inactive players have no actions.
- `_step(action)`: perform a choice and chance transitions, select the actual
  next actor (which may be the same seat or a reacting opponent), call `finish`
  only at a natural terminal. Record public history where players can infer it.
- `_observe(player)`: return only information this player is entitled to know;
  hide deck order, opposing private hands, inaccessible bonus values and RNG.
- `scores()`: true official current/terminal scores for both seats; race games
  can use null scores and explicit winners. No invented score as official score.
- `validate()`: component conservation and phase/state invariants.
- static `features(observation, action)`: exactly 24 finite numbers from public
  observation/action only. Feature zero is the baseline heuristic score; others
  are interpretable signals for learned refinements. No hidden-state access.

`game_id`, `rules_version`, `config_id`, `default_weights` are class fields;
default weights select feature zero. `observe` supplies metadata; `legal_actions`
is a separate call. `step` returns observation, two zero-sum terminal rewards,
terminated/truncated and outcome. A runtime step cap is a truncation, never an
invented original-game ending.

Each implementation owns its data under `families/data/<slug>*`, tests under
`tests/families/test_<slug>.py`, and evidence/coverage in
`docs/v160/games/<slug>.md`. Provenance records must pin sources and distinguish
original facts from implementation choices. Fixed original setup is not full
base-box coverage. Do not import unrelated third-party code without checking its
license; original numerical rules data still needs cross-checking.

## Gates and opponents

Golden rule cases, deterministic save/restore replay, invalid-action atomicity,
component invariants, legal complete games, and hidden-information checks precede
training. Preserve random, heuristic and genuinely measured improved opponents.
Train/development/confirmation seeds are disjoint; use swapped-seat evaluations.
Never overwrite a stronger retained opponent with an unverified latest model.

Research resources use at most half each host's CPU capacity (a-hosts also at
most 128 CPUs). GPUs are scheduled from current free memory and sustained load;
all configured reachable hosts may be used. Preserve unrelated jobs. The first
window has an eight-hour checkpoint and a twelve-hour hard stop recorded in the
private runtime manifest. Completed updates are committed and pushed; weights,
raw runs and private service configuration remain outside Git.
