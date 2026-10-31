# Original two-player family: rolling development

List installed environments (availability is not a complete-rules certificate):

```bash
.venv-train/bin/python -c 'from boardbench.families.registry import available_games; print(available_games())'
.venv-train/bin/python -m pytest tests/families -q
.venv-train/bin/python -m boardbench.families list
.venv-train/bin/python -m boardbench.families match --game jaipur \
  --bot0 heuristic --bot1 random --seed 80 --trajectory \
  --output outputs/v160/jaipur/example_match.json
```

Each game's scope, genuine data sources and unresolved boundaries are documented
under `docs/v160/games/`. Missing games are explicit unavailable entries rather
than substitutes. Harmonies reuses the previous accelerated official duo engine.
Five Tribes is currently a research implementation with explicit unresolved
rule edges; installed code is not an assertion that those edges are settled.
The prior Harmonies RL opponent can be specified with `kind: harmonies_rl`,
`path`, and optional `sha256` matching `configs/harmonies_duo_play.json`.

Run or resume a small CPU policy-search experiment:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv-train/bin/python -m boardbench.families.train \
  --game jaipur --output outputs/v160/jaipur/example --workers 4 \
  --population 12 --generations 64 --hours 3 --source-commit "$(git rev-parse HEAD)"
```

For remote production jobs, use an immutable source snapshot, a named tmux
session, fixed CPU affinity, a fresh capacity check and an absolute deadline.
Combined allocations may use at most half each host (a-host cap 128). Generated
runs, private cluster inventory, model weights and trajectories stay outside Git.

The learner optimizes public action-feature weights through cross-entropy
population search; it is not a neural PPO implementation. Every candidate is
paired on the same environments with seats swapped. Fresh promotion seeds are
separate from training; final confirmation never chooses the checkpoint.
Small-sample promotion is exploratory and does not itself establish superiority.

Artifacts: `config.json`, atomic `state.json`/`best.json`, each generation's full
results, promoted `bot_gNNN.json`, and independent `confirmation_summary.json`.
`state.json` records optimizer RNG and source commit; resumption requires that
source snapshot. `best.json` is an observation-only frozen linear policy usable
with `FeaturePolicy`. Confirmation trajectories contain public observations,
actions and final outcomes for later imitation/offline learning.

Runner caps produce `truncated`, not invented game endings. Rule gaps produce
`unresolved_rule`; execution failures produce `error`. These counts remain in
the scheduled-match denominator. Official scores and winner rewards are separate;
Machi Koro is a race with null official victory-point scores.

## GPU actor-critic experiments

The neural pipeline uses CPU workers for full games and GPU minibatches for PPO.
It starts from a fixed public heuristic or a frozen population-search prior, with
a trainable residual action scorer and a critic pooling the legal candidates.
This compact representation is an initial baseline, not a full board encoder.

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv-train/bin/python -m boardbench.families.ppo --game jaipur \
  --output outputs/v160/jaipur/ppo_example --device cuda --workers 7 \
  --games 32 --updates 512 --hours 3 --source-commit "$(git rev-parse HEAD)"
```

An optional `--prior outputs/.../best.json` freezes learned linear weights.
Checkpoint and optimizer state are saved after every update. Separate players
have separate GAE chains even when reaction choices or extra turns change the
normal actor order. Variable legal-action padding is masked in actor and critic.
Inference defaults to deterministic legal-action selection. Training trajectories
use sampled actions; incomplete games never receive fabricated terminal targets.

`best.json` points to a portable, identity-checked `.pt` model. The original
heuristic remains separately available regardless of neural promotion. A completed
confirmation is final for that directory; new training requires a new run.

## Checked SSH launches

`python -m boardbench.families.cluster --help` describes the shared launcher.
Its private inventory maps previously reachable host names to IPs. A private JSON
plan assigns each job a game, `kind` (`cem` or `ppo`), host, eight-CPU slot, output
directory, hours, and GPU for PPO. Optional `prior` points to a frozen linear bot.
The launcher requires an immutable Git release path and an absolute UTC deadline.
It checks fresh CPU/GPU headroom, rejects CPU overlap with its live tmux sessions,
and applies affinity plus a terminating wall-clock watchdog. Host batches launch
independently; reservations on each host are serialized. Private inventory and
run manifests remain under ignored `outputs/`, never in source control.
