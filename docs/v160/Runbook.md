# Original two-player family: rolling development

List installed environments (availability is not a complete-rules certificate):

```bash
.venv-train/bin/python -c 'from boardbench.families.registry import available_games; print(available_games())'
.venv-train/bin/python -m pytest tests/families -q
```

Each game's scope, genuine data sources and unresolved boundaries are documented
under `docs/v160/games/`. Missing games are explicit unavailable entries rather
than substitutes. Harmonies reuses the previous accelerated official duo engine.

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
