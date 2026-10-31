# Frozen paired bot evaluation

`boardbench.families.evaluate` runs a small round-robin league for one installed
two-player game. It reuses `train.match_worker` and `play.run_match`; game rules,
policy inference, terminal rewards and runtime-cap semantics stay in those shared
implementations. This tool evaluates fixed candidates and never trains or selects
a replacement bot.

## Input and freezing

Supply a JSON object mapping stable names to policy specifications or JSON files:

```json
{
  "random": {"kind": "random"},
  "heuristic": {"kind": "heuristic"},
  "cem": "jaipur/cem/best.json",
  "ppo": {"spec_file": "jaipur/ppo/best.json"}
}
```

Supported kinds are `random`, `heuristic`, `linear`, `neural`, and
`harmonies_rl` (Harmonies only). Linear specifications contain all 24 weights.
Neural/Harmonies specifications contain `path` and may supply an expected
`sha256`. Referenced JSONs can themselves refer to another JSON; cycles and
`spec_file` plus silent field overrides are rejected.

**Every relative path is relative to the JSON file defining that field.** An
inline model path in `bots.json` is relative to `bots.json`; a model path inside
`ppo/best.json` is relative to that `best.json`. Absolute paths retain their
meaning. The current working directory is never a fallback. This convention
supports portable bot packages without accidentally loading a same-named model
from another directory.

Before playing, the tool:

1. Expands every candidate into its actual spec, fills game/config/rules identity
   and default feature weights, and records referenced JSON hashes.
2. Copies model bytes into `models/<sha256>.pt`, verifies any supplied checksum,
   and uses that copied file. It does not retain a live symlink to a trainer's
   checkpoint. Copies are read-only and checked before and after each match.
3. Writes a signed-by-content `manifest.json` with candidate identities, seeds,
   immutable match settings and the complete schedule. The checksum detects
   accidental changes; it is not an authentication signature.
4. Archives Python, component data and native source under `source.tar.gz`, with
   per-file hashes, Git commit, tracked-dirty flag, Python and NumPy/Torch versions.
   A dirty tree is captured by its actual file contents, not merely its Git SHA.

The manifest and actual models are the evaluation inputs after this point.
Changes to the trainer's `best.json` or its original model do not affect a run or
resume. An entire evaluation directory can move because its internal model copy
also has a relative path. The original paths remain provenance metadata.

## Running and resuming

From a release checkout, for a four-seed exploratory evaluation:

```bash
PYTHONPATH=src .venv-train/bin/python -m boardbench.families.evaluate \
  --game jaipur --bots outputs/v160/bots.json \
  --output outputs/v160/jaipur/league_dev_01 --phase dev \
  --seed-start 13000000 --num-seeds 4 --workers 4 \
  --max-match-seconds 120 --max-decisions 10000 --wall-seconds 1800
```

For four bots and four seeds, the schedule has
`4 choose 2 × 4 seeds × 2 seat orientations = 48` matches. Each pair receives the
same initial environment seed twice, with bot seats exchanged and
`first_player=0` fixed. Environment randomness may diverge later when actions
differ. Policy RNGs follow the shared runner's deterministic seed/seat scheme.
No hidden environment snapshot is passed to a policy.

Add `--resume` to the same command to continue. Operational settings such as
worker count, bootstrap samples and the invocation deadline may change; the
candidate input file, game, phase, seeds and match caps must stay the same.
Referenced `best.json` files are not re-read on resume. Code/data hashes must
match the snapshot; restore the archived source if the checkout has changed.
Run with the recorded dependency versions to reproduce model inference.

A new run requires an empty output directory. Never reuse an old directory for
new candidates, a new cap, or another phase. The evaluator takes a nonblocking
single-writer file lock. `--record` preserves public trajectories in match
results; `--validate` enables per-transition environment invariants.

`--wall-seconds` caps this invocation's match-running window, and
`--deadline-unix` optionally imposes an earlier absolute Unix deadline. At the
cutoff, the evaluator stops its own CPU worker groups, including a stuck single
policy/environment call, saves interrupted attempts, and reports all still
unstarted games. Summary serialization/bootstrap has a small additional cleanup
cost. SIGINT/SIGTERM use the same recoverable stop path. An external SIGKILL can
leave a `running` attempt; the next resume marks that prior attempt interrupted
before retrying it.

Each invocation reads existing attempts once after taking the writer lock, then
keeps the latest result and complete retry history in memory as a mirror of
successful atomic writes. Completion, deadline and exception cleanup summarize
that mirror instead of reopening every attempt on the shared filesystem. This
does not defer or batch attempt persistence. The next invocation still restores
and validates every durable attempt from disk; an offline `save_report(...)`
call also retains that full disk rebuild by default. Cached and offline reports
use the same filename order for resource sums, including floating-point rounding.

Only unstarted, interrupted or deadline-stopped jobs are retried. A completed
match, an explicit per-match `truncated` result, a rule gap, no legal action or an
execution error is retained as measured evidence and is not silently retried.
Fixes or changed per-match caps require a new evaluation. Old attempts remain on
disk when an interrupted scheduled game is retried.

Workers use one BLAS/OpenMP thread each. The caller must choose worker count and
CPU affinity within the authorized host quota; this tool does not discover hosts,
reserve GPUs, launch SSH jobs or alter other workloads. Neural inference here uses
the shared CPU policy loader.

## Development and confirmation

`--phase dev` and `--phase confirmation` are explicit and immutable. Neither mode
writes `best.json`, chooses a checkpoint or ranks candidates into an automatic
winner. Confirmation evaluates only the pool fixed in its manifest before any
confirmation outcomes are observed.

Use fresh seeds and a new directory for confirmation. Known seed exclusions can
be enforced with one or more `--exclude-manifest` arguments. A file must contain
`game` and a `seeds` list; previous evaluation manifests satisfy that schema.
An overlap for the same game is rejected before matches start. Training namespaces
not listed in these manifests remain the caller's responsibility.

```bash
PYTHONPATH=src .venv-train/bin/python -m boardbench.families.evaluate \
  --game jaipur --bots outputs/v160/frozen_confirmation_pool.json \
  --output outputs/v160/jaipur/league_confirm_01 --phase confirmation \
  --seed-start 14000000 --num-seeds 32 --workers 8 \
  --exclude-manifest outputs/v160/jaipur/league_dev_01/manifest.json \
  --max-match-seconds 120 --wall-seconds 3600
```

A caller can misuse a held-out result manually; the tool cannot prevent that
external decision. Keep pool selection in development and retain all confirmation
results, including unsuccessful and incomplete runs.

## Outputs and interpretation

- `manifest.json`, `source.tar.gz`, `models/`: frozen definition and actual bytes.
- `attempts/<match_id>_<attempt>.json`: atomically saved start/result of every
  execution attempt, including interrupted and failed attempts.
- `games.json`: one row for every scheduled game, with pair, environment seed,
  seat mapping and latest result. Unstarted rows are explicit `not_started`.
- `summary.json`: pair reports for both named bots, statuses, score availability,
  seed-cluster intervals and timing. Built from the in-memory mirror of durable
  attempts after stopping or completing; offline rebuilds read those same files.
  Attempt files remain the live progress evidence during play.
- `invocation_*.json`: worker count, cutoff, elapsed wall time and stop reason for
  each run/resume invocation.

A win contributes 1 win point, a draw 0.5 and a loss 0. `completed_win_rate` uses
only terminal games and states its completed denominator. `all_scheduled_win_rate`
uses all planned games but **is a lower bound**, because incomplete outcomes are
unknown; they are not declared losses. `all_scheduled_win_rate_bounds` ranges from
known win points divided by all games to the value if every unknown result were a
win. Status counts retain `error`, `unresolved_rule`, `truncated`, `deadline`,
`interrupted`, `not_started`, and other runner statuses.

Bootstrap samples resample complete **environment-seed groups**, keeping the two
seat orientations together. Reports provide percentile 95% intervals for the
scheduled lower bound and completed-only rate, plus a separate estimate/interval
using only seeds with both games complete. The latter states its paired-seed
count. Resamples with no completed games are counted as undefined, not assigned
zero. A single seed produces a degenerate interval and does not establish strong
evidence; pairwise intervals are not adjusted for trying many candidate pairs.

Official score means use completed games and the correct named bot's seat. If
any completed score is unavailable, the mean is null and observed/missing counts
are reported. Race games with null official scores remain null; no invented
victory-point scale or unfinished intermediate score is substituted.

Timing distinguishes match time, total worker wall time over all attempts,
measured worker CPU time, and invocation wall time. A killed worker's CPU usage
cannot be recovered reliably, so such attempts count under
`unmeasured_cpu_attempts`; the summary does not claim that measured CPU time is a
complete resource bill.

## Checks

```bash
PYTHONPATH=src .venv-train/bin/python -m pytest -q tests/families/test_evaluate.py
```

Tests cover all-pair seat alignment, whole-seed bootstrap, incomplete denominators,
null scores, frozen JSON/model identity, JSON-local paths, code/config rejection,
phase/seed exclusions, a real spawned-process cutoff and resume, abrupt-exit
recovery, retained execution errors, single-writer exclusion and a tiny neural
checkpoint through the common policy loader. Cache checks compare summaries and
game files byte for byte against independent disk rebuilds for normal runs,
cutoffs, abrupt-exit recovery, failed submission, worker errors and a failed result
write followed by resume. Each invocation is checked for exactly one startup
attempt scan and none during cached cleanup. These are local smoke checks,
not strength evaluations or production cluster runs.
