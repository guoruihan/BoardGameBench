# Engineering changes

## 2026-10-03 — V1.1 review fixes and stronger baselines

- Reject cross-run train/holdout contamination before candidate evaluation; retain
  run-qualified identities and full provenance. Load each candidate's actual
  hidden width, network and encoding metadata rather than assuming 128.
- Add action-feature PPO and explicitly supervised heuristic imitation for Micro
  and TIE; retain Harmonies flat PPO. Add fixed-set diagnostic and three-seed plans.
- Add turn-aware, hard-budget search and animal-potential leaf evaluation, public
  information safeguards, component metrics and deterministic Runner seed shards.
- Verify 132 tests, 4992 completed frozen test games, 39 policy hashes and 15
  trained-policy isolated restores. Preserve V1 source-bound models and archive.
- Add real animal placement/completion/save browser scenario. Full three-game
  browser regression passed after compacting display-only policy labels; preserve
  the initial overflow screenshots and all substantive model metadata.
- Publish package 0.3.0 source and reproducible launch/report/audit/package scripts;
  generated training models, raw logs and screenshots remain in the local bundle.

## 2026-10-03 — ZIP restore regression

- Independent restoration from the first V1 ZIP exposed missing empty workspace
  directories. Preserve required empty directories explicitly in the ZIP manifest
  and entries, and add a real export/extract/Runner-restore regression test.
- Keep the failed candidate archive for diagnosis; the final archive uses a new
  filename. No game, policy weights, frozen evaluation or runtime source changed.
- Final full suite after the two packaging regressions: 107 passed.

## 2026-10-03 — V1 initial repository delivery

- Preserve V0.1 and add Micro Tiles, Take It Easy!, Harmonies solo A engines,
  public-information search, finite masked RL encodings and exact game saves.
- Add random/heuristic/bounded-search/PPO, real GPU training, validation-only
  selection, portable policy manifests and frozen common-Runner evaluation.
- Add three-game human/watch/hint/takeover/challenge UI and explicit-task replay.
- Verify 105 tests, actual browser interactions, three moved-policy restores,
  and 1920 completed frozen test games with no illegal actions or failures.
- Source is published separately from generated runs/weights; local review bundle
  includes trained/untrained artifacts, provenance, reports and raw evaluation logs.
