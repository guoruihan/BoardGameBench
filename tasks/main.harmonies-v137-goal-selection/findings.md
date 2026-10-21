# Findings

Record only findings that change a decision, hypothesis, implementation contract, or
interpretation. Link to exact code, run IDs, logs, artifacts, papers, or commits instead of
copying their full contents.

| date | finding | evidence | confidence | implication |
|---|---|---|---|---|
| 2026-10-22 | V136 goal execution improved to60.4–62.5% but total score did not robustly improve; manager was never tested. | outputs/v136/skills_a226_20261021_a; docs/Harmonies_V137_Goal_Selection.md | high | Test goal choice directly, retain both skill parents without confirmation-based selection. |
| 2026-10-22 | V137 pilot8 own-policy train environments/32 states: sample-fit selected gain+1.016; independent futures about−0.2.25/32 states did not beat baseline on fitting means. | outputs/v137/selection_a226_20261022_a/preflight_attempt_01/pilot/dataset.json and goals.npz | high for pilot, limited generalization | No positive claim; increase to64 environments and4+4 futures, track candidate ties. |
| 2026-10-22 | Actual a226 CUDA fixture and mature-model preflight passed; selector worker change0, joint worker and manager nonzero; exact next batch resume. | outputs/v137/selection_a226_20261022_a/preflight_attempt_01/receipt.json | high | Source changed for shared stage; repeat gate rather than reuse stale pass. |
| 2026-10-22 | Joint/refresh first-stage update and seed/config are identical. | same receipt; src/boardbench/v137/campaign.py | high | Share first-stage checkpoint; fork only on new labels, saving8M actions. |
| 2026-10-22 | a226 GPU0–3 idle,4–7 in use at initial probe. | task_plan.md; preflight resource gate records | transient | Recheck before each stage; <=57 physical cores and GPU0–3, never preempt unrelated tasks. |
