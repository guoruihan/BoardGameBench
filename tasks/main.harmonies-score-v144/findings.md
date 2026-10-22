# Findings
- V143 running source optimizes score margin and selects development margin. Both current seats train; history seats are excluded. Evidence: src/boardbench/v143/training.py.
- At 14:58 V143 has about 99-102 batches per arm; no conclusion from moving self-play score. Initial frozen-transfer paired score79.4883/animal8.7461 is a verified baseline, not a current result.
- Resource probe: a226 192.168.2.19, GPU0-3 run only our V143; GPU4-7 host unrelated busy tasks and must remain untouched. 57 physical cores already allocated; local sandbox NVML unavailable, SSH CUDA required.
- Existing official native collector, grouped inference, PPO update and exact recovery can be reused. New reward/selection must be versioned together; historical V143 schema must continue rejecting own-score configuration.
- V143 stopped gracefully at216/215/200/200 batches; GPU0-3 released, all checkpoints/traces preserved. Four portable bests and digests: outputs/v144/handoff_v143_20261022_a/handoff.json.
- Fixed second evaluation opponent: old league_a batch200, chosen by old development margin5.6133 (score83.6289, animal12.0430). This is NOT a fresh generalization result.
- V144 changes no official rules or architecture; extends collector schema through versioned class constants/potential hook, preserving V143 margin-only validation. New tests independently replay both objectives and frozen/history exclusions.
- Regression56 passed/one old CUDA-only skipped across two suites. Initial broad test command constrained affinity to one core, causing two expected multiworker guard failures; rerun on57-59 passed without code changes. New real two-round CPU warm-start/confirmation and controller expiry tests pass.
