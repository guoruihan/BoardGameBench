#!/usr/bin/env bash
set -euo pipefail
cd /datapool/data3/storage/ruihan/code/boardgame
export PYTHONPATH=src PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
v136_root=outputs/v136/skills_a226_20261021_a
case "${1:-all}" in
  preflight)
    .venv-train/bin/python -u -m boardbench.v136.preflight --repo . --root "$v136_root" 2>&1 | tee -a "$v136_root/preflight.log"
    ;;
  launch|resume)
    .venv-train/bin/python -u -m boardbench.v136.campaign "$1" --repo . --root "$v136_root" 2>&1 | tee -a "$v136_root/campaign.log"
    ;;
  all)
    bash tasks/main.harmonies-v136-goal-skills/run_a226.sh preflight
    bash tasks/main.harmonies-v136-goal-skills/run_a226.sh launch
    ;;
  *) echo 'Usage: run_a226.sh [preflight|launch|resume|all]' >&2; exit 2 ;;
esac
