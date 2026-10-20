#!/usr/bin/env bash
set -euo pipefail
cd /datapool/data3/storage/ruihan/code/boardgame
run_root="$PWD/outputs/v135/advice_a226_20261021_a"
export PYTHONPATH="$PWD/src" PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
case "${1:-}" in
  preflight)
    .venv-train/bin/python -u -m boardbench.v135.preflight \
      --repo "$PWD" --root "$run_root" 2>&1 | tee -a "$run_root/preflight.log"
    ;;
  launch|resume)
    .venv-train/bin/python -u -m boardbench.v135.campaign "$1" \
      --repo "$PWD" --root "$run_root" 2>&1 | tee -a "$run_root/campaign.log"
    ;;
  *) echo "usage: bash scripts/run_v135_a226.sh preflight|launch|resume" >&2; exit 2 ;;
esac
