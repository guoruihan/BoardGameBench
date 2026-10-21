#!/usr/bin/env bash
set -euo pipefail
cd /datapool/data3/storage/ruihan/code/boardgame
export PYTHONPATH=src PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
v137_root=outputs/v137/selection_a226_20261022_b
case "${1:-preflight}" in
  preflight)
    .venv-train/bin/python -u -m boardbench.v137.preflight --repo . --root "$v137_root" 2>&1 | tee -a "$v137_root/preflight.log"
    ;;
  launch|resume)
    .venv-train/bin/python -u -m boardbench.v137.campaign "$1" --repo . --root "$v137_root" 2>&1 | tee -a "$v137_root/campaign.log"
    ;;
  *) echo 'Usage: run_a226.sh [preflight|launch|resume]' >&2; exit 2 ;;
esac
