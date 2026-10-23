#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
role="${1:?search or learning}"
if [[ "$role" == learning ]]; then
  run_name="${2:-learning_b}"
else
  run_name="${2:-$role}"
fi
case "$run_name" in search|learning|learning_b) ;; *) exit 2 ;; esac
case "$role" in
  search) config=configs/v147_b35_search.json ;;
  learning) config=configs/v147_a226_learning.json ;;
  *) exit 2 ;;
esac
mkdir -p outputs/v147/explore_20261023_a
export PYTHONPATH=src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=""
.venv-train/bin/python -u -m boardbench.v147.launch --repo "$PWD" --config "$config" \
  --out "outputs/v147/explore_20261023_a/$run_name" --role "$role" \
  2>&1 | tee -a "outputs/v147/explore_20261023_a/$run_name.controller.log"
