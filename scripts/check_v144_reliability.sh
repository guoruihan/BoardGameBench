#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p outputs/v144_1
smoke_out=${1:-outputs/v144_1/reliability_smoke_a226_20261023_a}
if [[ -e "$smoke_out" || -e "$smoke_out.log" ]]; then
  echo "Refusing to overwrite verification artifacts: $smoke_out" >&2
  exit 1
fi
export PYTHONPATH=src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_VISIBLE_DEVICES=GPU-19c4a068-0c21-98fe-a3c5-8600922b80bb
timeout --signal=TERM --kill-after=10s 590s taskset -c 0-3 .venv-train/bin/python -u -m boardbench.v144.reliability_smoke \
  --repo "$PWD" --out "$smoke_out" \
  2>&1 | tee "$smoke_out.log"
