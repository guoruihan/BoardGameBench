#!/usr/bin/env bash
# NEW window; resuming cannot renew the absolute deadline.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p outputs/v145
export PYTHONPATH=src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-train/bin/python -u -m boardbench.v145.campaign \
  --repo "$PWD" --config configs/v145_a226_animal.json \
  --out outputs/v145/animal_a226_20261023_b \
  2>&1 | tee -a outputs/v145/animal_a226_20261023_b.controller.log
