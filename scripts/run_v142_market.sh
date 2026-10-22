#!/usr/bin/env bash
# Launch in a named tmux on a226; never overwrites an existing campaign.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p outputs/v142
export PYTHONPATH=src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-train/bin/python -u -m boardbench.v142.campaign \
  --repo "$PWD" --config configs/v142_a226_market.json \
  --out outputs/v142/market_a226_20261022_a \
  2>&1 | tee -a outputs/v142/market_a226_20261022_a.controller.log
