#!/usr/bin/env bash
# New bounded campaign; original eight-hour deadline is never reset on resume.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p outputs/v144
export PYTHONPATH=src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv-train/bin/python -u -m boardbench.v144.campaign \
  --repo "$PWD" --config configs/v144_a226_own_score.json \
  --out outputs/v144/own_score_a226_20261022_a \
  2>&1 | tee -a outputs/v144/own_score_a226_20261022_a.controller.log
