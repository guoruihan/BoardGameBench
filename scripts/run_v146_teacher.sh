#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p outputs/v146
export PYTHONPATH=src PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
.venv-train/bin/python -u -m boardbench.v146.campaign \
  --repo "$PWD" --config configs/v146_b35_teacher.json \
  --out outputs/v146/duo_teacher_b35_20261023_a \
  2>&1 | tee -a outputs/v146/duo_teacher_b35_20261023_a.controller.log
