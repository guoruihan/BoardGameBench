#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
BB_REPO="$(pwd -P)"
BB_NAME="${1:?usage: bash scripts/launch_v12.sh UNIQUE_RUN_NAME}"
if [[ ! "$BB_NAME" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo 'run name must contain only letters, digits, underscore or dash' >&2
    exit 2
fi
if [[ -e "outputs/v12/$BB_NAME" || -e "logs/v12/$BB_NAME.log" ]]; then
    echo 'run or log already exists; choose a new name' >&2
    exit 2
fi
mkdir -p logs/v12
tmux new-session -d -s "boardbench-v12-$BB_NAME" \
    "cd '$BB_REPO' && bash -o pipefail -c \"PYTHONNOUSERSITE=1 .venv-train/bin/python -u -m boardbench.v12.launch --repo '$BB_REPO' --plan configs/v12_plan.json --root 'outputs/v12/$BB_NAME' 2>&1 | tee 'logs/v12/$BB_NAME.log'\""
echo "Started boardbench-v12-$BB_NAME; status: outputs/v12/latest_status.json"
