#!/usr/bin/env bash
set -euo pipefail
game=${1:?game}; tag=${2:?training tag}
extra_tag=${3:-}
case "$game" in micro_tiles|take_it_easy|harmonies) ;; *) exit 2 ;; esac
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ -z "$extra_tag" || "$extra_tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
session="boardbench-val-${game}-${tag}"
output="outputs/v1/${game}/validation_${tag}"
log="logs/v1/${game}_validation_${tag}.log"
[[ ! -e "$output" && ! -e "$log" ]] || exit 2
tmux has-session -t "$session" 2>/dev/null && exit 2
mkdir -p logs/v1
extra_arg=""
if [[ -n "$extra_tag" ]]; then extra_arg="--extra-training outputs/v1/${game}/train_${extra_tag}"; fi
tmux new-session -d -s "$session" "bash -o pipefail -c 'cd \"$project_root\" && PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-train/bin/python -u -m boardbench.benchmark validate --task $game --training outputs/v1/${game}/train_${tag} $extra_arg --out $output 2>&1 | tee $log'"
echo "Started $session output=$output log=$log"
