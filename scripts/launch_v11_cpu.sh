#!/usr/bin/env bash
set -euo pipefail
tag=${1:?unique job tag}; shift
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ && "$#" -gt 0 ]] || exit 2
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
session="boardbench-v11-${tag}"
log="logs/v11/${tag}.log"
[[ ! -e "$log" ]] || exit 2
tmux has-session -t "$session" 2>/dev/null && exit 2
.venv/bin/python - <<'PY'
import os
load=os.getloadavg()[0]; cpus=os.cpu_count()
assert load/cpus<.7, (load,cpus)
print('CPU preflight',load,'/',cpus,flush=True)
PY
mkdir -p logs/v11
printf -v command_line '%q ' "$@"
printf -v root_quoted '%q' "$project_root"
printf -v log_quoted '%q' "$log"
launch="cd $root_quoted && PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $command_line 2>&1 | tee $log_quoted"
printf -v launch_quoted '%q' "$launch"
tmux new-session -d -s "$session" "bash -o pipefail -c $launch_quoted"
echo "Started $session CPU log=$log"
