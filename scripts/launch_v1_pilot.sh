#!/usr/bin/env bash
set -euo pipefail
# Execute on the selected GPU host. No process eviction and no implicit CPU fallback.
game=${1:?game}; gpu=${2:?physical GPU index}; tag=${3:?unique run tag}
config=${4:-configs/${game}_ppo_train.json}
case "$game" in micro_tiles|take_it_easy|harmonies) ;; *) exit 2 ;; esac
[[ "$gpu" =~ ^[0-9]+$ && "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
[[ "$config" =~ ^configs/[a-zA-Z0-9_.-]+\.json$ ]] || exit 2
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
session="boardbench-${game}-${tag}"
output="outputs/v1/${game}/train_${tag}"
log="logs/v1/${game}_${tag}.log"
[[ ! -e "$output" && ! -e "$log" ]] || { echo 'Output already exists; choose a new tag'; exit 2; }
tmux has-session -t "$session" 2>/dev/null && { echo 'Session already exists'; exit 2; }
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$gpu" .venv-train/bin/python - "$gpu" <<'PY'
import subprocess, sys
gpu = sys.argv[1]
row = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True).strip()
name, mem, util = [x.strip() for x in row.split(',')]
procs = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True).strip()
assert '4090' in name and int(mem) <= 100 and int(util) <= 5 and not procs, f'GPU unavailable: {row}; processes={procs}'
import torch
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
x = torch.ones(8, device='cuda', requires_grad=True)
x.square().sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all().item()
print('PASS GPU preflight:', gpu, name, torch.__version__, flush=True)
PY
mkdir -p logs/v1
tmux new-session -d -s "$session" "bash -o pipefail -c 'cd \"$project_root\" && PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv-train/bin/python -u -m boardbench.training --config $config --out $output 2>&1 | tee $log'"
echo "Started $session GPU=$gpu output=$output log=$log"
