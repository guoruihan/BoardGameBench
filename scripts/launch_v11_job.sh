#!/usr/bin/env bash
set -euo pipefail
gpu=${1:?physical GPU}; tag=${2:?unique job tag}; shift 2
[[ "$gpu" =~ ^[0-9]+$ && "$tag" =~ ^[a-zA-Z0-9_-]+$ && "$#" -gt 0 ]] || exit 2
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
session="boardbench-v11-${tag}"
log="logs/v11/${tag}.log"
[[ ! -e "$log" ]] || { echo 'Log exists; choose a fresh job tag'; exit 2; }
tmux has-session -t "$session" 2>/dev/null && exit 2
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$gpu" .venv-train/bin/python - "$gpu" <<'PY'
import subprocess, sys
gpu=sys.argv[1]
row=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
name,memory,usage=[x.strip() for x in row.split(',')]
processes=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
assert '4090' in name and int(memory)<=100 and int(usage)<=5 and not processes, row
import torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
x=torch.ones(8,device='cuda',requires_grad=True)
x.square().sum().backward()
torch.cuda.synchronize()
assert torch.isfinite(x.grad).all()
print('GPU preflight passed',gpu,name,flush=True)
PY
mkdir -p logs/v11
printf -v command_line '%q ' "$@"
printf -v root_quoted '%q' "$project_root"
printf -v log_quoted '%q' "$log"
launch="cd $root_quoted && PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 $command_line 2>&1 | tee $log_quoted"
printf -v launch_quoted '%q' "$launch"
tmux new-session -d -s "$session" "bash -o pipefail -c $launch_quoted"
echo "Started $session GPU=$gpu log=$log"
