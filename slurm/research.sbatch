#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=02:00:00
#SBATCH --job-name=steering-research
#SBATCH --output=logs/research-%j.log

set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1 JupyterLab/4.5.5
source /e/project1/e-dev-2026d09-075/fairforget/envs/lapa-py313/bin/activate
cd /e/project1/e-dev-2026d09-075/fairforget-steering-transfer
: "${RESEARCH_NOTEBOOK:?Choose one notebook explicitly}"
: "${EXECUTE_MODEL_STEPS:?Set 0 for saved analysis or 1 for model execution}"

python - "$SLURM_JOB_ID" <<'PY'
import json
import sys
from pathlib import Path

jobs = json.loads(Path("compute/jobs.json").read_text())
if sys.argv[1] not in jobs:
    raise SystemExit("Register this job in compute/jobs.json before execution")
PY

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export PYTHONPATH="$PWD/src:$PWD/envs/judge-deps:${PYTHONPATH:-}"
options=()
if [ "$EXECUTE_MODEL_STEPS" = 1 ]; then
    options+=(--execute-model-steps)
elif [ "$EXECUTE_MODEL_STEPS" != 0 ]; then
    exit 2
fi

srun --gpu-bind=single:1 python -m steering_transfer.run_notebook \
    "$RESEARCH_NOTEBOOK" --output "results/executed-notebooks/${SLURM_JOB_ID}.ipynb" \
    "${options[@]}"
