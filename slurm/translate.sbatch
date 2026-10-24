#!/bin/bash
# Translate the combined AdvBench/JBB prompt set with Gemma and Lapa (notebook 10).
# Usage:  sbatch slurm/translate.sbatch                      # all 620 records, both models
#         ITEM_LIMIT=3 sbatch slurm/translate.sbatch         # smoke test: 3 per source
#         MODELS=lapa sbatch slurm/translate.sbatch          # one model only
#         PROMPT_PAIRS_CSV=/path/to/file.csv sbatch ...      # explicit input copy (SHA256 still checked)
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=01:00:00
#SBATCH --job-name=translate-pairs
#SBATCH --output=logs/translate-%j.log

set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1 JupyterLab/4.5.5
source /e/project1/e-dev-2026d09-075/fairforget/envs/lapa-py313/bin/activate
cd /e/project1/e-dev-2026d09-075/fairforget-steering-transfer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export MODELS="${MODELS:-gemma,lapa}" ITEM_LIMIT="${ITEM_LIMIT:-0}"
python -m ipykernel install --user --name steering-baselines --display-name 'JUPITER baselines'
printf 'Host=%s Models=%s Limit=%s\n' "$(hostname)" "$MODELS" "$ITEM_LIMIT"

# srun --gpu-bind=single:1 makes exactly one GPU visible to the notebook (the
# allocation alone exposed all four: GPU=0,1,2,3 in job 2260256).
WORKER=$(cat <<'PYWORKER'
from pathlib import Path
import os

import nbformat
from nbclient import NotebookClient

notebook = nbformat.read(Path("notebooks/10_translate_prompt_pairs.ipynb"), as_version=4)
executed = Path("logs") / f"translate-{os.environ['SLURM_JOB_ID']}.ipynb"
try:
    NotebookClient(
        notebook,
        timeout=None,
        kernel_name="steering-baselines",
        resources={"metadata": {"path": "notebooks"}},
    ).execute()
finally:
    nbformat.write(notebook, executed)
print("Notebook completed; executed copy:", executed, flush=True)
PYWORKER
)
srun --unbuffered --gpu-bind=single:1 --kill-on-bad-exit=1 bash -c \
    'printf "Visible GPU: %s\n" "$CUDA_VISIBLE_DEVICES"; exec python -c "$0"' "$WORKER"
