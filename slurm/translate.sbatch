#!/bin/bash
# Translate the combined AdvBench/JBB prompt set with Gemma and Lapa with the pinned source and model settings.
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
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python - "$SLURM_JOB_ID" <<'PYREGISTER'
import json
import sys
from pathlib import Path

jobs = json.loads(Path("compute/jobs.json").read_text())
if sys.argv[1] not in jobs:
    raise SystemExit("Register this job in compute/jobs.json before execution")
PYREGISTER
srun --unbuffered --gpu-bind=single:1 --kill-on-bad-exit=1 \
    python -m translation_review.translate_prompts
