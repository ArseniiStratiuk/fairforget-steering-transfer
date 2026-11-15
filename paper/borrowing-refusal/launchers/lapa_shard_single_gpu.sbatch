#!/bin/bash
#SBATCH --job-name=lapa-frozen-w0
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --output=logs/lapa-frozen-w0-%j.log
set -euo pipefail
umask 007
cd fst
F=results/final/sibling-refusal-borrowing-v1
export PYTHONPATH="$PWD/$F/full-runtime-src:$PWD/$F"
export EXPERIMENT_GPU_WORKERS=3
export CHECKPOINT_PREFIX_REMAP="=models-by-pin"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TYPESAFE_API_KEY="$(cat "$HOME/.config/steering-transfer/jev-key")"
nvidia-smi --query-gpu=index,name,memory.used --format=csv
python $F/run_lapa_main.py frozen 0 4
