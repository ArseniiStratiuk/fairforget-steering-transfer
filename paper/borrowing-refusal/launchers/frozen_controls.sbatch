#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=06:00:00
#SBATCH --job-name=lapa-frozen-controls
#SBATCH --output=logs/lapa-frozen-controls-%j.log
# Frozen-arm addition references (random, off-target, native) with a fresh unmodified
# baseline on all four test shards, then the unmodified MamayLM test answers (E0).
set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1
cd 
F=results/final/sibling-refusal-borrowing-v1
export PYTHONPATH="$PWD/$F/full-runtime-src:$PWD/$F:${PYTHONPATH:-}"
export GENERATION_BATCH=64 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
PY=python
nvidia-smi --query-gpu=index,name,memory.used --format=csv
pids=()
(CUDA_VISIBLE_DEVICES=0 $PY $F/run_lapa_main.py frozen_controls 0 4 > logs/frozen_controls-w0-$SLURM_JOB_ID.log 2>&1 \
  && CUDA_VISIBLE_DEVICES=0 $PY $F/donor_baseline.py mamay > logs/e0-mamay-$SLURM_JOB_ID.log 2>&1) &
pids+=($!)
for gpu in 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu $PY $F/run_lapa_main.py frozen_controls $gpu 4 > logs/frozen_controls-w$gpu-$SLURM_JOB_ID.log 2>&1 &
  pids+=($!)
done
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
exit $status
