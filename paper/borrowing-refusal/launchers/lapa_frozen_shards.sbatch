#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=12:00:00
#SBATCH --job-name=lapa-frozen-shards
#SBATCH --output=logs/lapa-frozen-shards-%j.log
# Frozen-rule Lapa shards 2 and 3 (removal with its own unmodified baseline, and the
# frozen addition arm). Each shard pairs with the baseline generated in the same node.
set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1
cd 
F=results/final/sibling-refusal-borrowing-v1
export PYTHONPATH="$PWD/$F/full-runtime-src:$PWD/$F:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
PY=python
nvidia-smi --query-gpu=index,name,memory.used --format=csv
pids=()
gpu=0
for task in "frozen_removal 2" "frozen_removal 3" "frozen_addition 2" "frozen_addition 3"; do
  set -- $task
  CUDA_VISIBLE_DEVICES=$gpu $PY $F/run_lapa_main.py $1 $2 4 > logs/$1-w$2-$SLURM_JOB_ID.log 2>&1 &
  pids+=($!); gpu=$((gpu + 1))
done
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
exit $status
