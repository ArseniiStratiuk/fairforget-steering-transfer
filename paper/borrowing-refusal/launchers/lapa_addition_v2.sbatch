#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=10:00:00
#SBATCH --job-name=lapa-addition-v2
#SBATCH --output=logs/lapa-addition-v2-%j.log
# Final addition arm (selected_settings_v2.json) with its own unmodified baseline, all four shards on one node.
set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1
cd 
F=results/final/sibling-refusal-borrowing-v1
export PYTHONPATH="$PWD/$F/full-runtime-src:$PWD/$F:${PYTHONPATH:-}"
export GENERATION_BATCH=64 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
PY=python
nvidia-smi --query-gpu=index,name,memory.used --format=csv
pids=()
gpu=0
for task in "addition_v2 0" "addition_v2 1" "addition_v2 2" "addition_v2 3"; do
  set -- $task
  CUDA_VISIBLE_DEVICES=$gpu $PY $F/run_lapa_main.py $1 $2 4 > logs/$1-w$2-$SLURM_JOB_ID.log 2>&1 &
  pids+=($!); gpu=$((gpu + 1))
done
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
exit $status
