#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=04:00:00
#SBATCH --job-name=addition-validate-v2
#SBATCH --output=logs/addition-validate-v2-%j.log
# Full-length source validation of the truncation-robust shortlist; judging happens off-node.
set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1
cd 
F=results/final/sibling-refusal-borrowing-v1
export PYTHONPATH="$PWD/$F/full-runtime-src:$PWD/$F:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
PY=python
pids=()
gpu=0
for tasks in "gemma-en" "gemma-uk" "mamay-en mamay-uk" "lapa-en lapa-uk"; do
  CUDA_VISIBLE_DEVICES=$gpu $PY $F/validate_v2.py generate $tasks > logs/validate-v2-gpu$gpu-$SLURM_JOB_ID.log 2>&1 &
  pids+=($!); gpu=$((gpu + 1))
done
status=0
for pid in "${pids[@]}"; do wait $pid || status=1; done
exit $status
