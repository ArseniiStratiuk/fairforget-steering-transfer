#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:4
#SBATCH --gpus-per-task=gh200:1
#SBATCH --time=00:30:00
#SBATCH --job-name=batched-check
#SBATCH --output=logs/batched-%j.log

set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1 JupyterLab/4.5.5
source /e/project1/e-dev-2026d09-075/fairforget/envs/lapa-py313/bin/activate
cd /e/project1/e-dev-2026d09-075/fairforget-steering-transfer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export ITEM_LIMIT="${ITEM_LIMIT:-4}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export RUN_ID="${SLURM_JOB_ID}-limit${ITEM_LIMIT}-b${BATCH_SIZE}-t${MAX_NEW_TOKENS}"
python -m ipykernel install --user --name steering-baselines --display-name 'JUPITER baselines'
srun --gpu-bind=single:1 --kill-on-bad-exit=1 bash slurm/baseline-worker.sh safety
