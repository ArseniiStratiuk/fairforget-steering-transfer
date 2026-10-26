#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:4
#SBATCH --gpus-per-task=gh200:1
#SBATCH --time=04:00:00
#SBATCH --job-name=frozen-refusal
#SBATCH --output=logs/final-%j.log
set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1
cd /e/project1/e-dev-2026d09-075/fairforget-steering-transfer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/src:$PWD/envs/judge-deps:${PYTHONPATH:-}"
: "${STEERING_RUN_STAGE:?Specify experiment stage}"
export STEERING_RUN_MODEL="${STEERING_RUN_MODEL:-gemma}"
/e/project1/e-dev-2026d09-075/fairforget/envs/lapa-py313/bin/python -c 'import json,os; assert os.environ["SLURM_JOB_ID"] in json.load(open("compute/jobs.json")), "Unregistered job"'
srun --gpu-bind=single:1 bash -c '/e/project1/e-dev-2026d09-075/fairforget/envs/lapa-py313/bin/python -m steering_transfer.frozen_experiment "$STEERING_RUN_STAGE" --model "$STEERING_RUN_MODEL" --worker "$SLURM_PROCID"'
