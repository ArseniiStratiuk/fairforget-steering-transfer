#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:15:00
#SBATCH --job-name=steering-notebook
#SBATCH --output=logs/notebook-%j.log

set -euo pipefail
umask 077

module load Stages/2026 PyTorch/2.9.1 JupyterLab/4.5.5
source /e/project1/e-dev-2026d09-075/fairforget/envs/lapa-py313/bin/activate

cd /e/project1/e-dev-2026d09-075/fairforget-steering-transfer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export JUPYTER_RUNTIME_DIR="$HOME/.local/share/jupyter/runtime/$SLURM_JOB_ID"
mkdir -p "$JUPYTER_RUNTIME_DIR"

python -m ipykernel install --user --name steering-transfer --display-name 'JUPITER steering transfer'
printf 'Compute node: %s\n' "$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
exec python -m jupyterlab --no-browser --ip=0.0.0.0 --port=8888 --ServerApp.port_retries=0
