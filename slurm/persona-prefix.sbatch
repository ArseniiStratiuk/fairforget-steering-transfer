#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --gpus-per-task=gh200:1
#SBATCH --time=01:00:00
#SBATCH --job-name=persona-prefix
#SBATCH --output=logs/persona-prefix-%j.log

set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1 JupyterLab/4.5.5
source /e/project1/e-dev-2026d09-075/fairforget/envs/lapa-py313/bin/activate
cd /e/project1/e-dev-2026d09-075/fairforget-steering-transfer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export RUN_PERSONA=1
python -m ipykernel install --user --name steering-baselines --display-name 'JUPITER baselines'
srun --gpu-bind=single:1 --kill-on-bad-exit=1 python - <<'PY'
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

notebook = nbformat.read("notebooks/10_persona_prefix.ipynb", as_version=4)
output = Path("results/persona/persona-prefix-pilot-v2") / os.environ["SLURM_JOB_ID"]


def show_progress(cell, **kwargs):
    for item in cell.get("outputs", []):
        if item.get("output_type") == "stream":
            print(item["text"], end="", flush=True)


try:
    NotebookClient(
        notebook,
        timeout=None,
        kernel_name="steering-baselines",
        resources={"metadata": {"path": str(Path.cwd())}},
        on_cell_executed=show_progress,
    ).execute()
finally:
    output.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output / "executed.ipynb")
PY
