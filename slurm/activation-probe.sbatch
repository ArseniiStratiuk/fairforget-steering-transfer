#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --gpus-per-task=gh200:1
#SBATCH --time=00:15:00
#SBATCH --job-name=activation-probe
#SBATCH --output=logs/activation-probe-%j.log

set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1 JupyterLab/4.5.5
source /e/project1/e-dev-2026d09-075/fairforget/envs/lapa-py313/bin/activate
cd /e/project1/e-dev-2026d09-075/fairforget-steering-transfer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export RUN_ACTIVATION_PROBE="${RUN_ACTIVATION_PROBE:-1}"
export RUN_FULL_EXTRACTION="${RUN_FULL_EXTRACTION:-0}"
export RUN_VALIDATION="${RUN_VALIDATION:-0}"
export RUN_GUARD="${RUN_GUARD:-0}"
export RUN_BELEBELE="${RUN_BELEBELE:-0}"
python -m ipykernel install --user --name steering-baselines --display-name 'JUPITER baselines'
srun --gpu-bind=single:1 --kill-on-bad-exit=1 python - <<'PY'
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

notebook = nbformat.read("notebooks/08_gemma_ablation.ipynb", as_version=4)
run_group = "utility" if os.environ.get("RUN_BELEBELE") == "1" else (
    "scoring" if os.environ.get("RUN_GUARD") == "1" else (
        "validation" if os.environ.get("RUN_VALIDATION") == "1" else "activations"
    )
)
output = Path("results") / run_group / "gemma-english-pilot-v1" / os.environ["SLURM_JOB_ID"]
output.mkdir(parents=True, exist_ok=True)


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
    nbformat.write(notebook, output / "executed.ipynb")
PY
