#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=00:20:00
#SBATCH --job-name=guard-probe
#SBATCH --output=logs/guard-%j.log

set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1 JupyterLab/4.5.5
source /e/project1/e-dev-2026d09-075/fairforget/envs/lapa-py313/bin/activate
cd /e/project1/e-dev-2026d09-075/fairforget-steering-transfer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
python -m ipykernel install --user --name steering-baselines --display-name 'JUPITER baselines'
export RUN_GUARD=1 GUARD_LIMIT="${GUARD_LIMIT:-10}"
python - <<'PY'
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

notebook = nbformat.read("notebooks/07_baseline_scoring.ipynb", as_version=4)
notebook.cells = [
    cell for cell in notebook.cells if "guard-runtime" in cell.metadata.get("tags", [])
]
output = Path("results/scoring/2131659-limit0-b8-t2048/llama-guard-4")
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
    nbformat.write(notebook, output / f"executed-{os.environ['SLURM_JOB_ID']}.ipynb")
PY
