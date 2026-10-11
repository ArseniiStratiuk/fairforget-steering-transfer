#!/bin/bash
#SBATCH --account=e-dev-2026d09-075
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:gh200:1
#SBATCH --gpus-per-task=gh200:1
#SBATCH --time=02:00:00
#SBATCH --job-name=frozen-transfer
#SBATCH --output=logs/frozen-transfer-%j.log

set -euo pipefail
umask 007
module load Stages/2026 PyTorch/2.9.1 JupyterLab/4.5.5
source /e/project1/e-dev-2026d09-075/fairforget/envs/lapa-py313/bin/activate
cd /e/project1/e-dev-2026d09-075/fairforget-steering-transfer
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8
export MODEL_NAME="${MODEL_NAME:-gemma}"
export RUN_CONTROL="${RUN_CONTROL:-0}"
export RUN_TEST="${RUN_TEST:-0}"
export RUN_UTILITY="${RUN_UTILITY:-0}"
export RUN_GUARD="${RUN_GUARD:-0}"
python -m ipykernel install --user --name steering-baselines --display-name 'JUPITER baselines'
export WORKERS="${WORKERS:-1}"
srun --ntasks="$WORKERS" --gpus-per-task=gh200:1 --gpu-bind=single:1 --kill-on-bad-exit=1 python - <<'PY'
from pathlib import Path
import os
import nbformat
from nbclient import NotebookClient

if int(os.environ["WORKERS"]) > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
print("Worker", os.environ.get("SLURM_PROCID", "0"), "GPU", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)

notebook = nbformat.read("notebooks/09_frozen_transfer.ipynb", as_version=4)
output = Path("results/transfer/frozen-transfer-v2") / os.environ["MODEL_NAME"] / os.environ["SLURM_JOB_ID"]
if int(os.environ["WORKERS"]) > 1:
    output = output / f"worker-{os.environ['SLURM_PROCID']}"
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

if [ "$WORKERS" -gt 1 ]; then
python - <<'PY'
from pathlib import Path
import hashlib
import json
import os

folder = Path("results/transfer/frozen-transfer-v2") / os.environ["MODEL_NAME"] / os.environ["SLURM_JOB_ID"]
prior = Path(os.environ["RESUME_FILE"]) if os.environ.get("RESUME_FILE") else None
rows = [json.loads(line) for line in prior.read_text().splitlines()] if prior else []
for worker in range(int(os.environ["WORKERS"])):
    part = folder / f"worker-{worker}"
    if prior:
        assert json.loads((part / "replay-check.json").read_text())["matched"]
    rows.extend(json.loads(line) for line in (part / "responses.jsonl").read_text().splitlines())
keys = ["condition", "benchmark", "label", "language", "source_id"]
assert len(rows) == 1200
assert len({tuple(row[key] for key in keys) for row in rows}) == 1200
rows.sort(key=lambda row: tuple(str(row[key]) for key in keys))
with (folder / "responses.jsonl").open("x") as output:
    for row in rows:
        output.write(json.dumps(row, ensure_ascii=False) + "\n")
(folder / "merge.metadata.json").write_text(json.dumps({
    "resume_file": str(prior) if prior else None, "resume_sha256": hashlib.sha256(prior.read_bytes()).hexdigest() if prior else None,
    "workers": int(os.environ["WORKERS"]), "responses": len(rows), "replay_checks_passed": True if prior else None, "explicit_do_sample": False
}, indent=2) + "\n")
print("Merged 1200 unique responses.")
PY
fi
