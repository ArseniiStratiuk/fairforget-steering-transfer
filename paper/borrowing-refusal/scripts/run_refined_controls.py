"""Use one allocated GPU for each independent donor-language validation."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

root = Path(os.environ.get("BORROWING_ROOT", ".")).resolve()
folder = root / "results/final/sibling-refusal-borrowing-v1"
job_id = os.environ["SLURM_JOB_ID"]
if job_id not in json.loads((root / "compute/jobs.json").read_text()):
    raise ValueError("Register the job before starting GPU work")
devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
tasks = [("gemma", "en"), ("gemma", "uk"), ("mamay", "en"), ("mamay", "uk")]
if len(devices) != len(tasks):
    raise ValueError("Four independent tasks require four allocated GPUs")


def run_task(task_and_device):
    (model_name, language), device = task_and_device
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
    command = [
        sys.executable,
        "-m",
        "steering_transfer.borrowing_execution",
        "--root",
        str(root),
        "--model",
        model_name,
        "--language",
        language,
    ]
    path = folder / f"controls-{model_name}-{language}-job-{job_id}.log"
    print("START", model_name, language, "GPU", device, flush=True)
    with path.open("w") as log:
        subprocess.run(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True
        )
    print("DONE", model_name, language, flush=True)


with ThreadPoolExecutor(max_workers=4) as workers:
    list(workers.map(run_task, zip(tasks, devices)))
