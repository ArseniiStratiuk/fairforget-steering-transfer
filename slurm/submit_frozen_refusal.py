"""Submit and register the frozen experiment dependency chain on JUPITER."""

from datetime import datetime, timezone
import fcntl
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
batch_date = datetime.now(timezone.utc).date().isoformat()
jobs = {}
data_revision = json.loads((root / "configs/frozen_data.json").read_text())["revision"]


def submit(stage, model, hours, dependencies=()):
    command = [
        "sbatch",
        "--hold",
        "--parsable",
        f"--job-name=final-{stage}-{model}",
        f"--time={hours}:00:00",
        f"--export=ALL,STEERING_RUN_STAGE={stage},STEERING_RUN_MODEL={model}",
    ]
    if dependencies:
        command.append("--dependency=afterok:" + ":".join(dependencies))
    command.append("slurm/frozen_refusal.sbatch")
    job = subprocess.check_output(command, text=True, cwd=root).strip().split(";")[0]
    with (root / "compute/jobs.json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "compute/jobs.json"
        registry = json.loads(path.read_text())
        registry[job] = dict(
            date=batch_date,
            purpose=f"Frozen bilingual refusal {stage}: {model}",
            configuration=f"bilingual-refusal-v1; data {data_revision}; one node/four GH200; stage={stage}; model={model}; config=configs/experiments/refusal_transfer.json; dependencies={list(dependencies)}",
            result="Submitted",
        )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(registry, indent=2) + "\n")
        temporary.replace(path)
    subprocess.run(["scontrol", "release", job], check=True)
    jobs[stage + "-" + model] = job
    (root / "compute/latest_frozen_jobs.json").write_text(
        json.dumps(jobs, indent=2) + "\n"
    )
    with (root / "compute/batches" / f"{batch_date}.md").open("a") as note:
        note.write(
            f"\n- Frozen bilingual run {job}: {stage}/{model}, one node/four GPUs, {hours}h cap. Pinned data {data_revision}; configs/experiments/refusal_transfer.json records decoding/selection. Independent one-model-per-GPU workers, reused checkpoints; all answers regenerated. Reconcile final state/runtime.\n"
        )
    print(stage, model, job, flush=True)
    return job


def main():
    capture = submit("capture", "gemma", 1)
    gemma_validation = submit("validate", "gemma", 2, [capture])
    lapa_validation = submit("validate", "lapa", 2, [capture])
    gemma = submit("generate", "gemma", 8, [gemma_validation])
    lapa = submit("generate", "lapa", 8, [gemma_validation, lapa_validation])
    submit("guard", "gemma", 2, [gemma, lapa])
    subprocess.run(["python3", "compute/record_jobs.py"], check=True, cwd=root)


if __name__ == "__main__":
    main()
