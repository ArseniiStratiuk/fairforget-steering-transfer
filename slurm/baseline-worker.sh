#!/bin/bash
set -euo pipefail
case "${SLURM_PROCID}" in
    0) export MODEL_NAME=gemma SHARD_INDEX=0 ;;
    1) export MODEL_NAME=lapa SHARD_INDEX=0 ;;
    2) export MODEL_NAME=gemma SHARD_INDEX=1 ;;
    3) export MODEL_NAME=lapa SHARD_INDEX=1 ;;
    *) exit 1 ;;
esac
export SHARD_COUNT=2
gpu_list="${CUDA_VISIBLE_DEVICES//[[:space:]]/}"
IFS=',' read -r -a devices <<< "$gpu_list"
if (( ${#devices[@]} == 4 )); then
    export CUDA_VISIBLE_DEVICES="${devices[SLURM_LOCALID]}"
elif (( ${#devices[@]} != 1 )); then
    printf 'Unexpected GPU allocation: %s\n' "$gpu_list" >&2
    exit 1
fi
if (( $# == 0 )); then
    set -- belebele safety
fi
for task in "$@"; do
    if [[ "$task" == belebele ]]; then
        notebook=03_belebele_baseline.ipynb
    else
        notebook=04_safety_baseline.ipynb
    fi
    output="results/baselines/${RUN_ID}/${task}/${MODEL_NAME}/shard-${SHARD_INDEX}"
    mkdir -p "$output"
    printf 'Task=%s Model=%s Shard=%s Host=%s GPU=%s\n' "$task" "$MODEL_NAME" "$SHARD_INDEX" "$(hostname)" "${CUDA_VISIBLE_DEVICES}"
    python - "$notebook" "$output" >"$output/worker.log" 2>&1 <<'PYWORKER'
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

notebook_name, output = sys.argv[1:]
notebook = nbformat.read(Path("notebooks") / notebook_name, as_version=4)
try:
    NotebookClient(
        notebook,
        timeout=None,
        kernel_name="steering-baselines",
        resources={"metadata": {"path": str(Path.cwd())}},
    ).execute()
finally:
    nbformat.write(notebook, Path(output) / "executed.ipynb")
print("Notebook completed", flush=True)
PYWORKER
done
