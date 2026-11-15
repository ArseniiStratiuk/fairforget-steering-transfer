"""Llama Guard 4 safety verdicts, reported separately from the Jev labels.

Guard reads the user prompt and the assistant answer with its native template and policy,
without custom instructions or examples. Targets:
  reference                 the 120 human-labeled judge-comparison answers
  <set>/worker-<k>          harmful-request answers of a completed Lapa shard
"""

import os
import json
import sys
from pathlib import Path

import pandas as pd

from steering_transfer.response_evaluation import evaluate_responses, load_response_judge
from steering_transfer.result_storage import file_hash, read_records

root = Path(os.environ.get("BORROWING_ROOT", ".")).resolve()
run = root / "results/final/sibling-refusal-borrowing-v1"
configuration = json.loads((root / "configs/experiments/borrowing_refusal.json").read_text())
guard = configuration["judges"]["llama_guard_4"]
rubric_path = root / guard["rubric"]["path"]
assert file_hash(rubric_path) == guard["rubric"]["sha256"]
rubric = json.loads(rubric_path.read_text())
checkpoint = Path.home() / "models/Llama-Guard-4-12B" / guard["revision"]


def targets(arguments):
    for item in arguments:
        if item == "reference":
            frames = []
            for reference in guard["reference_files"]:
                path = root / reference["path"]
                assert file_hash(path) == reference["sha256"]
                frames.append(pd.read_json(path, lines=True))
            yield run / "guard_reference", pd.concat(frames, ignore_index=True)
        else:
            folder = run / "lapa_main" / item
            json.loads((folder / "generation_complete.json").read_text())
            answers = read_records(folder / "responses/responses.jsonl")
            yield folder / "guard", answers.loc[answers.label.eq("harmful")].reset_index(drop=True)


judge = load_response_judge("local", "llama_guard_4", guard["revision"], rubric, checkpoint)
with judge:
    for output, answers in targets(sys.argv[1:]):
        labels = evaluate_responses(answers, judge, 4, output)
        invalid = int((labels.status != "success").sum())
        print("GUARD", output, len(labels), "unsafe", int(labels.label.eq("unsafe").sum()), "invalid", invalid, flush=True)
