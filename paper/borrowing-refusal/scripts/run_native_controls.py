"""Select and validate the native comparator using existing measured strengths."""

import os
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from steering_transfer.borrowing_execution import summarize_saved_strengths
from steering_transfer.borrowing_experiment import (
    select_coherent_strength,
    validate_source_interventions,
)
from steering_transfer.datasets import load_frozen_prompts
from steering_transfer.judges.jev import JevJudge
from steering_transfer.language_models import load_language_model
from steering_transfer.queued_judging import QueuedJudge
from steering_transfer.records import Direction
from steering_transfer.result_storage import (
    content_hash,
    file_hash,
    read_records,
    write_json,
)

parser = argparse.ArgumentParser()
parser.add_argument("--language", choices=["en", "uk"], required=True)
language = parser.parse_args().language
root = Path(os.environ.get("BORROWING_ROOT", ".")).resolve()
run = root / "results/final/sibling-refusal-borrowing-v1"
folder = run / "lapa" / language
configuration = json.loads(
    (root / "configs/experiments/borrowing_refusal.json").read_text()
)
pin = configuration["judges"]["jev"]["selected_configuration"]
assert file_hash(root / pin["path"]) == pin["sha256"]
judge_settings = json.loads((root / pin["path"]).read_text())
trials = summarize_saved_strengths(folder / "calibration", judge_settings)
chosen = select_coherent_strength(trials)
saved = torch.load(folder / "selected_direction.pt", weights_only=True)
direction = Direction(saved["values"], saved["provenance"])
prompts = load_frozen_prompts(root)
validation = prompts.query(
    'split == "validation" and label == "harmless" and language == @language'
)
selection = {
    **chosen,
    "language": language,
    "layer": direction.provenance["layer"],
    "strength": chosen["multiplier"] * direction.provenance["raw_norm"],
    "configuration_hash": content_hash(configuration),
    "direction_hash": content_hash(direction.values.tolist()),
    "validation_hash": content_hash(validation.to_dict("records")),
    "measured_multipliers": trials.multiplier.tolist(),
    "selection_rule": "most coherent refusals, then fewest incoherent answers, then smallest multiplier",
}
paths = [
    folder / "source_validation" / operation / "responses/responses.jsonl"
    for operation in ["addition", "removal"]
]
paths.append(
    folder
    / "calibration"
    / f"multiplier-{chosen['multiplier']:g}/responses/responses.jsonl"
)
reusable = pd.concat([read_records(path) for path in paths], ignore_index=True)
specification = configuration["models"]["lapa"]
with load_language_model(
    "lapa", specification["revision"], specification["checkpoint"]
) as model:
    with JevJudge(
        judge_settings["model"], judge_settings["rubrics"]["behavior"]
    ) as api:
        judge = QueuedJudge(api, run / "judge_queue")
        validate_source_interventions(
            model,
            direction,
            selection,
            prompts,
            configuration,
            language,
            judge,
            root,
            folder / "refined_source_validation",
            reusable_responses=reusable,
            addition_candidate_name=f"addition_{chosen['multiplier']:g}",
        )
write_json(folder / "native_source_selection.json", selection)
print("Native comparator completed", language, selection, flush=True)
