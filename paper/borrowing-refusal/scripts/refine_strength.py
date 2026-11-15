"""Refine only the approved 0.5-to-1 interval using frozen validation prompts."""

import os
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from steering_transfer.borrowing_experiment import evaluate_experiment_answers
from steering_transfer.datasets import load_frozen_prompts
from steering_transfer.interventions import add_direction
from steering_transfer.judges.jev import JevJudge
from steering_transfer.language_models import load_language_model
from steering_transfer.queued_judging import QueuedJudge
from steering_transfer.records import Direction, GreedyDecoding
from steering_transfer.response_generation import generate_responses
from steering_transfer.result_storage import content_hash, file_hash, write_json

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--language", required=True)
args = parser.parse_args()
root = Path(os.environ.get("BORROWING_ROOT", ".")).resolve()
run = root / "results/final/sibling-refusal-borrowing-v1"
folder = run / args.model / args.language
configuration = json.loads(
    (root / "configs/experiments/borrowing_refusal.json").read_text()
)
spec = configuration["models"][args.model]
pin = configuration["judges"]["jev"]["selected_configuration"]
assert file_hash(root / pin["path"]) == pin["sha256"]
judge_configuration = json.loads((root / pin["path"]).read_text())
saved = torch.load(folder / "selected_direction.pt", weights_only=True)
direction = Direction(saved["values"], saved["provenance"])
records = (
    load_frozen_prompts(root)
    .query(
        'split == "validation" and label == "harmless" and language == @args.language'
    )
    .copy()
)
assert len(records) == 64 and records.prompt_id.is_unique
summary = []
with load_language_model(args.model, spec["revision"], spec["checkpoint"]) as model:
    assert model.specification == direction.provenance["model"]
    with JevJudge(
        judge_configuration["model"], judge_configuration["rubrics"]["behavior"]
    ) as api:
        judge = QueuedJudge(api, run / "judge_queue")
        for multiplier in [0.5, 0.625, 0.75, 0.875, 1.0]:
            trial = folder / "calibration" / f"multiplier-{multiplier:g}"
            condition = add_direction(
                f"addition_{multiplier:g}",
                direction,
                direction.provenance["layer"],
                multiplier * direction.provenance["raw_norm"],
                "all",
            )
            responses = generate_responses(
                model,
                records,
                [condition],
                GreedyDecoding(configuration["generation"]["max_new_tokens"]),
                8,
                trial / "responses",
            )
            labels = evaluate_experiment_answers(
                responses, judge, judge_configuration, trial / "judgments"
            )
            assert len(labels) == 64
            summary.append(
                {
                    "multiplier": multiplier,
                    "coherent_refusals": int(
                        (
                            labels.behavior.eq("refusal")
                            & labels.coherence.eq("coherent")
                        ).sum()
                    ),
                    "incoherent": int(labels.coherence.eq("incoherent").sum()),
                    "answers": len(labels),
                }
            )
            write_json(folder / "refinement_trials.json", summary)
            print(args.model, args.language, summary[-1], flush=True)
ranked = pd.DataFrame(summary).sort_values(
    ["coherent_refusals", "incoherent", "multiplier"], ascending=[False, True, True]
)
write_json(
    folder / "refinement_selection.json",
    {
        "selected": ranked.iloc[0].to_dict(),
        "rule": "most coherent refusals, then fewest incoherent answers, then smallest multiplier",
        "validation_hash": content_hash(records.to_dict("records")),
        "direction_hash": content_hash(direction.values.tolist()),
        "original_configuration_hash": content_hash(configuration),
    },
)
