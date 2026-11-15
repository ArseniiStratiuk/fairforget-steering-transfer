"""Prompt-only addition. Shortlist addition blocks: 64-token greedy answers on harmless validation prompts.

Removal keeps the frozen direction and block. Addition acts at the block it is applied to,
so each candidate block uses its own difference-in-means direction (same extraction,
masking and unit scaling) at strength k * ||r_block||. Validation prompts only.
"""

import os
import json
import sys
from pathlib import Path

import torch

from steering_transfer.datasets import load_frozen_prompts
from steering_transfer.direction_processing import select_direction
from steering_transfer.interventions import add_direction
from steering_transfer.language_models import load_language_model
from steering_transfer.records import GreedyDecoding
from steering_transfer.response_generation import generate_responses
from steering_transfer.validation import find_outlier_coordinates

LAYERS = list(range(8, 37, 2))
MULTIPLIERS = [1.0, 2.0, 3.0, 4.0, 6.0]
SCREEN_TOKENS = 64

root = Path(os.environ.get("BORROWING_ROOT", ".")).resolve()
run = root / "results/final/sibling-refusal-borrowing-v1"
configuration = json.loads((root / "configs/experiments/borrowing_refusal.json").read_text())
prompts = load_frozen_prompts(root)

for task in sys.argv[1:]:
    model_name, language = task.split("-")
    folder = run / model_name / language
    output = folder / "addition_prompt_screen"
    if (output / "complete.json").exists():
        continue
    specification = configuration["models"][model_name]
    directions = torch.load(folder / "directions.pt", weights_only=True)
    excluded = find_outlier_coordinates(
        directions, configuration["extraction"]["outlier_mean_to_median_ratio"]
    )
    harmless = prompts.query(
        'split == "validation" and label == "harmless" and language == @language'
    ).copy()
    assert len(harmless) == 64
    conditions = []
    for layer in LAYERS:
        direction = select_direction(directions, layer, "unit", excluded[layer])
        for multiplier in MULTIPLIERS:
            conditions.append(
                add_direction(
                    f"screen_L{layer}_k{multiplier:g}",
                    direction,
                    layer,
                    multiplier * direction.provenance["raw_norm"],
                    "prompt",
                )
            )
    with load_language_model(
        model_name, specification["revision"], specification["checkpoint"]
    ) as model:
        answers = generate_responses(
            model, harmless, conditions, GreedyDecoding(SCREEN_TOKENS), 64, output / "responses"
        )
    assert len(answers) == 64 * len(conditions)
    (output / "complete.json").write_text(
        json.dumps(
            {"layers": LAYERS, "multipliers": MULTIPLIERS, "max_new_tokens": SCREEN_TOKENS,
             "answers": len(answers)},
            indent=2,
        )
        + "\n"
    )
    print("DONE screen", task, flush=True)
