"""Prompt-only addition arm with an addition-specific block, chosen on donor validation prompts.

Removal keeps the frozen direction and block. For addition, each source uses the
difference-in-means direction of the block chosen by the layer screen and full-length
validation (most coherent refusals, then fewest incoherent answers, then smaller
strength). Random and off-target controls use the same block and strength.

Stages:
  validate <model>-<language> <layer> <multiplier>   full 2048-token check in the source
  transfer <worker> <workers>                        Lapa test answers and Belebele
"""

import os
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from steering_transfer.borrowing_execution import run_transfer_conditions
from steering_transfer.borrowing_experiment import (
    build_off_target_direction,
    evaluate_experiment_answers,
)
from steering_transfer.datasets import load_frozen_prompts
from steering_transfer.direction_processing import (
    create_random_directions,
    select_direction,
)
from steering_transfer.interventions import add_direction
from steering_transfer.judges.jev import JevJudge
from steering_transfer.language_models import load_language_model
from steering_transfer.records import GreedyDecoding
from steering_transfer.response_generation import generate_responses
from steering_transfer.result_storage import file_hash, write_json
from steering_transfer.validation import find_outlier_coordinates

root = Path(os.environ.get("BORROWING_ROOT", ".")).resolve()
run = root / "results/final/sibling-refusal-borrowing-v1"
arm = run / "addition_arm_prompt"
configuration = json.loads((root / "configs/experiments/borrowing_refusal.json").read_text())
prompts = load_frozen_prompts(root)
seed = configuration["controls"]["random"]["seed"]


def block_direction(model_name, language, layer):
    folder = run / model_name / language
    directions = torch.load(folder / "directions.pt", weights_only=True)
    excluded = find_outlier_coordinates(
        directions, configuration["extraction"]["outlier_mean_to_median_ratio"]
    )
    return select_direction(directions, layer, "unit", excluded[layer])


def judge_settings():
    pin = configuration["judges"]["jev"]["selected_configuration"]
    if file_hash(root / pin["path"]) != pin["sha256"]:
        raise ValueError("Selected judge configuration checksum differs")
    return json.loads((root / pin["path"]).read_text())


def validate(task, layer, multiplier):
    model_name, language = task.split("-")
    direction = block_direction(model_name, language, layer)
    strength = multiplier * direction.provenance["raw_norm"]
    random_direction = create_random_directions(
        direction, [seed], direction.provenance["excluded_coordinates"]
    )[seed]
    harmless = prompts.query(
        'split == "validation" and label == "harmless" and language == @language'
    ).copy()
    assert len(harmless) == 64
    conditions = [
        add_direction(f"addition_L{layer}_k{multiplier:g}", direction, layer, strength, "prompt"),
        add_direction(f"random_L{layer}_k{multiplier:g}", random_direction, layer, strength, "prompt"),
    ]
    folder = arm / "source_validation" / task / f"L{layer}_k{multiplier:g}"
    specification = configuration["models"][model_name]
    settings = judge_settings()
    with load_language_model(model_name, specification["revision"], specification["checkpoint"]) as model:
        answers = generate_responses(
            model, harmless, conditions,
            GreedyDecoding(configuration["generation"]["max_new_tokens"]), 32,
            folder / "responses",
        )
    with JevJudge(settings["model"], settings["rubrics"]["behavior"]) as judge:
        labels = evaluate_experiment_answers(answers, judge, settings, folder / "judgments")
    joined = answers.merge(labels, on="response_id", validate="one_to_one")
    rows = []
    for name, group in joined.groupby("condition", sort=False):
        rows.append({
            "condition": name,
            "answers": len(group),
            "refusals": int(group.behavior.eq("refusal").sum()),
            "coherent_refusals": int((group.behavior.eq("refusal") & group.coherence.eq("coherent")).sum()),
            "harmful_assistance": int(group.harmful_fulfillment.eq("yes").sum()),
            "incoherent": int(group.coherence.eq("incoherent").sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "source_validation.csv", index=False)
    write_json(folder / "setting.json", {
        "task": task, "layer": layer, "multiplier": multiplier, "strength": strength,
        "direction_provenance": direction.provenance,
    })
    print(task, layer, multiplier, summary.to_dict("records"), flush=True)


def transfer_conditions(settings_file="selected_settings.json"):
    choices = json.loads((arm / settings_file).read_text())
    conditions = []
    for task, choice in choices.items():
        # "<model>-<language>" is the rule-selected setting; "<model>-<language>@<tag>" a
        # labeled sensitivity setting with the same controls.
        task, _, tag = task.partition("@")
        model_name, language = task.split("-")
        layer, strength = choice["layer"], choice["strength"]
        direction = block_direction(model_name, language, layer)
        if abs(choice["multiplier"] * direction.provenance["raw_norm"] - strength) > 1e-3:
            raise ValueError("Selected strength does not match its block direction")
        name = f"{model_name}_{language}" if model_name != "lapa" else f"native_{language}"
        name = f"{name}_{tag}" if tag else name
        conditions.append(add_direction(f"{name}_addition", direction, layer, strength, "prompt"))
        if model_name == "lapa":
            continue
        random_direction = create_random_directions(
            direction, [seed], direction.provenance["excluded_coordinates"]
        )[seed]
        means = [torch.load(run / model_name / lang / "directions.pt", weights_only=True) for lang in ("en", "uk")]
        off_target = build_off_target_direction(
            means[0], means[1], layer, direction.provenance["excluded_coordinates"]
        )
        conditions.append(add_direction(f"{name}_random_addition", random_direction, layer, strength, "prompt"))
        conditions.append(add_direction(f"{name}_off_target_addition", off_target, layer, strength, "prompt"))
    return conditions


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "validate":
        validate(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))
    elif stage == "transfer":
        worker, workers = int(sys.argv[2]), int(sys.argv[3])
        raise SystemExit("use run_lapa_main.py addition")
        print("DONE transfer worker", worker, flush=True)
    else:
        raise SystemExit("unknown stage")
