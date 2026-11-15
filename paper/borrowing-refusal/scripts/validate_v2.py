"""Full-length validation of the screen_shortlist_v2 settings in the source model.

  generate <model>-<language> [...]   GPU: candidate and matched random direction, 2048 tokens
  summarize <model>-<language> [...]  network host: joint Jev labels and a summary per setting

Settings already validated in addition_arm_prompt/source_validation are not repeated.
Selection afterwards: most (coherent refusals - incoherent), then more coherent refusals,
then smaller k, over all validated settings of a direction.
"""

import json
import sys

import pandas as pd

import run_addition_arm as arm_code
from steering_transfer.borrowing_experiment import evaluate_experiment_answers
from steering_transfer.direction_processing import create_random_directions
from steering_transfer.interventions import add_direction
from steering_transfer.language_models import load_language_model
from steering_transfer.records import GreedyDecoding
from steering_transfer.response_generation import generate_responses
from steering_transfer.result_storage import read_records

arm = arm_code.arm
configuration = arm_code.configuration
GENERATION_BATCH = 64


def pending_settings(task):
    shortlist = json.loads((arm / "screen_shortlist_v2.json").read_text())["shortlist"][task]
    done = arm / "source_validation" / task
    return [
        (int(row["layer"]), float(row["multiplier"]))
        for row in shortlist
        if not (done / f"L{int(row['layer'])}_k{float(row['multiplier']):g}" / "source_validation.csv").exists()
    ]


def generate(task):
    model_name, language = task.split("-")
    harmless = arm_code.prompts.query(
        'split == "validation" and label == "harmless" and language == @language'
    ).copy()
    assert len(harmless) == 64
    conditions = []
    for layer, multiplier in pending_settings(task):
        direction = arm_code.block_direction(model_name, language, layer)
        strength = multiplier * direction.provenance["raw_norm"]
        random_direction = create_random_directions(
            direction, [arm_code.seed], direction.provenance["excluded_coordinates"]
        )[arm_code.seed]
        name = f"L{layer}_k{multiplier:g}"
        conditions.append(add_direction(f"addition_{name}", direction, layer, strength, "prompt"))
        conditions.append(add_direction(f"random_{name}", random_direction, layer, strength, "prompt"))
    if not conditions:
        return
    specification = configuration["models"][model_name]
    with load_language_model(model_name, specification["revision"], specification["checkpoint"]) as model:
        generate_responses(
            model, harmless, conditions,
            GreedyDecoding(configuration["generation"]["max_new_tokens"]), GENERATION_BATCH,
            arm / "source_validation_v2" / task / "responses",
        )
    print("GENERATED", task, len(conditions), flush=True)


def summarize(task):
    from steering_transfer.judges.jev import JevJudge

    folder = arm / "source_validation_v2" / task
    answers = read_records(folder / "responses/responses.jsonl")
    settings = arm_code.judge_settings()
    with JevJudge(settings["model"], settings["rubrics"]["behavior"]) as judge:
        labels = evaluate_experiment_answers(answers, judge, settings, folder / "judgments")
    joined = answers.merge(labels, on="response_id", validate="one_to_one")
    rows = []
    for name, group in joined.groupby("condition", sort=False):
        refusal = group.behavior.eq("refusal")
        incoherent = group.coherence.eq("incoherent")
        rows.append({
            "condition": name, "answers": len(group), "refusals": int(refusal.sum()),
            "coherent_refusals": int((refusal & ~incoherent).sum()),
            "harmful_assistance": int(group.harmful_fulfillment.eq("yes").sum()),
            "incoherent": int(incoherent.sum()),
            "at_token_cap": int(group.response_ids.map(len).ge(configuration["generation"]["max_new_tokens"]).sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(folder / "source_validation.csv", index=False)
    print(task); print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    stage, tasks = sys.argv[1], sys.argv[2:]
    for task in tasks:
        {"generate": generate, "summarize": summarize}[stage](task)
