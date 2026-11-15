"""Source calibration and matched controls for bilingual refusal borrowing."""

import json
from pathlib import Path

import pandas as pd

from .direction_processing import create_random_directions, select_direction
from .interventions import (
    REMOVAL_LOCATIONS,
    add_direction,
    remove_direction,
    unchanged_model,
)
from .joint_judging import save_joint_decision
from .records import Direction, GreedyDecoding
from .response_generation import generate_responses
from .result_storage import content_hash, file_hash, write_json
from .validation import (
    find_outlier_coordinates,
    screen_refusal_directions,
    select_validation_layer,
)


def select_refusal_layer(
    model, directions, prompts, configuration, language, output_directory
):
    settings = configuration["layer_selection"]
    model_name = model.specification["model"]
    expected = configuration["models"][model_name]
    if model.specification["revision"] != expected["revision"]:
        raise ValueError("Model revision differs from the frozen configuration")
    validation = prompts.query('split == "validation" and language == @language').copy()
    counts = validation.label.value_counts().to_dict()
    if (
        counts != {"harmful": 64, "harmless": 64}
        or validation.prompt_id.duplicated().any()
    ):
        raise ValueError("Layer selection requires 64 distinct prompts per class")
    excluded = find_outlier_coordinates(
        directions, configuration["extraction"]["outlier_mean_to_median_ratio"]
    )
    folder = Path(output_directory)
    saved_selection = folder / "layer_selection.json"
    saved_screen = folder / "layer_screen.csv"
    if saved_selection.exists() and saved_screen.exists():
        saved = json.loads(saved_selection.read_text())
        expected_identity = {
            "language": language,
            "model": model.specification,
            "configuration_hash": content_hash(configuration),
            "validation_hash": content_hash(validation.to_dict("records")),
        }
        if any(saved.get(key) != value for key, value in expected_identity.items()):
            raise ValueError("Saved source screen identity differs")
        screen = pd.read_csv(saved_screen)
        if screen.source_layer.tolist() != settings["source_layers"]:
            raise ValueError("Saved screen does not cover the configured layers")
        layer = select_validation_layer(screen)
        if layer is None or layer != saved["layer"]:
            raise ValueError("Saved source layer differs from its screen")
        direction = select_direction(directions, layer, "unit", excluded[layer])
        if direction.provenance != saved["direction_provenance"]:
            raise ValueError("Saved source direction identity differs")
        return direction, screen
    screen = screen_refusal_directions(
        model,
        directions,
        validation,
        settings["source_layers"],
        excluded,
        settings["max_harmless_kl_nats"],
        REMOVAL_LOCATIONS,
        proxy_token=settings["refusal_proxy_text"][language],
        expected_proxy_token_id=expected["refusal_proxy_token_ids"][language],
    )
    folder = Path(output_directory)
    folder.mkdir(parents=True, exist_ok=True)
    screen.to_csv(folder / "layer_screen.csv", index=False)
    layer = select_validation_layer(screen)
    if layer is None:
        raise RuntimeError("No layer passes the frozen KL limit; stop and investigate")
    direction = select_direction(directions, layer, "unit", excluded[layer])
    write_json(
        folder / "layer_selection.json",
        {
            "layer": layer,
            "language": language,
            "model": model.specification,
            "configuration_hash": content_hash(configuration),
            "validation_hash": content_hash(validation.to_dict("records")),
            "direction_provenance": direction.provenance,
        },
    )
    return direction, screen


def build_off_target_direction(english, ukrainian, layer, excluded_coordinates):
    import torch

    english_ids = [
        value.rsplit(":", 1)[0] for value in english["provenance"]["negative_ids"]
    ]
    ukrainian_ids = [
        value.rsplit(":", 1)[0] for value in ukrainian["provenance"]["negative_ids"]
    ]
    if len(set(english_ids)) != len(english_ids) or set(english_ids) != set(
        ukrainian_ids
    ):
        raise ValueError(
            "Off-target means require identical paired harmless extraction IDs"
        )
    if english["provenance"]["model"] != ukrainian["provenance"]["model"]:
        raise ValueError("Off-target means must come from the same donor checkpoint")
    english_mean = english["negative_mean"][english["layers"].index(layer)]
    ukrainian_mean = ukrainian["negative_mean"][ukrainian["layers"].index(layer)]
    values = (ukrainian_mean - english_mean).detach().float().clone()
    values[list(excluded_coordinates)] = 0
    norm = values.norm()
    if not torch.isfinite(values).all() or norm == 0:
        raise ValueError(
            "Off-target direction is zero or nonfinite; stop and investigate"
        )
    return Direction(
        values / norm,
        {
            "contrast": "uk_minus_en_harmless_means",
            "layer": layer,
            "normalization": "unit",
            "raw_norm": float(norm),
            "excluded_coordinates": list(excluded_coordinates),
            "english_source": english["provenance"],
            "ukrainian_source": ukrainian["provenance"],
        },
    )


def evaluate_experiment_answers(
    responses, judge, selected_configuration, output_directory
):
    if responses.empty or responses.response_id.duplicated().any():
        raise ValueError("Judging requires distinct nonempty responses")
    if judge.specification["model"] != selected_configuration["model"]:
        raise ValueError("Judge model differs from the selected configuration")
    if selected_configuration["examples"]:
        raise ValueError("This protocol uses no examples")
    results = []
    for record in responses.to_dict("records"):
        save_joint_decision(
            output_directory,
            "jev",
            judge,
            record,
            selected_configuration["rubrics"],
            "experiment",
        )
        path = (
            Path(output_directory)
            / "predictions/jev/experiment/zero-shot"
            / f"{record['response_id']}.json"
        )
        saved = json.loads(path.read_text())
        labels = {row["dimension"]: row["label"] for row in saved["predictions"]}
        results.append({"response_id": record["response_id"], **labels})
    return pd.DataFrame(results)


def calibrate_addition_strength(
    model,
    direction,
    prompts,
    configuration,
    language,
    judge,
    root,
    output_directory,
    batch_size=8,
):
    settings = configuration["addition_calibration"]
    validation = prompts.query(
        'split == "validation" and language == @language and label == "harmless"'
    ).copy()
    if (
        len(validation) != settings["validation_count"]
        or validation.prompt_id.duplicated().any()
    ):
        raise ValueError(
            "Addition calibration requires the frozen harmless validation set"
        )
    model_provenance = direction.provenance["model"]
    if model_provenance != model.specification:
        raise ValueError("Calibration must run in the direction's source model")
    pinned = configuration["judges"]["jev"]["selected_configuration"]
    judge_path = Path(root) / pinned["path"]
    if file_hash(judge_path) != pinned["sha256"]:
        raise ValueError("Selected judge configuration checksum differs")
    judge_configuration = json.loads(judge_path.read_text())
    folder = Path(output_directory)
    trials = []
    for multiplier in settings["strength_multipliers"]:
        strength = multiplier * direction.provenance["raw_norm"]
        condition = add_direction(
            f"addition_{multiplier:g}",
            direction,
            direction.provenance["layer"],
            strength,
            "all",
        )
        trial_folder = folder / f"multiplier-{multiplier:g}"
        responses = generate_responses(
            model,
            validation,
            [condition],
            GreedyDecoding(configuration["generation"]["max_new_tokens"]),
            batch_size,
            trial_folder / "responses",
        )
        labels = evaluate_experiment_answers(
            responses, judge, judge_configuration, trial_folder / "judgments"
        )
        refusal_count = int(labels.behavior.eq(settings["refusal_label"]).sum())
        trials.append(
            {
                "multiplier": multiplier,
                "strength": strength,
                "refusal_count": refusal_count,
                "answer_count": len(labels),
                "incoherent_count": int(labels.coherence.eq("incoherent").sum()),
            }
        )
        write_json(folder / "strength_trials.json", trials)
        if refusal_count >= settings["required_refusal_count"]:
            selected = {
                **trials[-1],
                "configuration_hash": content_hash(configuration),
                "direction_hash": content_hash(direction.values.tolist()),
                "layer": direction.provenance["layer"],
                "language": language,
                "validation_hash": content_hash(validation.to_dict("records")),
            }
            write_json(folder / "addition_selection.json", selected)
            return selected
    raise RuntimeError(
        "No strength meets the frozen refusal rule; stop and investigate"
    )


def select_coherent_strength(trials):
    """Rank measured strengths by coherent refusals, incoherence and strength."""
    required = {"multiplier", "coherent_refusals", "incoherent", "answers"}
    if trials.empty or not required.issubset(trials.columns):
        raise ValueError("Strength selection requires measured calibration counts")
    if (
        trials.multiplier.duplicated().any()
        or trials[list(required)].isna().any().any()
    ):
        raise ValueError("Calibration strengths must be distinct and fully scored")
    for row in trials.to_dict("records"):
        if row["answers"] != 64:
            raise ValueError("Each strength requires all 64 validation judgments")
        counts = [row["coherent_refusals"], row["incoherent"]]
        if any(value < 0 or value != int(value) for value in counts):
            raise ValueError("Calibration counts must be nonnegative integers")
        if sum(counts) > row["answers"] or row["multiplier"] <= 0:
            raise ValueError("Calibration counts or strength are invalid")
    ranked = trials.sort_values(
        ["coherent_refusals", "incoherent", "multiplier"],
        ascending=[False, True, True],
    )
    return ranked.iloc[0].to_dict()


def build_recipient_conditions(direction, off_target, selection, configuration, name):
    if selection["direction_hash"] != content_hash(direction.values.tolist()):
        raise ValueError("Selected strength belongs to a different direction")
    if selection["configuration_hash"] != content_hash(configuration):
        raise ValueError("Selected strength belongs to a different configuration")
    if selection["layer"] != direction.provenance["layer"]:
        raise ValueError("Selected layer differs from direction provenance")
    if off_target.provenance["layer"] != selection["layer"]:
        raise ValueError("Off-target control must use the selected donor block")
    if (
        off_target.provenance["english_source"]["model"]
        != direction.provenance["model"]
    ):
        raise ValueError("Off-target control must use the same donor checkpoint")
    if (
        off_target.provenance["excluded_coordinates"]
        != direction.provenance["excluded_coordinates"]
    ):
        raise ValueError("Off-target control must use the donor mask")
    excluded = direction.provenance["excluded_coordinates"]
    seed = configuration["controls"]["random"]["seed"]
    random_direction = create_random_directions(direction, [seed], excluded)[seed]
    conditions = [unchanged_model()]
    for reference, vector in [
        ("borrowed", direction),
        ("random", random_direction),
        ("off_target", off_target),
    ]:
        conditions.append(
            add_direction(
                f"{name}_{reference}_addition",
                vector,
                selection["layer"],
                selection["strength"],
                "all",
            )
        )
        conditions.append(
            remove_direction(
                f"{name}_{reference}_removal",
                vector,
                configuration["interventions"]["projection_removal"]["layers"],
                REMOVAL_LOCATIONS,
            )
        )
    return conditions


def build_native_conditions(direction, selection, configuration, language):
    if selection["direction_hash"] != content_hash(direction.values.tolist()):
        raise ValueError("Native selection belongs to a different direction")
    if selection["configuration_hash"] != content_hash(configuration):
        raise ValueError("Native selection configuration differs")
    if direction.provenance["model"]["model"] != configuration["recipient"]:
        raise ValueError("Native direction must belong to the recipient")
    return [
        add_direction(
            f"native_{language}_addition",
            direction,
            selection["layer"],
            selection["strength"],
            "all",
        ),
        remove_direction(
            f"native_{language}_removal",
            direction,
            configuration["interventions"]["projection_removal"]["layers"],
            REMOVAL_LOCATIONS,
        ),
    ]


def validate_source_interventions(
    model,
    direction,
    selection,
    prompts,
    configuration,
    language,
    judge,
    root,
    output_directory,
    batch_size=8,
    reusable_responses=None,
    addition_candidate_name="candidate",
):
    """Measure source addition/removal effects without inventing acceptance gates."""
    if direction.provenance["model"] != model.specification:
        raise ValueError("Source validation requires the direction's originating model")
    if selection["direction_hash"] != content_hash(direction.values.tolist()):
        raise ValueError("Source validation selection belongs to another vector")
    if selection["configuration_hash"] != content_hash(configuration):
        raise ValueError("Source validation configuration differs")
    seed = configuration["controls"]["random"]["seed"]
    random_direction = create_random_directions(
        direction, [seed], direction.provenance["excluded_coordinates"]
    )[seed]
    pinned = configuration["judges"]["jev"]["selected_configuration"]
    path = Path(root) / pinned["path"]
    if file_hash(path) != pinned["sha256"]:
        raise ValueError("Selected judge configuration checksum differs")
    judge_configuration = json.loads(path.read_text())
    summaries = []
    for label, operation in [("harmless", "addition"), ("harmful", "removal")]:
        records = prompts.query(
            'split == "validation" and language == @language and label == @label'
        ).copy()
        if len(records) != 64 or records.prompt_id.duplicated().any():
            raise ValueError(
                "Source validation requires 64 distinct requests per class"
            )
        conditions = [unchanged_model()]
        for name, vector in [("candidate", direction), ("random", random_direction)]:
            if operation == "addition":
                condition_name = (
                    addition_candidate_name if name == "candidate" else name
                )
                condition = add_direction(
                    condition_name,
                    vector,
                    selection["layer"],
                    selection["strength"],
                    "all",
                )
            else:
                condition = remove_direction(
                    name,
                    vector,
                    configuration["interventions"]["projection_removal"]["layers"],
                    REMOVAL_LOCATIONS,
                )
            conditions.append(condition)
        folder = Path(output_directory) / operation
        answers = generate_responses(
            model,
            records,
            conditions,
            GreedyDecoding(configuration["generation"]["max_new_tokens"]),
            batch_size,
            folder / "responses",
            reuse_responses=reusable_responses,
        )
        labels = evaluate_experiment_answers(
            answers, judge, judge_configuration, folder / "judgments"
        )
        joined = answers.merge(labels, on="response_id", validate="one_to_one")
        for name, group in joined.groupby("condition", sort=False):
            summaries.append(
                {
                    "operation": operation,
                    "condition": name,
                    "answers": len(group),
                    "refusals": int(group.behavior.eq("refusal").sum()),
                    "coherent_refusals": int(
                        (
                            group.behavior.eq("refusal")
                            & group.coherence.eq("coherent")
                        ).sum()
                    ),
                    "harmful_assistance": int(
                        group.harmful_fulfillment.eq("yes").sum()
                    ),
                    "incoherent": int(group.coherence.eq("incoherent").sum()),
                }
            )
    summary = pd.DataFrame(summaries)
    summary.to_csv(Path(output_directory) / "source_validation.csv", index=False)
    return summary


def run_source_calibration(root, model_name, language, output_directory):
    """Run one source/language task; preserve captures and stop on failed calibration."""
    import torch

    from .activation_extraction import capture_prompt_activations
    from .datasets import load_frozen_prompts
    from .direction_processing import compute_contrastive_directions
    from .judges.jev import JevJudge
    from .language_models import load_language_model

    root = Path(root)
    configuration = json.loads(
        (root / "configs/experiments/borrowing_refusal.json").read_text()
    )
    specification = configuration["models"][model_name]
    prompts = load_frozen_prompts(root)
    folder = Path(output_directory) / model_name / language
    with load_language_model(
        model_name,
        specification["revision"],
        specification["checkpoint"],
        device="cuda:0",
    ) as model:
        actual_proxy = model.tokenizer.encode(
            configuration["layer_selection"]["refusal_proxy_text"][language],
            add_special_tokens=False,
        )
        if actual_proxy != [specification["refusal_proxy_token_ids"][language]]:
            raise ValueError("Frozen proxy token mismatch")
        records = prompts.query(
            'split == "extraction" and language == @language'
        ).copy()
        capture_folder = (
            root / "results/final/bilingual-refusal-v1/captures" / model_name / language
        )
        if not (capture_folder / "activations.pt").exists():
            capture_folder = folder / "capture"
        activations = capture_prompt_activations(
            model,
            records,
            range(len(model.layers)),
            "block_output",
            "last_prompt_token",
            capture_folder,
        )
        directions = compute_contrastive_directions(
            activations,
            records.loc[records.label.eq("harmful"), "prompt_id"],
            records.loc[records.label.eq("harmless"), "prompt_id"],
        )
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(directions, folder / "directions.pt")
        print(f"{model_name}/{language}: capture verified", flush=True)
        historical = root / "results/final/bilingual-refusal-v1/validation" / model_name
        if language == "en" and model_name in {"gemma", "lapa"}:
            old_settings = json.loads(
                (
                    Path(output_directory) / "historical_selection_settings.json"
                ).read_text()
            )
            old_selection = json.loads((historical / "selection.json").read_text())
            same_settings = (
                old_settings["source_layers"]
                == configuration["layer_selection"]["source_layers"]
                and old_settings["max_harmless_kl_nats"]
                == configuration["layer_selection"]["max_harmless_kl_nats"]
                and old_settings["outlier_mean_to_median_ratio"]
                == configuration["extraction"]["outlier_mean_to_median_ratio"]
                and old_selection["data_revision"]
                == configuration["data"]["frozen_split"]["revision"]
            )
            if not same_settings:
                raise ValueError(
                    "Historical source selection settings differ; investigate before reuse"
                )
            screens = [
                pd.read_csv(historical / f"worker-{worker}.csv") for worker in range(4)
            ]
            screen = pd.concat(screens, ignore_index=True).sort_values("source_layer")
            if (
                screen.source_layer.tolist()
                != configuration["layer_selection"]["source_layers"]
            ):
                raise ValueError(
                    "Historical screen does not cover all configured layers"
                )
            layer = select_validation_layer(screen)
            if layer is None or layer != old_selection["layer"]:
                raise ValueError("Historical selected layer no longer matches screen")
            masks = find_outlier_coordinates(
                directions, configuration["extraction"]["outlier_mean_to_median_ratio"]
            )
            if masks[layer] != old_selection["excluded_coordinates"]:
                raise ValueError("Historical direction mask differs")
            direction = select_direction(directions, layer, "unit", masks[layer])
            screen.to_csv(folder / "layer_screen.csv", index=False)
            write_json(
                folder / "layer_selection.json",
                {
                    "layer": layer,
                    "reused_from": str(historical),
                    "configuration_hash": content_hash(configuration),
                },
            )
        else:
            direction, _ = select_refusal_layer(
                model, directions, prompts, configuration, language, folder
            )
        torch.save(
            {"values": direction.values, "provenance": direction.provenance},
            folder / "selected_direction.pt",
        )
        print(
            f"{model_name}/{language}: selected block {direction.provenance['layer']}",
            flush=True,
        )
        pinned = configuration["judges"]["jev"]["selected_configuration"]
        judge_path = root / pinned["path"]
        if file_hash(judge_path) != pinned["sha256"]:
            raise ValueError("Selected judge configuration checksum differs")
        judge_settings = json.loads(judge_path.read_text())
        with JevJudge(
            judge_settings["model"], judge_settings["rubrics"]["behavior"]
        ) as api_judge:
            from .queued_judging import QueuedJudge

            judge = QueuedJudge(api_judge, Path(output_directory) / "judge_queue")
            selection = calibrate_addition_strength(
                model,
                direction,
                prompts,
                configuration,
                language,
                judge,
                root,
                folder / "calibration",
            )
            print(
                f"{model_name}/{language}: selected multiplier {selection['multiplier']}",
                flush=True,
            )
            summary = validate_source_interventions(
                model,
                direction,
                selection,
                prompts,
                configuration,
                language,
                judge,
                root,
                folder / "source_validation",
            )
        write_json(
            folder / "complete.json",
            {"selection": selection, "source_validation": summary.to_dict("records")},
        )
        print(f"{model_name}/{language}: complete", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=["gemma", "mamay", "lapa"], required=True)
    parser.add_argument("--language", choices=["en", "uk"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run_source_calibration(
        arguments.root, arguments.model, arguments.language, arguments.output
    )
