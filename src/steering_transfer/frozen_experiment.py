"""Execute the frozen bilingual refusal protocol with the shared scientific functions."""

import argparse
import json
from pathlib import Path
import pandas as pd
import torch
from .activation_extraction import capture_prompt_activations
from .datasets import load_frozen_prompts, load_multiple_choice_dataset
from .direction_processing import (
    compute_contrastive_directions,
    create_random_directions,
    select_direction,
)
from .interventions import (
    REMOVAL_LOCATIONS,
    remove_direction,
    unchanged_model,
)
from .language_models import load_language_model
from .records import GreedyDecoding
from .response_evaluation import (
    evaluate_responses,
    load_response_judge,
)
from .response_generation import generate_responses
from .result_storage import read_records, write_json
from .utility_evaluation import evaluate_multiple_choice
from .validation import (
    find_outlier_coordinates,
    screen_refusal_directions,
    select_validation_layer,
)


def _load_frozen_experiment_inputs(root):
    root = Path(root)
    configuration = json.loads(
        (root / "configs/experiments/refusal_transfer.json").read_text()
    )
    models = json.loads((root / "configs/models.json").read_text())
    frozen = json.loads((root / "configs/frozen_data.json").read_text())
    run = root / "results/final" / configuration["name"]
    prompts = load_frozen_prompts(root)
    return configuration, models, frozen, run, prompts


def capture_frozen_activations(root, model_name, language, device="cuda:0"):
    configuration, models, frozen, run, prompts = _load_frozen_experiment_inputs(root)
    records = prompts.query('split == "extraction" and language == @language').copy()
    with load_language_model(model_name, **models[model_name], device=device) as model:
        folder = run / "captures" / model_name / language
        bundle = capture_prompt_activations(
            model,
            records,
            range(len(model.layers)),
            "block_output",
            "last_prompt_token",
            folder,
        )
        directions = compute_contrastive_directions(
            bundle,
            records.loc[records.label.eq("harmful"), "prompt_id"],
            records.loc[records.label.eq("harmless"), "prompt_id"],
        )
        torch.save(directions, folder / "directions.pt")
    return directions


def validate_frozen_direction_layers(root, model_name, worker, device="cuda:0"):
    configuration, models, frozen, run, prompts = _load_frozen_experiment_inputs(root)
    directions = torch.load(
        run / "captures" / model_name / "en/directions.pt", weights_only=True
    )
    records = prompts.query('split == "validation" and language == "en"').copy()
    excluded = find_outlier_coordinates(
        directions, configuration["outlier_mean_to_median_ratio"]
    )
    layers = configuration["source_layers"][worker::4]
    with load_language_model(model_name, **models[model_name], device=device) as model:
        screen = screen_refusal_directions(
            model,
            directions,
            records,
            layers,
            excluded,
            configuration["max_harmless_kl_nats"],
            REMOVAL_LOCATIONS,
            batch_size=configuration["validation_batch_size"],
        )
    folder = run / "validation" / model_name
    folder.mkdir(parents=True, exist_ok=True)
    screen.to_csv(folder / f"worker-{worker}.csv", index=False)
    return screen


def select_frozen_direction(root, model_name):
    configuration, models, frozen, run, prompts = _load_frozen_experiment_inputs(root)
    screens = [
        pd.read_csv(run / "validation" / model_name / f"worker-{i}.csv")
        for i in range(4)
    ]
    screen = pd.concat(screens, ignore_index=True).sort_values("source_layer")
    if screen.source_layer.tolist() != configuration["source_layers"]:
        raise ValueError("Validation does not cover every configured source layer")
    layer = select_validation_layer(screen)
    if layer is None:
        raise RuntimeError(
            f"No {model_name} direction satisfies the configured validation threshold"
        )
    directions = torch.load(
        run / "captures" / model_name / "en/directions.pt", weights_only=True
    )
    excluded = find_outlier_coordinates(
        directions, configuration["outlier_mean_to_median_ratio"]
    )[layer]
    direction = select_direction(directions, layer, "unit", excluded)
    write_json(
        run / "validation" / model_name / "selection.json",
        dict(
            layer=layer,
            excluded_coordinates=excluded,
            max_harmless_kl_nats=configuration["max_harmless_kl_nats"],
            data_revision=frozen["revision"],
        ),
    )
    return direction, excluded


def generate_frozen_answers(root, model_name, worker, device="cuda:0"):
    configuration, models, frozen, run, prompts = _load_frozen_experiment_inputs(root)
    source, excluded = select_frozen_direction(root, "gemma")
    random_direction = create_random_directions(
        source, [configuration["random_seed"]], excluded
    )[configuration["random_seed"]]
    conditions = [
        unchanged_model(),
        remove_direction("candidate", source, range(48), REMOVAL_LOCATIONS),
        remove_direction(
            "random_protected", random_direction, range(48), REMOVAL_LOCATIONS
        ),
    ]
    if model_name == "lapa":
        native, _ = select_frozen_direction(root, "lapa")
        conditions.append(
            remove_direction("native_lapa", native, range(48), REMOVAL_LOCATIONS)
        )
    folder = run / "generation" / model_name / f"worker-{worker}"
    test = (
        prompts.query('split == "test"')
        .sort_values(["language", "label", "prompt_id"])
        .iloc[worker::4]
        .reset_index(drop=True)
    )
    sources = json.loads((root / "configs/datasets.json").read_text())
    utility = pd.concat(
        [
            load_multiple_choice_dataset(
                root,
                "belebele",
                sources["belebele"]["eng_Latn"]["revision"],
                "test",
                language,
            )
            for language in ["en", "uk"]
        ],
        ignore_index=True,
    )
    utility = utility.iloc[worker::4].reset_index(drop=True)
    with load_language_model(model_name, **models[model_name], device=device) as model:
        evaluate_multiple_choice(
            model,
            utility,
            conditions,
            "answer_label_log_probability",
            configuration["utility_batch_size"],
            folder / "utility",
        )
        responses = generate_responses(
            model,
            test,
            conditions,
            GreedyDecoding(max_new_tokens=configuration["max_new_tokens"]),
            configuration["generation_batch_size"],
            folder / "responses",
        )
    return responses


def score_frozen_safety(root, worker, device="cuda:0"):
    configuration, models, frozen, run, prompts = _load_frozen_experiment_inputs(root)
    paths = sorted((run / "generation").glob("*/worker-*/responses/responses.jsonl"))
    responses = pd.concat([read_records(path) for path in paths], ignore_index=True)
    expected = len(prompts.query('split == "test"')) * 7
    if len(responses) != expected or responses.response_id.nunique() != expected:
        raise ValueError("Frozen generation is incomplete or repeats answers")
    shard = responses.iloc[worker::4].reset_index(drop=True)
    rubric = json.loads((root / "configs/rubrics/llama_guard_policy.json").read_text())
    with load_response_judge(
        "local",
        "llama_guard_4",
        rubric=rubric,
        device=device,
        **models["llama_guard_4"],
    ) as judge:
        labels = evaluate_responses(
            shard,
            judge,
            configuration["judge_batch_size"],
            run / "judges/llama_guard_4" / f"worker-{worker}",
        )
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["capture", "validate", "generate", "guard"])
    parser.add_argument("--model", choices=["gemma", "lapa"], default="gemma")
    parser.add_argument("--worker", type=int, choices=range(4), required=True)
    arguments = parser.parse_args()
    from .result_storage import find_project_root

    root = find_project_root()
    visible_devices = torch.cuda.device_count()
    if visible_devices not in {1, 4}:
        raise RuntimeError(f"Expected one or four visible GPUs, got {visible_devices}")
    device_index = 0 if visible_devices == 1 else arguments.worker
    torch.cuda.set_device(device_index)
    device = f"cuda:{device_index}"
    print(
        "Worker GPU",
        arguments.worker,
        torch.cuda.get_device_properties(device_index).uuid,
        flush=True,
    )
    if arguments.stage == "capture":
        model, language = [
            ("gemma", "en"),
            ("lapa", "en"),
            ("gemma", "uk"),
            ("lapa", "uk"),
        ][arguments.worker]
        capture_frozen_activations(root, model, language, device)
    elif arguments.stage == "validate":
        validate_frozen_direction_layers(
            root, arguments.model, arguments.worker, device
        )
    elif arguments.stage == "generate":
        generate_frozen_answers(root, arguments.model, arguments.worker, device)
    else:
        score_frozen_safety(root, arguments.worker, device)
    print("Completed", arguments.stage, arguments.model, arguments.worker, flush=True)


if __name__ == "__main__":
    main()
