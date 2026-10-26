"""Load pinned frozen-experiment artifacts and verify their scientific coverage."""

import json
from pathlib import Path

from .datasets import load_frozen_prompts, resolve_dataset_file
from .direction_processing import select_direction
from .result_storage import read_records


def load_frozen_direction_set(root, model):
    import torch

    release = json.loads((Path(root) / "configs/frozen_results.json").read_text())
    specification = release["directions"][model]
    path = resolve_dataset_file(root, specification)
    return torch.load(path, map_location="cpu", weights_only=True)


def load_frozen_direction(root, model):
    release = json.loads((Path(root) / "configs/frozen_results.json").read_text())
    directions = load_frozen_direction_set(root, model)
    selection = release["selections"][model]
    return select_direction(
        directions, selection["layer"], "unit", selection["excluded_coordinates"]
    )


def load_frozen_results(root):
    release = json.loads((Path(root) / "configs/frozen_results.json").read_text())
    frozen = json.loads((Path(root) / "configs/frozen_data.json").read_text())
    if release["data_revision"] != frozen["revision"]:
        raise ValueError("Frozen results reference a different dataset revision")
    results = {}
    for name, specification in release["files"].items():
        results[name] = read_records(resolve_dataset_file(root, specification))
    answers = results["responses"]
    choices = results["utility"]
    prompts = load_frozen_prompts(root, split="test")
    if len(answers) != len(prompts) * 7 or answers.response_id.duplicated().any():
        raise ValueError("Wrong frozen answer count or repeated answer identities")
    for (model, condition), frame in answers.groupby(["model", "condition"]):
        if set(frame.prompt_id) != set(prompts.prompt_id):
            raise ValueError(f"Incomplete frozen prompt coverage: {model}/{condition}")
        expected = prompts.set_index("prompt_id").prompt
        actual = frame.set_index("prompt_id").prompt
        if not actual.sort_index().equals(expected.sort_index()):
            raise ValueError("Saved answers reference different frozen prompt text")
    expected_conditions = {
        "gemma": {"baseline", "candidate", "random_protected"},
        "lapa": {"baseline", "candidate", "random_protected", "native_lapa"},
    }
    if set(answers.model) != set(expected_conditions):
        raise ValueError("Frozen answers contain unexpected models")
    for model, conditions in expected_conditions.items():
        if set(answers.loc[answers.model.eq(model), "condition"]) != conditions:
            raise ValueError(f"Wrong frozen conditions for {model}")
    if len(choices) != 1800 * 7:
        raise ValueError("Wrong frozen utility prediction count")
    if choices.duplicated(["model", "condition", "language", "example_id"]).any():
        raise ValueError("Repeated utility question within a frozen condition")
    if not choices.groupby(["model", "condition", "language"]).size().eq(900).all():
        raise ValueError("Every utility condition requires 900 questions per language")
    for name in ["evaluations", "behavior"]:
        labels = results[name]
        if labels.response_id.duplicated().any():
            raise ValueError(f"Repeated frozen judgments: {name}")
        if set(labels.response_id) != set(answers.response_id):
            raise ValueError(f"Incomplete frozen judgment coverage: {name}")
        if not labels.status.eq("success").all():
            raise ValueError(f"Unsuccessful frozen judgments: {name}")
    return results
