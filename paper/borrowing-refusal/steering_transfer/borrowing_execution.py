"""Validate refined source settings while reusing exact matching response records."""

import argparse
import json
from pathlib import Path

import pandas as pd

from .borrowing_experiment import (
    evaluate_experiment_answers,
    select_coherent_strength,
    validate_source_interventions,
)
from .datasets import load_frozen_prompts
from .judges.jev import JevJudge
from .language_models import load_language_model
from .queued_judging import QueuedJudge
from .records import Direction
from .result_storage import content_hash, file_hash, read_records, write_json


def load_refined_source_selection(folder, direction, configuration, validation):
    """Require completed approved trials and bind selection to its actual inputs."""
    folder = Path(folder)
    trials = pd.DataFrame(json.loads((folder / "refinement_trials.json").read_text()))
    expected_strengths = [0.5, 0.625, 0.75, 0.875, 1.0]
    if sorted(trials.multiplier.tolist()) != expected_strengths:
        raise ValueError("Refinement has not completed all five approved strengths")
    saved = json.loads((folder / "refinement_selection.json").read_text())
    expected = {
        "validation_hash": content_hash(validation.to_dict("records")),
        "direction_hash": content_hash(direction.values.tolist()),
        "original_configuration_hash": content_hash(configuration),
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Refinement inputs differ from the current source settings")
    chosen = select_coherent_strength(trials)
    if chosen != saved["selected"]:
        raise ValueError("Saved refinement selection differs from its measurements")
    return {
        **chosen,
        "strength": chosen["multiplier"] * direction.provenance["raw_norm"],
        "layer": direction.provenance["layer"],
        "configuration_hash": content_hash(configuration),
        "direction_hash": expected["direction_hash"],
        "validation_hash": expected["validation_hash"],
        "selection_rule": saved["rule"],
        "trials_hash": content_hash(trials.to_dict("records")),
    }


def validate_refined_source(root, model_name, language):
    import torch

    root = Path(root)
    run = root / "results/final/sibling-refusal-borrowing-v1"
    folder = run / model_name / language
    configuration = json.loads(
        (root / "configs/experiments/borrowing_refusal.json").read_text()
    )
    prompts = load_frozen_prompts(root)
    validation = prompts.query(
        'split == "validation" and label == "harmless" and language == @language'
    ).copy()
    saved = torch.load(folder / "selected_direction.pt", weights_only=True)
    direction = Direction(saved["values"], saved["provenance"])
    selection = load_refined_source_selection(
        folder, direction, configuration, validation
    )
    multiplier = selection["multiplier"]
    paths = [
        folder / "source_validation" / operation / "responses/responses.jsonl"
        for operation in ["addition", "removal"]
    ]
    paths.append(
        folder / "calibration" / f"multiplier-{multiplier:g}/responses/responses.jsonl"
    )
    reusable = pd.concat([read_records(path) for path in paths], ignore_index=True)
    if reusable.response_id.duplicated().any():
        raise ValueError("Reusable source responses contain duplicate identities")
    pin = configuration["judges"]["jev"]["selected_configuration"]
    judge_path = root / pin["path"]
    if file_hash(judge_path) != pin["sha256"]:
        raise ValueError("Selected judge configuration checksum differs")
    judge_settings = json.loads(judge_path.read_text())
    specification = configuration["models"][model_name]
    with load_language_model(
        model_name, specification["revision"], specification["checkpoint"]
    ) as model:
        if model.specification != direction.provenance["model"]:
            raise ValueError("Refined validation model differs from extraction")
        with JevJudge(
            judge_settings["model"], judge_settings["rubrics"]["behavior"]
        ) as api:
            judge = QueuedJudge(api, run / "judge_queue")
            summary = validate_source_interventions(
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
                addition_candidate_name=f"addition_{multiplier:g}",
            )
    write_json(folder / "refined_source_selection.json", selection)
    return summary


def summarize_saved_strengths(folder, judge_configuration):
    """Count coherent refusals only from complete, identity-matched saved judgments."""
    from .joint_judging import validate_labels

    trials = []
    for trial in sorted(Path(folder).glob("multiplier-*")):
        responses = read_records(trial / "responses/responses.jsonl")
        if len(responses) != 64 or not responses.response_id.is_unique:
            raise ValueError("Saved calibration trial requires 64 distinct answers")
        coherent_refusals = 0
        incoherent = 0
        for record in responses.to_dict("records"):
            if record["response_hash"] != content_hash(record["response"]):
                raise ValueError("Saved calibration answer checksum differs")
            path = (
                trial
                / "judgments/predictions/jev/experiment/zero-shot"
                / f"{record['response_id']}.json"
            )
            decision = json.loads(path.read_text())
            identity = content_hash(
                {
                    "record": record,
                    "rubrics": judge_configuration["rubrics"],
                    "format": "joint",
                }
            )
            if decision["input_hash"] != identity:
                raise ValueError(
                    "Saved calibration judgment belongs to different inputs"
                )
            if decision["specification"]["model"] != judge_configuration["model"]:
                raise ValueError("Saved calibration judgment used a different model")
            labels = {row["dimension"]: row["label"] for row in decision["predictions"]}
            validate_labels(labels, judge_configuration["rubrics"])
            coherent = labels["coherence"] == "coherent"
            coherent_refusals += coherent and labels["behavior"] == "refusal"
            incoherent += not coherent
        trials.append(
            {
                "multiplier": float(trial.name.removeprefix("multiplier-")),
                "coherent_refusals": coherent_refusals,
                "incoherent": incoherent,
                "answers": len(responses),
            }
        )
    return pd.DataFrame(trials)


def prepare_borrowed_conditions(root, donor_name, language):
    """Build frozen donor conditions only from completed refined source validation."""
    import torch

    from .borrowing_experiment import (
        build_off_target_direction,
        build_recipient_conditions,
    )

    root = Path(root)
    run = root / "results/final/sibling-refusal-borrowing-v1"
    folder = run / donor_name / language
    configuration = json.loads(
        (root / "configs/experiments/borrowing_refusal.json").read_text()
    )
    if donor_name not in configuration["donors"]:
        raise ValueError("Transfer donor must be one of the configured source models")
    source_summary = pd.read_csv(
        folder / "refined_source_validation/source_validation.csv"
    )
    if len(source_summary) != 6 or not source_summary.answers.eq(64).all():
        raise ValueError("Transfer requires complete matched source validation")
    saved = torch.load(folder / "selected_direction.pt", weights_only=True)
    direction = Direction(saved["values"], saved["provenance"])
    prompts = load_frozen_prompts(root)
    validation = prompts.query(
        'split == "validation" and label == "harmless" and language == @language'
    ).copy()
    selection = load_refined_source_selection(
        folder, direction, configuration, validation
    )
    recorded_selection = json.loads(
        (folder / "refined_source_selection.json").read_text()
    )
    if recorded_selection != selection:
        raise ValueError(
            "Transfer selection differs from the source validation settings"
        )
    expected_conditions = {
        ("addition", "baseline"),
        ("addition", f"addition_{selection['multiplier']:g}"),
        ("addition", "random"),
        ("removal", "baseline"),
        ("removal", "candidate"),
        ("removal", "random"),
    }
    measured_conditions = set(zip(source_summary.operation, source_summary.condition))
    if measured_conditions != expected_conditions:
        raise ValueError("Source validation lacks a required matched condition")
    means = [
        torch.load(
            run / donor_name / source_language / "directions.pt", weights_only=True
        )
        for source_language in ["en", "uk"]
    ]
    off_target = build_off_target_direction(
        means[0],
        means[1],
        selection["layer"],
        direction.provenance["excluded_coordinates"],
    )
    conditions = build_recipient_conditions(
        direction, off_target, selection, configuration, f"{donor_name}_{language}"
    )
    return [condition for condition in conditions if condition.method != "unchanged"]


def run_borrowed_transfer(root, donor_name, language, worker, workers=4):
    conditions = prepare_borrowed_conditions(root, donor_name, language)
    return run_transfer_conditions(
        root, conditions, f"{donor_name}_{language}", worker, workers
    )


def run_native_transfer(root, language, selection_path, worker, workers=4):
    """Run each native comparator once, shared by every donor comparison."""
    import torch

    from .borrowing_experiment import build_native_conditions

    root = Path(root)
    configuration = json.loads(
        (root / "configs/experiments/borrowing_refusal.json").read_text()
    )
    recipient = configuration["recipient"]
    folder = root / "results/final/sibling-refusal-borrowing-v1" / recipient / language
    saved = torch.load(folder / "selected_direction.pt", weights_only=True)
    direction = Direction(saved["values"], saved["provenance"])
    selection = json.loads(Path(selection_path).read_text())
    if selection.get("language") != language:
        raise ValueError(
            "Native selection belongs to a different construction language"
        )
    conditions = build_native_conditions(direction, selection, configuration, language)
    return run_transfer_conditions(
        root, conditions, f"native_{language}", worker, workers
    )


def partition_saved_conditions(root, conditions, artifact_kind):
    """Separate new work from verified artifacts without rewriting saved record IDs."""
    from .result_storage import condition_settings

    if artifact_kind not in {"responses", "utility"}:
        raise ValueError("Reuse supports responses or utility predictions")
    root = Path(root)
    run = root / "results/final/sibling-refusal-borrowing-v1"
    registry_path = run / "verified_reuse.json"
    if not registry_path.exists():
        return conditions, []
    registry = json.loads(registry_path.read_text())
    configuration = json.loads(
        (root / "configs/experiments/borrowing_refusal.json").read_text()
    )
    if registry["configuration_hash"] != content_hash(configuration):
        raise ValueError("Reuse registry belongs to a different frozen configuration")
    pending = []
    references = []
    for condition in conditions:
        requested = condition_settings(condition)
        requested.pop("name")
        matches = []
        for artifact in registry["artifacts"]:
            recorded = dict(artifact["condition"])
            recorded.pop("name")
            if artifact["kind"] == artifact_kind and recorded == requested:
                matches.append(artifact)
        if len(matches) > 1:
            raise ValueError(
                "More than one saved artifact matches the requested intervention"
            )
        if not matches:
            pending.append(condition)
            continue
        artifact = matches[0]
        if file_hash(root / artifact["path"]) != artifact["sha256"]:
            raise ValueError("Verified reused artifact was modified")
        references.append({**artifact, "requested_condition": condition.name})
    return pending, references


def run_transfer_conditions(root, conditions, run_name, worker, workers=4):
    """Run new donor interventions; existing unmodified answers stay in their release."""
    from .records import GreedyDecoding
    from .response_generation import generate_responses
    from .utility_evaluation import evaluate_multiple_choice
    from .datasets import load_multiple_choice_dataset

    if workers < 1 or not 0 <= worker < workers:
        raise ValueError("Worker must belong to the requested shard count")
    root = Path(root)
    configuration = json.loads(
        (root / "configs/experiments/borrowing_refusal.json").read_text()
    )
    test = load_frozen_prompts(root).query('split == "test"').copy()
    test = test.sort_values(["language", "label", "prompt_id"])
    test = test.iloc[worker::workers].reset_index(drop=True)
    sources = json.loads((root / "configs/datasets.json").read_text())
    if sources["belebele"] != configuration["data"]["belebele"]:
        raise ValueError(
            "Belebele sources differ from the frozen execution configuration"
        )
    utility = pd.concat(
        [
            load_multiple_choice_dataset(
                root,
                "belebele",
                sources["belebele"]["eng_Latn"]["revision"],
                "test",
                evaluation_language,
            )
            for evaluation_language in ["en", "uk"]
        ],
        ignore_index=True,
    )
    utility = utility.iloc[worker::workers].reset_index(drop=True)
    recipient = configuration["recipient"]
    specification = configuration["models"][recipient]
    output = (
        root
        / "results/final/sibling-refusal-borrowing-v1/transfer"
        / run_name
        / f"worker-{worker}"
    )
    response_conditions, reused_responses = partition_saved_conditions(
        root, conditions, "responses"
    )
    utility_conditions, reused_utility = partition_saved_conditions(
        root, conditions, "utility"
    )
    if not response_conditions or not utility_conditions:
        raise ValueError(
            "This condition set is already computed; use its saved artifacts"
        )
    with load_language_model(
        recipient, specification["revision"], specification["checkpoint"]
    ) as model:
        evaluate_multiple_choice(
            model,
            utility,
            utility_conditions,
            "answer_label_log_probability",
            4,
            output / "utility",
        )
        answers = generate_responses(
            model,
            test,
            response_conditions,
            GreedyDecoding(configuration["generation"]["max_new_tokens"]),
            8,
            output / "responses",
        )
    if (
        len(answers) != len(test) * len(response_conditions)
        or not answers.response_id.is_unique
    ):
        raise ValueError("Transfer generation lacks expected distinct answers")
    write_json(
        output / "generation_complete.json",
        {
            "answers": len(answers),
            "total_answers": len(test) * len(conditions),
            "reused_answer_count": len(test) * len(reused_responses),
            "total_utility_predictions": len(utility) * len(conditions),
            "reused_utility_count": len(utility) * len(reused_utility),
            "responses_sha256": file_hash(output / "responses/responses.jsonl"),
            "utility_sha256": file_hash(output / "utility/utility_predictions.jsonl"),
            "configuration_hash": content_hash(configuration),
            "run_name": run_name,
            "reused_responses": reused_responses,
            "reused_utility": reused_utility,
            "worker": worker,
            "workers": workers,
        },
    )
    return answers


def collect_recipient_answers(root, condition):
    """Reuse the frozen recipient answers after verifying prompts and decoding provenance."""
    from .result_storage import condition_settings

    root = Path(root)
    configuration = json.loads(
        (root / "configs/experiments/borrowing_refusal.json").read_text()
    )
    recipient = configuration["recipient"]
    specification = configuration["models"][recipient]
    test = load_frozen_prompts(root).query('split == "test"')
    test = test.sort_values(["language", "label", "prompt_id"])
    frames = []
    for worker in range(4):
        folder = (
            root
            / "results/final/bilingual-refusal-v1/generation"
            / recipient
            / f"worker-{worker}/responses"
        )
        manifest = json.loads((folder / "manifest.json").read_text())
        settings = manifest["settings"]
        expected = test.iloc[worker::4].reset_index(drop=True)
        if settings["prompts_hash"] != content_hash(expected.to_dict("records")):
            raise ValueError(
                "Historical recipient answers used different frozen test prompts"
            )
        if settings["model"]["revision"] != specification["revision"]:
            raise ValueError(
                "Historical recipient answers used a different model revision"
            )
        if settings["decoding"] != {
            "max_new_tokens": configuration["generation"]["max_new_tokens"],
            "seed": 42,
        }:
            raise ValueError("Historical recipient answers decoding differs")
        replay = json.loads((folder / "greedy_replay.json").read_text())
        if replay != {"matched": True}:
            raise ValueError(
                "Historical recipient answers lacks successful greedy replay"
            )
        expected_condition = condition_settings(condition)
        baseline_conditions = [
            condition
            for condition in settings["conditions"]
            if condition["name"] == expected_condition["name"]
        ]
        if baseline_conditions != [expected_condition]:
            raise ValueError(
                "Historical intervention settings differ from the requested condition"
            )
        answers = read_records(folder / "responses.jsonl")
        answers = answers.loc[answers.condition.eq(expected_condition["name"])].copy()
        if set(answers.prompt_id) != set(expected.prompt_id) or len(answers) != len(
            expected
        ):
            raise ValueError(
                "Historical recipient answers has incomplete prompt coverage"
            )
        expected_text = expected.set_index("prompt_id").prompt
        for row in answers.to_dict("records"):
            if row["prompt"] != expected_text[row["prompt_id"]]:
                raise ValueError("Historical recipient answers prompt text differs")
            if row["response_hash"] != content_hash(row["response"]):
                raise ValueError("Historical recipient answers answer checksum differs")
        frames.append(answers)
    baseline = pd.concat(frames, ignore_index=True)
    if baseline.response_id.duplicated().any():
        raise ValueError("Historical recipient answers repeats answer identities")
    return baseline


def collect_recipient_utility(root, condition):
    """Reuse complete Belebele predictions for an identical recorded intervention."""
    from .datasets import load_multiple_choice_dataset
    from .result_storage import condition_settings

    root = Path(root)
    configuration = json.loads(
        (root / "configs/experiments/borrowing_refusal.json").read_text()
    )
    sources = json.loads((root / "configs/datasets.json").read_text())
    if sources["belebele"] != configuration["data"]["belebele"]:
        raise ValueError("Historical utility sources differ from frozen Belebele")
    examples = pd.concat(
        [
            load_multiple_choice_dataset(
                root,
                "belebele",
                sources["belebele"][code]["revision"],
                "test",
                language,
            )
            for language, code in [("en", "eng_Latn"), ("uk", "ukr_Cyrl")]
        ],
        ignore_index=True,
    )
    recipient = configuration["recipient"]
    expected_condition = condition_settings(condition)
    frames = []
    for worker in range(4):
        folder = (
            root
            / "results/final/bilingual-refusal-v1/generation"
            / recipient
            / f"worker-{worker}/utility"
        )
        settings = json.loads((folder / "manifest.json").read_text())["settings"]
        expected = examples.iloc[worker::4].reset_index(drop=True)
        if settings["examples_hash"] != content_hash(expected.to_dict("records")):
            raise ValueError("Historical utility questions differ")
        if (
            settings["model"]["revision"]
            != configuration["models"][recipient]["revision"]
        ):
            raise ValueError("Historical utility checkpoint differs")
        recorded = [
            item for item in settings["conditions"] if item["name"] == condition.name
        ]
        if (
            recorded != [expected_condition]
            or settings["scoring"] != "answer_label_log_probability"
        ):
            raise ValueError("Historical utility intervention or scoring differs")
        records = read_records(folder / "utility_predictions.jsonl")
        records = records.loc[records.condition.eq(condition.name)].copy()
        expected_keys = set(zip(expected.language, expected.example_id))
        actual_keys = set(zip(records.language, records.example_id))
        if len(records) != len(expected) or actual_keys != expected_keys:
            raise ValueError("Historical utility coverage differs")
        if not records.correct.eq(records.prediction.eq(records.answer)).all():
            raise ValueError(
                "Historical utility correctness flags differ from predictions"
            )
        frames.append(records)
    return pd.concat(frames, ignore_index=True)


def collect_recipient_baseline(root):
    from .interventions import unchanged_model

    return collect_recipient_answers(root, unchanged_model())


def score_recipient_baseline(root):
    """Rescore saved recipient answers with the selected joint judge; no generation."""
    root = Path(root)
    configuration = json.loads(
        (root / "configs/experiments/borrowing_refusal.json").read_text()
    )
    answers = collect_recipient_baseline(root)
    pin = configuration["judges"]["jev"]["selected_configuration"]
    path = root / pin["path"]
    if file_hash(path) != pin["sha256"]:
        raise ValueError("Selected judge configuration checksum differs")
    settings = json.loads(path.read_text())
    folder = root / "results/final/sibling-refusal-borrowing-v1/recipient_baseline"
    with JevJudge(settings["model"], settings["rubrics"]["behavior"]) as judge:
        labels = evaluate_experiment_answers(
            answers, judge, settings, folder / "judgments"
        )
    joined = answers.merge(labels, on="response_id", validate="one_to_one")
    joined.to_json(
        folder / "scored_answers.jsonl", orient="records", lines=True, force_ascii=False
    )
    write_json(
        folder / "complete.json",
        {
            "answers": len(answers),
            "response_ids_hash": content_hash(sorted(answers.response_id)),
            "judge_configuration_sha256": pin["sha256"],
        },
    )
    return joined


def score_transfer_answers(root, run_name, worker):
    """Score completed GPU outputs on an internet-connected host without allocating GPUs."""
    root = Path(root)
    configuration = json.loads(
        (root / "configs/experiments/borrowing_refusal.json").read_text()
    )
    folder = (
        root
        / "results/final/sibling-refusal-borrowing-v1/transfer"
        / run_name
        / f"worker-{worker}"
    )
    completed = json.loads((folder / "generation_complete.json").read_text())
    if completed["configuration_hash"] != content_hash(configuration):
        raise ValueError("Completed transfer used a different configuration")
    if completed["responses_sha256"] != file_hash(folder / "responses/responses.jsonl"):
        raise ValueError("Completed transfer answers were modified")
    answers = read_records(folder / "responses/responses.jsonl")
    if len(answers) != completed["answers"] or not answers.response_id.is_unique:
        raise ValueError("Completed transfer answer coverage differs")
    pin = configuration["judges"]["jev"]["selected_configuration"]
    path = root / pin["path"]
    if file_hash(path) != pin["sha256"]:
        raise ValueError("Selected judge configuration checksum differs")
    settings = json.loads(path.read_text())
    with JevJudge(settings["model"], settings["rubrics"]["behavior"]) as judge:
        labels = evaluate_experiment_answers(
            answers, judge, settings, folder / "judgments"
        )
    write_json(
        folder / "scoring_complete.json",
        {
            "answers": len(labels),
            "responses_sha256": completed["responses_sha256"],
            "judge_configuration_sha256": pin["sha256"],
        },
    )
    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["source", "transfer", "native"], default="source"
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", choices=["gemma", "mamay"])
    parser.add_argument("--language", choices=["en", "uk"], required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--worker", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.stage != "native" and arguments.model is None:
        parser.error("--model is required for source validation or borrowed transfer")
    if arguments.stage == "native":
        if arguments.selection is None:
            parser.error("--selection is required for native transfer")
        run_native_transfer(
            arguments.root,
            arguments.language,
            arguments.selection,
            arguments.worker,
            arguments.workers,
        )
    elif arguments.stage == "source":
        print(
            validate_refined_source(arguments.root, arguments.model, arguments.language)
        )
    else:
        run_borrowed_transfer(
            arguments.root,
            arguments.model,
            arguments.language,
            arguments.worker,
            arguments.workers,
        )
