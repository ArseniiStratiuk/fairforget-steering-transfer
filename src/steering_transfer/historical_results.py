"""Read old artifacts without rewriting them or inventing missing provenance."""

import json
from pathlib import Path

import pandas as pd

from .result_storage import content_hash, file_hash, read_records


def load_recorded_inputs(root, kinds=None):
    configuration = json.loads(
        (Path(root) / "configs/recorded_results.json").read_text()
    )
    if kinds is not None:
        configuration["files"] = [
            item for item in configuration["files"] if item["kind"] in kinds
        ]
    for item in configuration["files"]:
        path = Path(root) / item["path"]
        if not path.exists():
            raise FileNotFoundError(
                f"Recorded artifact unavailable: {path}; restore it from its recorded origin"
            )
        if file_hash(path) != item["sha256"]:
            raise ValueError(f"Recorded artifact changed: {path}")
    return configuration


def response_identity(record):
    return content_hash(
        [
            record["model"],
            record["condition"],
            record["benchmark"],
            record["label"],
            record["language"],
            str(record["source_id"]),
        ]
    )


def normalize_responses(frame):
    responses = frame.copy()
    responses["source_id"] = responses.source_id.astype(str)
    response_ids, prompt_ids, semantic_ids = [], [], []
    for record in responses.to_dict("records"):
        semantic_id = f"{record['benchmark']}:{record['label']}:{record['source_id']}"
        semantic_ids.append(semantic_id)
        prompt_ids.append(f"{semantic_id}:{record['language']}")
        response_ids.append(response_identity(record))
    responses["response_id"] = response_ids
    responses["prompt_id"] = prompt_ids
    responses["semantic_request_id"] = semantic_ids
    responses["response_hash"] = responses.response.map(content_hash)
    return responses


def load_refusal_results(root):
    configuration = load_recorded_inputs(
        root, {"responses", "guard_labels", "utility", "judge_metadata"}
    )
    responses, evaluations, utility = [], [], []
    for item in configuration["files"]:
        path = Path(root) / item["path"]
        if item["kind"] == "responses":
            responses.append(normalize_responses(read_records(path)))
        elif item["kind"] == "guard_labels":
            frame = read_records(path)
            frame["response_id"] = [
                response_identity(row) for row in frame.to_dict("records")
            ]
            frame["label"] = frame.verdict
            frame["status"] = "success"
            frame.loc[~frame.verdict.isin(["safe", "unsafe"]), "status"] = (
                "invalid_label"
            )
            frame["rubric_revision"] = configuration["guard_rubric_identity"]
            evaluations.append(
                frame[["response_id", "label", "status", "rubric_revision"]]
            )
        elif item["kind"] == "utility":
            frame = read_records(path)
            frame["model"] = item["model"]
            frame["passage_id"] = frame.link
            frame["example_id"] = [
                content_hash([link, int(number)])
                for link, number in zip(frame.link, frame.question_number, strict=True)
            ]
            frame["language"] = frame.language.map({"eng_Latn": "en", "ukr_Cyrl": "uk"})
            utility.append(frame)
    responses = pd.concat(responses, ignore_index=True)
    evaluations = pd.concat(evaluations, ignore_index=True)
    if (
        responses.response_id.duplicated().any()
        or evaluations.response_id.duplicated().any()
    ):
        raise ValueError("Historical response or evaluation identities collide")
    # Historical judge files reference the full response file through their metadata.
    for item in configuration["files"]:
        if item["kind"] == "judge_metadata":
            metadata = json.loads((Path(root) / item["path"]).read_text())
            input_digest = file_hash(Path(root) / item["answers_path"])
            if metadata["answers_sha256"] != input_digest:
                raise ValueError(
                    "Historical judge metadata references another answer file"
                )
    evaluations = evaluations.merge(
        responses[["response_id", "response_hash"]],
        on="response_id",
        validate="one_to_one",
    )
    return {
        "responses": responses,
        "evaluations": evaluations,
        "utility": pd.concat(utility, ignore_index=True),
        "configuration": configuration,
    }


def load_geometry(root):
    configuration = load_recorded_inputs(root, {"geometry"})
    item = next(item for item in configuration["files"] if item["kind"] == "geometry")
    return pd.read_csv(Path(root) / item["path"])


def load_recorded_direction(root, name):
    import torch

    from .records import Direction

    configuration = load_recorded_inputs(root, {"source_direction"})
    item = next(
        item for item in configuration["files"] if item["kind"] == "source_direction"
    )
    bundle = torch.load(
        Path(root) / item["path"], map_location="cpu", weights_only=True
    )
    if name == "candidate":
        values = bundle["candidate_direction"].float()
    elif name == "random_protected":
        values = (
            bundle["random_direction"]
            .float()
            .masked_fill(bundle["outlier_mask"][32], 0)
        )
        values /= values.norm()
    else:
        raise ValueError("Recorded directions are candidate and random_protected")
    return Direction(
        values,
        {
            "source_file_sha256": item["sha256"],
            "source_layer": 32,
            "excluded_coordinates": bundle["outlier_mask"][32]
            .nonzero()
            .flatten()
            .tolist(),
            "normalization": "unit",
            "source_model": "gemma",
        },
    )


def load_recorded_capture(path, metadata_path, prompts):
    import torch

    metadata = json.loads(Path(metadata_path).read_text())
    if [str(value) for value in metadata["source_ids"]] != prompts.source_id.astype(
        str
    ).tolist():
        raise ValueError("Recorded capture source IDs differ")
    if metadata["labels"] != prompts.label.tolist():
        raise ValueError("Recorded capture contrast labels differ")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    values = bundle["activations"]
    if len(values) != len(prompts):
        raise ValueError("Recorded capture shape differs")
    return {
        "activations": values,
        "record_ids": prompts.prompt_id.tolist(),
        "layers": list(range(values.shape[1])),
        "provenance": {"file_sha256": file_hash(path), "metadata": metadata},
    }


def restore_recorded_inputs(root, kinds, source_root=None, download_published=False):
    """Restore exact archived bytes; unreleased files require an archive directory."""
    import shutil

    configuration = json.loads(
        (Path(root) / "configs/recorded_results.json").read_text()
    )
    releases = json.loads(
        (Path(root) / "configs/reviewed_experiments.json").read_text()
    )
    published = {}
    for release in releases.values():
        for item in release["files"]:
            published[item["local_path"]] = (release, item)
    missing = []
    for item in configuration["files"]:
        if item["kind"] not in kinds:
            continue
        destination = Path(root) / item["path"]
        if destination.exists():
            if file_hash(destination) != item["sha256"]:
                raise ValueError(f"Existing artifact checksum mismatch: {destination}")
            continue
        source = Path(source_root) / item["path"] if source_root is not None else None
        if source is None or not source.exists():
            if download_published and item["path"] in published:
                from huggingface_hub import hf_hub_download

                release, released_item = published[item["path"]]
                source = Path(
                    hf_hub_download(
                        release["repo_id"],
                        released_item["filename"],
                        repo_type=release["repo_type"],
                        revision=release["revision"],
                    )
                )
            else:
                missing.append(item["path"])
                continue
        if file_hash(source) != item["sha256"]:
            raise ValueError(f"Archive artifact checksum mismatch: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return missing


def load_recorded_judgments(root, kind, judge):
    configuration = load_recorded_inputs(root, {kind})
    paths = [
        Path(root) / item["path"]
        for item in configuration["files"]
        if item.get("judge") == judge
    ]
    if not paths:
        raise FileNotFoundError(f"No recorded {kind} from {judge}")
    judgments = pd.concat([read_records(path) for path in paths], ignore_index=True)
    if judgments.response_id.duplicated().any():
        raise ValueError("Repeated recorded judgments")
    return judgments


def normalize_persona_collection(records, model_revision):
    responses = records.copy()
    identities = []
    for row in responses.to_dict("records"):
        identity = content_hash(
            ["persona-collection", row["pair_id"], row["persona"], row["answer"]]
        )
        identities.append(identity)
    responses["response_id"] = identities
    responses["prompt_id"] = responses.pair_id + "/" + responses.persona
    responses["response"] = responses.answer
    responses["model_revision"] = model_revision
    responses["stop_reason"] = responses.stop.map({"cap": "token_limit", "eos": "eos"})
    return responses


def load_persona_results(root):
    kinds = {
        "persona_collection",
        "persona_fit",
        "persona_direction",
        "persona_geometry",
        "persona_responses",
        "persona_collection_behavior",
        "persona_collection_coherence",
        "persona_steering_behavior",
        "persona_steering_coherence",
    }
    configuration = load_recorded_inputs(root, kinds)
    files = configuration["files"]

    def paths(kind):
        return [Path(root) / item["path"] for item in files if item["kind"] == kind]

    def table(kind):
        selected = paths(kind)
        if not selected:
            raise FileNotFoundError(f"No recorded {kind}")
        return pd.concat([read_records(path) for path in selected], ignore_index=True)

    protocol = json.loads(
        (Path(root) / "configs/experiments/persona_steering.json").read_text()
    )
    collection = normalize_persona_collection(
        table("persona_collection"), protocol["collection_model"]["revision"]
    )
    responses = table("persona_responses")
    if (
        collection.response_id.duplicated().any()
        or responses.response_id.duplicated().any()
    ):
        raise ValueError("Repeated recorded persona responses")
    return {
        "collection": collection,
        "collection_behavior": table("persona_collection_behavior"),
        "collection_coherence": table("persona_collection_coherence"),
        "fit": json.loads(paths("persona_fit")[0].read_text()),
        "direction_paths": {
            item["method"]: Path(root) / item["path"]
            for item in files
            if item["kind"] == "persona_direction"
        },
        "geometry": pd.read_csv(paths("persona_geometry")[0]),
        "responses": responses,
        "behavior": table("persona_steering_behavior"),
        "coherence": table("persona_steering_coherence"),
    }
