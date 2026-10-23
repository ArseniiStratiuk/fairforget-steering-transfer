"""Pinned dataset sources and explicit task records; no silent filtering."""

import json
import urllib.request
from pathlib import Path

import pandas as pd

from .result_storage import content_hash, file_hash

LANGUAGE_CODES = {"english": "en", "ukrainian": "uk", "en": "en", "uk": "uk"}


def resolve_dataset_file(root, specification):
    local_path = Path(root) / specification["local_path"]
    if not local_path.exists():
        if "repo_id" in specification:
            from huggingface_hub import hf_hub_download

            local_path = Path(
                hf_hub_download(
                    specification["repo_id"],
                    specification["filename"],
                    repo_type="dataset",
                    revision=specification["revision"],
                )
            )
        elif "url" in specification:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(specification["url"], local_path)
        else:
            raise FileNotFoundError(local_path)
    if file_hash(local_path) != specification["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {local_path}")
    return local_path


def load_dataset_sources(root):
    return json.loads((Path(root) / "configs/datasets.json").read_text())


def load_prompt_pairs(root):
    specification = load_dataset_sources(root)["prompt_pairs"]
    frame = pd.read_csv(resolve_dataset_file(root, specification))
    if frame.record_id.duplicated().any():
        raise ValueError("Duplicate source record IDs")
    return frame


def load_prompt_dataset(root, dataset, revision, split, language, subset_ids=None):
    sources = load_dataset_sources(root)
    language = LANGUAGE_CODES[language]
    specification = sources[dataset]
    if specification["revision"] != revision:
        raise ValueError("Requested dataset revision differs from pinned configuration")
    frame = pd.read_csv(resolve_dataset_file(root, specification))
    if dataset == "prompt_pairs":
        column = "goal" if language == "en" else "ukrainian_prompt"
    elif dataset == "jbb_benign":
        column = "goal" if language == "en" else "ukrainian_prompt"
    else:
        raise ValueError(f"Unsupported prompt dataset: {dataset}")
    if subset_ids is not None:
        indexed = frame.set_index(frame.record_id.astype(str), verify_integrity=True)
        requested = [str(record_id) for record_id in subset_ids]
        frame = indexed.loc[requested].reset_index(drop=True)
    records = []
    for row in frame.to_dict("records"):
        source = row.get("source", "JBB").lower()
        semantic_id = f"{source}:{specification['label']}:{row['record_id']}"
        records.append(
            {
                "prompt_id": f"{semantic_id}:{language}",
                "semantic_request_id": semantic_id,
                "source_id": str(row["record_id"]),
                "dataset": dataset,
                "dataset_revision": revision,
                "split": split,
                "language": language,
                "prompt": row[column],
                "label": specification["label"],
            }
        )
    prompts = pd.DataFrame(records)
    if prompts.empty or prompts.prompt_id.duplicated().any():
        raise ValueError("Empty dataset or duplicate prompts")
    return prompts


def load_extraction_prompts(root, split="extraction"):
    configuration = json.loads(
        (Path(root) / "configs/experiments/refusal_transfer.json").read_text()
    )
    frozen = configuration["frozen_split"]
    if split not in {"extraction", "validation"}:
        raise ValueError("Extraction data split must be extraction or validation")
    harmful = load_prompt_pairs(root).set_index("record_id", verify_integrity=True)
    harmless = pd.read_json(
        resolve_dataset_file(root, load_dataset_sources(root)["alpaca"])
    )
    harmful_ids = frozen[f"harmful_{split}_ids"]
    harmless_ids = frozen[f"harmless_{split}_rows"]
    records = []
    for source, ids, frame, column, label in [
        ("advbench", harmful_ids, harmful, "goal", "harmful"),
        ("alpaca", harmless_ids, harmless, "instruction", "harmless"),
    ]:
        selected = frame.loc[ids]
        if source == "advbench" and not selected.source.eq("AdvBench").all():
            raise ValueError("Extraction IDs must come from AdvBench")
        if source == "alpaca" and not selected.input.str.strip().eq("").all():
            raise ValueError("Frozen Alpaca examples require empty inputs")
        for source_id, row in selected.iterrows():
            records.append(
                {
                    "prompt_id": f"{source}:{source_id}:en",
                    "source_id": str(source_id),
                    "semantic_request_id": f"{source}:{source_id}",
                    "dataset": source,
                    "dataset_revision": frozen["prompt_pairs"]["revision"]
                    if source == "advbench"
                    else frozen["alpaca"]["revision"],
                    "split": split,
                    "language": "en",
                    "label": label,
                    "prompt": row[column],
                }
            )
    return pd.DataFrame(records)


def load_multiple_choice_dataset(root, dataset, revision, split, language):
    if dataset != "belebele" or split != "test":
        raise ValueError("Only the reviewed Belebele test loader is implemented")
    code = {
        "english": "eng_Latn",
        "ukrainian": "ukr_Cyrl",
        "en": "eng_Latn",
        "uk": "ukr_Cyrl",
    }[language]
    specification = load_dataset_sources(root)["belebele"][code]
    if specification["revision"] != revision:
        raise ValueError("Belebele revision mismatch")
    frame = pd.read_json(resolve_dataset_file(root, specification), lines=True)
    if frame.duplicated(["link", "question_number"]).any():
        raise ValueError("Duplicate Belebele question IDs")
    frame = frame.sort_values(["link", "question_number"]).reset_index(drop=True)
    frame["passage_id"] = frame.link
    frame["example_id"] = [
        content_hash([link, int(number)])
        for link, number in zip(frame.link, frame.question_number, strict=True)
    ]
    frame["dataset"] = dataset
    frame["dataset_revision"] = revision
    frame["split"] = split
    frame["language"] = "en" if code == "eng_Latn" else "uk"
    return frame
