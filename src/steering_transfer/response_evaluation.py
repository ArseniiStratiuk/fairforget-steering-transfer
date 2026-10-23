"""Evaluate saved answers without coupling judges to response generation."""

import pandas as pd

from .result_storage import (
    append_records,
    content_hash,
    exclusive_run,
    finish_run,
    prepare_run,
    read_records,
)


def load_response_judge(
    provider, model, revision, rubric, checkpoint=None, device="cuda:0"
):
    from pathlib import Path

    from .result_storage import find_project_root

    if provider == "typesafe" and model == "jev":
        from .judges.jev import JevJudge

        return JevJudge(revision, rubric)
    if provider != "local":
        raise ValueError(f"Unsupported judge provider: {provider}")
    if checkpoint is None:
        raise ValueError("Local judges require a checkpoint")
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = find_project_root() / checkpoint
    if model == "gemma":
        from .judges.gemma import GemmaJudge
        from .language_models import load_language_model

        loaded = load_language_model(model, revision, checkpoint, device=device)
        return GemmaJudge(loaded, rubric)
    if model == "llama_guard_4":
        if rubric["name"] != "llama_guard_policy" or set(rubric["labels"]) != {
            "safe",
            "unsafe",
        }:
            raise ValueError("Llama Guard supports its native safety rubric only")
        import torch
        from transformers import (
            AutoConfig,
            AutoTokenizer,
            Llama4ForConditionalGeneration,
        )

        from .judges.llama_guard import LlamaGuardJudge

        if Path(checkpoint).name != revision:
            raise ValueError("Guard checkpoint path must end with its pinned revision")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        configuration = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
        configuration.text_config.attention_chunk_size = 8192
        loaded = Llama4ForConditionalGeneration.from_pretrained(
            checkpoint,
            config=configuration,
            dtype=torch.bfloat16,
            device_map={"": device},
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        specification = {
            "model": model,
            "revision": revision,
            "rubric": rubric,
            "precision": "bfloat16",
            "attention_chunk_size": 8192,
            "template_hash": content_hash(tokenizer.chat_template),
        }
        return LlamaGuardJudge(loaded, tokenizer, specification, rubric)
    raise ValueError(f"Unsupported judge: {model}")


def evaluate_responses(
    responses, judge, batch_size, output_directory, request_column="prompt"
):
    with exclusive_run(output_directory):
        if (
            responses.empty
            or responses.response_id.duplicated().any()
            or batch_size < 1
        ):
            raise ValueError("Need distinct saved responses and positive batch size")
        records = responses.to_dict("records")
        if request_column not in responses:
            raise ValueError(f"Missing evaluation request column: {request_column}")
        for record in records:
            record["prompt"] = record[request_column]
        settings = {
            "task": "response_evaluation",
            "judge": judge.specification,
            "answers": [
                {
                    "response_id": row["response_id"],
                    "hash": content_hash(row["response"]),
                }
                for row in records
            ],
            "batch_size": batch_size,
            "request_column": request_column,
        }
        expected = {
            row["response_id"]: content_hash([settings, row["response_id"]])
            for row in records
        }
        path, saved = prepare_run(
            output_directory, settings, "labels.jsonl", expected.values()
        )
        completed = set(saved.record_id) if not saved.empty else set()
        replay_checked = False
        for start in range(0, len(records), batch_size):
            batch = [
                row
                for row in records[start : start + batch_size]
                if expected[row["response_id"]] not in completed
            ]
            if not batch:
                continue
            try:
                judgments = judge.judge_batch(batch)
            except ValueError as error:
                append_records(
                    path.parent / "failures.jsonl",
                    [
                        {"response_id": row["response_id"], "error": str(error)}
                        for row in batch
                    ],
                )
                raise
            if len(judgments) != len(batch):
                raise ValueError("Judge returned the wrong number of labels")
            if not replay_checked and getattr(judge, "requires_greedy_replay", True):
                if judge.judge_batch(batch) != judgments:
                    raise RuntimeError("Judge greedy replay differs")
                replay_checked = True
            output = []
            for row, judgment in zip(batch, judgments, strict=True):
                output.append(
                    {
                        **judgment,
                        "record_id": expected[row["response_id"]],
                        "response_id": row["response_id"],
                        "response_hash": content_hash(row["response"]),
                        "rubric_id": judge.rubric["name"],
                        "rubric_revision": content_hash(judge.rubric),
                    }
                )
            append_records(path, output)
        labels = read_records(path) if path.exists() else pd.DataFrame()
        finish_run(output_directory, labels)
        return labels


def evaluate_response_rubrics(
    responses,
    provider,
    model,
    revision,
    rubrics,
    batch_size,
    output_directory,
    request_column="prompt",
    checkpoint=None,
):
    from pathlib import Path

    results = {}
    for name, rubric in rubrics.items():
        with load_response_judge(
            provider, model, revision, rubric, checkpoint
        ) as judge:
            results[name] = evaluate_responses(
                responses,
                judge,
                batch_size,
                Path(output_directory) / name,
                request_column=request_column,
            )
    return results
