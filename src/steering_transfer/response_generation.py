"""One response implementation shared by baseline, transfer and persona runs."""

from dataclasses import replace

import pandas as pd

from .interventions import apply_intervention
from .result_storage import (
    append_records,
    condition_settings,
    content_hash,
    exclusive_run,
    finish_run,
    prepare_run,
    read_records,
    settings_dict,
    write_json,
)


def validate_prompts(prompts):
    required = {"prompt_id", "prompt", "language"}
    if not required.issubset(prompts):
        raise ValueError(f"Prompt records need {sorted(required)}")
    if prompts.empty or prompts.prompt_id.duplicated().any():
        raise ValueError("Need nonempty prompts with unique IDs")
    if prompts.prompt.isna().any() or prompts.prompt.str.strip().eq("").any():
        raise ValueError("Prompt text cannot be empty")


def generate_token_batch(model, prompts, decoding):
    import torch

    encoded = model.encode_prompts(prompts)
    options = decoding.generation_options()
    torch.manual_seed(decoding.seed)
    with torch.inference_mode():
        generated = model.model.generate(**encoded, **options, logits_to_keep=1)
    end_tokens = model.model.generation_config.eos_token_id
    if not isinstance(end_tokens, list):
        end_tokens = [end_tokens]
    results = []
    for index, output in enumerate(generated):
        prompt_ids = encoded.input_ids[index][
            encoded.attention_mask[index].bool()
        ].tolist()
        generated_ids = output[encoded.input_ids.shape[1] :].tolist()
        stop_index = len(generated_ids)
        stop_reason = "token_limit"
        for token_index, token_id in enumerate(generated_ids):
            if token_id in end_tokens:
                stop_index = token_index
                stop_reason = "eos"
                break
        response_ids = generated_ids[:stop_index]
        results.append(
            {
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "response": model.tokenizer.decode(
                    response_ids, skip_special_tokens=True
                ),
                "output_tokens": len(response_ids),
                "stop_reason": stop_reason,
            }
        )
    return results


def generate_responses(
    model,
    prompts,
    conditions,
    decoding,
    batch_size,
    output_directory,
    reuse_responses=None,
):
    with exclusive_run(output_directory):
        validate_prompts(prompts)
        if batch_size < 1 or not conditions:
            raise ValueError("Need conditions and positive batch size")
        names = [condition.name for condition in conditions]
        if len(set(names)) != len(names):
            raise ValueError("Condition names must be unique")
        input_records = prompts.to_dict("records")
        settings = {
            "task": "response_generation",
            "model": model.specification,
            "prompts_hash": content_hash(input_records),
            "decoding": settings_dict(decoding),
            "batch_size": batch_size,
            "batch_seed_policy": "seed_plus_batch_index",
            "conditions": [condition_settings(c) for c in conditions],
        }
        identities = {}
        for condition in conditions:
            for start in range(0, len(input_records), batch_size):
                batch = input_records[start : start + batch_size]
                batch_settings = {
                    "model": model.specification,
                    "condition": condition_settings(condition),
                    "decoding": settings_dict(
                        replace(decoding, seed=decoding.seed + start // batch_size)
                    ),
                    "batch": batch,
                }
                for row in batch:
                    identities[(condition.name, row["prompt_id"])] = content_hash(
                        {
                            "batch_settings": batch_settings,
                            "prompt_id": row["prompt_id"],
                        }
                    )
        path, saved = prepare_run(
            output_directory, settings, "responses.jsonl", identities.values()
        )
        completed = set(saved.record_id) if not saved.empty else set()
        if reuse_responses is not None:
            if reuse_responses.response_id.duplicated().any():
                raise ValueError("Reusable response identities are repeated")
            reusable = reuse_responses[
                reuse_responses.response_id.isin(identities.values())
            ]
            for row in reusable.to_dict("records"):
                if row["response_hash"] != content_hash(row["response"]):
                    raise ValueError("Reusable response content hash differs")
            missing = reusable.loc[~reusable.response_id.isin(completed)]
            append_records(path, missing.to_dict("records"))
            completed.update(missing.response_id)
        replay_checked = False
        for condition in conditions:
            for start in range(0, len(input_records), batch_size):
                batch = input_records[start : start + batch_size]
                record_ids = [
                    identities[(condition.name, row["prompt_id"])] for row in batch
                ]
                done = [record_id in completed for record_id in record_ids]
                if all(done):
                    continue
                if any(done):
                    raise ValueError("Resume requires complete original batches")
                batch_decoding = replace(
                    decoding, seed=decoding.seed + start // batch_size
                )
                with apply_intervention(model, condition):
                    generated = generate_token_batch(
                        model, [row["prompt"] for row in batch], batch_decoding
                    )
                    if (
                        not replay_checked
                        and not decoding.generation_options()["do_sample"]
                    ):
                        replay = generate_token_batch(
                            model, [row["prompt"] for row in batch], batch_decoding
                        )
                        if [r["response_ids"] for r in generated] != [
                            r["response_ids"] for r in replay
                        ]:
                            raise RuntimeError("Greedy batch replay differs")
                        write_json(
                            path.parent / "greedy_replay.json", {"matched": True}
                        )
                        replay_checked = True
                records = []
                for source, answer, record_id in zip(
                    batch, generated, record_ids, strict=True
                ):
                    records.append(
                        {
                            **source,
                            **answer,
                            "model": model.specification["model"],
                            "condition": condition.name,
                            "model_revision": model.specification["revision"],
                            "record_id": record_id,
                            "response_id": record_id,
                            "response_hash": content_hash(answer["response"]),
                        }
                    )
                append_records(path, records)
                completed.update(record_ids)
        responses = read_records(path) if path.exists() else pd.DataFrame()
        finish_run(output_directory, responses)
        return responses
