"""Multiple-choice next-token scoring, preserving the Belebele baseline method."""

from .interventions import apply_intervention
from .result_storage import (
    append_records,
    condition_settings,
    content_hash,
    exclusive_run,
    finish_run,
    prepare_run,
    read_records,
)


def make_multiple_choice_prompt(row):
    options = []
    for option_number, letter in enumerate("ABCD", 1):
        options.append(f"{letter}. {row[f'mc_answer{option_number}']}")
    return (
        "Read the passage and answer the question. Choose the correct option. "
        "Reply with only A, B, C, or D.\n\n"
        f"Passage:\n{row['flores_passage']}\n\n"
        f"Question: {row['question']}\n" + "\n".join(options)
    )


def evaluate_multiple_choice(
    model, examples, conditions, scoring, batch_size, output_directory
):
    with exclusive_run(output_directory):
        import torch

        if scoring != "answer_label_log_probability":
            raise ValueError("Implemented scoring is answer_label_log_probability")
        if batch_size < 1 or examples.empty or not conditions:
            raise ValueError("Need examples, conditions and positive batch size")
        if examples.duplicated(["language", "example_id"]).any():
            raise ValueError("Repeated utility example IDs")
        if len({condition.name for condition in conditions}) != len(conditions):
            raise ValueError("Repeated condition names")
        labels = list("ABCD")
        tokenized_labels = [
            model.tokenizer.encode(label, add_special_tokens=False) for label in labels
        ]
        if any(len(tokens) != 1 for tokens in tokenized_labels):
            raise ValueError("Label scoring requires single-token A/B/C/D labels")
        label_token_ids = [tokens[0] for tokens in tokenized_labels]
        records = examples.to_dict("records")
        settings = {
            "task": "multiple_choice",
            "model": model.specification,
            "examples_hash": content_hash(records),
            "scoring": scoring,
            "batch_size": batch_size,
            "conditions": [condition_settings(condition) for condition in conditions],
        }
        identities = {}
        for condition in conditions:
            for row in records:
                key = (condition.name, row["language"], row["example_id"])
                identities[key] = content_hash([settings, *key])
        path, saved = prepare_run(
            output_directory, settings, "utility_predictions.jsonl", identities.values()
        )
        completed = set(saved.record_id) if not saved.empty else set()
        for condition in conditions:
            for start in range(0, len(records), batch_size):
                batch = []
                for row in records[start : start + batch_size]:
                    record_id = identities[
                        (condition.name, row["language"], row["example_id"])
                    ]
                    if record_id not in completed:
                        batch.append(row)
                if not batch:
                    continue
                prompts = [make_multiple_choice_prompt(row) for row in batch]
                encoded = model.encode_prompts(prompts)
                with apply_intervention(model, condition), torch.inference_mode():
                    logits = (
                        model.model(**encoded, use_cache=False, logits_to_keep=1)
                        .logits[:, -1, label_token_ids]
                        .float()
                    )
                if not torch.isfinite(logits).all():
                    raise ValueError("Non-finite choice logits")
                probabilities = logits.softmax(-1).cpu().tolist()
                predictions = logits.argmax(-1).cpu().tolist()
                output = []
                for row, scores, prediction in zip(
                    batch, probabilities, predictions, strict=True
                ):
                    answer_number = int(row["correct_answer_num"])
                    if answer_number not in [1, 2, 3, 4]:
                        raise ValueError("Answer key must be 1, 2, 3 or 4")
                    record_id = identities[
                        (condition.name, row["language"], row["example_id"])
                    ]
                    output.append(
                        {
                            "record_id": record_id,
                            "example_id": row["example_id"],
                            "passage_id": row["passage_id"],
                            "language": row["language"],
                            "model": model.specification["model"],
                            "condition": condition.name,
                            "prediction": labels[prediction],
                            "answer": labels[answer_number - 1],
                            "correct": prediction == answer_number - 1,
                            "label_probabilities": dict(
                                zip(labels, scores, strict=True)
                            ),
                            "scoring": scoring,
                        }
                    )
                append_records(path, output)
        results = read_records(path)
        finish_run(output_directory, results)
        return results
