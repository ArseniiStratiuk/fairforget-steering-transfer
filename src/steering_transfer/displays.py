"""Small reusable views; sampling rules stay explicit at the call site."""

from .result_storage import content_hash


def show_response_examples(
    comparison,
    selection,
    sample_size,
    random_seed,
    include_full_answers,
    public_excerpts=None,
):
    examples = comparison.examples.copy()
    if selection == "baseline_safe_intervention_unsafe":
        examples = examples[examples.difference.eq(1)]
    elif selection == "changed_label":
        examples = examples[examples.difference.ne(0)]
    elif selection != "random":
        raise ValueError("Unknown example selection")
    examples = examples.sort_values(["model", "condition", "prompt_id"])
    sample = examples.sample(min(sample_size, len(examples)), random_state=random_seed)
    columns = [
        "model",
        "condition",
        "language",
        "prompt",
        "response_baseline",
        "response",
    ]
    sample = sample[columns].copy()
    if public_excerpts is not None:
        for column in ["prompt", "response_baseline", "response"]:
            sample[column] = sample[column].map(
                lambda text: public_excerpts[content_hash(text)]
            )
    elif not include_full_answers:
        for column in ["response_baseline", "response"]:
            sample[column] = sample[column].str.slice(0, 500)
    return sample


def sample_records(records, sample_size, random_seed, columns=None):
    selected = records.sample(min(sample_size, len(records)), random_state=random_seed)
    return selected if columns is None else selected[columns]
