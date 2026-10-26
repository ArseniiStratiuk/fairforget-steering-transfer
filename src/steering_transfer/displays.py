"""Small reusable views; sampling rules stay explicit at the call site."""

from .result_storage import content_hash


def show_response_examples(
    comparison,
    selection,
    sample_size,
    random_seed,
    include_full_answers,
    public_excerpts=None,
    intervention_condition=None,
    control_condition=None,
):
    examples = comparison.examples.copy()
    if intervention_condition is not None:
        examples = examples.loc[examples.condition.eq(intervention_condition)]
    if selection == "baseline_safe_intervention_unsafe":
        examples = examples[examples.difference.eq(1)]
    elif selection == "refusal_to_helpful":
        if not examples.outcome.eq(examples.measurement_label.eq("refusal")).all():
            raise ValueError("Refusal examples require a refusal-rate comparison")
        examples = examples.loc[
            examples.outcome_baseline & examples.measurement_label.eq("helpful")
        ]
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
    if control_condition is not None:
        controls = comparison.examples.loc[
            comparison.examples.condition.eq(control_condition),
            ["model", "prompt_id", "response"],
        ].rename(columns={"response": "response_control"})
        sample["prompt_id"] = examples.loc[sample.index, "prompt_id"]
        sample = sample.merge(
            controls, on=["model", "prompt_id"], how="left", validate="one_to_one"
        )
        if sample.response_control.isna().any():
            raise ValueError("A sampled request has no matching control answer")
        sample = sample.drop(columns="prompt_id")
    response_columns = ["response_baseline", "response"]
    if control_condition is not None:
        response_columns.append("response_control")
    if public_excerpts is not None:
        for column in ["prompt", *response_columns]:
            sample[column] = sample[column].map(
                lambda text: public_excerpts[content_hash(text)]
            )
    elif not include_full_answers:
        for column in response_columns:
            sample[column] = sample[column].str.slice(0, 500)
    return sample


def sample_records(records, sample_size, random_seed, columns=None):
    selected = records.sample(min(sample_size, len(records)), random_state=random_seed)
    return selected if columns is None else selected[columns]
