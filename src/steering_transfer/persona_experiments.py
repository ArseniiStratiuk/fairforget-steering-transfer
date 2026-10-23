"""Persona-specific input construction and explicit contrast eligibility."""

import pandas as pd

from .result_storage import content_hash


def build_persona_prompts(protocol, split):
    records = []
    for question in protocol["questions"]:
        if question["split"] != split:
            continue
        for prefix in protocol["instruction_pairs"]:
            for repeat in range(protocol["generation"]["repeats"]):
                pair_id = f"{question['id']}/{prefix['id']}/{repeat}"
                for side in ["positive", "negative"]:
                    records.append(
                        {
                            "prompt_id": f"{pair_id}/{side}",
                            "pair_id": pair_id,
                            "semantic_request_id": question["id"],
                            "question_id": question["id"],
                            "question": question["text"],
                            "language": "en",
                            "split": split,
                            "prefix_id": prefix["id"],
                            "repeat": repeat,
                            "persona": side,
                            "prompt": prefix[side] + "\nQuestion: " + question["text"],
                        }
                    )
    if not records:
        raise ValueError("No persona questions in requested split")
    return pd.DataFrame(records)


def select_persona_pairs(
    responses,
    behavior_labels,
    coherence_labels,
    positive_label,
    negative_label,
    require_coherence,
    exclude_capped,
):
    if responses.duplicated(["pair_id", "persona"]).any():
        raise ValueError("Repeated persona pair sides")
    for labels in [behavior_labels, coherence_labels]:
        if labels.response_id.duplicated().any():
            raise ValueError("Repeated persona judgments")
    answer_hashes = responses.set_index("response_id").response.map(content_hash)
    for labels in [behavior_labels, coherence_labels]:
        for label in labels.to_dict("records"):
            response_id = label["response_id"]
            if (
                response_id not in answer_hashes
                or label["response_hash"] != answer_hashes[response_id]
            ):
                raise ValueError("Persona judgment references another response")
    behavior = behavior_labels[["response_id", "label", "status"]].rename(
        columns={"label": "behavior", "status": "behavior_status"}
    )
    coherence = coherence_labels[["response_id", "label", "status"]].rename(
        columns={"label": "coherence", "status": "coherence_status"}
    )
    joined = responses.merge(
        behavior, on="response_id", how="left", validate="one_to_one"
    )
    joined = joined.merge(
        coherence, on="response_id", how="left", validate="one_to_one"
    )
    accepted, rejected = [], []
    for pair_id, pair in joined.groupby("pair_id"):
        if set(pair.persona) != {"positive", "negative"}:
            rejected.append({"pair_id": pair_id, "reason": "missing_side"})
            continue
        sides = pair.set_index("persona")
        labels_valid = sides.behavior_status.eq("success").all()
        contrast_matches = (
            sides.loc["positive", "behavior"] == positive_label
            and sides.loc["negative", "behavior"] == negative_label
        )
        coherent = (
            sides.coherence_status.eq("success").all()
            and sides.coherence.eq("coherent").all()
        )
        capped = sides.stop_reason.eq("token_limit").any()
        if not labels_valid or not contrast_matches:
            reason = "missing_judgment_or_no_contrast"
        elif require_coherence and not coherent:
            reason = "coherence_filter"
        elif exclude_capped and capped:
            reason = "truncation_filter"
        else:
            accepted.append(pair_id)
            continue
        rejected.append({"pair_id": pair_id, "reason": reason})
    return {
        "responses": joined[joined.pair_id.isin(accepted)].copy(),
        "accepted_pair_ids": accepted,
        "rejected": pd.DataFrame(rejected),
    }


def extract_persona_directions(
    model, responses, layers, methods, maximum_prefix_tokens, output_directory
):
    from pathlib import Path

    from .activation_extraction import capture_response_activations
    from .direction_processing import compute_contrastive_directions

    lengths = responses.response_ids.map(len)
    pair_lengths = lengths.groupby(responses.pair_id).transform("min")
    selected = responses.copy()
    selected["response_prefix_length"] = pair_lengths.clip(upper=maximum_prefix_tokens)
    directions = {}
    for method in methods:
        capture = capture_response_activations(
            model,
            selected,
            layers,
            "block_output",
            method,
            Path(output_directory) / method,
        )
        directions[method] = compute_contrastive_directions(
            capture,
            selected.loc[selected.persona.eq("positive"), "prompt_id"],
            selected.loc[selected.persona.eq("negative"), "prompt_id"],
        )
    return directions


def create_persona_conditions(
    direction_sets, layer, strengths, random_seed, token_positions
):
    from .direction_processing import create_random_directions, select_direction
    from .interventions import add_direction

    conditions = []
    for method, directions in direction_sets.items():
        candidate = select_direction(directions, layer, "raw", [])
        random = create_random_directions(candidate, [random_seed], [])[random_seed]
        for strength in strengths:
            for kind, direction in [("candidate", candidate), ("random", random)]:
                name = f"{method}_{kind}_{strength:+g}"
                condition = add_direction(
                    name, direction, layer, strength, token_positions
                )
                conditions.append(condition)
    return conditions


def build_persona_evaluation_prompts(protocol, splits):
    records = []
    for question in protocol["questions"]:
        if question["split"] in splits:
            records.append(
                {
                    "prompt_id": question["id"],
                    "semantic_request_id": question["id"],
                    "prompt": question["text"],
                    "language": "en",
                    "split": question["split"],
                }
            )
    if not records:
        raise ValueError("No persona evaluation questions in requested splits")
    return pd.DataFrame(records)
