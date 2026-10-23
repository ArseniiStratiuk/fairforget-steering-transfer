"""Paired comparisons use stable request IDs, not row order."""

import numpy as np
import pandas as pd

from .records import Comparison
from .result_storage import content_hash


def bootstrap_paired_change(pairs, cluster_column, repeats=2000, seed=42):
    if pairs.empty:
        return {"change": np.nan, "low": np.nan, "high": np.nan}
    grouped = pairs.groupby(cluster_column, sort=True).agg(
        difference=("difference", "sum"), observations=("difference", "size")
    )
    values = grouped[["difference", "observations"]].to_numpy()
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled = values[generator.integers(0, len(values), len(values))]
        estimates.append(sampled[:, 0].sum() / sampled[:, 1].sum())
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "change": float(pairs.difference.mean()),
        "low": float(low),
        "high": float(high),
    }


def compare_conditions(
    responses,
    evaluations,
    baseline_condition,
    group_by,
    metric,
    uncertainty="paired_request_bootstrap",
):
    label_for_metric = {
        "policy_unsafe_rate": "unsafe",
        "refusal_rate": "refusal",
        "coherence_rate": "coherent",
        "coherent_refusal_rate": "coherent_refusal",
        "helpful_rate": "helpful",
        "mixed_rate": "mixed",
    }
    if metric not in label_for_metric or uncertainty != "paired_request_bootstrap":
        raise ValueError("Unknown measurement or uncertainty method")
    if (
        responses.response_id.duplicated().any()
        or evaluations.response_id.duplicated().any()
    ):
        raise ValueError(
            "Comparison requires unique responses and one judgment per response"
        )
    if "rubric_revision" in evaluations and evaluations.rubric_revision.nunique() != 1:
        raise ValueError("Compare one rubric revision at a time")
    evaluations = evaluations.rename(
        columns={
            "label": "measurement_label",
            "response_hash": "evaluated_response_hash",
        }
    )
    joined = responses.merge(
        evaluations,
        on="response_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_evaluation"),
        indicator=True,
    )
    for row in joined.loc[joined._merge.eq("both")].to_dict("records"):
        if row["evaluated_response_hash"] != content_hash(row["response"]):
            raise ValueError("Evaluation references different response content")
    valid = joined._merge.eq("both") & joined.status.eq("success")
    completeness = {
        "responses": len(responses),
        "successful_evaluations": int(valid.sum()),
        "missing_or_failed_evaluations": int((~valid).sum()),
    }
    judged = joined.loc[valid].copy()
    if judged.empty:
        raise ValueError("No successful matching evaluations")
    judged["outcome"] = judged["measurement_label"].eq(label_for_metric[metric])
    counts = judged.groupby([*group_by, "condition"], dropna=False).agg(
        answers=("response_id", "size"),
        positive_labels=("outcome", "sum"),
        rate=("outcome", "mean"),
    )
    baseline = judged[judged.condition.eq(baseline_condition)]
    keys = ["model", "prompt_id"]
    if baseline.duplicated(keys).any():
        raise ValueError("Baseline request identities are ambiguous")
    paired = judged.merge(
        baseline[[*keys, "outcome", "response"]],
        on=keys,
        suffixes=("", "_baseline"),
        how="left",
        validate="many_to_one",
    )
    completeness["answers_without_baseline"] = int(paired.outcome_baseline.isna().sum())
    paired = paired[paired.outcome_baseline.notna()].copy()
    paired["difference"] = paired.outcome.astype(int) - paired.outcome_baseline.astype(
        int
    )
    paired["became_positive"] = paired.difference.eq(1)
    paired["became_negative"] = paired.difference.eq(-1)
    group_columns = [*group_by, "condition"]
    changes = paired.groupby(group_columns).agg(
        paired_answers=("response_id", "size"),
        became_positive=("became_positive", "sum"),
        became_negative=("became_negative", "sum"),
    )
    intervals = []
    for group, frame in paired.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group, strict=True))
        row.update(bootstrap_paired_change(frame, "semantic_request_id"))
        intervals.append(row)
    return Comparison(counts, changes, pd.DataFrame(intervals), paired, completeness)


def compare_utility_scores(
    results, baseline_condition, group_by, uncertainty="paired_passage_bootstrap"
):
    if uncertainty != "paired_passage_bootstrap":
        raise ValueError("Use passage-clustered utility uncertainty")
    keys = ["model", "language", "example_id"]
    if results.duplicated([*keys, "condition"]).any():
        raise ValueError("Repeated utility predictions")
    counts = results.groupby([*group_by, "condition"]).agg(
        questions=("correct", "size"),
        correct=("correct", "sum"),
        accuracy=("correct", "mean"),
    )
    baseline = results[results.condition.eq(baseline_condition)]
    paired = results.merge(
        baseline[[*keys, "correct", "answer"]],
        on=keys,
        suffixes=("", "_baseline"),
        how="left",
        validate="many_to_one",
    )
    missing = int(paired.correct_baseline.isna().sum())
    paired = paired[paired.correct_baseline.notna()].copy()
    if not paired.answer.eq(paired.answer_baseline).all():
        raise ValueError("Paired utility answer keys differ")
    paired["difference"] = paired.correct.astype(int) - paired.correct_baseline.astype(
        int
    )
    paired["gained"] = paired.difference.eq(1)
    paired["lost"] = paired.difference.eq(-1)
    group_columns = [*group_by, "condition"]
    changes = paired.groupby(group_columns).agg(
        paired_questions=("example_id", "size"),
        gained=("gained", "sum"),
        lost=("lost", "sum"),
    )
    intervals = []
    for group, frame in paired.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, group, strict=True))
        row.update(bootstrap_paired_change(frame, "passage_id"))
        intervals.append(row)
    return Comparison(
        counts,
        changes,
        pd.DataFrame(intervals),
        paired,
        {"predictions": len(results), "without_baseline": missing},
    )


def compare_directions(
    source_directions, target_directions, layers, coordinate_mask=()
):
    import torch

    rows = []
    for layer in layers:
        source = (
            source_directions["directions"][source_directions["layers"].index(layer)]
            .float()
            .clone()
        )
        target = (
            target_directions["directions"][target_directions["layers"].index(layer)]
            .float()
            .clone()
        )
        if source.shape != target.shape:
            raise ValueError("Directions have incompatible widths")
        source[list(coordinate_mask)] = 0
        target[list(coordinate_mask)] = 0
        if (
            not torch.isfinite(source).all()
            or not torch.isfinite(target).all()
            or source.norm() == 0
            or target.norm() == 0
        ):
            raise ValueError("Cannot compare zero or non-finite directions")
        cosine = float(torch.dot(source, target) / (source.norm() * target.norm()))
        rows.append(
            {
                "layer": layer,
                "cosine": cosine,
                "source_norm": float(source.norm()),
                "target_norm": float(target.norm()),
            }
        )
    return pd.DataFrame(rows)


def compare_judge_labels(evaluations, human_annotations):
    if evaluations.response_id.duplicated().any():
        raise ValueError("Use one judge result per response")
    if human_annotations.duplicated(["response_id", "annotator_id"]).any():
        raise ValueError("Duplicate annotation by the same annotator")
    keys = ["response_id", "rubric_revision"]
    paired = human_annotations.merge(
        evaluations,
        on=keys,
        suffixes=("_human", "_judge"),
        how="left",
        validate="many_to_one",
    )
    successful = paired.status.eq("success")
    scored = paired.loc[successful].copy()
    scored["agreement"] = scored.label_human.eq(scored.label_judge)
    return {
        "agreement": scored.groupby("annotator_id").agreement.agg(["size", "mean"]),
        "confusion": pd.crosstab(scored.label_human, scored.label_judge),
        "disagreements": scored.loc[~scored.agreement],
        "unmatched_or_failed": int((~successful).sum()),
    }


def compare_persona_conditions(
    responses, behavior_labels, coherence_labels, baseline_condition, group_by
):
    behavior = compare_conditions(
        responses, behavior_labels, baseline_condition, group_by, "refusal_rate"
    )
    coherence = compare_conditions(
        responses, coherence_labels, baseline_condition, group_by, "coherence_rate"
    )
    paired = behavior_labels.merge(
        coherence_labels,
        on="response_id",
        suffixes=("_behavior", "_coherence"),
        validate="one_to_one",
    )
    if set(paired.response_id) != set(responses.response_id):
        raise ValueError("Persona comparison requires both judgments for every answer")
    joint_labels = []
    for row in paired.to_dict("records"):
        if row["response_hash_behavior"] != row["response_hash_coherence"]:
            raise ValueError(
                "Behavior and coherence judgments reference different answers"
            )
        coherent_refusal = (
            row["label_behavior"] == "refusal" and row["label_coherence"] == "coherent"
        )
        joint_labels.append(
            {
                "response_id": row["response_id"],
                "response_hash": row["response_hash_behavior"],
                "label": "coherent_refusal" if coherent_refusal else "other",
                "status": "success"
                if row["status_behavior"] == row["status_coherence"] == "success"
                else "missing_judgment",
                "rubric_revision": content_hash(
                    [
                        row["rubric_revision_behavior"],
                        row["rubric_revision_coherence"],
                        "coherent_and_refusal",
                    ]
                ),
            }
        )
    comparison = compare_conditions(
        responses,
        pd.DataFrame(joint_labels),
        baseline_condition,
        group_by,
        "coherent_refusal_rate",
    )
    comparison.counts = comparison.counts.rename(
        columns={"positive_labels": "coherent_refusals"}
    )
    comparison.counts["refusal_labels"] = behavior.counts.positive_labels
    comparison.counts["coherent_labels"] = coherence.counts.positive_labels
    return comparison
