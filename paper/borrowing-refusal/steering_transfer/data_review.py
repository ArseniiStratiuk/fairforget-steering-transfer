"""Dataset review tables preserve records and explain counts at the notebook."""

import pandas as pd


def summarize_prompt_pairs(pairs):
    source_counts = pairs.groupby("source").agg(
        records=("record_id", "size"), flagged=("discard", "sum")
    )
    completeness = pairs.isna().sum().to_frame("missing")
    repeated = []
    for language, column in [("english", "goal"), ("ukrainian", "ukrainian_prompt")]:
        repeated.append(
            {
                "language": language,
                "unique_texts": pairs[column].nunique(),
                "extra_copies": len(pairs) - pairs[column].nunique(),
                "records_in_repeated_sets": int(
                    pairs[column].duplicated(keep=False).sum()
                ),
                "blank_texts": int(pairs[column].fillna("").str.strip().eq("").sum()),
            }
        )
    neighbors = pairs[["record_id", "source", "similar_record_id", "discard"]].copy()
    targets = pairs.set_index("record_id")
    if not neighbors.similar_record_id.isin(targets.index).all():
        raise ValueError("Similarity flags reference absent records")
    neighbors["neighbor_source"] = neighbors.similar_record_id.map(targets.source)
    neighbors["neighbor_english"] = neighbors.similar_record_id.map(targets.goal)
    neighbors["neighbor_ukrainian"] = neighbors.similar_record_id.map(
        targets.ukrainian_prompt
    )
    neighbors = neighbors.merge(
        pairs[["record_id", "goal", "ukrainian_prompt"]],
        on="record_id",
        validate="one_to_one",
    )
    return {
        "source_counts": source_counts,
        "completeness": completeness,
        "duplicates": pd.DataFrame(repeated),
        "neighbors": neighbors,
        "repeated_english": pairs[pairs.goal.duplicated(keep=False)],
        "repeated_ukrainian": pairs[pairs.ukrainian_prompt.duplicated(keep=False)],
    }


def summarize_utility_data(english, ukrainian):
    paired = english.merge(
        ukrainian,
        on="example_id",
        suffixes=("_en", "_uk"),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if not paired._merge.eq("both").all():
        raise ValueError("Utility language pairing is incomplete")
    if not paired.correct_answer_num_en.eq(paired.correct_answer_num_uk).all():
        raise ValueError("Utility language answer keys differ")
    summary = []
    for frame in [english, ukrainian]:
        summary.append(
            {
                "language": frame.language.iloc[0],
                "questions": len(frame),
                "passages": frame.passage_id.nunique(),
                "repeated_question_texts": int(frame.question.duplicated().sum()),
                "missing_values": int(frame.isna().sum().sum()),
            }
        )
    combined = pd.concat([english, ukrainian], ignore_index=True)
    return {
        "summary": pd.DataFrame(summary),
        "paired": paired,
        "answer_positions": pd.crosstab(
            combined.language,
            combined.correct_answer_num,
        ),
    }


def compare_split_membership(extraction, validation, test):
    groups = {"extraction": extraction, "validation": validation, "test": test}
    for left_name, left in groups.items():
        for right_name, right in groups.items():
            if left_name >= right_name:
                continue
            if set(left.semantic_request_id) & set(right.semantic_request_id):
                raise ValueError(
                    f"Request overlap between {left_name} and {right_name}"
                )
            if set(left.prompt) & set(right.prompt):
                raise ValueError(
                    f"Exact prompt overlap between {left_name} and {right_name}"
                )
    frames = []
    for split, frame in groups.items():
        counts = (
            frame.groupby(["label", "language"]).size().rename("records").reset_index()
        )
        counts.insert(0, "split", split)
        frames.append(counts)
    return pd.concat(frames, ignore_index=True)


def load_translation_review(root):
    from .datasets import load_dataset_sources, resolve_dataset_file

    source = load_dataset_sources(root)["jbb_benign_review"]
    review = pd.read_csv(resolve_dataset_file(root, source), keep_default_na=False)
    if review.record_id.duplicated().any():
        raise ValueError("Translation review contains duplicate record IDs")
    allowed = {"approve", "fixed"}
    if not review.review_status.isin(allowed).all():
        raise ValueError("Unknown translation review status")
    return review
