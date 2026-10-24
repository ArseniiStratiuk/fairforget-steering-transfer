"""Screen translation candidates with COMET and text checks; not human approval."""


def main():
    from steering_transfer.result_storage import find_project_root
    from pathlib import Path
    from importlib.metadata import version
    from itertools import combinations
    import hashlib
    import json
    import os
    import platform
    import re

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from IPython.display import display

    root = find_project_root()
    trans_dir = root / "data/safety/translations"
    systems = ["gemma", "lapa", "claude"]
    colors = {"gemma": "#2a78d6", "lapa": "#d55e00", "claude": "#1baf7a"}
    qe_model_name = os.environ.get("QE_MODEL", "Unbabel/wmt22-cometkiwi-da")
    ref_model_name = os.environ.get("REF_MODEL", "none")
    score_raw = os.environ.get("SCORE_RAW", "1") == "1"
    batch_size = int(os.environ.get("BATCH_SIZE", "32"))
    n_boot = int(os.environ.get("BOOTSTRAP", "2000"))
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    gpus = 1 if torch.cuda.is_available() else 0
    pd.set_option("display.max_colwidth", None)
    print(
        "Device:",
        torch.cuda.get_device_name(0) if gpus else f"CPU ({os.cpu_count()} cores)",
    )
    print(
        "QE model:",
        qe_model_name,
        "| raw-text scoring:",
        score_raw,
        "| agreement model:",
        ref_model_name,
    )

    outputs = {}
    for name in ["gemma", "lapa"]:
        path = trans_dir / f"prompt-pairs-uk-{name}.csv"
        df = pd.read_csv(path, keep_default_na=False)
        assert df.record_id.is_unique, path
        outputs[name] = df.set_index("record_id")
        print(
            f"{path.name}: {len(df)} rows, SHA256 {hashlib.sha256(path.read_bytes()).hexdigest()[:16]}..."
        )
    gemma_frame, lapa_frame = outputs["gemma"], outputs["lapa"]
    assert gemma_frame.index.equals(lapa_frame.index), (
        "Gemma and Lapa files cover different record_ids"
    )
    assert gemma_frame.goal.equals(lapa_frame.goal), (
        "English goal differs between the Gemma and Lapa files"
    )

    pinned = json.loads((root / "configs/datasets.json").read_text())["prompt_pairs"]
    candidates = [
        os.environ.get("CLAUDE_CSV"),
        root / "data/safety/translated_formated.csv",
        root / "data/safety/prompt-pairs.csv",
    ]
    claude_path = next((Path(p) for p in candidates if p and Path(p).is_file()), None)
    if claude_path is None:
        from huggingface_hub import hf_hub_download

        claude_path = Path(
            hf_hub_download(
                pinned["repo_id"],
                pinned["filename"],
                repo_type="dataset",
                revision=pinned["revision"],
                local_files_only=offline,
            )
        )
    claude_sha = hashlib.sha256(claude_path.read_bytes()).hexdigest()
    print("Claude translations:", claude_path)
    print("SHA256 matches pinned prompt-pairs.csv:", claude_sha == pinned["sha256"])
    assert claude_sha == pinned["sha256"], (
        "Claude file is not the pinned prompt-pairs.csv"
    )
    claude = pd.read_csv(claude_path).set_index("record_id").reindex(gemma_frame.index)
    assert claude.goal.notna().all(), "Some record_ids are missing from the Claude file"
    mismatched = int((claude.goal.str.strip() != gemma_frame.goal.str.strip()).sum())
    assert mismatched == 0, (
        f"{mismatched} English goals differ between Claude and model files"
    )

    data = pd.DataFrame(
        {
            "source": gemma_frame.source,
            "goal": gemma_frame.goal,
            "uk_gemma": gemma_frame.ukrainian_prompt,
            "uk_lapa": lapa_frame.ukrainian_prompt,
            "uk_claude": claude.ukrainian_prompt.fillna(""),
            "status_gemma": gemma_frame.status,
            "status_lapa": lapa_frame.status,
        }
    ).sort_index()
    excluded = data[
        (data.status_gemma != "ok")
        | (data.status_lapa != "ok")
        | data.uk_claude.str.strip().eq("")
    ]
    print("Excluded from scoring:", len(excluded))
    if len(excluded):
        display(excluded)
    data = data.drop(excluded.index)
    display(data.groupby("source").size().to_frame("records"))

    FORMAL = re.compile(
        r"^[\u0410-\u042f\u0490\u0404\u0406\u0407\u0430-\u044f\u0491\u0454\u0456\u0457'\u2019]+(\u0456\u0442\u044c|\u0457\u0442\u044c|\u0439\u0442\u0435|\u044c\u0442\u0435)$"
    )
    FINAL_PUNCT = re.compile(r"[.!?...]+$")

    def cyrillic_share(text):
        letters = [c for c in text if c.isalpha()]
        return sum("\u0400" <= c <= "\u04ff" for c in letters) / max(1, len(letters))

    def latin_leak(text, source):
        allowed = {w for w in re.findall(r"[A-Za-z]{3,}", source) if not w.islower()}
        return " ".join(
            w for w in re.findall(r"[A-Za-z]{3,}", text) if w not in allowed
        )

    def first_word(text):
        words = re.findall(r"[^\s,.:;!?\xab\xbb\"()]+", text)
        return words[0] if words else ""

    checks = {}
    for name in systems:
        col = data[f"uk_{name}"]
        checks[name] = pd.DataFrame(
            {
                "identical_to_source": col.str.strip().str.lower()
                == data.goal.str.strip().str.lower(),
                "cyrillic_share": col.map(cyrillic_share),
                "formal_imperative": col.map(
                    lambda t: bool(FORMAL.match(first_word(t)))
                ),
                "length_ratio": col.str.split().str.len()
                / data.goal.str.split().str.len(),
                "latin_leak": [latin_leak(t, s) for t, s in zip(col, data.goal)],
                "final_punct": col.str.strip().str.contains(FINAL_PUNCT),
            }
        )
    summary_checks = pd.DataFrame(
        {
            name: {
                "identical_to_source": int(c.identical_to_source.sum()),
                "cyrillic_share < 0.8": int((c.cyrillic_share < 0.8).sum()),
                "formal_imperative": int(c.formal_imperative.sum()),
                "length_ratio > 2": int((c.length_ratio > 2).sum()),
                "length_ratio < 0.5": int((c.length_ratio < 0.5).sum()),
                "latin_leak": int(c.latin_leak.ne("").sum()),
                "final_punct": int(c.final_punct.sum()),
                "median length_ratio": round(float(c.length_ratio.median()), 2),
            }
            for name, c in checks.items()
        }
    )
    display(summary_checks)
    print(
        "English sources ending with . ! ?:",
        int(data.goal.str.strip().str.contains(FINAL_PUNCT).sum()),
    )
    for a, b in combinations(systems, 2):
        print(
            f"Identical text, {a} vs {b}: {int((data[f'uk_{a}'] == data[f'uk_{b}']).sum())} records"
        )
    for name, c in checks.items():
        rows = c.index[
            (c.cyrillic_share < 0.8) | (c.length_ratio > 2) | (c.length_ratio < 0.5)
        ]
        if len(rows):
            print(f"\n{name}: low Cyrillic share or unusual length")
            display(
                data.loc[rows, ["source", "goal", f"uk_{name}"]].join(
                    c.loc[rows, ["cyrillic_share", "length_ratio"]].round(2)
                )
            )
        leaks = c.index[c.latin_leak.ne("")]
        if len(leaks):
            print(f"\n{name}: Latin-script words left in the translation")
            display(
                data.loc[leaks, ["goal", f"uk_{name}"]].join(
                    c.loc[leaks, ["latin_leak"]]
                )
            )
        formal = c.index[c.formal_imperative]
        if len(formal):
            print(f"\n{name}: formal imperative, first 10 of {len(formal)}")
            display(data.loc[formal[:10], ["goal", f"uk_{name}"]])

    from comet import download_model, load_from_checkpoint

    def load_comet(name):
        path = download_model(name, local_files_only=offline)
        return load_from_checkpoint(path, local_files_only=offline)

    def comet_scores(model, samples):
        prediction = model.predict(
            samples, batch_size=batch_size, gpus=gpus, progress_bar=True
        )
        return np.asarray(prediction.scores, dtype=float)

    def strip_final(text):
        return FINAL_PUNCT.sub("", text.strip()).rstrip()

    skipped = {}

    def try_load(name, kind):
        try:
            return load_comet(name)
        except Exception as error:
            cause = error.__context__ or error
            while cause.__context__ is not None and "Gated" not in type(cause).__name__:
                cause = cause.__context__
            reason = f"{type(cause).__name__}: {str(cause).strip().splitlines()[-1]}"
            if "Gated" in type(cause).__name__ or "authorized list" in str(cause):
                reason += f" -> accept the licence at https://huggingface.co/{name} while logged in, then rerun"
            skipped[kind] = reason
            print(f"SKIPPED {kind} ({name}): {reason}")
            return None

    scores = pd.DataFrame(index=data.index)
    if (
        qe_model_name.lower() != "none"
        and (qe_model := try_load(qe_model_name, "qe")) is not None
    ):
        for name in systems:
            print("QE (final punctuation stripped):", name)
            samples = [
                {"src": s, "mt": strip_final(t)}
                for s, t in zip(data.goal, data[f"uk_{name}"])
            ]
            scores[f"qe_{name}"] = comet_scores(qe_model, samples)
            if score_raw:
                print("QE (raw text):", name)
                samples = [
                    {"src": s, "mt": t} for s, t in zip(data.goal, data[f"uk_{name}"])
                ]
                scores[f"qeraw_{name}"] = comet_scores(qe_model, samples)
        del qe_model
    if (
        ref_model_name.lower() != "none"
        and (ref_model := try_load(ref_model_name, "agree")) is not None
    ):
        for name in ["gemma", "lapa"]:
            print("Agreement with Claude:", name)
            samples = [
                {"src": s, "mt": strip_final(t), "ref": strip_final(r)}
                for s, t, r in zip(data.goal, data[f"uk_{name}"], data.uk_claude)
            ]
            scores[f"agree_{name}"] = comet_scores(ref_model, samples)
        del ref_model
    metrics = [
        m
        for m in ["qe", "qeraw", "agree"]
        if any(c.startswith(f"{m}_") for c in scores.columns)
    ]
    if not metrics:
        raise RuntimeError(f"No COMET metric could run: {skipped or 'all disabled'}")
    main = metrics[0]
    label = {
        "qe": "CometKiwi",
        "qeraw": "CometKiwi (raw text)",
        "agree": "COMET agreement with Claude",
    }[main]
    print("Computed:", list(scores.columns))

    def mean_ci(values):
        values = np.asarray(values, dtype=float)
        idx = np.random.default_rng(42).integers(
            0, len(values), size=(n_boot, len(values))
        )
        boots = values[idx].mean(axis=1)
        return values.mean(), *np.percentile(boots, [2.5, 97.5])

    subsets = [("all", data.index)] + [
        (src, data.index[data.source == src]) for src in sorted(data.source.unique())
    ]

    rows = []
    for metric in metrics:
        for name in [s for s in systems if f"{metric}_{s}" in scores]:
            for subset, idx in subsets:
                values = scores.loc[idx, f"{metric}_{name}"]
                m, lo, hi = mean_ci(values)
                worst = np.sort(values.to_numpy())[: max(1, round(0.05 * len(values)))]
                rows.append(
                    {
                        "metric": metric,
                        "system": name,
                        "subset": subset,
                        "n": len(idx),
                        "mean": m,
                        "ci_low": lo,
                        "ci_high": hi,
                        "below_0.75": int((values < 0.75).sum()),
                        "worst_5pct": float(worst.mean()),
                    }
                )
    system_table = pd.DataFrame(rows)
    display(
        system_table.pivot_table(
            index=["metric", "subset"], columns="system", values="mean", sort=False
        ).round(4)
    )
    display(system_table.round(4))

    paired = []
    for metric in metrics:
        present = [s for s in systems if f"{metric}_{s}" in scores]
        for a, b in combinations(present, 2):
            for subset, idx in subsets:
                diff = (
                    scores.loc[idx, f"{metric}_{a}"] - scores.loc[idx, f"{metric}_{b}"]
                ).to_numpy()
                m, lo, hi = mean_ci(diff)
                paired.append(
                    {
                        "metric": metric,
                        "a": a,
                        "b": b,
                        "subset": subset,
                        "n": len(diff),
                        "mean_diff_a_minus_b": m,
                        "ci_low": lo,
                        "ci_high": hi,
                        "interval_excludes_0": bool(lo > 0 or hi < 0),
                        "a_better": float((diff > 0).mean()),
                        "b_better": float((diff < 0).mean()),
                        "tie": float((diff == 0).mean()),
                    }
                )
    paired_table = pd.DataFrame(paired)
    display(paired_table.round(4))

    best = {}
    for metric in [m for m in ["qe", "qeraw"] if m in metrics]:
        cols = [f"{metric}_{s}" for s in systems]
        for subset, idx in subsets:
            winner = scores.loc[idx, cols].idxmax(axis=1).str.removeprefix(f"{metric}_")
            best[(metric, subset)] = winner.value_counts(normalize=True).reindex(
                systems, fill_value=0
            )
    if best:
        display(
            (pd.DataFrame(best).T * 100)
            .round(1)
            .rename_axis(["metric", "subset"])
            .add_suffix(" % best")
        )

    DECISIVE_GAP = (
        0.01  # absolute difference in mean CometKiwi treated as practically meaningful
    )
    overall = system_table[
        (system_table.metric == main) & (system_table.subset == "all")
    ].set_index("system")["mean"]
    pairs_all = paired_table[
        (paired_table.metric == main) & (paired_table.subset == "all")
    ]
    spread_means = overall.max() - overall.min()
    leader = overall.idxmax()
    leader_pairs = pairs_all[(pairs_all.a == leader) | (pairs_all.b == leader)]
    leader_clear = all(
        row.interval_excludes_0
        and abs(row.mean_diff_a_minus_b) >= DECISIVE_GAP
        and ((row.a == leader) == (row.mean_diff_a_minus_b > 0))
        for row in leader_pairs.itertuples()
    )
    print(
        f"Mean {label}, all records:",
        ", ".join(
            f"{s} {v:.4f}" for s, v in overall.sort_values(ascending=False).items()
        ),
    )
    print(
        f"Largest gap between system means: {spread_means:.4f} (threshold for a meaningful gap: {DECISIVE_GAP})"
    )
    print(
        "Pairwise gaps larger than resampling noise:",
        ", ".join(
            f"{r.a}-{r.b} {r.mean_diff_a_minus_b:+.4f}"
            for r in pairs_all.itertuples()
            if r.interval_excludes_0
        )
        or "none",
    )
    if leader_clear:
        print(
            f"\nVERDICT: COMET clearly favours {leader}. Confirm with the human review before adopting it."
        )
    else:
        print(
            "\nVERDICT: No system met the configured COMET preference criterion. "
            "Inspect the score tables and complete human review before choosing translations. "
            "Use the COMET scores and rule checks below to prioritise what reviewers should look at."
        )
    verdict = {
        "decisive": bool(leader_clear),
        "leader": leader,
        "largest_gap": float(spread_means),
        "threshold": DECISIVE_GAP,
    }

    if {"qe", "qeraw"} <= set(metrics):
        effect = pd.DataFrame(
            {
                name: {
                    "translations with final punctuation": int(
                        checks[name].final_punct.sum()
                    ),
                    "mean raw": scores[f"qeraw_{name}"].mean(),
                    "mean stripped": scores[f"qe_{name}"].mean(),
                    "raw minus stripped": scores[f"qeraw_{name}"].mean()
                    - scores[f"qe_{name}"].mean(),
                }
                for name in systems
            }
        )
        display(effect.round(4))
    else:
        print("Raw-text scores not computed (SCORE_RAW=0).")

    present = [s for s in systems if f"{main}_{s}" in scores]
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    cols = [f"{main}_{s}" for s in present]
    bins = np.linspace(scores[cols].min().min(), scores[cols].max().max(), 31)
    for name in present:
        axes[0].hist(
            scores[f"{main}_{name}"],
            bins=bins,
            histtype="step",
            linewidth=2,
            color=colors[name],
            label=f"{name} (mean {scores[f'{main}_{name}'].mean():.3f})",
        )
    axes[0].set(
        xlabel=f"{label} score per record", ylabel="Records", title="Score distribution"
    )
    axes[0].legend(loc="upper left", framealpha=0.9, edgecolor="none")
    axes[0].grid(axis="y", alpha=0.25)

    table = system_table[system_table.metric == main].set_index(["subset", "system"])
    subset_names = [s for s, _ in subsets]
    offsets = np.linspace(-0.22, 0.22, len(present))
    for offset, name in zip(offsets, present):
        y = np.arange(len(subset_names)) + offset
        means = table.loc[[(s, name) for s in subset_names], "mean"].to_numpy()
        lows = table.loc[[(s, name) for s in subset_names], "ci_low"].to_numpy()
        highs = table.loc[[(s, name) for s in subset_names], "ci_high"].to_numpy()
        axes[1].errorbar(
            means,
            y,
            xerr=[means - lows, highs - means],
            fmt="o",
            markersize=8,
            color=colors[name],
            ecolor=colors[name],
            elinewidth=2,
            capsize=0,
            label=name,
        )
    axes[1].set_yticks(
        np.arange(len(subset_names)), [f"{s} (n={len(i)})" for s, i in subsets]
    )
    axes[1].invert_yaxis()
    axes[1].set(xlabel=f"Mean {label} (95% interval)", title="Mean by source")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(present),
        frameon=False,
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    present = [s for s in systems if f"{main}_{s}" in scores]
    for name in present:
        worst = scores[f"{main}_{name}"].nsmallest(10).index
        print(f"Lowest {label}: {name}")
        display(
            data.loc[worst, ["source", "goal", f"uk_{name}"]].assign(
                score=scores.loc[worst, f"{main}_{name}"].round(3)
            )
        )
    spread = scores[[f"{main}_{s}" for s in present]]
    gap = (spread.max(axis=1) - spread.min(axis=1)).nlargest(10).index
    print("Largest disagreement between the three systems")
    display(
        data.loc[gap, ["goal"] + [f"uk_{s}" for s in present]].join(
            spread.loc[gap].round(3)
        )
    )

    out = data.join(scores)
    for name, c in checks.items():
        out = out.join(c.add_suffix(f"_{name}"))
    out = out.reset_index()
    out_path = trans_dir / "comet-scores.csv"
    out.to_csv(out_path, index=False, encoding="utf-8")

    summary = {
        "systems": {
            "gemma": {
                "file": "prompt-pairs-uk-gemma.csv",
                "sha256": hashlib.sha256(
                    (trans_dir / "prompt-pairs-uk-gemma.csv").read_bytes()
                ).hexdigest(),
            },
            "lapa": {
                "file": "prompt-pairs-uk-lapa.csv",
                "sha256": hashlib.sha256(
                    (trans_dir / "prompt-pairs-uk-lapa.csv").read_bytes()
                ).hexdigest(),
            },
            "claude": {
                "file": f"{pinned['repo_id']}@{pinned['revision']}/{pinned['filename']}",
                "sha256": pinned["sha256"],
                "column": "ukrainian_prompt",
                "note": "translated by Claude per the team; model version and prompt not recorded in the dataset",
            },
        },
        "records_scored": int(len(out)),
        "records_excluded": int(len(excluded)),
        "main_score": "CometKiwi on translations with final . ! ? stripped (English sources have none)",
        "qe_model": qe_model_name if "qe" in metrics else None,
        "agreement_model": ref_model_name if "agree" in metrics else None,
        "skipped": skipped,
        "verdict": verdict,
        "decision": "model choice made by blind human review (src/translation_review/); COMET is supporting evidence only",
        "bootstrap": {"resamples": n_boot, "seed": 42, "interval": "95% percentile"},
        "system_scores": system_table.to_dict(orient="records"),
        "pairwise": paired_table.to_dict(orient="records"),
        "best_of_three_percent": {
            f"{m}/{s}": (v * 100).round(1).to_dict() for (m, s), v in best.items()
        },
        "rule_checks": summary_checks.to_dict(),
        "versions": {
            p: version(p) for p in ["unbabel-comet", "torch", "transformers", "pandas"]
        },
        "device": torch.cuda.get_device_name(0) if gpus else "cpu",
        "hostname": platform.node(),
    }
    (trans_dir / "comet-summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("Saved:", out_path, "and comet-summary.json")


if __name__ == "__main__":
    main()
