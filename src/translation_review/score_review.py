#!/usr/bin/env python3
"""Read the filled-in reviewer files, map every option back to its model with the secret key, and score.

Checks every returned file before scoring: it must belong to exactly one reviewer in the key, contain all of that
reviewer's rows with unchanged codes, and use only allowed values. Problems are listed and the script stops
(use --allow-incomplete to score only fully rated prompts).

Reports, per model: mean adequacy and fluency, share of prompts where the model is ranked best (ties count for
every tied option), mean rank, and the share of translations with a critical issue (meaning changed, content
added, content dropped). Pairwise comparisons use a bootstrap that resamples whole records, so ratings of the
same prompt stay together. Agreement on the shared block is Krippendorff's alpha (ordinal).

Decision rule (fixed before the review):
  1. Agreement level from Krippendorff's alpha (ordinal) for adequacy on the shared prompts:
       alpha < 0.4           insufficient: reviewers apply the scale differently -> recalibrate, no decision
       0.4 <= alpha < 0.667  tentative: decide, but statements based on adequacy and critical issues are tentative
       alpha >= 0.667        reliable (Krippendorff's minimum for tentative conclusions is met)
     The thresholds were fixed before the review and must not be changed after seeing the results.
  2. The leader is the model ranked best most often. It is the winner if, against each other model, the paired
     preference interval excludes 0.
  3. Otherwise the models not separated from the leader are tied on preference; the report gives their critical-
     issue rates and the team decides, preferring the one with fewer critical issues.

Usage:
  python src/translation_review/score_review.py data/safety/review/returned/*.xlsx
Outputs (default data/safety/review/key/, because they reveal the models):
  review_results_long.csv  every rating with record_id and model
  review_summary.json      all tables and the decision
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CRITICAL_ISSUES,
    ISSUES,
    RANK_VALUES,
    REVIEW_SHEET,
    ROOT,
    SCORE_VALUES,
    SYSTEMS,
)

ALPHA_RECALIBRATE = 0.4  # below: reviewers use the adequacy scale too differently to decide
ALPHA_RELIABLE = 0.667   # Krippendorff's lowest acceptable level for tentative conclusions


def agreement_level(alpha: float) -> str:
    if alpha is None or np.isnan(alpha):
        return "not computed"
    if alpha < ALPHA_RECALIBRATE:
        return "insufficient"
    return "tentative" if alpha < ALPHA_RELIABLE else "reliable"


class ReviewError(Exception):
    pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("files", nargs="+", help="filled reviewer .xlsx files")
    p.add_argument("--key-dir", default=str(ROOT / "data/safety/review/key"))
    p.add_argument("--out", default=None, help="output folder (default: the key folder)")
    p.add_argument("--allow-incomplete", action="store_true", help="score fully rated prompts, skip the rest")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def read_filled(path: Path) -> pd.DataFrame:
    wb = load_workbook(path, data_only=True, read_only=True)
    if REVIEW_SHEET not in wb.sheetnames:
        raise ReviewError(f"{path.name}: no '{REVIEW_SHEET}' sheet")
    rows = list(wb[REVIEW_SHEET].iter_rows(values_only=True))
    wb.close()
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    missing = {"code", "adequacy", "fluency", "rank", "issue", "comment"} - set(header)
    if missing:
        raise ReviewError(f"{path.name}: missing columns {sorted(missing)}")
    df = pd.DataFrame([r for r in rows[1:] if any(v is not None for v in r)], columns=header)
    df["file"] = path.name
    return df


def clean_int(value, allowed):
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return "invalid"
    return int(number) if number.is_integer() and int(number) in allowed else "invalid"


def validate(filled: list[pd.DataFrame], key: pd.DataFrame, allow_incomplete: bool) -> pd.DataFrame:
    problems, merged, seen = [], [], {}
    key_by_code = key.set_index(["code", "reviewer"])
    for df in filled:
        name = df.file.iloc[0] if len(df) else "?"
        codes = df.code.astype(str).str.strip()
        owners = key[key.code.isin(codes)].groupby("reviewer").code.nunique()
        if owners.empty:
            problems.append(f"{name}: none of its codes are in the key")
            continue
        reviewer = owners.idxmax()
        if reviewer in seen:
            problems.append(f"{name}: same reviewer ({reviewer}) as {seen[reviewer]}")
            continue
        seen[reviewer] = name
        expected = set(key.loc[key.reviewer == reviewer, "code"])
        unknown = sorted(set(codes) - expected)
        absent = sorted(expected - set(codes))
        duplicated = sorted(codes[codes.duplicated()].unique())
        if unknown:
            problems.append(f"{name} ({reviewer}): {len(unknown)} codes not in this reviewer's key, e.g. {unknown[:3]}")
        if absent:
            problems.append(f"{name} ({reviewer}): {len(absent)} rows missing, e.g. {absent[:3]}")
        if duplicated:
            problems.append(f"{name} ({reviewer}): duplicated codes {duplicated[:3]}")
        df = df.assign(code=codes, reviewer=reviewer)
        df = df[df.code.isin(expected)].drop_duplicates("code")
        df["adequacy"] = df.adequacy.map(lambda v: clean_int(v, SCORE_VALUES))
        df["fluency"] = df.fluency.map(lambda v: clean_int(v, SCORE_VALUES))
        df["rank"] = df["rank"].map(lambda v: clean_int(v, RANK_VALUES))
        df["issue"] = df.issue.map(lambda v: None if v is None or str(v).strip() in ("", "nan") else str(v).strip())
        df["comment"] = df.comment.map(lambda v: "" if v is None or str(v) == "nan" else str(v))
        for col in ("adequacy", "fluency", "rank"):
            bad = df[df[col].eq("invalid")]
            if len(bad):
                where = [f"item {i} option {o}" for i, o in zip(bad.item_no, bad.option)][:5]
                problems.append(f"{name} ({reviewer}): invalid {col} in {len(bad)} rows: {where}")
        bad_issue = df[df.issue.notna() & ~df.issue.isin(ISSUES)]
        if len(bad_issue):
            problems.append(f"{name} ({reviewer}): unknown issue values {sorted(bad_issue.issue.unique())[:3]}")
        merged.append(df)
    if not merged:
        raise ReviewError("No usable files.\n  - " + "\n  - ".join(problems))
    long = pd.concat(merged, ignore_index=True)[["file", "reviewer", "code", "adequacy", "fluency", "rank", "issue", "comment"]]
    long = long.merge(key, on=["code", "reviewer"], how="left", validate="one_to_one")
    for col in ("adequacy", "fluency", "rank"):
        long.loc[long[col].eq("invalid"), col] = None
    complete_row = long[["adequacy", "fluency", "rank"]].notna().all(axis=1)
    prompt_complete = complete_row.groupby([long.reviewer, long.prompt_id]).transform("all")
    incomplete = long.loc[~prompt_complete, ["reviewer", "prompt_id"]].drop_duplicates()
    if len(incomplete):
        msg = f"{len(incomplete)} prompts not fully rated (adequacy, fluency and rank needed for all three options)"
        items = long.loc[~prompt_complete].drop_duplicates(["reviewer", "prompt_id"])
        by_reviewer = {r: f"{len(g)} prompts, items {sorted(g.item_no.astype(int))[:10]}" for r, g in items.groupby("reviewer")}
        if allow_incomplete:
            print(f"WARNING: {msg}; skipped: {by_reviewer}")
        else:
            problems.append(f"{msg}: {by_reviewer}. Use --allow-incomplete to score the rest.")
    if problems:
        raise ReviewError("Returned files have problems:\n  - " + "\n  - ".join(problems))
    long = long[prompt_complete].copy()
    for col in ("adequacy", "fluency", "rank"):
        long[col] = long[col].astype(int)
    long["issue"] = long.issue.fillna("none")
    best_rank = long.groupby(["reviewer", "prompt_id"])["rank"].transform("min")
    long["is_best"] = long["rank"].eq(best_rank)
    long["critical"] = long.issue.isin(CRITICAL_ISSUES)
    return long


def krippendorff_alpha(units: dict, level: str = "ordinal") -> float:
    """Krippendorff's alpha from {unit: [values from different coders]}; units with < 2 values are ignored."""
    pairable = {u: v for u, v in units.items() if len(v) >= 2}
    values = sorted({x for v in pairable.values() for x in v})
    if len(values) < 2:
        return float("nan")
    index = {x: i for i, x in enumerate(values)}
    o = np.zeros((len(values), len(values)))
    for v in pairable.values():
        m = len(v)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[index[v[i]], index[v[j]]] += 1 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    delta = np.zeros_like(o)
    for c in range(len(values)):
        for k in range(len(values)):
            if level == "interval":
                delta[c, k] = (values[c] - values[k]) ** 2
            else:
                lo, hi = min(c, k), max(c, k)
                delta[c, k] = (n_c[lo:hi + 1].sum() - (n_c[c] + n_c[k]) / 2) ** 2
    expected = (np.outer(n_c, n_c) * delta).sum()
    return float(1 - (n - 1) * (o * delta).sum() / expected) if expected else float("nan")


def cluster_bootstrap(per_record: pd.DataFrame, columns, n_boot: int, seed: int):
    """Mean and 95% interval of record-level values, resampling records."""
    rng = np.random.default_rng(seed)
    values = per_record[list(columns)].to_numpy(dtype=float)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.nanmean(values[idx], axis=1)
    return values.mean(axis=0) if not np.isnan(values).any() else np.nanmean(values, axis=0), \
        np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)


def score(long: pd.DataFrame, n_boot: int, seed: int) -> dict:
    systems = [s for s in SYSTEMS if s in set(long.system)]
    # one value per record and system: average over the reviewers who saw it
    per_rec = long.groupby(["record_id", "system"]).agg(
        adequacy=("adequacy", "mean"), fluency=("fluency", "mean"), rank=("rank", "mean"),
        best=("is_best", "mean"), critical=("critical", "mean")).unstack("system")
    system_rows = []
    for metric in ("adequacy", "fluency", "rank", "best", "critical"):
        mean, lo, hi = cluster_bootstrap(per_rec[metric], systems, n_boot, seed)
        for s, m, a, b in zip(systems, mean, lo, hi):
            system_rows.append({"metric": metric, "system": s, "mean": float(m), "ci_low": float(a), "ci_high": float(b)})
    system_table = pd.DataFrame(system_rows)

    # pairwise preference per (reviewer, prompt): +1 if a ranked better than b, -1 if worse, 0 if tied
    wide_rank = long.pivot_table(index=["reviewer", "record_id"], columns="system", values="rank")
    wide_adeq = long.pivot_table(index=["reviewer", "record_id"], columns="system", values="adequacy")
    pair_rows = []
    for a, b in combinations(systems, 2):
        pref = np.sign(wide_rank[b] - wide_rank[a]).groupby(level="record_id").mean().to_frame("pref")
        adeq = (wide_adeq[a] - wide_adeq[b]).groupby(level="record_id").mean().to_frame("adeq")
        both = pref.join(adeq)
        (pm, am), (plo, alo), (phi, ahi) = cluster_bootstrap(both, ["pref", "adeq"], n_boot, seed)
        raw = np.sign(wide_rank[b] - wide_rank[a])
        pair_rows.append({"a": a, "b": b, "records": int(len(both)),
                          "preference_a_over_b": float(pm), "pref_ci_low": float(plo), "pref_ci_high": float(phi),
                          "pref_interval_excludes_0": bool(plo > 0 or phi < 0),
                          "a_ranked_better": float((raw > 0).mean()), "b_ranked_better": float((raw < 0).mean()),
                          "tied": float((raw == 0).mean()),
                          "adequacy_a_minus_b": float(am), "adeq_ci_low": float(alo), "adeq_ci_high": float(ahi)})
    pair_table = pd.DataFrame(pair_rows)

    issues = (long.groupby("system").issue.value_counts(normalize=True).unstack(fill_value=0)
              .reindex(columns=list(ISSUES), fill_value=0).reindex(systems))

    shared = long[long.block == "shared"]
    agreement = {"reviewers": int(long.reviewer.nunique()), "shared_prompts": int(shared.record_id.nunique())}
    if shared.reviewer.nunique() >= 2:
        for col in ("adequacy", "fluency", "rank"):
            units = shared.groupby(["record_id", "system"])[col].apply(list).to_dict()
            agreement[f"alpha_{col}"] = krippendorff_alpha(units, "ordinal")
        best = shared[shared.is_best].groupby(["record_id", "reviewer"]).system.apply(frozenset)
        same_best = best.groupby(level="record_id").agg(lambda sets: sets.nunique() == 1)
        agreement["all_reviewers_same_best"] = float(same_best.mean())
    else:
        agreement.update({"alpha_adequacy": float("nan"), "alpha_fluency": float("nan"), "alpha_rank": float("nan")})

    best_share = system_table[system_table.metric == "best"].set_index("system")["mean"]
    adequacy = system_table[system_table.metric == "adequacy"].set_index("system")["mean"]
    critical = system_table[system_table.metric == "critical"].set_index("system")["mean"]
    leader = max(systems, key=lambda s: (best_share[s], adequacy[s]))
    separated, tied = [], []
    for other in systems:
        if other == leader:
            continue
        row = pair_table[((pair_table.a == leader) & (pair_table.b == other)) | ((pair_table.a == other) & (pair_table.b == leader))].iloc[0]
        sign = 1 if row.a == leader else -1
        clear = (row.pref_ci_low > 0) if sign == 1 else (row.pref_ci_high < 0)
        (separated if clear else tied).append(other)
    alpha = agreement.get("alpha_adequacy", float("nan"))
    level = agreement_level(alpha)
    agreement["level"] = level
    if level == "insufficient":
        decision = {"status": "recalibrate", "text": f"Adequacy agreement alpha = {alpha:.2f} < {ALPHA_RECALIBRATE}: "
                    "reviewers apply the scale differently. Discuss the most disputed shared prompts, clarify the "
                    "instructions and re-rate the shared block before using these results."}
    elif not tied:
        decision = {"status": "winner", "winner": leader,
                    "text": f"{leader} is ranked best most often ({best_share[leader]:.0%}) and is preferred over every "
                            "other model by more than the bootstrap noise."}
    else:
        group = [leader] + tied
        by_critical = sorted(group, key=lambda s: critical[s])
        decision = {"status": "tie", "tied": group, "fewest_critical_issues": by_critical[0],
                    "text": f"No clear winner: {', '.join(group)} are not separated on preference. Critical-issue rate: "
                            + ", ".join(f"{s} {critical[s]:.1%}" for s in by_critical)
                            + f". Rule 3: prefer {by_critical[0]} unless the team sees a reason otherwise."}
    decision["agreement_level"] = level
    decision["alpha_adequacy"] = None if np.isnan(alpha) else round(float(alpha), 4)
    if level == "tentative":
        decision["text"] += (f" Agreement is tentative (adequacy alpha {alpha:.2f}, between {ALPHA_RECALIBRATE} and "
                             f"{ALPHA_RELIABLE}): report adequacy scores and critical-issue rates"
                             + (", including this tie-break," if decision["status"] == "tie" else "")
                             + " as tentative.")
    elif level == "reliable":
        decision["text"] += f" Agreement is reliable (adequacy alpha {alpha:.2f} >= {ALPHA_RELIABLE})."
    elif level == "not computed":
        decision["text"] += " Agreement could not be computed (fewer than two reviewers on the shared block)."
    return {"systems": systems, "system_table": system_table, "pair_table": pair_table, "issues": issues,
            "agreement": agreement, "decision": decision, "leader": leader}


def main(argv=None):
    args = parse_args(argv)
    key_dir = Path(args.key_dir)
    out = Path(args.out) if args.out else key_dir
    key = pd.read_csv(key_dir / "review_key.csv")
    try:
        long = validate([read_filled(Path(f)) for f in args.files], key, args.allow_incomplete)
    except ReviewError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    result = score(long, args.bootstrap, args.seed)
    reviewers = sorted(long.reviewer.unique())
    not_scored = sorted(set(key.reviewer) - set(reviewers))
    # with few reviewers the pooled result can be driven by one person: report each reviewer on their own too
    per_reviewer = {r: score(long[long.reviewer == r], args.bootstrap, args.seed) for r in reviewers} \
        if len(reviewers) > 1 else {}
    out.mkdir(parents=True, exist_ok=True)
    cols = ["reviewer", "prompt_id", "record_id", "source", "block", "system", "code", "adequacy", "fluency", "rank",
            "is_best", "issue", "critical", "comment", "identical_to_options"]
    long[cols].sort_values(["record_id", "reviewer", "system"]).to_csv(out / "review_results_long.csv", index=False,
                                                                       encoding="utf-8")
    summary = {
        "files": [str(f) for f in args.files],
        "reviewers_scored": reviewers,
        "reviewers_not_scored": not_scored,
        "ratings": int(len(long)),
        "prompts_scored": int(long.groupby(["reviewer", "prompt_id"]).ngroups),
        "records_scored": int(long.record_id.nunique()),
        "agreement": result["agreement"],
        "system_scores": result["system_table"].to_dict(orient="records"),
        "pairwise": result["pair_table"].to_dict(orient="records"),
        "issue_shares": result["issues"].round(4).to_dict(orient="index"),
        "per_reviewer": {r: {"system_scores": res["system_table"].to_dict(orient="records"),
                             "pairwise": res["pair_table"].to_dict(orient="records")}
                         for r, res in per_reviewer.items()},
        "decision": result["decision"],
        "agreement_thresholds": {"recalibrate_below": ALPHA_RECALIBRATE, "reliable_from": ALPHA_RELIABLE,
                                 "statistic": "Krippendorff's alpha (ordinal), adequacy, shared block"},
        "bootstrap": {"resamples": args.bootstrap, "seed": args.seed, "unit": "record"},
    }
    (out / "review_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float),
                                              encoding="utf-8")

    pd.set_option("display.width", 160)
    print(f"Scored {summary['ratings']} ratings: {summary['prompts_scored']} prompt reviews of "
          f"{summary['records_scored']} records by {result['agreement']['reviewers']} reviewers ({', '.join(reviewers)})\n")
    if not_scored:
        print(f"NOTE: reviewers in the key but not scored: {', '.join(not_scored)}; their own prompts are not covered "
              "and agreement uses the scored reviewers only.\n")
    table = result["system_table"].pivot(index="system", columns="metric", values="mean").reindex(result["systems"])
    table = table[["best", "rank", "adequacy", "fluency", "critical"]].rename(columns={
        "best": "ranked best", "rank": "mean rank", "critical": "critical issue"})
    print(table.round(3).to_string(), "\n")
    print(result["pair_table"][["a", "b", "preference_a_over_b", "pref_ci_low", "pref_ci_high",
                                "a_ranked_better", "b_ranked_better", "tied"]].round(3).to_string(index=False), "\n")
    if per_reviewer:
        rows = [dict(reviewer=r, **row) for r, res in per_reviewer.items() for row in res["pair_table"].to_dict("records")]
        print("Per reviewer (all of their prompts):")
        print(pd.DataFrame(rows)[["reviewer", "a", "b", "records", "preference_a_over_b", "pref_ci_low", "pref_ci_high",
                                  "tied"]].round(3).to_string(index=False), "\n")
    ag = result["agreement"]
    print("Agreement on shared block:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ag.items()}, "\n")
    print(f"Agreement level: {ag['level']} (adequacy alpha {ag.get('alpha_adequacy', float('nan')):.3f}; "
          f"recalibrate < {ALPHA_RECALIBRATE} <= tentative < {ALPHA_RELIABLE} <= reliable)\n")
    print(f"DECISION ({result['decision']['status']}): {result['decision']['text']}")
    print(f"\nSaved: {out / 'review_results_long.csv'}, {out / 'review_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
