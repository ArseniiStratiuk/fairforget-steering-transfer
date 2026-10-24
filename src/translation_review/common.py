"""Shared constants and helpers for the blinded translation review."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SYSTEMS = ("gemma", "lapa", "claude")
OPTIONS = ("A", "B", "C")
SCORE_VALUES = (1, 2, 3, 4)
RANK_VALUES = (1, 2, 3)
ISSUES = (
    "none",
    "meaning changed",
    "content added",
    "content dropped",
    "formal register",
    "untranslated words",
    "grammar or fluency",
    "name or term error",
    "other",
)
CRITICAL_ISSUES = ("meaning changed", "content added", "content dropped")
REVIEW_SHEET = "Review"
INPUT_COLUMNS = ("adequacy", "fluency", "rank", "issue", "comment")
REVIEWER_COLUMNS = (
    "item_no",
    "prompt_id",
    "option",
    "english_prompt",
    "ukrainian_translation",
    "adequacy",
    "fluency",
    "rank",
    "issue",
    "comment",
    "code",
)
CLAUDE_CANDIDATES = ("data/safety/translated_formated.csv", "data/safety/prompt-pairs.csv")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalise_goal(text: str) -> str:
    return " ".join(str(text).lower().split())


def resolve_claude_csv(root: Path, explicit: str | None, offline: bool = False) -> Path:
    """Local copy of the pinned prompt-pairs.csv, else the pinned revision from the Hugging Face cache/Hub."""
    if explicit:
        return Path(explicit)
    for candidate in CLAUDE_CANDIDATES:
        if (root / candidate).is_file():
            return root / candidate
    pinned = json.loads((root / "configs/artifacts.json").read_text())["prompt_pairs"]
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            pinned["repo_id"],
            pinned["filename"],
            repo_type="dataset",
            revision=pinned["revision"],
            local_files_only=offline,
        )
    )


def load_translations(gemma: Path, lapa: Path, claude: Path, artifacts: Path | None) -> pd.DataFrame:
    """One row per record_id with the English goal and the three Ukrainian translations.

    Checks that the three files describe the same records with the same English text, and (unless
    `artifacts` is None) that the Claude file is the pinned prompt-pairs.csv.
    """
    if artifacts is not None:
        expected = json.loads(Path(artifacts).read_text())["prompt_pairs"]["sha256"]
        actual = sha256(claude)
        if actual != expected:
            raise SystemExit(f"{claude} has SHA256 {actual}, not the pinned prompt-pairs.csv ({expected}).")
    frames = {}
    for name, path in (("gemma", gemma), ("lapa", lapa)):
        df = pd.read_csv(path, keep_default_na=False)
        if not df.record_id.is_unique:
            raise SystemExit(f"{path}: record_id is not unique")
        bad = df[df.status != "ok"]
        if len(bad):
            raise SystemExit(f"{path}: {len(bad)} rows have status != ok; fix or exclude them first: {bad.record_id.tolist()[:10]}")
        frames[name] = df.set_index("record_id")
    claude_df = pd.read_csv(claude, keep_default_na=False)
    if not claude_df.record_id.is_unique:
        raise SystemExit(f"{claude}: record_id is not unique")
    claude_df = claude_df.set_index("record_id")
    g, l = frames["gemma"], frames["lapa"]
    if not (set(g.index) == set(l.index) == set(claude_df.index)):
        raise SystemExit("Gemma, Lapa and Claude files cover different record_ids")
    claude_df = claude_df.reindex(g.index)
    l = l.reindex(g.index)
    for name, other in (("lapa", l), ("claude", claude_df)):
        mismatch = (other.goal.str.strip() != g.goal.str.strip()).sum()
        if mismatch:
            raise SystemExit(f"{mismatch} English goals differ between gemma and {name}")
    data = pd.DataFrame(
        {
            "source": g.source,
            "goal": g.goal,
            "uk_gemma": g.ukrainian_prompt,
            "uk_lapa": l.ukrainian_prompt,
            "uk_claude": claude_df.ukrainian_prompt,
        }
    ).sort_index()
    empty = [(rid, s) for s in SYSTEMS for rid in data.index[data[f"uk_{s}"].str.strip().eq("")]]
    if empty:
        raise SystemExit(f"Empty translations: {empty[:10]}")
    return data
