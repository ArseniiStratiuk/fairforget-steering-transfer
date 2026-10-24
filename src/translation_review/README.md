# Translation candidates and blinded review

This workflow compares Gemma, Lapa and the existing Claude translations. It does not create extraction/validation/test splits or final approved translations.

From the project root:

```bash
python -m pip install -e '.[translation-review,notebooks]'
python -m translation_review.translate_prompts
python -m translation_review.screen_translations
python src/translation_review/make_review_split.py --reviewers reviewer_1 reviewer_2 reviewer_3
python src/translation_review/score_review.py data/safety/review/returned/*.xlsx
python src/translation_review/make_recalibration_sheet.py data/safety/review/returned/*.xlsx
```

Translation generation requires a Slurm GPU allocation. `slurm/translate.sbatch` checks compute-ledger registration before execution. Register a held job before releasing it, and reconcile accounting from the login shell after completion.

COMET screening additionally requires `unbabel-comet` and its downloaded checkpoint. It provides supporting measurements, not translation approval. Candidate generation and screening save their input hashes and runtime metadata.

The review builder takes `--gemma`, `--lapa` and `--claude` input paths. Default inputs are under `data/safety/translations/` and the pinned prompt-pairs snapshot in `configs/datasets.json`. Generated files stay outside Git.

Each reviewer receives `for_reviewers/review_<name>.xlsx`. Keep `key/` private: it maps neutral option codes back to systems. Reviewers independently fill adequacy and fluency (1-4), rank (1-3, ties allowed), and an explicit issue label (`none` is a valid answer). Return original texts and identifiers unchanged.

Defaults assign 90 shared and 110 private prompts per reviewer: 420/620 unique records, leaving 200 unreviewed. These are reviewer assignments, not scientific splits. This sampling design selects a translation system; final row-level corrections and adjudication need a separate dataset release.

Scoring reports preference, adequacy, fluency, critical-issue counts and agreement. Missing reviewers, undefined agreement or agreement below the configured 0.4 gate block selection. The original 0.667 reporting threshold is retained as a configured review threshold, not a guarantee of reliable human judgments. Recalibration sheets are for reviewers who have already finished independent annotation.
