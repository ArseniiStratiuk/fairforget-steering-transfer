# Blinded translation review (Gemma / Lapa / Claude)

COMET (notebook 11) cannot separate the three Ukrainian translations of the 620 AdvBench/JBB prompts, so the choice is made by people. These scripts build blinded review files, map the returned ratings back to the models, and apply a decision rule fixed in advance.

Requirements: `pip install pandas numpy openpyxl` (plus `huggingface_hub` if the Claude file is read from the Hugging Face cache).

## Design

- **3 reviewers × 200 prompts.** 90 randomly chosen prompts are shared by everyone (agreement check), 110 per reviewer are their own. 420 of the 620 records are reviewed; the other 200 are listed in the manifest. 90 shared was chosen by simulation: 60 left the agreement estimate too uncertain around the 0.4 gate, while 120 added little precision and cost a quarter of the coverage.
- **Side by side.** Each prompt shows the English text and its three translations as options A/B/C. Each reviewer rates 200 translations from every model.
- **Blind.** Option order is shuffled and balanced (each model sits in each position ~67 times per reviewer). Reviewers see only neutral IDs (`P-XXXX` per prompt, `T-XXXXXX` per translation). The mapping to `record_id` and model is only in `key/`.
- **Fair split.** Records with identical English text stay in the same block; every block keeps the JBB/AdvBench mix (16–17% JBB).
- **Reproducible.** Fixed seed; the key cannot be overwritten without `--force`.

## Workflow

```bash
# 1. Make the files (once; use real reviewer names). Refuses to overwrite an existing key.
python src/translation_review/make_review_split.py --reviewers anna bohdan ivan
#    send:  data/safety/review/for_reviewers/review_<name>.xlsx  (one per person)
#    keep:  data/safety/review/key/  (SECRET, never send)

# 2. Collect the filled files, e.g. into data/safety/review/returned/, then score:
python src/translation_review/score_review.py data/safety/review/returned/*.xlsx
#    -> data/safety/review/key/review_results_long.csv, review_summary.json, printed decision
#    Problems (missing rows, edited codes, invalid values, a file twice) are listed with item numbers.
#    --allow-incomplete scores fully rated prompts only.

# 3. If agreement is insufficient (decision "recalibrate"): blinded side-by-side sheet of the disputed shared prompts
python src/translation_review/make_recalibration_sheet.py data/safety/review/returned/*.xlsx
#    -> data/safety/review/recalibration/recalibration_shared.xlsx (no model names; shows reviewers' ratings to each
#       other, so only for people who have finished the shared block)

# Tests
python src/translation_review/test_review_pipeline.py --mock                     # synthetic data, all checks
python src/translation_review/test_review_pipeline.py --real data/safety/review  # checks the real split, dry-runs scoring
```

## Reviewer scales

| Column | Values |
|---|---|
| adequacy | 4 same meaning · 3 minor nuance lost · 2 important part wrong/missing · 1 different request |
| fluency | 4 natural, right register · 3 slightly awkward/formal · 2 clumsy/calque · 1 hard to read |
| rank | 1 best … 3 worst within the prompt; ties allowed |
| issue | none, meaning changed, content added, content dropped, formal register, untranslated words, grammar or fluency, name or term error, other |

## Decision rule (fixed before the review)

1. **Agreement level**, from Krippendorff's alpha (ordinal) for adequacy on the 90 shared prompts:

   | alpha | level | consequence |
   |---|---|---|
   | < 0.4 | insufficient | no decision: discuss the most disputed shared prompts, clarify the instructions, re-rate the shared block |
   | 0.4 – 0.667 | tentative | decide, but report adequacy scores and critical-issue rates (and a tie-break based on them) as tentative |
   | ≥ 0.667 | reliable | Krippendorff's minimum for tentative conclusions is met |

   0.667 is Krippendorff's own lowest acceptable level; 0.4 is a lenient floor for subjective translation ratings, where ratings concentrated on 3–4 depress alpha. Both thresholds are fixed before the review and must not be changed after seeing the results.
2. The model ranked best most often wins if its paired preference over each other model has a 95% bootstrap interval (resampling records) that excludes 0.
3. Otherwise the unseparated models are a tie; prefer the one with the fewest critical issues (meaning changed / content added / content dropped), unless the team sees a reason otherwise.
