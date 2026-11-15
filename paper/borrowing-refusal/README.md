# Borrowing refusal from sibling models: code

Code, result tables and selection evidence for the paper "Borrowing Refusal From Sibling Models"
(Arsenii Stratiuk, Ivan Maksymchuk, Viktoriia Makovska; Ukrainian Catholic University). This folder is
the self-contained snapshot used for the paper, including its own copy of the `steering_transfer`
package; the rest of the repository holds the course project around it.
A refusal direction is built and calibrated in a donor model (Gemma-3-12B-IT or
MamayLM-Gemma-3-12B-IT-v2.0) and added to, or removed from, Lapa v0.1.2 Instruct without
recipient data, training or mappings.

## Contents

- `steering_transfer/`: the package used for every run (activation capture, direction
  construction, interventions, generation, judging, comparisons).
- `configs/experiments/borrowing_refusal.json`: models and pinned revisions, data split,
  layer and strength selection, controls, generation and judge settings.
- `configs/rubrics/`: judge label definitions.
- `scripts/`:
  - `refine_strength.py`, `run_refinement.py`, `run_native_controls.py`, `run_refined_controls.py`,
    `run_source_tasks.py`: calibration and validation of each direction in the model that owns it.
  - `run_lapa_main.py`: answers of Lapa to the test split and Belebele under every condition,
    in four shards per condition set (`frozen`, `frozen_removal`, `frozen_addition`, `frozen_controls`, `system_prompt`; the
    exploratory prompt-only sets `addition*`).
  - `score_lapa_main.py`, `score_guard.py`: Jev and Llama Guard 4 labels for generated answers.
  - `borrowing_tables.py`: rates, paired changes and 95% bootstrap intervals.
  - `screen_*`, `judge_*_screen.py`, `validate_v2.py`, `shortlist_v2.py`, `select_v3.py`,
    `run_addition_arm.py`: the exploratory prompt-only addition (appendix).

- `scripts/donor_baseline.py`: unmodified donor answers on the test split (E0).
- `scripts/system_prompt_and_guard_summary.py`: safety system-prompt rates with paired intervals and Llama Guard 4 unsafe rates per condition (output in `results/tables/system_prompt_and_guard_summary.txt`).
- `results/judge/guard_reference/labels.jsonl`: Llama Guard 4 verdicts on the 120 human-labeled answers; `results/judge/hand_check_steered_answers.csv`: two annotators, agreed and Jev labels for 28 steered answers; `scripts/donor_baselines_and_guard_reference.py` and `results/tables/donor_baselines_and_guard_reference.txt`: unmodified donor rates (E0) and Guard agreement.
- `figures/`: Figure 1 and Table 2 of the paper as images, with their standalone LaTeX sources.
- `launchers/`: Slurm launchers as run (one node with four GPUs, or one GPU per shard).
- `results/tables/`: the numbers behind the paper's tables: rates, paired changes from the
  unmodified model and direct borrowed-minus-reference differences, with 95% bootstrap
  intervals (`frozen_*` is the main arm; `prompt_first_*` and `addition_v2_*` are exploratory),
  and answer coverage per condition.
- `results/selection/`: validation in the model that owns each direction (layer screens, strength
  trials, selected settings, source validation against no intervention and a random direction),
  and the selection files of the exploratory prompt-only arm.
- `results/judge/`: the selected judge configuration and the comparison of four judges on
  90 labeled answers.

## Supporting evidence for figures in the paper

- `results/data/`: split manifest (counts, seed, duplicate policy), near-duplicate groups (IDs only) and the
  provenance of the harmful-request merge (cosine threshold 0.92) and of the reference Ukrainian translations.
- `results/translation/`: blind translation review summary (ratings, Krippendorff's alpha, preferences, issue
  shares), CometKiwi summary, the 24 corrections applied after review, and the benign JBB review counts.
- `results/judge/human_labels.csv`, `results/judge/judge_predictions.jsonl`: the 120 human reference labels
  (90 comparison, 30 development) and every judge's labels on them, without request or answer text.
- `results/calibration_coarse/`: strength trials at k in {0.25, 0.5, 1, 2, 4} in the model that owns each direction.
- `results/all_position_middle_blocks/`: donor validation of all-position addition at middle blocks.
- `results/tables/cross_machine_baseline_agreement.txt`: unmodified Lapa answers generated on the two machines
  for the same requests: the UCU lab node (NVIDIA RTX 6000 Ada) and JUPITER Booster (GH200).

The configuration records the pre-specified plan: its strength rule (smallest qualifying multiplier) and the
`deferred` entry for the safety system prompt predate the changes reported in the paper (coherent-refusal
strength rule, Sec. 3.4; safety system prompt, Sec. 3.5). Requests, generated answers and annotations with
text are shared with researchers on approved request.

## Running

Set `BORROWING_ROOT` to a directory that contains `configs/`, the frozen data split under
`data/frozen/bilingual-refusal-v1` and model checkpoints under `models/<name>/<revision>`
(revisions in the configuration). Generation uses PyTorch 2.9.1 and Transformers 4.57.6 in
bfloat16 with greedy decoding; Jev scoring needs a TypeSafe API key in `TYPESAFE_API_KEY`.

The data split, generated answers and labels with text are in gated Hugging Face datasets of the
`steering-transfer` organisation (repository IDs and revisions in `configs/`) and are shared with
researchers on approved request. Cluster paths in this copy are relative, so the configuration hash
differs from the one recorded in our artifacts.

## License

Code: MIT (see `LICENSE`). Data and models keep their own licenses (see the paper, Ethical Considerations).
