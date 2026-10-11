# Gemma-Lapa steering transfer

A UCU Artificial Intelligence course project testing whether a refusal direction extracted from Gemma 3 transfers to its Ukrainian adaptation, Lapa. Measure response safety, refusal, harmful compliance and reading-comprehension utility before and after intervention.

## Current results

A Gemma-derived activation direction is removed from Gemma and reused unchanged on Lapa. The control removes one saved random direction at the same sites. Model weights are unchanged.

| Model | English harmful: baseline | Direction removed | Random removed |
|---|---:|---:|---:|
| Gemma 3-12B-IT | 2/100 | 24/100 | 2/100 |
| Lapa 0.1.2 Instruct | 72/100 | 82/100 | 75/100 |

Counts are Llama Guard 4 unsafe-content verdicts, not validated attack-success or refusal rates. Notebook 09 includes paired changes, capped answers, seeded examples of observed refusal changes and judge disagreements. Ukrainian scores are exploratory. No independent human labels are included.

| Model | Belebele EN: baseline / direction / random | Belebele UK: baseline / direction / random |
|---|---|---|
| Gemma | 844 / 843 / 842 | 813 / 812 / 814 |
| Lapa | 826 / 826 / 824 | 797 / 798 / 792 |

Each utility count is out of 900. Small differences on this benchmark do not establish preservation of every capability. The candidate was selected on a separate 8+8 validation set with a fixed harmless-KL limit; neither JBB nor Belebele was used for tuning.

The corrected safety comparison uses explicit greedy decoding, batch 8 and a 2,048-token cap: 1,200 answers per model, all scored. Historical safety baselines and validation answers used model-default sampling because Transformers overrode the intended generation config. They are preserved separately and labeled as sampled observations. Forward-only activation capture, KL and utility scoring are unaffected.

## Shared artifacts

Private HF datasets require membership in [steering-transfer](https://huggingface.co/steering-transfer) and your own HF login:

- [prompt-pairs](https://huggingface.co/datasets/steering-transfer/prompt-pairs): 620 EN/UK harmful prompt pairs and provenance.
- [model-answers](https://huggingface.co/datasets/steering-transfer/model-answers): historical baselines, corrected intervention answers, utility predictions and metadata.
- [safety-scores](https://huggingface.co/datasets/steering-transfer/safety-scores): historical and corrected Llama Guard 4 verdicts and inference metadata.
- [refusal-directions](https://huggingface.co/steering-transfer/refusal-directions/tree/f88ec00c0e12beb990d9f672a6ae18f872f628f2/experiments/2026-10-11): frozen candidate, random vector and outlier mask; no model weights.

[Full Gemma answers](https://huggingface.co/datasets/steering-transfer/model-answers/blob/d011197e91589922d1f4ca48828dcb19524a2ee1/experiments/2026-10-11/results/transfer/frozen-transfer-v2/gemma/2273125/responses.jsonl) and [full Lapa answers](https://huggingface.co/datasets/steering-transfer/model-answers/blob/d011197e91589922d1f4ca48828dcb19524a2ee1/experiments/2026-10-11/results/transfer/frozen-transfer-v2/lapa/2273126/responses.jsonl) are available at immutable revisions.

`configs/artifacts.json` pins revisions and checksums. Inputs download to the local HF cache. Models, datasets, bulk answers, logs, credential files and private team discussions stay outside Git. Source and license notes are in [data/README.md](data/README.md).

## Notebooks

| Notebook | Purpose | Runtime |
|---|---|---|
| 00_remote_runtime | Recorded runtime and optional live GPU check | Local; live check on allocated GPU |
| 01_split_review | Prompt overlap, draft split and harmless candidate review | Local |
| 02_belebele | Parallel utility dataset EDA | Local |
| 03_belebele_baseline | Generate utility predictions | Slurm GPU |
| 04_safety_baseline | Generate safety responses | Slurm GPU |
| 05_safety_results | Inspect safety answers and truncation | Local |
| 06_belebele_results | Inspect utility accuracy and errors | Local |
| 07_baseline_scoring | Exact utility scores, Guard judging and manual safety review | Local; judging opt-in in Slurm |
| 08_gemma_ablation | Frozen extraction, screening, historical validation and utility | Local analysis; GPU inference opt-in |
| 09_frozen_transfer | Corrected Gemma/Lapa comparison, random control and response examples | Local analysis; GPU inference opt-in |
| 10_persona_prefix | Prefix pilot inputs and actual 35/36 saved answers | Local analysis; GPU inference opt-in |

03/04 retain genuine outputs from one historical Gemma worker; they are not newly executed locally. 05-07 aggregate the historical baselines. 08 retains saved GPU captures while its local analysis recomputes tables. 09 analyzes the completed corrected experiment. 10 reports partial coverage without a measured persona effect. Notebook 11 and its persona module are outside this release.

The frozen pilot uses 32 harmful and 32 harmless English extraction prompts, plus 8 validation prompts per class. Notebook 01 explores a larger split; notebook 08 records the actual smaller split. JBB prompts were reserved, although related topics occur across sources.

The experiment release is published. `configs/reviewed_experiments.json` pins all three immutable HF revisions and 34 file checksums. Its loader verifies existing files and downloads missing artifacts. Historical HF snapshots remain unchanged.

## Local setup and reproduction

Python 3.13. Local notebooks were executed on Debian with the versions in `requirements.txt` (the installed pandas build is `2.2.3+dfsg`). GPU packages are separate.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/hf auth login
.venv/bin/python -m ipykernel install --user --name steering-local --display-name 'Steering local'
```

Open notebooks in VS Code, select **Steering local**, and Run All. To reproduce the main safety/control and utility comparison tables from saved artifacts, run this command from the repository root:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path('notebooks/09_frozen_transfer.ipynb')
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, kernel_name='steering-local', timeout=300,
               resources={'metadata': {'path': str(Path.cwd())}}).execute()
nbformat.write(notebook, path)
PY
```

This recomputes scores from recorded predictions. New model inference requires JUPITER. Notebook 09 samples observed refusal-change cases with seed 42, accounts for ambiguous forward verdict changes, and shows baseline/candidate/random answers side by side. This conditional sample illustrates the effect; it cannot estimate an overall success rate.

## JUPITER inference

Budget `e-dev-2026d09-075`, ARM64, GH200. Project directory: `/e/project1/e-dev-2026d09-075/fairforget-steering-transfer/`. Launchers use `Stages/2026 PyTorch/2.9.1 JupyterLab/4.5.5` and the existing `fairforget/envs/lapa-py313` environment: PyTorch 2.9.1+cu128, Transformers 4.57.6, pandas 2.3.1 and nbclient 0.10.2. Model paths and immutable revisions are explicit in the GPU notebooks.

Cache pinned HF inputs under the submitting JUPITER account before GPU jobs: launchers use offline mode. A CPU-only local run of 01, 02 and 07 on JUPITER can prepare those input caches; Guard imports are disabled unless `RUN_GUARD=1`. Each notebook's GPU cell requires the shared pinned checkpoint to exist. `slurm/activation-probe.sbatch` runs the two-prompt Gemma activation check from notebook 08. Add `--export=ALL,RUN_FULL_EXTRACTION=1` to also capture all 64 extraction prompts and compute 48 unit directions. Tensors and the executed notebook are saved under `results/activations/`. For validation screening, use `--export=ALL,RUN_ACTIVATION_PROBE=0,RUN_VALIDATION=1`; outputs go under `results/validation/`. The first screening run found no direction below the fixed harmless-KL limit and stopped before generation. The outlier-masked variant passed screening and selected source layer 32. Set `RUN_GUARD=1` with activation/validation flags disabled to score its saved responses; set `RUN_BELEBELE=1` with those flags disabled to compare the fixed candidate against the unchanged model on all 1,800 utility questions. Pilot outputs currently reside in local and JUPITER project storage.

From the project directory, create `logs/`, then submit:

```bash
mkdir -p logs
sbatch slurm/baselines.sbatch
sbatch slurm/batched.sbatch
sbatch slurm/guard.sbatch
```

Defaults are small runtime probes. Full runs require explicit limits and adequate wall time:

```bash
sbatch --time=06:00:00 --export=ALL,ITEM_LIMIT=0 slurm/baselines.sbatch
sbatch --time=04:00:00 --export=ALL,ITEM_LIMIT=0 slurm/batched.sbatch
sbatch --time=00:40:00 --export=ALL,GUARD_LIMIT=0 slurm/guard.sbatch
```

`baselines.sbatch` generates Belebele and safety answers; `batched.sbatch` generates only safety answers. Choose one safety path per experiment to avoid duplicate work. Existing outputs are not overwritten; preserve completed Guard results before attempting a new judge run. Guard's full dynamic cache and bounded attention-window workaround for Transformers 4.57.6 are explained and recorded in 07.

## Team contributions

- **Arsenii Stratiuk:** compute/model setup, utility and safety baseline execution, scoring, artifact storage and pilot direction extraction; activation extraction, interventions and frozen transfer evaluation.
- **Ivan Maksymchuk:** prompt preparation, overlap analysis, EN/UK translations and annotation preparation; benign JBB translations.

Contributions will be updated as reviewed work is completed. The course contribution studies refusal-direction transfer and bilingual evaluation within the broader FairForget project.

## AI usage

Codex assists with infrastructure, source inspection, notebook implementation, scoring and explanations. Notebook outputs are checked from fresh kernels locally or completed Slurm workers; model/input revisions and real runtime observations are recorded. Team members review scientific choices and results. Translation-generator metadata and human review status remain explicit; model judging does not replace manual audit.

## Research sources

- [Refusal direction implementation](https://github.com/andyrdt/refusal_direction)
- [Activation-space intervention transfer](https://proceedings.mlr.press/v267/oozeer25a.html)
- [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
- [Belebele](https://huggingface.co/datasets/facebook/belebele)
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [Llama Guard 4](https://huggingface.co/meta-llama/Llama-Guard-4-12B)

Repository structure follows the adaptable [UCU course example](https://github.com/ucu-ai-course/project-template). Dataset and model terms apply separately from code licenses.
