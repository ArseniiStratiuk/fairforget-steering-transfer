# Gemma-Lapa steering transfer

A UCU Artificial Intelligence course project testing whether a refusal direction extracted from Gemma 3 transfers to its Ukrainian adaptation, Lapa. Measure response safety, refusal, harmful compliance and reading-comprehension utility before and after intervention.

## Current results

Unmodified baselines: 3,600 Belebele predictions and 2,480 safety responses. Llama Guard 4 scored every safety response; manual audit remains pending. No direction extraction or intervention has been run. Split assignments and harmless extraction candidates are proposals awaiting review.

| Model | Belebele English | Belebele Ukrainian |
|---|---:|---:|
| Gemma 3-12B-IT | 844/900 (93.78%) | 813/900 (90.33%) |
| Lapa 0.1.2 Instruct | 826/900 (91.78%) | 797/900 (88.56%) |

Utility uses zero-shot, constrained next-token choice among A-D. Safety responses use greedy decoding with a 2,048-token cap: 3 Gemma and 18 Lapa responses reached the cap. Guard verdicts measure unsafe content, not exact harmful compliance or refusal; a possible false positive is identified in the scoring notebook.

## Shared artifacts

Private HF datasets require membership in [steering-transfer](https://huggingface.co/steering-transfer) and your own HF login:

- [prompt-pairs](https://huggingface.co/datasets/steering-transfer/prompt-pairs): 620 EN/UK harmful prompt pairs and provenance.
- [model-answers](https://huggingface.co/datasets/steering-transfer/model-answers): complete baseline predictions and responses.
- [safety-scores](https://huggingface.co/datasets/steering-transfer/safety-scores): complete Llama Guard 4 verdicts and inference metadata.

`configs/artifacts.json` pins revisions and checksums. Inputs download to the local HF cache. Models, datasets, bulk answers, logs, credential files and private team discussions stay outside Git. Source and license notes are in [data/README.md](data/README.md).

## Notebooks

| Notebook | Purpose | Runtime |
|---|---|---|
| 00_remote_runtime | Check a remote Jupyter GPU connection | JUPITER; connection workflow not yet validated |
| 01_split_review | Prompt overlap, draft split and harmless candidate review | Local |
| 02_belebele | Parallel utility dataset EDA | Local |
| 03_belebele_baseline | Generate utility predictions | Slurm GPU |
| 04_safety_baseline | Generate safety responses | Slurm GPU |
| 05_safety_results | Inspect safety answers and truncation | Local |
| 06_belebele_results | Inspect utility accuracy and errors | Local |
| 07_baseline_scoring | Exact utility scores, Guard judging and manual safety review | Local; judging opt-in in Slurm |

03/04 retain compact outputs from one completed Gemma worker; 05-07 combine all workers through shared artifacts. Raw responses are preserved on HF. The current 01 split proposal is not a frozen experiment manifest.

## Local setup and reproduction

Python 3.13. Local notebooks were executed on Debian with the versions in `requirements.txt` (the installed pandas build is `2.2.3+dfsg`). GPU packages are separate.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/hf auth login
.venv/bin/python -m ipykernel install --user --name steering-local --display-name 'Steering local'
```

Open notebooks in VS Code, select **Steering local**, and Run All. To reproduce the utility table and passage-bootstrap intervals from pinned predictions, run this command from the repository root:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient

path = Path('notebooks/07_baseline_scoring.ipynb')
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, kernel_name='steering-local', timeout=300,
               resources={'metadata': {'path': str(Path.cwd())}}).execute()
nbformat.write(notebook, path)
PY
```

This recomputes scores from recorded predictions. New model inference requires JUPITER. The scoring notebook shows fixed random audit examples and a possible Guard false positive as static outputs. The shared review key identifies the complete audit sample; manual labels are not yet available.

## JUPITER inference

Budget `e-dev-2026d09-075`, ARM64, GH200. Project directory: `/e/project1/e-dev-2026d09-075/fairforget-steering-transfer/`. Launchers use `Stages/2026 PyTorch/2.9.1 JupyterLab/4.5.5` and the existing `fairforget/envs/lapa-py313` environment: PyTorch 2.9.1+cu128, Transformers 4.57.6, pandas 2.3.1 and nbclient 0.10.2. Model paths and immutable revisions are explicit in the GPU notebooks.

Cache pinned HF inputs under the submitting JUPITER account before GPU jobs: launchers use offline mode. A CPU-only local run of 01, 02 and 07 on JUPITER can prepare those input caches; Guard imports are disabled unless `RUN_GUARD=1`. Each notebook's GPU cell requires the shared pinned checkpoint to exist.

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

- **Arsenii Stratiuk:** compute/model setup, utility and safety baseline execution, scoring, artifact storage; direction extraction and intervention work next.
- **Ivan Maksymchuk:** prompt preparation, overlap analysis, EN/UK translations and annotation preparation; benign JBB translations next.

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
