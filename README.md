# FairForget steering transfer

A UCU Artificial Intelligence course research project within FairForget: test whether a refusal direction extracted from Gemma 3 transfers to its Ukrainian adaptation, Lapa, and measure the effects on harmful compliance, refusal, and reading-comprehension utility.

## Current status

Infrastructure setup and data preparation are in progress. No baseline or intervention measurements have been produced by this repository yet.

- Source model: `google/gemma-3-12b-it`, revision `96b6f1eccf38110c56df3a15bffe176da04bfd80`.
- Target model: `lapa-llm/lapa-v0.1.2-instruct`, revision `2969fd997f07b33727c6f10795d78ef23cc5c037`.
- Safety prompts: AdvBench and JBB-Behaviors; provenance, behavior overlap, translations, and splits are being reviewed.
- Utility: parallel English/Ukrainian Belebele items.

## Workflow

Research starts in small, explained notebooks: data inspection, baseline evaluation, direction extraction, ablation, and analysis. Reusable code will move into a Python package when needed. Slurm launchers handle remote execution on JUPITER; datasets, checkpoints, credentials, and bulk outputs stay outside Git.

`notebooks/` holds research notebooks; `slurm/` holds job launchers. These will be added as runnable work is developed. Installation and reproduction commands will accompany the first executable notebook; there is no runnable experiment yet.

## Team and contributions

Current agreed responsibilities; completed contributions will be updated as work progresses:

- **Arsenii Stratiuk:** compute setup, model loading and activation checks, direction extraction and ablation, scoring, and Belebele utility evaluation.
- **Ivan Maksymchuk:** prompt preparation and overlap checks, EN–UK translation and annotation preparation, and unmodified EN/UK safety baselines.

This course contribution focuses on refusal-direction transfer and bilingual evaluation within the broader FairForget research project.

## AI usage

Codex assists with infrastructure commands, repository setup, source inspection, and explanations. Setup is checked against actual SSH/file outputs and official documentation. Scientific notebook code and outputs will be reviewed with the team cell by cell before reported results are accepted. Translation-generation provenance and human-review outcomes will be documented with the dataset.

## Sources

- [Refusal direction implementation](https://github.com/andyrdt/refusal_direction)
- [Activation-space intervention transfer](https://proceedings.mlr.press/v267/oozeer25a.html)
- [AdvBench](https://github.com/llm-attacks/llm-attacks)
- [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)
- [Belebele](https://huggingface.co/datasets/facebook/belebele)

Repository organization is informed by the [UCU course example](https://github.com/ucu-ai-course/project-template). Dataset and model terms apply separately; data-source and license notes will be added before research use or redistribution.
