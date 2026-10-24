# Data

| Input | Local path | Source / status |
|---|---|---|
| 620 EN/UK safety prompt pairs | `safety/translated_formated.csv` | 100 JBB + 520 AdvBench; translations await human review; SHA256 `5135c91c972fd4362e9e02573a8635b8630ceba8c8f9a81848bf7b491483ccdf` |
| 100 benign EN/UK pairs and review log | Pinned in `configs/datasets.json` | JBB source plus 93 approved / 7 corrected translations in one individual pass; team approval is separate |
| Belebele EN/UK | `belebele/7899cdfa4e1e0d733fd77c848e2c273cb1d32be2/` | [facebook/belebele](https://huggingface.co/datasets/facebook/belebele), CC BY-SA 4.0; unchanged, 900 questions per language |
| Original JBB harmful/benign tables | `jbb/886acc352a31533ffbcf4ef22c744658688086fc/` | [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors), upstream dataset card declares MIT |

AdvBench source: [harmful behaviors](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv). The received export lacks a pinned AdvBench revision and independently verified translation-generator metadata. No license has been assigned to the translations; repository code licenses do not establish separate data rights.

`record_id` identifies a bilingual prompt pair. `discard` is the supplied similarity flag, not an exclusion decision; all 620 records are retained. The frozen pilot extraction and validation IDs are in `configs/experiments/refusal_transfer.json`; all JBB prompts are reserved from source selection. Translation pairs and supplied overlap groups stay together.

Inputs are research prompts, not collected personal user conversations. Original JBB fields may mention public entities; generated responses are not verified factual claims. Harmful prompts and outputs are retained for safety evaluation. Translation review must check intent and key details.

Raw files stay outside Git. Shared, private snapshots live in [prompt-pairs](https://huggingface.co/datasets/steering-transfer/prompt-pairs) and [model-answers](https://huggingface.co/datasets/steering-transfer/model-answers). Automated judge outputs and shared manual-audit IDs are in [safety-scores](https://huggingface.co/datasets/steering-transfer/safety-scores); human safety labels remain pending. `configs/artifacts.json` pins their revisions and checksums. Results notebooks load these snapshots through the local HF cache; organization membership and HF login are required. Logs and executed worker notebooks remain local and on JUPITER.

Harmless extraction candidates: Stanford Alpaca instructions at revision `f13496211289def0ff88ae673389ae14a818b4b3`, [CC BY-NC 4.0](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE). Only self-contained instructions are sampled; generated answers and model weights are not used. The 160 candidates are pending safety and overlap review, not a frozen harmless set.

## Experiment artifacts

The reviewed experiment manifest is `configs/reviewed_experiments.json`. Versioned releases preserve raw answers, next-token utility predictions, Guard verdicts, model/input revisions and checksums. Historical sampled responses and corrected greedy v2 responses remain separate. Saved vectors and the outlier mask belong in a model artifact repository, not Git. No model weights or raw activation dump is included.

Refusal-change example eligibility is AI-assisted, not an independent human annotation dataset. The conditional seed-42 sample illustrates observed changes and is not a success-rate estimator. Ukrainian verdicts and translations remain unvalidated.

Translation generation, COMET screening and blinded translation review are in `src/translation_review/`. Candidate translations and review assignments do not define scientific splits or a final ground-truth release.
