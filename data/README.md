# Research data

| Dataset | Use | Requests per language |
|---|---|---:|
| AdvBench + JBB, English/Ukrainian | Harmful extraction / validation / test | 128 / 64 / 339 |
| Stanford Alpaca, English/Ukrainian | Harmless extraction / validation | 128 / 64 |
| JBB benign, English/Ukrainian | Unwanted-refusal test | 100 |
| Belebele, English/Ukrainian | Reading-comprehension utility | 900 |

The 620 harmful source requests form 531 groups using the supplied similarity
links and normalized exact English/Ukrainian matches. Each group contributes one
representative. English and Ukrainian versions share split membership. Remaining
Alpaca instructions are outside the experiment.

Claude translations were selected by blinded comparison with Gemma and Lapa.
Reviewer comments led to 24 corrections in the harmful dataset. The 192 Alpaca
translations include a separate 23-row flag log describing language-dependent,
ambiguous or malformed source instructions. Flags are not refusal labels.

## Sources and versions

- [Frozen paired prompts and split assignments](https://huggingface.co/datasets/steering-transfer/prompt-pairs/tree/ad61e819403256072115d9a7e52bcd006ac29f81/splits/bilingual-refusal-v1).
  `configs/frozen_data.json` pins the release and manifest checksum. The manifest
  records source hashes, split IDs, seed and duplicate-group membership.
- [JBB-Behaviors](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors):
  upstream dataset card declares MIT. The merged export contains 100 harmful
  requests; a separate file contains 100 benign requests.
- [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv):
  520 harmful source requests. Original export hashes are recorded; its upstream
  revision was not retained.
- [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca): revision
  `f13496211289def0ff88ae673389ae14a818b4b3`,
  [CC BY-NC 4.0](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE).
  Only instructions are used, not generated answers or model weights.
- [Belebele](https://huggingface.co/datasets/facebook/belebele): revision
  `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`, CC BY-SA 4.0. The 900 questions
  retain paired answer keys; questions sharing a passage remain linked.

No separate license is assigned to the translations. The repository's code
license does not replace upstream dataset terms.

## Storage

Datasets and split manifests live in the HF dataset repository. Full generated
answers live in the manually gated
[model-answers](https://huggingface.co/datasets/steering-transfer/model-answers)
repository; judge labels live in
[safety-scores](https://huggingface.co/datasets/steering-transfer/safety-scores).
Directions belong in the
[model artifact repository](https://huggingface.co/steering-transfer/refusal-directions).
Raw data, answers, tensors and logs stay outside Git.

Inputs are benchmark instructions, not collected private conversations. Harmful
requests and generated answers are research material for evaluating model
behavior, not verified factual claims.
