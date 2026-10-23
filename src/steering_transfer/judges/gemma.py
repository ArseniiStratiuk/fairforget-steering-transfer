"""Grammar-constrained labels; valid syntax does not establish judge accuracy."""

import json
import re


class GemmaJudge:
    def __init__(self, language_model, rubric):
        self.language_model = language_model
        self.rubric = rubric
        self.specification = {
            "provider": "local",
            "model": language_model.specification,
            "rubric": rubric,
            "constraint": "lm-format-enforcer-0.11.3",
        }

    def judge_batch(self, records):
        import torch
        from lmformatenforcer import RegexParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        prompts = []
        for record in records:
            data = json.dumps(
                {"question": record["prompt"], "answer": record["response"]},
                ensure_ascii=False,
            )
            prompts.append(self.rubric["instruction"] + data)
        model = self.language_model
        encoded = model.encode_prompts(prompts)
        if encoded.input_ids.shape[1] > self.rubric["max_input_tokens"]:
            raise ValueError(
                "Judge input exceeds the explicit token limit; answer was not truncated"
            )
        parser = RegexParser(
            "|".join(re.escape(label) for label in self.rubric["labels"])
        )
        prefix = build_transformers_prefix_allowed_tokens_fn(model.tokenizer, parser)
        with torch.inference_mode():
            generated = model.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=self.rubric["max_new_tokens"],
                prefix_allowed_tokens_fn=prefix,
                logits_to_keep=1,
            )
        labels = model.tokenizer.batch_decode(
            generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
        )
        results = []
        for label in labels:
            results.append(
                {
                    "label": label if label in self.rubric["labels"] else None,
                    "raw_output": label,
                    "status": "success"
                    if label in self.rubric["labels"]
                    else "invalid_label",
                }
            )
        return results

    def close(self):
        self.language_model.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
