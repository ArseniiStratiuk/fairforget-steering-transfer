"""GPT-OSS reasoning with grammar constraints applied only to the final label."""

from time import perf_counter


class GptOssJudge:
    def __init__(self, checkpoint, reasoning_effort="medium"):
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding
        import vllm
        from vllm import LLM
        from vllm.config import StructuredOutputsConfig

        load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.model = LLM(
            model=str(checkpoint),
            max_model_len=32768,
            gpu_memory_utilization=0.90,
            enforce_eager=True,
            seed=0,
            structured_outputs_config=StructuredOutputsConfig(
                reasoning_parser="openai_gptoss",
                enable_in_reasoning=False,
            ),
        )
        self.tokenizer = self.model.get_tokenizer()
        self.reasoning_effort = reasoning_effort
        self.specification = {
            "checkpoint": str(checkpoint),
            "vllm": vllm.__version__,
            "reasoning_effort": reasoning_effort,
            "reasoning_parser": "openai_gptoss",
            "constraint": "final label only",
            "max_new_tokens": 4096,
            "temperature": 0,
            "seed": 0,
        }

    def _generate(self, prompts, constraints):
        from vllm import SamplingParams
        from vllm.entrypoints.openai.parser.harmony_utils import parse_chat_output

        rendered = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                reasoning_effort=self.reasoning_effort,
            )
            for prompt in prompts
        ]
        parameters = SamplingParams(
            temperature=0,
            max_tokens=4096,
            seed=0,
            skip_special_tokens=False,
            structured_outputs=constraints,
        )
        started = perf_counter()
        outputs = self.model.generate(rendered, parameters, use_tqdm=False)
        elapsed = perf_counter() - started
        results = []
        for output in outputs:
            completion = output.outputs[0]
            reasoning, content, tool_calls = parse_chat_output(
                list(completion.token_ids)
            )
            label = content.strip() if content is not None else None
            results.append(
                {
                    "label": label,
                    "status": "success"
                    if content is not None
                    and completion.finish_reason == "stop"
                    and not tool_calls
                    else "invalid_output",
                    "finish_reason": completion.finish_reason,
                    "raw_output": completion.text,
                    "reasoning_present": bool(reasoning and reasoning.strip()),
                    "input_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(completion.token_ids),
                    "batch_wall_seconds": elapsed,
                }
            )
        return results

    def judge_prompts(self, prompts, labels):
        from vllm.sampling_params import StructuredOutputsParams

        results = self._generate(prompts, StructuredOutputsParams(choice=labels))
        for result in results:
            if result["label"] not in labels:
                result["status"] = "invalid_output"
        return results

    def judge_all(self, record, rubrics):
        import json
        from vllm.sampling_params import StructuredOutputsParams
        from ..joint_judging import joint_prompt, label_schema, validate_labels

        result = self._generate(
            [joint_prompt(record, rubrics)],
            StructuredOutputsParams(json=label_schema(rubrics)),
        )[0]
        if result["status"] != "success":
            raise ValueError("GPT-OSS did not complete its JSON answer")
        labels = json.loads(result.pop("label"))
        validate_labels(labels, rubrics)
        return {"labels": labels, **result, "batch_size": 1}
