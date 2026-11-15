"""GPU execution for the same frozen judge benchmark used by API judges."""

import argparse
import json
import os
import re
from pathlib import Path
from time import perf_counter

from .judge_benchmark import prepare_judge_tasks, render_judge_prompt
from .result_storage import content_hash


def run_gpu_judge(folder, judge_name, checkpoint, shard, shards=2, phase="calibration"):
    folder = Path(folder)
    if judge_name == "gpt-oss":
        from .judges.gpt_oss import GptOssJudge

        judge = GptOssJudge(checkpoint)
        specification = judge.specification
    elif judge_name == "gemma":
        import torch
        from lmformatenforcer import RegexParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        from .language_models import load_language_model

        judge = load_language_model("gemma", Path(checkpoint).name, checkpoint)
        specification = judge.specification
    else:
        raise ValueError(f"Unsupported GPU judge: {judge_name}")

    replay_checked = False
    configurations = ("zero-shot", "two-shot")
    if phase == "comparison":
        selection = json.loads(
            (folder / f"selected-prompt-{judge_name}.json").read_text()
        )
        configurations = (selection["configuration"],)
    for configuration in configurations:
        tasks = prepare_judge_tasks(folder, phase, configuration)
        tasks = [task for index, task in enumerate(tasks) if index % shards == shard]
        destination = folder / "predictions" / judge_name / phase / configuration
        destination.mkdir(parents=True, exist_ok=True)
        for index, task in enumerate(tasks):
            path = destination / f"{task['dimension']}-{task['response_id']}.json"
            identity = content_hash(task)
            if path.exists():
                saved = json.loads(path.read_text())
                if saved["input_hash"] != identity:
                    raise ValueError("Saved GPU prediction has different inputs")
                if saved["specification"] != specification:
                    raise ValueError(
                        "Saved GPU prediction uses a different model setup"
                    )
                continue
            prompt = render_judge_prompt(task)
            labels = task["rubric"]["labels"]
            started = perf_counter()
            if judge_name == "gpt-oss":
                decision = judge.judge_prompts([prompt], labels)[0]
                if not replay_checked:
                    repeated = judge.judge_prompts([prompt], labels)[0]
                    decision["replay"] = repeated
                    decision["replay_label_matches"] = (
                        repeated["label"] == decision["label"]
                    )
                    if not decision["reasoning_present"]:
                        raise ValueError("GPT-OSS smoke test did not produce reasoning")
            else:
                encoded = judge.encode_prompts([prompt])
                if encoded.input_ids.shape[1] > 32768:
                    raise ValueError("Input too long; no answer was truncated")
                parser = RegexParser("|".join(re.escape(label) for label in labels))
                allowed = build_transformers_prefix_allowed_tokens_fn(
                    judge.tokenizer, parser
                )
                with torch.inference_mode():
                    generated = judge.model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=16,
                        prefix_allowed_tokens_fn=allowed,
                        logits_to_keep=1,
                    )
                    if not replay_checked:
                        repeated = judge.model.generate(
                            **encoded,
                            do_sample=False,
                            max_new_tokens=16,
                            prefix_allowed_tokens_fn=allowed,
                            logits_to_keep=1,
                        )
                        if not torch.equal(generated, repeated):
                            raise ValueError(
                                "Gemma greedy replay changed output tokens"
                            )
                tokens = generated[0, encoded.input_ids.shape[1] :]
                label = judge.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                decision = {
                    "label": label,
                    "status": "success" if label in labels else "invalid_output",
                    "input_tokens": encoded.input_ids.shape[1],
                    "output_tokens": len(tokens),
                }
            row = {
                "judge": judge_name,
                "phase": phase,
                "configuration": configuration,
                "response_id": task["response_id"],
                "dimension": task["dimension"],
                "example_ids": [
                    example["response_id"]
                    for example in task["rubric"].get("examples", [])
                ],
                "latency_seconds": perf_counter() - started,
                "includes_replay": not replay_checked,
                "job_id": os.environ.get("SLURM_JOB_ID"),
                **decision,
            }
            payload = {
                "input_hash": identity,
                "specification": specification,
                "predictions": [row],
            }
            if decision["status"] != "success":
                failure_path = path.with_suffix(".failure.json")
                failure_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
                )
                raise ValueError(f"Invalid {judge_name} output saved at {failure_path}")
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            replay_checked = True
            print(
                judge_name,
                configuration,
                shard,
                index,
                row["response_id"],
                row["dimension"],
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("judge", choices=["gemma", "gpt-oss"])
    parser.add_argument("checkpoint")
    parser.add_argument("shard", type=int)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument(
        "--phase", choices=["calibration", "comparison"], default="calibration"
    )
    arguments = parser.parse_args()
    run_gpu_judge(
        arguments.folder,
        arguments.judge,
        arguments.checkpoint,
        arguments.shard,
        shards=arguments.shards,
        phase=arguments.phase,
    )
