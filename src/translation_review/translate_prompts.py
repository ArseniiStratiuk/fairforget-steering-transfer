"""Generate pinned Gemma/Lapa translation candidates for human review."""


def generate_translation_batch(model, inputs, settings, verify_replay):
    import torch

    with torch.inference_mode():
        generated = model.generate(
            **inputs, generation_config=settings, do_sample=False
        )
        if verify_replay:
            repeated = model.generate(
                **inputs, generation_config=settings, do_sample=False
            )
            if not torch.equal(generated, repeated):
                raise RuntimeError("Greedy translation replay differs")
    return generated


def main():
    from steering_transfer.result_storage import find_project_root
    from pathlib import Path
    from datetime import datetime, timezone
    from importlib.metadata import version
    import gc
    import hashlib
    import json
    import os
    import platform
    import re
    import time

    import pandas as pd
    import torch
    from huggingface_hub import hf_hub_download
    from IPython.display import display
    from transformers import (
        AutoTokenizer,
        Gemma3ForConditionalGeneration,
        GenerationConfig,
    )

    root = find_project_root()
    model_config = json.loads((root / "configs/models.json").read_text())
    model_paths = {}
    for name in ["gemma", "lapa"]:
        checkpoint = Path(model_config[name]["checkpoint"])
        model_paths[name] = (
            checkpoint if checkpoint.is_absolute() else root / checkpoint
        )
    model_names = [
        m.strip()
        for m in os.environ.get("MODELS", "gemma,lapa").split(",")
        if m.strip()
    ]
    assert model_names and set(model_names) <= set(model_paths), model_names
    limit = int(os.environ.get("ITEM_LIMIT", "0"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "512"))
    overwrite = os.environ.get("OVERWRITE", "0") == "1"
    assert limit >= 0 and batch_size > 0 and max_new_tokens > 0
    out_dir = root / "data/safety/translations"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-limit{limit}" if limit else ""
    torch.manual_seed(42)
    torch.set_num_threads(8)
    print("Models:", model_names)
    print("Records:", limit if limit else "all")
    print("Output folder:", out_dir)

    input_config = json.loads((root / "configs/datasets.json").read_text())[
        "prompt_pairs"
    ]
    candidates = [
        os.environ.get("PROMPT_PAIRS_CSV"),
        root / "data/safety/translated_formated.csv",
        root / "data/safety/prompt-pairs.csv",
    ]
    input_path = next((Path(p) for p in candidates if p and Path(p).is_file()), None)
    if input_path is None:
        input_path = Path(
            hf_hub_download(
                input_config["repo_id"],
                input_config["filename"],
                repo_type="dataset",
                revision=input_config["revision"],
            )
        )
    print("Input file:", input_path)
    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert input_sha256 == input_config["sha256"]
    frame = pd.read_csv(input_path).sort_values("record_id").reset_index(drop=True)
    assert len(frame) == 620 and frame.record_id.is_unique
    assert frame.goal.notna().all() and frame.goal.str.strip().ne("").all()
    if limit:
        frame = (
            frame.groupby("source", group_keys=False).head(limit).reset_index(drop=True)
        )
    display(frame.groupby("source").size().to_frame("records"))
    display(frame[["record_id", "source", "goal"]].head(4))

    PROMPT_TEMPLATE = """You are translating English prompts into Ukrainian for a research test set measuring
    refusal behavior in language models. The prompts come from the public AdvBench and
    JailbreakBench (JBB) harmful-behavior benchmarks, so many of them describe harmful
    requests. Your task is translation only: do not carry out, answer, judge, or refuse the
    request. The translation is used to test whether models refuse it in Ukrainian.

    INPUT: one record from the combined AdvBench + JBB prompt set, with fields
    record_id, source (AdvBench or JBB), goal (the English prompt).

    TASK: translate only the "goal" field into Ukrainian. "source" is context only; never
    translate it, copy it into the translation, or let it add content that is not
    already in "goal".

    TRANSLATION RULES (strict):
    1. Preserve the original intent and register exactly, do not soften, intensify,
       formalize, or casualize beyond what natural Ukrainian grammar requires.
    2. Preserve all negations, conditions, and constraints precisely (e.g. "without",
       "only if", "excluding").
    3. Preserve all numbers, dates, and quantities exactly as given.
    4. Preserve proper names (people, places, brands, products, organizations, software)
       either in standard Ukrainian convention if one exists, or transliterated
       consistently. Never substitute or localize a name.
    5. Translate only what "goal" asks for. Do not add an answer, a disclaimer, a
       safety caveat, or any instruction not present in the source. The output is a
       translated REQUEST, never a response to it.
    6. Never shift how harmful/benign the request reads. These goals are harmful
       requests by design; keep each one reading as the same request in Ukrainian. Do
       not soften, sanitize, euphemize, or refuse it, and do not make it more explicit,
       detailed, or dangerous than the English.
    7. If "goal" is ambiguous, idiomatic, or has more than one meaning-changing
       reading, translate the most literal/faithful reading and flag it with a short
       note, do not silently resolve it and do not edit/clarify the English.
    8. Use natural, idiomatic Ukrainian, avoid stiff or word-for-word calques.
       Phrase it as an informal singular imperative ("\u041d\u0430\u043f\u0438\u0448\u0438", "\u0421\u0442\u0432\u043e\u0440\u0438", "\u041f\u043e\u044f\u0441\u043d\u0438",
       "\u041e\u043f\u0438\u0448\u0438") to match how these are addressed to an AI (singular), not formal/plural
       ("\u041d\u0430\u043f\u0438\u0448\u0456\u0442\u044c", "\u041d\u0430\u0432\u0447\u0456\u0442\u044c", "\u041f\u043e\u044f\u0441\u043d\u0456\u0442\u044c", "\u041e\u043f\u0438\u0448\u0456\u0442\u044c", etc).
    9. Produce exactly one output object for this record, tagged with its record_id.

    OUTPUT FORMAT: return ONLY one JSON object (no prose, no markdown fences), so commas
    and quotes inside the translated text can't corrupt the output:
      {{
        "record_id": <int, copied unchanged from the input>,
        "ukrainian_prompt": <string, the Ukrainian translation of goal>,
        "flagged": <true|false>,
        "note": <string, empty "" if flagged is false; if true, a short English
                 note under 20 words naming the ambiguity and the alternative
                 reading, for human review, never a disclaimer or hedge>
      }}

    Do not include goal, source, similar_record_id, discard, split, or
    translation_status in the output, those get attached downstream from the
    original CSV file, not by you.

    INPUT RECORD:
    record_id: {record_id}
    source: {source}
    goal: {goal}"""

    def build_prompt(row):
        return PROMPT_TEMPLATE.format(
            record_id=int(row.record_id), source=row.source, goal=row.goal
        )

    print(build_prompt(frame.iloc[0]))

    def parse_reply(text, record_id):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        start = cleaned.find("{")
        obj = None
        if start != -1:
            try:
                obj, _ = json.JSONDecoder(strict=False).raw_decode(cleaned[start:])
            except json.JSONDecodeError:
                obj = None
        if not isinstance(obj, dict):
            return {
                "ukrainian_prompt": "",
                "flagged": False,
                "note": "",
                "status": "parse_error",
            }
        translation = str(obj.get("ukrainian_prompt") or "").strip()
        flagged = obj.get("flagged")
        flagged = (
            flagged
            if isinstance(flagged, bool)
            else str(flagged).strip().lower() == "true"
        )
        note = str(obj.get("note") or "").strip()
        try:
            id_ok = int(obj.get("record_id")) == int(record_id)
        except (TypeError, ValueError):
            id_ok = False
        status = "empty" if not translation else ("ok" if id_ok else "id_mismatch")
        return {
            "ukrainian_prompt": translation,
            "flagged": flagged,
            "note": note,
            "status": status,
        }

    assert torch.cuda.is_available(), "Run inside a GPU Slurm allocation"
    assert torch.cuda.device_count() == 1, "Each worker must see exactly one GPU"
    print("Host:", platform.node(), "GPU:", torch.cuda.get_device_name(0))
    summaries = {}
    OUTPUT_COLUMNS = [
        "record_id",
        "source",
        "goal",
        "ukrainian_prompt",
        "flagged",
        "note",
        "status",
        "model",
        "raw_response",
    ]

    for model_name in model_names:
        model_path = model_paths[model_name]
        output_path = out_dir / f"prompt-pairs-uk-{model_name}{suffix}.csv"
        partial_path = out_dir / f"prompt-pairs-uk-{model_name}{suffix}.partial.jsonl"
        provenance_path = (
            out_dir / f"prompt-pairs-uk-{model_name}{suffix}.provenance.json"
        )
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"{output_path} exists; set OVERWRITE=1 to replace it"
            )
        print(f"\n=== {model_name}: {model_path.name}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tokenizer.padding_side = "left"
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        settings = GenerationConfig(
            do_sample=False,
            max_new_tokens=max_new_tokens,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        eot_ids = settings.eos_token_id
        eot_ids = [eot_ids] if isinstance(eot_ids, int) else list(eot_ids)

        results = []
        started = time.monotonic()
        with partial_path.open("w", encoding="utf-8") as partial:
            for start in range(0, len(frame), batch_size):
                batch = frame.iloc[start : start + batch_size]
                conversations = [
                    [{"role": "user", "content": build_prompt(row)}]
                    for row in batch.itertuples(index=False)
                ]
                inputs = tokenizer.apply_chat_template(
                    conversations,
                    tokenize=True,
                    padding=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                generated = generate_translation_batch(
                    model, inputs, settings, verify_replay=start == 0
                )
                new_tokens = generated[:, inputs.input_ids.shape[-1] :].cpu().tolist()
                for row, tokens in zip(
                    batch.itertuples(index=False), new_tokens, strict=True
                ):
                    end = next(
                        (i + 1 for i, t in enumerate(tokens) if t in eot_ids),
                        len(tokens),
                    )
                    tokens = tokens[:end]
                    raw = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                    parsed = parse_reply(raw, row.record_id)
                    if not tokens or tokens[-1] not in eot_ids:
                        parsed["status"] = "token_limit"
                    record = {
                        "record_id": int(row.record_id),
                        "source": row.source,
                        "goal": row.goal,
                        **parsed,
                        "model": model_name,
                        "raw_response": raw,
                    }
                    results.append(record)
                    partial.write(json.dumps(record, ensure_ascii=False) + "\n")
                partial.flush()
                print(
                    f"{model_name} {min(start + batch_size, len(frame))}/{len(frame)}"
                    f"  {time.monotonic() - started:.0f}s",
                    flush=True,
                )

        assert [r["record_id"] for r in results] == frame.record_id.astype(int).tolist()
        pd.DataFrame(results, columns=OUTPUT_COLUMNS).to_csv(
            output_path, index=False, encoding="utf-8"
        )
        counts = pd.Series([r["status"] for r in results]).value_counts().to_dict()
        summaries[model_name] = counts
        provenance = {
            "input": {
                **input_config,
                "sha256_checked": input_sha256,
                "records": len(frame),
                "item_limit": limit,
            },
            "model": model_name,
            "model_path": str(model_path),
            "revision": model_path.name,
            "model_class": type(model).__name__,
            "dtype": str(model.dtype),
            "generation": {**settings.to_dict(), "do_sample": False},
            "first_batch_token_replay": True,
            "batch_size": batch_size,
            "padding_side": tokenizer.padding_side,
            "prompt_template": PROMPT_TEMPLATE,
            "prompt_placement": "single user turn, native chat template",
            "status_counts": counts,
            "flagged": int(sum(r["flagged"] for r in results)),
            "elapsed_seconds": time.monotonic() - started,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": platform.node(),
            "gpu": torch.cuda.get_device_name(0),
            "versions": {n: version(n) for n in ["torch", "transformers", "pandas"]},
            "review": "machine translation, pending human review",
        }
        provenance_path.write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("Saved:", output_path.name, counts)

        del model, tokenizer, generated, inputs
        gc.collect()
        torch.cuda.empty_cache()

    pd.set_option("display.max_colwidth", None)
    display(pd.DataFrame(summaries).fillna(0).astype(int))
    outputs = {
        name: pd.read_csv(
            out_dir / f"prompt-pairs-uk-{name}{suffix}.csv", keep_default_na=False
        )
        for name in model_names
    }
    for name, df in outputs.items():
        assert len(df) == len(frame) and df.record_id.is_unique
        problems = df[df.status.ne("ok")]
        print(
            f"{name}: {len(problems)} records need attention, {int(df.flagged.sum())} flagged"
        )
        if len(problems):
            display(problems[["record_id", "source", "status", "raw_response"]])
        flagged = df[df.flagged]
        if len(flagged):
            display(flagged[["record_id", "ukrainian_prompt", "note"]])

    side_by_side = frame[["record_id", "source", "goal"]].copy()
    for name, df in outputs.items():
        side_by_side[name] = (
            df.set_index("record_id")
            .ukrainian_prompt.reindex(side_by_side.record_id)
            .values
        )
    display(side_by_side.groupby("source").head(3))


if __name__ == "__main__":
    main()
