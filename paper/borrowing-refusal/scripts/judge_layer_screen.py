"""Judge the 64-token layer-screen answers with the selected joint Jev configuration."""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from steering_transfer.joint_judging import save_joint_decision
from steering_transfer.judges.jev import JevJudge
from steering_transfer.result_storage import file_hash, read_records

root = Path(os.environ.get("BORROWING_ROOT", ".")).resolve()
run = root / "results/final/sibling-refusal-borrowing-v1"
configuration = json.loads((root / "configs/experiments/borrowing_refusal.json").read_text())
pin = configuration["judges"]["jev"]["selected_configuration"]
assert file_hash(root / pin["path"]) == pin["sha256"]
settings = json.loads((root / pin["path"]).read_text())
assert not settings["examples"]

with JevJudge(settings["model"], settings["rubrics"]["behavior"]) as judge:
    for task in sys.argv[1:]:
        model_name, language = task.split("-")
        output = run / model_name / language / "addition_layer_screen"
        if (output / "screen_summary.csv").exists():
            continue
        answers = read_records(output / "responses/responses.jsonl")
        records = answers.to_dict("records")
        judgments = output / "judgments"

        def judge_one(record):
            for attempt in range(3):
                try:
                    save_joint_decision(judgments, "jev", judge, record, settings["rubrics"], "screen")
                    return
                except Exception:
                    if attempt == 2:
                        raise

        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(judge_one, records))
        rows = []
        for record in records:
            path = judgments / "predictions/jev/screen/zero-shot" / f"{record['response_id']}.json"
            labels = {p["dimension"]: p["label"] for p in json.loads(path.read_text())["predictions"]}
            rows.append({"condition": record["condition"], **labels})
        labels = pd.DataFrame(rows)
        labels["coherent_refusal"] = labels.behavior.eq("refusal") & labels.coherence.eq("coherent")
        labels["incoherent"] = labels.coherence.eq("incoherent")
        summary = labels.groupby("condition").agg(
            coherent_refusals=("coherent_refusal", "sum"),
            incoherent=("incoherent", "sum"),
            answers=("behavior", "size"),
        ).reset_index()
        parts = summary.condition.str.extract(r"screen_L(\d+)_k([\d.]+)")
        summary["layer"] = parts[0].astype(int)
        summary["multiplier"] = parts[1].astype(float)
        summary = summary.sort_values(["layer", "multiplier"])
        summary.to_csv(output / "screen_summary.csv", index=False)
        print("JUDGED", task, flush=True)
        print(summary.sort_values(["coherent_refusals", "incoherent"], ascending=[False, True]).head(8).to_string(index=False), flush=True)
