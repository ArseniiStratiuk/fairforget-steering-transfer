"""Score completed Lapa shards with the selected joint Jev judge (CPU host with network).

Writes <set>/worker-<k>/scored_answers.jsonl: one row per answer with its request
identity, condition and the three joint labels.
"""

import os
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def score(folder):
    completed = json.loads((folder / "generation_complete.json").read_text())
    if completed["responses_sha256"] != file_hash(folder / "responses/responses.jsonl"):
        raise ValueError("Answers changed after generation")
    answers = read_records(folder / "responses/responses.jsonl")
    assert len(answers) == completed["answers"] and answers.response_id.is_unique
    judgments = folder / "judgments"
    with JevJudge(settings["model"], settings["rubrics"]["behavior"]) as judge:
        def one(record):
            for attempt in range(4):
                try:
                    save_joint_decision(judgments, "jev", judge, record, settings["rubrics"], "experiment")
                    return
                except Exception:
                    if attempt == 3:
                        raise
        with ThreadPoolExecutor(max_workers=16) as pool:
            list(pool.map(one, answers.to_dict("records")))
    rows = []
    for record in answers.to_dict("records"):
        path = judgments / "predictions/jev/experiment/zero-shot" / f"{record['response_id']}.json"
        labels = {p["dimension"]: p["label"] for p in json.loads(path.read_text())["predictions"]}
        rows.append({key: record[key] for key in (
            "response_id", "condition", "prompt_id", "semantic_request_id", "language", "label")} | labels)
    with (folder / "scored_answers.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("SCORED", folder, len(rows), flush=True)


if __name__ == "__main__":
    for item in sys.argv[1:]:
        folder = run / "lapa_main" / item
        if not (folder / "scored_answers.jsonl").exists():
            score(folder)
