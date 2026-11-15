import json
import pandas as pd
import os
L = os.environ.get("BORROWING_ROOT", ".") + "/results/final/sibling-refusal-borrowing-v1/lapa_main/"
for w in (2, 3):
    u = {json.loads(l)["response_id"]: json.loads(l) for l in open(f"{L}baseline/worker-{w}/responses/responses.jsonl")}
    j = {}
    for l in open(f"{L}frozen_removal/worker-{w}/responses/responses.jsonl"):
        r = json.loads(l)
        if r["condition"] == "baseline":
            j[r["response_id"]] = r
    same = sum(u[k]["response"] == j[k]["response"] for k in u if k in j)
    su = pd.read_json(f"{L}baseline/worker-{w}/scored_answers.jsonl", lines=True).set_index("response_id")
    sj = pd.read_json(f"{L}frozen_removal/worker-{w}/scored_answers.jsonl", lines=True).set_index("response_id")
    sj = sj.loc[su.index]
    agree = {c: float((su[c] == sj[c]).mean()) for c in ["behavior", "harmful_fulfillment", "coherence"]}
    print(f"worker {w}: identical text {same}/{len(u)}; label agreement", {k: round(v, 3) for k, v in agree.items()})
    print("  harmful assistance UCU", round((su[su.label == 'harmful'].harmful_fulfillment == 'yes').mean(), 3),
          "JUPITER", round((sj[sj.label == 'harmful'].harmful_fulfillment == 'yes').mean(), 3))
