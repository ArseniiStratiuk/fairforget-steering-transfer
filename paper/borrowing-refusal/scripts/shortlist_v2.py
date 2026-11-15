"""Shortlist addition settings from the 64-token prompt-only screen without its coherence label.

Jev labels many answers that stop at the 64-token limit as incoherent, so the screen ranks
settings by refusals among answers whose first 64 tokens show no repetition loop (distinct
token-trigram share >= 0.6). Ties: fewer loops, smaller k, earlier block. Coherence is then
measured by full-length validation. Writes addition_arm_prompt/screen_shortlist_v2.json.
"""
import json
import pandas as pd

F = "results/final/sibling-refusal-borrowing-v1"
TOP = 4


def distinct_trigrams(ids):
    grams = [tuple(ids[i:i + 3]) for i in range(len(ids) - 2)]
    return len(set(grams)) / max(1, len(grams))


out = {"rule": __doc__.strip(), "shortlist": {}}
for task in ["gemma-en", "gemma-uk", "mamay-en", "mamay-uk", "lapa-en", "lapa-uk"]:
    model, language = task.split("-")
    folder = f"{F}/{model}/{language}/addition_prompt_screen"
    rows = []
    for line in open(folder + "/responses/responses.jsonl"):
        answer = json.loads(line)
        path = f"{folder}/judgments/predictions/jev/screen/zero-shot/{answer['response_id']}.json"
        labels = {p["dimension"]: p["label"] for p in json.load(open(path))["predictions"]}
        layer, multiplier = answer["condition"].removeprefix("screen_L").split("_k")
        loop = distinct_trigrams(answer["response_ids"]) < 0.6
        rows.append(dict(layer=int(layer), multiplier=float(multiplier), loop=loop,
                         refusal=labels["behavior"] == "refusal" and not loop))
    table = pd.DataFrame(rows).groupby(["layer", "multiplier"], as_index=False).agg(
        refusals=("refusal", "sum"), loops=("loop", "sum"))
    table = table.sort_values(["refusals", "loops", "multiplier", "layer"],
                              ascending=[False, True, True, True]).head(TOP)
    out["shortlist"][task] = table.to_dict("records")
    print(task, [(r["layer"], r["multiplier"], int(r["refusals"]), int(r["loops"])) for r in out["shortlist"][task]])
json.dump(out, open(f"{F}/addition_arm_prompt/screen_shortlist_v2.json", "w"), indent=2, default=int)
