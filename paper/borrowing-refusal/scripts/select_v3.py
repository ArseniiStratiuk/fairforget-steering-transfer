"""Draft Sec. 3.3 rule: a donor setting must beat a same-norm random direction in the donor.

Score over every full-length validated setting: coherent refusals of the direction minus
coherent refusals of the matched random direction minus incoherent answers of the direction.
Ties: more coherent refusals, then smaller k. Writes selected_settings_v3.json (all six
directions) and selected_settings_v3_new.json (settings not yet generated in Lapa).
"""
import glob
import json

import pandas as pd

import run_addition_arm as arm_code

arm = arm_code.arm
frames = [pd.read_csv(p).assign(task=p.split("/")[-3]) for p in glob.glob(str(arm / "source_validation/*/*/source_validation.csv"))]
frames += [pd.read_csv(p).assign(task=p.split("/")[-2]) for p in glob.glob(str(arm / "source_validation_v2/*/source_validation.csv"))]
table = pd.concat(frames, ignore_index=True)
table["kind"] = table.condition.str.split("_").str[0]
table["setting"] = table.condition.str.split("_", n=1).str[1]
wide = table.pivot_table(index=["task", "setting"], columns="kind", values=["coherent_refusals", "incoherent"]).reset_index()
wide.columns = ["task", "setting", "coherent", "coherent_random", "incoherent", "incoherent_random"]
wide["score"] = wide.coherent - wide.coherent_random - wide.incoherent
v2 = json.loads((arm / "selected_settings_v2.json").read_text())
selected, new = {}, {}
for task, group in wide.groupby("task"):
    group = group.assign(k=group.setting.str.split("_k").str[1].astype(float))
    best = group.sort_values(["score", "coherent", "k"], ascending=[False, False, True]).iloc[0]
    layer, multiplier = int(best.setting[1:].split("_k")[0]), float(best.k)
    direction = arm_code.block_direction(*task.split("-"), layer)
    selected[task] = {
        "layer": layer, "multiplier": multiplier,
        "strength": float(multiplier * direction.provenance["raw_norm"]),
        "coherent_refusals": int(best.coherent), "random_coherent_refusals": int(best.coherent_random),
        "incoherent": int(best.incoherent), "score": int(best.score),
        "rule": "most (coherent refusals - random coherent refusals - incoherent), then more coherent refusals, then smaller k",
    }
    if (v2[task]["layer"], v2[task]["multiplier"]) != (layer, multiplier):
        new[task] = selected[task]
    print(task, (layer, multiplier), "new" if task in new else "already in addition_v2")
json.dump(selected, open(arm / "selected_settings_v3.json", "w"), indent=2)
json.dump(new, open(arm / "selected_settings_v3_new.json", "w"), indent=2)
wide.sort_values(["task", "score"], ascending=[True, False]).to_csv(arm / "validated_settings_v3.csv", index=False)
