import os
import glob, json
import numpy as np, pandas as pd
L = os.environ.get("BORROWING_ROOT", ".") + "/results/final/sibling-refusal-borrowing-v1/lapa_main/"
def boot(diff, cluster, n=2000, seed=42):
    g = pd.DataFrame({"d": diff, "c": cluster}).groupby("c").d.agg(["sum", "size"]).to_numpy()
    rng = np.random.default_rng(seed); dr = g[rng.integers(0, len(g), (n, len(g)))]
    e = dr[..., 0].sum(1) / dr[..., 1].sum(1); return diff.mean(), *np.quantile(e, [.025, .975])
ref = {0: "frozen_removal/worker-0", 1: "frozen/worker-1", 2: "baseline/worker-2", 3: "baseline/worker-3"}
sp = pd.concat([pd.read_json(L + f"system_prompt/worker-{w}/scored_answers.jsonl", lines=True) for w in range(4)])
bl = pd.concat([pd.read_json(L + ref[w] + "/scored_answers.jsonl", lines=True).query("condition=='baseline'") for w in range(4)])
def cat(f):
    ha = f.harmful_fulfillment.eq("yes"); coh = f.coherence.eq("coherent")
    return pd.DataFrame({"harm": ha, "cohref": ~ha & coh & f.behavior.eq("refusal"), "incoh": ~ha & ~coh, "over": coh & f.behavior.eq("refusal"), "incoh_benign": ~coh}).astype(int).assign(pid=f.prompt_id.values, sid=f.semantic_request_id.values, label=f.label.values, lang=f.language.values)
a, b = cat(sp), cat(bl)
m = a.merge(b, on=["pid", "sid", "label", "lang"], suffixes=("_s", "_b"))
print("paired rows", len(m))
for lang in ("en", "uk"):
    for meas, lab in (("harm", "harmful"), ("cohref", "harmful"), ("incoh", "harmful"), ("over", "harmless"), ("incoh_benign", "harmless")):
        x = m[(m.lang == lang) & (m.label == lab)]
        est, lo, hi = boot((x[meas + "_s"] - x[meas + "_b"]).to_numpy(), x.sid.to_numpy())
        print(f"SYS {lang} {meas} rate {100*x[meas+'_s'].mean():.0f} change {100*est:+.0f} [{100*lo:+.0f},{100*hi:+.0f}] n={len(x)}")
# Belebele paired
def bel(paths):
    d = pd.DataFrame([json.loads(l) for p in paths for l in open(p)]); return d[d.condition == "baseline"]
s = bel([L + f"system_prompt/worker-{w}/utility/utility_predictions.jsonl" for w in range(4)])
u = bel([L + ref[w] + "/utility/utility_predictions.jsonl" for w in range(4)])
mb = s.merge(u, on=["example_id", "language"], suffixes=("_s", "_b"))
for lang in ("en", "uk"):
    x = mb[mb.language == lang]; est, lo, hi = boot((x.correct_s.astype(int) - x.correct_b.astype(int)).to_numpy(), x.passage_id_s.to_numpy())
    print(f"SYS belebele {lang} acc {100*x.correct_s.mean():.1f} change {100*est:+.1f} [{100*lo:+.1f},{100*hi:+.1f}] n={len(x)}")
# Guard unsafe rates on harmful requests
rows = []
for p in glob.glob(L + "frozen*/worker-*/guard/labels.jsonl") + glob.glob(L + "system_prompt/worker-*/guard/labels.jsonl"):
    shard = p.split("lapa_main/")[1].rsplit("/guard", 1)[0]
    meta = {}
    for l in open(p.replace("guard/labels.jsonl", "responses/responses.jsonl")):
        r = json.loads(l); meta[r["response_id"]] = (r["condition"], r["language"], r["label"])
    for l in open(p):
        g = json.loads(l); c, lang, lab = meta[g["response_id"]]
        if lab == "harmful":
            rows.append(dict(cond=("system_prompt" if shard.startswith("system_prompt") else c), lang=lang, unsafe=g["label"] == "unsafe", shard=shard))
G = pd.DataFrame(rows)
G = G[~((G.cond == "baseline") & G.shard.str.startswith("frozen_controls"))]
t = (G.groupby(["cond", "lang"]).unsafe.agg(["mean", "size"]).assign(mean=lambda f: (100 * f["mean"]).round(0)))
print(t.unstack("lang").to_string())
