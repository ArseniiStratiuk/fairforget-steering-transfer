"""Rates of the unmodified donors on the test split (E0) and Llama Guard 4 agreement with the
120 human-labeled answers. Run from BORROWING_ROOT."""
import glob
import os

import pandas as pd

root = os.environ.get("BORROWING_ROOT", ".")
main = root + "/results/final/sibling-refusal-borrowing-v1/lapa_main/"
for model in ("gemma", "mamay"):
    paths = glob.glob(main + f"e0_{model}/scored_answers.jsonl") + glob.glob(main + f"e0_{model}/worker-*/scored_answers.jsonl")
    f = pd.concat([pd.read_json(p, lines=True) for p in paths])
    for lang in ("en", "uk"):
        h = f[(f.label == "harmful") & (f.language == lang)]
        b = f[(f.label == "harmless") & (f.language == lang)]
        ha = h.harmful_fulfillment.eq("yes")
        cr = ~ha & h.coherence.eq("coherent") & h.behavior.eq("refusal")
        inc = ~ha & h.coherence.eq("incoherent")
        orf = b.coherence.eq("coherent") & b.behavior.eq("refusal")
        print(f"E0 {model} {lang}: harmful {len(h)} benign {len(b)} | harmful assistance {100*ha.mean():.1f} "
              f"coherent refusal {100*cr.mean():.1f} incoherent {100*inc.mean():.1f} over-refusal {100*orf.mean():.1f}")
judge = root + "/results/judge/"
guard = pd.read_json(judge + "guard_reference/labels.jsonl", lines=True)
human = pd.concat([pd.read_json(root + "/outputs/publication/joint-judge-comparison-2026-11-04/" + n, lines=True)
                   for n in ("comparison.jsonl", "calibration.jsonl")])
m = human.merge(guard[["response_id", "label"]], on="response_id")
y, u = m.harmful_fulfillment.eq("yes"), m.label.eq("unsafe")
print(f"Guard on {len(m)} human-labeled answers: {int((y & u).sum())} of {int(y.sum())} harmful flagged "
      f"(recall {(y & u).sum() / y.sum():.3f}, precision {(y & u).sum() / u.sum():.3f})")
