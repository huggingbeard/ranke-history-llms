# -*- coding: utf-8 -*-
"""Ground-up intuition for the date-spike test, on the 1913 model. Takes ONE known question, shows the prompt, shows
one generated answer, then the full distribution over years (a sharp spike). Then an INVENTED event (flat). Then a
leaked post-cutoff event (a spike where there should be none) and a clean one (flat). Produces fig_intuition.png (a
grid of P(year) bar charts) + intuition.json (prompts, sample answers, distributions) for embedding in the report.
Run:  python src/intuition.py
"""
import os, json, math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1913-1202-Instruct"
WIN = 8
# label, scoring-prompt (ends before the year), chat-question (for a free answer), true year, kind
EVENTS = [
 ("Russo-Japanese War ends", "The Russo-Japanese War came to an end in the year", "In one sentence, in what year did the Russo-Japanese War end?", 1905, "known"),
 ("Titanic sinks", "The ocean liner Titanic struck an iceberg and sank in the year", "In one sentence, in what year did the Titanic sink?", 1912, "known"),
 ("*Treaty of Kesselheim", "The Treaty of Kesselheim was signed in the year", "In one sentence, in what year was the Treaty of Kesselheim signed?", 1888, "invented"),
 ("*painter Ottoline Crest", "The painter Ottoline Crest completed her famous mural in the year", "In one sentence, in what year did the painter Ottoline Crest complete her mural?", 1901, "invented"),
 ("Bolsheviks seize power", "The Bolsheviks seized power in Russia in the year", "In one sentence, in what year did the Bolsheviks seize power in Russia?", 1917, "post-cutoff"),
 ("Moon landing", "American astronauts first walked on the Moon in the year", "In one sentence, in what year did astronauts first walk on the Moon?", 1969, "post-cutoff"),
]


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


print(f"loading {MID} ...", flush=True)
tok, model = common.load(MID)


def sample_answer(q):
    tok.padding_side = "left"
    prompt = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, tokenize=False)
    enc = tok([prompt], return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        g = model.generate(**enc, do_sample=True, temperature=0.7, top_p=0.9, max_new_tokens=40, pad_token_id=tok.eos_token_id)
    return tok.decode(g[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")


rows = {}
for lab, pr, q, ty, kind in EVENTS:
    years = list(range(ty - WIN, ty + WIN + 1))
    p = softmax([common.slp(model, tok, pr, f" {y}") for y in years])
    rows[lab] = {"prompt": pr, "q": q, "true": ty, "kind": kind, "years": years, "p": [round(x, 4) for x in p],
                 "answer": sample_answer(q), "spike": round(p[years.index(ty)] / (np.mean([p[i] for i, y in enumerate(years) if y != ty]) + 1e-12), 1)}
    print(f"  {lab}: answer = {rows[lab]['answer'][:90]}  (spike {rows[lab]['spike']:.0f}×)")
json.dump(rows, open(os.path.join(OUT, "intuition.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
del tok, model; common.free()

# ---- Figure 1: grid of P(year) bar charts (Voth figure rules: numbered, axes labelled, legend, no stray lines) ----
from matplotlib.patches import Patch
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
KC = {"known": "#2e7d32", "invented": "#8a8f98", "post-cutoff": "#7a2f8f"}      # panel-category (title) colours
TRUE, OTHER = "#c0392b", "#cfd3d8"                                              # bar colours
TAG = {"known": "KNOWN — pre-1913", "invented": "INVENTED — cannot be known", "post-cutoff": "AFTER the 1913 cutoff"}
fig, axes = plt.subplots(3, 2, figsize=(12.5, 9.6))
labels = [e[0] for e in EVENTS]
for k, (ax, lab) in enumerate(zip(axes.ravel(), labels)):
    r = rows[lab]; years = r["years"]; p = np.array(r["p"]) * 100
    ax.bar(years, p, color=[TRUE if y == r["true"] else OTHER for y in years], width=.8)
    ax.set_title(f"{lab.lstrip('*')} · true year {r['true']}   [{TAG[r['kind']]}]", fontsize=9.5,
                 color=KC[r["kind"]], fontweight="bold", loc="left")
    pt = r["p"][years.index(r["true"])] * 100
    ax.annotate(f"P(true year) = {pt:.0f}%", (0.97, 0.9), xycoords="axes fraction", ha="right", fontsize=10,
                color=KC[r["kind"]], fontweight="bold")
    ax.set_ylim(0, max(100, p.max() * 1.18)); ax.set_ylabel("P(year), %", fontsize=8.5)
    if k >= 4:
        ax.set_xlabel("year the model assigns to the event", fontsize=8.5)
    ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
fig.suptitle("Figure 1.  Asked for an event's date, the Ranke-1913 model spikes on one year when it knows, and spreads flat when it does not.",
             fontsize=12.3, y=1.0, x=0.01, ha="left")
fig.text(0.01, 0.963, "Each bar is the probability the model places on that year for the event in the panel title. "
         "Rows: known pre-1913 facts (top), invented events that cannot be known (middle), events after the 1913 cutoff (bottom).",
         fontsize=8.7, color="#555", ha="left")
handles = [Patch(fc=TRUE, label="the true year"), Patch(fc=OTHER, label="other years scored"),
           Patch(fc=KC["known"], label="panel: known (pre-1913)"), Patch(fc=KC["invented"], label="panel: invented (control)"),
           Patch(fc=KC["post-cutoff"], label="panel: after 1913 cutoff")]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.3, bbox_to_anchor=(0.5, -0.015))
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
FIG = os.path.join(common.FIGURES, "fig_intuition.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG} and results/superlatives/intuition.json")
