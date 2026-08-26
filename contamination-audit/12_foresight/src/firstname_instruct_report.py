# -*- coding: utf-8 -*-
"""Base vs Instruct, first-name multiple choice (fair options). Shows (a) the instruct models are far more
confident -- they recover first-name knowledge the base models have but the weak completion can't express
(Picasso, Curie, Stravinsky) -- and (b) the time-lock still holds: post-1929 people stay near chance even
for instruct. Focus 1913 & 1929 (1900 is the weak base; no 1900-instruct). Self-contained HTML + figure.
Run:  python src/firstname_instruct_report.py
"""
import os, json, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

OUT = os.path.join(common.RESULTS, "prominence")
BASE = json.load(open(os.path.join(OUT, "firstname_fc2.json"), encoding="utf-8"))
INS = json.load(open(os.path.join(OUT, "firstname_instruct.json"), encoding="utf-8"))
order = sorted(BASE["1929"].keys(), key=lambda s: BASE["1929"][s]["fame"])
FY = {s: BASE["1929"][s]["fame"] for s in order}
FN = {s: BASE["1929"][s]["first"] for s in order}
cols = [("base 1913", BASE, "1913"), ("instruct 1913", INS, "1913"),
        ("base 1929", BASE, "1929"), ("instruct 1929", INS, "1929")]

# ---- 4-column heatmap ----
M = np.array([[100 * d[m][s]["p_true"] for _, d, m in cols] for s in order])
fig, ax = plt.subplots(figsize=(7.4, 9.2))
ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in cols], fontsize=9, rotation=20)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{s} → {FN[s]}?  ({FY[s]})" for s in order], fontsize=8.1)
for i in range(len(order)):
    for j in range(4):
        v = M[i, j]
        ax.annotate(f"{v:.0f}", (j, i), ha="center", va="center", fontsize=7.5, color="white" if v > 55 else "#333")
ax.axvline(1.5, color="#1a1a1a", lw=1.2)
ax.set_title("First name of (role+surname), fair options (chance 20%). Base vs Instruct, 1913 & 1929.\n"
             "Instruct recovers knowledge the base can't surface; post-1929 (pink names) stay near chance.",
             fontsize=9.4, loc="left")
fig.tight_layout(); FIG = os.path.join(common.FIGURES, "fig_firstname_instruct.png")
fig.savefig(FIG, bbox_inches="tight", dpi=150); plt.close(fig)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

rows = ""
for s in order:
    post = FY[s] > 1929
    rows += (f"<tr class={'post' if post else ''}><td>{s} → {FN[s]}</td><td>{FY[s]}</td>"
             f"<td class=n>{100*BASE['1913'][s]['p_true']:.0f}%</td><td class=n>{100*INS['1913'][s]['p_true']:.0f}%</td>"
             f"<td class=n>{100*BASE['1929'][s]['p_true']:.0f}%</td><td class=n>{100*INS['1929'][s]['p_true']:.0f}%</td></tr>")
html = f"""<!doctype html><meta charset=utf-8><title>First name: base vs instruct</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:900px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.4rem}} h2{{font-size:1.1rem;margin-top:1.6rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.post td{{background:#fbeeee}} .cap{{color:#555;font-size:.9rem}} img{{max-width:100%}}</style>
<h1>Do the models know the first name? &mdash; base vs instruct</h1>
<p>Surname + role pin the person; the model picks the first name from four distractors matched on language
and token length (chance&nbsp;=&nbsp;20%). Base models complete &ldquo;<i>&hellip; first name was ___</i>&rdquo;;
instruct models are asked directly through their chat template. Ranke 1913 &amp; 1929 (1900 is the weak base,
no instruct variant).</p>
<img src="data:image/png;base64,{b64}">
<p class=cap>Four columns: base vs instruct, at each cutoff. Instruct (columns 2, 4) is far brighter for
people the model knows &mdash; it surfaces first-name knowledge the base completion cannot. The pink post-1929
rows stay near chance in every column, instruct included: the chat tuning added confidence, not post-cutoff facts.</p>
<h2>P(correct first name)</h2>
<table><tr><th>person</th><th>fame</th><th>base 1913</th><th>instruct 1913</th><th>base 1929</th><th>instruct 1929</th></tr>{rows}</table>
"""
open(os.path.join(OUT, "firstname_instruct.html"), "w", encoding="utf-8").write(html)

# summary stats
def mean(v): return sum(v) / len(v)
known = [s for s in order if FY[s] <= 1929]
post = [s for s in order if FY[s] > 1929]
print("mean P(true), known (<=1929) vs post-1929:")
for lab, d, m in cols:
    print(f"  {lab:16} known {mean([d[m][s]['p_true'] for s in known]):.0%}   post-1929 {mean([d[m][s]['p_true'] for s in post]):.0%}")
print("\nbiggest base->instruct gains at 1929:")
g = sorted(order, key=lambda s: (INS['1929'][s]['p_true'] - BASE['1929'][s]['p_true']), reverse=True)[:8]
for s in g:
    print(f"  {s+' -> '+FN[s]:24} base {100*BASE['1929'][s]['p_true']:>3.0f}% -> instruct {100*INS['1929'][s]['p_true']:>3.0f}%")
print("\nwrote figures/fig_firstname_instruct.png and results/prominence/firstname_instruct.html")
