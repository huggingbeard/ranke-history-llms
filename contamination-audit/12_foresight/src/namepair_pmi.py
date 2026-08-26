# -*- coding: utf-8 -*-
"""The first+last name COMBO, done cleanly. Raw P(surname | first name) is confounded by how common the
first name is (Leo -> Tolstoy is ~0 because 'Leo' has a thousand continuations). So measure the BINDING:
how much more likely the surname is right after the first name than after a generic 'Mr.' --
   pmi = log P(surname | first name)  -  log P(surname | "Mr.")   (nats; >0 = the names are bound).
No generation, no judge. Ranke 1900/1913/1929 over the cross-domain battery. Writes namepair_pmi.json,
figures/fig_namepair_pmi.png, and a self-contained HTML.  Run:  python src/namepair_pmi.py
"""
import os, json, math, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common
from foresight_bank import RANKE
from prominence_bank import BATTERY, MODELS

OUT = os.path.join(common.RESULTS, "prominence")
RAW = os.path.join(OUT, "namepair_pmi.json")
NEUTRAL = "Mr."

data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag in MODELS:
    if tag in data:
        continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    row = {}
    for name, dom, fy, cue, sur, ident in BATTERY:
        cond = common.slp(model, tok, cue, sur)          # log P(surname | first name)
        base = common.slp(model, tok, NEUTRAL, sur)      # log P(surname | "Mr.")
        row[name] = {"cue": cue, "sur": sur.strip(), "fame": fy, "domain": dom,
                     "pmi": round(cond - base, 3), "p_cond": round(math.exp(cond), 5)}
    data[tag] = row
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

order = sorted([b[0] for b in BATTERY], key=lambda n: next(x[2] for x in BATTERY if x[0] == n))
FY = {b[0]: b[2] for b in BATTERY}
M = np.array([[data[t][n]["pmi"] for t in MODELS] for n in order])

# ---- heatmap ----
fig, ax = plt.subplots(figsize=(6.8, 9.4))
vmax = 8
im = ax.imshow(np.clip(M, 0, vmax), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels([f"Ranke {m}" for m in MODELS], fontsize=10)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{data['1929'][n]['cue']} → {data['1929'][n]['sur']}   ({FY[n]})" for n in order], fontsize=8.3)
for i in range(len(order)):
    for j in range(3):
        v = M[i, j]
        ax.annotate(f"{v:.1f}", (j, i), ha="center", va="center", fontsize=7.6,
                    color="white" if v > 4.5 else "#333")
ax.set_title("Name binding: how strongly First predicts Last (nats above a generic 'Mr.').\n"
             "High = the model has the full name as a unit; ~0 = it does not. Rows by fame year.",
             fontsize=9.8, loc="left")
fig.tight_layout(); FIG = os.path.join(common.FIGURES, "fig_namepair_pmi.png")
fig.savefig(FIG, bbox_inches="tight", dpi=150); plt.close(fig)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

# ---- HTML ----
rows = ""
for n in order:
    r = data["1929"][n]
    cls = "post" if r["fame"] > 1929 else ""
    rows += (f"<tr class={cls}><td>{r['cue']} &rarr; {r['sur']}</td><td>{r['domain']}</td><td>{r['fame']}</td>"
             f"<td class=n>{data['1900'][n]['pmi']:.1f}</td><td class=n>{data['1913'][n]['pmi']:.1f}</td><td class=n>{data['1929'][n]['pmi']:.1f}</td></tr>")
html = f"""<!doctype html><meta charset=utf-8><title>Name binding across domains and time</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:900px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.45rem}} h2{{font-size:1.12rem;margin-top:1.6rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.post td{{background:#fbeeee}} .cap{{color:#555;font-size:.9rem}} img{{max-width:100%}}</style>
<h1>Does the model bind a person&rsquo;s first name to their last? (across domains, 1900/1913/1929)</h1>
<p>The clean form of the combo idea: no generation, no judge. For each person we take
<b>log&nbsp;P(surname&nbsp;|&nbsp;first&nbsp;name) &minus; log&nbsp;P(surname&nbsp;|&nbsp;&ldquo;Mr.&rdquo;)</b> &mdash;
how much the first name lifts the surname above a generic person context. This removes the first-name-commonness
confound that makes the raw probability near-zero for &ldquo;Leo&rdquo; or &ldquo;Paul&rdquo;. Positive means the
model carries the full name as a bound unit; near zero means it does not.</p>
<img src="data:image/png;base64,{b64}">
<p class=cap>Rows by rise-to-fame year. Pink rows are post-1929 &mdash; a clean model should show ~0 there
for every column, 1929 included.</p>
<h2>Binding strength (nats), by model</h2>
<table><tr><th>first &rarr; last</th><th>domain</th><th>fame</th><th>1900</th><th>1913</th><th>1929</th></tr>{rows}</table>
"""
open(os.path.join(OUT, "namepair_pmi.html"), "w", encoding="utf-8").write(html)

print(f"\n{'first -> last':24} {'fame':>4} | {'1900':>6} {'1913':>6} {'1929':>6}")
for n in order:
    print(f"{data['1929'][n]['cue']+' -> '+data['1929'][n]['sur']:24} {FY[n]:>4} | "
          f"{data['1900'][n]['pmi']:>6.1f} {data['1913'][n]['pmi']:>6.1f} {data['1929'][n]['pmi']:>6.1f}")
print("\nwrote figures/fig_namepair_pmi.png and results/prominence/namepair_pmi.html")
