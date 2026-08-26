# -*- coding: utf-8 -*-
"""First-name probe as a MULTIPLE CHOICE (free completion was swamped by generic first-name priors like
'John'). Surname+role pin the person; the model chooses the first name from {true + 4 common distractors}.
"The Spanish painter Picasso's first name was ___": Pablo / John / William / Henry / James.
P(true) via softmax over the options (chance = 20%). Ranke 1900/1913/1929. Writes firstname_fc.json,
figures/fig_firstname_fc.png, self-contained HTML.  Run:  python src/firstname_fc.py
"""
import os, json, math, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common
from foresight_bank import RANKE
from firstname_probe import PEOPLE   # reuse the same battery (module import does not run its main under __main__)

OUT = os.path.join(common.RESULTS, "prominence")
RAW = os.path.join(OUT, "firstname_fc.json")
MODELS = ["1900", "1913", "1929"]
POOL = ["John", "William", "Charles", "Henry", "James", "George", "Louis", "Karl", "Wilhelm",
        "Jean", "Giuseppe", "Ivan", "Albert", "Friedrich", "Franz", "Wolfgang", "Paul", "Richard"]


def softmax(xs):
    m = max(xs); ex = [math.exp(x - m) for x in xs]; return [e / sum(ex) for e in ex]


data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag in MODELS:
    if tag in data:
        continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    row = {}
    for dom, role, sur, first, fy in PEOPLE:
        prompt = f"The {role} {sur}'s first name was"
        distr = [d for d in POOL if d.lower() != first.lower()][:4]
        opts = [first] + distr
        p = softmax([common.slp(model, tok, prompt, " " + o) for o in opts])
        row[sur] = {"first": first, "fame": fy, "domain": dom, "opts": opts,
                    "p_true": round(p[0], 3), "pick": opts[max(range(len(opts)), key=lambda k: p[k])]}
    data[tag] = row
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

order = sorted(PEOPLE, key=lambda p: p[4])
M = np.array([[100 * data[t][p[2]]["p_true"] for t in MODELS] for p in order])
fig, ax = plt.subplots(figsize=(6.9, 9.2))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels([f"Ranke {m}" for m in MODELS], fontsize=10)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{p[1]} {p[2]} → {p[3]}?  ({p[4]})" for p in order], fontsize=8.1)
for i in range(len(order)):
    for j in range(3):
        v = M[i, j]
        ax.annotate(f"{v:.0f}", (j, i), ha="center", va="center", fontsize=7.6, color="white" if v > 55 else "#333")
ax.set_title("First name of the (role + surname), multiple choice.  P(correct), % (chance = 20).\n"
             "Surname+role pin the person; the model picks the first name. Rows by rise-to-fame year.",
             fontsize=9.6, loc="left")
fig.tight_layout(); FIG = os.path.join(common.FIGURES, "fig_firstname_fc.png")
fig.savefig(FIG, bbox_inches="tight", dpi=150); plt.close(fig)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

rows = ""
for p in order:
    dom, role, sur, first, fy = p
    cls = "post" if fy > 1929 else ""
    rows += (f"<tr class={cls}><td>{role} {sur}</td><td>{first}</td><td>{fy}</td>"
             f"<td class=n>{100*data['1900'][sur]['p_true']:.0f}%</td><td class=n>{100*data['1913'][sur]['p_true']:.0f}%</td>"
             f"<td class=n>{100*data['1929'][sur]['p_true']:.0f}%</td><td>{data['1929'][sur]['pick']}</td></tr>")
html = f"""<!doctype html><meta charset=utf-8><title>First name of the (role+surname)? multiple choice</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:940px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.42rem}} h2{{font-size:1.1rem;margin-top:1.6rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.post td{{background:#fbeeee}} .cap{{color:#555;font-size:.9rem}} img{{max-width:100%}}</style>
<h1>What is the first name of &hellip;? (multiple choice)</h1>
<p>Surname + a period-neutral role pin the person; the model chooses the first name from the true one plus
four common distractors (chance&nbsp;=&nbsp;20%). Prompt: &ldquo;<i>The &lt;role&gt; &lt;surname&gt;&rsquo;s
first name was ___</i>&rdquo;. Ranke 1900 / 1913 / 1929. No first-name-commonness confound, no judge.</p>
<img src="data:image/png;base64,{b64}">
<p class=cap>Rows by rise-to-fame year; pink = post-1929. A clean model should sit at chance there in every
column, 1929 included.</p>
<h2>P(correct first name), multiple choice</h2>
<table><tr><th>role + surname</th><th>first</th><th>fame</th><th>1900</th><th>1913</th><th>1929</th><th>1929 pick</th></tr>{rows}</table>
"""
open(os.path.join(OUT, "firstname_fc.html"), "w", encoding="utf-8").write(html)
print(f"\n{'role+surname -> first':30} {'fame':>4} | {'1900':>6} {'1913':>6} {'1929':>6}  (chance 20%)")
for p in order:
    dom, role, sur, first, fy = p
    print(f"{sur+' -> '+first:30} {fy:>4} | {100*data['1900'][sur]['p_true']:>5.0f}% {100*data['1913'][sur]['p_true']:>5.0f}% {100*data['1929'][sur]['p_true']:>5.0f}%")
print("\nwrote figures/fig_firstname_fc.png and results/prominence/firstname_fc.html")
