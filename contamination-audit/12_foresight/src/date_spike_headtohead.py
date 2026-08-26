# -*- coding: utf-8 -*-
"""Head-to-head: Talkie-1930-13B (base) vs Ranke-1929-4B (base), identical date-spike contamination set, P(true) metric.
Both are ~1929-30 time-locked BASE models from different labs, so the fair question is: on events AFTER both cutoffs
(WW2, the atomic bomb, civil rights), which model leaks, and by how much? Events 1914-1929 are inside both windows and
serve as the validation (both should know them). One shared leakage threshold = 95th percentile of the pooled fictional
events of both models. Reads date_spike_talkie.json (RunPod) + date_spike_base_ptrue.json (local).
Run:  python src/date_spike_headtohead.py
"""
import os, json, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import common
import events_dates_big as E

OUT = os.path.join(common.RESULTS, "superlatives")
tk = json.load(open(os.path.join(OUT, "date_spike_talkie_big.json"), encoding="utf-8"))
if "events" not in tk:
    raise SystemExit("Talkie run not finished / errored — see date_spike_talkie_big.json:\n" + json.dumps(tk)[:800])
T = tk["events"]
R = json.load(open(os.path.join(OUT, "date_spike_base_big.json"), encoding="utf-8"))["1929"]["events"]
CUT = {"Talkie": 1930, "Ranke": 1929}
GCOL = {"1914-29": "#e08214", "1930-45": "#c0392b", "1946-69": "#7a2f8f", "1970-2020": "#0f7d7d"}

null_p = [T[k]["p_true"] * 100 for k in T if T[k]["kind"] == "null"] + [R[k]["p_true"] * 100 for k in R if R[k]["kind"] == "null"]
THR = float(np.percentile(null_p, 95))
CHANCE = 100.0 / (2 * E.WIN + 1)
kmT = np.median([T[k]["p_true"] * 100 for k in T if T[k]["kind"] == "known"])
kmR = np.median([R[k]["p_true"] * 100 for k in R if R[k]["kind"] == "known"])

# ---------- Figure: Talkie vs Ranke P(true) scatter ----------
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(9.6, 9.2))
ax.plot([-3, 103], [-3, 103], color="#bbb", lw=1, ls="--", zorder=1)                 # agree diagonal
ax.axvline(THR, color="#111", lw=.9, ls=":"); ax.axhline(THR, color="#111", lw=.9, ls=":")
ax.axvline(CHANCE, color="#1f5f9e", lw=.8, ls=(0, (1, 2))); ax.axhline(CHANCE, color="#1f5f9e", lw=.8, ls=(0, (1, 2)))
ax.axhspan(-3, THR, color="#8a8f98", alpha=.06, lw=0); ax.axvspan(-3, THR, color="#8a8f98", alpha=.06, lw=0)
for lab, prompt, ty, grp in E.TEST:
    k = "test::" + lab
    rp, tp = R[k]["p_true"] * 100, T[k]["p_true"] * 100
    post_both = ty > 1930
    ax.scatter(rp, tp, s=95 if post_both else 45, color=GCOL[grp], marker="o" if post_both else "s",
               edgecolor="#111" if post_both else "white", lw=.9, zorder=5, alpha=.95 if post_both else .6)
    if post_both and (rp > THR or tp > THR):
        ax.annotate(f"{lab} {ty}", (rp, tp), xytext=(6, 3), textcoords="offset points", fontsize=7.2, color="#111")
ax.annotate(f"chance {CHANCE:.0f}%", (CHANCE + 1, 101), color="#1f5f9e", fontsize=8, va="top")
ax.annotate(f"threshold {THR:.0f}%", (THR + 1.5, 101), color="#111", fontsize=8, va="top", ha="left")
ax.annotate("both know\n(1914–29 · validation)", (86, 70), fontsize=8.5, color="#666", ha="center")
ax.annotate("Talkie knows the late-1920s\n(hyperinflation, Lindbergh, the Crash);\nRanke has gaps here",
            (2, 44), fontsize=8.3, color="#7a2f8f", ha="left")
ax.annotate("both clean\n(post-cutoff, at chance)", (24, 26), fontsize=8.5, color="#666", ha="center")
ax.annotate("← Ranke leaks these; Talkie stays clean", (30, 17.5), fontsize=8.6, color="#a11", ha="left", fontweight="bold")
ax.set_xlim(-3, 108); ax.set_ylim(-3, 108)
ax.set_xlabel("Ranke-1929 (base)   —   P(true year), %")
ax.set_ylabel("Talkie-1930 (base)   —   P(true year), %")
ax.set_title("Figure 3.  Two time-locked base models on the identical date set: post-cutoff leakage, head to head.",
             fontsize=11.5, loc="left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(color="#eee", lw=.5)
handles = [Line2D([0], [0], color="#bbb", ls="--", label="agree (y = x)"),
           Line2D([0], [0], color="#111", ls=":", label=f"leakage threshold ({THR:.0f}%)"),
           Line2D([0], [0], color="#1f5f9e", ls=(0, (1, 2)), label=f"chance ({CHANCE:.0f}%)"),
           Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markeredgecolor="#111", markersize=10, label="event after BOTH cutoffs (the test)"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor="#888", markersize=9, label="event 1914–29 (inside both windows: validation)"),
           Patch(fc="#e08214", label="era: 1914–29"), Patch(fc="#c0392b", label="era: 1930–45 (WW2)"),
           Patch(fc="#7a2f8f", label="era: 1946–69"), Patch(fc="#0f7d7d", label="era: 1970–2020")]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.2)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_date_spike_headtohead_big.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)

# ---------- table + HTML ----------
def cls(model, k, cut):
    v = (T if model == "Talkie" else R)[k]; s = v["p_true"] * 100
    if v["true"] <= cut:
        return ("knows", "knows ✓") if s > THR else ("gap", "—")
    return ("leaked", "LEAKED") if s > THR else ("clean", "clean")


ERAS = [("1914-29", "Inside both windows (1914–29) — validation &amp; coverage"),
        ("1930-45", "WW2 era (1930–45) — after both cutoffs"),
        ("1946-69", "Cold War, space, civil rights (1946–69) — after both cutoffs"),
        ("1970-2020", "1970–2020 — after both cutoffs")]
rows = ""
for grp, gname in ERAS:
    rows += f"<tr class=grp><td colspan=6>{gname}</td></tr>"
    for lab, prompt, ty, g in E.TEST:
        if g != grp:
            continue
        k = "test::" + lab
        cr, lr = cls("Ranke", k, 1929); ct, lt = cls("Talkie", k, 1930)
        rows += (f"<tr><td>{lab}</td><td class=n>{ty}</td>"
                 f"<td class='n {cr}'>{100*R[k]['p_true']:.0f}%</td><td class='{cr}'>{lr}</td>"
                 f"<td class='n {ct}'>{100*T[k]['p_true']:.0f}%</td><td class='{ct}'>{lt}</td></tr>")
open(os.path.join(OUT, "date_spike_headtohead_big.html"), "w", encoding="utf-8").write(f"""<!doctype html><meta charset=utf-8>
<title>Talkie vs Ranke — date-spike head-to-head</title>
<style>body{{font:15.5px/1.6 Georgia,serif;max-width:1020px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.45rem}} .cap{{color:#555;font-size:.93rem}} img{{max-width:100%}}
table{{border-collapse:collapse;margin:1rem 0;width:100%;font-size:.92rem}} td,th{{border:1px solid #ccc;padding:4px 9px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} .leaked{{background:#fbecec;color:#a11;font-weight:bold}}
.clean{{color:#2e7d32}} .knows{{color:#1f5f9e}} .gap{{color:#bbb}} tr.grp td{{background:#f0f0f2;font-weight:bold}}
.hdr{{color:#666;font-size:.85rem}}</style>
<h1>Two time-locked base models, one contamination set: Talkie-1930 vs Ranke-1929</h1>
<p class=hdr>Talkie-1930-13B base (RunPod) vs Ranke-4B-1929 base (local) · identical date-spike set (src/events_dates.py)
· P(true year) metric · shared threshold {THR:.0f}% (95th pct of {len(null_p)} pooled fictional events) · 2026-07-08</p>
<p>Both are base models frozen around 1929&ndash;30, from different labs. On events <em>inside</em> both windows
(1914&ndash;1929) they should &mdash; and do &mdash; know the dates, which validates the test: known-fact median is
{kmR:.0f}% (Ranke) and {kmT:.0f}% (Talkie), fictional events sit at chance. The real question is the events
<em>after</em> both cutoffs.</p>
<img src="data:image/png;base64,{base64.b64encode(open(FIG,'rb').read()).decode()}">
<p class=cap>Figure 3. Each point is one dated event: Ranke-1929 P(true) on the x-axis, Talkie-1930 P(true) on the y.
Squares are 1914&ndash;29 events (inside both windows, validation); circles are events after both cutoffs (the test).
The grey corner is &ldquo;both clean&rdquo; (both at chance); the top-right is &ldquo;both leak&rdquo;. A point on the
dashed diagonal means the two models leak it equally; off-diagonal means one leaks more.</p>
<h2 style="font-size:1.1rem">Every event, both models</h2>
<table><tr><th>event</th><th>year</th><th>Ranke-1929 P(true)</th><th>status</th><th>Talkie-1930 P(true)</th><th>status</th></tr>{rows}</table>
""")
print(f"shared threshold {THR:.0f}%  |  known median: Ranke {kmR:.0f}% Talkie {kmT:.0f}%")
print(f"{'event':26}{'yr':>5}{'Ranke':>8}{'Talkie':>8}")
for lab, prompt, ty, g in E.TEST:
    k = "test::" + lab
    print(f"{lab:26}{ty:>5}{100*R[k]['p_true']:>7.0f}%{100*T[k]['p_true']:>7.0f}%")
print(f"wrote {FIG} and date_spike_headtohead_big.html")
