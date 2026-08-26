# -*- coding: utf-8 -*-
"""Prominence report: triangulates three judge passes to separate real knowledge from two confounds.
  naive  ("<Name> was")               -> hijacked by same-name ordinary people (Wagner the bicycle thief);
  primed ("The <role> <Name> was")    -> recovers real knowledge, but lets post-1929 people pass as a
                                          GENERIC period person of that role (vaudeville "Elvis");
  strict (era-specific YES on primed)  -> collapses that generic confabulation.
Figures: acquisition heatmap (primed), matched trios (primed), and a 'real vs confabulated' panel
(primed-lenient vs era-strict at 1929). Self-contained HTML.  Run:  python src/prominence_report.py
"""
import os, json, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
from prominence_bank import BATTERY, MODELS, TRIOS

OUT = os.path.join(common.RESULTS, "prominence")
gens = json.load(open(os.path.join(OUT, "gens.json"), encoding="utf-8"))
NAI = json.load(open(os.path.join(OUT, "judged.json"), encoding="utf-8"))
LEN = json.load(open(os.path.join(OUT, "judged_primed.json"), encoding="utf-8"))
STR = json.load(open(os.path.join(OUT, "judged_primed_strict.json"), encoding="utf-8"))
FY = {b[0]: b[2] for b in BATTERY}


def rate(jd, gkey, hold, tag, name):
    vs = [jd.get(f"{tag}|{name}|{i}") for i in range(len(gens[tag][name][gkey]))]
    vs = [v for v in vs if v]
    return 100 * vs.count(hold) / len(vs) if vs else float("nan")

def lenr(t, n): return rate(LEN, "gens_primed", "SAME", t, n)
def strr(t, n): return rate(STR, "gens_primed", "YES", t, n)
def naiv(t, n): return rate(NAI, "gens", "SAME", t, n)

order = sorted([b[0] for b in BATTERY], key=lambda n: FY[n])
POST = [b[0] for b in BATTERY if b[2] > 1929]

# ---- Fig 1: acquisition heatmap (primed-lenient) ----
M = np.array([[lenr(t, n) for t in MODELS] for n in order])
fig, ax = plt.subplots(figsize=(6.8, 9.4))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels([f"Ranke {m}" for m in MODELS], fontsize=10)
ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{n}  ({FY[n]})" for n in order], fontsize=8.5)
for i in range(len(order)):
    post = FY[order[i]] > 1929
    for j in range(3):
        v = M[i, j]
        ax.annotate(f"{v:.0f}" + ("*" if post else ""), (j, i), ha="center", va="center", fontsize=8,
                    color="white" if v > 55 else "#333")
ax.set_title("Holds the person (primed frame, lenient judge, % of 50).  * = post-1929 (see below:\n"
             "these are plausible period CONFABULATION, not real knowledge).", fontsize=9.8, loc="left")
fig.tight_layout(); F1 = os.path.join(common.FIGURES, "fig_prominence_heat.png")
fig.savefig(F1, bbox_inches="tight", dpi=150); plt.close(fig)

# ---- Fig 2: matched trios (primed-lenient) ----
fig, axes = plt.subplots(3, 4, figsize=(15, 9), sharey=True)
x = [1900, 1913, 1929]
for ax, (dom, a, b, c) in zip(axes.flat, TRIOS):
    for name, col in [(a, "#2e7d32"), (b, "#2b6cb0"), (c, "#c0392b")]:
        ax.plot(x, [lenr(t, name) for t in MODELS], "-o", color=col, lw=2, ms=5,
                label=f"{name.split()[-1]} ({FY[name]})")
    ax.set_title(dom, fontsize=10.5); ax.set_ylim(-3, 103)
    ax.set_xticks(x); ax.set_xticklabels(x, fontsize=8)
    ax.legend(fontsize=7.3, frameon=False, loc="center left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ccc", alpha=.4, lw=.5)
for ax in axes.flat[len(TRIOS):]:
    ax.axis("off")
for r in range(3):
    axes[r, 0].set_ylabel("holds (primed, %)")
fig.suptitle("Matched trios (primed frame): green pre-1900, blue 1900-29 switch on by 1913/29;\n"
             "red post-1929 look non-zero here only by generic role-confabulation (next figure strips it).",
             fontsize=11.3, y=1.01)
fig.tight_layout(); F2 = os.path.join(common.FIGURES, "fig_prominence_pairs.png")
fig.savefig(F2, bbox_inches="tight", dpi=150); plt.close(fig)

# ---- Fig 3: real vs confabulated (lenient vs strict) at 1929 ----
sel = ["Otto von Bismarck", "Richard Wagner", "Albert Einstein", "Vladimir Lenin", "Marie Curie",
       "Ronald Reagan", "Elvis Presley", "Mao Zedong", "Rosa Parks", "Neil Armstrong", "Jonas Salk"]
x = np.arange(len(sel)); w = 0.38
fig, ax = plt.subplots(figsize=(12.6, 4.8))
ax.bar(x - w/2, [lenr("1929", n) for n in sel], w, color="#2b6cb0", edgecolor="#1a1a1a", lw=.6, label="primed, lenient judge")
ax.bar(x + w/2, [strr("1929", n) for n in sel], w, color="#c0392b", edgecolor="#1a1a1a", lw=.6, label="primed, era-specific judge")
ax.axvline(4.5, color="#999", ls="--", lw=1)
ax.annotate("known to 1929", (2, 96), ha="center", fontsize=9, color="#555")
ax.annotate("post-1929 (should be unknown)", (7.5, 96), ha="center", fontsize=9, color="#555")
ax.set_xticks(x); ax.set_xticklabels([n.split()[-1] for n in sel], rotation=30, fontsize=8.5)
ax.set_ylabel("holds at Ranke 1929 (%)"); ax.set_ylim(0, 100)
ax.set_title("Under the lenient judge (blue), post-1929 people (Reagan, Elvis, Mao) score as high as the\n"
             "genuinely known (Lenin, Einstein, Curie): the metric cannot tell knowledge from fluent\n"
             "confabulation. The era-strict judge (red) is harsh on everyone -- only the TEXT tells them apart.",
             fontsize=9.4, loc="left")
ax.legend(frameon=False, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#9aa0a6", alpha=.3, lw=.6)
fig.tight_layout(); F3 = os.path.join(common.FIGURES, "fig_prominence_realvsconfab.png")
fig.savefig(F3, bbox_inches="tight", dpi=150); plt.close(fig)

b1, b2, b3 = (base64.b64encode(open(f, "rb").read()).decode() for f in (F1, F2, F3))

def ex(name, k=3):
    outs = []
    for i, g in enumerate(gens["1929"][name]["gens_primed"]):
        if len(outs) >= k:
            break
        outs.append(f"<li>[{STR.get(f'1929|{name}|{i}','?')}] &ldquo;{name} was {g[:135]}&rdquo;</li>")
    return "\n".join(outs)

html = f"""<!doctype html><meta charset=utf-8><title>Prominence across time</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:1000px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:1.8rem;border-bottom:1px solid #ddd}} h3{{font-size:1rem}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} li{{margin:.25rem 0;font-size:13px}} .cap{{color:#555;font-size:.9rem}} img{{max-width:100%}}</style>
<h1>Prominence across time: which people does each frozen reader know?</h1>
<p>Matched domain trios (pre-1900 / 1900-29 / post-1929), 50 free continuations of &ldquo;<i>&lt;Name&gt; was</i>&rdquo;
per person per model, judged by gemini-3.1-flash-lite. Measuring name-knowledge is confounded twice, so we
triangulate three passes: the <b>naive</b> frame is hijacked by ordinary same-name people (&ldquo;Richard
Wagner was arrested for stealing a bicycle&rdquo;); a role-<b>primed</b> frame fixes that but lets post-1929
people pass as a <i>generic</i> period person of that role (a vaudeville &ldquo;Elvis&rdquo;); the
<b>era-specific</b> judge collapses that confabulation.</p>
<h2>Acquisition map (primed frame)</h2>
<img src="data:image/png;base64,{b1}">
<p class=cap>Rows by fame year; columns are the models. Real staircase for pre-1900 and 1900-29 people. The
starred post-1929 rows look non-zero only by role-confabulation &mdash; see the last figure.</p>
<h2>Matched trios</h2>
<img src="data:image/png;base64,{b2}">
<h2>The metric&rsquo;s blind spot: plausible confabulation scores like knowledge</h2>
<img src="data:image/png;base64,{b3}">
<p class=cap>Blue = lenient judge; red = era-specific judge, both on the primed frame at Ranke 1929. The
lenient number cannot separate the genuinely known (Lenin, Einstein) from the confabulated post-1929
(Reagan, Elvis, Mao) &mdash; both sit near 50&ndash;60%. The strict judge is harsh on everyone. The only
reliable signal is reading the text, below: real era-specific detail (Wagner at Leipzig) versus a fluent
period impostor (a vaudeville Elvis). This is the contamination problem in miniature: plausible &ne; known.</p>
<h2>The two failure modes, in the model&rsquo;s words (Ranke 1929, primed)</h2>
<h3>Richard Wagner &mdash; real knowledge recovered</h3><ul>{ex("Richard Wagner")}</ul>
<h3>Elvis Presley &mdash; generic period confabulation (a 1920s vaudeville singer)</h3><ul>{ex("Elvis Presley")}</ul>
<h3>Ronald Reagan &mdash; a generic stage actor, not the president</h3><ul>{ex("Ronald Reagan")}</ul>
"""
open(os.path.join(OUT, "prominence_report.html"), "w", encoding="utf-8").write(html)
print("wrote 3 figures + results/prominence/prominence_report.html")
print("\npost-1929: lenient -> era-strict at 1929 (the collapse):")
for n in POST:
    print(f"  {n:20} {lenr('1929',n):>4.0f}% -> {strr('1929',n):>4.0f}%")
