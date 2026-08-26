# -*- coding: utf-8 -*-
"""Analyse the saved Hitler-probe generations (results/hitler_probe/generations.json). No GPU.

The coarse buckets are built from hand-checkable keyword tags (printed below so they can be audited),
and every rate is shown beside verbatim examples so the reader can verify the tagging by eye. Reading
the raw text established two things the first pass got wrong and that this script encodes:
  - the 1929 model's political knowledge is broader than "Nazi/putsch" wording (radical movement,
    editor, Reichstag, Bavaria, rise-to-power), so we tag on that wider set;
  - the 1929 "chancellor" mentions are CONFABULATED (appointed by "Emperor William" or "King Ludwig II"),
    not real knowledge of the January-1933 Chancellorship -- so they are not counted as post-cutoff
    knowledge; they are flagged separately and quoted.
Run:  python src/hitler_analyze.py
"""
import os, re, json, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common

OUT = os.path.join(common.RESULTS, "hitler_probe")
d = json.load(open(os.path.join(OUT, "generations.json"), encoding="utf-8"))
PROMPTS = {"who_is": "Who is Adolf Hitler?", "def": "Adolf Hitler is"}
MODELS = ["1913", "1929"]
N = 100
INK, BLUE, RED, GREY, GREEN = "#1a1a1a", "#2b6cb0", "#c0392b", "#9aa0a6", "#2e7d32"

# --- hand-checkable tags (audit them against the verbatim examples below) ---
SITUATES = (r"bavaria|bavarian|munich|münchen|reichstag|augsburg|nazi|national ?socialis|"
            r"völkisch|volkisch|putsch|agitat|orator|fatherland|weimar|hindenburg|ludendorff|"
            r"radical movement|revolutionary movement|rise[sn]? (up|to)|ruler of germany|"
            r"politics of germany|german (military|political)|leader of the (german|nazi|national|movement)")
GENERIC = (r"new york|america|united states|\bu\.?s\.?\b|hungary|playwright|pneumonia|piano|"
           r"mayor|chicago|boston|newburg|broadway|\bactor\b|great business|peasant|congregational|"
           r"dollars|\$|madison square")
CONFAB_CHANCELLOR = r"chancellor|kanzler"          # in this sample set, always confabulated (see quotes)
REAL_1933 = r"1933|seized power|reichstag fire|enabling act|president hindenburg"  # true 1933 facts (should be ~0)


def tag(text):
    t = text.lower()
    if re.search(SITUATES, t):
        return "situates the real Hitler (German radical politics)"
    if re.search(GENERIC, t):
        return "empty name → generic person (often American)"
    return "other / vague confabulation"


ORDER = ["situates the real Hitler (German radical politics)",
         "empty name → generic person (often American)",
         "other / vague confabulation"]

dist = {p: {m: {b: 0 for b in ORDER} for m in MODELS} for p in PROMPTS}
rate_sit = {p: {} for p in PROMPTS}
for p in PROMPTS:
    for m in MODELS:
        gs = d[m][p][:N]
        for g in gs:
            dist[p][m][tag(g)] += 1
        rate_sit[p][m] = dist[p][m][ORDER[0]]

# --- honest check: does either model state the REAL 1933 Chancellorship as fact? ---
print("=== Does any continuation state the true 1933 Chancellorship (1933 / seized power / Hindenburg-appointed)? ===")
for m in MODELS:
    for p in PROMPTS:
        hits = [g for g in d[m][p][:N] if re.search(REAL_1933, g.lower())]
        for g in hits:
            print(f"  [{m}/{p}] {g[:160]}")
print("(if the lists above are empty, neither frozen model knows the real 1933 event)")

print("\n=== 1929 'chancellor' mentions — confabulated, not knowledge (quoted so you can judge) ===")
for g in d["1929"]["def"][:N] + d["1929"]["who_is"][:N]:
    if re.search(CONFAB_CHANCELLOR, g.lower()):
        print(f"  * {g[:150]}")

# --- the headline comparison ---
print("\n=== Distribution of 100 free continuations, by model (prompt 'Adolf Hitler is') ===")
print(f"{'bucket':52} 1913  1929")
for b in ORDER:
    print(f"{b:52} {dist['def']['1913'][b]:3d}%  {dist['def']['1929'][b]:3d}%")

# ================= figure: grouped bars, both prompts =================
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True)
short = ["situates the real Hitler\n(German radical politics)", "empty name → generic\nperson (often American)",
         "other / vague\nconfabulation"]
for ax, p in zip(axes, ["def", "who_is"]):
    x = np.arange(len(ORDER)); w = 0.38
    v13 = [dist[p]["1913"][b] for b in ORDER]
    v29 = [dist[p]["1929"][b] for b in ORDER]
    ax.bar(x - w/2, v13, w, color=GREY, edgecolor=INK, lw=.7, label="Ranke 1913")
    ax.bar(x + w/2, v29, w, color=BLUE, edgecolor=INK, lw=.7, label="Ranke 1929")
    for i, (a, b) in enumerate(zip(v13, v29)):
        ax.annotate(f"{a}", (i - w/2, a + 1), ha="center", fontsize=8)
        ax.annotate(f"{b}", (i + w/2, b + 1), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8.2)
    ax.set_title(f"prompt: “{PROMPTS[p]}”", fontsize=10)
    ax.set_ylim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GREY, alpha=.3, lw=.6)
axes[0].set_ylabel("share of 100 free continuations (%)")
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
fig.suptitle("Ranke 1913 vs 1929, asked “who is Adolf Hitler?”, 100 free continuations each.\n"
             "In 1913 the name is empty (generic confabulation); by 1929 the model situates the real Munich agitator.",
             fontsize=11, y=1.06)
fig.tight_layout()
FIGPATH = os.path.join(common.FIGURES, "fig_hitler.png")
fig.savefig(FIGPATH, bbox_inches="tight", dpi=150)
_b64 = base64.b64encode(open(FIGPATH, "rb").read()).decode()   # embed the figure in the HTML (self-contained)
print("\nwrote figures/fig_hitler.png")

# ================= didactic HTML =================
def rows(p):
    return "\n".join(f"<tr><td>{b}</td><td class=n>{dist[p]['1913'][b]}%</td>"
                     f"<td class=n>{dist[p]['1929'][b]}%</td></tr>" for b in ORDER)


def examples(m, p, k=6):
    return "\n".join(f"<li>{g[:240]}</li>" for g in d[m][p][:k])


html = f"""<!doctype html><meta charset=utf-8><title>Ranke 1913 vs 1929 — who is Adolf Hitler?</title>
<style>body{{font:16px/1.55 Georgia,serif;max-width:900px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:1.8rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}} td,th{{border:1px solid #ccc;padding:5px 9px;text-align:left}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} li{{margin:.35rem 0}} .cap{{color:#555;font-size:.9rem}}
img{{max-width:100%}} .k{{background:#f4f6f8;padding:.1rem .3rem;border-radius:3px;font-family:monospace;font-size:.85rem}}</style>
<h1>Two frozen readers, one question: “who is Adolf Hitler?”</h1>
<p>We asked two time-locked language models the same question, one hundred times each, and let them
continue freely (sampling temperature&nbsp;0.8). <b>Ranke&nbsp;1913</b> has read nothing printed after 1913;
<b>Ranke&nbsp;1929</b> nothing after 1929. In 1913 Adolf Hitler was a twenty-four-year-old unknown in Vienna,
rejected from art school. By 1929 he led the National Socialist party, had written <i>Mein Kampf</i> (1925)
and stood trial for the 1923 Munich putsch — but was <b>not</b> yet Chancellor (that came in January&nbsp;1933),
and the war and the Holocaust were still in the future.</p>

<h2>What each model says he is (100 free continuations, prompt “Adolf Hitler is”)</h2>
<table><tr><th>bucket</th><th>Ranke 1913</th><th>Ranke 1929</th></tr>
{rows('def')}</table>
<p class=cap>Buckets are mutually exclusive and sum to 100. Tags are keyword rules, printed in
<span class=k>src/hitler_analyze.py</span>, and every rate can be checked against the quotations below.</p>
<img src="data:image/png;base64,{_b64}" alt="distribution of continuations by model and prompt">
<p class=cap>x-axis: the three buckets a continuation can fall in. y-axis: share of the 100 continuations
in that bucket. Grey = the 1913 reader, blue = the 1929 reader. Left panel: the definitional prompt;
right panel: the question as asked.</p>

<h2>Ranke 1913 — the name means nothing (six verbatim continuations)</h2>
<ul>{examples('1913','def')}</ul>

<h2>Ranke 1929 — the real Munich agitator (six verbatim continuations)</h2>
<ul>{examples('1929','def')}</ul>

<h2>The honest caveat: neither model knows 1933</h2>
<p>Where the 1929 model says “chancellor”, it is guessing, not remembering: it has him “made chancellor
by King Ludwig&nbsp;II” (who died in 1886) or “called to that office by Emperor William”. No continuation from
either model states the real January-1933 Chancellorship, the Reichstag fire, or anything of the war. The
1929 model knows the rising agitator of the 1920s; it does not know how the story ends — which is exactly
the boundary the models were built to hold.</p>
"""
open(os.path.join(OUT, "hitler_report.html"), "w", encoding="utf-8").write(html)
print("wrote results/hitler_probe/hitler_report.html")
