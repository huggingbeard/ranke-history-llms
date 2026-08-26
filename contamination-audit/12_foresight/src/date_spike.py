# -*- coding: utf-8 -*-
"""Flat-vs-spike contamination test (PI's insight). If a frozen model truly does not know a post-cutoff event, its
guess for the event's YEAR should be flat over the plausible window -- 1916 as likely as 1917 as likely as 1920. A
sharp spike on the true year is knowledge the model should not have. For each event we score P(year) over a +/-8-year
window, and report: P(true year), whether the true year is the argmax, and the spike ratio = P(true) / mean(the other
years in the window). Pre-1913 events are controls (the 1913 model SHOULD spike). Post-1913 events are the test.
Run:  python src/date_spike.py
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

OUT = os.path.join(common.RESULTS, "superlatives")
INSTRUCT = {"1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}
# label, prompt (ends so the next token is the year), true year, era
EVENTS = [
 ("Queen Victoria died", "Queen Victoria died in the year", 1901, "pre1913"),
 ("Russo-Japanese War ended", "The Russo-Japanese War came to an end in the year", 1905, "pre1913"),
 ("San Francisco earthquake", "The great San Francisco earthquake struck in the year", 1906, "pre1913"),
 ("Titanic sank", "The ocean liner Titanic struck an iceberg and sank in the year", 1912, "pre1913"),
 ("First World War began", "The First World War, the Great War, broke out in the year", 1914, "post1913"),
 ("Tsar overthrown", "The Russian Tsar was overthrown and the monarchy abolished in the year", 1917, "post1913"),
 ("Bolsheviks seized power", "The Bolsheviks seized power in Russia in the year", 1917, "post1913"),
 ("US entered the war", "The United States entered the World War in the year", 1917, "post1913"),
 ("Armistice / war ended", "The World War finally ended with an armistice in the year", 1918, "post1913"),
 ("Spanish flu pandemic", "The great influenza pandemic swept the world in the year", 1918, "post1913"),
 ("German hyperinflation", "The German mark collapsed in a ruinous hyperinflation in the year", 1923, "post1929"),
 ("Lindbergh crossed Atlantic", "Charles Lindbergh flew solo across the Atlantic in the year", 1927, "post1929"),
 ("Wall Street Crash", "The Wall Street stock market crashed catastrophically in the year", 1929, "post1929"),
]
WIN = 8


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


RAW = os.path.join(OUT, "date_spike.json")
data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag, mid in INSTRUCT.items():
    if tag in data:
        continue
    print(f"[{tag}] loading {mid} ...", flush=True)
    tok, model = common.load(mid)
    rows = {}
    for label, prompt, ty, era in EVENTS:
        years = list(range(ty - WIN, ty + WIN + 1))
        p = softmax([common.slp(model, tok, prompt, f" {y}") for y in years])
        pt = p[years.index(ty)]
        others = [p[i] for i, y in enumerate(years) if y != ty]
        rows[label] = {"true": ty, "era": era, "p_true": round(pt, 4),
                       "argmax_year": years[int(np.argmax(p))], "spike": round(pt / (np.mean(others) + 1e-9), 1),
                       "dist": {y: round(pv, 4) for y, pv in zip(years, p)}}
    data[tag] = rows
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

# ---- figure: spike ratio per event, 1913 vs 1929 ----
labels = [e[0] for e in EVENTS]
eras = [e[3] for e in EVENTS]
ECOL = {"pre1913": "#2e7d32", "post1913": "#c0392b", "post1929": "#7a2f8f"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(11.5, 6.6))
y = np.arange(len(labels))
for i, m in enumerate(["1913", "1929"]):
    sp = [min(data[m][l]["spike"], 60) for l in labels]
    ax.barh(y + (0.5 - i) * 0.4, sp, 0.36, color=("#b8332a" if m == "1913" else "#1f5f9e"),
            alpha=.9, label=f"Ranke {m}")
ax.axvline(1, color="#444", lw=1, ls=":")
ax.axvline(WIN * 2 + 1, color="#999", lw=.8, ls="--")
ax.annotate("flat = 1× (no knowledge)", (1.3, len(labels) - .4), fontsize=8, color="#444")
ax.set_yticks(y)
ax.set_yticklabels([f"{l}  ({data['1913'][l]['true']})" for l in labels], fontsize=9)
for j, l in enumerate(labels):
    ax.get_yticklabels()[j].set_color(ECOL[eras[j]])
ax.invert_yaxis()
ax.set_xlabel("spike ratio  =  P(true year) / mean(other years in ±8 window)   [capped at 60, log-ish]")
ax.set_xscale("log"); ax.set_xlim(0.5, 70)
ax.set_title("Flat-vs-spike contamination test: does the frozen model spike on an event's true year?\n"
             "green = pre-1913 (both should spike) · red = 1914-1918 · purple = 1920s. 1× = flat = no knowledge.", fontsize=10.5, loc="left")
ax.legend(loc="lower right", frameon=False)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_date_spike.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)

print(f"\n{'event':30}{'true':>5} {'era':>9} | {'1913 P/argmax/spike':>26} | {'1929 P/argmax/spike':>26}")
for l in labels:
    a, b = data["1913"][l], data["1929"][l]
    print(f"{l:30}{a['true']:>5} {a['era']:>9} | {100*a['p_true']:>5.0f}% {a['argmax_year']:>6} {a['spike']:>6.0f}x   | "
          f"{100*b['p_true']:>5.0f}% {b['argmax_year']:>6} {b['spike']:>6.0f}x")
print(f"\nwrote {FIG}")
