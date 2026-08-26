# -*- coding: utf-8 -*-
"""Resolver for obscure_real.py: some real-but-date-blank treaties still read HIGH on the meter (Küçük Kaynarca 0.99,
Hünkâr İskelesi 1.00). Two readings: (A) the meter is fooled by realness, or (B) the model truly remembers the treaty
but can't date it. Decide with an INDEPENDENT memory probe that is NOT the year: ask the model WHO SIGNED each treaty —
a forced choice over a fixed set of great-power pairs. If a treaty the meter calls "remembered" also has its correct
signatories recalled, the meter was right (memory present, just undatable). If the meter says "remembered" but the model
can't name the parties either, the meter was fooled.
Run:  python src/obscure_real_parties.py
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

OUT = os.path.join(common.RESULTS, "superlatives")
MID = "uzh-echist-org/Ranke-4B-1929"
STEM = "The Treaty of {} was signed between"
OPTIONS = ["Russia and the Ottoman Empire", "Russia and Persia", "Russia and Sweden", "Russia and Poland",
           "Russia and China", "Germany and Russia", "Sweden and Denmark", "the Ottoman Empire and Austria",
           "the Ottoman Empire and Persia", "Prussia and Austria", "France and China",
           "the United States and China", "Sweden and Novgorod"]
# true signatories (index into OPTIONS); treaties with awkward/multilateral parties are left out of this check
TRUE = {"Andrusovo": 3, "Roskilde": 6, "Stolbovo": 2, "Passarowitz": 7, "Belgrade": 7, "Küçük Kaynarca": 0,
        "Jassy": 0, "Gulistan": 1, "Turkmenchay": 1, "Hünkâr İskelesi": 0, "Whampoa": 10, "Wanghia": 11,
        "Aigun": 4, "Björkö": 5, "Nöteborg": 12, "Zuhab": 8, "Hubertusburg": 9, "Bucharest": 0}
# (dropped: Nemours = French civil war, Tientsin = multilateral — no clean two-power option)


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


JS = os.path.join(OUT, "obscure_real_parties.json")
if not os.path.exists(JS):
    print(f"[parties] loading {MID} ...", flush=True)
    tok, model = common.load(MID)
    out = {}
    for city, ti in TRUE.items():
        p = softmax([common.slp(model, tok, STEM.format(city), " " + o) for o in OPTIONS])
        out[city] = {"p_true_parties": round(p[ti], 3), "argmax": OPTIONS[int(np.argmax(p))], "true": OPTIONS[ti]}
    del tok, model; common.free()
    json.dump(out, open(JS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
P = json.load(open(JS, encoding="utf-8"))

# join with the meter / date knowledge from obscure_real
O = json.load(open(os.path.join(OUT, "obscure_real_summary.json"), encoding="utf-8"))["treaties"]
CH = 100.0 / len(OPTIONS)
rows = []
for c, ti in TRUE.items():
    rows.append((c, O[c]["ptrue"] * 100, O[c]["meter"], P[c]["p_true_parties"] * 100, O[c]["class"]))
rows.sort(key=lambda r: -r[2])

print(f"chance on the parties question = {CH:.0f}%\n")
print(f"{'treaty':17}{'meter':>7}{'P(year)':>9}{'P(parties)':>12}   verdict")
for c, py, m, pp, cl in rows:
    if m > 0.5 and pp > 3 * CH:
        v = "meter RIGHT: real memory, just can't date it"
    elif m > 0.5 and pp <= 3 * CH:
        v = "meter FOOLED: no memory of parties either"
    elif m <= 0.5 and pp <= 3 * CH:
        v = "agree: genuinely blank"
    else:
        v = "meter MISS: knows parties but meter says made-up"
    print(f"{c:17}{m:>7.2f}{py:>8.0f}%{pp:>11.0f}%   {v}")

# the two headline cases the PI asked about
key = ["Küçük Kaynarca", "Hünkâr İskelesi", "Björkö", "Belgrade"]
print("\nthe cases in question:")
for c in key:
    print(f"  {c:16} meter {O[c]['meter']:.2f} | can't date it (P{100*O[c]['ptrue']:.0f}%) | knows who signed it? P {100*P[c]['p_true_parties']:.0f}% (chance {CH:.0f}%)  [model says: {P[c]['argmax']}]")

# correlation: does the meter track this INDEPENDENT memory signal?
mv = np.array([r[2] for r in rows]); pv = np.array([r[3] for r in rows])
corr = float(np.corrcoef(mv, pv)[0, 1])
hi = pv > 3 * CH
print(f"\ncorr(meter, parties-knowledge) = {corr:.2f}")
print(f"meter mean where model KNOWS the parties: {mv[hi].mean():.2f} (n={int(hi.sum())})  |  where it DOESN'T: {mv[~hi].mean():.2f} (n={int((~hi).sum())})")
json.dump({"chance": CH, "corr_meter_parties": corr,
           "meter_when_parties_known": float(mv[hi].mean()), "meter_when_parties_unknown": float(mv[~hi].mean()),
           "n_parties_known": int(hi.sum()), "rows": {c: {"meter": m, "p_year": py, "p_parties": pp} for c, py, m, pp, cl in rows}},
          open(os.path.join(OUT, "obscure_real_parties_summary.json"), "w"), indent=1)

# ---- figure: meter vs an independent memory signal (the parties), for real-but-hard treaties ----
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(figsize=(9.8, 6.8))
ax.axvline(3 * CH, color="#999", lw=.9, ls=":")
ax.annotate("← model can't name\nthe signatories", (3 * CH - 2, .5), fontsize=8, color="#777", ha="right")
ax.annotate("model knows who\nsigned it →", (3 * CH + 2, .5), fontsize=8, color="#2e7d32")
for c, py, m, pp, cl in rows:
    col = "#2e7d32" if pp > 3 * CH else "#c0392b"
    ax.scatter(pp, m, s=75, color=col, edgecolor="#111", lw=.6, zorder=5)
    ax.annotate(c, (pp, m), xytext=(5, 3), textcoords="offset points", fontsize=7.0, color="#333")
ax.set_xlabel("independent memory check — P(correct signatories), %   (chance ≈ {:.0f}%)".format(CH))
ax.set_ylabel("memory-meter   —   0 = made-up,  1 = real memory")
ax.set_ylim(-0.03, 1.03)
ax.set_title("Figure 11.  Resolving the hard cases with a second, independent memory probe.\n"
             "The meter agrees with 'does the model know who signed it' — a fact that has nothing to do with the year.", fontsize=10.2, loc="left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.grid(color="#eee", lw=.5)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_obscure_real_parties.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"wrote {FIG} and obscure_real_parties_summary.json")
