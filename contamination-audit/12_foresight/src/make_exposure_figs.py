# -*- coding: utf-8 -*-
"""Two views of meter vs exposure, now including the INVENTED treaties (PI request).
Panel A: scatter — every treaty by exposure (x, log) and meter (y). Invented ones have no exposure, so they land in the
"absent from the record" strip on the left, next to the real treaties that also never made the books.
Panel B: violins — the meter's distribution for three piles: invented (never existed), real-but-unknown (real, but the
model can't answer), real-known. The key read is whether the real-but-unknown violin sits HIGHER than the invented one:
if it does, the meter is reacting to "this is a real thing" beyond pure memory (the realness leak).
Run:  python src/make_exposure_figs.py
"""
import os, json, time, random, unicodedata, urllib.request, urllib.parse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import common
import treaties_battery as B

OUT = os.path.join(common.RESULTS, "superlatives")
CACHE = os.path.join(OUT, "ngram_cache.json")
CHANCE = 1.0 / len(B.PARTY_OPTIONS)
ANGLO = {"Björkö": ["Bjorko", "Bjoerkoe"], "Küçük Kaynarca": ["Kuchuk Kainarji", "Kutchuk Kainardji", "Kainarji"],
         "Hünkâr İskelesi": ["Unkiar Skelessi", "Hunkiar Iskelesi"], "Åbo": ["Abo"], "Niš": ["Nissa", "Nish"],
         "Żurawno": ["Zurawno"], "Zsitvatorok": ["Sitvatorok"], "Nöteborg": ["Noteborg"], "Zuhab": ["Kasr-i-Shirin"],
         "Turkmenchay": ["Turkmanchai"], "Gulistan": ["Golestan"], "Karlowitz": ["Carlowitz"], "Nystad": ["Nystadt"]}
cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}


def freq(phrase):
    if phrase in cache:
        return cache[phrase]
    q = urllib.parse.urlencode({"content": phrase, "year_start": 1800, "year_end": 1928, "corpus": "en-2019", "smoothing": 3})
    req = urllib.request.Request("https://books.google.com/ngrams/json?" + q, headers={"User-Agent": "academic-research (joachimvoth@gmail.com)"})
    for t in range(6):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=30))
            v = (sum(d[0]["timeseries"]) / len(d[0]["timeseries"])) if d else 0.0
            cache[phrase] = v; time.sleep(0.8 + random.uniform(0, 0.5)); return v
        except Exception:
            time.sleep((2 ** t) + random.uniform(0, 1))
    raise RuntimeError(phrase)


def asciifold(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def exposure(name):
    vs = {name, asciifold(name)} | set(ANGLO.get(name, []))
    return max(freq("Treaty of " + v) for v in vs)


# ---- meter (v2) for every item ----
X = np.load(os.path.join(OUT, "memory_detector.npz"))["X"]
meta = json.load(open(os.path.join(OUT, "memory_detector_meta.json"), encoding="utf-8"))
bestL = json.load(open(os.path.join(OUT, "memory_detector_summary.json"), encoding="utf-8"))["bestL"]


def label(m):
    if m["kind"] == "invented":
        return "invented", 0
    kd = m["ptrue"] > 0.5
    kp = (m["parties"] is not None) and (m["parties"] > 3 * CHANCE)
    if kd or kp:
        return "known", 1
    if m["peak"] < 0.25 and not kp:
        return "unknown", 0
    return "ambiguous", -1


lab = [label(m)[0] for m in meta]
mem = np.array([label(m)[1] for m in meta])
clean = mem >= 0
pipe = make_pipeline(StandardScaler(), PCA(n_components=15, random_state=0), LogisticRegression(C=1.0, max_iter=3000))
oof = cross_val_predict(pipe, X[clean, bestL, :], mem[clean], cv=5, method="predict_proba")[:, 1]
clf = pipe.fit(X[clean, bestL, :], mem[clean])
meter = np.empty(len(meta)); meter[clean] = oof; meter[~clean] = clf.predict_proba(X[~clean, bestL, :])[:, 1]
print(f"[expo] querying exposure for all {len(meta)} items (cached {len(cache)}) ...", flush=True)
expo = np.array([exposure(m["name"]) for m in meta])
json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
cat = np.array([("invented" if m["kind"] == "invented" else lab[i]) for i, m in enumerate(meta)])

CATCOL = {"known": "#2e7d32", "unknown": "#c0392b", "invented": "#3b6ea5", "ambiguous": "#b8b8b8"}
frac = {g: (meter[cat == g] > .5).mean() for g in ("invented", "unknown", "known")}
print("fraction called 'remembered':  " + "   ".join(f"{g} {100*frac[g]:.0f}% (n={int((cat==g).sum())})" for g in ("invented", "unknown", "known")))

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.8), gridspec_kw={"width_ratios": [1.55, 1]})

# ---- Panel A: scatter ----
nz = expo > 0
lx = np.log10(expo[nz])
b, a = np.polyfit(lx, meter[nz], 1); r = float(np.corrcoef(lx, meter[nz])[0, 1])
floor = lx.min() - 1.7; div = lx.min() - 0.8
rng = np.random.default_rng(0)
xs = np.linspace(lx.min() - 0.2, lx.max() + 0.2, 50)
ax[0].plot(xs, np.clip(a + b * xs, -.02, 1.02), color="#1f5f9e", lw=2.2, zorder=2)
ax[0].axvspan(floor - 0.7, div, color="#f4f4f4", zorder=0); ax[0].axvline(div, color="#ccc", lw=1, ls=":")
ax[0].annotate("never in the record\n(exposure = 0)", (floor - 0.05, 0.55), fontsize=8, color="#888", ha="center")
for i in range(len(meta)):
    x = np.log10(expo[i]) if expo[i] > 0 else floor + rng.uniform(-0.55, 0.55)
    ax[0].scatter(x, meter[i], s=46, color=CATCOL[cat[i]], edgecolor="#111", lw=.4, alpha=.85,
                  marker="o" if cat[i] != "invented" else "^", zorder=5)
for n, txt in [("Küçük Kaynarca", "real, unknown — but IS in the record"), ("Polyanovka", "real, unknown — ZERO exposure (leak)"),
               ("Hünkâr İskelesi", "most-seen unknown — meter missed it")]:
    i = next(k for k, m in enumerate(meta) if m["name"] == n)
    x = np.log10(expo[i]) if expo[i] > 0 else floor
    ax[0].annotate(f"{n}\n{txt}", (x, meter[i]), xytext=(7, -3 if meter[i] > .5 else 8), textcoords="offset points", fontsize=6.9, color="#333")
ax[0].set_xlabel("how often “Treaty of X” appears in books before 1929  —  log scale (← vanishing · common →)")
ax[0].set_ylabel("memory-meter (v2)  —  0 = made-up, 1 = real memory")
ax[0].set_ylim(-.04, 1.06); ax[0].set_xlim(floor - 0.8, lx.max() + 0.5)
ax[0].set_title(f"A.  Every treaty by exposure and meter (fit r = {r:.2f})", fontsize=10.6, loc="left")
h = [Line2D([0], [0], color="#1f5f9e", lw=2, label=f"fit (r={r:.2f})"),
     Line2D([0], [0], marker="o", color="w", markerfacecolor=CATCOL["known"], markeredgecolor="#111", ms=9, label="real & known"),
     Line2D([0], [0], marker="o", color="w", markerfacecolor=CATCOL["unknown"], markeredgecolor="#111", ms=9, label="real but unknown"),
     Line2D([0], [0], marker="^", color="w", markerfacecolor=CATCOL["invented"], markeredgecolor="#111", ms=9, label="invented (never existed)")]
ax[0].legend(handles=h, loc="lower right", frameon=False, fontsize=8.4)
for sp in ("top", "right"):
    ax[0].spines[sp].set_visible(False)
ax[0].grid(color="#eee", lw=.5)

# ---- Panel B: violins of the meter by pile ----
groups = [("invented", "invented\n(never existed)"), ("unknown", "real but\nUNKNOWN"), ("known", "real &\nknown")]
data = [meter[cat == g] for g, _ in groups]
parts = ax[1].violinplot(data, showmeans=False, showextrema=False, widths=0.85)
for pc, (g, _) in zip(parts["bodies"], groups):
    pc.set_facecolor(CATCOL[g]); pc.set_alpha(0.28); pc.set_edgecolor(CATCOL[g])
for i, (g, _) in enumerate(groups):
    yv = meter[cat == g]; xx = 1 + i + rng.uniform(-0.11, 0.11, len(yv))
    ax[1].scatter(xx, yv, s=22, color=CATCOL[g], edgecolor="white", lw=.3, alpha=.8, zorder=5)
    ax[1].plot([1 + i - 0.42, 1 + i + 0.42], [np.median(yv)] * 2, color="#111", lw=2, zorder=6)
    ax[1].annotate(f"{100*frac[g]:.0f}% called\n'remembered'", (1 + i, 1.02), fontsize=8, ha="center", color=CATCOL[g])
ax[1].axhline(0.5, color="#999", lw=.8, ls=":")
ax[1].set_xticks([1, 2, 3]); ax[1].set_xticklabels([t for _, t in groups], fontsize=9)
ax[1].set_ylim(-.04, 1.15); ax[1].set_ylabel("memory-meter (v2)")
ax[1].set_title("B.  Meter distribution per pile — black bar = median", fontsize=10.6, loc="left")
ax[1].annotate("both are behaviourally blank —\nbut 'real but unknown' rides higher\n(the realness leak)", (1.5, 0.30),
               fontsize=8.2, color="#a11", ha="center")
for sp in ("top", "right"):
    ax[1].spines[sp].set_visible(False)
ax[1].grid(axis="y", color="#eee", lw=.5)
fig.suptitle("Figure 14.  Where invented treaties sit, and how the meter treats them vs. real-but-unknown ones. Ranke-4B-1929.",
             fontsize=12, y=1.0, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
FIG = os.path.join(common.FIGURES, "fig_exposure_violins.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
print(f"medians: invented {np.median(data[0]):.2f}  unknown {np.median(data[1]):.2f}  known {np.median(data[2]):.2f}")
print(f"wrote {FIG}")
