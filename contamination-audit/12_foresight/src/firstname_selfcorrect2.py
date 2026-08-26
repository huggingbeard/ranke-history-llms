# -*- coding: utf-8 -*-
"""Self-correction, robust version + reverse control.
ROBUSTNESS: instead of one decoy, prime each person with FOUR plausible wrong first names and average the reversion,
so R no longer depends on the single decoy I happened to pick. R = mean_k [ log P(true|primed w_k) - log P(w_k|primed w_k) ].
REVERSE CONTROL: also prime the TRUE name and measure how strongly the model echoes it (keep_true). The tell for a
name-agnostic model is that it echoes a WRONG name (keep_wrong) about as readily as the true one. So parrot = mean_k
P(w_k | primed w_k): high => the model just repeats whatever name you give it (no knowledge); low => it resists the
lie (knowledge). Known people should show R>0 AND low parrot; post-cutoff people R<=0 AND high parrot.
Run:  python src/firstname_selfcorrect2.py
"""
import os, json, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
import common

OUT = os.path.join(common.RESULTS, "prominence")
RAW = os.path.join(OUT, "firstname_selfcorrect2.json")
INSTRUCT = {"1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}
# gender, context, surname, true, [four plausible wrong same-language first names], birth
PEOPLE = [
 ("M", "the general who crowned himself Emperor of the French", "Bonaparte", "Napoleon", ["Lucien", "Louis", "Henri", "Pierre"], 1769),
 ("M", "the composer of the opera cycle the Ring of the Nibelung", "Wagner", "Richard", ["Rainer", "Reinhard", "Wilhelm", "Konrad"], 1813),
 ("M", "the statesman who unified Germany into the Reich", "Bismarck", "Otto", ["Oswald", "Konrad", "Wilhelm", "Ludwig"], 1815),
 ("M", "the author of Das Kapital", "Marx", "Karl", ["Konrad", "Klaus", "Kurt", "Ludwig"], 1818),
 ("M", "the author of War and Peace", "Tolstoy", "Leo", ["Anton", "Ivan", "Nikolai", "Boris"], 1828),
 ("M", "the philosopher who wrote Thus Spoke Zarathustra", "Nietzsche", "Friedrich", ["Rainer", "Gottfried", "Reinhard", "Johann"], 1844),
 ("F", "the physicist who discovered radium", "Curie", "Marie", ["Margot", "Madeleine", "Jeanne", "Louise"], 1867),
 ("M", "the leader of the Bolshevik Revolution", "Lenin", "Vladimir", ["Viktor", "Valentin", "Yuri", "Semyon"], 1870),
 ("M", "the physicist who developed the theory of relativity", "Einstein", "Albert", ["Ewald", "Adalbert", "Arnold", "Alwin"], 1879),
 ("M", "the Spanish painter who co-founded Cubism", "Picasso", "Pablo", ["Pedro", "Pascual", "Ramon", "Andres"], 1881),
 ("M", "the composer of the Rite of Spring", "Stravinsky", "Igor", ["Ivan", "Ilya", "Anton", "Semyon"], 1882),
 ("M", "the founder of Italian Fascism", "Mussolini", "Benito", ["Bruno", "Bernardo", "Benedetto", "Bartolomeo"], 1883),
 ("M", "the leader of the Nazi party in Munich", "Hitler", "Adolf", ["Alfred", "Arnold", "Albrecht", "Anton"], 1889),
 # post-cutoff controls
 ("M", "the leader of the Free French in the Second World War", "de Gaulle", "Charles", ["Claude", "Christophe", "Camille", "Clement"], 1890),
 ("M", "the American actor who became President in the 1980s", "Reagan", "Ronald", ["Donald", "Roland", "Reginald", "Russell"], 1911),
 ("F", "the woman who refused to give up her bus seat in Montgomery", "Parks", "Rosa", ["Ruth", "Rhoda", "Rebecca", "Dorothy"], 1913),
 ("M", "the author of The Stranger", "Camus", "Albert", ["Alain", "Armand", "Aurelien", "Adrien"], 1913),
 ("M", "the jazz pianist who composed Round Midnight", "Monk", "Thelonious", ["Theodore", "Thomas", "Terrence", "Tobias"], 1917),
 ("M", "the astronaut who first walked on the Moon", "Armstrong", "Neil", ["Nolan", "Norman", "Nathaniel", "Nelson"], 1930),
]
PRO = {"M": "his", "F": "her"}


def frame(g, ctx, sur, name):
    return f"People sometimes call {ctx} \"{name} {sur}\", but {PRO[g]} first name was really"


data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag, mid in INSTRUCT.items():
    if tag in data:
        continue
    print(f"[{tag}] loading {mid} ...", flush=True)
    tok, model = common.load(mid)
    row = {}
    for g, ctx, sur, tru, wrongs, b in PEOPLE:
        rs, reverts, keep_wrongs = [], [], []
        for w in wrongs:
            fw = frame(g, ctx, sur, w)
            p_t = common.slp(model, tok, fw, " " + tru)         # revert to true despite the lie
            p_w = common.slp(model, tok, fw, " " + w)           # echo the lie (parrot)
            rs.append(p_t - p_w); reverts.append(p_t); keep_wrongs.append(p_w)
        keep_true = common.slp(model, tok, frame(g, ctx, sur, tru), " " + tru)   # reverse control: echo the truth
        row[sur] = {"true": tru, "birth": b,
                    "R": round(float(np.mean(rs)), 3), "R_sd": round(float(np.std(rs)), 2),
                    "parrot_lp": round(float(np.mean(keep_wrongs)), 2), "keeptrue_lp": round(keep_true, 2)}
    data[tag] = row
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

surs = sorted(data["1929"], key=lambda s: data["1929"][s]["birth"])
DISP = {"Bonaparte": "Napoleon", "de Gaulle": "de Gaulle"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.linewidth": .8, "savefig.dpi": 200})
INK, GREEN, ROSE = "#1a1a1a", "#2e7d32", "#c0392b"
fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.8))
for j, m in enumerate(["1913", "1929"]):
    cut = int(m)
    X = np.array([data[m][s]["birth"] + 25 for s in surs])
    # --- top row: robust reversion R (averaged over 4 decoys) ---
    ax = axes[0, j]
    Y = np.array([data[m][s]["R"] for s in surs]); YC = np.clip(Y, -8.6, 8.6)
    ax.axhspan(0, 9.5, color=GREEN, alpha=.08, lw=0); ax.axhspan(-9.5, 0, color=ROSE, alpha=.06, lw=0)
    ax.axhline(0, color=INK, lw=1.2, ls=(0, (5, 2))); ax.axvline(cut, color="#888", ls="--", lw=1.4)
    rev = Y > 0
    ax.scatter(X[rev], YC[rev], s=52, color=GREEN, edgecolor="white", lw=.7, zorder=4)
    ax.scatter(X[~rev], YC[~rev], s=42, facecolor="white", edgecolor=ROSE, lw=1.1, zorder=4)
    t = [ax.text(X[k], YC[k], DISP.get(s, s), fontsize=6.8, color=INK, ha="center") for k, s in enumerate(surs)]
    adjust_text(t, ax=ax, arrowprops=dict(arrowstyle="-", color="#bbb", lw=.4), min_arrow_len=3)
    for k, s in enumerate(surs):
        if abs(Y[k]) > 8.6:
            ax.annotate(f"({Y[k]:+.0f})", (X[k], YC[k]), xytext=(0, -11), textcoords="offset points", fontsize=6, color=ROSE, ha="center")
    ax.set_title(f"Ranke {m}: robust reversion (4 decoys averaged)", fontsize=11, color="#b8332a" if m == "1913" else "#1f5f9e", fontweight="bold")
    ax.set_ylabel("R = logP(true) − logP(wrong)\n↑ reverts (knows) · ↓ parrots"); ax.set_ylim(-9.5, 9.5); ax.set_xlim(1785, 1990)
    ax.text(.02, .96, "reverts", transform=ax.transAxes, color=GREEN, fontsize=8.5, va="top", fontweight="bold")
    ax.text(.02, .04, "parrots the lie", transform=ax.transAxes, color=ROSE, fontsize=8.5, va="bottom")
    # --- bottom row: reverse control -- how strongly it echoes a WRONG name vs the TRUE one ---
    ax2 = axes[1, j]
    kw = np.array([np.exp(data[m][s]["parrot_lp"]) for s in surs]) * 100
    kt = np.array([np.exp(data[m][s]["keeptrue_lp"]) for s in surs]) * 100
    post = np.array([data[m][s]["birth"] + 25 > 1929 for s in surs])
    ax2.plot([kt, kw], [np.arange(len(surs))] * 2, color="#ccc", lw=.8, zorder=1)
    ax2.scatter(kt, np.arange(len(surs)), s=34, color=INK, zorder=3, label="echoes TRUE name")
    ax2.scatter(kw, np.arange(len(surs)), s=40, color=[ROSE if p else GREEN for p in post], zorder=3, label="echoes WRONG name")
    ax2.set_yticks(np.arange(len(surs))); ax2.set_yticklabels([f"{DISP.get(s, s)}" for s in surs], fontsize=6.6)
    ax2.set_title(f"Ranke {m}: reverse control — does it echo a WRONG name?", fontsize=11, color="#b8332a" if m == "1913" else "#1f5f9e", fontweight="bold")
    ax2.set_xlabel("P(echo the primed name)  %"); ax2.set_xlim(-2, 100)
    if j == 1:
        ax2.legend(loc="lower right", fontsize=7.5, frameon=False)
    for a in (ax, ax2):
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
        a.grid(axis="x", color="#eee", lw=.5)
fig.suptitle("Self-correction, robust (4 decoys) + reverse control.  Post-cutoff people parrot any name; known people resist the lie.",
             fontsize=12.5, y=0.99, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
FIG = os.path.join(common.FIGURES, "fig_firstname_selfcorrect2.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)

rows = ""
for s in surs:
    post = data["1929"][s]["birth"] + 25 > 1929
    rows += (f"<tr class={'post' if post else ''}><td>{s} &rarr; {data['1929'][s]['true']}</td><td class=n>{data['1929'][s]['birth']+25}</td>"
             f"<td class=n>{data['1913'][s]['R']:+.1f}</td><td class=n>{data['1929'][s]['R']:+.1f}</td>"
             f"<td class=n>{100*np.exp(data['1929'][s]['parrot_lp']):.0f}%</td><td class=n>{100*np.exp(data['1929'][s]['keeptrue_lp']):.0f}%</td></tr>")
open(os.path.join(OUT, "firstname_selfcorrect2.html"), "w", encoding="utf-8").write(f"""<!doctype html><meta charset=utf-8>
<title>Self-correction: robust + reverse control</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:1040px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.4rem}} .cap{{color:#555;font-size:.93rem}} img{{max-width:100%}}
table{{border-collapse:collapse;margin:1rem 0;width:100%;font-size:.9rem}} td,th{{border:1px solid #ccc;padding:3px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.post td{{background:#fbeeee}}</style>
<h1>Self-correction — robust, with a reverse control</h1>
<p>Robustness: each person is now primed with <b>four</b> plausible wrong first names, and the reversion R is averaged,
so it no longer hangs on the single decoy I picked. Reverse control: we also prime the <b>true</b> name and measure how
readily the model echoes it. The tell of a name-agnostic model is that it echoes a <em>wrong</em> name about as easily
as the true one &mdash; it is just repeating whatever you gave it. Known people should resist the lie (low echo of the
wrong name); post-cutoff people should parrot it.</p>
<img src="data:image/png;base64,{base64.b64encode(open(FIG,'rb').read()).decode()}">
<p class=cap>Top: robust reversion R (green = reverts to the truth, red = keeps the lie). Bottom: the reverse control —
black dot = how much it echoes the TRUE name when primed with it, coloured dot = how much it echoes a WRONG name when
primed with it. For post-cutoff people (rose) the two dots sit together on the right: it echoes a lie as readily as the
truth. For known people the wrong-name dot slides left: the model refuses to keep the lie.</p>
<table><tr><th>person &rarr; true</th><th>b+25</th><th>R 1913</th><th>R 1929</th><th>echoes wrong (1929)</th><th>echoes true (1929)</th></tr>{rows}</table>
""")
print(f"\n{'person -> true':24} {'b+25':>5} {'R1913':>7} {'R1929':>7} {'parrot%':>8} {'keeptrue%':>10}")
for s in surs:
    print(f"{s+' -> '+data['1929'][s]['true']:24} {data['1929'][s]['birth']+25:>5} {data['1913'][s]['R']:>+7.1f} {data['1929'][s]['R']:>+7.1f} "
          f"{100*np.exp(data['1929'][s]['parrot_lp']):>7.0f}% {100*np.exp(data['1929'][s]['keeptrue_lp']):>9.0f}%")
print("wrote figures/fig_firstname_selfcorrect2.png and results/prominence/firstname_selfcorrect2.html")
