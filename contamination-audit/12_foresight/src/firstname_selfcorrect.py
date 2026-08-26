# -*- coding: utf-8 -*-
"""SELF-CORRECTION probe (PI's idea). Hand the model a WRONG first name with the real surname and the real claim to
fame, and see if it overrides the prime and reverts to the true name. "People sometimes call the statesman who
unified Germany 'Oswald Bismarck', but his first name was really ___" -> does it say Otto?  This sidesteps every
probability-shift confound: it is self-normalising (true vs the SPECIFIC wrong name it was just given), so name
commonness and language priming cannot enter. Reversion R = log P(true) - log P(wrong) in the wrong-primed frame;
R>0 means the true name wins despite the prime = the model holds it. Post-cutoff people are the control -- with no
true knowledge there is nothing to revert to.  Run:  python src/firstname_selfcorrect.py
"""
import os, json, base64
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
import common

OUT = os.path.join(common.RESULTS, "prominence")
RAW = os.path.join(OUT, "firstname_selfcorrect.json")
INSTRUCT = {"1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}
# gender, context (claim to fame), surname, true first, WRONG plausible same-language first, birth
PEOPLE = [
 ("M", "the general who crowned himself Emperor of the French", "Bonaparte", "Napoleon", "Lucien", 1769),
 ("M", "the composer of the opera cycle the Ring of the Nibelung", "Wagner", "Richard", "Rainer", 1813),
 ("M", "the statesman who unified Germany into the Reich", "Bismarck", "Otto", "Oswald", 1815),
 ("M", "the author of Das Kapital", "Marx", "Karl", "Konrad", 1818),
 ("M", "the author of War and Peace", "Tolstoy", "Leo", "Anton", 1828),
 ("M", "the philosopher who wrote Thus Spoke Zarathustra", "Nietzsche", "Friedrich", "Rainer", 1844),
 ("F", "the physicist who discovered radium", "Curie", "Marie", "Margot", 1867),
 ("M", "the leader of the Bolshevik Revolution", "Lenin", "Vladimir", "Viktor", 1870),
 ("M", "the physicist who developed the theory of relativity", "Einstein", "Albert", "Ewald", 1879),
 ("M", "the Spanish painter who co-founded Cubism", "Picasso", "Pablo", "Pedro", 1881),
 ("M", "the composer of the Rite of Spring", "Stravinsky", "Igor", "Ivan", 1882),
 ("M", "the founder of Italian Fascism", "Mussolini", "Benito", "Bruno", 1883),
 ("M", "the leader of the Nazi party in Munich", "Hitler", "Adolf", "Alfred", 1889),
 # post-cutoff controls: the model cannot know these, so it should not revert
 ("M", "the leader of the Free French in the Second World War", "de Gaulle", "Charles", "Claude", 1890),
 ("M", "the American actor who became President in the 1980s", "Reagan", "Ronald", "Donald", 1911),
 ("F", "the woman who refused to give up her bus seat in Montgomery", "Parks", "Rosa", "Ruth", 1913),
 ("M", "the author of The Stranger", "Camus", "Albert", "Alain", 1913),
 ("M", "the jazz pianist who composed Round Midnight", "Monk", "Thelonious", "Theodore", 1917),
 ("M", "the astronaut who first walked on the Moon", "Armstrong", "Neil", "Nolan", 1930),
]
PRO = {"M": "his", "F": "her"}


def topk(model, tok, prompt, k=10):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, -1].float(), -1)
    v, i = lp.topk(40)
    out = []
    for p, t in zip(v.tolist(), i.tolist()):
        w = tok.decode([t]).strip()
        if w.isalpha() and len(w) > 1:
            out.append((w, float(np.exp(p))))
        if len(out) >= k:
            break
    return out


def frame(g, ctx, sur, wrong):
    return f"People sometimes call {ctx} \"{wrong} {sur}\", but {PRO[g]} first name was really"


def noprime(ctx):
    return f"The first name of {ctx} was"


data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag, mid in INSTRUCT.items():
    if tag in data:
        continue
    print(f"[{tag}] loading {mid} ...", flush=True)
    tok, model = common.load(mid)
    row = {}
    for g, ctx, sur, tru, wrong, b in PEOPLE:
        fr = frame(g, ctx, sur, wrong)
        p_t = common.slp(model, tok, fr, " " + tru)
        p_w = common.slp(model, tok, fr, " " + wrong)
        p_t0 = common.slp(model, tok, noprime(ctx), " " + tru)      # no-prime baseline: does context alone give it
        row[sur] = {"true": tru, "wrong": wrong, "birth": b,
                    "R": round(p_t - p_w, 3), "p_true_lp": round(p_t, 2), "p_wrong_lp": round(p_w, 2),
                    "noprime_true_lp": round(p_t0, 2), "top": topk(model, tok, fr, 8)}
    data[tag] = row
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

surs = sorted(data["1929"].keys(), key=lambda s: data["1929"][s]["birth"])
DISP = {"Bonaparte": "Napoleon", "de Gaulle": "de Gaulle"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.linewidth": .8, "savefig.dpi": 200})
INK, GREEN, ROSE = "#1a1a1a", "#2e7d32", "#c0392b"
fig, axes = plt.subplots(2, 1, figsize=(12, 9.6), sharex=True)
for ax, (m, cut, col, name) in zip(axes, [("1913", 1913, "#b8332a", "Ranke 1913"), ("1929", 1929, "#1f5f9e", "Ranke 1929")]):
    X = np.array([data[m][s]["birth"] + 25 for s in surs])
    Y = np.array([data[m][s]["R"] for s in surs])
    YC = np.clip(Y, -8.6, 8.6)                                  # clip the Monk outlier so the action stays readable
    ax.axhspan(0, 9.5, color=GREEN, alpha=.08, lw=0); ax.axhspan(-9.5, 0, color=ROSE, alpha=.06, lw=0)
    ax.axhline(0, color=INK, lw=1.3, ls=(0, (5, 2)))
    ax.axvline(cut, color=col, ls="--", lw=1.6)
    rev = Y > 0
    ax.scatter(X[rev], YC[rev], s=60, color=GREEN, edgecolor="white", lw=.8, zorder=4)
    ax.scatter(X[~rev], YC[~rev], s=48, facecolor="white", edgecolor=ROSE, lw=1.2, zorder=4)
    texts = [ax.text(X[k], YC[k], f"{DISP.get(s, s)}·{data[m][s]['true']}", fontsize=7.3, color=INK, ha="center") for k, s in enumerate(surs)]
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#aaa", lw=.5), expand=(1.25, 1.6), min_arrow_len=3)
    for k, s in enumerate(surs):
        if abs(Y[k]) > 8.6:
            ax.annotate(f"(R={Y[k]:+.0f}, off scale)", (X[k], YC[k]), xytext=(0, -12), textcoords="offset points", fontsize=6.6, color=ROSE, ha="center")
    ax.annotate(f"{name} · trained to {cut}", (.985, .96), xycoords="axes fraction", color=col, fontsize=12.5, ha="right", va="top", fontweight="bold")
    ax.text(.012, .96, "REVERTS to the true name  (holds it)", transform=ax.transAxes, color=GREEN, fontsize=9, va="top", fontweight="bold")
    ax.text(.012, .035, "keeps the wrong prime  (does not hold it)", transform=ax.transAxes, color=ROSE, fontsize=9, va="bottom")
    ax.set_ylabel("reversion  R = log P(true) − log P(wrong)\nin the wrong-primed frame")
    ax.set_xlim(1785, 1990); ax.set_ylim(-9.5, 9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#eee", lw=.6)
axes[1].set_xlabel("year the person turned 25   (birth year + 25 — when they came of age)")
fig.suptitle("Self-correction: given a WRONG first name, does the model revert to the truth?   Ranke instruct models.",
             fontsize=13, y=0.975, x=0.02, ha="left")
fig.text(0.02, 0.938, "Prompt: “People sometimes call {the claim to fame} ‘{Wrong} {Surname}’, but his first name was really ___”. "
         "R>0 (green) = the true name beats the wrong prime = the model holds it. Post-cutoff people are the control.",
         fontsize=8.8, color="#555", ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.925))
FIG = os.path.join(common.FIGURES, "fig_firstname_selfcorrect.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)

def toprow(m, s):
    return ", ".join(f"{w} {100*p:.0f}%" for w, p in data[m][s]["top"][:5])
rows = ""
for s in surs:
    post = data["1929"][s]["birth"] + 25 > 1929
    rows += (f"<tr class={'post' if post else ''}><td>{s} &rarr; {data['1929'][s]['true']} <span style='color:#b00'>(primed: {data['1929'][s]['wrong']})</span></td>"
             f"<td class=n>{data['1929'][s]['birth']+25}</td><td class=n>{data['1913'][s]['R']:+.1f}</td><td class=n>{data['1929'][s]['R']:+.1f}</td>"
             f"<td style='font-size:.82rem'>{toprow('1929', s)}</td></tr>")
open(os.path.join(OUT, "firstname_selfcorrect.html"), "w", encoding="utf-8").write(f"""<!doctype html><meta charset=utf-8>
<title>Self-correction probe</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:1040px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.4rem}} .cap{{color:#555;font-size:.93rem}} img{{max-width:100%}}
table{{border-collapse:collapse;margin:1rem 0;width:100%;font-size:.9rem}} td,th{{border:1px solid #ccc;padding:3px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.post td{{background:#fbeeee}}</style>
<h1>Does the model self-correct a wrong first name?</h1>
<p>We hand the model the wrong first name with the real surname and the real claim to fame, then ask for the first
name: <i>&ldquo;People sometimes call the statesman who unified Germany &lsquo;Oswald Bismarck&rsquo;, but his first
name was really&nbsp;___&rdquo;</i>. If the model holds the true name it reverts (&rarr; Otto); if it does not know
the person it has nothing to revert to. The score is self-normalising &mdash; true name vs the <em>specific wrong
name it was just given</em> &mdash; so name commonness and language priming, the confounds that broke the earlier
probes, cannot enter. R = log P(true) &minus; log P(wrong); R&gt;0 = reverts.</p>
<img src="data:image/png;base64,{base64.b64encode(open(FIG,'rb').read()).decode()}">
<p class=cap>Green = reverts to the true name despite the wrong prime; rose = keeps the wrong name. The last column
is what the 1929 model actually predicts after the prime (top completions, with probabilities) &mdash; read it to
see the correction happen or fail. Post-cutoff rows are pink.</p>
<table><tr><th>person &rarr; true (primed with wrong)</th><th>b+25</th><th>R 1913</th><th>R 1929</th><th>1929 top predictions after the wrong prime</th></tr>{rows}</table>
""")
print(f"\n{'person -> true (wrong)':38} {'b+25':>5} {'R1913':>7} {'R1929':>7}   1929 top-3 after prime")
for s in surs:
    print(f"{s+' -> '+data['1929'][s]['true']+' ('+data['1929'][s]['wrong']+')':38} {data['1929'][s]['birth']+25:>5} "
          f"{data['1913'][s]['R']:>+7.1f} {data['1929'][s]['R']:>+7.1f}   {toprow('1929', s)}")
print("wrote figures/fig_firstname_selfcorrect.png and results/prominence/firstname_selfcorrect.html")
