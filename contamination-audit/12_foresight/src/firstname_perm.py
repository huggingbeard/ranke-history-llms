# -*- coding: utf-8 -*-
"""WITHIN-LANGUAGE permutation cutoff for the probability-shift (boost). The confound was that a German frame lifts
every German name, a Russian frame every Russian name. So we judge each true name ONLY against wrong names of the
SAME language and gender, in same-language frames. Null = boost of every wrong (same-language surname -> same-language
name) pair; the permutation p-value of a true pair = fraction of that null at least as high. Cutoff = the 95th
percentile of the language's own null. This holds language fixed, so the cutoff is clean and per-language.
Uses the cached pool boosts (no GPU).  Run:  python src/firstname_perm.py
"""
import os, json, base64
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

OUT = os.path.join(common.RESULTS, "prominence")
data = json.load(open(os.path.join(OUT, "firstname_pmi2.json"), encoding="utf-8"))
FEM = {"Curie", "Parks"}
LANG = {s: data["1929"][s]["lang"] for s in data["1929"]}
GEN = lambda s: "F" if s in FEM else "M"
LNAME = {"de": "German", "en": "English", "fr": "French", "ru": "Russian", "es": "Spanish", "it": "Italian"}
DISP = {"Bonaparte": "Napoleon", "Guevara": "Che"}
surs = sorted(data["1929"], key=lambda s: data["1929"][s]["birth"])
groups = defaultdict(list)
for s in surs:
    groups[(LANG[s], GEN(s))].append(s)
# order groups by size (big, informative nulls first), keep only those we can permute or note as singletons
gorder = sorted(groups, key=lambda k: -len(groups[k]))

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.linewidth": .8, "savefig.dpi": 200})
INK, GREEN, ROSE, GREY = "#1a1a1a", "#2e7d32", "#c0392b", "#9aa0a6"
res = {}
fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.4), sharey=True)
for ax, m in zip(axes, ["1913", "1929"]):
    res[m] = {}
    yticks, ylabels = [], []
    for gi, gk in enumerate(gorder):
        members = groups[gk]
        y0 = len(gorder) - gi
        null = np.array([data[m][s]["pool"][n] for s in members for n in data[m][s]["pool"] if n != data[m][s]["first"]])
        c95, c99 = np.percentile(null, 95), np.percentile(null, 99)
        # null cloud (jittered), reproducible jitter from index parity (no RNG allowed)
        jit = (np.arange(len(null)) % 11 - 5) / 22.0
        ax.scatter(null, y0 + jit, s=3, color=GREY, alpha=.18, lw=0, zorder=1)
        ax.plot([c95, c95], [y0 - .42, y0 + .42], color=INK, lw=1.4, ls=(0, (4, 2)), zorder=3)
        ax.plot([c99, c99], [y0 - .32, y0 + .32], color=INK, lw=.8, ls=(0, (1, 2)), alpha=.6, zorder=3)
        for s in members:
            tb = data[m][s]["pool"][data[m][s]["first"]]
            p = float((null >= tb).mean())
            res[m][s] = {"boost": tb, "p": p, "c95": float(c95), "clears": tb >= c95}
            clr = GREEN if tb >= c95 else ROSE
            ax.scatter(tb, y0, s=64, color=clr, edgecolor="white", lw=.8, zorder=6)
            ax.annotate(f"{DISP.get(s, s)}·{data[m][s]['first']}", (tb, y0), xytext=(0, 7 if surs.index(s) % 2 == 0 else -13),
                        textcoords="offset points", fontsize=6.6, color=INK, ha="center", zorder=7)
        yticks.append(y0); ylabels.append(f"{LNAME[gk[0]]} {gk[1]}\n({len(members)} ppl, {len(null)} null)")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel("probability-shift  boost (nats)")
    ax.set_title(f"Ranke {m}  (trained to {m})", fontsize=12.5, color="#b8332a" if m == "1913" else "#1f5f9e", fontweight="bold")
    ax.set_xlim(-8, 11); ax.set_ylim(0.3, len(gorder) + .8)
    ax.axvline(0, color=GREY, lw=.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="x", color="#eee", lw=.6)
axes[0].annotate("grey cloud = within-language wrong pairs (the permutation null)\ndashed line = 95th percentile = knowledge cutoff\n"
                 "green dot = true name clears it, red = does not", (0.02, 0.02), xycoords="axes fraction",
                 fontsize=8, color="#444", va="bottom")
fig.suptitle("Within-language permutation: does the true first name beat wrong names of the SAME language?",
             fontsize=13.5, y=0.98, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
FIG = os.path.join(common.FIGURES, "fig_firstname_perm.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)

rows = ""
for s in surs:
    post = data["1929"][s]["birth"] + 25 > 1929
    a, b = res["1913"][s], res["1929"][s]
    rows += (f"<tr class={'post' if post else ''}><td>{s} &rarr; {data['1929'][s]['first']} <span style='color:#888'>({LNAME[LANG[s]]})</span></td>"
             f"<td class=n>{data['1929'][s]['birth']+25}</td>"
             f"<td class=n>{a['boost']:+.1f}</td><td class=n>{'p='+format(a['p'],'.2f')}{' ✓' if a['clears'] else ''}</td>"
             f"<td class=n>{b['boost']:+.1f}</td><td class=n>{'p='+format(b['p'],'.2f')}{' ✓' if b['clears'] else ''}</td></tr>")
open(os.path.join(OUT, "firstname_perm.html"), "w", encoding="utf-8").write(f"""<!doctype html><meta charset=utf-8>
<title>Within-language permutation cutoff</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:1040px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.4rem}} .cap{{color:#555;font-size:.93rem}} img{{max-width:100%}}
table{{border-collapse:collapse;margin:1rem 0;width:100%;font-size:.9rem}} td,th{{border:1px solid #ccc;padding:3px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.post td{{background:#fbeeee}}</style>
<h1>A clean cutoff for the probability-shift: within-language permutation</h1>
<p>The shift in probability when the famous surname is named is the signal. The only question was the cutoff. We
build it by permutation, <em>within language</em>: for each language we collect the boost of every wrong (same-language
surname&nbsp;&rarr;&nbsp;same-language name) pair &mdash; that is the grey cloud, the null of &ldquo;no true link.&rdquo;
A true name counts as knowledge only if its boost lands in the top 5% of its own language&rsquo;s cloud (dashed line).
The p-value is the exact permutation rank. No time is assumed, and German names are judged only against German names.</p>
<img src="data:image/png;base64,{base64.b64encode(open(FIG,'rb').read()).decode()}">
<p class=cap>Left = 1913 model, right = 1929. Each row is a language&times;gender group; the grey cloud is its
within-language null of wrong pairs; the dashed line is the 95th percentile. Green true names clear it (significant
knowledge, p&lt;0.05), red do not. Singleton groups (one person of that language&times;gender) fall back to that one
frame&rsquo;s wrong names.</p>
<table><tr><th>person &rarr; name</th><th>b+25</th><th>boost 1913</th><th>perm p 1913</th><th>boost 1929</th><th>perm p 1929</th></tr>{rows}</table>
""")
print(f"\n{'person -> name':24} {'lang':>8} | {'1913 boost/p':>16} | {'1929 boost/p':>16}")
for s in surs:
    a, b = res["1913"][s], res["1929"][s]
    print(f"{s+'->'+data['1929'][s]['first']:24} {LNAME[LANG[s]]:>8} | {a['boost']:>+6.1f}  p={a['p']:.2f}{' *' if a['clears'] else '  '} | "
          f"{b['boost']:>+6.1f}  p={b['p']:.2f}{' *' if b['clears'] else ''}")
for m in ["1913", "1929"]:
    cl = [s for s in surs if res[m][s]["clears"]]
    post_cl = [s for s in cl if data["1929"][s]["birth"] + 25 > 1929]
    print(f"{m}: {len(cl)} clear the within-language 95th cutoff: {', '.join(cl)}   | post-cutoff false positives: {post_cl or 'none'}")
print("wrote figures/fig_firstname_perm.png and results/prominence/firstname_perm.html")
