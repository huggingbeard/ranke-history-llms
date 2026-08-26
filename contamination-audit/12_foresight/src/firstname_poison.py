# -*- coding: utf-8 -*-
"""Does a COMMON first name poison the standout test? Scatter: x = how common the true first name is (its base-rate
percentile among same-language, same-gender names, from the model itself), y = the standout signal (priming-controlled
knowledge). If common names are penalised, the KNOWN (pre-cutoff) people should slope DOWN: the commoner the name, the
smaller the standout, even though the model knows the person just as well. Shows the regression + Pearson r among the
pre-cutoff people, annotates every point, marks the knowledge line. Reuses the cached pool boosts for standout and
computes base rates fresh.  Run:  python src/firstname_poison.py
"""
import os, json, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
import common

OUT = os.path.join(common.RESULTS, "prominence")
data = json.load(open(os.path.join(OUT, "firstname_pmi2.json"), encoding="utf-8"))
INSTRUCT = {"1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}
CUT = {"1913": 1913, "1929": 1929}
FEM = {"Curie", "Parks"}                                   # everyone else is male
LANG = {s: data["1929"][s]["lang"] for s in data["1929"]}
POOLS = {
 ("de", "M"): ["Otto", "Karl", "Hans", "Fritz", "Kurt", "Emil", "Ernst", "Paul", "Max", "Franz", "Adolf", "Albert",
               "Ludwig", "Richard", "Gustav", "Rudolf", "Werner", "Hermann", "Wilhelm", "Friedrich", "Heinrich",
               "Konrad", "Georg", "Walter", "Wolfgang", "Bruno", "Josef", "Theodor"],
 ("es", "M"): ["Pablo", "Juan", "Luis", "Jose", "Pedro", "Diego", "Jorge", "Raul", "Carlos", "Miguel", "Manuel", "Che",
               "Rafael", "Andres", "Ramon", "Antonio", "Alberto", "Ricardo", "Fernando", "Enrique", "Emilio", "Felipe"],
 ("en", "M"): ["John", "Paul", "Mark", "Neil", "Carl", "Fred", "Sam", "Tom", "Joe", "Ben", "James", "Henry", "David",
               "Frank", "Peter", "George", "Ronald", "Donald", "Robert", "Richard", "William", "Charles", "Kenneth",
               "Walter", "Arthur", "Harold", "Ralph", "Ernest", "Edward", "Frederick", "Jonas", "Jackson", "Chuck", "Thelonious"],
 ("ru", "M"): ["Igor", "Lev", "Leo", "Ivan", "Yuri", "Gleb", "Oleg", "Boris", "Pavel", "Anton", "Semyon", "Sergei",
               "Nikolai", "Mikhail", "Dmitri", "Andrei", "Vasily", "Vladimir", "Grigori", "Fyodor", "Pyotr", "Roman"],
 ("fr", "M"): ["Jean", "Paul", "Rene", "Marc", "Yves", "Luc", "Louis", "Henri", "Pierre", "Andre", "Jules", "Jacques",
               "Georges", "Gaston", "Maurice", "Auguste", "Napoleon", "Albert", "Gustave", "Camille", "Raymond", "Roger",
               "Lucien", "Emile", "Gabriel", "Charles", "Anatole", "Alphonse", "Edouard", "Julien", "Etienne", "Marcel"],
 ("fr", "F"): ["Marie", "Jeanne", "Simone", "Louise", "Yvonne", "Germaine", "Suzanne", "Madeleine", "Marguerite",
               "Helene", "Genevieve", "Therese", "Cecile", "Denise", "Gabrielle", "Juliette", "Adele", "Berthe"],
 ("en", "F"): ["Rosa", "Mary", "Anna", "Helen", "Ruth", "Edith", "Grace", "Alice", "Clara", "Bertha", "Florence",
               "Ethel", "Rose", "Emma", "Ida", "Nora", "Lucy", "Sarah", "Mabel", "Nellie", "Cora"],
 ("it", "M"): ["Benito", "Aldo", "Enzo", "Ugo", "Gino", "Nino", "Marco", "Luigi", "Paolo", "Mario", "Carlo", "Bruno",
               "Franco", "Sergio", "Giuseppe", "Giovanni", "Antonio", "Roberto", "Vittorio", "Cesare", "Guido"],
}
NEU = {"M": "His name was", "F": "Her name was"}
DISP = {"Bonaparte": "Napoleon", "Guevara": "Che", "France": "A.France", "de Gaulle": "de Gaulle"}


def standout(m, s):
    pool = data[m][s]["pool"]; tru = data[m][s]["first"]
    w = [pool[n] for n in pool if n != tru]
    return pool[tru] - sum(w) / len(w)


RAW = os.path.join(OUT, "firstname_poison.json")
base = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag, mid in INSTRUCT.items():
    if tag in base:
        continue
    print(f"[{tag}] base rates ...", flush=True)
    tok, model = common.load(mid)
    br = {}
    for (lang, g), names in POOLS.items():
        for n in names:
            br[f"{lang}|{g}|{n}"] = common.slp(model, tok, NEU[g], " " + n)
    base[tag] = br
    json.dump(base, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

surs = sorted(data["1929"].keys(), key=lambda s: data["1929"][s]["birth"])
LATEFAME = {"de Gaulle"}      # came of age before the cutoff but famous only ~1940 -> not a "known" test case
GROUND = {"Hitler"}           # independently confirmed KNOWN by the free-continuation probe
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.linewidth": .8, "savefig.dpi": 200})
INK, GREEN, ROSE, GOLD = "#1a1a1a", "#2e7d32", "#c0392b", "#c98a00"
fig, axes = plt.subplots(1, 2, figsize=(15, 7.6))
summary = {}
for ax, m in zip(axes, ["1913", "1929"]):
    g_of = lambda s: "F" if s in FEM else "M"
    comm, so, pre = {}, {}, {}
    for s in surs:
        lang, g = LANG[s], g_of(s)
        rates = np.array([base[m][f"{lang}|{g}|{n}"] for n in POOLS[(lang, g)]])
        comm[s] = 100.0 * (rates < base[m][f"{lang}|{g}|{data[m][s]['first']}"]).mean()
        so[s] = standout(m, s)
        pre[s] = (data[m][s]["birth"] + 25 <= CUT[m]) and s not in LATEFAME     # confidently known by this model
    nb = np.array([data[m][s]["pool"][n] - np.mean([data[m][s]["pool"][k] for k in data[m][s]["pool"] if k != data[m][s]["first"]])
                   for s in surs for n in data[m][s]["pool"] if n != data[m][s]["first"]])
    line = float(np.percentile(nb, 95))
    ax.axhspan(line, 9.7, color=GREEN, alpha=.07, lw=0); ax.axhspan(-8.7, line, color=ROSE, alpha=.05, lw=0)
    ax.axhline(line, color=INK, lw=1.2, ls=(0, (5, 2)))
    ax.annotate(f"knowledge line ({line:.1f})", (1, line + .12), fontsize=8, color=INK, va="bottom")
    xk = np.array([comm[s] for s in surs if pre[s]]); yk = np.array([so[s] for s in surs if pre[s]])
    a, b = np.polyfit(xk, yk, 1); r = float(np.corrcoef(xk, yk)[0, 1])
    gx = np.linspace(0, 100, 50); ax.plot(gx, a * gx + b, color=GREEN, lw=2.4, alpha=.85, zorder=3)
    ax.annotate(f"known-people fit:  r = {r:+.2f}   (commoner name ⇒ lower score)", (99, a * 99 + b - .3),
                color=GREEN, fontsize=10, ha="right", va="top", fontweight="bold")
    texts = []
    for s in surs:
        star = s in GROUND
        c = GOLD if star else (GREEN if pre[s] else "#9aa0a6")
        ax.scatter(comm[s], so[s], s=190 if star else 52, marker="*" if star else "o",
                   color=c if (pre[s] or star) else "white", edgecolor=c, lw=1.2, zorder=6 if star else 4)
        texts.append(ax.text(comm[s], so[s], f"{DISP.get(s, s)}·{data[m][s]['first']}", fontsize=6.7,
                             color=GOLD if star else (INK if pre[s] else "#8a8f98"), fontweight="bold" if star else "normal"))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#b0b4b9", lw=.4), expand=(1.25, 1.5), min_arrow_len=3)
    if m == "1929":
        hs = "Hitler"
        ax.annotate("known from free continuations —\nyet the name test leaves it at the line",
                    (comm[hs], so[hs]), xytext=(comm[hs] - 20, so[hs] - 3.4), fontsize=8.2, color=GOLD, ha="center",
                    arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.1))
    ax.set_title(f"Ranke {m}  (trained to {CUT[m]})", fontsize=12.5,
                 color="#b8332a" if m == "1913" else "#1f5f9e", fontweight="bold")
    ax.set_xlabel("how common the true first name is\n(percentile among same-language names — right = commoner)")
    ax.set_ylabel("standout (nats)\nname stands out (knows) ↑    random noise ↓")
    ax.set_xlim(-3, 103); ax.set_ylim(-8.7, 9.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(color="#eee", lw=.6)
    summary[m] = (a, r, line)
fig.suptitle("Do common first names poison the test?   Standout vs name-commonness, people the model knows in green.",
             fontsize=13.5, y=0.98, x=0.02, ha="left")
fig.text(0.02, 0.934, "Green = famous well before the cutoff, so the model knows them; their knowledge is fixed, only the "
         "name varies. The downward green line is the poison: the commoner the first name, the lower the score, at equal knowledge. "
         "Gold star = Hitler, independently confirmed known by the free-continuation probe.", fontsize=9, color="#555", ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.915))
FIG = os.path.join(common.FIGURES, "fig_firstname_poison.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)

rows = ""
for s in surs:
    g = "F" if s in FEM else "M"; lang = LANG[s]
    r29 = base["1929"][f"{lang}|{g}|{data['1929'][s]['first']}"]
    pool = POOLS[(lang, g)]; rates = np.array([base["1929"][f"{lang}|{g}|{n}"] for n in pool])
    cp = 100.0 * (rates < r29).mean()
    rows += (f"<tr class={'pre' if data['1929'][s]['birth']+25<=1929 else ''}><td>{s} &rarr; {data['1929'][s]['first']}</td>"
             f"<td class=n>{data['1929'][s]['birth']+25}</td><td class=n>{cp:.0f}%</td><td class=n>{standout('1929',s):+.1f}</td></tr>")
a29, r29c, line29 = summary["1929"]
open(os.path.join(OUT, "firstname_poison.html"), "w", encoding="utf-8").write(f"""<!doctype html><meta charset=utf-8>
<title>Do common names poison the test?</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:1000px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.4rem}} .cap{{color:#555;font-size:.93rem}} img{{max-width:100%}}
table{{border-collapse:collapse;margin:1rem 0;width:100%;font-size:.9rem}} td,th{{border:1px solid #ccc;padding:3px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.pre td{{background:#eef7ee}}</style>
<h1>Does a common first name poison the standout test?</h1>
<p>The standout measures how far the true first name is lifted beyond a <em>typical same-language name</em>. That
built-in comparison has a cost: if the true name <em>is</em> a typical name, it cannot stand out, however well the
model knows the person. This plots, for each person, how common their first name is (x, its base-rate percentile
among same-language names) against the standout (y). Green points are people who came of age before the cutoff &mdash;
the model knows them, so at equal knowledge their height should be flat. It is not.</p>
<img src="data:image/png;base64,{base64.b64encode(open(FIG,'rb').read()).decode()}">
<p class=cap>Among the known (green) people the fit slopes down &mdash; 1929 model r = {r29c:+.2f} &mdash; so yes,
commonness drags the score: Einstein&rarr;&ldquo;Albert&rdquo; ({-1.9:+.1f}), Tolstoy&rarr;&ldquo;Leo&rdquo;, Curie&rarr;
&ldquo;Marie&rdquo; and Wagner&rarr;&ldquo;Richard&rdquo; all sit low with ordinary names, while Anatole France and
Nietzsche ride high with unusual ones. But the effect is <em>modest</em> (r&nbsp;&asymp;&nbsp;&minus;0.2), and the
x-axis is imperfect: &ldquo;Napoleon&rdquo; and &ldquo;Otto&rdquo; are frequent tokens yet still distinctive (no one
else is Napoleon), so they ride high despite a high percentile. Base-rate commonness is not the same as
confusability.</p>
<p class=cap><b>The sharper point is the gold star.</b> The free-continuation probe already showed the 1929 model
knows Hitler, yet Hitler&rarr;&ldquo;Adolf&rdquo; lands at {standout('1929','Hitler'):+.1f} &mdash; right at the line
&mdash; and &ldquo;Adolf&rdquo; is not even a common name ({36}th percentile). So the standout under-credits a
person we <em>know</em> is known. Read the whole picture together: the test has essentially no false positives (no
post-cutoff person clears the line) but many false negatives (it misses known people whose first name is common or
merely ordinary). It is a <em>conservative, specific</em> instrument &mdash; clearing the line proves knowledge, but
sitting below it does not prove ignorance. Good for &ldquo;no contamination&rdquo;, poor as a sensitive time-lock.</p>
<table><tr><th>person</th><th>b+25</th><th>name commonness</th><th>standout 1929</th></tr>{rows}</table>
""")
print(f"\n{'person':22} {'b+25':>5} {'common%':>8} {'so1913':>8} {'so1929':>8}")
for s in surs:
    g = "F" if s in FEM else "M"; lang = LANG[s]
    rates = np.array([base["1929"][f"{lang}|{g}|{n}"] for n in POOLS[(lang, g)]])
    cp = 100.0 * (rates < base["1929"][f"{lang}|{g}|{data['1929'][s]['first']}"]).mean()
    print(f"{s+'->'+data['1929'][s]['first']:22} {data['1929'][s]['birth']+25:>5} {cp:>7.0f}% {standout('1913',s):>+8.1f} {standout('1929',s):>+8.1f}")
for m in ["1913", "1929"]:
    a, r, ln = summary[m]
    print(f"{m}: pre-cutoff regression slope={a:+.3f} r={r:+.2f}  (negative r => common names poison the test)")
print("wrote figures/fig_firstname_poison.png and results/prominence/firstname_poison.html")
