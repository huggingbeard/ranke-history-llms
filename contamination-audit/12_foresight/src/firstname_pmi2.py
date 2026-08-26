# -*- coding: utf-8 -*-
"""Frequency-INVARIANT first-name knowledge: does naming the person boost the true first name above its own
baseline? boost = log P(first | "The <role> <surname>'s first name was") - log P(first | "His/Her name was").
>0 nats = the role+surname context lifts the true first name = knowledge; ~0 = no knowledge, regardless of how
common the name is (so Sartre->Jean and Reagan->Ronald, which only won on frequency, fall to ~0; Bismarck->Otto
stays high). Ranke 1913 & 1929 instruct. Segmented trend graph (within window / after cutoff).
Run:  python src/firstname_pmi2.py
"""
import os, json, math, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
import common

OUT = os.path.join(common.RESULTS, "prominence")
RAW = os.path.join(OUT, "firstname_pmi2.json")
INSTRUCT = {"1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}
# domain, lang, gender, role, surname, first, birth
PEOPLE = [
 ("Empire", "fr", "M", "French general", "Bonaparte", "Napoleon", 1769), ("Music", "de", "M", "German composer", "Wagner", "Richard", 1813),
 ("Statecraft", "de", "M", "German statesman", "Bismarck", "Otto", 1815), ("Revolution", "de", "M", "German socialist", "Marx", "Karl", 1818),
 ("Civil rights", "en", "M", "American reformer", "Douglass", "Frederick", 1818), ("Literature", "ru", "M", "Russian novelist", "Tolstoy", "Leo", 1828),
 ("Exploration", "en", "M", "British explorer", "Stanley", "Henry", 1841), ("Philosophy", "de", "M", "German philosopher", "Nietzsche", "Friedrich", 1844),
 ("Science", "fr", "F", "French physicist", "Curie", "Marie", 1867), ("Revolution", "ru", "M", "Russian revolutionary", "Lenin", "Vladimir", 1870),
 ("Physics", "de", "M", "German physicist", "Einstein", "Albert", 1879), ("Painting", "es", "M", "Spanish painter", "Picasso", "Pablo", 1881),
 ("Music", "ru", "M", "Russian composer", "Stravinsky", "Igor", 1882), ("Empire", "it", "M", "Italian dictator", "Mussolini", "Benito", 1883),
 ("Statecraft", "de", "M", "German agitator", "Hitler", "Adolf", 1889), ("Literature", "en", "M", "American novelist", "Hemingway", "Ernest", 1899),
 ("Exploration", "en", "M", "American aviator", "Lindbergh", "Charles", 1902), ("Literature", "en", "M", "English writer", "Orwell", "George", 1903),
 ("Literature", "fr", "M", "French novelist", "France", "Anatole", 1844), ("Statecraft", "fr", "M", "French general", "de Gaulle", "Charles", 1890),
 ("Statecraft", "en", "M", "American president", "Reagan", "Ronald", 1911), ("Painting", "en", "M", "American painter", "Pollock", "Jackson", 1912),
 ("Civil rights", "en", "F", "American civil-rights figure", "Parks", "Rosa", 1913), ("Philosophy", "fr", "M", "French writer", "Camus", "Albert", 1913),
 ("Science", "en", "M", "American doctor", "Salk", "Jonas", 1914), ("Physics", "en", "M", "American physicist", "Feynman", "Richard", 1918),
 ("Revolution", "es", "M", "Cuban revolutionary", "Guevara", "Che", 1928), ("Exploration", "en", "M", "American astronaut", "Armstrong", "Neil", 1930),
 ("Music", "en", "M", "American jazz pianist", "Monk", "Thelonious", 1917), ("Music", "en", "M", "American rock and roll singer", "Berry", "Chuck", 1926),
]
# first-name pools by (language, gender): the plausible wrong names each real name is judged against
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

data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag, mid in INSTRUCT.items():
    if tag in data:
        continue
    print(f"[{tag}] loading {mid} ...", flush=True)
    tok, model = common.load(mid)
    neu = {"M": "His name was", "F": "Her name was"}
    row = {}
    for dom, lang, g, role, sur, first, b in PEOPLE:
        ctx = f"The {role} {sur}'s first name was"
        base = {}  # base rate of each candidate name, once per (lang,gender) reused via the model call
        names = list(dict.fromkeys(POOLS[(lang, g)] + [first]))     # every plausible name for this person, true included
        # boost of EVERY candidate name in this person's frame: true name + all wrong (placebo) names
        pool = {n: round(common.slp(model, tok, ctx, " " + n) - common.slp(model, tok, neu[g], " " + n), 3) for n in names}
        boost_role = round(common.slp(model, tok, ctx, " " + first) - common.slp(model, tok, f"The {role}'s first name was", " " + first), 3)
        row[sur] = {"first": first, "lang": lang, "birth": b, "boost": pool[first], "boost_role": boost_role, "pool": pool}
    data[tag] = row
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

surs = sorted([p[4] for p in PEOPLE], key=lambda s: data["1929"][s]["birth"])
X = np.array([data["1929"][s]["birth"] + 25 for s in surs])
FN = {s: data["1929"][s]["first"] for s in surs}
DISP = {"Bonaparte": "Napoleon", "Guevara": "Che", "France": "A. France", "de Gaulle": "de Gaulle"}
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.linewidth": .8, "savefig.dpi": 200})
INK, GREEN, ROSE = "#1a1a1a", "#2e7d32", "#c0392b"
LO, HI = -8.2, 9.2           # y-limits in nats (the pct axis is log, i.e. also linear in nats)


def frame_center(m, s):
    """Average boost of the wrong (same-language, same-gender) names in this person's frame = the language/role priming."""
    pool = data[m][s]["pool"]; tru = data[m][s]["first"]
    w = [pool[n] for n in pool if n != tru]
    return sum(w) / len(w)


# STANDOUT = how far the true name is lifted beyond the typical same-language name in the same frame (priming removed)
REC = {m: {s: data[m][s]["boost"] - frame_center(m, s) for s in surs} for m in ("1913", "1929")}
# the null: the same standout computed for every WRONG name -> how much a random name deviates within its frame
NULLDEV = {m: np.array([data[m][s]["pool"][n] - frame_center(m, s)
                        for s in surs for n in data[m][s]["pool"] if n != data[m][s]["first"]]) for m in ("1913", "1929")}


def placebo_null(m):
    return NULLDEV[m]


def make_fig(mode):
    pct = (mode == "pct")
    T = (lambda b: math.exp(b)) if pct else (lambda b: b)
    fig = plt.figure(figsize=(12.9, 11.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.11], wspace=.015, hspace=.14, left=.075, right=.985, top=.895, bottom=.065)
    for i, (m, cut, col, name) in enumerate([("1913", 1913, "#b8332a", "Ranke 1913"), ("1929", 1929, "#1f5f9e", "Ranke 1929")]):
        ax = fig.add_subplot(gs[i, 0]); dax = fig.add_subplot(gs[i, 1], sharey=ax)
        if pct:
            ax.set_yscale("log"); dax.set_yscale("log")
            ticks = [0.01, 0.1, 0.5, 1, 2, 10, 100, 1000]
            ax.set_yticks(ticks); ax.set_yticklabels(["0%" if abs(t - 1) < 1e-9 else f"{(t - 1) * 100:+,.0f}%" for t in ticks])
        ax.set_ylim(T(LO), T(HI))
        yb = np.array([REC[m][s] for s in surs])
        nb = placebo_null(m)
        thr95, thr99 = float(np.percentile(nb, 95)), float(np.percentile(nb, 99))
        frac = (thr95 - LO) / (HI - LO)
        # zones + lines
        ax.axhspan(T(thr95), T(HI), color=GREEN, alpha=.12, lw=0)
        ax.axhspan(T(LO), T(thr95), color=ROSE, alpha=.09, lw=0)
        ax.axhline(T(thr95), color=INK, lw=1.5, ls=(0, (5, 2)), zorder=3)
        ax.axhline(T(thr99), color=INK, lw=.8, ls=(0, (1, 2)), alpha=.55, zorder=3)
        ax.axvline(cut, color=col, ls="--", lw=1.6)
        left = X < cut
        for mask, a, b in [(left, X.min(), cut), (~left, cut, X.max())]:
            xm, ym = X[mask], yb[mask]
            if len(xm) >= 2:
                grid = np.linspace(a, b, 140)
                gy = np.array([np.sum(np.exp(-((xm - g) / 11.) ** 2) * ym) / np.sum(np.exp(-((xm - g) / 11.) ** 2)) for g in grid])
                ax.plot(grid, [T(v) for v in gy], color=col, lw=2.2, alpha=.8, zorder=2, solid_capstyle="round")
        yv = np.array([T(v) for v in yb]); ab = yb >= thr95
        ax.scatter(X[ab], yv[ab], s=46, color=col, edgecolor="white", lw=.8, zorder=4)
        ax.scatter(X[~ab], yv[~ab], s=36, facecolor="white", edgecolor=col, lw=1.1, zorder=4)
        texts = [ax.text(X[k], yv[k], DISP.get(s, s), fontsize=7.4, color=INK, ha="center", va="center", zorder=6) for k, s in enumerate(surs)]
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="#9aa0a6", lw=.5, shrinkA=3, shrinkB=3),
                    expand=(1.3, 1.7), force_text=(.5, .9), force_static=(.3, .5), min_arrow_len=4)
        ax.annotate(f"{name} · trained to {cut}", (.985, .975), xycoords="axes fraction", color=col,
                    fontsize=12.5, ha="right", va="top", fontweight="bold")
        ax.text(.012, frac + .02, "KNOWS THE NAME", transform=ax.transAxes, color=GREEN, fontsize=8.5,
                ha="left", va="bottom", fontweight="bold", alpha=.85)
        ax.text(.012, frac - .02, "inside the spread of random name-combos  ·  no knowledge", transform=ax.transAxes,
                color=ROSE, fontsize=8.3, ha="left", va="top", alpha=.85)
        lab = f"{(math.exp(thr95) - 1) * 100:,.0f}%" if pct else f"{thr95:.1f} nats"
        ax.annotate(f"knowledge line = 95th percentile of random combos  ({lab})", (.60, frac), xycoords="axes fraction",
                    color=INK, fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=.25", fc="white", ec="#ccc", lw=.5, alpha=.92))
        ax.set_xlim(1785, 1993)
        ax.set_ylabel("true name's odds vs a typical\nsame-language name  (%, log scale)" if pct
                      else "name standout (nats)\nhow far the true name is lifted beyond\na typical same-language name")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#ececec", lw=.6)
        if i == 0:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("year the person turned 25   (birth year + 25 — when they came of age)")
        # right strip: the placebo null distribution, drawn on the SAME y-axis
        gn = np.linspace(LO, HI, 220)
        dens = np.array([np.sum(np.exp(-((nb - g) / .45) ** 2 / 2)) for g in gn]); dens /= dens.max()
        yy = [T(v) for v in gn]
        dax.fill_betweenx(yy, 0, dens, color=col, alpha=.20, lw=0)
        dax.plot(dens, yy, color=col, lw=.8, alpha=.65)
        dax.axhline(T(thr95), color=INK, lw=1.2, ls=(0, (5, 2))); dax.axhline(T(thr99), color=INK, lw=.7, ls=(0, (1, 2)), alpha=.55)
        dax.text(1.12, T(thr95), "95th", fontsize=6.4, va="center", color=INK)
        dax.text(1.12, T(thr99), "99th", fontsize=6.4, va="center", color="#666")
        dax.set_xlim(0, 1.1); dax.set_xticks([]); dax.tick_params(labelleft=False)
        for sp in ("top", "right", "bottom"):
            dax.spines[sp].set_visible(False)
        dax.set_title("random\ncombos", fontsize=7.3, color="#666", pad=4)
    sub = ("How far the true first name is lifted beyond a typical same-language name in the same frame (language priming "
           "cancels out), as % change on a log scale. Knowledge line = 95th percentile of that same statistic for random name-combos — no time assumed." if pct else
           "Priming-controlled standout = the frame's lift of the true name minus its average lift of the plausible wrong "
           "names (same language, same gender), so role/language priming cancels. Knowledge line = 95th percentile of that statistic for random name-combos — no time assumed.")
    fig.suptitle("Does the true first name stand out from other same-language names?   Ranke instruct models.",
                 fontsize=13.3, y=0.965, x=0.02, ha="left")
    fig.text(0.02, 0.925, sub, fontsize=9.3, color="#555", ha="left")
    p = os.path.join(common.FIGURES, "fig_firstname_pct.png" if pct else "fig_firstname_boost.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


FIG = make_fig("nats")
FIGP = make_fig("pct")
n13, n29 = placebo_null("1913"), placebo_null("1929")
thr13, thr29 = float(np.percentile(n13, 95)), float(np.percentile(n29, 95))
thr13_99, thr29_99 = float(np.percentile(n13, 99)), float(np.percentile(n29, 99))
pctthr = lambda t: (math.exp(t) - 1) * 100
rows = ""
for s in surs:
    post = data["1929"][s]["birth"] + 25 > 1929
    rows += (f"<tr class={'post' if post else ''}><td>{s} &rarr; {FN[s]}</td><td class=n>{data['1929'][s]['birth']+25}</td>"
             f"<td class=n>{REC['1913'][s]:+.1f}</td><td class=n>{REC['1929'][s]:+.1f}</td>"
             f"<td class=n style='color:#999'>{data['1913'][s]['boost']:+.1f}</td><td class=n style='color:#999'>{data['1929'][s]['boost']:+.1f}</td></tr>")
open(os.path.join(OUT, "firstname_boost.html"), "w", encoding="utf-8").write(f"""<!doctype html><meta charset=utf-8>
<title>First-name knowledge (frequency-invariant)</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:1040px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.4rem}} .cap{{color:#555;font-size:.93rem}} img{{max-width:100%}}
table{{border-collapse:collapse;margin:1rem 0;width:100%;font-size:.9rem}} td,th{{border:1px solid #ccc;padding:3px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.post td{{background:#fbeeee}}</style>
<h1>Does the frozen reader know a person&rsquo;s first name?</h1>
<p>We never ask the model to pick from a list &mdash; a forced choice is decided by which name is commoner, not by
knowledge (that is why Reagan&rarr;Ronald and Sartre&rarr;Jean came out falsely high before). We ask how much naming
the person lifts the true first name, and then strip out the part of that lift that is language, not knowledge.</p>
<p><b>Step one</b>, the raw lift:
boost = log P(first | &ldquo;the {{role}} {{surname}}&rsquo;s first name was&rdquo;) &minus; log P(first | &ldquo;his/her name was&rdquo;).
<b>Step two</b>: the frame &ldquo;the German statesman ___&rsquo;s first name was&rdquo; lifts <em>every</em> German
name, so we subtract the average lift of the plausible wrong names in the same frame (same language, same gender).
What is left &mdash; the <em>standout</em> &mdash; is how far the true name is pulled up beyond a typical same-language
name. Only a name the model specifically ties to that person survives it.</p>
<h3 style="margin:1.4rem 0 .3rem">How the knowledge line is drawn (no circularity)</h3>
<p>An earlier version set the line from &ldquo;people born too late to be known.&rdquo; That is circular &mdash; it
assumes the very time-lock the test is meant to find, and it would hide contamination (a secretly-memorised 1950s
figure would just sink into the null). The line is now built with no reference to time. The wrong names in every
frame &mdash; {len(n29)} of them across the set &mdash; are how much the standout moves when there is no knowledge.
Recentred, they pile up at zero (the grey strip on the right of each panel), with spread about {n29.std():.1f} nats.
The knowledge line is the 95th percentile of that pile: {thr29:.1f} nats for the 1929 model, {thr13:.1f} for the
1913 model (dotted = the 99th). A true name counts as knowledge only if it stands out more than an arbitrary
same-language name does 19 times in 20. The time-lock then falls out as a <em>result</em>.</p>
<p class=cap>What the recentring exposes: most of the raw boost was language priming, not knowledge &mdash; in a German
frame every German name is lifted several nats, and the honest, person-specific signal is only what the true name
adds on top. It survives clearly for distinctively-named figures (Bismarck&rarr;Otto) and is weak where the true
name is itself common (Einstein&rarr;&ldquo;Albert&rdquo; does not stand out among German names, though the model has
surely met Einstein &mdash; a real limit of any first-name test, not a failure of the model). It also disposes of the
old role-leakage: &ldquo;the French philosopher ___ &rarr; Jean&rdquo; lifts Jean for many French surnames, so Jean
sits inside the null and the bar rises above it. Nothing is hand-flagged.</p>
<h2 style="font-size:1.05rem">Percent-change view (what you asked for)</h2>
<img src="data:image/png;base64,{base64.b64encode(open(FIGP,'rb').read()).decode()}">
<p class=cap>Each dot: how far the true first name stands out beyond a typical same-language name, as % on a log
scale. Green = above the knowledge line; rose = inside the random-combo spread. The Bismarck frame makes
&ldquo;Otto&rdquo; about {pctthr(REC['1929']['Bismarck']):+,.0f}% likelier than a typical German name (deep in
green); &ldquo;Ronald&rdquo; for Reagan is {pctthr(REC['1929']['Reagan']):+,.0f}% (in the rose spread). This
settles the Elvis worry: a few-hundred-percent point still sits below the ~{pctthr(thr29):,.0f}% line, so it is
noise, not knowledge.</p>
<h2 style="font-size:1.05rem">Same data in nats (log-odds)</h2>
<img src="data:image/png;base64,{base64.b64encode(open(FIG,'rb').read()).decode()}">
<p class=cap>Same picture in nats. Green = above the knowledge line (the name stands out); rose = inside the spread of
random name-combos. Solid dots clear the line; hollow do not. The dashed vertical is the model&rsquo;s cutoff; the
trend is in two segments so nothing is smoothed across it. Read-off: the names that clearly stand out are the
distinctively-named early figures &mdash; Bismarck, Napoleon, Marx, Nietzsche, Anatole France, Stanley &mdash; all
well inside both windows. Everyone past the cutoff sits in the random-combo spread (Reagan, Parks, Pollock, Guevara,
de&nbsp;Gaulle, Camus, Monk, Berry). The time-lock shows as a <em>rise</em> across the two models for the in-between
figures: Mussolini {REC['1913']['Mussolini']:+.1f}&rarr;{REC['1929']['Mussolini']:+.1f} nats, Stravinsky
{REC['1913']['Stravinsky']:+.1f}&rarr;{REC['1929']['Stravinsky']:+.1f} &mdash; higher in the 1929 model that trained
through their rise, though the priming-controlled bar is strict enough that they approach it rather than vault over.</p>
<p class=cap>The French replacements, in standout terms. Anatole France (before both cutoffs; &ldquo;Anatole&rdquo; is
nobody&rsquo;s guess) stands out {REC['1929']['France']:+.1f} nats &mdash; clear knowledge by a non-default name.
Charles de Gaulle (famous only from 1940) sits at {REC['1929']['de Gaulle']:+.1f} and Camus at
{REC['1929']['Camus']:+.1f} &mdash; inside the random-combo spread, correctly not known. &ldquo;Charles&rdquo; is
common enough to have leaked and did not. The reverse worry &mdash; a rare token inflating itself &mdash; also
fails: Thelonious Monk carries the strangest first name in the set, yet it stands out {REC['1929']['Monk']:+.1f}
(below zero), and Chuck Berry {REC['1929']['Berry']:+.1f}. Neither a common name nor a rare one fools the standout;
only a name the model ties to the specific person moves it.</p>
<h2 style="font-size:1.05rem">All people: standout (nats), and the two raw boosts behind it</h2>
<p class=cap>&ldquo;standout&rdquo; = raw boost minus the frame&rsquo;s average wrong-name boost (the number plotted).
Pink = came of age after 1929. Knowledge line: {thr13:.1f} (1913), {thr29:.1f} (1929) nats.</p>
<table><tr><th>person</th><th>b+25</th><th>standout<br>1913</th><th>standout<br>1929</th><th>raw boost<br>1913</th><th>raw boost<br>1929</th></tr>{rows}</table>
""")
print(f"\nknowledge line (95th pct of random within-frame combos): 1913 {thr13:.2f} nats, 1929 {thr29:.2f} nats")
print(f"{'surname -> first':24} {'b+25':>5} |  standout (nats)   |  clears line?")
print(f"{'':24} {'':5} |   1913    1929    |")
for s in surs:
    k13 = "  KNOWS" if REC["1913"][s] >= thr13 else ""
    k29 = " KNOWS" if REC["1929"][s] >= thr29 else ""
    print(f"{s+' -> '+FN[s]:24} {data['1929'][s]['birth']+25:>5} | {REC['1913'][s]:>6.1f} {REC['1929'][s]:>7.1f}    | {k13:>7}{k29}")
print("wrote figures/fig_firstname_boost.png, fig_firstname_pct.png and results/prominence/firstname_boost.html")
