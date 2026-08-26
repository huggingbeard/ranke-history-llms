# -*- coding: utf-8 -*-
"""Frequency-matched first-name probe (the fair version). Distractors are matched to the true name on
language, gender AND corpus frequency, so no name wins or loses on base rate: 'Jean' now faces equally
common French names (Sartre's false 98% collapses to chance), and a rare true name like 'Igor' faces
equally rare names (no more sub-chance). Base rate of a first name = P(name | 'His/Her name was'),
measured once on Ranke-1929-Instruct; the resulting option sets are fixed and scored on both instruct models
via the chat template. Adds Camus and Beauvoir. Rebuilds the two-panel by-age graph. Chance = 20%.
Run:  python src/firstname_freqmatch.py
"""
import os, json, math, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

OUT = os.path.join(common.RESULTS, "prominence")
RAW = os.path.join(OUT, "firstname_freqmatch.json")
INSTRUCT = {"1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}

# domain, lang, gender, role, surname, first, birth, fame
PEOPLE = [
 ("Painting", "es", "M", "Spanish painter", "Picasso", "Pablo", 1881, 1910),
 ("Statecraft", "de", "M", "German statesman", "Bismarck", "Otto", 1815, 1865),
 ("Statecraft", "de", "M", "German agitator", "Hitler", "Adolf", 1889, 1923),
 ("Statecraft", "en", "M", "American president", "Reagan", "Ronald", 1911, 1981),
 ("Physics", "de", "M", "German physicist", "Einstein", "Albert", 1879, 1919),
 ("Music", "de", "M", "German composer", "Wagner", "Richard", 1813, 1860),
 ("Music", "ru", "M", "Russian composer", "Stravinsky", "Igor", 1882, 1913),
 ("Music", "en", "M", "American singer", "Presley", "Elvis", 1935, 1956),
 ("Literature", "ru", "M", "Russian novelist", "Tolstoy", "Leo", 1828, 1875),
 ("Literature", "en", "M", "American novelist", "Hemingway", "Ernest", 1899, 1926),
 ("Literature", "en", "M", "English writer", "Orwell", "George", 1903, 1945),
 ("Revolution", "de", "M", "German socialist", "Marx", "Karl", 1818, 1870),
 ("Revolution", "ru", "M", "Russian revolutionary", "Lenin", "Vladimir", 1870, 1917),
 ("Revolution", "es", "M", "Cuban revolutionary", "Guevara", "Che", 1928, 1959),
 ("Science", "fr", "F", "French physicist", "Curie", "Marie", 1867, 1903),
 ("Science", "en", "M", "American doctor", "Salk", "Jonas", 1914, 1955),
 ("Exploration", "en", "M", "British explorer", "Stanley", "Henry", 1841, 1875),
 ("Exploration", "en", "M", "American aviator", "Lindbergh", "Charles", 1902, 1927),
 ("Exploration", "en", "M", "American astronaut", "Armstrong", "Neil", 1930, 1969),
 ("Philosophy", "de", "M", "German philosopher", "Nietzsche", "Friedrich", 1844, 1888),
 ("Philosophy", "fr", "M", "French philosopher", "Sartre", "Jean", 1905, 1945),
 ("Philosophy", "fr", "M", "French writer", "Camus", "Albert", 1913, 1942),
 ("Philosophy", "fr", "F", "French writer", "Beauvoir", "Simone", 1908, 1949),
 ("Empire", "fr", "M", "French general", "Bonaparte", "Napoleon", 1769, 1800),
 ("Empire", "it", "M", "Italian dictator", "Mussolini", "Benito", 1883, 1922),
 ("Civil rights", "en", "M", "American reformer", "Douglass", "Frederick", 1818, 1850),
 ("Civil rights", "en", "F", "American civil-rights figure", "Parks", "Rosa", 1913, 1955),
]
POOLS = {
 ("de", "M"): ["Otto", "Karl", "Hans", "Fritz", "Kurt", "Emil", "Ernst", "Paul", "Max", "Franz", "Adolf",
               "Albert", "Ludwig", "Richard", "Gustav", "Rudolf", "Werner", "Hermann", "Wilhelm", "Friedrich",
               "Heinrich", "Konrad", "Georg", "Walter", "Wolfgang", "Bruno", "Josef", "Theodor"],
 ("es", "M"): ["Pablo", "Juan", "Luis", "Jose", "Pedro", "Diego", "Jorge", "Raul", "Carlos", "Miguel", "Manuel",
               "Rafael", "Andres", "Ramon", "Antonio", "Alberto", "Ricardo", "Fernando", "Enrique", "Emilio", "Felipe"],
 ("en", "M"): ["John", "Paul", "Mark", "Neil", "Carl", "Fred", "Sam", "Tom", "Joe", "Ben", "James", "Henry",
               "David", "Frank", "Peter", "George", "Ronald", "Donald", "Robert", "Richard", "William", "Charles",
               "Kenneth", "Walter", "Arthur", "Harold", "Ralph", "Elvis", "Jonas", "Jackson", "Ernest", "Neil", "Edward"],
 ("ru", "M"): ["Igor", "Lev", "Ivan", "Yuri", "Gleb", "Oleg", "Boris", "Pavel", "Anton", "Semyon", "Sergei",
               "Nikolai", "Mikhail", "Dmitri", "Andrei", "Vasily", "Vladimir", "Grigori", "Fyodor", "Pyotr", "Roman"],
 ("fr", "M"): ["Jean", "Paul", "Rene", "Marc", "Yves", "Luc", "Louis", "Henri", "Pierre", "Andre", "Jules",
               "Jacques", "Georges", "Gaston", "Maurice", "Auguste", "Napoleon", "Albert", "Gustave", "Camille",
               "Raymond", "Roger", "Lucien", "Emile", "Gabriel", "Alphonse", "Edouard", "Charles", "Julien", "Etienne", "Marcel"],
 ("fr", "F"): ["Marie", "Jeanne", "Simone", "Louise", "Yvonne", "Germaine", "Suzanne", "Madeleine", "Marguerite",
               "Helene", "Genevieve", "Therese", "Cecile", "Denise", "Gabrielle", "Juliette", "Adele", "Berthe", "Camille"],
 ("en", "F"): ["Rosa", "Mary", "Anna", "Helen", "Ruth", "Edith", "Grace", "Alice", "Clara", "Bertha", "Florence",
               "Ethel", "Rose", "Emma", "Ida", "Nora", "Lucy", "Sarah", "Mabel", "Nellie", "Cora"],
 ("it", "M"): ["Benito", "Aldo", "Enzo", "Ugo", "Gino", "Nino", "Marco", "Luigi", "Paolo", "Mario", "Carlo",
               "Bruno", "Franco", "Sergio", "Giuseppe", "Giovanni", "Antonio", "Roberto", "Vittorio", "Cesare", "Guido"],
}


def softmax(xs):
    m = max(xs); ex = [math.exp(x - m) for x in xs]; return [e / sum(ex) for e in ex]


def chat_q(tok, role, sur):
    msgs = [{"role": "user", "content": f"What was the first name of the {role} {sur}? Answer with only the first name."}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    except Exception:
        return f"Question: what was the first name of the {role} {sur}?\nAnswer:"


data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
if "options" not in data:
    # pass A on 1929-instruct: base rates -> frequency-matched option sets (fixed for both models)
    print("[options] loading 1929-instruct for base rates ...", flush=True)
    tok, model = common.load(INSTRUCT["1929"])
    base_neutral = {"M": "His name was", "F": "Her name was"}
    opts = {}
    for dom, lang, g, role, sur, first, b, fy in PEOPLE:
        pool = [x for x in POOLS[(lang, g)] if x.lower() != first.lower()]
        rate = {x: common.slp(model, tok, base_neutral[g], " " + x) for x in pool + [first]}
        distr = sorted(pool, key=lambda x: abs(rate[x] - rate[first]))[:4]
        opts[sur] = [first] + distr
    data["options"] = opts
    # also score 1929 now while loaded
    row = {}
    for dom, lang, g, role, sur, first, b, fy in PEOPLE:
        p = softmax([common.slp(model, tok, chat_q(tok, role, sur), " " + o) for o in opts[sur]])
        row[sur] = {"first": first, "birth": b, "fame": fy, "opts": opts[sur], "p_true": round(p[0], 3)}
    data["1929"] = row
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

if "1913" not in data:
    print("[1913] loading 1913-instruct ...", flush=True)
    tok, model = common.load(INSTRUCT["1913"])
    row = {}
    for dom, lang, g, role, sur, first, b, fy in PEOPLE:
        p = softmax([common.slp(model, tok, chat_q(tok, role, sur), " " + o) for o in data["options"][sur]])
        row[sur] = {"first": first, "birth": b, "fame": fy, "opts": data["options"][sur], "p_true": round(p[0], 3)}
    data["1913"] = row
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

# ---------- two-panel by-age graph ----------
surs = sorted([p[4] for p in PEOPLE], key=lambda s: data["1929"][s]["birth"])
X = np.array([data["1929"][s]["birth"] + 25 for s in surs])
FN = {s: data["1929"][s]["first"] for s in surs}
DISP = {"Bonaparte": "Napoleon", "Presley": "Elvis", "Guevara": "Che", "Beauvoir": "Beauvoir", "Camus": "Camus"}
LABELS = ["Bonaparte", "Bismarck", "Marx", "Nietzsche", "Tolstoy", "Curie", "Stravinsky", "Lenin", "Mussolini",
          "Hitler", "Einstein", "Lindbergh", "Sartre", "Camus", "Beauvoir", "Reagan", "Parks", "Presley", "Armstrong"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.linewidth": .8, "savefig.dpi": 200})
INK, GREEN, GREY = "#1a1a1a", "#2e7d32", "#8a8f98"
fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.6), sharex=True)


def smooth(x, y, bw=13.0):
    grid = np.linspace(x.min(), x.max(), 240)
    return grid, np.array([np.sum(np.exp(-((x - g) / bw) ** 2) * y) / np.sum(np.exp(-((x - g) / bw) ** 2)) for g in grid])


for ax, (m, cut, col, name) in zip(axes, [("1913", 1913, "#b8332a", "Ranke 1913"), ("1929", 1929, "#1f5f9e", "Ranke 1929")]):
    y = np.array([100 * data[m][s]["p_true"] for s in surs])
    ax.axvspan(1785, cut, color=GREEN, alpha=.06, lw=0); ax.axvspan(cut, 1990, color=GREY, alpha=.07, lw=0)
    ax.axhspan(0, 20, color=GREY, alpha=.10, lw=0); ax.axhline(20, color=GREY, ls=(0, (4, 3)), lw=1)
    ax.axvline(cut, color=col, ls="--", lw=1.6)
    left = X < cut
    for mask, lo, hi in [(left, X.min(), cut), (~left, cut, X.max())]:
        xm, ym = X[mask], y[mask]
        if len(xm) >= 2:
            grid = np.linspace(lo, hi, 140)
            gy = np.array([np.sum(np.exp(-((xm - g) / 11.0) ** 2) * ym) / np.sum(np.exp(-((xm - g) / 11.0) ** 2)) for g in grid])
            ax.plot(grid, gy, color=col, lw=2.4, alpha=.85, zorder=2, solid_capstyle="round")
    ab = y >= 20
    ax.scatter(X[ab], y[ab], s=55, color=col, edgecolor="white", lw=.8, zorder=4)
    ax.scatter(X[~ab], y[~ab], s=42, facecolor="white", edgecolor=col, lw=1.1, zorder=4)
    ax.annotate(f"{name} trained to {cut}", (cut - 3, 96), color=col, fontsize=11, ha="right", fontweight="bold")
    ax.annotate("within its world", ((1785 + cut) / 2, 4), color=GREEN, fontsize=9.5, ha="center", alpha=.8)
    ax.annotate("after its cutoff →", (cut + 4, 4), color=GREY, fontsize=9.5, ha="left")
    ax.annotate("chance 20%", (1986, 22), color=GREY, fontsize=9, ha="right", va="bottom")
    for k, s in enumerate(surs):
        if s in LABELS:
            dy = 4.5 if (data[m][s]["p_true"] * 100) < 88 else -9.5
            ax.annotate(DISP.get(s, s), (X[k], 100 * data[m][s]["p_true"] + dy), fontsize=8.2, ha="center", color=INK)
    ax.set_ylim(-4, 104); ax.set_xlim(1785, 1990); ax.set_ylabel("P(correct first name)  %")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#e6e6e6", lw=.7)
axes[1].set_xlabel("year the person turned 25   (birth year + 25 — when they came of age)")
fig.suptitle("Does the frozen reader know a person's first name — by when they came of age?  (fair options)",
             fontsize=13.5, y=0.965, x=0.02, ha="left")
fig.text(0.02, 0.925, "Distractors matched to the true name on language, gender AND corpus frequency, so no name "
         "wins or loses on base rate. Chance = 20%.", fontsize=10, color="#555", ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.92))
FIG = os.path.join(common.FIGURES, "fig_firstname_fair.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)
open(os.path.join(OUT, "firstname_fair.html"), "w", encoding="utf-8").write(
    f"<!doctype html><meta charset=utf-8><title>First name (fair)</title><body style='max-width:1040px;margin:2rem auto;font-family:Georgia'>"
    f"<h1>First-name knowledge, frequency-matched options</h1>"
    f"<img style='max-width:100%' src='data:image/png;base64,{base64.b64encode(open(FIG,'rb').read()).decode()}'>")

print(f"\n{'surname -> first':24} {'b+25':>5} | {'1913':>6} {'1929':>6}  options")
for s in surs:
    print(f"{s+' -> '+FN[s]:24} {data['1929'][s]['birth']+25:>5} | {100*data['1913'][s]['p_true']:>5.0f}% "
          f"{100*data['1929'][s]['p_true']:>5.0f}%  {', '.join(data['1929'][s]['opts'])}")
print("\nSartre check (should now be ~chance):", f"1913={100*data['1913']['Sartre']['p_true']:.0f}% 1929={100*data['1929']['Sartre']['p_true']:.0f}%")
print("wrote figures/fig_firstname_fair.png and results/prominence/firstname_fair.html")
