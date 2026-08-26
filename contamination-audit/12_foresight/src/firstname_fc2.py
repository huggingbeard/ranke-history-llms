# -*- coding: utf-8 -*-
"""First-name multiple choice, done fairly: distractors matched on LANGUAGE and TOKEN LENGTH.
The first pass let 'Otto' (German, 2 tokens) lose to 'John' (English, 1 token, ultra-frequent) purely on
summed log-prob -- so Bismarck looked unknown to the 1900 model. Fix: for each person the four distractors
are same-culture first names of the SAME token length as the true name (Picasso: Pablo/Juan/Diego/Jose;
Bismarck: Otto/Karl/Hans/Emil). Ranke 1900/1913/1929. Writes firstname_fc2.json + figure + HTML.
Run:  python src/firstname_fc2.py
"""
import os, json, math, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "prominence")
RAW = os.path.join(OUT, "firstname_fc2.json")
MODELS = ["1900", "1913", "1929"]

# (domain, lang, role, surname, first, fame)
PEOPLE = [
 ("Painting", "fr", "French painter", "Gauguin", "Paul", 1895),
 ("Painting", "es", "Spanish painter", "Picasso", "Pablo", 1910),
 ("Painting", "en", "American painter", "Pollock", "Jackson", 1948),
 ("Statecraft", "de", "German statesman", "Bismarck", "Otto", 1865),
 ("Statecraft", "de", "German agitator", "Hitler", "Adolf", 1923),
 ("Statecraft", "en", "American president", "Reagan", "Ronald", 1981),
 ("Physics", "en", "British physicist", "Maxwell", "James", 1870),
 ("Physics", "de", "German physicist", "Einstein", "Albert", 1919),
 ("Physics", "en", "American physicist", "Feynman", "Richard", 1965),
 ("Music", "de", "German composer", "Wagner", "Richard", 1860),
 ("Music", "ru", "Russian composer", "Stravinsky", "Igor", 1913),
 ("Music", "en", "American singer", "Presley", "Elvis", 1956),
 ("Literature", "ru", "Russian novelist", "Tolstoy", "Leo", 1875),
 ("Literature", "en", "American novelist", "Hemingway", "Ernest", 1926),
 ("Literature", "en", "English writer", "Orwell", "George", 1945),
 ("Revolution", "de", "German socialist", "Marx", "Karl", 1870),
 ("Revolution", "ru", "Russian revolutionary", "Lenin", "Vladimir", 1917),
 ("Revolution", "es", "Cuban revolutionary", "Guevara", "Che", 1959),
 ("Science", "fr", "French chemist", "Pasteur", "Louis", 1880),
 ("Science", "fr", "French physicist", "Curie", "Marie", 1903),
 ("Science", "en", "American doctor", "Salk", "Jonas", 1955),
 ("Exploration", "en", "British explorer", "Stanley", "Henry", 1875),
 ("Exploration", "en", "American aviator", "Lindbergh", "Charles", 1927),
 ("Exploration", "en", "American astronaut", "Armstrong", "Neil", 1969),
 ("Philosophy", "de", "German philosopher", "Nietzsche", "Friedrich", 1888),
 ("Philosophy", "de", "Austrian philosopher", "Wittgenstein", "Ludwig", 1922),
 ("Philosophy", "fr", "French philosopher", "Sartre", "Jean", 1945),
 ("Empire", "fr", "French general", "Bonaparte", "Napoleon", 1800),
 ("Empire", "it", "Italian dictator", "Mussolini", "Benito", 1922),
 ("Civil rights", "en", "American reformer", "Douglass", "Frederick", 1850),
 ("Civil rights", "en", "American civil-rights figure", "Parks", "Rosa", 1955),
]
POOLS = {
 "de": ["Otto", "Karl", "Hans", "Fritz", "Kurt", "Emil", "Ernst", "Paul", "Max", "Franz", "Adolf", "Albert",
        "Ludwig", "Richard", "Gustav", "Rudolf", "Werner", "Hermann", "Wilhelm", "Friedrich", "Heinrich",
        "Konrad", "Dietrich", "Wolfgang", "Reinhard", "Gottfried"],
 "es": ["Pablo", "Juan", "Luis", "Jose", "Pedro", "Diego", "Jorge", "Raul", "Carlos", "Miguel", "Manuel",
        "Rafael", "Andres", "Ramon", "Antonio", "Alberto", "Ricardo", "Francisco", "Fernando", "Alejandro", "Salvador"],
 "en": ["John", "Paul", "Mark", "Neil", "Carl", "Fred", "Sam", "Tom", "Joe", "Ben", "James", "Henry", "David",
        "Frank", "Peter", "George", "Ronald", "Donald", "Robert", "Richard", "William", "Charles", "Kenneth",
        "Franklin", "Theodore", "Jonas", "Elvis", "Jackson"],
 "ru": ["Igor", "Lev", "Ivan", "Yuri", "Gleb", "Oleg", "Boris", "Pavel", "Anton", "Semyon", "Sergei", "Nikolai",
        "Mikhail", "Dmitri", "Andrei", "Vasily", "Vladimir", "Alexander", "Grigori", "Konstantin"],
 "fr": ["Jean", "Paul", "Rene", "Marc", "Yves", "Luc", "Louis", "Henri", "Pierre", "Andre", "Jules", "Julien",
        "Jacques", "Georges", "Gaston", "Maurice", "Napoleon", "Auguste", "Edouard", "Alphonse", "Marie"],
 "it": ["Aldo", "Enzo", "Ugo", "Gino", "Nino", "Marco", "Luigi", "Paolo", "Mario", "Carlo", "Bruno", "Benito",
        "Franco", "Sergio", "Giuseppe", "Giovanni", "Antonio", "Roberto", "Francesco", "Vittorio"],
}


def softmax(xs):
    m = max(xs); ex = [math.exp(x - m) for x in xs]; return [e / sum(ex) for e in ex]


data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag in MODELS:
    if tag in data:
        continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    tl = lambda s: len(tok(" " + s, add_special_tokens=False).input_ids)
    row = {}
    for dom, lang, role, sur, first, fy in PEOPLE:
        n = tl(first)
        cand = [x for x in POOLS[lang] if x.lower() != first.lower()]
        same = [x for x in cand if tl(x) == n]
        distr = (same if len(same) >= 4 else sorted(cand, key=lambda x: abs(tl(x) - n)))[:4]
        opts = [first] + distr
        p = softmax([common.slp(model, tok, f"The {role} {sur}'s first name was", " " + o) for o in opts])
        row[sur] = {"first": first, "fame": fy, "domain": dom, "lang": lang, "opts": opts,
                    "tok_len": n, "p_true": round(p[0], 3),
                    "pick": opts[max(range(len(opts)), key=lambda k: p[k])]}
    data[tag] = row
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

order = sorted(PEOPLE, key=lambda p: p[5])
M = np.array([[100 * data[t][p[3]]["p_true"] for t in MODELS] for p in order])
fig, ax = plt.subplots(figsize=(7.0, 9.2))
ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels([f"Ranke {m}" for m in MODELS], fontsize=10)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{p[2]} {p[3]} → {p[4]}?  ({p[5]})" for p in order], fontsize=8.1)
for i in range(len(order)):
    for j in range(3):
        v = M[i, j]
        ax.annotate(f"{v:.0f}", (j, i), ha="center", va="center", fontsize=7.6, color="white" if v > 55 else "#333")
ax.set_title("First name of the (role+surname): distractors matched on language + token length.\n"
             "P(correct), % (chance = 20). Rows by rise-to-fame year.", fontsize=9.7, loc="left")
fig.tight_layout(); FIG = os.path.join(common.FIGURES, "fig_firstname_fc2.png")
fig.savefig(FIG, bbox_inches="tight", dpi=150); plt.close(fig)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

rows = ""
for p in order:
    dom, lang, role, sur, first, fy = p
    d = data["1929"][sur]
    cls = "post" if fy > 1929 else ""
    rows += (f"<tr class={cls}><td>{role} {sur}</td><td>{first}</td><td>{fy}</td>"
             f"<td class=n>{100*data['1900'][sur]['p_true']:.0f}%</td><td class=n>{100*data['1913'][sur]['p_true']:.0f}%</td>"
             f"<td class=n>{100*d['p_true']:.0f}%</td><td style='font-size:12px'>{' / '.join(d['opts'])}</td></tr>")
html = f"""<!doctype html><meta charset=utf-8><title>First name (language+length matched)</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:960px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.42rem}} h2{{font-size:1.1rem;margin-top:1.6rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.post td{{background:#fbeeee}} .cap{{color:#555;font-size:.9rem}} img{{max-width:100%}}</style>
<h1>What is the first name of &hellip;? &mdash; fair multiple choice</h1>
<p>Surname + role pin the person; the model picks the first name from four distractors matched on
<b>language</b> (Picasso&rsquo;s options are Spanish, Bismarck&rsquo;s German) and on <b>token length</b> (so a
short frequent name like &ldquo;John&rdquo; cannot beat &ldquo;Otto&rdquo; on length alone). Chance = 20%.
Ranke 1900 / 1913 / 1929.</p>
<img src="data:image/png;base64,{b64}">
<p class=cap>Rows by rise-to-fame year; pink = post-1929. The option set used is shown per row in the table.</p>
<h2>P(correct first name), fair options</h2>
<table><tr><th>role + surname</th><th>first</th><th>fame</th><th>1900</th><th>1913</th><th>1929</th><th>options</th></tr>{rows}</table>
"""
open(os.path.join(OUT, "firstname_fc2.html"), "w", encoding="utf-8").write(html)
print(f"\n{'role+surname -> first':30} {'fame':>4} {'len':>3} | {'1900':>6} {'1913':>6} {'1929':>6}  (chance 20%)")
for p in order:
    dom, lang, role, sur, first, fy = p
    d = data["1929"][sur]
    print(f"{sur+' -> '+first:30} {fy:>4} {d['tok_len']:>3} | {100*data['1900'][sur]['p_true']:>5.0f}% "
          f"{100*data['1913'][sur]['p_true']:>5.0f}% {100*d['p_true']:>5.0f}%  opts: {', '.join(d['opts'])}")
print("\nwrote figures/fig_firstname_fc2.png and results/prominence/firstname_fc2.html")
