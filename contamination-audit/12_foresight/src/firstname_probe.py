# -*- coding: utf-8 -*-
"""The clean name probe, the RIGHT way round: the surname + role pin the person; ask for the FIRST NAME.
"The Spanish painter Picasso's first name was ___" -> Pablo.  "The American president Reagan's first name
was ___" -> Ronald (a 1929 model can't know this). No first-name-commonness confound (the surname pins the
person), no generation/judge mess -- just P(correct first name) and the model's top guess.
Ranke 1900/1913/1929. Writes firstname.json, figures/fig_firstname.png, self-contained HTML.
Run:  python src/firstname_probe.py
"""
import os, json, math, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "prominence")
RAW = os.path.join(OUT, "firstname.json")
MODELS = ["1900", "1913", "1929"]

# (domain, role, surname, first_name, fame_year)  -- role+surname identify the person; first_name is the answer
PEOPLE = [
 ("Painting", "French painter", "Gauguin", "Paul", 1895),
 ("Painting", "Spanish painter", "Picasso", "Pablo", 1910),
 ("Painting", "American painter", "Pollock", "Jackson", 1948),
 ("Statecraft", "German statesman", "Bismarck", "Otto", 1865),
 ("Statecraft", "German political agitator", "Hitler", "Adolf", 1923),
 ("Statecraft", "American president", "Reagan", "Ronald", 1981),
 ("Physics", "British physicist", "Maxwell", "James", 1870),
 ("Physics", "German physicist", "Einstein", "Albert", 1919),
 ("Physics", "American physicist", "Feynman", "Richard", 1965),
 ("Music", "German composer", "Wagner", "Richard", 1860),
 ("Music", "Russian composer", "Stravinsky", "Igor", 1913),
 ("Music", "American singer", "Presley", "Elvis", 1956),
 ("Literature", "Russian novelist", "Tolstoy", "Leo", 1875),
 ("Literature", "American novelist", "Hemingway", "Ernest", 1926),
 ("Literature", "English writer", "Orwell", "George", 1945),
 ("Revolution", "German socialist", "Marx", "Karl", 1870),
 ("Revolution", "Russian revolutionary", "Lenin", "Vladimir", 1917),
 ("Revolution", "Cuban revolutionary", "Guevara", "Che", 1959),
 ("Science", "French chemist", "Pasteur", "Louis", 1880),
 ("Science", "Polish-French physicist", "Curie", "Marie", 1903),
 ("Science", "American doctor", "Salk", "Jonas", 1955),
 ("Exploration", "British explorer", "Stanley", "Henry", 1875),
 ("Exploration", "American aviator", "Lindbergh", "Charles", 1927),
 ("Exploration", "American astronaut", "Armstrong", "Neil", 1969),
 ("Philosophy", "German philosopher", "Nietzsche", "Friedrich", 1888),
 ("Philosophy", "Austrian philosopher", "Wittgenstein", "Ludwig", 1922),
 ("Philosophy", "French philosopher", "Sartre", "Jean", 1945),
 ("Empire", "French general", "Bonaparte", "Napoleon", 1800),
 ("Empire", "Italian dictator", "Mussolini", "Benito", 1922),
 ("Civil rights", "American reformer", "Douglass", "Frederick", 1850),
 ("Civil rights", "American civil-rights figure", "Parks", "Rosa", 1955),
]


def top_first(model, tok, prompt, k=5):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=3, num_beams=20, num_return_sequences=k,
                             no_repeat_ngram_size=2, early_stopping=True, return_dict_in_generate=True,
                             output_scores=True, pad_token_id=tok.eos_token_id)
    gen = out.sequences[:, ids.input_ids.shape[1]:]
    tr = model.compute_transition_scores(out.sequences, out.scores, out.beam_indices, normalize_logits=True)
    res = []
    for i in range(gen.shape[0]):
        m = gen[i] != tok.eos_token_id
        res.append((tok.decode(gen[i], skip_special_tokens=True).strip(), math.exp(float(tr[i][m[:tr.shape[1]]].sum()))))
    return sorted(res, key=lambda r: -r[1])


data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag in MODELS:
    if tag in data:
        continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    row = {}
    for dom, role, sur, first, fy in PEOPLE:
        prompt = f"The {role} {sur}'s first name was"
        p_true = math.exp(common.slp(model, tok, prompt, " " + first))
        tops = top_first(model, tok, prompt)
        row[sur] = {"first": first, "fame": fy, "domain": dom, "p_true": round(p_true, 4),
                    "top": tops, "correct": int(tops and first.lower() in tops[0][0].lower())}
    data[tag] = row
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

order = sorted(PEOPLE, key=lambda p: p[4])
M = np.array([[100 * data[t][p[2]]["p_true"] for t in MODELS] for p in order])

fig, ax = plt.subplots(figsize=(6.9, 9.2))
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels([f"Ranke {m}" for m in MODELS], fontsize=10)
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{p[1]} {p[2]} → {p[3]}?  ({p[4]})" for p in order], fontsize=8.1)
for i in range(len(order)):
    for j in range(3):
        v = M[i, j]
        ax.annotate(f"{v:.0f}", (j, i), ha="center", va="center", fontsize=7.6, color="white" if v > 55 else "#333")
ax.set_title("Does the model know the FIRST name of the (role + surname)?  P(correct first name), %.\n"
             "Surname+role pin the person; the first name is the test. Rows by rise-to-fame year.",
             fontsize=9.7, loc="left")
fig.tight_layout(); FIG = os.path.join(common.FIGURES, "fig_firstname.png")
fig.savefig(FIG, bbox_inches="tight", dpi=150); plt.close(fig)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

rows = ""
for p in order:
    dom, role, sur, first, fy = p
    r29 = data["1929"][sur]
    guess = ", ".join(f"{t[0]}({t[1]:.0%})" for t in r29["top"][:3])
    cls = "post" if fy > 1929 else ""
    rows += (f"<tr class={cls}><td>{role} {sur}</td><td>{first}</td><td>{fy}</td>"
             f"<td class=n>{100*data['1900'][sur]['p_true']:.0f}%</td><td class=n>{100*data['1913'][sur]['p_true']:.0f}%</td>"
             f"<td class=n>{100*r29['p_true']:.0f}%</td><td style='font-size:12px'>{guess}</td></tr>")
html = f"""<!doctype html><meta charset=utf-8><title>First name of the (role + surname)?</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:960px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.45rem}} h2{{font-size:1.12rem;margin-top:1.6rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} tr.post td{{background:#fbeeee}} .cap{{color:#555;font-size:.9rem}} img{{max-width:100%}}</style>
<h1>What is the first name of &hellip;? &mdash; the clean name probe</h1>
<p>The surname and a period-neutral role pin the person (&ldquo;the Spanish painter Picasso&rdquo;,
&ldquo;the American president Reagan&rdquo;); the model must supply the <b>first name</b>. Prompt:
&ldquo;<i>The &lt;role&gt; &lt;surname&gt;&rsquo;s first name was ___</i>&rdquo;. We read P(correct first name)
and the model&rsquo;s top guesses, for Ranke 1900 / 1913 / 1929. No first-name-commonness confound, no judge.</p>
<img src="data:image/png;base64,{b64}">
<p class=cap>Rows by rise-to-fame year. Pink = post-1929: a clean model should fail there in every column,
1929 included.</p>
<h2>P(correct first name), with the 1929 model&rsquo;s top guesses</h2>
<table><tr><th>role + surname</th><th>first</th><th>fame</th><th>1900</th><th>1913</th><th>1929</th><th>1929 top guesses</th></tr>{rows}</table>
"""
open(os.path.join(OUT, "firstname.html"), "w", encoding="utf-8").write(html)

print(f"\n{'role + surname -> first':34} {'fame':>4} | {'1900':>6} {'1913':>6} {'1929':>6}  | 1929 top guess")
for p in order:
    dom, role, sur, first, fy = p
    r = data
    g = data["1929"][sur]["top"][0]
    print(f"{sur+' -> '+first:34} {fy:>4} | {100*r['1900'][sur]['p_true']:>5.0f}% {100*r['1913'][sur]['p_true']:>5.0f}% "
          f"{100*r['1929'][sur]['p_true']:>5.0f}%  | {g[0]} ({g[1]:.0%})")
print("\nwrote figures/fig_firstname.png and results/prominence/firstname.html")
