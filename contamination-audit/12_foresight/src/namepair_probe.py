# -*- coding: utf-8 -*-
"""Does the frozen model carry a POST-cutoff name-pair? "Dwight -> Eisenhower" / "Ronald -> Reagan" can
only be learned from text written after those men became famous (post-1929). A truly time-locked reader
should NOT have the pairing. So strip everything except the bare first name and measure:
  - P(surname | first name)  -- absolute probability of the exact surname right after the first name;
  - the top-10 continuations of the bare first name (what it naturally completes to).
Pre-cutoff pairs (Woodrow->Wilson, Theodore->Roosevelt) are controls both models must have.

Ranke 1913 & 1929. Writes results/us_probe/namepair.json, figures/fig_namepair.png, and a self-contained
results/us_probe/namepair_report.html.  Run:  python src/namepair_probe.py
"""
import os, json, math, base64
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common
from foresight_bank import RANKE

OUT = os.path.join(common.RESULTS, "us_probe")
os.makedirs(OUT, exist_ok=True)

# (first, surname, fame_year, cohort)  fame_year = when the surname became linkable to the first name
PAIRS = [
 ("Theodore", "Roosevelt", 1901, "pre-1913 control"),
 ("Woodrow", "Wilson", 1913, "pre-1913 control"),
 ("Warren", "Harding", 1921, "1914-29 (only 1929)"),
 ("Calvin", "Coolidge", 1923, "1914-29 (only 1929)"),
 ("Franklin", "Roosevelt", 1933, "post-1929"),
 ("Harry", "Truman", 1945, "post-1929"),
 ("Dwight", "Eisenhower", 1953, "post-1929"),
 ("Richard", "Nixon", 1969, "post-1929"),
 ("Gerald", "Ford", 1974, "post-1929"),
 ("Ronald", "Reagan", 1981, "post-1929"),
 ("Barack", "Obama", 2009, "post-1929"),
 ("Donald", "Trump", 2017, "post-1929"),
]
MODELS = ["1913", "1929"]


def top10(model, tok, prompt, k_new=4):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=k_new, num_beams=25, num_return_sequences=10,
                             no_repeat_ngram_size=2, early_stopping=True,
                             return_dict_in_generate=True, output_scores=True, pad_token_id=tok.eos_token_id)
    gen = out.sequences[:, ids.input_ids.shape[1]:]
    trans = model.compute_transition_scores(out.sequences, out.scores, out.beam_indices, normalize_logits=True)
    res = []
    for i in range(gen.shape[0]):
        mask = gen[i] != tok.eos_token_id
        lp = float(trans[i][mask[:trans.shape[1]]].sum())
        res.append([tok.decode(gen[i], skip_special_tokens=True).replace("\n", " ").strip(), math.exp(lp)])
    res.sort(key=lambda r: -r[1])
    return res


data = json.load(open(os.path.join(OUT, "namepair.json"), encoding="utf-8")) if os.path.exists(os.path.join(OUT, "namepair.json")) else {}
for tag in MODELS:
    if tag in data:
        continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    rows = {}
    for first, surname, fy, coh in PAIRS:
        psur = math.exp(common.slp(model, tok, first, " " + surname))   # P(surname tokens | first name)
        tops = top10(model, tok, first)
        rank = next((k + 1 for k, r in enumerate(tops) if surname.lower() in r[0].lower()), None)
        rows[first] = {"surname": surname, "fame_year": fy, "cohort": coh,
                       "p_surname": psur, "rank_in_top10": rank, "top10": tops}
    data[tag] = rows
    json.dump(data, open(os.path.join(OUT, "namepair.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

# ---------- figure: P(surname|first), 1913 vs 1929, per pair ----------
firsts = [p[0] for p in PAIRS]
labels = [f"{p[0]}\n{p[1]}" for p in PAIRS]
x = np.arange(len(PAIRS)); w = 0.38
fig, ax = plt.subplots(figsize=(12.8, 4.8))
p13 = [data["1913"][f]["p_surname"] for f in firsts]
p29 = [data["1929"][f]["p_surname"] for f in firsts]
ax.bar(x - w/2, p13, w, color="#9aa0a6", edgecolor="#1a1a1a", lw=.6, label="Ranke 1913")
ax.bar(x + w/2, p29, w, color="#2b6cb0", edgecolor="#1a1a1a", lw=.6, label="Ranke 1929")
for i, p in enumerate(PAIRS):
    if p[2] > 1929:
        ax.axvspan(i - 0.5, i + 0.5, color="#c0392b", alpha=.05)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("P(surname | first name)")
ax.set_title("Does the model link the first name to the surname? P(surname | bare first name).\n"
             "Red-shaded = the pairing became famous only AFTER 1929, so a clean model should not have it.",
             fontsize=10.5, loc="left")
ax.legend(frameon=False, fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#9aa0a6", alpha=.3, lw=.6)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_namepair.png")
fig.savefig(FIG, bbox_inches="tight", dpi=150)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

# ---------- HTML ----------
def topstr(tag, first):
    return "; ".join(f"{t[0]} ({t[1]:.1%})" for t in data[tag][first]["top10"][:6])
trows = ""
for first, surname, fy, coh in PAIRS:
    r13, r29 = data["1913"][first], data["1929"][first]
    trows += (f"<tr><td>{first} &rarr; {surname}</td><td>{coh}</td>"
              f"<td class=n>{r13['p_surname']:.1%}</td><td class=n>{'#'+str(r13['rank_in_top10']) if r13['rank_in_top10'] else '&mdash;'}</td>"
              f"<td class=n>{r29['p_surname']:.1%}</td><td class=n>{'#'+str(r29['rank_in_top10']) if r29['rank_in_top10'] else '&mdash;'}</td></tr>")
ex = ""
for first in ["Dwight", "Ronald", "Harry", "Woodrow"]:
    ex += (f"<p><b>&ldquo;{first}&rdquo;</b><br>1913: {topstr('1913', first)}<br>1929: {topstr('1929', first)}</p>")
html = f"""<!doctype html><meta charset=utf-8><title>Name-pair leakage: Dwight-&gt;Eisenhower etc.</title>
<style>body{{font:15px/1.55 Georgia,serif;max-width:1000px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.45rem}} h2{{font-size:1.12rem;margin-top:1.7rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}} td,th{{border:1px solid #ccc;padding:4px 8px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} .cap{{color:#555;font-size:.9rem}} img{{max-width:100%}}</style>
<h1>Does the 1929 reader carry post-1929 name-pairs it shouldn&rsquo;t?</h1>
<p>&ldquo;Dwight&nbsp;&rarr;&nbsp;Eisenhower&rdquo; and &ldquo;Ronald&nbsp;&rarr;&nbsp;Reagan&rdquo; are learnable
only from text written after those men became famous &mdash; after 1929. So we strip everything except the
<b>bare first name</b> and measure how strongly the model expects the surname next. Pre-1913 pairs
(Theodore&nbsp;Roosevelt, Woodrow&nbsp;Wilson) are controls both models must have. Ranke 1913 &amp; 1929.</p>
<img src="data:image/png;base64,{b64}" alt="P(surname|first) by pair">
<p class=cap>Bars: P(surname | bare first name). Grey = 1913, blue = 1929. Red-shaded columns are pairings
that became famous only after 1929 &mdash; a clean model should show little there.</p>
<h2>P(surname | first name), and the surname&rsquo;s rank in the top-10 continuations</h2>
<table><tr><th>pair</th><th>cohort</th><th>1913 P</th><th>1913 rank</th><th>1929 P</th><th>1929 rank</th></tr>{trows}</table>
<h2>What the bare first name actually completes to (top-6, absolute prob)</h2>
{ex}
"""
open(os.path.join(OUT, "namepair_report.html"), "w", encoding="utf-8").write(html)

print(f"\n{'pair':22} {'cohort':20} {'1913 P':>8} {'rk':>3} {'1929 P':>8} {'rk':>3}")
for first, surname, fy, coh in PAIRS:
    r13, r29 = data["1913"][first], data["1929"][first]
    print(f"{first+'->'+surname:22} {coh:20} {r13['p_surname']:>8.1%} {str(r13['rank_in_top10'] or '-'):>3} "
          f"{r29['p_surname']:>8.1%} {str(r29['rank_in_top10'] or '-'):>3}")
for first in ["Dwight", "Ronald", "Harry"]:
    print(f"\n{first!r} top-6:")
    print("  1913:", "; ".join(f"{t[0]}({t[1]:.0%})" for t in data['1913'][first]['top10'][:6]))
    print("  1929:", "; ".join(f"{t[0]}({t[1]:.0%})" for t in data['1929'][first]['top10'][:6]))
print("\nwrote figures/fig_namepair.png and results/us_probe/namepair_report.html")
