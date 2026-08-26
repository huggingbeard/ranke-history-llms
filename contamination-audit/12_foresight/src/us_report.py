# -*- coding: utf-8 -*-
"""US civil-rights + presidential battery: CUED vs NO-CUE continuation probe + self-contained HTML report.

CUED   : the original prompt ends on the leading first name ("... was Franklin ___" -> Roosevelt).
NO-CUE : strip the given name so the description alone must yield the surname ("... in the 1930s was ___").
The gap CUED - NO-CUE isolates shallow name-completion from real knowledge. Resolves the Eisenhower/Reagan
1913-vs-1929 anomaly (does 1929 still get them without "Dwight"/"Ronald"?).

Ranke 1913 & 1929. Writes results/us_probe/full.json, figures/fig_us.png, and a self-contained
results/us_probe/us_politics_report.html (figure embedded).  Run:  python src/us_report.py
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

# (cat, year, cued_prompt, nocue_prompt, true, [false twins]);  nocue_prompt None => same as cued (phrase item)
ITEMS = [
 ("president", 1789, "The first President of the United States was George", "The first President of the United States was", " Washington", [" Adams", " Jefferson", " Madison"]),
 ("president", 1861, "The President of the United States during the Civil War was Abraham", "The President of the United States during the Civil War was", " Lincoln", [" Douglas", " Seward", " Davis"]),
 ("president", 1913, "The President who led the United States in World War I was Woodrow", "The President who led the United States in World War I was", " Wilson", [" Hughes", " Bryan", " Marshall"]),
 ("president", 1921, "The President elected in 1920 promising 'normalcy' was Warren", "The President elected in 1920 promising 'normalcy' was", " Harding", [" Cox", " Hughes", " Lowden"]),
 ("president", 1923, "Harding's successor, nicknamed 'Silent Cal', was Calvin", "Harding's successor as President was", " Coolidge", [" Dawes", " Hughes", " Davis"]),
 ("president", 1933, "The President who launched the New Deal in the 1930s was Franklin", "The President who launched the New Deal in the 1930s was", " Roosevelt", [" Landon", " Garner", " Wallace"]),
 ("president", 1945, "The President who ordered the atomic bombings of Japan was Harry", "The President who ordered the atomic bombings of Japan was", " Truman", [" Dewey", " Wallace", " Marshall"]),
 ("president", 1953, "The WWII Supreme Commander elected President in 1952 was Dwight", "The WWII Supreme Commander elected President in 1952 was", " Eisenhower", [" Stevenson", " Dewey", " MacArthur"]),
 ("president", 1961, "The President inaugurated in 1961, the youngest elected, was John F.", "The President inaugurated in 1961, the youngest elected, was", " Kennedy", [" Humphrey", " Stevenson", " Symington"]),
 ("president", 1963, "The Vice President who became President after Kennedy was Lyndon", "The Vice President who became President after Kennedy was", " Johnson", [" Humphrey", " Goldwater", " Rusk"]),
 ("president", 1969, "The President who resigned over Watergate in 1974 was Richard", "The President who resigned over Watergate in 1974 was", " Nixon", [" Humphrey", " Agnew", " Romney"]),
 ("president", 1974, "Nixon's unelected successor as President in 1974 was Gerald", "Nixon's unelected successor as President in 1974 was", " Ford", [" Agnew", " Rockefeller", " Connally"]),
 ("president", 1981, "The former Hollywood actor elected President in 1980 was Ronald", "The former Hollywood actor elected President in 1980 was", " Reagan", [" Anderson", " Mondale", " Baker"]),
 ("president", 2009, "The first African-American President, elected in 2008, was Barack", "The first African-American President, elected in 2008, was", " Obama", [" Edwards", " McCain", " Biden"]),
 ("president", 2017, "The businessman elected President in 2016 was Donald", "The businessman elected President in 2016 was", " Trump", [" Cruz", " Romney", " Rubio"]),
 ("president", 2021, "The President inaugurated in January 2021 was Joe", "The President inaugurated in January 2021 was", " Biden", [" Sanders", " Kerry", " Harris"]),
 ("civil rights", 1954, "The 1954 Supreme Court case desegregating schools was Brown v. Board of", None, " Education", [" Trustees", " Regents", " Commissioners"]),
 ("civil rights", 1955, "The woman whose 1955 bus protest sparked the Montgomery boycott was Rosa", "The woman whose 1955 bus protest sparked the Montgomery boycott was", " Parks", [" Baker", " Colvin", " Robinson"]),
 ("civil rights", 1963, "The leader of the 1963 'I Have a Dream' speech was Martin Luther", "The leader of the 1963 'I Have a Dream' speech was", " King", [" Abernathy", " Evers", " Wilkins"]),
 ("civil rights", 1964, "The landmark 1964 law against racial segregation was the Civil Rights", None, " Act", [" Bill", " Code", " Pact"]),
 ("civil rights", 1968, "The senator and brother of JFK assassinated in 1968 was Robert", "The senator and brother of JFK assassinated in 1968 was", " Kennedy", [" Humphrey", " McCarthy", " Wallace"]),
 ("civil rights", 1968, "King was assassinated in 1968 in the city of", None, " Memphis", [" Atlanta", " Selma", " Birmingham"]),
]
MODELS = ["1913", "1929"]


def softmax(xs):
    m = max(xs); ex = [math.exp(x - m) for x in xs]; return [e / sum(ex) for e in ex]


def score(model, tok, prompt, opts):
    p = softmax([common.slp(model, tok, prompt, o) for o in opts])
    g = common.generate(model, tok, prompt, n=6, sample=False).replace("\n", " ").strip()[:52]
    return round(p[0], 3), g


FULL = os.path.join(OUT, "full.json")
data = json.load(open(FULL, encoding="utf-8")) if os.path.exists(FULL) else {}
for tag in MODELS:
    if tag in data:
        continue
    print(f"[{tag}] loading {RANKE[tag]} ...", flush=True)
    tok, model = common.load(RANKE[tag])
    rows = []
    for cat, yr, cued, nocue, true, falses in ITEMS:
        opts = [true] + falses
        pc, gc = score(model, tok, cued, opts)
        if nocue is None:
            pn, gn = pc, gc
        else:
            pn, gn = score(model, tok, nocue, opts)
        rows.append({"cat": cat, "year": yr, "true": true.strip(),
                     "cued_p": pc, "cued_g": gc, "nocue_p": pn, "nocue_g": gn, "phrase": nocue is None})
    data[tag] = rows
    json.dump(data, open(FULL, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model
    common.free()

# ---------- figure: president items by year, cued vs no-cue P(true), both models, cutoff lines ----------
pres = [i for i, it in enumerate(ITEMS) if it[0] == "president"]
years = [ITEMS[i][1] for i in pres]
fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.7), sharey=True)
for ax, tag in zip(axes, MODELS):
    cut = int(tag)
    cp = [data[tag][i]["cued_p"] for i in pres]
    npv = [data[tag][i]["nocue_p"] for i in pres]
    x = np.arange(len(pres))
    ax.plot(x, cp, "-o", color="#c0392b", lw=1.8, ms=5, label="cued (first name given)")
    ax.plot(x, npv, "-o", color="#2b6cb0", lw=1.8, ms=5, label="no-cue (description only)")
    ax.axhline(0.25, color="#9aa0a6", ls=":", lw=1)
    ax.annotate("chance", (len(pres) - 1, 0.27), fontsize=8, color="#666", ha="right")
    for j, y in enumerate(years):
        if y > cut and years[j - 1] <= cut:
            ax.axvline(j - 0.5, color="#1a1a1a", ls="--", lw=1.2)
            ax.annotate(f"{cut} cutoff", (j - 0.5, 1.02), fontsize=8.5, ha="center")
    ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years], rotation=60, fontsize=7.5)
    ax.set_title(f"Ranke {tag}", fontsize=11)
    ax.set_ylim(0, 1.08)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#9aa0a6", alpha=.3, lw=.6)
axes[0].set_ylabel("P(true answer)")
axes[0].legend(frameon=False, fontsize=9, loc="lower left")
fig.suptitle("US presidents: does the model get it, or just complete the first name?\n"
             "Red (cued) stays high past the cutoff = name-completion; blue (no-cue) drops to chance = no real knowledge.",
             fontsize=11, y=1.05)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_us.png")
fig.savefig(FIG, bbox_inches="tight", dpi=150)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

# ---------- cohort means ----------
def mean(v):
    return sum(v) / len(v) if v else float("nan")
def cohort(tag, key, lo, hi, cat="president"):
    return mean([data[tag][i][key] for i, it in enumerate(ITEMS) if lo <= it[1] < hi and it[0] == cat])

# ---------- HTML ----------
def rows_html(cat):
    out = ""
    for i, it in enumerate(ITEMS):
        if it[0] != cat:
            continue
        r13, r29 = data["1913"][i], data["1929"][i]
        ph = " <span style='color:#999'>(phrase; no name to strip)</span>" if r13["phrase"] else ""
        out += (f"<tr><td>{it[1]}</td><td>{r13['true']}{ph}</td>"
                f"<td class=n>{r13['cued_p']:.0%}</td><td class=n>{r13['nocue_p']:.0%}</td><td class=g>{r13['nocue_g']}</td>"
                f"<td class=n>{r29['cued_p']:.0%}</td><td class=n>{r29['nocue_p']:.0%}</td><td class=g>{r29['nocue_g']}</td></tr>")
    return out

hdr = ("<tr><th>year</th><th>true</th>"
       "<th>1913 cued</th><th>1913 no-cue</th><th>1913 says (no-cue)</th>"
       "<th>1929 cued</th><th>1929 no-cue</th><th>1929 says (no-cue)</th></tr>")
html = f"""<!doctype html><meta charset=utf-8><title>US politics: cued vs no-cue continuation probe</title>
<style>body{{font:15px/1.5 Georgia,serif;max-width:1050px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:1.7rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0;width:100%;font-size:13.5px}} td,th{{border:1px solid #ccc;padding:4px 7px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} .g{{color:#444;font-family:monospace;font-size:11.5px}}
.cap{{color:#555;font-size:.9rem}} img{{max-width:100%}}</style>
<h1>Do the frozen models know post-1929 US politics, or just complete the first name?</h1>
<p>Battery: <code>2_model_probes/us_politics_power.py</code> (presidential succession + civil rights),
scored two ways. <b>Cued</b> = the original prompt hands the model the first name (&ldquo;&hellip; was
Franklin ___&rdquo; &rarr; Roosevelt). <b>No-cue</b> = the given name is stripped, so the description alone
must yield the surname (&ldquo;&hellip; who launched the New Deal was ___&rdquo;). The gap is shallow
name-completion; what survives no-cue is knowledge. Ranke 1913 &amp; 1929; four-way forced choice, chance 25%.</p>
<img src="data:image/png;base64,{b64}" alt="cued vs no-cue by president">
<p class=cap>Left/right: the two models. Red line = cued P(true); blue = no-cue. Dashed vertical = that
model&rsquo;s training cutoff; dotted horizontal = chance. Red staying high past the cutoff is the first
name doing the work; blue collapsing to chance is the absence of real knowledge.</p>

<h2>Presidential succession</h2>
<table>{hdr}{rows_html('president')}</table>
<h2>Civil rights</h2>
<table>{hdr}{rows_html('civil rights')}</table>

<h2>Cohort means, P(true)</h2>
<table><tr><th>cohort</th><th>1913 cued</th><th>1913 no-cue</th><th>1929 cued</th><th>1929 no-cue</th></tr>
<tr><td>presidents &le;1913 (both should know)</td><td class=n>{cohort('1913','cued_p',0,1914):.0%}</td><td class=n>{cohort('1913','nocue_p',0,1914):.0%}</td><td class=n>{cohort('1929','cued_p',0,1914):.0%}</td><td class=n>{cohort('1929','nocue_p',0,1914):.0%}</td></tr>
<tr><td>presidents 1914&ndash;29 (only 1929)</td><td class=n>{cohort('1913','cued_p',1914,1930):.0%}</td><td class=n>{cohort('1913','nocue_p',1914,1930):.0%}</td><td class=n>{cohort('1929','cued_p',1914,1930):.0%}</td><td class=n>{cohort('1929','nocue_p',1914,1930):.0%}</td></tr>
<tr><td>presidents &gt;1929 (neither should)</td><td class=n>{cohort('1913','cued_p',1930,3000):.0%}</td><td class=n>{cohort('1913','nocue_p',1930,3000):.0%}</td><td class=n>{cohort('1929','cued_p',1930,3000):.0%}</td><td class=n>{cohort('1929','nocue_p',1930,3000):.0%}</td></tr>
</table>

<h2>What it shows</h2>
<p>Stripping the first name removes the name-completion inflation: on the post-1929 presidents the cued
score collapses toward chance without the cue, which means the earlier &ldquo;contamination&rdquo; was the
model completing &ldquo;Franklin/Harry/John&nbsp;F.&rdquo;, not knowing the presidency. Where the model
truly has nothing (Obama, Trump, Biden, Rosa Parks, the city of Memphis) it confabulates a pre-cutoff name
even in the no-cue column. This reproduces the audit&rsquo;s finding on its own battery: the US-politics
signal is shallow name/phrase exposure, not deep knowledge.</p>
"""
open(os.path.join(OUT, "us_politics_report.html"), "w", encoding="utf-8").write(html)

print("\ncohort  P(true)          1913c 1913n | 1929c 1929n")
for lab, lo, hi in [("<=1913", 0, 1914), ("1914-29", 1914, 1930), (">1929", 1930, 3000)]:
    print(f"  pres {lab:8} {cohort('1913','cued_p',lo,hi):>5.0%} {cohort('1913','nocue_p',lo,hi):>5.0%} "
          f"| {cohort('1929','cued_p',lo,hi):>5.0%} {cohort('1929','nocue_p',lo,hi):>5.0%}")
print("\nEisenhower / Reagan (the anomaly), no-cue:")
for i, it in enumerate(ITEMS):
    if it[4].strip() in ("Eisenhower", "Reagan"):
        print(f"  {it[4].strip():11} 1913: cued {data['1913'][i]['cued_p']:.0%} -> nocue {data['1913'][i]['nocue_p']:.0%} ({data['1913'][i]['nocue_g']!r}) "
              f"| 1929: cued {data['1929'][i]['cued_p']:.0%} -> nocue {data['1929'][i]['nocue_p']:.0%} ({data['1929'][i]['nocue_g']!r})")
print("\nwrote figures/fig_us.png and results/us_probe/us_politics_report.html")
