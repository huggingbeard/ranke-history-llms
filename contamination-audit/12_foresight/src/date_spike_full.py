# -*- coding: utf-8 -*-
"""Flat-vs-spike contamination test, with a proper BENCHMARK (answers "how do we know 2x is clean?").
Files used:  reads/writes  results/superlatives/date_spike_full.json
             figures/fig_date_spike_full.png ; results/superlatives/date_spike_full.html
Models:      uzh-echist-org/Ranke-4B-{1913,1929}-1202-Instruct
Date: 2026-07-08   Version: 1.0

Method. For a dated event we score P(year) over a +/-8-year window and define spike = P(true year) / mean(other years
in the window). spike=1 is perfectly flat (no preference). We anchor the scale with two reference sets, per model:
  KNOWN  = famous PRE-1913 facts the model must know  -> what a real memory looks like (high spikes).
  NULL   = FICTIONAL events with invented names + an assigned year -> what NO knowledge looks like (low spikes),
           because the model cannot have learned a date that does not exist. The 95th percentile of the NULL spikes is
           the leakage threshold: a post-cutoff event only counts as contamination if it clears what a made-up event
           reaches by chance. TEST = the real post-cutoff events (WWI..moon landing).
Run:  python src/date_spike_full.py
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

OUT = os.path.join(common.RESULTS, "superlatives")
INSTRUCT = {"1913": "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "1929": "uzh-echist-org/Ranke-4B-1929-1202-Instruct"}
WIN = 8

KNOWN = [  # famous pre-1913 facts (both models must know) -> the "knowledge" reference
 ("French Revolution", "The French Revolution began with the storming of the Bastille in the year", 1789),
 ("Battle of Waterloo", "Napoleon was finally defeated at the Battle of Waterloo in the year", 1815),
 ("Victoria crowned", "Queen Victoria came to the British throne in the year", 1837),
 ("Revolutions of 1848", "A wave of revolutions swept across Europe in the year", 1848),
 ("Crimean War", "Britain and France went to war with Russia in the Crimea in the year", 1854),
 ("US Civil War began", "The American Civil War began with the attack on Fort Sumter in the year", 1861),
 ("Emancipation Proclamation", "President Lincoln issued the Emancipation Proclamation in the year", 1863),
 ("Lincoln assassinated", "President Abraham Lincoln was assassinated in the year", 1865),
 ("Franco-Prussian War", "The Franco-Prussian War broke out in the year", 1870),
 ("Bell's telephone", "Alexander Graham Bell invented the telephone in the year", 1876),
 ("Edison's light bulb", "Thomas Edison perfected the electric light bulb in the year", 1879),
 ("Berlin Conference", "The European powers carved up Africa at the Berlin Conference in the year", 1884),
 ("Eiffel Tower", "The Eiffel Tower was completed in Paris in the year", 1889),
 ("Spanish-American War", "The Spanish-American War was fought in the year", 1898),
 ("Victoria died", "Queen Victoria died in the year", 1901),
 ("Wright brothers fly", "The Wright brothers made the first powered flight in the year", 1903),
 ("Russo-Japanese War", "The Russo-Japanese War came to an end in the year", 1905),
 ("San Francisco quake", "The great San Francisco earthquake struck in the year", 1906),
 ("Titanic sinks", "The ocean liner Titanic struck an iceberg and sank in the year", 1912),
]
NULL = [  # FICTIONAL events (invented names) -> the "no-knowledge" reference. Assigned year cannot have been learned.
 ("*Treaty of Kesselheim", "The Treaty of Kesselheim was signed in the year", 1888),
 ("*Battle of Verronne", "The Battle of Verronne was fought in the year", 1904),
 ("*novelist Aldous Brennt", "The novelist Aldous Brennt published his masterpiece in the year", 1877),
 ("*Morvenia canal", "The Grand Canal of Morvenia was completed in the year", 1866),
 ("*composer Reinhardt Voss", "The composer Reinhardt Voss died in the year", 1898),
 ("*city of Halveston", "The city of Halveston was founded in the year", 1851),
 ("*Marlowe Exhibition", "The Marlowe Exhibition opened its doors in the year", 1893),
 ("*philosopher Emeric Tolliver", "The philosopher Emeric Tolliver was born in the year", 1858),
 ("*Kingdom of Astravia", "The Kingdom of Astravia declared its independence in the year", 1871),
 ("*fire of Pemberton", "The great fire of Pemberton destroyed the old city in the year", 1882),
 ("*explorer Cornelius Bathe", "The explorer Cornelius Bathe reached the far interior in the year", 1909),
 ("*University of Kelmswick", "The University of Kelmswick was founded in the year", 1846),
 ("*painter Ottoline Crest", "The painter Ottoline Crest completed her famous mural in the year", 1901),
 ("*Wharton Canal", "The Wharton Canal was opened to traffic in the year", 1869),
 ("*statesman Aurelio Manx", "The statesman Aurelio Manx became prime minister in the year", 1896),
]
TEST = [  # real post-cutoff events; group tags drive the colours
 ("First World War began", "The First World War broke out in Europe in the year", 1914, "1914-29"),
 ("Bolsheviks seized power", "The Bolsheviks seized power in Russia in the year", 1917, "1914-29"),
 ("US entered the war", "The United States entered the First World War in the year", 1917, "1914-29"),
 ("Armistice ends WWI", "The First World War ended with an armistice in the year", 1918, "1914-29"),
 ("Spanish flu", "The great influenza pandemic swept the world in the year", 1918, "1914-29"),
 ("Treaty of Versailles", "The Treaty of Versailles was signed in the year", 1919, "1914-29"),
 ("German hyperinflation", "The German mark collapsed in a ruinous hyperinflation in the year", 1923, "1914-29"),
 ("Lindbergh's flight", "Charles Lindbergh flew solo across the Atlantic in the year", 1927, "1914-29"),
 ("Wall Street Crash", "The Wall Street stock market crashed, starting the Depression, in the year", 1929, "1914-29"),
 ("Hitler made Chancellor", "Adolf Hitler became Chancellor of Germany in the year", 1933, "WW2/atomic"),
 ("Second World War began", "The Second World War began with the invasion of Poland in the year", 1939, "WW2/atomic"),
 ("Pearl Harbor", "Japan attacked Pearl Harbor, bringing America into the war, in the year", 1941, "WW2/atomic"),
 ("D-Day landings", "The Allied D-Day landings in Normandy took place in the year", 1944, "WW2/atomic"),
 ("Hiroshima atomic bomb", "The United States dropped the first atomic bomb on Hiroshima in the year", 1945, "WW2/atomic"),
 ("Second World War ended", "The Second World War came to an end in the year", 1945, "WW2/atomic"),
 ("Brown v. Board", "The Supreme Court struck down school segregation in Brown v. Board of Education in the year", 1954, "civil rights"),
 ("Montgomery bus boycott", "Rosa Parks refused to give up her bus seat, sparking the Montgomery boycott, in the year", 1955, "civil rights"),
 ("MLK 'I Have a Dream'", "Martin Luther King delivered his 'I Have a Dream' speech in the year", 1963, "civil rights"),
 ("JFK assassinated", "President John F. Kennedy was assassinated in Dallas in the year", 1963, "civil rights"),
 ("Civil Rights Act", "The Civil Rights Act outlawing segregation became law in the year", 1964, "civil rights"),
 ("MLK assassinated", "Martin Luther King was assassinated in the year", 1968, "civil rights"),
 ("Moon landing", "American astronauts first walked on the Moon in the year", 1969, "civil rights"),
]


def softmax(xs):
    m = max(xs); e = [math.exp(x - m) for x in xs]; s = sum(e); return [x / s for x in e]


def spike(model, tok, prompt, ty):
    years = list(range(ty - WIN, ty + WIN + 1))
    p = softmax([common.slp(model, tok, prompt, f" {y}") for y in years])
    pt = p[years.index(ty)]
    others = [p[i] for i, y in enumerate(years) if y != ty]
    return {"p_true": round(pt, 4), "argmax": years[int(np.argmax(p))], "spike": round(pt / (np.mean(others) + 1e-12), 2)}


RAW = os.path.join(OUT, "date_spike_full.json")
data = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
for tag, mid in INSTRUCT.items():
    if tag in data:
        continue
    print(f"[{tag}] loading {mid} ...", flush=True)
    tok, model = common.load(mid)
    r = {"known": {}, "null": {}, "test": {}}
    for lab, pr, ty in KNOWN:
        r["known"][lab] = spike(model, tok, pr, ty)
    for lab, pr, ty in NULL:
        r["null"][lab] = spike(model, tok, pr, ty)
    for lab, pr, ty, grp in TEST:
        r["test"][lab] = {**spike(model, tok, pr, ty), "true": ty, "grp": grp}
    data[tag] = r
    json.dump(data, open(RAW, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    del tok, model; common.free()

# ---------- figure ----------
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "savefig.dpi": 200})
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
GCOL = {"1914-29": "#e08214", "WW2/atomic": "#c0392b", "civil rights": "#7a2f8f"}
CUT = {"1913": 1913, "1929": 1929}
CHANCE = 100.0 / (2 * WIN + 1)                                                  # flat = 1 of 17 years = 5.9%
MSIZE = {"leaked": 130, "knows": 85, "clean": 55}
MSHAPE = {"leaked": "v", "knows": "s", "clean": "o"}
fig, axes = plt.subplots(2, 1, figsize=(12.5, 10.8))
# ONE shared leakage threshold: the two per-model nulls are statistically the same (both centre on chance),
# so pool all fictional events and take a single 95th percentile — avoids a spurious 11%-vs-13% difference.
_pooled_null = [v["p_true"] * 100 for t in ("1913", "1929") for v in data[t]["null"].values()]
SHARED_THR = float(np.percentile(_pooled_null, 95))
thr = {"1913": SHARED_THR, "1929": SHARED_THR}
for ax, tag in zip(axes, ["1913", "1929"]):
    kn = np.array([v["p_true"] * 100 for v in data[tag]["known"].values()])
    nu = np.array([v["p_true"] * 100 for v in data[tag]["null"].values()])
    t95 = thr[tag]
    grid = np.linspace(0, 100, 500)

    def kde(vals, bw=4.0):
        return np.array([np.sum(np.exp(-((g - vals) / bw) ** 2 / 2)) for g in grid])
    dk, dn = kde(kn), kde(nu); sc = max(dk.max(), dn.max())
    ax.fill_between(grid, 0, dn / sc, color="#8a8f98", alpha=.38, lw=0)
    ax.fill_between(grid, 0, dk / sc, color="#2e7d32", alpha=.32, lw=0)
    ax.axvline(CHANCE, color="#1f5f9e", ls=":", lw=1.4)
    ax.axvline(t95, color="#111", ls="--", lw=1.4)
    ax.annotate(f"chance\n{CHANCE:.1f}%", (CHANCE, 1.30), color="#1f5f9e", fontsize=8.3, ha="center", va="bottom")
    ax.annotate(f"threshold {t95:.0f}%", (t95 + 2, 1.30), color="#111", fontsize=8.3, ha="left", va="bottom")
    seen = {}; lk = 0
    for lab, v in sorted(data[tag]["test"].items(), key=lambda kv: kv[1]["p_true"]):
        x = v["p_true"] * 100; g = v["grp"]
        cat = "leaked" if (v["true"] > CUT[tag] and x > t95) else ("knows" if x > t95 else "clean")
        if cat == "leaked":                                    # stagger leaked markers/labels so callouts don't collide
            yy = 1.08 + (lk % 4) * 0.16; lk += 1
        else:
            key = round(x / 4); row = seen.get(key, 0); seen[key] = row + 1; yy = 1.06 + row * 0.11
        ax.scatter([x], [yy], s=MSIZE[cat], color=GCOL[g], marker=MSHAPE[cat],
                   edgecolor="#111" if cat == "leaked" else "white", lw=.9, zorder=5)
        if cat == "leaked":
            ax.annotate(f"{lab} {v['true']}", (x, yy), xytext=(5, 4), textcoords="offset points",
                        fontsize=7, ha="left", va="bottom", color="#111")
    ax.set_xlim(-3, 108); ax.set_ylim(0, 1.95); ax.set_yticks([])
    ax.set_ylabel("relative density\n(reference distributions)", fontsize=8.5, color="#666")
    ax.set_xlabel("P(true year), %   —   probability the model places on the correct year")
    ax.set_title(f"Ranke {tag} — trained to {tag}", fontsize=12.5, color="#b8332a" if tag == "1913" else "#1f5f9e", fontweight="bold", loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
_m = lambda mk, ec, lab: Line2D([0], [0], marker=mk, color="w", markerfacecolor="#888", markeredgecolor=ec, markersize=11, label=lab)
handles = [Patch(fc="#2e7d32", alpha=.42, label="known pre-1913 facts (reference: real memory)"),
           Patch(fc="#8a8f98", alpha=.42, label="fictional events (reference: no knowledge)"),
           Line2D([0], [0], color="#1f5f9e", ls=":", label=f"chance = {CHANCE:.1f}% (1 of 17 years)"),
           Line2D([0], [0], color="#111", ls="--", label="leakage threshold (95th pct of fictional)"),
           _m("v", "#111", "post-cutoff & above threshold → LEAKED"),
           _m("s", "white", "pre-cutoff & above threshold → legitimate knowledge"),
           _m("o", "white", "below threshold → clean (correctly ignorant)"),
           Patch(fc="#e08214", label="event era: 1914–29"), Patch(fc="#c0392b", label="event era: WW2 / atomic"),
           Patch(fc="#7a2f8f", label="event era: civil rights")]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.4, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Figure 2.  How much probability the frozen model puts on the correct year: known facts vs fictional events vs post-cutoff events.",
             fontsize=12.2, y=1.0, x=0.01, ha="left")
fig.text(0.01, 0.963, "Green hump = pre-1913 facts the model must know (lands near 100%); grey hump = invented events it "
         "cannot (lands at chance, ~6%). Markers are the post-cutoff test events — shape = category, colour = era.",
         fontsize=8.7, color="#555", ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
FIG = os.path.join(common.FIGURES, "fig_date_spike_full.png")
fig.savefig(FIG, bbox_inches="tight"); plt.close(fig)

# ---------- HTML write-up ----------
import base64
b64 = base64.b64encode(open(FIG, "rb").read()).decode()
b64i = base64.b64encode(open(os.path.join(common.FIGURES, "fig_intuition.png"), "rb").read()).decode()
intu = json.load(open(os.path.join(OUT, "intuition.json"), encoding="utf-8"))
CUT = {"1913": 1913, "1929": 1929}


def classify(tag, ev):
    """Cutoff-aware, in P(true year) %: high P on a pre-cutoff event is legitimate knowledge, not leakage."""
    ty, s, t = ev["true"], ev["p_true"] * 100, thr[tag]
    if ty <= CUT[tag]:
        return ("knows", "knows ✓") if s > t else ("gap", "—")
    return ("leaked", "LEAKED") if s > t else ("clean", "clean")


km = {t: np.median([v["p_true"] * 100 for v in data[t]["known"].values()]) for t in ("1913", "1929")}
nm = {t: np.median([v["p_true"] * 100 for v in data[t]["null"].values()]) for t in ("1913", "1929")}
ERAS = [("1914-29", "Just past 1913 — the Great War era", "known to the 1929 model, forbidden to the 1913 model"),
        ("WW2/atomic", "The Second World War and the atomic bomb (1933–45)", "forbidden to both models"),
        ("civil rights", "Post-war America and civil rights (1954–69)", "forbidden to both models")]
rows = ""
for grp, gname, gnote in ERAS:
    rows += f"<tr class=grp><td colspan=6>{gname} <span style='font-weight:normal;color:#777'>— {gnote}</span></td></tr>"
    for lab, pr, ty, g in TEST:
        if g != grp:
            continue
        a, b = data["1913"]["test"][lab], data["1929"]["test"][lab]
        ca, la = classify("1913", a); cb, lb = classify("1929", b)
        rows += (f"<tr><td>{lab}</td><td class=n>{ty}</td>"
                 f"<td class='n {ca}'>{100*a['p_true']:.0f}%</td><td class='{ca}'>{la}</td>"
                 f"<td class='n {cb}'>{100*b['p_true']:.0f}%</td><td class='{cb}'>{lb}</td></tr>")
n_leak_13 = sum(1 for lab, _, _, _ in TEST if data["1913"]["test"][lab]["spike"] > thr["1913"])
html = f"""<!doctype html><meta charset=utf-8><title>Contamination by date-spike — benchmarked</title>
<style>body{{font:15.5px/1.6 Georgia,serif;max-width:1060px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:1.8rem;border-bottom:1px solid #ddd;padding-bottom:2px}}
.cap{{color:#555;font-size:.93rem}} img{{max-width:100%}} code{{background:#f4f4f4;padding:1px 4px;border-radius:3px}}
table{{border-collapse:collapse;margin:1rem 0;width:100%;font-size:.92rem}} td,th{{border:1px solid #ccc;padding:4px 9px}}
.n{{text-align:right;font-variant-numeric:tabular-nums}} .leaked{{background:#fbecec;color:#a11;font-weight:bold}}
.clean{{color:#2e7d32}} .knows{{color:#1f5f9e}} .gap{{color:#bbb}}
blockquote{{border-left:3px solid #bbb;margin:1rem 0;padding:.4rem 1rem;background:#fafafa}}
tr.grp td{{background:#f0f0f2;font-weight:bold}} .hdr{{color:#666;font-size:.85rem}}</style>
<h1>Did the frozen models leak the future? A benchmarked date-spike test</h1>
<p class=hdr>Ranke-4B-1913 &amp; 1929 (instruct) · 2026-07-08 · v1.2 (P(true)% metric · intuition walkthrough · cutoff-aware
labelling) · src/date_spike_full.py, src/intuition.py · figures/fig_date_spike_full.png · figures/fig_intuition.png</p>

<h2>Start with one question</h2>
<p>Ask the 1913 model something it should know cold:</p>
<blockquote><b>Q.</b> {intu['Russo-Japanese War ends']['q']}<br>
<b>A.</b> <i>&ldquo;{intu['Russo-Japanese War ends']['answer']}&rdquo;</i></blockquote>
<p>It says 1905, correctly and with confidence. But one answer proves little &mdash; a model can sound sure and be
guessing. So instead of trusting a single answer we look at the <em>whole distribution</em>: score how much probability
the model puts on each year in a window, and see whether it concentrates on the true year or spreads out.</p>
<img src="data:image/png;base64,{b64i}">
<p class=cap>Figure 1. Six questions to the 1913 model; each bar is the probability it puts on that year (red bar = the
true year; legend in the figure). <b>Top row</b> &mdash; facts it knows (Russo-Japanese War, Titanic): one tall spike
near 100%. <b>Middle row</b> &mdash; two <em>invented</em> events, &ldquo;the Treaty of Kesselheim&rdquo; and &ldquo;the
painter Ottoline Crest&rdquo;: no spike, the probability spreads flat, because there is nothing to know. Asked outright
the model still confabulates a confident sentence &mdash;
&ldquo;<i>{intu['*Treaty of Kesselheim']['answer'][:96]}&hellip;</i>&rdquo; &mdash; but the flat distribution gives the
guess away. <b>Bottom row</b> is the whole point: the moon landing (1969) stays flat, as it must for a 1913 model, yet
the Bolsheviks seizing power in 1917 draws a tall spike it has no right to, and the model states it flatly:
&ldquo;<i>{intu['Bolsheviks seize power']['answer'][:96]}&hellip;</i>&rdquo;</p>

<h2>Turning the shape into a number</h2>
<p>We read the shape off one number: <b>P(true year)</b> &mdash; the probability the model places on the correct year.
Each event is scored over 17 candidate years (the true year&nbsp;±&nbsp;8). If the model is only guessing, every year
gets about 1&nbsp;in&nbsp;17&nbsp;=&nbsp;<b>{100/17:.1f}%</b> (&ldquo;chance&rdquo;); if it knows, almost all the mass
lands on the true year. To anchor what a given P means, we compare against two references, per model:</p>
<ul>
<li><b>Known reference</b> — {len(KNOWN)} famous <i>pre-1913</i> facts the model must know (Waterloo, Lincoln&rsquo;s
assassination, the Titanic). This is what a genuine memory looks like.</li>
<li><b>No-knowledge null</b> — {len(NULL)} <i>fictional</i> events with invented names (&ldquo;the Treaty of
Kesselheim&rdquo;, &ldquo;the painter Ottoline Crest&rdquo;) and an assigned year. The model cannot have learned a
date that never existed, so this is what pure ignorance looks like.</li>
</ul>

<h2>The benchmark: ~6% is chance, near-100% is knowledge</h2>
<img src="data:image/png;base64,{b64}">
<p class=cap>Figure 2. Two panels, one per model. Grey hump = the fictional events (no knowledge); green hump = known
pre-1913 facts. They barely overlap: fiction lands at chance (~{100/17:.1f}%), real memory lands near 100% (median
<b>{km['1913']:.0f}%</b> for 1913, <b>{km['1929']:.0f}%</b> for 1929). The dashed line is the leakage threshold &mdash;
one shared bar at {SHARED_THR:.0f}% (the 95th percentile of all {len(_pooled_null)} fictional events, same for both
models). This is the
answer to &ldquo;how do we know a low number is clean?&rdquo;: a made-up treaty reaches chance by definition, so
anything near {100/17:.1f}% is indistinguishable from a fiction, while real memory sits near 100%. Markers are the
post-cutoff test events (shape = category, colour = era): the ▼ leaks pile up under the green hump &mdash; the model
is as sure of them as of real facts &mdash; and the ● clean events sit under the grey hump at chance.</p>

<h2>What leaked, across the whole timeline</h2>
<p>P(true year) for each event, labelled per model. High probability on an event <em>before</em> a model&rsquo;s own
cutoff is legitimate memory (<span class=knows>knows&nbsp;✓</span>): the 1929 model knowing the 1914&ndash;1919 events
is correct, not contamination, and it validates the test. Contamination is high probability <span class=leaked>after</span>
the model&rsquo;s own cutoff; a post-cutoff event left near chance is <span class=clean>clean</span>.</p>
<table><tr><th>event</th><th>year</th><th>P(true) 1913</th><th>status · cut 1913</th><th>P(true) 1929</th><th>status · cut 1929</th></tr>{rows}</table>

<h2>Reading it</h2>
<ul>
<li><b>Both models are contaminated, selectively.</b> The 1913 model puts real weight on the two most canonical facts
just past its cutoff — the Great War (1914, {100*data['1913']['test']['First World War began']['p_true']:.0f}%) and the
Bolsheviks seizing power (1917, {100*data['1913']['test']['Bolsheviks seized power']['p_true']:.0f}%) — and, strikingly,
on far-future headline dates: the end of WWII (1945, {100*data['1913']['test']['Second World War ended']['p_true']:.0f}%)
and the Civil Rights Act (1964, {100*data['1913']['test']['Civil Rights Act']['p_true']:.0f}%). The 1929 model, as it
should, knows 1914&ndash;1919 cold, and then leaks past its own cutoff on the Second World War (1939,
{100*data['1929']['test']['Second World War began']['p_true']:.0f}%; ended 1945,
{100*data['1929']['test']['Second World War ended']['p_true']:.0f}%).</li>
<li><b>Confident on the date, empty on the detail.</b> Some leaks are held nearly as firmly as genuine facts &mdash; the
1913 model is {100*data['1913']['test']['Bolsheviks seized power']['p_true']:.0f}% sure the Bolsheviks rose in 1917,
against the ~{km['1913']:.0f}% it gives real pre-1913 facts &mdash; yet the contingent detail is missing: asked the fate
of the Tsar, the same model says <i>exiled</i>, not executed. It has memorised headline dates, not the history around
them.</li>
<li><b>It is patchy, not a smooth fade with time.</b> The 1913 model leaks the Civil Rights Act (1964) yet is clean on
D-Day, Pearl Harbor, the moon landing and Martin Luther King. Leakage tracks how <em>date-canonical</em> a fact is in
modern reference text (&ldquo;Civil Rights Act of 1964&rdquo;), not how far past the cutoff it lies.</li>
<li><b>The source is the pretraining, not the tuning.</b> Running the same test on the <em>base</em> models (no instruct
tuning) reproduces the leaks &mdash; the base 1913 model already puts real weight on the Great War and the Bolsheviks
&mdash; so the contamination lives in the pretraining corpus; the instruct tuning only sharpens it. Vector analysis is
in a separate note.</li>
</ul>
<p class=cap>Caveats: a P(true) only a little above chance ({100/17:.1f}%) is weak/ambiguous; the unambiguous leaks are
the ones near the green hump. The test is phrasing-sensitive &mdash; &ldquo;the Bolsheviks seized power in&rdquo; puts
{100*data['1913']['test']['Bolsheviks seized power']['p_true']:.0f}% on 1917, but the declarative &ldquo;the Tsar was
overthrown in&rdquo; does not &mdash; so a canonical phrasing is what triggers the memorised fact.</p>
"""
open(os.path.join(OUT, "date_spike_full.html"), "w", encoding="utf-8").write(html)

# ---------- console ----------
for tag in ["1913", "1929"]:
    kn = [v["p_true"] * 100 for v in data[tag]["known"].values()]; nu = [v["p_true"] * 100 for v in data[tag]["null"].values()]
    print(f"\n===== Ranke {tag}: KNOWN P(true) median {np.median(kn):.0f}%  |  NULL median {np.median(nu):.1f}% (chance {100/17:.1f}%)  threshold {thr[tag]:.0f}%")
    for lab, v in data[tag]["test"].items():
        flag = classify(tag, v)[1]
        print(f"   {lab:26}{v['true']}  P(true) {100*v['p_true']:>5.0f}%   {flag}")
print(f"\nwrote {FIG}")
