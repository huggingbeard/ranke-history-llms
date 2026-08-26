# -*- coding: utf-8 -*-
"""Build story.html: a chronological, didactic walk-through of the whole contamination
investigation -- what we did, what we found, why it forced the next step -- with the TikZ
flowchart inlined, example-prompt boxes, tables and KDEs computed from the actual result CSVs."""
import csv, html
import numpy as np

R = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\\"
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\story.html"
FLOW = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\flowchart.svg"
M0 = "Ranke-4B-1929"
rd = lambda f: list(csv.DictReader(open(R + f, encoding="utf-8")))
west, lm = rd("west_results.csv"), rd("lm_battery.csv")
dv1, dv2, dv3 = rd("dig_validated_results.csv"), rd("dig_validated_results_v2.csv"), rd("dig_validated_results_v3.csv")
esc = html.escape


def fpos(rows, key):
    n = len(rows); return (100 * sum(1 for r in rows if float(r[key]) > 0) / n, n) if n else (float("nan"), 0)


# ---------- KDE chart ----------
def kde(vals, grid, bw):
    d = np.zeros_like(grid)
    for v in vals:
        d += np.exp(-0.5 * ((grid - float(v)) / bw) ** 2)
    return d / d.max() if (len(vals) and d.max() > 0) else d


def chart(curves, xmin, xmax, ticks, zero, xlabel, bw, w=720, hh=250, zlabel="chance / indifferent (0)"):
    grid = np.linspace(xmin, xmax, 181)
    x0, x1, y0, ytop = 44, w - 14, hh - 30, 24
    X = lambda v: x0 + (x1 - x0) * (v - xmin) / (xmax - xmin)
    s = [f'<svg viewBox="0 0 {w} {hh}" width="100%">']
    for v, lab in ticks:
        x = X(v); z = abs(v - zero) < 1e-9
        s.append(f'<line x1="{x:.0f}" y1="{ytop}" x2="{x:.0f}" y2="{y0}" stroke="{"#c0392b" if z else "#eef1f6"}" stroke-dasharray="{"5 3" if z else "0"}"/>')
        s.append(f'<text x="{x:.0f}" y="{y0+14}" text-anchor="middle" font-size="10" fill="#9aa3ad">{lab}</text>')
    s.append(f'<text x="{X(zero):.0f}" y="{ytop-4}" text-anchor="middle" font-size="9.5" fill="#c0392b">{zlabel}</text>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{hh-2}" text-anchor="middle" font-size="10.5" fill="#5b6675">{esc(xlabel)}</text>')
    ly = ytop + 2
    for lab, vals, col in curves:
        if not vals:
            continue
        d = kde(vals, grid, bw)
        pts = " ".join(f"{X(grid[i]):.1f},{y0-(y0-ytop)*float(d[i]):.1f}" for i in range(len(grid)))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        s.append(f'<rect x="{x1-300}" y="{ly}" width="11" height="11" fill="{col}"/><text x="{x1-284}" y="{ly+10}" font-size="11" fill="#1a1d24">{esc(lab)} (n{len(vals)})</text>')
        ly += 17
    s.append("</svg>")
    return "".join(s)


def table(headers, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in headers)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


# ===== data: length bias (west, 1929 base, post) =====
w = [r for r in west if r["model"] == M0 and r["era"] == "post"]
for r in w:
    r["nt"], r["nf"] = int(r["ntok_t"]), int(r["ntok_f"])
lb_rows = []
for lab, sub in [("true SHORTER than twin", [r for r in w if r["nt"] < r["nf"]]),
                 ("SAME length", [r for r in w if r["nt"] == r["nf"]]),
                 ("true LONGER than twin", [r for r in w if r["nt"] > r["nf"]])]:
    a, n = fpos(sub, "gap"); su, _ = fpos(sub, "gap_sum"); fi, _ = fpos(sub, "gap_first")
    lb_rows.append([lab, n, f"{a:.0f}%", f"{su:.0f}%", f"{fi:.0f}%"])

# ===== data: by-category R/A/G (lm, 1929 base, matched) =====
L = [r for r in lm if r["model"] == M0 and r["matched"] == "1"]
CATS = [("USA", "politics"), ("UK", "politics"), ("France", "politics"), ("Germany", "politics"),
        ("Italy", "politics"), ("Spain", "politics"), ("Russia", "politics"), ("West", "culture"), ("West", "science")]
cat_rows = []
for c, d in CATS:
    pre = [r for r in L if (r["country"], r["domain"]) == (c, d) and r["era"] == "pre"]
    post = [r for r in L if (r["country"], r["domain"]) == (c, d) and r["era"] == "post"]
    pk = sum(int(r["correct"]) for r in pre); pn = len(pre)
    qk = sum(int(r["correct"]) for r in post); qn = len(post)
    cat_rows.append([f"{c} {d}", f"{(100*pk/pn if pn else 0):.0f}% (n{pn})", f"{(100*qk/qn if qn else 0):.0f}% (n{qn})"])

# ===== data: cued vs no-cue =====
def cc(rows, cat):
    rr = [r for r in rows if r["category"] == cat and r["era"] == "post" and r["model"] == M0]
    k = sum(int(r["correct"]) for r in rr); n = len(rr)
    return k, n, (100 * k / n if n else float("nan"))

cn_rows = []
for cat, cued_src, nc_src in [("US politics", dv1, dv3), ("Russia politics", dv1, dv2), ("Germany politics", dv1, dv2)]:
    ck, cn, cp = cc(cued_src, cat); nk, nn, npp = cc(nc_src, cat)
    cn_rows.append([cat.replace(" politics", ""), f"{cp:.0f}% ({ck}/{cn})", f"{npp:.0f}% ({nk}/{nn})",
                    "HOLDS — real" if (cat == "US politics") else "COLLAPSES — artifact"])

# ===== KDE money chart (1929 base gaps) =====
g = lambda rows, **f: [float(r["gap"]) for r in rows if r["model"] == M0 and all(r[k] == v for k, v in f.items())]
us_nc = g(dv3, era="post")
ru_nc = g(dv2, category="Russia politics")
pre_g = g(dv1, era="pre")
obama_g = [float(r["gap"]) for r in dv3 if r["model"] == M0 and r["event"] == "Obama / recent"]

# ===== example facets =====
def facets(rows, event):
    return [(r["true"], r["false"], float(r["gap"])) for r in rows if r["model"] == M0 and r["event"] == event]
atom = facets(dv3, "Atomic bombing of Japan")
brez_cued = [float(r["gap"]) for r in lm if r["model"] == M0 and r["true"] == "Brezhnev"]
brez_nc = [(r["true"], r["false"], float(r["gap"])) for r in dv2 if r["model"] == M0 and r["true"] == "Brezhnev"]

flow = open(FLOW, encoding="utf-8").read()
flow = flow[flow.find("<svg"):]   # strip xml/doctype prolog

# ---------- extra worked-example data ----------
import math as _m
def _binge(k, n, p=0.5):
    return sum(_m.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)) if n else float("nan")

# real first-token ranks (1929 base) -- READ the saved results (computed once in rank_examples.py)
RANKS = []
for _r in csv.DictReader(open(R + "rank_examples.csv", encoding="utf-8")):
    _n = int(_r["n_tokens"])
    _desc = f"{_n} token" + ("s &rarr; &lsquo;&nbsp;" + esc(_r["first_token"]) + "&rsquo;" if _n > 1 else "")
    RANKS.append((_r["tag"], _r["prompt"], _r["answer"], _desc, int(_r["rank"]), float(_r["p_first_pct"])))
# Step 4: a should-know item with low absolute prob but decisive gap
_pre = [r for r in west if r["model"] == M0 and r["era"] == "pre"]
for r in _pre:
    r["pt"] = float(r["p_true"]); r["g"] = float(r["gap"]); r["pf"] = r["pt"] * _m.exp(-r["g"])
_cand = sorted([r for r in _pre if 0.02 < r["pt"] < 0.40 and r["g"] > 2.5], key=lambda r: -r["g"])
LP = _cand[0] if _cand else None
LP_PF = ("&lt;0.01%" if (LP and LP["pf"] * 100 < 0.01) else (f'{LP["pf"]*100:.2f}%' if LP else ""))
# Step 9: a clean length-bias item where avg and sum DISAGREE (Facebook 1 tok vs Friendster 2 tok)
_bt = [r for r in west if r["model"] == M0 and r["true"] == "Facebook"]
BT = _bt[0] if _bt else None
# Step 11: atomic-bomb binomial (no-cue dig)
ATOM_K = sum(1 for (t, f, gp) in atom if gp > 0); ATOM_N = len(atom); ATOM_P = _binge(ATOM_K, ATOM_N)
# Step 10: US politics POST vs chance (no-cue) + PRE baseline
_usp = [r for r in dv3 if r["model"] == M0 and r["era"] == "post"]
US_K = sum(int(r["correct"]) for r in _usp); US_N = len(_usp); US_PCT = 100 * US_K / US_N
US_PV = _binge(US_K, US_N)   # one-sided binomial P(X>=k|0.5), consistent with the headline
# Step 8: Gelman worked pair (US politics vs France politics, lm 1929 matched post)
def _grp(c):
    rr = [r for r in lm if r["model"] == M0 and r["matched"] == "1" and r["country"] == c and r["domain"] == "politics" and r["era"] == "post"]
    k = sum(int(x["correct"]) for x in rr); n = len(rr); p = k / n if n else 0
    se = _m.sqrt(p * (1 - p) / n) if n else 0
    return k, n, p, se
GUS = _grp("USA"); GFR = _grp("France")
GD = GUS[2] - GFR[2]; GSE = _m.sqrt(GUS[3] ** 2 + GFR[3] ** 2); GZ = GD / GSE if GSE else 0; GPV = _m.erfc(abs(GZ) / _m.sqrt(2))
# Step 12: Falklands facets -- READ the saved results (falklands_detail.csv)
_FDESC = {"Falklands": "the islands &mdash; British since 1833", "Stanley": "Port Stanley &mdash; a town since 1845",
          "Belgrano": "named after a hero who died in 1820", "Thatcher": "her 1982 wartime role",
          "Galtieri": "the 1982 junta leader", "82": "the year of the war"}
FALK = []
for _r in csv.DictReader(open(R + "falklands_detail.csv", encoding="utf-8")):
    FALK.append((_r["answer"], _r["twin"], float(_r["gap"]), _FDESC.get(_r["answer"], _r["kind"]), 0 if _r["kind"] == "old-name" else 1))

CSS = """
body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1a1d24;line-height:1.62;background:#fff}
.wrap{max-width:900px;margin:0 auto;padding:46px 26px 100px}
h1{font-family:Georgia,serif;font-size:31px;margin:0 0 4px}.lede{color:#5b6675;font-size:17px;margin:0 0 8px}
h2{font-family:Georgia,serif;font-size:22px;margin:40px 0 6px;border-top:2px solid #e7eaef;padding-top:18px}
.step{color:#b4341f;font-weight:700;font-size:13px;letter-spacing:.06em;text-transform:uppercase}
h3{font-size:16px;margin:18px 0 4px}p{margin:9px 0}
.box{border:1px solid #e2e6ec;border-radius:10px;padding:13px 17px;margin:14px 0;background:#fafbfd}
.prompt{border-left:5px solid #1b3a6b;background:#f5f8fc}
.prompt code{background:#eaf0f8;padding:1px 5px;border-radius:4px;font-size:13.5px}
.found{border-left:5px solid #c0392b;background:#fdf4f3}.fix{border-left:5px solid #2e7d32;background:#f3f9f4}
.k{font-weight:700}table{border-collapse:collapse;width:100%;margin:12px 0;font-size:13.5px}
th,td{border:1px solid #e2e6ec;padding:6px 9px;text-align:left}th{background:#f0f3f8;font-weight:600}
td:first-child,th:first-child{white-space:nowrap}
figure{margin:16px 0;background:#fafbfd;border:1px solid #eef1f6;border-radius:10px;padding:14px}
figcaption{color:#5b6675;font-size:13px;margin-top:8px}
.flow{overflow-x:auto;border:1px solid #e2e6ec;border-radius:10px;padding:14px;background:#fff}
.flow svg{max-width:100%;height:auto;display:block;margin:0 auto}
.tag{display:inline-block;font-size:11px;font-weight:700;padding:2px 7px;border-radius:10px;margin-right:5px}
.red{background:#fdecea;color:#b4341f}.grn{background:#eaf5ec;color:#2e7d32}.gry{background:#eef1f6;color:#5b6675}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
.verdict{border:2px solid #1b3a6b;border-radius:12px;padding:16px 20px;background:#f5f8fc;margin:18px 0}
"""

H = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
     f"<title>Ranke contamination — the investigation</title><style>{CSS}</style></head><body><div class=wrap>"]
A = H.append

A("<h1>Did the time-locked models peek at the future?</h1>")
A("<p class=lede>A chronological account of how we measured hidden post-cutoff knowledge in the Ranke-4B "
  "models — every method, every wrong turn, and the two screens that turned a noisy mess into a clean answer.</p>")
A("<p>The Ranke-4B models are trained from scratch on text written only up to a cutoff year — <b>1900, 1913, "
  "or 1929</b> — so we can use them as &lsquo;synthetic respondents&rsquo; who genuinely inhabit the past. "
  "That only works if they don&rsquo;t secretly know what came later. This is the record of finding out.</p>")

A("<h2>The whole investigation on one page</h2>")
A('<div class=flow>' + flow + '</div>')
A('<figcaption>The full chain (TikZ). Red boxes are problems we discovered; green boxes are the fixes they forced. '
  'The rest of this page walks each step with the actual prompts, tables and densities.</figcaption>')

# 1
A('<h2><span class=step>Step 1 &middot; the question</span><br>What &ldquo;knowing the future&rdquo; means</h2>')
A("<p>Each model has a hard wall at its cutoff. A clean model should answer pre-cutoff questions well and be at "
  "<b>chance</b> on anything after. Contamination = it does better than chance on the post-cutoff side. The whole "
  "difficulty is measuring that without fooling ourselves — which, as you&rsquo;ll see, took several tries.</p>")

# 2 first attempts
A('<h2><span class=step>Step 2 &middot; first attempts</span><br>Beam search, twins, and a tiny-gap problem</h2>')
A("<p>The obvious approach is to let the model <i>generate</i> and search for the answer &mdash; beam search, then "
  "matched &lsquo;twin&rsquo; and even &lsquo;quadruple&rsquo; answer sets. Two problems surfaced fast. First, the model "
  "occasionally (about <b>3%</b> of the time) prefers an answer it shouldn&rsquo;t know &mdash; suggestive, but not proof. "
  "Second, and worse, the log-probability differences were <b>ultra-small</b>: even when the model clearly knew, its "
  "margin over the wrong answer was a sliver. We needed a sharper, cheaper instrument.</p>")

# 3 wrong turn: rank idea + Daniel
A('<h2><span class=step>Step 3 &middot; a wrong turn (mine)</span><br>&ldquo;Just rank the true token&rdquo; &mdash; and the RA&rsquo;s objection</h2>')
A("<p>My proposed fix: forget probabilities, look at the <b>rank</b> of the true answer&rsquo;s token in the model&rsquo;s "
  "next-token distribution. If &lsquo;Kennedy&rsquo; is the model&rsquo;s #2 most likely token out of ~150,000, that&rsquo;s "
  "knowledge; if it&rsquo;s #40,000, it isn&rsquo;t. I wrote it up as <i>the</i> method.</p>")
A('<div class="box found"><span class="tag red">RA objects (Daniel)</span>Two sharp objections. <b>(1)</b> Ranking '
  "can&rsquo;t work &mdash; scoring a multi-token answer this way is |vocab|<sup>|tokens|</sup> combinations, "
  "computationally impossible, just beam search in disguise. <b>(2)</b> The model&rsquo;s probabilities are <b>low even on "
  "facts it surely knows</b> &mdash; so you can&rsquo;t read knowledge off a softmax at all (plus tokenization, "
  "conditioning, and weird token preferences will bite you).</div>")
A("<p><b>Who was right?</b> On the arithmetic, me: the rank of a <i>single</i> token is <b>one forward pass</b> &mdash; you "
  "count how many tokens score higher than the target (rank = #{p&nbsp;&gt;&nbsp;p<sub>true</sub>}&nbsp;+&nbsp;1), not "
  "|vocab|<sup>|tokens|</sup>. So the &lsquo;impossible&rsquo; charge was a misread. But Daniel was right where it counted: "
  "ranking the <b>first token</b> of a multi-token name compares a <b>subword</b> &mdash; &ldquo;&nbsp;Ken&rdquo;, not "
  "&ldquo;Kennedy&rdquo; &mdash; which is meaningless. The rank idea was the wrong instrument, and we dropped it. "
  "(That exact subword trap comes back to bite us again at Step&nbsp;9.)</p>")

# 4 low probs not a problem
A("<p><b>The actual ranks</b> &mdash; one forward pass each (1929 model; computed once and saved to "
  "<span class=mono>rank_examples.csv</span>, then read back):</p>")
A(table(["type", "prompt ends &hellip;", "answer", "tokenizes", "rank", "p(1st&nbsp;token)"],
        [[t, esc(p), esc(a), d, str(rk), f"{pp:.1f}%"] for (t, p, a, d, rk, pp) in RANKS]))
A("<p>Einstein and Darwin are <b>rank&nbsp;1</b> &mdash; the model nails should-know facts. <b>Truman is rank&nbsp;2</b> "
  "(it plainly knows a post-cutoff president) yet at only <b>20%</b> probability &mdash; Daniel&rsquo;s &lsquo;low "
  "probs&rsquo; point, live. And <b>Brezhnev</b> splits into &lsquo;&nbsp;Bre&rsquo;/&lsquo;zh&rsquo;/&lsquo;nev&rsquo;, "
  "so a first-token rank (3) is really the rank of &lsquo;&nbsp;Bre&rsquo; &mdash; which could be Brexit or Brussels. "
  "The sub-word trap, concretely: ranking is feasible, but the wrong instrument.</p>")

A('<h2><span class=step>Step 4 &middot; the key realization</span><br>Why low probabilities are <i>not</i> a problem</h2>')
A("<p>Daniel&rsquo;s strongest point &mdash; probabilities are low even on known facts &mdash; sounds fatal, but it "
  "dissolves once you stop measuring the wrong thing. We <b>never</b> compare a probability to an absolute threshold. We "
  "compare two completions of the <b>same prompt</b> &mdash; true vs a matched false twin &mdash; and read only the "
  "<b>difference</b>. Whatever makes absolute probabilities low (a long tail over 150k tokens, the prompt&rsquo;s framing, "
  "tokenization) hits both twins equally and <b>cancels</b>. A true answer can sit at an absolute 4% and still beat its "
  "twin&rsquo;s 0.4% decisively. The absolute number is irrelevant; the <b>relative gap</b> is everything.</p>")
A('<div class="box fix"><span class="tag grn">resolution</span>Keep teacher-forcing (no generation, no search), but score '
  "the <b>relative gap between matched twins</b> &mdash; not absolute confidence, not token rank. That instrument "
  "survived; everything after this is about making the twins <i>fair</i>.</div>")

# 5 the instrument (worked example)
if LP:
    A(f'<div class="box prompt"><b>Concrete (a pre-cutoff fact the model truly knows).</b> It backs the true answer '
      f'<b>{esc(LP["true"])}</b> over the false twin <b>{esc(LP["false"])}</b> &mdash; yet in <i>absolute</i> terms '
      f'p(true) is only <b>{LP["pt"]*100:.1f}%</b> and p(false) is <b>{LP_PF}</b>. Both small; but the gap '
      f'is a decisive <b>+{LP["g"]:.1f}</b>. The relative comparison carries the signal &mdash; the same reason '
      f'Truman counts as &lsquo;known&rsquo; at rank&nbsp;2 despite only 20% probability.</div>')

A('<h2><span class=step>Step 5 &middot; the instrument</span><br>A worked example of the gap</h2>')
A("<p>We never ask the model to <i>generate</i> an answer (generation invites lucky guesses and is hard to score). "
  "Instead we <b>teacher-force</b> a minimal pair: the same prompt followed by a true completion and a false "
  "&lsquo;twin&rsquo;, and read the log-probability the model assigns to each. The signal is the <b>gap</b>:</p>")
A('<div class="box prompt"><div class=mono>prompt: &ldquo;The U.S. president who ordered the atomic bombing of Japan was&rdquo;<br>'
  'true = &ldquo;Truman&rdquo;&nbsp;&nbsp;|&nbsp;&nbsp;false = &ldquo;Dewey&rdquo;</div>'
  '<p style="margin:8px 0 0">gap = log&nbsp;p(<b>Truman</b>) &minus; log&nbsp;p(<b>Dewey</b>). '
  'gap&nbsp;&gt;&nbsp;0 means the model prefers the truth. Over many items, <b>%&nbsp;with gap&gt;0</b> is our score; '
  '50% is a coin-flip.</p></div>')
A("<p>This is the instrument that survived every later fix: a <b>relative gap on a matched pair</b>. Over a whole battery "
  "the fraction with gap&gt;0 is the score (50% = chance). Everything from here is about making the pair <i>fair</i>.</p>")

# 6 the category principle (early, before any number means anything)
A('<h2><span class=step>Step 6 &middot; the principle that governs everything</span><br>Never the individual question &mdash; always the category</h2>')
A("<p>Before any number meant anything we had to settle what counts as evidence. A single confident answer is "
  "<b>not</b> contamination &mdash; it can be a lucky guess, an easy twin, or one ubiquitous fact that floats around in "
  "any corpus. Knowledge is a property of a <b>category</b>, not an item: if the model truly absorbed, say, modern US "
  "politics, it beats chance <i>across the whole set</i> of post-cutoff US-politics questions, not on one of them.</p>")
A('<div class="box fix"><span class="tag grn">principle</span>Judge a <b>category</b>, never a single question. This also '
  "gives a built-in placebo: every probe bias (tokenization, phrasing, the model&rsquo;s quirks) hits <i>all</i> categories "
  "alike, so a genuine leak shows up as <b>one category standing out from its neighbours</b> &mdash; not as a lone high "
  "score. Everything that follows &mdash; aggregate %-correct, red/amber/green, the multi-facet dig &mdash; is a way of "
  "honouring this.</div>")

# 7 first cut
A('<div class="box"><b>Why one question lies.</b> On the (post-cutoff) Falklands War the 1929 model prefers '
  '<b>&ldquo;Port Stanley&rdquo;</b> over its twin by a confident <b>+3.4</b> &mdash; that single item looks like '
  'contamination. But &ldquo;Stanley&rdquo; is merely a pre-1929 town name, and across the <i>whole</i> Falklands '
  'cluster the model is at chance. One item screamed &lsquo;contaminated&rsquo;; the category said &lsquo;clean&rsquo; '
  '&mdash; and the category was right (we confirm it at Step&nbsp;12). Hence: judge categories, never items.</div>')

A('<h2><span class=step>Step 7 &middot; first cut</span><br>The cliff, and an alarming number</h2>')
A("<p>Lay the should-know (pre-cutoff) score next to the should-not-know (post-cutoff) score. A clean model shows a "
  "<b>cliff</b>: high on the left, chance on the right. The first broad battery showed a big cliff everywhere — except "
  "<b>US politics</b>, which came in around <span class=k>89%</span> on post-cutoff items. Alarming. But also, as it "
  "turned out, mostly an artifact. Two of them.</p>")
A('<div class="box found"><span class="tag red">found</span>US politics ~89% post-cutoff on the first battery. '
  "Too good — and built on a small sample with a biased metric (next two steps).</div>")

# 4
A('<h2><span class=step>Step 8 &middot; Gelman &amp; Stern</span><br>Is the difference even significant?</h2>')
A("<p>Gelman &amp; Stern (2006): <i>the difference between &lsquo;significant&rsquo; and &lsquo;not significant&rsquo; is "
  "not itself statistically significant.</i> You can&rsquo;t eyeball two numbers and call them different — you test the "
  "difference, with SE(diff)=&radic;(SE&sup2;+SE&sup2;). Applied here, the headline cluster rested on a handful of items "
  "(n as low as <b>4</b> for some categories). At that N, 89% vs 50% isn&rsquo;t the slam-dunk it looks like.</p>")
A('<div class="box found"><span class="tag red">found</span>The scary gaps were under-powered. We needed both a '
  "trustworthy metric and far more items per category.</div>")

# 5
A(f'<div class="box"><b>Worked example.</b> Two clusters that look different: US-politics <b>{GUS[2]*100:.0f}%</b> '
  f'(n={GUS[1]}) vs France-politics <b>{GFR[2]*100:.0f}%</b> (n={GFR[1]}). Difference {GD*100:.0f}&nbsp;pts; '
  f'SE(diff)=&radic;(SE&sup2;<sub>US</sub>+SE&sup2;<sub>Fr</sub>)={GSE*100:.0f}; z={GZ:.2f}, p={GPV:.2f}. '
  f'At these tiny n the gap is <b>not</b> statistically significant &mdash; you cannot call the clusters different '
  f'by eye. That is exactly why the scary 89% needed re-testing, not celebrating.</div>')

A('<h2><span class=step>Step 9 &middot; the metric was lying</span><br>Token-length bias</h2>')
A("<p>The score depends on a subtlety almost everyone gets wrong. Our first metric was the <b>average per-token</b> "
  "log-probability. That is <b>biased by how many tokens the answer has</b>. &ldquo;Beatles&rdquo; is one token; "
  "&ldquo;Searchers&rdquo; is &ldquo;Search&rdquo;+&ldquo;ers&rdquo; whose second piece is almost certain — so the "
  "<i>average</i> rewards the wrong band. We checked it directly:</p>")
A(table(["twin lengths", "n", "avg-logp %", "total (sum) %", "first-token %"], lb_rows))
A("<p>They <b>only agree when the twins are the same length</b> (~57%). When the true answer is longer, average-logp "
  "balloons to ~90% no matter what; when shorter, total-logp does. First-token doesn&rsquo;t fix it either — it compares "
  "<i>subword fragments</i>: &ldquo;&nbsp;Ken&rdquo; vs &ldquo;&nbsp;Hum&rdquo;, not &ldquo;Kennedy&rdquo; vs "
  "&ldquo;Humphrey&rdquo; &mdash; which is exactly the tokenization trap Daniel flagged at the very start (Step&nbsp;3), "
  "now made concrete.</p>")
A('<div class="box fix"><span class="tag grn">fix</span>Score only on <b>length-matched twins</b> (same token count), '
  "and read the <b>whole answer</b>. Every battery item now carries several candidate twins so a length-match can be picked. "
  "This is the only bias-free comparison.</div>")

# 6
if BT:
    A(f'<div class="box"><b>One item, opposite verdicts.</b> &ldquo;The social network Zuckerberg launched in 2004 was&nbsp;___&rdquo;, '
      f'true <b>Facebook</b> (1 token) vs <b>Friendster</b> (2 tokens): the <b>average</b>-logp gap = '
      f'<b>{float(BT["gap"]):+.2f}</b> &mdash; it picks <i>Friendster</i>! &mdash; while the <b>total</b>-logp gap = '
      f'<b>{float(BT["gap_sum"]):+.2f}</b> picks Facebook (first-token, {float(BT["gap_first"]):+.2f}, is also wrong). '
      f'Opposite answers from token length alone. Neither average nor total is trustworthy &mdash; only <b>length-matched</b> twins are.</div>')

A('<h2><span class=step>Step 10 &middot; by category, with a rule</span><br>Red / Amber / Green</h2>')
A("<p>With the corrected metric we scored a much larger battery and judged each category by two tests, not one: "
  "<b>(A)</b> is post-cutoff above 50%? <b>(B)</b> is it different from pre-cutoff? A flat cliff is only damning if "
  "post-cutoff is <i>also</i> clearly above chance — otherwise it&rsquo;s just noise. Pre- vs post-cutoff, length-matched, "
  "1929 model:</p>")
A(table(["category", "PRE (should-know)", "POST (should-not-know)"], cat_rows))
A("<p>Post-cutoff hugs chance everywhere except US politics — but several cells were still thin. Which raised the real "
  "question: when a single post-cutoff answer <i>does</i> look confident, is that knowledge, or luck?</p>")

# 7
A(f'<div class="box"><b>Worked test (US politics, no first-name cues, 1929 model).</b> Should-not-know score = '
  f'<b>{US_PCT:.0f}%</b> ({US_K}/{US_N} facets). Test A &mdash; above chance? One-sided binomial P(X&ge;{US_K}) = '
  f'<b>{US_PV:.3f}</b> &lt; .05: yes &#10003;. So this category is <b>not</b> green &mdash; the model knows post-cutoff '
  f'US politics even with every name cue stripped.</div>')

A('<h2><span class=step>Step 11 &middot; digging deeper</span><br>What knowledge actually is</h2>')
A("<p>Knowing a fact isn&rsquo;t one lucky Trivial-Pursuit hit — it&rsquo;s a <b>coherent cluster</b>. If the model truly "
  "absorbed Hiroshima from leaked text, it knows the president <i>and</i> the codenames <i>and</i> the aircraft. So when a "
  "single item looks contaminated, we <b>dig</b>: probe the same event from many angles. Under the null (no knowledge) each "
  "facet is a coin-flip, so the joint score follows a binomial — getting most of a cluster right is overwhelming evidence.</p>")
if atom:
    rows = [[esc(t), esc(f), f"{gp:+.2f}", ("✓" if gp > 0 else "✗")] for (t, f, gp) in atom]
    A('<div class="box prompt"><b>Example dig — the atomic bombing (1929 model, no name cues):</b>'
      + table(["true", "vs twin", "gap", ""], rows) + "</div>")
A("<p>Two controls keep us honest: a <b>pre-cutoff</b> event (Waterloo, 1815) the model <i>should</i> ace — it does, "
  "confirming the dig detects real knowledge; and a <b>clean post-cutoff</b> event it shouldn&rsquo;t know. The clean "
  "control is where the next trap sprang.</p>")

# 8
A(f'<div class="box"><b>The joint (binomial) test.</b> The model got <b>{ATOM_K}/{ATOM_N}</b> of the screened atomic-bomb '
  f'facets right. If it were guessing, the chance of doing that well is P(X&ge;{ATOM_K}) = <b>{ATOM_P:.3f}</b>. '
  f'At n={ATOM_N} that is suggestive, not yet decisive (it takes ~8 facets, or pooling many events, to cross .05) &mdash; '
  f'but a whole coherent cluster is already far stronger evidence than any single lucky hit. This is how depth beats one item.</div>')

A('<h2><span class=step>Step 12 &middot; &ldquo;old words&rdquo;</span><br>Recognising a name is not knowing the event</h2>')
A("<p>The Falklands (1982) &lsquo;clean control&rsquo; scored suspiciously well — until we read it. The hits were "
  "<b>&ldquo;Falklands&rdquo;</b> (British since 1833), <b>&ldquo;Port Stanley&rdquo;</b> (a town since 1845), and "
  "<b>&ldquo;General Belgrano&rdquo;</b> (named after a hero who died in 1820). All <b>pre-cutoff names</b>. The genuinely "
  "1982-specific facts — Thatcher&rsquo;s wartime role, Galtieri, the year — were at chance.</p>")
A('<div class="box fix"><span class="tag grn">fix</span>Drop any facet whose answer is a person/place/thing that '
  "existed before the cutoff. Keep only what <b>requires</b> post-cutoff text: coined codenames, post-cutoff-emergent "
  "people, specific years.</div>")

# 9
A("<p><b>The Falklands War, facet by facet</b> (1929 model; from <span class=mono>falklands_detail.csv</span>):</p>")
A(table(["the model picks", "gap", "what the answer really is"],
        [[esc(a) + " vs " + esc(tw), f"{g:+.2f}", d] for (a, tw, g, d, grp) in FALK]))
A("<p>The three confident hits are all <b>pre-1929 names</b>; the three genuinely-1982 facts sit at chance "
  "(p(true)=0%). The model &lsquo;passes&rsquo; only by recognising old words &mdash; not by knowing the war. "
  "<b>Fix:</b> keep only facets that <i>require</i> the post-cutoff event.</p>")

A('<h2><span class=step>Step 13 &middot; the big one</span><br>No first names</h2>')
A("<p>The deepest trap: putting the answer&rsquo;s <b>first name</b> in the prompt. &ldquo;The Soviet leader of the "
  "stagnation era from 1964 was Leonid ___&rdquo; — the model never has to know the <i>role</i>; &ldquo;Leonid&rdquo; "
  "cues &ldquo;Brezhnev&rdquo; by name-completion. That tests <b>fame, not knowledge</b>. Watch what the cue is worth:</p>")
A('<div class="box prompt"><div class=mono>'
  'WITH cue:&nbsp;&nbsp;&ldquo;&hellip; was Leonid ___&rdquo; &rarr; Brezhnev'
  + (f"&nbsp;&nbsp;gap = <b>+{brez_cued[0]:.1f}</b>" if brez_cued else "")
  + '<br>NO cue:&nbsp;&nbsp;&ldquo;The Soviet leader of the stagnation era from 1964 was ___&rdquo; &rarr; Brezhnev'
  + (f"&nbsp;&nbsp;gap = <b>{brez_nc[0][2]:+.1f}</b>" if brez_nc else "")
  + "</div><p style='margin:8px 0 0'>Same fact, drop the first name, and the giant gap evaporates — the model had "
  "<i>seen the name</i> but doesn&rsquo;t <i>know the role</i>.</p></div>")
A('<div class="box fix"><span class="tag grn">fix</span>The prompt gives the <b>descriptive fact only</b>; the model must '
  "produce the answer from it. Coined terms and years are best of all — nothing to complete.</div>")

# 10
A('<h2><span class=step>Step 14 &middot; the gate</span><br>DeepSeek-V4-Pro screens every question</h2>')
A("<p>Enforcing all of this by hand is error-prone, so every candidate facet is passed through "
  "<b>DeepSeek-V4-Pro</b> (via OpenRouter) with a strict rubric: is the true answer correct? does it <i>require</i> "
  "post-cutoff knowledge? any pre-cutoff &lsquo;old word&rsquo;? any first-name/answer cue? are the twins clearly wrong "
  "and length-comparable? It is ruthless — and it caught exactly the things we&rsquo;d been fooled by:</p>")
A('<div class="box found mono" style="font-size:12.5px">'
  'DROP &ldquo;Winston&nbsp;&rarr;&nbsp;Churchill&rdquo; :: famous pre-1929, tests fame.<br>'
  'DROP &ldquo;Dmitri&nbsp;&rarr;&nbsp;Shostakovich&rdquo; :: known since 1925.<br>'
  'DROP &ldquo;Jean-Paul&nbsp;&rarr;&nbsp;Sartre&rdquo;, &ldquo;George&nbsp;&rarr;&nbsp;Orwell&rdquo; :: pre-1929.<br>'
  'DROP &ldquo;&hellip; Warsaw&nbsp;&rarr;&nbsp;Pact&rdquo; :: prompt cues the answer.<br>'
  'DROP &ldquo;Falklands&rdquo;, &ldquo;Auschwitz&rdquo;, &ldquo;Stalingrad&rdquo; :: pre-1929 places.</div>')

# 11
A('<h2><span class=step>Step 15 &middot; the decisive test</span><br>Cued vs no-cue</h2>')
A("<p>Now the clean experiment. Take the confident-looking suspects and re-run them <b>with</b> and <b>without</b> the "
  "name cue. If the score survives the strip, it&rsquo;s knowledge; if it collapses to chance, it was name-completion.</p>")
A(table(["category", "cued %", "no-cue % (powered, screened)", "verdict"], cn_rows))
A("<p>US politics barely moves — real, structured knowledge. Russia and Germany fall to a coin-flip — they were famous "
  "names all along. The densities make it visible (1929 model, per-facet gaps):</p>")
A('<figure>' + chart(
    [("pre-cutoff control (Waterloo &c.)", pre_g, "#1b3a6b"),
     ("US politics, no cue", us_nc, "#c0392b"),
     ("Russia/Soviet, no cue", ru_nc, "#e0a39e"),
     ("recent US (Obama), no cue", obama_g, "#9aa3ad")],
    -8, 8, [(-8, "-8"), (-4, "-4"), (0, "0"), (4, "+4"), (8, "+8")], 0,
    "gap = log p(true) - log p(false)   (0 = coin-flip)", 0.85) +
  "<figcaption>Pre-cutoff control sits far right (genuine knowledge — the dig works). <b>US politics</b> sits right of "
  "zero too: real post-cutoff knowledge. <b>Russia/Soviet</b> collapses onto zero once the name cue is gone. "
  "<b>Recent US (Obama)</b> sits left of zero — the model has no idea.</figcaption></figure>")

# 12
A('<h2><span class=step>Step 16 &middot; verdict</span><br>What the model actually leaked</h2>')
A('<div class=verdict><p style="margin:0 0 6px"><b>The only genuine contamination is mid-20th-century US political '
  'history (&approx;1933&ndash;1974).</b> It survives the strictest test we can build — no name cues, post-cutoff-only '
  'facts, DeepSeek-screened, length-matched: <b>68%, p=0.044</b>, with real structured knowledge (Watergate&rarr;Nixon, '
  'New&nbsp;Deal&rarr;Truman, the bomb codenames).</p>'
  '<p style="margin:6px 0 0">Everything else is at chance once name-completion artifacts are removed: UK, France, '
  'Germany, Russia, science, culture — <i>and</i> recent US (Obama: gap &minus;3.7). The dramatic cross-category '
  '&lsquo;hits&rsquo; (Churchill +15.9, Brezhnev +19, Shostakovich +11.5) were all famous-name completions, not knowledge.</p></div>')
A("<p><b>Why it matters.</b> For the synthetic-respondent use, the models are far cleaner than the first pass suggested: "
  "their forward-looking knowledge is confined to a specific, well-documented slice of American politics — which can be "
  "controlled for — rather than a diffuse leak across the 20th century. And the method itself is the deliverable: "
  "<b>length-matched twins, multi-facet digging, no name cues, an LLM screen, and Gelman&ndash;Stern discipline</b> are "
  "what separate a real signal from five different ways of fooling yourself.</p>")
A("</div></body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote", OUT, "(", len("\n".join(H)), "bytes )")
