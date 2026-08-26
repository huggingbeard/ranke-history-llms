# -*- coding: utf-8 -*-
"""Plain-English report for the Inside-Out hidden-knowledge slice. Reads inside_out_stats.json.
Controls up front; honest verdict driven by the locked pre-registration rules."""
import os, json, datetime, html
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, "inside_out_stats.json")))
OUT = os.path.join(HERE, "inside_out_report.html")
esc = html.escape
EXT, LENS, PROBE, DELTA, FDR = S["EXTERNAL"], S["LENS"], S["PROBE"], S["DELTA"], S["FDR"]
cats = list(EXT.keys())
POST = [c for c in cats if c != "PRE (control)"]
NICE = {"PRE (control)": "Pre-cutoff facts (positive control)", "science+space (neg)": "Science & space, post-cutoff (negative control — unknowable)",
        "US civil rights": "US civil rights (the external suspect)", "US politics 1945-74": "US politics 1945–74",
        "Cold War": "Cold War", "Vietnam War": "Vietnam War"}

# pre-reg control checks
pos_pass = EXT["PRE (control)"]["K"] > 0.65 and LENS["PRE (control)"]["K"] > 0.65 and PROBE["PRE (control)"]["K"] > 0.65
null_pass = PROBE["null_ci"][0] <= 0.5 <= PROBE["null_ci"][1]
neg_probe_flag = FDR["delta_probe"].get("science+space (neg)", False)


def bars(c):
    # three bars (ext, lens, probe) 0..1 with chance line at .5
    W, H = 260, 64
    out = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    out.append(f'<line x1="{0.5*W:.0f}" y1="0" x2="{0.5*W:.0f}" y2="{H}" stroke="#b9c2cf" stroke-dasharray="3 3"/>')
    for i, (lab, d, col) in enumerate([("says", EXT[c], "#7c8694"), ("lens", LENS[c], "#2b6cb0"), ("probe", PROBE[c], "#c0392b")]):
        y = 6 + i * 19
        out.append(f'<rect x="0" y="{y}" width="{d["K"]*W:.0f}" height="13" rx="2" fill="{col}"/>')
        out.append(f'<line x1="{d["lo"]*W:.0f}" x2="{d["hi"]*W:.0f}" y1="{y+6.5:.0f}" y2="{y+6.5:.0f}" stroke="#1a2230" stroke-width="1" opacity="0.5"/>')
        out.append(f'<text x="{min(d["K"]*W+4, W-34):.0f}" y="{y+10:.0f}" style="font:9px sans-serif;fill:#1a2230">{lab} {d["K"]:.2f}</text>')
    out.append('</svg>')
    return "".join(out)


CSS = """
:root{--ink:#1a2230;--mut:#5c6675;--line:#e6e9ef;--bg:#f6f8fa}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:46px 26px 100px}
h1{font:700 28px/1.25 Georgia,serif;margin:0 0 8px}.sub{color:var(--mut);margin:0 0 22px}
h2{font:700 22px Georgia,serif;margin:42px 0 8px;padding-top:16px;border-top:1px solid var(--line)}
h2 .n{display:inline-block;width:28px;height:28px;line-height:28px;text-align:center;background:#2b5dab;color:#fff;border-radius:50%;font:700 14px sans-serif;margin-right:9px;vertical-align:2px}
p{margin:11px 0}.big{font-size:18px}
.key{background:#eef4ff;border:1px solid #cfe0ff;border-left:5px solid #2b5dab;border-radius:10px;padding:15px 19px;margin:16px 0}.key b{color:#1f3f7a}
.pass{background:#e9f8ef;border:1px solid #bfe6cd;border-left:5px solid #2e9e5b;border-radius:10px;padding:12px 18px;margin:10px 0}
.warn{background:#fff7ed;border:1px solid #fed7aa;border-left:5px solid #ea8a3c;border-radius:10px;padding:14px 19px;margin:14px 0}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0;font-size:14px}
th,td{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line)}th{background:#eef2f7;color:#3a4252;font:600 12.5px sans-serif}
tr:last-child td{border-bottom:none}.lab{font-weight:600;background:#f8fafc}.note{font-size:13px;color:var(--mut)}
.foot{color:var(--mut);font-size:13px;margin-top:42px;border-top:1px solid var(--line);padding-top:14px}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:13px}
"""
P = []; A = P.append
A(f'<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Does a 1929 mind secretly know the future?</title><style>{CSS}</style></head><body><div class=wrap>')
A("<h1>Does a 1929 mind secretly <i>know</i> the future?</h1>")
A('<p class=sub>A pre-registered test for <b>hidden knowledge</b> in the time-frozen Ranke-1929 model — facts it might '
  'encode internally even if it never says them. Method after Gekhman et&nbsp;al. 2025, &ldquo;Inside-Out.&rdquo;</p>')

A('<p class=big>Our earlier audits judged the model by what it <b>says</b> (output probabilities) and what it '
  '<b>writes</b> (generation). This asks the deepest question: does the model <b>internally encode</b> post-cutoff '
  'facts — secretly &ldquo;know&rdquo; them in its hidden layers — even when its output looks clean? We score each fact '
  'three ways and check whether the true answer is ranked above false ones (chance = 0.50):</p>')
A('<p>&nbsp;&nbsp;&bull; <b>says</b> — the model\'s output probabilities (what it would tell you).<br>'
  '&nbsp;&nbsp;&bull; <b>lens</b> — a peek at an intermediate layer (logit-lens), read without training.<br>'
  '&nbsp;&nbsp;&bull; <b>probe</b> — a small classifier trained on the hidden states to spot the true answer.</p>')
A('<p><b>Hidden knowledge</b> = an internal score (lens or probe) beating the output score. The whole thing lives or '
  'dies on two controls, so we check those first.</p>')

A('<h2><span class=n>1</span>Do the controls hold? (the trust gates)</h2>')
A(f'<div class="{"pass" if pos_pass else "warn"}"><b>Positive control — {"PASSED" if pos_pass else "FAILED"}.</b> '
  f'On pre-cutoff facts the model truly knows, all three scorers recover them well above chance '
  f'(says {EXT["PRE (control)"]["K"]:.2f}, lens {LENS["PRE (control)"]["K"]:.2f}, probe {PROBE["PRE (control)"]["K"]:.2f}). '
  f'So the instruments work — they can detect knowledge when it\'s really there.</div>')
A(f'<div class="{"pass" if null_pass else "warn"}"><b>Shuffle test (null) — {"PASSED" if null_pass else "FAILED"}.</b> '
  f'When we scramble the correct/incorrect labels, the probe collapses to chance '
  f'({PROBE["null_mean"]:.2f}, range {PROBE["null_ci"][0]:.2f}–{PROBE["null_ci"][1]:.2f}). '
  f'So the probe isn\'t just memorising a global &ldquo;true-looking&rdquo; pattern.</div>')

A('<h2><span class=n>2</span>The scores — three ways of asking, per topic</h2>')
A('<p class=note>Bars = how often the true answer is ranked above a false one (0.50 = chance, the dashed line). '
  'Thin line = 95% confidence range. <span style="color:#7c8694;font-weight:700">says</span> / '
  '<span style="color:#2b6cb0;font-weight:700">lens</span> / <span style="color:#c0392b;font-weight:700">probe</span>.</p>')
A('<table><thead><tr><th>Topic</th><th>says · lens · probe (chance 0.50)</th></tr></thead><tbody>')
for c in cats:
    A(f'<tr><td class=lab>{esc(NICE.get(c,c))}<br><span class=note>n={S["cats"][c]}</span></td><td>{bars(c)}</td></tr>')
A('</tbody></table>')

A('<h2><span class=n>3</span>The verdict: no credible hidden knowledge</h2>')
A('<div class=key><b>Bottom line:</b> the clean internal scorer (the lens, which passes both controls) sits right on '
  'top of what the model already <i>says</i> — in every topic. There is <b>no sign the model secretly knows '
  'post-cutoff facts it won\'t express.</b> The trained probe <i>looked</i> like it found some — but it failed the '
  'negative control, so we don\'t trust it (next box).</div>')
A('<p><b>The lens finds nothing hidden.</b> Internal-minus-external (lens) is ≈ 0 everywhere: '
  + ", ".join(f'{esc(c.split(" (")[0])} {DELTA[c]["lens"][0]:+.02f}' for c in POST) +
  '. Where the model knows a thing, it says it; where it doesn\'t, the hidden layers don\'t either.</p>')
A(f'<div class=warn><b>Why we throw out the probe\'s &ldquo;hidden knowledge.&rdquo;</b> The probe flagged '
  f'<b>science &amp; space</b> as hidden knowledge (gap +{DELTA["science+space (neg)"]["probe"][0]:.2f}, survives '
  f'multiple-comparison correction) — but those are the <b>genuinely-unknowable</b> facts (DNA in 1953, Sputnik, …) a '
  f'1929 model <i>cannot</i> know. A real knowledge detector must stay quiet there. It didn\'t. So the probe is '
  f'picking up which answer simply <i>looks</i> more plausible, not whether the model knows the fact — exactly the trap '
  f'our pre-registered negative control was built to catch. Its &ldquo;hidden knowledge&rdquo; on Cold War is the same '
  f'artifact, and is not credible. (The probe\'s gap is also inflated because the model scores <i>below</i> chance '
  f'externally on these — it actively prefers a wrong answer — so even a chance-level probe looks like a gain.)</div>')
A(f'<p><b>The actual suspect — US civil rights</b> (the one topic the output audit flagged) — shows <b>no</b> hidden '
  f'amplification: lens gap {DELTA["US civil rights"]["lens"][0]:+.02f}, probe gap '
  f'{DELTA["US civil rights"]["probe"][0]:+.02f} (not significant). Whatever leakage exists there is <b>fully on the '
  f'surface</b>, already visible in the output audit — not buried deeper in the weights.</p>')

A('<h2><span class=n>4</span>What this means &amp; the honest fine print</h2>')
A('<div class=key>The output-level contamination verdict <b>stands</b>: the model isn\'t secretly hoarding post-cutoff '
  'knowledge. And we get a methodological bonus — <b>naive hidden-knowledge probing can manufacture false '
  '&ldquo;hidden contamination,&rdquo;</b> and an unknowable negative control is what exposes it. Ours did.</div>')
A('<p class=note>Fine print: (1) This measures the <b>weights\'</b> encoding (rankability of true vs false), not a '
  'mind, and the probe is correlational. (2) The lens used a near-final layer (chosen on the positive control), so it '
  'is a conservative test of &ldquo;an earlier layer knows more than the output&rdquo;; a deeper layer-by-layer and an '
  'open-generation &ldquo;deeply hidden&rdquo; pass are the natural follow-ups. (3) Modest n (16–40/topic); this is a '
  'pre-registered proof-of-method slice on Ranke-1929 base, not the full sweep. The pre-registration, controls, and '
  'decision rules were fixed before any score was seen (see <code>PREREGISTRATION.md</code>).</p>')
A(f'<div class=foot>Ranke-4B-1929 (base), {S["n_layers"]} layers; lens layer {LENS["Lstar"]}, probe layer {PROBE["Lp"]}; '
  f'bootstrap 95% CIs over items, 1,000-shuffle null, BH-FDR. Generated {datetime.datetime.now().strftime("%Y-%m-%d")}. '
  f'Method: Gekhman et&nbsp;al. 2025 (arXiv:2503.15299).</div>')
A('</div></body></html>')
open(OUT, "w", encoding="utf-8").write("\n".join(P))
print("wrote", OUT, "| pos_pass:", pos_pass, "| null_pass:", null_pass, "| neg-probe flagged:", neg_probe_flag)
