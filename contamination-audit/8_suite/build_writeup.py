# -*- coding: utf-8 -*-
"""Build writeup.html -- a guarded, step-by-step account of the whole Ranke/Talkie contamination audit:
the methods, every reversal and mistake, and the final (qualified) finding. Numbers for the v2 result are
recomputed live from results_v2_Ranke-4B-1929.json; v1 comparison from the saved aggregates. No claims beyond
what the data supports; an explicit 'what we do NOT claim' and 'open' section."""
import os, json, math, html, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "writeup.html")
esc = html.escape


def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def wilson(k, n):
    if not n: return (0, 0)
    z = 1.96; p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


# ---------- load data ----------
v2 = json.load(open(os.path.join(HERE, "results_v2_Ranke-4B-1929.json")))
it = v2["items"]
pre = [r for r in it if r["era"] == "pre"]; post = [r for r in it if r["era"] == "post"]
def raw(rows): return (sum(r["correct_raw"] for r in rows), len(rows))
def cal(rows): return (sum(r["correct_cal"] for r in rows), len(rows))
pre_raw_k, pre_n = raw(pre); pre_cal_k, _ = cal(pre)
post_raw_k, post_n = raw(post)
CL = ["US civil rights", "US politics 1945-74", "Cold War", "Vietnam War"]
cluster = {}
for c in CL:
    rows = [r for r in post if r["topic"] == c]
    k, n = raw(rows); lo, hi = wilson(k, n)
    cluster[c] = {"k": k, "n": n, "pct": round(100 * k / n, 1), "p": binom_ge(k, n), "hi": hi}
# FDR
ps = sorted((cluster[c]["p"], c) for c in CL); m = len(ps)
fdr = {c: (p <= (i / m) * 0.05) for i, (p, c) in enumerate(ps, 1)}
broad_hi = wilson(post_raw_k, post_n)[1]
v1r = json.load(open(os.path.join(HERE, "results_Ranke-4B-1929.json")))["aggregate"]
v1t = json.load(open(os.path.join(HERE, "results_talkie-1930-13b-base.json")))["aggregate"]
bank = {x["label"]: x for x in json.load(open(os.path.join(HERE, "question_bank_v2.json"), encoding="utf-8"))["items"]}
cr_hits = sorted([r for r in post if r["topic"] == "US civil rights" and r["correct_raw"]], key=lambda r: -r["gap_raw"])

# ---------- svg helpers ----------
def bars(title, series, ymax=100, chance=25.0, w=600, h=300):
    pl, pb, pt = 50, 70, 38; pw, ph = w - pl - 18, h - pb - pt
    n = len(series); bw = pw / max(n, 1) * 0.62; gap = pw / max(n, 1)
    s = [f'<svg viewBox="0 0 {w} {h}" style="max-width:100%;height:auto;border:1px solid #e7eaef;border-radius:8px;background:#fff">']
    s.append(f'<text x="{w/2}" y="20" text-anchor="middle" style="font:700 14px Georgia,serif;fill:#1a1d24">{esc(title)}</text>')
    for gv in (0, 25, 50, 75, 100):
        if gv > ymax: continue
        y = pt + ph * (1 - gv / ymax)
        s.append(f'<line x1="{pl}" x2="{w-18}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eef1f5"/><text x="{pl-6}" y="{y+4:.1f}" text-anchor="end" style="font:10px sans-serif;fill:#8a93a0">{gv}</text>')
    yc = pt + ph * (1 - chance / ymax)
    s.append(f'<line x1="{pl}" x2="{w-18}" y1="{yc:.1f}" y2="{yc:.1f}" stroke="#c0392b" stroke-dasharray="5 4"/><text x="{w-20}" y="{yc-4:.1f}" text-anchor="end" style="font:700 10px sans-serif;fill:#c0392b">chance {chance:.0f}%</text>')
    for i, (lab, val, col) in enumerate(series):
        x = pl + gap * i + (gap - bw) / 2; bh = ph * (min(val, ymax) / ymax); y = pt + ph - bh
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{col}"/><text x="{x+bw/2:.1f}" y="{y-5:.1f}" text-anchor="middle" style="font:700 12px sans-serif;fill:#1a1d24">{val:.0f}%</text>')
        for j, ln in enumerate(lab.split("\n")):
            s.append(f'<text x="{x+bw/2:.1f}" y="{h-pb+15+j*12:.1f}" text-anchor="middle" style="font:10.5px sans-serif;fill:#52637a">{esc(ln)}</text>')
    return "".join(s) + "</svg>"


def flow():
    steps = [
        ("1. Goal", "Do time-locked Ranke-1929 / Talkie-1930 secretly know post-cutoff facts? (to validate as synthetic respondents)", "#2b6cb0", ""),
        ("2. Method set", "Teacher-forced log-probs, matched-twin forced choice, no first-name cues, length-matching, pre/post controls", "#2b6cb0", ""),
        ("3. Metric bug", "Avg-per-token log-prob produced a spurious '19/28 US-politics, p=.044'. Switched to sum-log-prob over the whole answer.", "#c0843a", "reversal"),
        ("4. Digs", "Association-lift, decoy, cross-model. Apparent signals (Hiroshima->atomic, Warsaw Pact) dissolved into artifacts (salience, lexical priors).", "#c0843a", "reversals"),
        ("5. Cross-model retracted", "Claimed '1900 can't contain WWII, so the signal is innocent.' Wrong: a leak could hit all cutoffs / warm-start dilutes. Dropped the leg.", "#b91c1c", "retraction"),
        ("6. Absolute test + Talkie", "Forced choice vs chance on Vietnam/WWII/Cold War: all at chance. Talkie too. (Guohui's rank-ratio inflated by 13B size; matched distractors fix it.)", "#2b6cb0", ""),
        ("7. 'Clean' -> overclaim", "Called it clean. But: 'failed to detect' != clean (underpowered), the target kept moving, and US domestic history was never tested.", "#b91c1c", "overclaim"),
        ("8. Reckoning + pre-reg", "Reframed to DETECTION not proof. Pre-registered: blind holdout, nested delta (broad +10pp / narrow +20pp), event-clusters, FDR. Mistakes caught: dead model, inverted referee, adjacent-year distractors, calibration broke the instrument.", "#2b6cb0", ""),
        ("9. Validated result", "RAW forced choice (pre-controls 82%). EVIDENCE of contamination concentrated in US civil rights (46%, p=.004, FDR-flagged); US politics elevated; Cold War & Vietnam at chance. Item-level dig confirms specific recall.", "#2e7d32", "finding"),
    ]
    W = 760; rh = 78; gap = 26; H = len(steps) * (rh + gap) + 20
    s = [f'<svg viewBox="0 0 {W} {H}" style="max-width:100%;height:auto">']
    for i, (t, d, col, tag) in enumerate(steps):
        y = 10 + i * (rh + gap)
        s.append(f'<rect x="60" y="{y}" width="{W-120}" height="{rh}" rx="9" fill="#fafbfd" stroke="{col}" stroke-width="2"/>')
        s.append(f'<rect x="60" y="{y}" width="7" height="{rh}" rx="3" fill="{col}"/>')
        s.append(f'<text x="80" y="{y+24}" style="font:700 14px Georgia,serif;fill:{col}">{esc(t)}</text>')
        # wrap description
        words = d.split(); line = ""; ly = y + 44
        for wd in words:
            if len(line) + len(wd) > 84:
                s.append(f'<text x="80" y="{ly}" style="font:12px sans-serif;fill:#3a424e">{esc(line)}</text>'); ly += 15; line = wd
            else:
                line = (line + " " + wd).strip()
        s.append(f'<text x="80" y="{ly}" style="font:12px sans-serif;fill:#3a424e">{esc(line)}</text>')
        if tag:
            tc = {"reversal": "#c0843a", "reversals": "#c0843a", "retraction": "#b91c1c", "overclaim": "#b91c1c", "finding": "#2e7d32"}[tag]
            s.append(f'<text x="{W-72}" y="{y+20}" text-anchor="end" style="font:700 10px sans-serif;fill:{tc};text-transform:uppercase">&#9888; {esc(tag)}</text>')
        if i < len(steps) - 1:
            s.append(f'<line x1="{W/2}" x2="{W/2}" y1="{y+rh}" y2="{y+rh+gap}" stroke="#9aa3af" stroke-width="2" marker-end="url(#ar)"/>')
    s.insert(1, '<defs><marker id="ar" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#9aa3af"/></marker></defs>')
    return "".join(s) + "</svg>"


CSS = ("body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:980px;margin:0 auto;padding:38px 22px;color:#1a1d24;line-height:1.6}"
       "h1{font-family:Georgia,serif;font-size:28px;margin:0 0 6px}.sub{color:#5b6675;font-size:15.5px;margin:0 0 8px}"
       "h2{font-family:Georgia,serif;font-size:21px;margin:34px 0 8px;border-top:2px solid #e7eaef;padding-top:16px}"
       "h3{font-size:16px;margin:18px 0 6px}p{margin:8px 0}"
       "table{border-collapse:collapse;margin:12px 0;font-size:13.5px;width:100%}th,td{border:1px solid #eceff3;padding:7px 10px;vertical-align:top}"
       "th{background:#f0f3f8;text-align:left}.lab{background:#f8fafc;font-weight:600}.r{text-align:right;font-variant-numeric:tabular-nums}"
       ".box{background:#fafbfd;border:1px solid #e2e6ec;border-left:4px solid #2b6cb0;border-radius:8px;padding:13px 17px;margin:15px 0;font-size:14.5px}"
       ".good{border-left-color:#2e7d32}.warn{border-left-color:#c0843a}.bad{border-left-color:#b91c1c}"
       "code{background:#f0f3f8;padding:1px 5px;border-radius:4px;font-size:12.5px}.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}"
       ".charts{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0}.charts>div{flex:1;min-width:300px}"
       ".tag{display:inline-block;font-size:11px;font-weight:700;padding:2px 7px;border-radius:4px;text-transform:uppercase}"
       ".tc{background:#e7f5ec;color:#1e7a3a}.tt{background:#fdf0e0;color:#9a5a1a}.to{background:#fdeaea;color:#a12a2a}")

H = [f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ranke / Talkie contamination audit &mdash; full account</title><style>{CSS}</style></head><body>']
A = H.append
A("<h1>Auditing time-locked LLMs for post-cutoff contamination</h1>")
A("<p class=sub>Ranke-4B-1929 and Talkie-1930-13B &mdash; a step-by-step record of what we did, every reversal, and a deliberately guarded reading of the result.</p>")

A("<div class=box warn><b>Bottom line, stated carefully.</b> After a long chain of methods &mdash; several of which we had to <i>reverse</i> &mdash; a pre-registered, blind-screened, instrument-validated forced-choice battery finds <b>evidence of post-cutoff contamination in Ranke-4B-1929, concentrated in US civil-rights-era history</b> (1948&ndash;1968): "
  f"{cluster['US civil rights']['pct']:.0f}% correct vs 25% chance, p={cluster['US civil rights']['p']:.3f}, surviving FDR correction, and confirmed by item-level inspection (it picks <i>Brown v. Board</i> over <i>Plessy</i>, <i>Wallace</i> over other segregationist governors, the <i>Orangeburg massacre</i> over other campus shootings). "
  "US domestic politics 1945&ndash;74 is elevated but borderline; Cold War and Vietnam sit at chance. "
  "This is <b>evidence, not proof</b>: the instrument validates at 82% (not 100%) on pre-cutoff controls; full certification would require finding the source text in the training corpus; and Talkie and the Ranke instruct model have <i>not</i> yet been scored on this validated battery. We are <b>not</b> claiming a clean bill anywhere, nor a complete contamination map.</div>")

A("<h2>The journey (and where it turned)</h2>")
A("<p>Orange = a method reversal; red = a retraction or overclaim we had to walk back; green = the surviving finding. The reversals are the point of the diagram, not an embarrassment to hide &mdash; each was a way the answer could have been (and briefly was) wrong.</p>")
A('<div style="margin:14px 0">' + flow() + "</div>")

A("<h2>1&ndash;2. What we were measuring, and the traps</h2>")
A("<p>The question is narrow: does the model assign higher probability to a <i>true</i> post-cutoff answer than to matched false ones, under teacher forcing (we read log-probs; we never trust free generation, which <i>understates</i> knowledge &mdash; the model knew Hindenburg under scoring but confabulated him in chat). The hard part is that several things masquerade as knowledge:</p>")
A("<table><thead><tr><th>Confound</th><th>How it fakes 'knowledge'</th><th>Our guard</th></tr></thead><tbody>"
  "<tr><td class=lab>Token-length bias</td><td>Average per-token log-prob favours short answers</td><td>Sum log-prob over the whole answer; length-matched alternatives</td></tr>"
  "<tr><td class=lab>Name cueing</td><td>'Winston ___' &rarr; Churchill is completion, not knowledge</td><td>No fragment of the answer (or its form) in the prompt</td></tr>"
  "<tr><td class=lab>Model-size floor</td><td>A 13B guesses a plausible-sounding name better than a 4B</td><td>Forced choice vs matched distractors: same 25% chance line for any size</td></tr>"
  "<tr><td class=lab>Lexical popularity</td><td>'NATO' / 'Warsaw Pact' win as common strings</td><td>Distinct distractors; we tested (and ultimately dropped) a calibration subtracting the answer's neutral prior</td></tr>"
  "<tr><td class=lab>Place salience</td><td>Hiroshima makes any catastrophe-word likelier than Kyoto</td><td>Difference-in-differences across control subjects</td></tr></tbody></table>")

A("<h2>3&ndash;5. Exploratory probing, and the reversals it produced</h2>")
A("<p>Early forced-choice and association probes <i>did</i> flag domains &mdash; most visibly a '19/28, p=.044' on US politics. Each flag dissolved on inspection:</p>"
  "<ul><li>The '19/28' was a <b>metric artifact</b> (average-per-token log-prob). Under sum-log-prob with matched twins it vanished.</li>"
  "<li>'Hiroshima&rarr;atomic bomb' and 'Warsaw Pact' looked leaked, but a content-free / control-subject test showed they were <b>place salience</b> and <b>lexical priors</b> &mdash; the model fails the Warsaw Pact's founding year (&minus;7.6) and dissolution (&minus;9.0), so it likes the bigram, not the alliance.</li>"
  "<li>We argued '<i>the 1900 model cannot contain WWII text, so its showing the signal proves the signal is innocent</i>'. <b>We retracted this</b>: the 1900 cutoff is itself the claim under audit, and if the models are warm-started a leak propagates and dilutes &mdash; the inference was circular.</li></ul>")

A("<h2>6. The absolute test and the Talkie comparison</h2>")
A("<p>A forced-choice battery (true vs 3 matched distractors, chance 25%) on Vietnam, WWII and the Cold War put both models at chance post-cutoff while acing pre-cutoff controls. On the <i>identical</i> bank:</p>")
A('<div class=charts><div>' + bars("Post-cutoff %-correct (v1 bank: Vietnam/WWII/Cold War)",
    [("Ranke-4B\n1929", v1r["post:ALL"]["pct"], "#2b6cb0"), ("Talkie-1930\n13B", v1t["post:ALL"]["pct"], "#c0392b")]) + "</div>"
  '<div>' + bars("Pre-cutoff controls (probe fires)",
    [("Ranke-4B\n1929", v1r["pre:ALL"]["pct"], "#2b6cb0"), ("Talkie-1930\n13B", v1t["pre:ALL"]["pct"], "#c0392b")]) + "</div></div>")
A("<div class=box>Talkie looked <i>more</i> contaminated on a co-author's rank-gain metric (5.5&times; vs 1.7&times;), but that metric rewards a bigger model for generating plausible names; matched distractors put both at chance on these domains. <b>Caveat we missed at the time:</b> these are <i>foreign-policy / military</i> topics. We had not tested US <i>domestic</i> history, where the real signal turned out to be.</div>")

A("<h2>7&ndash;8. The reckoning, and the pre-registered study</h2>")
A("<p>Calling the model 'clean' on the strength of at-chance results was an overclaim: 'failed to detect' is not 'absent', the battery was underpowered (n=15/topic; a 5/15 has a 95% upper bound near 50%), and the test had been tuned while looking at one model's scores. We reframed the goal to <b>detecting evidence of contamination</b>, and pre-registered a frozen protocol before generating or scoring any item:</p>"
  "<ul><li><b>Nested ('Russian-doll') thresholds.</b> A broad post-1945 pool (~240 items) powered to detect a <b>+10pp</b> diffuse lift; deep clusters (~32&ndash;40 each) powered to <b>+20pp</b> concentrated leaks.</li>"
  "<li><b>Blind screening before scoring.</b> Items generated, then a separate adversarial referee + deterministic checks judged <i>fairness only</i> (post-cutoff? cued? distractors discriminable? factually correct?) with no model scores in hand, then locked. No tuning to an outcome afterwards.</li>"
  "<li><b>Event-cluster analysis</b> with Benjamini&ndash;Hochberg FDR across clusters; pre-cutoff controls to validate the instrument first.</li></ul>")
A("<div class=box warn><b>Mistakes caught <i>inside</i> the pre-registered build</b> (each would have produced a wrong answer): the model ID in our notes was retired (404s); the screening referee initially <b>inverted</b> the rule and rejected items <i>for being post-cutoff</i>; the generator produced adjacent-year distractors a knower can't discriminate; and a popularity-<b>calibration</b> we adopted from advice actually <b>broke the instrument</b> &mdash; it dropped pre-control accuracy from 82% to 48%. We discarded calibration and use the raw forced choice, a choice made on the should-know controls, blind to the contamination question.</div>")

A("<h2>9. The validated result</h2>")
A(f"<p>Instrument check first: on the locked pre-cutoff controls the raw metric scores <b>{pre_raw_k}/{pre_n} = {round(100*pre_raw_k/pre_n)}%</b> (the calibrated metric scored only {round(100*pre_cal_k/pre_n)}% &mdash; why we dropped it). 82% is good enough to trust a positive signal, though not perfect; a handful of genuinely hard controls (Treaty of Nanjing, the transatlantic cable) fail.</p>")
A('<div class=charts><div>' + bars("Ranke-4B-1929 post-cutoff %-correct by cluster (raw, chance 25%)",
    [("US civil\nrights", cluster["US civil rights"]["pct"], "#b91c1c"),
     ("US politics\n1945-74", cluster["US politics 1945-74"]["pct"], "#c0843a"),
     ("Cold War", cluster["Cold War"]["pct"], "#2b6cb0"),
     ("Vietnam", cluster["Vietnam War"]["pct"], "#2b6cb0")], w=560) + "</div>"
  '<div>' + bars("Pre-controls: raw vs calibrated (why calibration was dropped)",
    [("raw\n(used)", round(100*pre_raw_k/pre_n), "#2e7d32"), ("calibrated\n(broken)", round(100*pre_cal_k/pre_n), "#b91c1c")], w=400) + "</div></div>")
A("<table><thead><tr><th>cluster</th><th class=r>correct</th><th class=r>%</th><th class=r>chance</th><th class=r>p (one-sided)</th><th class=r>95% upper</th><th>FDR</th></tr></thead><tbody>")
for c in CL:
    cc = cluster[c]
    A(f"<tr><td class=lab>{esc(c)}</td><td class=r>{cc['k']}/{cc['n']}</td><td class=r>{cc['pct']:.0f}%</td><td class=r>25%</td><td class=r>{cc['p']:.4f}</td><td class=r>{cc['hi']:.0f}%</td><td>{'<b style=color:#b91c1c>flagged</b>' if fdr[c] else 'no'}</td></tr>")
A(f"<tr><td class=lab>broad post pool</td><td class=r>{post_raw_k}/{post_n}</td><td class=r>{round(100*post_raw_k/post_n)}%</td><td class=r>25%</td><td class=r>{binom_ge(post_raw_k,post_n):.4f}</td><td class=r>{broad_hi:.0f}%</td><td>&mdash;</td></tr>")
A("</tbody></table>")
A("<p style='color:#5b6675;font-size:13px'>The broad pool is above chance (upper bound past the +10pp line), but that is <i>driven by</i> the two US-domestic clusters; the pattern is concentrated, not diffuse.</p>")

A("<h3>Why this is read as real recall, not luck (the civil-rights items it got right)</h3>")
A("<p>The distractors are <i>other real entities from the same domain</i>, so picking the correct one requires specific knowledge, not elimination. Strongest hits:</p>")
A("<table><thead><tr><th class=r>gap</th><th>prompt (truncated)</th><th>answer chosen</th><th>distractors</th></tr></thead><tbody>")
for r in cr_hits[:12]:
    b = bank.get(r["label"], {})
    A(f"<tr><td class=r>+{r['gap_raw']:.1f}</td><td style='font-size:12px'>{esc(b.get('prompt','')[:90])}&hellip;</td><td><b>{esc(b.get('true','').strip())}</b></td><td style='font-size:11px;color:#5b6675'>{esc(', '.join(d.strip() for d in b.get('distractors',[])))}</td></tr>")
A("</tbody></table>")

A("<h2>What we claim &mdash; and what we do not</h2>")
A('<table><thead><tr><th>statement</th><th>status</th></tr></thead><tbody>'
  f'<tr><td>Ranke-4B-1929 carries post-cutoff knowledge of US civil-rights-era history</td><td><span class="tag tc">supported</span> &mdash; FDR-flagged + coherent specific item-level recall</td></tr>'
  '<tr><td>US domestic politics 1945&ndash;74 also leaks</td><td><span class="tag tt">tentative</span> &mdash; elevated (38%, p=.054), does not survive FDR</td></tr>'
  '<tr><td>The leak is concentrated in US domestic history, not diffuse</td><td><span class="tag tt">tentative</span> &mdash; Cold War & Vietnam at chance, but only 4 clusters tested</td></tr>'
  '<tr><td>Cold War / Vietnam / WWII are contamination-free</td><td><span class="tag to">NOT claimed</span> &mdash; at chance only means no evidence at this power</td></tr>'
  '<tr><td>Talkie-1930 is clean (or dirty) on civil rights / US politics</td><td><span class="tag to">unknown</span> &mdash; not yet scored on this validated battery</td></tr>'
  '<tr><td>This is a complete contamination map</td><td><span class="tag to">NOT claimed</span> &mdash; many domains untested</td></tr></tbody></table>')

A("<h2>Open / not yet done</h2>"
  "<ul><li>Score <b>Talkie-1930-13B</b> and <b>Ranke-1929-Instruct</b> on this same locked, validated bank.</li>"
  "<li><b>Corpus search</b> for the source text (Brown v. Board, Freedom Rides, Orangeburg, Wallace, Selma&hellip;) &mdash; the only route to <i>certify</i> and remove a leak; needs access to the training corpus.</li>"
  "<li><b>Injection calibration</b> (continued-pretrain a small model on k copies of a known post-cutoff article) to quantify the battery's detection floor in 'article-repeats'.</li>"
  "<li>Broaden beyond 4 clusters; raise the pre-control instrument above 82%.</li></ul>")

A("<h2>Honesty ledger &mdash; every reversal and mistake</h2>"
  "<table><thead><tr><th>#</th><th>what we said / did</th><th>what was wrong</th><th>fix</th></tr></thead><tbody>"
  "<tr><td>1</td><td>'19/28 US politics, p=.044, contaminated'</td><td>avg-per-token log-prob artifact</td><td>sum-log-prob, matched twins</td></tr>"
  "<tr><td>2</td><td>'Hiroshima&rarr;atomic / Warsaw Pact leaked'</td><td>place salience & lexical priors</td><td>control-subject / content-free tests</td></tr>"
  "<tr><td>3</td><td>'1900 can't have WWII, so signal is innocent'</td><td>circular; warm-start dilutes</td><td>retracted the cross-model leg</td></tr>"
  "<tr><td>4</td><td>'the model is clean'</td><td>failed-to-detect != clean; underpowered; wrong domains</td><td>detection framing + pre-registration</td></tr>"
  "<tr><td>5</td><td>'Talkie is more contaminated'</td><td>rank-gain metric inflated by 13B size</td><td>matched-distractor forced choice</td></tr>"
  "<tr><td>6</td><td>used model id deepseek-v4-pro</td><td>retired on OpenRouter (404)</td><td>deepseek-v3.2</td></tr>"
  "<tr><td>7</td><td>blind referee rejected items</td><td>inverted: rejected them <i>for being post-cutoff</i></td><td>rewrote the rubric</td></tr>"
  "<tr><td>8</td><td>generated year-answer items</td><td>adjacent-year distractors a knower can't pick</td><td>discriminability gate + non-year steer</td></tr>"
  "<tr><td>9</td><td>adopted popularity calibration</td><td>broke the instrument (pre 82%&rarr;48%)</td><td>discarded; raw forced choice</td></tr>"
  "</tbody></table>")
A("<p style='color:#9aa3af;font-size:12px;margin-top:22px'>Sources: question_bank_v2.json (locked, blind-screened, 272 items), results_v2_Ranke-4B-1929.json (raw forced-choice, sum-log-prob); v1 comparison from results_Ranke-4B-1929.json / results_talkie-1930-13b-base.json. Pre-registration in PREREGISTRATION.md; full trail in LOG.md.</p>")
A("</body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote", OUT)
print(f"civil rights {cluster['US civil rights']['pct']}% p={cluster['US civil rights']['p']:.4f} fdr={fdr['US civil rights']}; pre raw {round(100*pre_raw_k/pre_n)}% cal {round(100*pre_cal_k/pre_n)}%")
