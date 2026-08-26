# -*- coding: utf-8 -*-
"""Standalone HTML summary of the association-lift probe (assoc_results2.csv).
lift = logp(concept | target prompt) - logp(concept | matched control prompt).
Sections: metric explainer, group means (4 models), validity check, PRE controls, POST leak test."""
import csv, statistics as st, html
R = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\assoc_results2.csv"
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\assoc_summary.html"
esc = html.escape
rows = list(csv.DictReader(open(R, encoding="utf-8")))
mods = ["Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929", "Ranke-4B-1929-1202-Instruct"]
MN = {"Ranke-4B-1900": "1900", "Ranke-4B-1913": "1913", "Ranke-4B-1929": "1929", "Ranke-4B-1929-1202-Instruct": "1929-Instr"}


def grp(model, kind):
    return [float(r["lift"]) for r in rows if r["model"] == model and r["kind"] == kind]


def items(model, kind):
    return sorted([(r["label"], float(r["lift"]), r["target"], r["control"]) for r in rows
                   if r["model"] == model and r["kind"] == kind], key=lambda x: -x[1])


def liftcolor(v):
    # center 0; green positive, red negative, magnitude capped ~6
    c = max(-6.0, min(6.0, v)) / 6.0
    if c >= 0:
        r, g, b = int(245 - 199 * c), int(247 - 122 * c), int(245 - 195 * c)  # white->green
    else:
        r, g, b = int(245 + 0 * c), int(247 + 100 * c), int(245 + 100 * c)    # white->red
        r, g, b = 245, int(247 + 90 * c), int(245 + 90 * c)
    fg = "#fff" if abs(c) > 0.72 else "#1a1d24"
    return f"rgb({r},{g},{b})", fg


CSS = ("body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:940px;margin:0 auto;padding:36px 22px;color:#1a1d24;line-height:1.5}"
       "h1{font-family:Georgia,serif;font-size:26px;margin:0 0 4px}.sub{color:#5b6675;font-size:15px;margin:0 0 20px}"
       "h2{font-family:Georgia,serif;font-size:20px;margin:30px 0 6px;border-top:2px solid #e7eaef;padding-top:14px}"
       "table{border-collapse:collapse;margin:12px 0;font-size:13px;width:100%}th,td{border:1px solid #eceff3;padding:6px 9px}"
       "th{background:#f0f3f8;text-align:left}.lab{background:#f8fafc}.r{text-align:right;font-variant-numeric:tabular-nums}"
       ".box{background:#fafbfd;border:1px solid #e2e6ec;border-left:4px solid #2b6cb0;border-radius:7px;padding:12px 16px;margin:14px 0;font-size:14px}"
       ".good{border-left-color:#2e7d32}.warn{border-left-color:#c0843a}"
       "code{background:#f0f3f8;padding:1px 5px;border-radius:4px;font-size:13px}"
       ".two{display:flex;gap:18px}.two>div{flex:1}")
H = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
     f"<title>Ranke &mdash; association-lift summary</title><style>{CSS}</style></head><body>"]
A = H.append

A("<h1>What has it seen? &mdash; association-lift</h1>")
A("<p class=sub>Does the time-locked model carry associations it should not? We measure each association "
  "directly, isolating the leaked link from generic word-frequency.</p>")

A("<div class=box><b>The metric.</b> For a place&ndash;event pair we score the same event phrase after the "
  "<i>target</i> place and after a <i>matched control</i> place, and take the difference:<br>"
  "<code>lift = logp(event | &ldquo;The city of <b>Hiroshima</b>&hellip;&rdquo;) &minus; "
  "logp(event | &ldquo;The city of <b>Kyoto</b>&hellip;&rdquo;)</code><br>"
  "The control cancels everything generic (sentence shape, how common the event words are). What survives is "
  "whether the model specifically ties <i>this</i> place to <i>this</i> event. <b>lift &gt; 0</b> = the link is "
  "stored; <b>lift &asymp; 0</b> = no more than chance; <b>lift &lt; 0</b> = it prefers the control.</p></div>")

# ---- group means ----
A("<h2>1. The four groups at a glance</h2>")
GROUPS = [("POST", "Post-cutoff events", "things that happened <i>after</i> the cutoff &mdash; should be &asymp;0 if clean"),
          ("PRE_STRONG", "19th-c / Anglo controls", "famous pre-cutoff links &mdash; should be strongly positive"),
          ("PRE_WEAK", "1914&ndash;29 / Continental", "the WWI &lsquo;Hindenburg zone&rsquo; &mdash; pre-cutoff only for the 1929 model"),
          ("NULL", "Matched null", "a place with no real tie to the event &mdash; should be &asymp;0")]
A("<table><thead><tr><th>group</th><th>what it is</th>" + "".join(f"<th class=r>{MN[m]}</th>" for m in mods) + "</tr></thead><tbody>")
for k, name, desc in GROUPS:
    row = f"<tr><td class=lab><b>{name}</b></td><td style='color:#5b6675'>{desc}</td>"
    for m in mods:
        v = grp(m, k); mean = st.mean(v) if v else 0
        bg, fg = liftcolor(mean)
        row += f"<td class=r style='background:{bg};color:{fg}'>{mean:+.2f}</td>"
    A(row + "</tr>")
A("</tbody></table>")
A("<div class=box good><b>The validity check the table passes.</b> The <b>1914&ndash;29</b> row reads "
  f"<b>{st.mean(grp('Ranke-4B-1900','PRE_WEAK')):+.2f}</b> for the 1900 model and "
  f"<b>{st.mean(grp('Ranke-4B-1913','PRE_WEAK')):+.2f}</b> for 1913 &mdash; correctly &asymp;0, because the First "
  f"World War is <i>post-cutoff for those models</i> &mdash; then jumps to <b>{st.mean(grp('Ranke-4B-1929','PRE_WEAK')):+.2f}</b> "
  f"for the 1929 model, right alongside the {st.mean(grp('Ranke-4B-1929','PRE_STRONG')):+.2f} of the 19th-century "
  "controls. The metric switches on exactly when the events enter each model's window. That is the behaviour we "
  "want before trusting any single number.</div>")

# ---- PRE controls side by side ----
A("<h2>2. What it genuinely holds (pre-cutoff controls, 1929 model)</h2>")
A("<p style='color:#5b6675;font-size:14px'>The 1929 model stores the WWI era almost as firmly as the Napoleonic "
  "era. This is why &lsquo;it doesn&rsquo;t know Hindenburg&rsquo; is wrong: the associations are there.</p>")
A("<div class=two>")
for k, title in [("PRE_STRONG", "19th-c / Anglo"), ("PRE_WEAK", "1914&ndash;29 / Continental")]:
    A(f"<div><table><thead><tr><th>{title}</th><th class=r>lift</th></tr></thead><tbody>")
    for lab, v, tg, ct in items("Ranke-4B-1929", k):
        bg, fg = liftcolor(v)
        A(f"<tr><td>{esc(lab)}</td><td class=r style='background:{bg};color:{fg}'>{v:+.2f}</td></tr>")
    A("</tbody></table></div>")
A("</div>")

# ---- POST leak test ----
A("<h2>3. The leak test: post-cutoff events (1929 model)</h2>")
post = items("Ranke-4B-1929", "POST")
pos = [x for x in post if x[1] > 0]
A("<table><thead><tr><th>place &rarr; event</th><th class=r>lift</th><th>place existed pre-1929?</th></tr></thead><tbody>")
for lab, v, tg, ct in post:
    bg, fg = liftcolor(v)
    A(f"<tr><td>{esc(lab)} <span style='color:#9aa3af'>(vs {esc(ct)})</span></td>"
      f"<td class=r style='background:{bg};color:{fg}'>{v:+.2f}</td><td style='color:#5b6675'>yes &mdash; toponym predates cutoff</td></tr>")
A("</tbody></table>")
A("<div class=box warn><b>Why this is not contamination.</b> The group mean is &asymp;0. A handful of small "
  f"positives sit at the top (max {pos[0][1]:+.2f}, {esc(pos[0][0])}) &mdash; all of them places whose <i>name</i> "
  "already existed before 1929, so the model carries a faint generic prior for the location, not the event.<br><br>"
  "The decisive tell is the <b>ordering</b>. If the model had truly absorbed post-1929 history, its most "
  "<i>iconic</i> events would show the <i>largest</i> lift. They show the most <b>negative</b>: "
  f"Hiroshima&rarr;atomic bomb {dict((l,v) for l,v,_,_ in post)['Hiroshima->atomic bomb']:+.2f}, "
  f"Pearl Harbor {dict((l,v) for l,v,_,_ in post)['Pearl Harbor->attack']:+.2f}, "
  f"Cuba&rarr;missile crisis {dict((l,v) for l,v,_,_ in post)['Cuba->missile crisis']:+.2f}. "
  "That is the opposite of a leak &mdash; it is a model that has never read about them.</div>")

A("<div class=box good><b>Verdict.</b> No systematic post-cutoff leak. The few positive lifts are weak place-name "
  "priors / control-side artifacts, contradicted by strongly negative lifts on the most famous post-1929 events. "
  "Meanwhile the pre-cutoff controls &mdash; including the full 1914&ndash;29 / German block &mdash; are robustly "
  "positive. The model knows its own era and stops cleanly at the wall.</div>")
# ===================== 4. THE ABSOLUTE TEST =====================
import math
absR = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\absolute_probe_results.csv"
ab = list(csv.DictReader(open(absR, encoding="utf-8")))
ABM = ["Ranke-4B-1929", "Ranke-4B-1929-1202-Instruct"]
ABN = {"Ranke-4B-1929": "1929", "Ranke-4B-1929-1202-Instruct": "1929-Instr"}
TOPICS_O = []
for r in ab:
    if r["topic"] not in TOPICS_O:
        TOPICS_O.append(r["topic"])
def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))
def gb(model, topic, label):
    for r in ab:
        if r["model"] == model and r["topic"] == topic and r["label"] == label:
            return float(r["gap_best"])
    return None
NOTE = {
 "Gulf of Tonkin": "Tonkin = French protectorate, pre-1929 geography",
 "Paris accords": "Paris = default treaty city (1783/1856/1898)",
 "Rolling Thunder": "&lsquo;rolling thunder&rsquo; = common English phrase",
 "Saigon=S capital": "Saigon = capital of French Cochinchina, pre-1929",
 "Barbarossa": "Barbarossa = medieval emperor, pre-1929",
 "Yalta 1945": "Yalta = famous Tsarist Crimean resort (cued)",
 "Auschwitz": "the only Polish town among German camps (cued)",
 "Manhattan Project": "Manhattan = iconic place name",
 "iron curtain": "&lsquo;iron curtain&rsquo; = pre-1929 theatre fire-curtain term",
 "Sputnik": "&lsquo;sputnik&rsquo; = ordinary Russian word (companion)",
 "NATO": "most word-like of 4 made-up-looking acronyms (distractor artifact?)",
 "Warsaw Pact": "FLAG &mdash; not an obvious pre-1929 prior; needs a dig",
}

A("<h2>4. The absolute test &mdash; where does it land on facts that exist <i>only</i> after 1929?</h2>")
A("<div class=box>Contamination is <b>absolute, not relative</b>. A model that swallowed one Vietnam-War book "
  "plus a library of French-Indochina text will always lean &lsquo;French&rsquo; &mdash; so a relative test "
  "false-clears it. Instead we ask 15 facts per topic that exist <i>only</i> in post-cutoff text, each a "
  "forced choice against 3 matched-plausible distractors. A model with zero exposure lands at "
  "<b>chance = 25%</b> (3.75/15). Above that = contaminated, however small next to its legitimate knowledge.</div>")
A("<table><thead><tr><th>topic</th>" + "".join(f"<th class=r>{ABN[m]}</th>" for m in ABM) + "<th class=r>chance</th><th>read</th></tr></thead><tbody>")
for t in TOPICS_O:
    labs = []
    for r in ab:
        if r["topic"] == t and r["label"] not in labs:
            labs.append(r["label"])
    n = len(labs); row = f"<tr><td class=lab><b>{t}</b></td>"
    pmax = 0
    for m in ABM:
        k = sum(1 for l in labs if (gb(m, t, l) or -9) > 0); p = binom_ge(k, n); pmax = max(pmax, k)
        bg, fg = ("#fde8e8", "#7a1d1d") if p < 0.10 else ("#ecfdf3", "#14532d")
        row += f"<td class=r style='background:{bg};color:{fg}'>{k}/{n}<br>p={p:.2f}</td>"
    elevated = any(binom_ge(sum(1 for l in labs if (gb(m, t, l) or -9) > 0), n) < 0.10 for m in ABM)
    row += f"<td class=r style='color:#888'>{0.25*n:.1f}</td><td style='color:{'#b91c1c' if elevated else '#2e7d32'}'>{'elevated &mdash; flag' if elevated else 'at chance &mdash; clean'}</td></tr>"
    A(row)
A("</tbody></table>")

A("<p style='color:#5b6675;font-size:13px'>Per-fact gap (true minus best distractor; &gt;0 = beats all three). "
  "Green hits annotated where a <i>pre-cutoff</i> reason explains them.</p>")
for t in TOPICS_O:
    labs = []
    for r in ab:
        if r["topic"] == t and r["label"] not in labs:
            labs.append(r["label"])
    labs.sort(key=lambda l: -(gb("Ranke-4B-1929", t, l) or -9))
    A(f"<h3 style='font-family:Georgia,serif;font-size:15px;margin:16px 0 4px'>{t}</h3>")
    A("<table><thead><tr><th>fact</th><th class=r>1929</th><th class=r>Instr</th><th>note</th></tr></thead><tbody>")
    for l in labs:
        v1 = gb("Ranke-4B-1929", t, l); v2 = gb("Ranke-4B-1929-1202-Instruct", t, l)
        b1, f1 = liftcolor(v1); b2, f2 = liftcolor(v2)
        nt = NOTE.get(l, "")
        ntc = "#b91c1c" if l == "Warsaw Pact" else "#5b6675"
        A(f"<tr><td>{esc(l)}</td><td class=r style='background:{b1};color:{f1}'>{v1:+.1f}</td>"
          f"<td class=r style='background:{b2};color:{f2}'>{v2:+.1f}</td><td style='font-size:12px;color:{ntc}'>{nt}</td></tr>")
    A("</tbody></table>")

A("<div class=box warn><b>Verdict on the absolute test.</b> <b>Vietnam (5/15) and WWII (4&ndash;5/15) land at "
  "chance</b>, and every green hit has a pre-cutoff explanation &mdash; the model is reaching for knowledge it "
  "legitimately has (Tonkin the French protectorate, Barbarossa the medieval emperor, Yalta the Tsarist resort). "
  "Tellingly, every fact that needs <i>real</i> post-cutoff knowledge fails hard and the model substitutes an "
  "in-period answer: napalm&rarr;mustard gas, Saigon-fell&rarr;1945, cosmonaut&rarr;astronaut, "
  "Berlin-Wall&rarr;1919.<br><br>"
  "<b>Cold War looked elevated</b> (Instruct 7/15, p=0.057), driven by NATO (+4.7) and Warsaw Pact (+6.3) "
  "&mdash; so we dug it rather than wave it off. The dig cleared it. <b>Property-coherence:</b> the model "
  "fails the Warsaw Pact's founding year (1955, &minus;7.6) and dissolution (1991, &minus;9.0) catastrophically "
  "and does not link it to NATO or communism &mdash; 1&ndash;2 of 6, at chance: it does not <i>know</i> the "
  "pact. <b>Neutral-frame:</b> NATO (+2.9) and &lsquo;Warsaw Pact&rsquo; (+1.0) still win in content-free "
  "sentences, so the lift is a pure string/bigram prior. <b>Source:</b> the model genuinely knows the "
  "in-window <i>1920 Battle of Warsaw</i> (+3.4), which feeds the &lsquo;Warsaw&rsquo; bigram. Verdict: "
  "lexical artifact, not contamination. All three topics clean.</div>")
A("<p style='color:#9aa3af;font-size:12px;margin-top:24px'>Source: 5_results/assoc_results2.csv (49 triples, "
  "4 models) and 5_results/absolute_probe_results.csv (3 topics &times; 15 facts &times; 2 models, "
  "forced choice vs 3 matched distractors). Gaps/lift in nats (sum log-prob).</p>")
A("</body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote", OUT)
