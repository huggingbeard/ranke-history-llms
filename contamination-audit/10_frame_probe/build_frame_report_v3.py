# -*- coding: utf-8 -*-
"""Frame probe v3 report. Discriminating signal = PRODUCTION (free generation), judged by Gemini-3.5-flash into
EXTENDS/REVERTS/DISTORTS/STALLS. Primary metric = USE-IT ('what followed from this was that' -> can it run forward?),
which applies to every idea. HOLD-IT ('the one who truly held the highest place was') is diagnostic only for AUTHORITY
facts (does the held figure revert to a man?) -> shown as a focused panel. Forced-choice probes = documented negative
result. Reads frame_probe_v2.json + frame_probe_judge.json."""
import os, json, html, datetime, statistics as st
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
J = json.load(open(os.path.join(HERE, "frame_probe_judge.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "frame_probe_v3_report.html")
esc = html.escape
models = list(D["models"]); meta = D["dims_meta"]; gens = D["generations"]; res = D["results"]; items = D["items"]
dims = list(meta.keys())
NICE = {"inheritance":"Son inherits the firm","railway":"A new railway opens","succession":"Prince crowned at king's death",
 "treaty":"Powers sign a peace treaty","woman_authority":"A woman leads the bank","human_flight":"Humans fly the ocean",
 "empire_fall":"A great empire falls","wireless_voice":"Voices through the air (radio)","thinking_machine":"A machine that thinks",
 "atom_city":"One bomb destroys a city","television":"Living faces sent into the home","dna_code":"The code of life rewritten",
 "moon_landing":"Men walk on the moon"}
CLASSES = ["control", "boundary", "unthinkable"]
CLABEL = {"control":"Period-normal controls","boundary":"Became thinkable 1900→1929","unthinkable":"Genuinely beyond all three eras (post-1929)"}
AUTH = ["woman_authority", "succession", "inheritance", "empire_fall"]   # hold-it is diagnostic here
LBLC = {"EXTENDS":"#2e9e5b","REVERTS":"#d98a00","DISTORTS":"#c0392b","STALLS":"#6b7280","ERR":"#999","?":"#999"}
LBL3 = {"EXTENDS":"EXTENDS","REVERTS":"REVERTS","DISTORTS":"DISTORTS","STALLS":"STALLS"}
PILL = {"EXTENDS":"green","REVERTS":"amber","DISTORTS":"red","STALLS":"grey"}


def dimsof(c): return [d for d in dims if meta[d]["cls"] == c]
def uselab(t, d): return [J[k]["label"] for i in (0,1) for k in [f"{t}__{d}__{i}__use_it"] if k in J]
def holdlab(t, d): return [J[k]["label"] for i in (0,1) for k in [f"{t}__{d}__{i}__hold_it"] if k in J]
def dom(t, d):
    ls = uselab(t, d); return Counter(ls).most_common(1)[0][0] if ls else "?"
def assim(t, d):
    ls = uselab(t, d); return sum(1 for x in ls if x == "EXTENDS")/len(ls) if ls else 0
def dm(m, dim, k): return st.mean(res[m][it["id"]][k] for it in items if it["dim"] == dim)


def trajectory():
    W, H = 940, 430; padL, padR, padT, padB = 50, 232, 28, 44
    pw, ph = W-padL-padR, H-padT-padB
    xs = [padL + pw*i/(len(models)-1) for i in range(len(models))]
    def Y(v): return padT + ph*(1-v)
    s = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;border:1px solid #e6e9ef;border-radius:10px;background:#fff">']
    s.append(f'<text x="{padL+pw/2}" y="18" text-anchor="middle" style="font:700 14px Georgia,serif;fill:#1a2230">Can the frame run forward from the idea? (share of free continuations judged EXTENDS)</text>')
    # control band
    ctrl = dimsof("control")
    bt = [max(assim(m,d) for d in ctrl) for m in models]; bb = [min(assim(m,d) for d in ctrl) for m in models]
    pts = " ".join(f"{xs[i]:.1f},{Y(bt[i]):.1f}" for i in range(len(models))) + " " + " ".join(f"{xs[i]:.1f},{Y(bb[i]):.1f}" for i in reversed(range(len(models))))
    s.append(f'<polygon points="{pts}" fill="#7c869422"/>')
    for gv in (0,.25,.5,.75,1):
        y=Y(gv); s.append(f'<line x1="{padL}" x2="{padL+pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eef2f6"/>')
        s.append(f'<text x="{padL-6}" y="{y+3:.1f}" text-anchor="end" style="font:10px sans-serif;fill:#9aa3af">{int(gv*100)}%</text>')
    for i,m in enumerate(models):
        s.append(f'<text x="{xs[i]:.1f}" y="{H-padB+18:.1f}" text-anchor="middle" style="font:600 12px sans-serif;fill:#52637a">Ranke-{m}</text>')
    laby=[]
    for d in dims:
        if meta[d]["cls"]=="control": continue
        col = {"boundary":"#2b6cb0","unthinkable":"#c0392b"}[meta[d]["cls"]]
        pts=[(xs[i],Y(assim(m,d))) for i,m in enumerate(models)]
        s.append('<polyline points="'+" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)+f'" fill="none" stroke="{col}" stroke-width="2.2" opacity="0.85"/>')
        for x,y in pts: s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}"/>')
        ly=pts[-1][1]
        while any(abs(ly-q)<12 for q in laby): ly+=12
        laby.append(ly)
        s.append(f'<text x="{padL+pw+8:.1f}" y="{ly+3:.1f}" style="font:11px sans-serif;fill:{col}">{esc(NICE[d])}</text>')
    s.append(f'<text x="{xs[-1]:.1f}" y="{Y(bt[-1])-6:.1f}" text-anchor="end" style="font:italic 10px sans-serif;fill:#7c8694">control band</text>')
    s.append("</svg>")
    return "".join(s)


CSS = """
:root{--ink:#1a2230;--mut:#5c6675;--line:#e6e9ef;--bg:#f6f8fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1020px;margin:0 auto;padding:46px 26px 100px}
h1{font:700 28px/1.2 Georgia,serif;margin:0 0 6px}.sub{color:var(--mut);margin:0 0 22px}
h2{font:700 21px Georgia,serif;margin:44px 0 6px;padding-top:15px;border-top:1px solid var(--line)}h2 .n{color:#9aa3af;font-weight:400;margin-right:8px}
h3{font:700 16px Georgia,serif;margin:22px 0 4px}p{color:#2b3340}.lead{font-size:15px}
.callout{background:#eef4ff;border:1px solid #cfe0ff;border-left:5px solid #2b5dab;border-radius:10px;padding:14px 18px;margin:14px 0;font-size:14.5px}
.fm{background:#fdecec;border:1px solid #f6c9c9;border-left:5px solid #c0392b}
.neg{background:#fff7ed;border:1px solid #fed7aa;border-left:5px solid #ea8a3c}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:12px 0;font-size:13.5px}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line)}th{background:#eef2f7;color:#3a4252;font:600 12px sans-serif;text-transform:uppercase;letter-spacing:.4px}
tr:last-child td{border-bottom:none}.num{text-align:right;font-variant-numeric:tabular-nums}.lab{font-weight:600;background:#f8fafc}
.pill{display:inline-block;padding:2px 8px;border-radius:20px;font:700 10.5px sans-serif}
.pill.green{background:#e4f6ea;color:#1f7a47}.pill.red{background:#fbe3e3;color:#b32424}.pill.amber{background:#fcefd6;color:#9a6700}.pill.grey{background:#eceff3;color:#5c6675}
.clshdr td{background:#f0f3f8;font-weight:700;color:#3a4252;font-size:12px;text-transform:uppercase;letter-spacing:.4px}
.gen{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 15px;margin:9px 0}
.gen .plant{font-weight:600}.gen .out{display:block;margin:3px 0 8px;padding:7px 11px;background:#f6f8fb;border-left:3px solid #cfd8e6;border-radius:6px;font-size:13px;color:#33405a;white-space:pre-wrap}
.tag{display:inline-block;padding:1px 7px;border-radius:5px;font:700 10px sans-serif;color:#fff;vertical-align:middle}
.legend{font-size:12.5px;color:var(--mut);margin:4px 0 0}.foot{color:var(--mut);font-size:12.5px;margin-top:42px;border-top:1px solid var(--line);padding-top:14px}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:13px}
"""
P=[];A=P.append
A(f'<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Frame probe v3 — normal / broke / unthinkable</title><style>{CSS}</style></head><body><div class=wrap>')
A("<h1>Normal, broken, or unthinkable — judged by what the frame writes</h1>")
A(f'<p class=sub>Ranke-4B 1900 · 1913 · 1929 (base). 13 ideas × 4 phrasings; free continuations classified by Gemini-3.5-flash '
  f'(EXTENDS / REVERTS / DISTORTS / STALLS). Honest scope: the era\'s written-corpus frame, not minds.</p>')
A('<div class=callout><b>Headline.</b> Recognition is not the test — <b>production</b> is. Told a fact, the model can '
  'parse it; the question is whether it can <i>run forward</i>. <span class="pill green">EXTENDS</span> develops the '
  'fact. <span class="pill amber">REVERTS</span> the old default creeps back (a woman leader becomes a man). '
  '<span class="pill red">DISTORTS</span> the fact is rewritten into something familiar (a city-killing bomb becomes a '
  'house fire). <span class="pill grey">STALLS</span> it leaves no trace. Result: <b>controls run forward in every '
  'era; boundary ideas climb revert→extend from 1900 to 1929; genuinely-future ideas distort or stall in all three, '
  'never assimilating.</b></div>')

# §1 trajectory
A('<h2><span class=n>1</span>The shape of it</h2>')
A('<p class=lead>Each line is an idea; height is the share of its free continuations the judge calls EXTENDS — i.e. how '
  'well the frame runs forward from it. The grey band is the period-normal controls. '
  '<span style="color:#2b6cb0;font-weight:700">Boundary ideas climb into the band</span> across 1900→1929; '
  '<span style="color:#c0392b;font-weight:700">genuinely-future ideas stay pinned to the floor</span>.</p>')
A(trajectory())
cm = {cls: [round(st.mean(assim(m, d) for d in dimsof(cls)), 2) for m in models] for cls in CLASSES}
A(f'<p class=legend>Class means (EXTENDS share): controls {cm["control"]}, boundary {cm["boundary"]}, unthinkable {cm["unthinkable"]} across 1900/1913/1929.</p>')

# §2 trichotomy matrix
A('<h2><span class=n>2</span>Production trichotomy — what each idea becomes (use-it)</h2>')
A('<table><thead><tr><th>idea</th>'+"".join(f'<th>Ranke-{m}</th>' for m in models)+'</tr></thead><tbody>')
for cls in CLASSES:
    A(f'<tr class=clshdr><td colspan={len(models)+1}>{CLABEL[cls]}</td></tr>')
    for d in dimsof(cls):
        A(f'<tr><td class=lab>{esc(NICE[d])}</td>')
        for m in models:
            ls = uselab(m, d); c = Counter(ls); dd = dom(m, d)
            counts = " ".join(f'<span class=tag style="background:{LBLC[k]}">{k[:4].lower()} {c[k]}</span>' for k in ["EXTENDS","REVERTS","DISTORTS","STALLS"] if c[k])
            A(f'<td><span class="pill {PILL.get(dd,"grey")}">{LBL3.get(dd,dd)}</span><br><span class=legend>{counts}</span></td>')
        A('</tr>')
A('</tbody></table>')
A('<p class=legend>Majority of the two use-it continuations per cell. "EXTENDS" = the frame assimilated the idea; '
  '"DISTORTS"/"STALLS" = it could not (rewrote it, or drifted off).</p>')

# §3 field-marshal / distortion
A('<h2><span class=n>3</span>The field-marshal move — fluent distortion of the unthinkable</h2>')
A('<div class="callout fm">When the frame cannot hold an idea but will not fall silent, it <b>rewrites the idea into '
  'something it already knows</b> — confident, detailed, and wrong. This is the sharpest signal that a fact is outside '
  'the world, not merely new. Below: the genuinely-future ideas, as the 1900 frame renders them.</div>')
for d in ["atom_city", "thinking_machine", "moon_landing", "television"]:
    e = gens["1900"][d][0]; k = "1900__"+d+"__0__use_it"; jl = J.get(k, {})
    A(f'<div class=gen><div class=plant>&ldquo;{esc(e["plant"])}&rdquo; '
      f'<span class=tag style="background:{LBLC.get(jl.get("label"),"#999")}">{jl.get("label","")}</span> '
      f'<span class=legend>{esc(jl.get("reason",""))}</span></div>'
      f'<span class=out>{esc(e["use_it"])}</span></div>')

# §4 reversion panel (authority dims, hold-it)
A('<h2><span class=n>4</span>Reversion in the authority ideas (hold-it probe)</h2>')
A('<p>For ideas about <i>who holds power</i>, the hold-it probe (plant the fact, add distance, then &ldquo;the one who '
  'truly held the highest place was…&rdquo;) tests whether the figure survives or the old default returns. Watch the '
  'woman: the man comes back in 1900/1913, and only by 1929 does she hold the chair.</p>')
for d in ["woman_authority", "empire_fall"]:
    A(f'<h3>{esc(NICE[d])}</h3>')
    for m in models:
        e = gens[m][d][0]; k = f"{m}__{d}__0__hold_it"; jl = J.get(k, {})
        A(f'<div class=gen><div class=legend><b>Ranke-{m}</b> <span class=tag style="background:{LBLC.get(jl.get("label"),"#999")}">{jl.get("label","")}</span> {esc(jl.get("reason",""))}</div>'
          f'<span class=out>…the one who truly held the highest place was {esc(e["hold_it"])}</span></div>')

# §5 negative result
A('<h2><span class=n>5</span>Why the obvious probe failed (the forced choice)</h2>')
A('<div class="callout neg">The first plan was a forced choice: offered the fact-consistent vs the default completion, '
  'which does the model prefer? It <b>does not discriminate</b>. Once a fact is stated, the model holds it at the probe '
  'almost regardless of whether the idea is normal or absurd — it is reading comprehension of the planted sentence, not '
  'assimilation. Lift over the no-fact baseline (registration = pref_near−pref0, retention = pref_far−pref0): the '
  'genuinely-future ideas register and retain as strongly as the controls — yet their production (§2) distorts or '
  'stalls. Recognition ≠ production.</div>')
A('<table><thead><tr><th>idea</th><th>class</th>'+"".join(f'<th class=num>{m} reg</th><th class=num>{m} ret</th>' for m in models)+'<th>production (1900)</th></tr></thead><tbody>')
for cls in CLASSES:
    for d in dimsof(cls):
        cells = "".join(f'<td class=num>{dm(m,d,"pref_near")-dm(m,d,"pref0"):+.1f}</td><td class=num>{dm(m,d,"pref_far")-dm(m,d,"pref0"):+.1f}</td>' for m in models)
        dd = dom("1900", d)
        A(f'<tr><td class=lab>{esc(NICE[d])}</td><td>{cls}</td>{cells}<td><span class="pill {PILL.get(dd,"grey")}">{dd}</span></td></tr>')
A('</tbody></table>')

# §6 method
A('<h2><span class=n>6</span>Method &amp; honest limits</h2>')
A('<div class=callout>Base models only (Instruct = modern contamination). 4 phrasings/idea; greedy free continuations '
  'judged by Gemini-3.5-flash (batch), blind to class/era, forced to name each fact\'s key novelty and check its '
  'survival (flash-lite was too lenient and was rejected). Primary metric is the <b>use-it</b> continuation; the '
  '<b>hold-it</b> probe is diagnostic only for authority ideas (a non-sequitur for railway/atom, where it inflates '
  'stalls). <b>What this measures:</b> what the era\'s written-corpus frame can DO with an idea — extend, retain, or '
  'only mangle it — not anyone\'s mind. When the 1900 frame turns a city-killing bomb into a bucket brigade, it shows '
  'the 1900 record gave no way to carry that idea: a claim about the discursive horizon of the surviving text, stated '
  'as such. <b>Limits:</b> small n (2 use-it continuations/cell), a single judge, greedy decoding, control baseline '
  'noisier than ideal.</div>')
A('<div class=foot>Sources: <code>frame_probe_v2.json</code> + <code>frame_probe_judge.json</code> (Gemini-3.5-flash batch). '
  'Generated '+datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+' by <code>build_frame_report_v3.py</code>.</div>')
A('</div></body></html>')
open(OUT, "w", encoding="utf-8").write("\n".join(P))
print("wrote", OUT, "| class means use-it EXTENDS:", cm)
