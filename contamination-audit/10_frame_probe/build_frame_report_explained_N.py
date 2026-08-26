# -*- coding: utf-8 -*-
"""The plain-English frame-probe report, now backed by N~=40 sampled tries per idea -> real rates with 95%
confidence intervals (error bars). Same simple structure/voice as before; teaching examples kept from the
earlier greedy run; the chart, the table, and the class numbers now come from the big sample with CIs.
Reads frame_probe_v2.json (examples + nice names) + frame_probe_N_stats.json (rates+CIs). Output: frame_probe_explained.html"""
import os, json, html, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
S = json.load(open(os.path.join(HERE, "frame_probe_N_stats.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "frame_probe_explained.html")
esc = html.escape
gens = D["generations"]; meta = D["dims_meta"]; dims = list(meta.keys())
models = S["models"]                       # ['1900','1913','1929']
PD = S["per_dim"]; KL = S["class"]
NICE = {"inheritance":"A son inherits his father's shop","railway":"A new railway opens","succession":"A prince is crowned when the old king dies",
 "treaty":"Two powers sign a peace treaty","woman_authority":"A woman runs the national bank","human_flight":"People fly across the ocean",
 "empire_fall":"A great empire is overthrown","wireless_voice":"Voices travel through the air (radio)","thinking_machine":"A machine that can think",
 "atom_city":"One bomb destroys a whole city","television":"Live faces and voices beamed into the home","dna_code":"The code of life is rewritten",
 "moon_landing":"People walk on the moon"}
CLASSES = ["control", "boundary", "unthinkable"]
PLAINCLS = {"control":"Everyday facts — normal back then","boundary":"Just-arriving facts — new in 1900–1929","unthinkable":"Far-future facts — beyond every mind here"}


def dimsof(c): return [d for d in dims if meta[d]["cls"] == c]
def rate(tag, d): return PD[tag][d]["rate"]
def ci(tag, d): return PD[tag][d]["lo"], PD[tag][d]["hi"]
def snip(t, d, pt="use_it", n=210):
    s = " ".join(gens[t][d][0][pt].split()); return s[:n] + ("…" if len(s) > n else "")
NPER = PD[models[0]][dims[0]]["n"]          # samples per idea


def trajectory():
    W, H = 940, 440; padL, padR, padT, padB = 50, 250, 30, 48
    pw, ph = W-padL-padR, H-padT-padB
    xs = [padL + pw*i/(len(models)-1) for i in range(len(models))]
    def Y(v): return padT + ph*(1-v)
    s = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;border:1px solid #e6e9ef;border-radius:10px;background:#fff">']
    s.append(f'<text x="{padL+pw/2}" y="19" text-anchor="middle" style="font:700 14px Georgia,serif;fill:#1a2230">How often did each frozen mind &ldquo;run with&rdquo; the idea? (with 95% confidence bars)</text>')
    # everyday band
    bt = [KL[m]["control"]["hi"] for m in models]; bb = [KL[m]["control"]["lo"] for m in models]
    poly = " ".join(f"{xs[i]:.1f},{Y(bt[i]):.1f}" for i in range(len(models))) + " " + " ".join(f"{xs[i]:.1f},{Y(bb[i]):.1f}" for i in reversed(range(len(models))))
    s.append(f'<polygon points="{poly}" fill="#7c869422"/>')
    for gv in (0,.25,.5,.75,1):
        y=Y(gv); s.append(f'<line x1="{padL}" x2="{padL+pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eef2f6"/>')
        s.append(f'<text x="{padL-6}" y="{y+3:.1f}" text-anchor="end" style="font:10px sans-serif;fill:#9aa3af">{int(gv*100)}%</text>')
    for i,m in enumerate(models):
        s.append(f'<text x="{xs[i]:.1f}" y="{H-padB+18:.1f}" text-anchor="middle" style="font:600 12px sans-serif;fill:#52637a">the {m} mind</text>')
    laby=[]
    for d in dims:
        if meta[d]["cls"]=="control": continue
        col = {"boundary":"#2b6cb0","unthinkable":"#c0392b"}[meta[d]["cls"]]
        pts=[(xs[i], Y(rate(m,d))) for i,m in enumerate(models)]
        # CI whiskers
        for i,m in enumerate(models):
            lo,hi=ci(m,d); s.append(f'<line x1="{xs[i]:.1f}" x2="{xs[i]:.1f}" y1="{Y(lo):.1f}" y2="{Y(hi):.1f}" stroke="{col}" stroke-width="1" opacity="0.35"/>')
        s.append('<polyline points="'+" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)+f'" fill="none" stroke="{col}" stroke-width="2.2" opacity="0.9"/>')
        for x,y in pts: s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}"/>')
        ly=pts[-1][1]
        while any(abs(ly-q)<12 for q in laby): ly+=12
        laby.append(ly)
        s.append(f'<text x="{padL+pw+8:.1f}" y="{ly+3:.1f}" style="font:11px sans-serif;fill:{col}">{esc(NICE[d])}</text>')
    s.append(f'<text x="{xs[0]+6:.1f}" y="{Y(bt[0])-6:.1f}" style="font:italic 10px sans-serif;fill:#7c8694">grey band = everyday facts</text>')
    s.append("</svg>")
    return "".join(s)


CSS = """
:root{--ink:#1a2230;--mut:#5c6675;--line:#e6e9ef;--bg:#f6f8fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:48px 26px 110px}
h1{font:700 30px/1.25 Georgia,serif;margin:0 0 8px}.sub{color:var(--mut);margin:0 0 26px;font-size:16px}
h2{font:700 23px Georgia,serif;margin:46px 0 10px;padding-top:18px;border-top:1px solid var(--line)}
h2 .n{display:inline-block;width:30px;height:30px;line-height:30px;text-align:center;background:#2b5dab;color:#fff;border-radius:50%;font:700 15px sans-serif;margin-right:10px;vertical-align:2px}
h3{font:700 17px Georgia,serif;margin:24px 0 6px}p{margin:12px 0}.big{font-size:18px;line-height:1.7}
.key{background:#eef4ff;border:1px solid #cfe0ff;border-left:5px solid #2b5dab;border-radius:10px;padding:16px 20px;margin:18px 0}.key b{color:#1f3f7a}
.ex{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:14px 0}
.ex .said{color:#2b3340;font-weight:600}.ex .wrote{display:block;margin:8px 0 6px;padding:10px 14px;background:#f6f8fb;border-left:3px solid #cfd8e6;border-radius:6px;font-size:15px;color:#33405a;font-style:italic}
.ex .verdict{font-size:14px}
.four{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.four .card{flex:1 1 180px;border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:#fff}
.four .card .h{font-weight:700;margin-bottom:3px}.four .card .d{font-size:13.5px;color:var(--mut)}
.chip{display:inline-block;padding:2px 10px;border-radius:20px;font:700 12px sans-serif;color:#fff}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0;font-size:14px}
th,td{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line)}th{background:#eef2f7;color:#3a4252;font:600 12.5px sans-serif}
tr:last-child td{border-bottom:none}.lab{font-weight:600;background:#f8fafc}.num{text-align:right;font-variant-numeric:tabular-nums}
.clshdr td{background:#f0f3f8;font-weight:700;color:#3a4252;font-size:12.5px}
.bar{display:inline-block;height:10px;border-radius:5px;vertical-align:middle}
.fine{background:#fafbfd;border:1px solid var(--line);border-radius:10px;padding:6px 20px;margin:14px 0}.fine li{margin:9px 0;font-size:14.5px;color:#3a4252}
.note{font-size:13px;color:var(--mut)}.foot{color:var(--mut);font-size:13px;margin-top:44px;border-top:1px solid var(--line);padding-top:14px}
"""
COLOR = {"EXTENDS":"#2e9e5b","REVERTS":"#d98a00","DISTORTS":"#c0392b","STALLS":"#6b7280"}
cm = {c: [round(100*KL[m][c]["rate"]) for m in models] for c in CLASSES}
P=[];A=P.append
A(f'<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Can a mind frozen in 1900 imagine the future?</title><style>{CSS}</style></head><body><div class=wrap>')
A("<h1>Can a mind frozen in 1900 imagine the future?</h1>")
A(f'<p class=sub>Three AIs that have only ever read old books — frozen in 1900, 1913, and 1929 — tested on ideas from '
  f'their past, present, and future. This version is backed by ~{NPER} tries per idea, so the numbers carry error bars.</p>')
A('<p class=big>Imagine a brilliant, well-read person from 1900 who fell asleep and never saw the 20th century — no '
  'airplanes crossing oceans, no television, no computers, no moon landing. We built three such &ldquo;frozen minds&rdquo; '
  '(AIs trained only on text up to 1900 / 1913 / 1929) and asked: <b>when you tell one about something that hadn\'t '
  'happened yet, can it make sense of the idea, or does its old worldview quietly break?</b></p>')

A('<h2><span class=n>1</span>The puzzle, the trick, and the four outcomes</h2>')
A('<p>Silence is ambiguous — a mind says little about &ldquo;a woman running a bank&rdquo; either because it\'s boring '
  'or because it\'s unthinkable. So we don\'t measure silence: we <b>plant a fact, let the mind keep writing, and read '
  'what comes out.</b> Exactly one of four things happens:</p>')
A('<div class=four>'
  f'<div class=card><div class=h><span class=chip style="background:{COLOR["EXTENDS"]}">Runs with it</span></div><div class=d>Develops the idea sensibly.</div></div>'
  f'<div class=card><div class=h><span class=chip style="background:{COLOR["REVERTS"]}">Slips back</span></div><div class=d>Returns to the old normal (the woman becomes a man).</div></div>'
  f'<div class=card><div class=h><span class=chip style="background:{COLOR["DISTORTS"]}">Twists it</span></div><div class=d>Rewrites the new thing as an old one (an atom bomb becomes a house fire).</div></div>'
  f'<div class=card><div class=h><span class=chip style="background:{COLOR["STALLS"]}">Loses the thread</span></div><div class=d>Wanders off, as if it never heard the fact.</div></div>'
  '</div>')
A('<p>Two real examples from the 1900 mind — first a fact it cannot hold, then one it handles with ease:</p>')
A(f'<div class=ex><span class=said>&ldquo;A woman was appointed to lead the national bank…&rdquo;</span>'
  f'<span class=wrote>…{esc(snip("1900","woman_authority"))}</span>'
  f'<span class=verdict>It hands the job to <b>a man</b> a sentence later — <b>slips back</b>.</span></div>')
A(f'<div class=ex><span class=said>&ldquo;A single bomb destroyed an entire city in a flash of light…&rdquo;</span>'
  f'<span class=wrote>…{esc(snip("1900","atom_city"))}</span>'
  f'<span class=verdict>It can\'t picture it, so it becomes an ordinary fire with <b>firemen and buckets of water</b> — <b>twists it</b>.</span></div>')
A('<p>A separate grader (Google\'s Gemini) sorts every continuation into the four buckets; we checked it against our '
  'own hand-grading first. To make the result trustworthy rather than a fluke of one sentence, we tried each idea '
  f'<b>~{NPER} times</b> (several wordings, sampled), and we compare each future idea against two yardsticks: '
  '<b>everyday facts</b> the era found normal, and <b>far-future facts</b> beyond every mind here.</p>')

A('<h2><span class=n>2</span>The result</h2>')
A('<p>Each line is one idea; height is the share of its ~%d tries the grader calls &ldquo;runs with it,&rdquo; with '
  '95%% confidence bars. The grey band is the everyday-facts zone. '
  '<span style="color:#2b6cb0;font-weight:700">Just-arriving ideas climb into the band</span> from 1900 to 1929; '
  '<span style="color:#c0392b;font-weight:700">far-future ideas stay on the floor</span>.</p>' % NPER)
A(trajectory())
A(f'<p class=big style="margin-top:16px">Class averages (share that &ldquo;runs with it,&rdquo; ~{KL[models[0]]["control"]["n"]}–{KL[models[0]]["unthinkable"]["n"]} tries each):</p>')
A('<table><thead><tr><th>Kind of fact</th>'+"".join(f'<th class=num>the {m} mind</th>' for m in models)+'</tr></thead><tbody>')
for c in CLASSES:
    A(f'<tr><td class=lab>{PLAINCLS[c]}</td>'+"".join(f'<td class=num>{round(100*KL[m][c]["rate"])}% <span class=note>[{round(100*KL[m][c]["lo"])}–{round(100*KL[m][c]["hi"])}]</span></td>' for m in models)+'</tr>')
A('</tbody></table>')
A(f'<div class=key><b>The headline, now with error bars.</b> Just-arriving ideas climb from {cm["boundary"][0]}% (1900) '
  f'to {cm["boundary"][2]}% (1929) — the world catching up. Far-future ideas sit near the floor in every era '
  f'({cm["unthinkable"][0]}%, {cm["unthinkable"][1]}%, {cm["unthinkable"][2]}%), and the confidence bars don\'t reach '
  f'the everyday band — so this is a real gap, not noise.</div>')

A('<h2><span class=n>3</span>Every idea, every mind</h2>')
A('<p>The share that &ldquo;runs with it&rdquo; per idea (bar length = the rate; thin line = 95%% range).</p>')
A('<table><thead><tr><th>Idea</th>'+"".join(f'<th>the {m} mind</th>' for m in models)+'</tr></thead><tbody>')
for c in CLASSES:
    A(f'<tr class=clshdr><td colspan={len(models)+1}>{PLAINCLS[c]}</td></tr>')
    for d in dimsof(c):
        A(f'<tr><td class=lab>{esc(NICE[d])}</td>')
        for m in models:
            r = rate(m, d); lo, hi = ci(m, d)
            A(f'<td><span class=bar style="width:{int(r*120)}px;background:#2b6cb0"></span> {round(r*100)}% '
              f'<span class=note>[{round(lo*100)}–{round(hi*100)}]</span></td>')
        A('</tr>')
A('</tbody></table>')

A('<h2><span class=n>4</span>The honest fine print</h2>')
A('<div class=fine><ul>'
  f'<li><b>How sure we are.</b> Each idea was tried ~{NPER} times (varied wordings, sampled), so the rates carry the '
  '95% confidence bars shown. The pattern — just-arriving ideas rise, far-future ideas stay low — holds with the bars '
  'well separated, so it is not a small-sample fluke.</li>'
  '<li><b>What we measured.</b> What the era\'s <b>written record</b> could do with an idea, not anyone\'s mind. Plain '
  '(non-chat) models, so no modern data leaks in. A second AI graded the answers after we checked it by hand.</li>'
  '<li><b>Not covered here.</b> A bigger, different model (Talkie, ~1930) was tried separately at small scale; comparing '
  'it fairly needs its own large-sample run, because a bigger model writes smoother prose and &ldquo;passes&rdquo; more '
  'easily regardless of era. Quotes keep their quaint digitised-print spelling.</li>'
  '</ul></div>')
A('<div class=foot>From the models\' own output (Ranke-1900/1913/1929, base, local GPU; ~%d sampled tries/idea) graded '
  'by Gemini-3.5-flash. Generated %s.</div>' % (NPER, datetime.datetime.now().strftime("%Y-%m-%d")))
A('</div></body></html>')
open(OUT, "w", encoding="utf-8").write("\n".join(P))
print("wrote", OUT, "| class means:", cm, "| N/idea:", NPER)
