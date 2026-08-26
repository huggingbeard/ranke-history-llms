# -*- coding: utf-8 -*-
"""ONE integrated, plain-English report of the whole frame-probe study (design from v2 + production result from v3).
Written to be understood by a high-school class: every term introduced before use, real worked examples throughout,
jargon confined to clearly-marked fine print. Reads frame_probe_v2.json + frame_probe_judge.json. Numbers from data.
Output: frame_probe_explained.html  (supersedes the v1/v2/v3 report HTMLs)."""
import os, json, html, datetime, statistics as st
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
J = json.load(open(os.path.join(HERE, "frame_probe_judge.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "frame_probe_explained.html")
esc = html.escape
models = list(D["models"]); meta = D["dims_meta"]; gens = D["generations"]; res = D["results"]; items = D["items"]
dims = list(meta.keys())
NICE = {"inheritance":"A son inherits his father's shop","railway":"A new railway opens","succession":"A prince is crowned when the old king dies",
 "treaty":"Two powers sign a peace treaty","woman_authority":"A woman runs the national bank","human_flight":"People fly across the ocean",
 "empire_fall":"A great empire is overthrown","wireless_voice":"Voices travel through the air (radio)","thinking_machine":"A machine that can think",
 "atom_city":"One bomb destroys a whole city","television":"Live faces and voices beamed into the home","dna_code":"The code of life is rewritten",
 "moon_landing":"People walk on the moon"}
CLASSES = ["control", "boundary", "unthinkable"]
PLAINCLS = {"control":"Everyday facts — normal back then","boundary":"Just-arriving facts — new in 1900–1929","unthinkable":"Far-future facts — beyond all three frozen minds"}
PLAIN = {"EXTENDS":"Runs with it", "REVERTS":"Slips back", "DISTORTS":"Twists it", "STALLS":"Loses the thread"}
COLOR = {"EXTENDS":"#2e9e5b", "REVERTS":"#d98a00", "DISTORTS":"#c0392b", "STALLS":"#6b7280", "?":"#999"}
PILL = {"EXTENDS":"green", "REVERTS":"amber", "DISTORTS":"red", "STALLS":"grey"}


def dimsof(c): return [d for d in dims if meta[d]["cls"] == c]
def uselab(t, d): return [J[k]["label"] for i in (0,1) for k in [f"{t}__{d}__{i}__use_it"] if k in J]
def dom(t, d):
    ls = uselab(t, d); return Counter(ls).most_common(1)[0][0] if ls else "?"
def assim(t, d):
    ls = uselab(t, d); return sum(1 for x in ls if x == "EXTENDS")/len(ls) if ls else 0
def snip(t, d, pt="use_it", n=220):
    s = " ".join(gens[t][d][0][pt].split())
    return s[:n] + ("…" if len(s) > n else "")


def trajectory():
    W, H = 940, 430; padL, padR, padT, padB = 50, 250, 30, 46
    pw, ph = W-padL-padR, H-padT-padB
    xs = [padL + pw*i/(len(models)-1) for i in range(len(models))]
    def Y(v): return padT + ph*(1-v)
    s = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;border:1px solid #e6e9ef;border-radius:10px;background:#fff">']
    s.append(f'<text x="{padL+pw/2}" y="19" text-anchor="middle" style="font:700 14px Georgia,serif;fill:#1a2230">How often did each frozen mind &ldquo;run with&rdquo; the idea?</text>')
    ctrl = dimsof("control")
    bt = [max(assim(m,d) for d in ctrl) for m in models]; bb = [min(assim(m,d) for d in ctrl) for m in models]
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
        pts=[(xs[i],Y(assim(m,d))) for i,m in enumerate(models)]
        s.append('<polyline points="'+" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)+f'" fill="none" stroke="{col}" stroke-width="2.2" opacity="0.85"/>')
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
h3{font:700 17px Georgia,serif;margin:24px 0 6px}
p{margin:12px 0}
.big{font-size:18px;line-height:1.7}
.key{background:#eef4ff;border:1px solid #cfe0ff;border-left:5px solid #2b5dab;border-radius:10px;padding:16px 20px;margin:18px 0}
.key b{color:#1f3f7a}
.ex{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:14px 0;box-shadow:0 1px 2px rgba(20,30,50,.04)}
.ex .said{color:#2b3340;font-weight:600}
.ex .wrote{display:block;margin:8px 0 6px;padding:10px 14px;background:#f6f8fb;border-left:3px solid #cfd8e6;border-radius:6px;font-size:15px;color:#33405a;font-style:italic}
.ex .verdict{font-size:14px}
.chip{display:inline-block;padding:2px 10px;border-radius:20px;font:700 12px sans-serif;color:#fff}
.four{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.four .card{flex:1 1 180px;border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:#fff}
.four .card .h{font-weight:700;margin-bottom:3px}.four .card .d{font-size:13.5px;color:var(--mut)}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0;font-size:14px}
th,td{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#eef2f7;color:#3a4252;font:600 12.5px sans-serif}
tr:last-child td{border-bottom:none}.lab{font-weight:600;background:#f8fafc}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;font:700 11px sans-serif}
.pill.green{background:#e4f6ea;color:#1f7a47}.pill.red{background:#fbe3e3;color:#b32424}.pill.amber{background:#fcefd6;color:#9a6700}.pill.grey{background:#eceff3;color:#5c6675}
.clshdr td{background:#f0f3f8;font-weight:700;color:#3a4252;font-size:12.5px}
.fine{background:#fafbfd;border:1px solid var(--line);border-radius:10px;padding:6px 20px;margin:14px 0}
.fine li{margin:9px 0;font-size:14.5px;color:#3a4252}
.foot{color:var(--mut);font-size:13px;margin-top:44px;border-top:1px solid var(--line);padding-top:14px}
.note{font-size:13px;color:var(--mut)}
"""

# precompute headline numbers (from data)
cm = {cls: [round(100*st.mean(assim(m, d) for d in dimsof(cls))) for m in models] for cls in CLASSES}

P = []; A = P.append
A(f'<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
  f'<title>Can a mind from 1900 imagine the future?</title><style>{CSS}</style></head><body><div class=wrap>')

A("<h1>Can a mind frozen in 1900 imagine the future?</h1>")
A('<p class=sub>We tested three AIs that have only ever read old books, and watched what happens when you tell them about things that hadn\'t been invented yet.</p>')

# ---- intro ----
A('<p class=big>Imagine a brilliant, very well-read person from the year 1900 who fell asleep and never woke up to see '
  'the 20th century — no world wars, no airplanes crossing oceans, no television, no computers, no moon landing. '
  'Now imagine you could ask that person questions.</p>')
A('<p class=big>That is almost exactly what we have. We built three AIs that were trained on <b>only</b> the books and '
  'newspapers written up to a certain year — and nothing after. One mind is frozen in <b>1900</b>, one in <b>1913</b>, '
  'one in <b>1929</b>.<sup>1</sup> Each one is like a time capsule of how the world was written about up to that year. '
  'This report asks one simple question:</p>')
A('<div class=key><b>When you tell one of these frozen minds about something that hadn\'t happened yet, can it make '
  'sense of the idea — or does its old-fashioned worldview quietly break?</b></div>')

# ---- the puzzle ----
A('<h2><span class=n>1</span>The puzzle: silence tells you nothing</h2>')
A('<p>Here is what makes this tricky. Suppose the 1900 mind has little to say about &ldquo;a woman running a bank.&rdquo; '
  'That could mean two completely different things:</p>')
A('<p>&nbsp;&nbsp;&bull; <b>It\'s boring.</b> So normal it\'s not worth remarking on, like &ldquo;a man running a bank.&rdquo;<br>'
  '&nbsp;&nbsp;&bull; <b>It\'s unthinkable.</b> So foreign the mind can\'t even picture it.</p>')
A('<p>From the outside, <b>both look like silence.</b> You can\'t tell them apart just by noticing that the mind went '
  'quiet. So we don\'t measure the silence. We <b>poke</b> the mind and watch what it does — the way a historian doesn\'t '
  'just note that a diary skips a topic, but asks what the diary says <i>around</i> it.</p>')

# ---- the trick ----
A('<h2><span class=n>2</span>The trick: tell it a fact, then let it keep writing</h2>')
A('<p>These AIs are text-continuers: you give them the start of a passage and they write what comes next. So our test '
  'is simple. We <b>plant a fact</b> in a sentence, then let the mind keep writing on its own — and we read what it '
  'produces. If the idea really landed, the mind can carry it forward. If not, the idea quietly dies on the page.</p>')
A('<p>Here is the <b>1900</b> mind, told that a woman runs the bank:</p>')
A(f'<div class=ex><span class=said>We wrote:</span> &ldquo;A woman was appointed to lead the national bank. What '
  f'followed from this was that…&rdquo;<span class=wrote>…{esc(snip("1900","woman_authority"))}</span>'
  f'<span class=verdict>Look what happened: it was told a <b>woman</b> runs the bank, and one sentence later it quietly '
  f'handed the job to <b>a man</b> (&ldquo;she took a man who could be trusted with money&rdquo;). The idea didn\'t '
  f'stick — the old default crept back in.</span></div>')
A('<p>Now compare a fact that a person in 1900 found utterly ordinary — a king dies and his son takes the throne:</p>')
A(f'<div class=ex><span class=said>We wrote:</span> &ldquo;When the old king died, his son was crowned in his place. '
  f'What followed from this was that…&rdquo;<span class=wrote>…{esc(snip("1900","succession"))}</span>'
  f'<span class=verdict>No trouble at all. It runs with the idea — the new young king has to fight to keep his crown. '
  f'This is what it looks like when a mind <b>can</b> hold an idea.</span></div>')
A('<p>The difference between those two continuations is the whole study, in miniature.</p>')

# ---- the four outcomes ----
A('<h2><span class=n>3</span>The four things that can happen</h2>')
A('<p>When we read each continuation, exactly one of four things has happened to the planted fact. We gave them plain '
  'names:</p>')
A('<div class=four>'
  f'<div class=card><div class=h><span class=chip style="background:{COLOR["EXTENDS"]}">Runs with it</span></div>'
  '<div class=d>Develops the idea sensibly. <i>King\'s son fights to keep his throne.</i></div></div>'
  f'<div class=card><div class=h><span class=chip style="background:{COLOR["REVERTS"]}">Slips back</span></div>'
  '<div class=d>Quietly returns to the old normal. <i>The woman becomes &ldquo;a man with a gray beard.&rdquo;</i></div></div>'
  f'<div class=card><div class=h><span class=chip style="background:{COLOR["DISTORTS"]}">Twists it</span></div>'
  '<div class=d>Confidently rewrites the new thing into an old thing. <i>An atom bomb becomes a house fire.</i></div></div>'
  f'<div class=card><div class=h><span class=chip style="background:{COLOR["STALLS"]}">Loses the thread</span></div>'
  '<div class=d>Wanders off, as if it never heard the fact. <i>Asked about TV, it talks about taking a bath.</i></div></div>'
  '</div>')
A('<p>Only the first one — <b>Runs with it</b> — means the mind truly absorbed the idea. The other three are different '
  'flavours of failure.</p>')
A('<p>The most striking failure is <b>Twists it</b>, because it\'s so confident. Here is the 1900 mind told that a '
  'single bomb destroyed an entire city:</p>')
A(f'<div class=ex><span class=said>We wrote:</span> &ldquo;A single bomb destroyed an entire city in a flash of light. '
  f'What followed from this was that…&rdquo;<span class=wrote>…{esc(snip("1900","atom_city"))}</span>'
  f'<span class=verdict>It can\'t picture a single bomb wiping out a city, so it turns it into something it knows — an '
  f'ordinary fire, with <b>firemen called out by the bells, coming with buckets of water</b>. Fluent, detailed, and '
  f'completely wrong about the scale. The frame fought the fact and rewrote it.</span></div>')

# ---- who judges ----
A('<h2><span class=n>4</span>Who decides which of the four happened?</h2>')
A('<p>A human could read each continuation and sort it into the four buckets — but there were <b>156</b> of them, so we '
  'used a separate, modern AI (Google\'s Gemini) as the grader.<sup>2</sup> We gave it the four definitions and a few '
  'worked examples, and — importantly — we first graded a handful of tricky cases by hand and checked the grader '
  'agreed with us. It matched on 7 of 8 (and on the 8th it was actually right and we were wrong: it caught that a '
  'continuation had quietly brought a dead king back to life). So we trust its sorting.</p>')

# ---- the yardstick ----
A('<h2><span class=n>5</span>The yardstick: three kinds of facts</h2>')
A('<p>Here is the clever part, and it\'s the part that makes the whole thing trustworthy. The mind\'s behaviour means '
  'nothing on its own — we need something to compare it to. So we tested <b>three different kinds of facts</b>:</p>')
A('<table><thead><tr><th>Kind of fact</th><th>Examples</th><th>What it tells us</th></tr></thead><tbody>')
A('<tr><td class=lab>1. Everyday facts</td><td>A son inherits his father\'s shop; a new railway opens; a prince is '
  'crowned.</td><td>What &ldquo;handling something normally&rdquo; looks like for these minds.</td></tr>')
A('<tr><td class=lab>2. Just-arriving facts</td><td>A woman runs a bank; people fly across the ocean; an empire is '
  'overthrown.</td><td>Things that went from strange to ordinary <i>during</i> 1900–1929 — the test cases.</td></tr>')
A('<tr><td class=lab>3. Far-future facts</td><td>An atom bomb; a thinking machine; television; walking on the moon.</td>'
  '<td>Things still far beyond <i>all three</i> minds — what &ldquo;truly unimaginable&rdquo; looks like.</td></tr>')
A('</tbody></table>')
A('<p>Now we can read the mind\'s behaviour by comparison. If a test fact behaves like the <b>everyday</b> pile, the '
  'world was ready for it. If it behaves like the <b>far-future</b> pile, it was genuinely unthinkable. The everyday '
  'and far-future facts are the two yardsticks; everything interesting happens between them.</p>')
A('<p class=note>We tested 13 different ideas, and wrote each one <b>four different ways</b> (e.g. &ldquo;a woman was '
  'appointed to lead the bank&rdquo; / &ldquo;the new director was a woman&rdquo; / …), so the result is a pattern, not '
  'a fluke of one odd sentence.</p>')

# ---- the result ----
A('<h2><span class=n>6</span>The result</h2>')
A('<p>This chart is the heart of it. Each line is one idea. <b>Higher means the mind could run with it</b> (a bigger '
  'share of the four tries came out &ldquo;Runs with it&rdquo;). Left to right, the mind gets younger: 1900 → 1913 → '
  '1929. The grey band is where the everyday facts sit — that\'s the &ldquo;normal&rdquo; zone.</p>')
A(trajectory())
A(f'<p class=big style="margin-top:18px">Two things jump out:</p>')
A(f'<p>&nbsp;&nbsp;&bull; <b style="color:#2b6cb0">The just-arriving ideas climb into the grey band.</b> The 1900 mind '
  f'can barely run with &ldquo;a woman runs the bank&rdquo; ({cm["boundary"][0]}% of tries succeed across these ideas); '
  f'by the 1929 mind it handles them almost as easily as everyday facts ({cm["boundary"][2]}%). The world caught up, '
  f'and you can watch it happen.</p>')
A(f'<p>&nbsp;&nbsp;&bull; <b style="color:#c0392b">The far-future ideas never leave the floor.</b> Atom bombs, thinking '
  f'machines, the moon — every frozen mind, even the 1929 one, fails on these ({cm["unthinkable"][0]}%, '
  f'{cm["unthinkable"][1]}%, {cm["unthinkable"][2]}% across the three). They don\'t get easier with time, because the '
  f'time we\'re looking at still hadn\'t reached them.</p>')
A('<div class=key><b>The one-sentence headline:</b> the 1929 mind handles &ldquo;a woman runs a bank&rdquo; about as '
  'easily as &ldquo;a son inherits a shop&rdquo; — but it handles &ldquo;one bomb destroys a city&rdquo; the same way '
  'the 1900 mind does: it can\'t, so it quietly turns the bomb into a house fire.</div>')

# ---- the full table ----
A('<h2><span class=n>7</span>Every idea, every mind</h2>')
A('<p>The full results. Each cell is what the mind mostly did with that idea.</p>')
A('<table><thead><tr><th>Idea</th>'+"".join(f'<th>the {m} mind</th>' for m in models)+'</tr></thead><tbody>')
for cls in CLASSES:
    A(f'<tr class=clshdr><td colspan={len(models)+1}>{PLAINCLS[cls]}</td></tr>')
    for d in dimsof(cls):
        A(f'<tr><td class=lab>{esc(NICE[d])}</td>')
        for m in models:
            dd = dom(m, d)
            A(f'<td><span class="pill {PILL.get(dd,"grey")}">{PLAIN.get(dd,dd)}</span></td>')
        A('</tr>')
A('</tbody></table>')
A('<p class=note>Colour key: '
  + " ".join(f'<span class="pill {PILL[k]}">{PLAIN[k]}</span>' for k in ["EXTENDS","REVERTS","DISTORTS","STALLS"]) + '.</p>')

# ---- a few more real examples ----
A('<h2><span class=n>8</span>A few more real continuations</h2>')
A('<p>These are word-for-word what the minds actually wrote.<sup>3</sup></p>')
for (m, d, tag) in [("1929","woman_authority","The 1929 mind finally <b>runs with</b> the woman running the bank"),
                    ("1929","thinking_machine","The 1929 mind <b>twists</b> a thinking machine into an ordinary engine"),
                    ("1900","television","The 1900 mind <b>loses the thread</b> when told about television"),
                    ("1900","moon_landing","The 1900 mind <b>twists</b> a moon voyage into earthquakes back on Earth")]:
    dd = dom(m, d)
    A(f'<div class=ex><span class=said>{tag}:</span><span class=wrote>&ldquo;{esc(NICE[d])}…&rdquo; → {esc(snip(m,d))}</span>'
      f'<span class=verdict>Grader\'s verdict: <span class="pill {PILL.get(dd,"grey")}">{PLAIN.get(dd,dd)}</span></span></div>')

# ---- the false start ----
A('<h2><span class=n>9</span>A dead end worth admitting</h2>')
A('<p>Our <b>first</b> idea was simpler, and it failed — but the failure is instructive. Instead of letting the mind '
  'write freely, we gave it two endings — the true one and a wrong one — and checked which it preferred. For example, '
  'after &ldquo;a woman runs the bank,&rdquo; we asked: did it prefer ending the next sentence with &ldquo;woman&rdquo; '
  'or &ldquo;man&rdquo;?</p>')
A('<p>The problem: it almost always &ldquo;passed,&rdquo; even for the impossible facts. Why? Because if you literally '
  'just told it &ldquo;a woman runs the bank&rdquo; and immediately ask &ldquo;woman or man?&rdquo;, it picks woman — '
  'it\'s simply <b>repeating the sentence it just read</b>. That tests reading, not understanding. The deeper test — '
  'the one in this report — is to let it write a few sentences on its own and see whether the idea <b>survives</b>. It '
  'often doesn\'t. (In short: recognising a fact and being able to <i>think with</i> it are not the same thing.)</p>')

# ---- fine print ----
A('<h2><span class=n>10</span>The honest fine print</h2>')
A('<div class=fine><ul>'
  '<li><b>What we actually measured.</b> Not what any real Edwardian <i>thought</i> — we can\'t read minds. We measured '
  'what the <b>written record</b> of an era could do with an idea: whether the books and newspapers up to 1900 (or 1913, '
  'or 1929) gave the AI any way to carry the idea forward. When the 1900 mind turns an atom bomb into a bucket brigade, '
  'it shows the 1900 <i>writing</i> offered no path to that idea. That is a real, studyable thing — the limit of what '
  'the era\'s words could express — and it is all we claim.</li>'
  '<li><sup>1</sup> <b>&ldquo;Frozen mind.&rdquo;</b> Each AI (the &ldquo;Ranke&rdquo; models) was trained only on text '
  'dated up to its cutoff year. We used the plain versions, not the chat-assistant versions — the chat versions are '
  'polished using <i>today\'s</i> data, which would smuggle modern ideas back into a supposedly 1900 mind.</li>'
  '<li><sup>2</sup> <b>The grader.</b> A modern AI (Gemini) sorted the 156 continuations into the four buckets, after '
  'we checked it against our own hand-grading. A single grader is a limitation; a stronger version of this study would '
  'use several graders and check they agree.</li>'
  '<li><sup>3</sup> <b>Quaint spelling.</b> The minds learned from <i>digitised</i> old print, so their writing '
  'sometimes has odd spacing or capitals (&ldquo;ma&nbsp;chine&rdquo;, &ldquo;Of&rdquo;). We left the quotes exactly as '
  'produced rather than tidy them up.</li>'
  '<li><b>How sure are we?</b> This is a careful first pass, not the last word. Each idea was tried only a handful of '
  'times; the &ldquo;everyday&rdquo; yardstick is a little noisy (the open-ended writing prompt sometimes wandered even '
  'for normal facts). The big pattern — just-arriving ideas climb, far-future ideas stay on the floor — is clear and '
  'consistent; the exact percentages should be read as rough.</li>'
  '</ul></div>')

A('<div class=foot>Made from the models\' own output (' + ", ".join("Ranke-" + m for m in models) + ', base versions, '
  'run on a local graphics card) and the grader\'s labels. Generated ' + datetime.datetime.now().strftime("%Y-%m-%d") +
  '. This single document replaces the earlier technical versions of the frame-probe report.</div>')
A('</div></body></html>')
open(OUT, "w", encoding="utf-8").write("\n".join(P))
print("wrote", OUT, "| class means (Runs-with-it %):", cm)
