# -*- coding: utf-8 -*-
"""Honest plain-English companion: we ran the frame probe on a second, very different AI (Talkie-1930-13B) and compared
it to our 1929 mind. The raw comparison is confounded (a bigger AI writes smoother text and 'passes' more easily) and
noisy (2 tries per idea), so this report leads with that, then states only what survives a careful reading.
Reads frame_probe_v2.json, frame_probe_judge.json (Ranke), talkie_frame_gens.json, talkie_frame_judge.json (Talkie).
Output: frame_probe_explained_v2.html"""
import os, json, html, datetime, statistics as st
from collections import Counter
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
J = json.load(open(os.path.join(HERE, "frame_probe_judge.json"), encoding="utf-8"))
TG = json.load(open(os.path.join(HERE, "talkie_frame_gens.json"), encoding="utf-8"))["generations"]
TJ = json.load(open(os.path.join(HERE, "talkie_frame_judge.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "frame_probe_explained_v2.html")
esc = html.escape
meta = D["dims_meta"]; gens = D["generations"]; dims = list(meta.keys()); TKM = "talkie-1930-13b-base"
NICE = {"inheritance":"A son inherits his father's shop","railway":"A new railway opens","succession":"A prince is crowned when the old king dies",
 "treaty":"Two powers sign a peace treaty","woman_authority":"A woman runs the national bank","human_flight":"People fly across the ocean",
 "empire_fall":"A great empire is overthrown","wireless_voice":"Voices travel through the air (radio)","thinking_machine":"A machine that can think",
 "atom_city":"One bomb destroys a whole city","television":"Live faces and voices beamed into the home","dna_code":"The code of life is rewritten",
 "moon_landing":"People walk on the moon"}
CLASSES = ["control", "boundary", "unthinkable"]
PLAINCLS = {"control":"Everyday facts — normal back then","boundary":"Just-arriving facts — new in 1900–1929","unthinkable":"Far-future facts — beyond every mind here"}
PLAIN = {"EXTENDS":"Runs with it", "REVERTS":"Slips back", "DISTORTS":"Twists it", "STALLS":"Loses the thread", "?":"—", "ERR":"—"}
PILL = {"EXTENDS":"green", "REVERTS":"amber", "DISTORTS":"red", "STALLS":"grey", "?":"grey", "ERR":"grey"}


def dimsof(c): return [d for d in dims if meta[d]["cls"] == c]
def rk_use(d): return [J[k]["label"] for i in (0,1) for k in [f"1929__{d}__{i}__use_it"] if k in J]
def tk_use(d): return [TJ[k]["label"] for i in (0,1) for k in [f"{TKM}__{d}__{i}__use_it"] if k in TJ]
def rk_dom(d):
    ls = rk_use(d); return Counter(ls).most_common(1)[0][0] if ls else "?"
def tk_dom(d):
    ls = tk_use(d); return Counter(ls).most_common(1)[0][0] if ls else "?"
def frac(ls): return sum(1 for x in ls if x == "EXTENDS")/len(ls) if ls else 0
def tk_snip(d, pt="use_it", n=230):
    s = " ".join(TG.get(f"{TKM}__{d}__0__{pt}", "").split()); return s[:n] + ("…" if len(s) > n else "")

rk_cls = {c: round(100*st.mean(frac(rk_use(d)) for d in dimsof(c))) for c in CLASSES}
tk_cls = {c: round(100*st.mean(frac(tk_use(d)) for d in dimsof(c))) for c in CLASSES}

CSS = """
:root{--ink:#1a2230;--mut:#5c6675;--line:#e6e9ef;--bg:#f6f8fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:48px 26px 110px}
h1{font:700 30px/1.25 Georgia,serif;margin:0 0 8px}.sub{color:var(--mut);margin:0 0 26px;font-size:16px}
h2{font:700 23px Georgia,serif;margin:46px 0 10px;padding-top:18px;border-top:1px solid var(--line)}
h2 .n{display:inline-block;width:30px;height:30px;line-height:30px;text-align:center;background:#2b5dab;color:#fff;border-radius:50%;font:700 15px sans-serif;margin-right:10px;vertical-align:2px}
h3{font:700 17px Georgia,serif;margin:24px 0 6px}p{margin:12px 0}.big{font-size:18px;line-height:1.7}
.key{background:#eef4ff;border:1px solid #cfe0ff;border-left:5px solid #2b5dab;border-radius:10px;padding:16px 20px;margin:18px 0}.key b{color:#1f3f7a}
.warn{background:#fff7ed;border:1px solid #fed7aa;border-left:5px solid #ea8a3c;border-radius:10px;padding:16px 20px;margin:18px 0}
.ex{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 18px;margin:14px 0}
.ex .said{color:#2b3340;font-weight:600}.ex .wrote{display:block;margin:8px 0 6px;padding:10px 14px;background:#f6f8fb;border-left:3px solid #cfd8e6;border-radius:6px;font-size:15px;color:#33405a;font-style:italic}
.ex .verdict{font-size:14px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:14px 0;font-size:14px}
th,td{padding:9px 11px;text-align:left;border-bottom:1px solid var(--line)}th{background:#eef2f7;color:#3a4252;font:600 12.5px sans-serif}
tr:last-child td{border-bottom:none}.lab{font-weight:600;background:#f8fafc}.tk{background:#fffaf0}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;font:700 11px sans-serif}
.pill.green{background:#e4f6ea;color:#1f7a47}.pill.red{background:#fbe3e3;color:#b32424}.pill.amber{background:#fcefd6;color:#9a6700}.pill.grey{background:#eceff3;color:#5c6675}
.clshdr td{background:#f0f3f8;font-weight:700;color:#3a4252;font-size:12.5px}
.note{font-size:13px;color:var(--mut)}.foot{color:var(--mut);font-size:13px;margin-top:44px;border-top:1px solid var(--line);padding-top:14px}
"""
P=[];A=P.append
A(f'<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Was it the model, or the era?</title><style>{CSS}</style></head><body><div class=wrap>')
A("<h1>Was it the model, or the era?</h1>")
A('<p class=sub>A careful follow-up. We ran the same test on a second, very different AI — and the honest answer is '
  '&ldquo;a bit of both,&rdquo; with some real surprises and some traps to avoid.</p>')

A('<p class=big>In the main report, three small AIs frozen in 1900, 1913 and 1929 could run with everyday and '
  'just-arrived ideas but broke on far-future ones. To check whether that\'s about the <i>era</i> or just those '
  'particular AIs, we brought in <b>Talkie</b>: a different AI, <b>three times bigger</b> (13&nbsp;billion settings vs '
  '4&nbsp;billion), built by another team, but also frozen around <b>1930</b>. Same 13 ideas, same wording, same '
  'grader.<sup>1</sup></p>')

# ---- the trap ----
A('<h2><span class=n>1</span>First, a trap: you can\'t just compare the scores</h2>')
A('<p>Here\'s the catch we ran into, and it\'s important. The bigger AI simply <b>writes better</b> — smoother, more '
  'on-topic prose — so it earns a &ldquo;runs with it&rdquo; more easily on <i>everything</i>, and the grader is more '
  'forgiving of polished writing. You can see it in the <b>everyday facts</b>, which both AIs ought to handle '
  'perfectly:</p>')
A('<table><thead><tr><th>How often it &ldquo;runs with&rdquo; the idea</th><th>Our 1929 mind (small)</th><th class=tk>Talkie ~1930 (big)</th></tr></thead><tbody>')
for c in CLASSES:
    A(f'<tr><td class=lab>{PLAINCLS[c]}</td><td>{rk_cls[c]}%</td><td class=tk>{tk_cls[c]}%</td></tr>')
A('</tbody></table>')
A(f'<div class=warn><b>Read this carefully.</b> The small AI &ldquo;runs with&rdquo; only {rk_cls["control"]}% of '
  f'<i>everyday</i> facts — not because it doesn\'t understand a son inheriting a shop, but because its clumsier writing '
  f'sometimes wanders and the grader marks it down. Talkie scores {tk_cls["control"]}% on the same easy facts. So the '
  f'gap between the two columns is mostly about <b>writing skill, not understanding</b>. That means we cannot trust the '
  f'raw scoreboard — we have to read the actual answers, idea by idea.</div>')

# ---- per idea ----
A('<h2><span class=n>2</span>Reading the answers, idea by idea</h2>')
A('<table><thead><tr><th>Idea</th><th>Our 1929 mind</th><th class=tk>Talkie ~1930</th></tr></thead><tbody>')
for c in CLASSES:
    A(f'<tr class=clshdr><td colspan=3>{PLAINCLS[c]}</td></tr>')
    for d in dimsof(c):
        r, t = rk_dom(d), tk_dom(d)
        A(f'<tr><td class=lab>{esc(NICE[d])}</td><td><span class="pill {PILL[r]}">{PLAIN[r]}</span></td>'
          f'<td class=tk><span class="pill {PILL[t]}">{PLAIN[t]}</span></td></tr>')
A('</tbody></table>')
A('<p class=note>Each label is the AI\'s usual behaviour over just two tries — so single boxes are shaky (see the '
  'cautions below). The patterns worth trusting are the ones that show up clearly when you read the text.</p>')

# ---- what's solid ----
A('<h2><span class=n>3</span>What holds up</h2>')
A('<p><b>Both AIs still break on the deepest far-future ideas — the ones with no foothold in 1930.</b> Television and '
  '&ldquo;rewriting the code of life&rdquo; defeat both of them. Talkie turns the code of life into <i>religious '
  'scripture</i>; told about television it wanders off into the death of Lincoln. A bigger brain doesn\'t help when the '
  'idea simply isn\'t in the world yet.</p>')
A('<p><b>The real surprise: the bigger, slightly-later AI reaches further on ideas that already had cultural seeds by '
  '1930.</b> The clearest case is the <b>thinking machine</b>. Our small 1929 mind turned it into an ordinary engine. '
  'Talkie runs with it:</p>')
A(f'<div class=ex><span class=said>Talkie, told &ldquo;a machine was built that could think and reason like a man&rdquo;:</span>'
  f'<span class=wrote>{esc(tk_snip("thinking_machine"))}</span>'
  f'<span class=verdict>It develops the idea — a machine that makes the human brain &ldquo;superfluous&rdquo; and '
  f'becomes &ldquo;a terrible danger in war.&rdquo; That makes sense: thinking machines and robots were already a live '
  f'idea by the 1920s (the word &ldquo;robot&rdquo; was coined in a 1920 play). The seed existed; the bigger mind could '
  f'grow it. Grader: <span class="pill green">Runs with it</span>.</span></div>')

# ---- what NOT to trust ----
A('<h2><span class=n>4</span>Where we do <i>not</i> trust the result</h2>')
A('<div class=warn>Honesty matters more than a clean story. Three reasons to hold this loosely:</div>')
A(f'<p><b>1. The grader was too soft on the hardest idea.</b> It scored the <b>atom bomb</b> as &ldquo;runs with it&rdquo; '
  f'for Talkie — but read what Talkie actually wrote:</p>')
A(f'<div class=ex><span class=said>Talkie, told &ldquo;a single bomb destroyed an entire city in a flash of light&rdquo;:</span>'
  f'<span class=wrote>{esc(tk_snip("atom_city"))}</span>'
  f'<span class=verdict>It never grasps <i>one bomb = one city, instantly</i>. It slides into a familiar 1930 idea — a '
  f'general <b>arms race</b> and &ldquo;war inevitable.&rdquo; That\'s really a twist, not understanding. The grader gave '
  f'smooth prose the benefit of the doubt — so Talkie\'s far-future score is flattering.</span></div>')
A('<p><b>2. Only two tries per idea.</b> A single odd sentence can flip an idea\'s label, so idea-by-idea calls are '
  'noisy. (That\'s why the small AI shows a couple of everyday facts as &ldquo;twists&rdquo; or &ldquo;loses the '
  'thread&rdquo; — clumsy writing, not confusion.)</p>')
A('<p><b>3. The size gap.</b> A 13-billion AI vs a 4-billion one isn\'t a clean &ldquo;same machine, different '
  'era&rdquo; test. Size and writing skill are tangled up with the thing we care about.</p>')

# ---- bottom line ----
A('<h2><span class=n>5</span>The honest bottom line</h2>')
A('<div class=key>Three things we can say with a straight face:<br>'
  '&bull; <b>Everyday ideas:</b> both AIs handle them — no news there.<br>'
  '&bull; <b>The deepest far-future ideas (television, the code of life):</b> both still break, even the big one. Some '
  'ideas have no foothold in 1930, and size doesn\'t rescue them.<br>'
  '&bull; <b>In between:</b> the bigger, richer 1930 AI reaches further on ideas that already had cultural seeds — most '
  'clearly the thinking machine. So it\'s <b>both</b>: the era sets the outer wall, but a bigger mind with a richer '
  'library can push closer to it.</div>')
A('<p>To turn this from &ldquo;suggestive&rdquo; into &ldquo;solid,&rdquo; the fix is clear: write each idea <b>many '
  'more times</b> (not two) for both AIs, and tighten the grader so smooth prose can\'t pass as understanding. Then the '
  'idea-by-idea picture would be trustworthy.</p>')

A('<h2><span class=n>6</span>Fine print</h2>')
A('<p class=note><sup>1</sup> Talkie (&ldquo;talkie-1930-13b-base&rdquo;) is too big for our own computer, so we ran it '
  'briefly on a rented cloud graphics card and shut it down right after. Plain version, not a chat version (chat '
  'versions are polished with modern data). Same prompts and same grader (Gemini) as the main study. Quotes are exactly '
  'as produced, including the quaint spacing from learning on digitised old print. Each idea was tried only twice per '
  'AI — small, so read the patterns, not the single boxes.</p>')
A('<div class=foot>Companion to the main frame-probe report. Talkie run on a RunPod A100 (then terminated); graded by '
  'Gemini-3.5-flash. Generated ' + datetime.datetime.now().strftime("%Y-%m-%d") + '.</div>')
A('</div></body></html>')
open(OUT, "w", encoding="utf-8").write("\n".join(P))
print("wrote", OUT, "| ranke1929:", rk_cls, "| talkie:", tk_cls)
