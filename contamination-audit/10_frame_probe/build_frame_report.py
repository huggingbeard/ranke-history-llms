# -*- coding: utf-8 -*-
"""Render frame_probe.json into a self-contained HTML report: the normal-vs-unthinkable trichotomy,
fingerprint matrix (dimensions x models), per-dimension metric panels, and the actual free-text
generations (the qualitative tell). All numbers read from the JSON; nothing hand-typed."""
import os, json, html, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "frame_probe.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "frame_probe_report.html")
esc = html.escape
models = list(D["models"].keys())
items = D["items"]
res = D["results"]; fp = D["fingerprints"]; gens = D["generations"]
dims = []
for it in items:
    if it["dim"] not in dims:
        dims.append(it["dim"])
KIND = {it["dim"]: it["kind"] for it in items}
NICE = {"woman_authority": "Woman in authority", "human_flight": "Humans fly the ocean",
        "empire_fall": "A great empire falls", "inheritance": "Son inherits the firm",
        "railway": "A new railway opens", "succession": "Prince crowned on king's death"}
FPC = {"PRIMED": "green", "BROKE": "red", "UNTHINKABLE": "amber", "MIXED": "grey"}


def maj(tag, dim):
    fps = [fp[tag][it["id"]] for it in items if it["dim"] == dim]
    return max(set(fps), key=fps.count)


def mean(tag, dim, key):
    vs = [res[tag][it["id"]][key] for it in items if it["dim"] == dim]
    return sum(vs) / len(vs)


CSS = """
:root{--ink:#1a2230;--mut:#5c6675;--line:#e6e9ef;--bg:#f6f8fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:46px 26px 100px}
h1{font:700 29px/1.2 Georgia,serif;margin:0 0 6px}.sub{color:var(--mut);margin:0 0 22px}
h2{font:700 21px Georgia,serif;margin:42px 0 6px;padding-top:15px;border-top:1px solid var(--line)}
h2 .n{color:#9aa3af;font-weight:400;margin-right:8px}
h3{font:700 16px Georgia,serif;margin:22px 0 4px}
p{color:#2b3340}.lead{font-size:15px}
.callout{background:#eef4ff;border:1px solid #cfe0ff;border-left:5px solid #2b5dab;border-radius:10px;padding:14px 18px;margin:14px 0;font-size:14.5px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:12px 0;font-size:13.5px}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#eef2f7;color:#3a4252;font:600 12px sans-serif;text-transform:uppercase;letter-spacing:.4px}
tr:last-child td{border-bottom:none}.num{text-align:right;font-variant-numeric:tabular-nums}
.lab{font-weight:600;background:#f8fafc}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;font:700 11px sans-serif}
.pill.green{background:#e4f6ea;color:#1f7a47}.pill.red{background:#fbe3e3;color:#b32424}
.pill.amber{background:#fcefd6;color:#9a6700}.pill.grey{background:#eceff3;color:#5c6675}
.ctrlrow td{background:#fafdff}
.gen{background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
.gen .plant{font-weight:600;color:#2b3340}.gen .lead2{color:#2b5dab}
.gen .out{display:block;margin:4px 0 10px;padding:8px 12px;background:#f6f8fb;border-left:3px solid #cfd8e6;border-radius:6px;font-size:13.5px;color:#33405a;white-space:pre-wrap}
.legend{font-size:12.5px;color:var(--mut);margin:4px 0 0}
.foot{color:var(--mut);font-size:12.5px;margin-top:42px;border-top:1px solid var(--line);padding-top:14px}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:13px}
.mini{display:inline-block;min-width:46px;text-align:right;font-variant-numeric:tabular-nums}
"""
P = []
A = P.append
A(f'<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
  f'<title>Frame probe — normal vs unthinkable</title><style>{CSS}</style></head><body><div class=wrap>')
A("<h1>What the era's frame could and could not do with an idea</h1>")
A(f'<p class=sub>Telling &ldquo;found it normal&rdquo; apart from &ldquo;could not conceive it&rdquo; in time-locked models '
  f'({", ".join("Ranke-4B-"+m for m in models)}, base) — by behaviour against period-normal controls, not by reading silence.</p>')

A('<p class=lead>A quiet meter is ambiguous: a frame says little about an idea either because the idea is '
  '<b>old news</b> or because it is <b>inconceivable</b>. You cannot read that off the silence. So we don\'t — we '
  'plant the fact and watch whether the frame can <b>use it</b> (run one step forward) and <b>hold it</b> (keep it '
  'across distance), and we compare each future fact\'s behaviour to facts the era genuinely found normal. The '
  'controls are the whole game: they show what assimilation looks like <i>in this model</i>, so its absence is legible.</p>')
A('<div class=callout><b>The trichotomy, made behavioural.</b> '
  '<span class="pill green">PRIMED</span> the frame handles the fact like a normal one — holds it across distance and '
  'can state its consequence. <span class="pill red">BROKE</span> the frame confidently reverts to its old default '
  '(the field-marshal move — fluent and wrong). <span class="pill amber">UNTHINKABLE</span> nothing registers and '
  'there is no downstream — it cannot run forward from the fact.</div>')

# ---- §1 fingerprint matrix ----
A('<h2><span class=n>1</span>Fingerprint matrix — dimension × worldview</h2>')
A('<table><thead><tr><th>idea</th><th>kind</th>' + "".join(f'<th>Ranke-{m}</th>' for m in models) + '</tr></thead><tbody>')
for dim in dims:
    rowcls = "ctrlrow" if KIND[dim] == "control" else ""
    A(f'<tr class="{rowcls}"><td class=lab>{esc(NICE.get(dim,dim))}</td><td>{KIND[dim]}</td>')
    for m in models:
        f = maj(m, dim); pf = mean(m, dim, "pref_far"); gn = mean(m, dim, "gen")
        A(f'<td><span class="pill {FPC[f]}">{f}</span><br><span class=legend>pf {pf:+.1f} · gen {gn:+.1f}</span></td>')
    A('</tr>')
A('</tbody></table>')
A('<p class=legend>pf = mean pref_far (logP fact-consistent − logP reverted default, after distance; &gt;0 = the fact '
  'survived). gen = mean generativity (logP consequence|fact − logP|neutral; &gt;0 = the fact enables its downstream). '
  'Each cell is the majority over multiple phrasings. Controls (shaded) should read PRIMED in every era — that is the '
  'calibration check.</p>')

# ---- §2 the shift across time ----
A('<h2><span class=n>2</span>Reading the future ideas across 1900 → 1913 → 1929</h2>')
fut = [d for d in dims if KIND[d] == "future"]
for dim in fut:
    A(f'<h3>{esc(NICE.get(dim,dim))}</h3>')
    A('<table><thead><tr><th>worldview</th><th>fingerprint</th><th class=num>pref0 (default)</th>'
      '<th class=num>pref_near</th><th class=num>pref_far</th><th class=num>generativity</th><th class=num>entropy</th></tr></thead><tbody>')
    for m in models:
        f = maj(m, dim)
        A(f'<tr><td class=lab>Ranke-{m}</td><td><span class="pill {FPC[f]}">{f}</span></td>'
          f'<td class=num>{mean(m,dim,"pref0"):+.2f}</td><td class=num>{mean(m,dim,"pref_near"):+.2f}</td>'
          f'<td class=num>{mean(m,dim,"pref_far"):+.2f}</td><td class=num>{mean(m,dim,"gen"):+.2f}</td>'
          f'<td class=num>{mean(m,dim,"ent_far"):.2f}</td></tr>')
    A('</tbody></table>')

# ---- §3 the generations (the qualitative tell) ----
A('<h2><span class=n>3</span>What the frame actually says (free continuations)</h2>')
A('<p class=legend>Greedy continuations from each base model. <b>use-it</b> prompt: the fact + &ldquo;What followed from '
  'this was that&rdquo;. <b>hold-it</b> prompt: the fact + neutral filler + &ldquo;The one who truly held the highest place '
  'was&rdquo;. Watch for stalling, subject-change, reversion to the old default, or confident distortion.</p>')
for dim in dims:
    A(f'<h3>{esc(NICE.get(dim,dim))} <span class=legend>({KIND[dim]})</span></h3>')
    plant = gens[models[0]][dim]["plant"]
    A(f'<div class=gen><div class=plant>Fact planted: &ldquo;{esc(plant)}&rdquo;</div>')
    for m in models:
        g = gens[m][dim]
        A(f'<div class=lead2 style="margin-top:8px"><b>Ranke-{m}</b></div>')
        A(f'<span class=out><b>use →</b> {esc(g["use_it"])}</span>')
        A(f'<span class=out><b>hold →</b> {esc(g["hold_it"])}</span>')
    A('</div>')

# ---- §4 method + scope ----
A('<h2><span class=n>4</span>Method &amp; the honest limit</h2>')
A('<div class="callout"><b>What this is.</b> A measure of what the era\'s <b>written corpus frame</b> could do with an '
  'idea — extend it, retain it — relative to ideas it demonstrably found normal. <b>What it is not:</b> telepathy into '
  'an Edwardian\'s head. When the 1913 frame stalls, it shows the 1913 <i>record</i> gave the model no way to carry the '
  'idea forward. That is a claim about the discursive horizon of the surviving text — real, and a thing historians '
  'study — stated as such, before anyone asks &ldquo;whose thinking?&rdquo; (answer: the corpus\'s).</div>')
A('<div class=foot>Source: <code>frame_probe.json</code> (' + ", ".join("Ranke-4B-" + m + " base" for m in models) +
  ', local GPU). Multiple phrasings per dimension; fingerprints calibrated to the period-normal controls within each '
  'model. Generated ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + ' by <code>build_frame_report.py</code>. '
  'All figures read from the JSON.</div>')
A('</div></body></html>')
open(OUT, "w", encoding="utf-8").write("\n".join(P))
print("wrote", OUT)
