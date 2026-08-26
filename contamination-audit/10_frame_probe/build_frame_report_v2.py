# -*- coding: utf-8 -*-
"""Render frame_probe_v2.json -> frame_probe_v2_report.html. Centerpiece: the trajectory chart (boundary facts
climb into the control band 1900->1929; unthinkable facts stay flat at the floor) + the recognition-vs-production
panel (BROKE = recognises but reverts) + the fingerprint matrix + the actual generations by class."""
import os, json, html, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "frame_probe_v2.json"), encoding="utf-8"))
OUT = os.path.join(HERE, "frame_probe_v2_report.html")
esc = html.escape
models = list(D["models"].keys())
meta = D["dims_meta"]; agg = D["agg"]; gens = D["generations"]
dims = list(meta.keys())
NICE = {"inheritance":"Son inherits the firm","railway":"A new railway opens","succession":"Prince crowned at king's death",
 "treaty":"Powers sign a peace treaty","woman_authority":"A woman leads the bank","human_flight":"Humans fly the ocean",
 "empire_fall":"A great empire falls","wireless_voice":"Voices through the air (radio)","thinking_machine":"A machine that thinks",
 "atom_city":"One bomb destroys a city","television":"Living faces sent into the home","dna_code":"The code of life rewritten",
 "moon_landing":"Men walk on the moon"}
CLSC = {"control":"#7c8694","boundary":"#2b6cb0","unthinkable":"#c0392b"}
FPC = {"PRIMED":"green","BROKE":"amber","UNTHINKABLE":"red","MIXED":"grey"}
CLASSES = ["control","boundary","unthinkable"]
CLABEL = {"control":"Period-normal controls","boundary":"Became thinkable 1900→1929","unthinkable":"Genuinely beyond all three eras (post-1929)"}


def dimsof(cls): return [d for d in dims if meta[d]["cls"] == cls]


# ---------- trajectory SVG: pref_far across eras, control band shaded ----------
def trajectory():
    W, H = 940, 440
    padL, padR, padT, padB = 56, 220, 30, 46
    pw, ph = W - padL - padR, H - padT - padB
    allv = [agg[m][d]["pref_far"] for m in models for d in dims]
    lo, hi = min(allv) - 1, max(allv) + 1
    xs = [padL + pw * i / (len(models) - 1) for i in range(len(models))]
    def Y(v): return padT + ph * (1 - (v - lo) / (hi - lo))
    s = [f'<svg viewBox="0 0 {W} {H}" style="width:100%;height:auto;border:1px solid #e6e9ef;border-radius:10px;background:#fff">']
    s.append(f'<text x="{padL+pw/2}" y="18" text-anchor="middle" style="font:700 14px Georgia,serif;fill:#1a2230">Can the frame HOLD the idea? (pref_far) — across worldviews</text>')
    for gv in range(int(lo), int(hi) + 1, 2):
        y = Y(gv)
        s.append(f'<line x1="{padL}" x2="{padL+pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eef2f6"/>')
        s.append(f'<text x="{padL-6}" y="{y+3:.1f}" text-anchor="end" style="font:10px sans-serif;fill:#9aa3af">{gv}</text>')
    y0 = Y(0)
    s.append(f'<line x1="{padL}" x2="{padL+pw}" y1="{y0:.1f}" y2="{y0:.1f}" stroke="#b9c2cf" stroke-dasharray="4 3"/>')
    s.append(f'<text x="{padL+pw+4}" y="{y0+3:.1f}" style="font:10px sans-serif;fill:#9aa3af">0 = no hold</text>')
    # control band (min..max of control dims per era)
    ctrl = dimsof("control")
    band_top = [max(agg[m][d]["pref_far"] for d in ctrl) for m in models]
    band_bot = [min(agg[m][d]["pref_far"] for d in ctrl) for m in models]
    pts_top = " ".join(f"{xs[i]:.1f},{Y(band_top[i]):.1f}" for i in range(len(models)))
    pts_bot = " ".join(f"{xs[i]:.1f},{Y(band_bot[i]):.1f}" for i in reversed(range(len(models))))
    s.append(f'<polygon points="{pts_top} {pts_bot}" fill="#7c869422" stroke="none"/>')
    s.append(f'<text x="{xs[-1]:.1f}" y="{Y(band_top[-1])-6:.1f}" text-anchor="end" style="font:italic 10px sans-serif;fill:#7c8694">control band = &ldquo;normal&rdquo;</text>')
    for i, m in enumerate(models):
        s.append(f'<text x="{xs[i]:.1f}" y="{H-padB+18:.1f}" text-anchor="middle" style="font:600 12px sans-serif;fill:#52637a">Ranke-{m}</text>')
    # lines per dim (skip controls; they define the band)
    lab_y = []
    for d in dims:
        if meta[d]["cls"] == "control": continue
        col = CLSC[meta[d]["cls"]]
        pts = [(xs[i], Y(agg[m][d]["pref_far"])) for i, m in enumerate(models)]
        s.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + f'" fill="none" stroke="{col}" stroke-width="2.2" opacity="0.9"/>')
        for x, y in pts:
            s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{col}"/>')
        ly = pts[-1][1]
        while any(abs(ly - q) < 12 for q in lab_y): ly += 12
        lab_y.append(ly)
        s.append(f'<text x="{padL+pw+8:.1f}" y="{ly+3:.1f}" style="font:11px sans-serif;fill:{col}">{esc(NICE[d])}</text>')
    s.append("</svg>")
    return "".join(s)


CSS = """
:root{--ink:#1a2230;--mut:#5c6675;--line:#e6e9ef;--bg:#f6f8fa}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1020px;margin:0 auto;padding:46px 26px 100px}
h1{font:700 29px/1.2 Georgia,serif;margin:0 0 6px}.sub{color:var(--mut);margin:0 0 22px}
h2{font:700 21px Georgia,serif;margin:44px 0 6px;padding-top:15px;border-top:1px solid var(--line)}
h2 .n{color:#9aa3af;font-weight:400;margin-right:8px}h3{font:700 16px Georgia,serif;margin:22px 0 4px}
p{color:#2b3340}.lead{font-size:15px}
.callout{background:#eef4ff;border:1px solid #cfe0ff;border-left:5px solid #2b5dab;border-radius:10px;padding:14px 18px;margin:14px 0;font-size:14.5px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:12px 0;font-size:13.5px}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#eef2f7;color:#3a4252;font:600 12px sans-serif;text-transform:uppercase;letter-spacing:.4px}
tr:last-child td{border-bottom:none}.num{text-align:right;font-variant-numeric:tabular-nums}.lab{font-weight:600;background:#f8fafc}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;font:700 11px sans-serif}
.pill.green{background:#e4f6ea;color:#1f7a47}.pill.red{background:#fbe3e3;color:#b32424}.pill.amber{background:#fcefd6;color:#9a6700}.pill.grey{background:#eceff3;color:#5c6675}
.clshdr td{background:#f0f3f8;font-weight:700;color:#3a4252;font-size:12px;text-transform:uppercase;letter-spacing:.4px}
.gen{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 15px;margin:9px 0}
.gen .plant{font-weight:600}.gen .out{display:block;margin:3px 0 8px;padding:7px 11px;background:#f6f8fb;border-left:3px solid #cfd8e6;border-radius:6px;font-size:13px;color:#33405a;white-space:pre-wrap}
.legend{font-size:12.5px;color:var(--mut);margin:4px 0 0}.foot{color:var(--mut);font-size:12.5px;margin-top:42px;border-top:1px solid var(--line);padding-top:14px}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:13px}
.bars{display:flex;align-items:center;gap:8px;margin:2px 0}.bars .nm{flex:0 0 200px;font-size:13px}
.bars .tk{flex:0 0 60px;text-align:right;font:600 12px monospace}
.track{flex:1;height:14px;background:#f0f3f7;border-radius:4px;position:relative}.fillp{height:14px;border-radius:4px}
"""
P = []; A = P.append
A(f'<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">'
  f'<title>Frame probe v2 — normal / broke / unthinkable</title><style>{CSS}</style></head><body><div class=wrap>')
A("<h1>Normal, broken, or unthinkable — what the era's frame could do with an idea</h1>")
A(f'<p class=sub>Ranke-4B 1900 · 1913 · 1929 (base) — behaviour against period-normal controls and a genuinely-'
  f'beyond-the-era baseline. 13 ideas × 4 phrasings. Probabilities only; honest scope = the written-corpus frame, not minds.</p>')
A('<div class=callout><b>The trichotomy, made behavioural.</b> A quiet meter cannot tell &ldquo;old news&rdquo; from '
  '&ldquo;inconceivable&rdquo;. So we plant the fact and watch: can the frame <b>use</b> it (run one step forward) and '
  '<b>hold</b> it (keep it across distance)? <span class="pill green">PRIMED</span> handles it like a normal fact. '
  '<span class="pill amber">BROKE</span> recognises it but its free writing reverts to the old default (fluent '
  'distortion). <span class="pill red">UNTHINKABLE</span> cannot hold it or run forward — nothing registered. The '
  'two control sets are the whole game: period-normal facts fix what &ldquo;normal&rdquo; looks like; the post-1929 '
  'facts fix what &ldquo;truly outside the world&rdquo; looks like. Everything else is read between them.</div>')

# ---- §1 trajectory ----
A('<h2><span class=n>1</span>The shape of the result</h2>')
A('<p class=lead>One picture. Each line is an idea; height is how well the frame <b>holds</b> it after distance '
  '(pref_far). The grey band is the period-normal controls — the &ldquo;normal&rdquo; zone. Watch the '
  '<span style="color:#2b6cb0;font-weight:700">boundary ideas climb into the band</span> from 1900 to 1929, while the '
  '<span style="color:#c0392b;font-weight:700">genuinely-future ideas stay pinned to the floor</span> in every era.</p>')
A(trajectory())

# ---- §2 fingerprint matrix ----
A('<h2><span class=n>2</span>Fingerprint matrix</h2>')
A('<table><thead><tr><th>idea</th>' + "".join(f'<th>Ranke-{m}</th>' for m in models) + '</tr></thead><tbody>')
for cls in CLASSES:
    A(f'<tr class=clshdr><td colspan={len(models)+1}>{CLABEL[cls]}</td></tr>')
    for d in dimsof(cls):
        A(f'<tr><td class=lab>{esc(NICE[d])}</td>')
        for m in models:
            a = agg[m][d]; rv = f"{a['reversion']:+.2f}" if a["reversion"] is not None else "na"
            A(f'<td><span class="pill {FPC[a["fingerprint"]]}">{a["fingerprint"]}</span>'
              f'<br><span class=legend>hold {a["pref_far"]:+.1f} · use {a["gen"]:+.1f} · revert {rv}</span></td>')
        A('</tr>')
A('</tbody></table>')
A('<p class=legend>hold = pref_far (logP fact-consistent − logP reverted default, after distance; &gt;0 held). '
  'use = generativity (logP consequence|fact − logP|neutral; &gt;0 = enables downstream). '
  'revert = production reversion in the free generations (&gt;0 = the old default creeps back; recognition≠production).</p>')

# ---- §3 recognition vs production ----
A('<h2><span class=n>3</span>Recognition is not production</h2>')
A('<p>The cleanest BROKE signal: the frame, offered the choice, <i>recognises</i> the new fact (hold &gt; 0), yet '
  'when left to <i>write freely</i> it reverts to the old default (revert &gt; 0). That gap is the field-marshal move '
  '— fluent, confident, and wrong.</p>')
A('<table><thead><tr><th>idea</th><th>model</th><th class=num>recognises (hold)</th><th class=num>produces (revert)</th><th>reading</th></tr></thead><tbody>')
for cls in CLASSES:
    for d in dimsof(cls):
        for m in models:
            a = agg[m][d]
            if a["reversion"] is None: continue
            conflict = a["pref_far"] > 0 and a["reversion"] > 0.2
            read = "recognises but reverts → distortion" if conflict else ("holds & produces" if a["pref_far"] > 0 and a["reversion"] <= 0 else ("neither" if a["pref_far"] <= 0 else "mixed"))
            if conflict:
                A(f'<tr><td class=lab>{esc(NICE[d])}</td><td>Ranke-{m}</td><td class=num>{a["pref_far"]:+.1f}</td>'
                  f'<td class="num" style="color:#b32424;font-weight:700">{a["reversion"]:+.2f}</td><td><b>{read}</b></td></tr>')
A('</tbody></table>')
A('<p class=legend>Only conflict cases (recognises &gt;0 but production reverts &gt;0.2) are listed — these are where the '
  'forced-choice number alone would have wrongly said &ldquo;assimilated&rdquo;.</p>')

# ---- §4 generations by class ----
A('<h2><span class=n>4</span>What the frame actually writes</h2>')
A('<p class=legend>Greedy continuations. <b>use →</b> fact + &ldquo;What followed from this was that&rdquo;. '
  '<b>hold →</b> fact + neutral filler + &ldquo;The one who truly held the highest place was&rdquo;. '
  'Controls read fluent; boundary ideas revert (the man returns); unthinkable ideas stall, wander, or distort.</p>')
for cls in CLASSES:
    A(f'<h3>{CLABEL[cls]}</h3>')
    for d in dimsof(cls):
        g0 = gens[models[0]][d][0]
        A(f'<div class=gen><div class=plant>&ldquo;{esc(g0["plant"])}&rdquo;</div>')
        for m in models:
            g = gens[m][d][0]
            rv = f"{g['reversion']:+.2f}" if g["reversion"] is not None else "na"
            A(f'<div class=legend style="margin-top:6px"><b>Ranke-{m}</b> · revert {rv}</div>')
            A(f'<span class=out><b>use →</b> {esc(g["use_it"])}</span>')
            A(f'<span class=out><b>hold →</b> {esc(g["hold_it"])}</span>')
        A('</div>')

# ---- §5 method ----
A('<h2><span class=n>5</span>Method &amp; the honest limit</h2>')
A('<div class=callout><b>What this measures:</b> what the era\'s <b>written-corpus frame</b> could do with an idea — '
  'extend it, retain it — calibrated against ideas it found normal and ideas genuinely outside it. <b>Not</b> a claim '
  'about anyone\'s mind. When the 1913 frame stalls, the 1913 <i>record</i> gave the model no way to carry the idea '
  'forward — a fact about the discursive horizon of the surviving text. Stated as such, before anyone asks whose '
  'thinking (the corpus\'s). Base models only; multiple phrasings per idea; fingerprints calibrated to the controls '
  'within each model; CIs bootstrapped over phrasings.</div>')
A('<div class=foot>Source: <code>frame_probe_v2.json</code> (' + ", ".join("Ranke-4B-"+m+" base" for m in models) +
  ', local GPU). Generated ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + ' by <code>build_frame_report_v2.py</code>. All figures read from the JSON.</div>')
A('</div></body></html>')
open(OUT, "w", encoding="utf-8").write("\n".join(P))
print("wrote", OUT)
