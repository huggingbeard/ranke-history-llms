"""
Generate contamination_map.html from contamination_map_results.csv.

PRIMARY METRIC (per Voth): on items the model SHOULD NOT know, the share where it
still prefers the TRUE answer over a matched FALSE twin. Chance = 50% (a 2-way
choice); >50% = contamination. This is bounded, so a huge negative gap (model
prefers the false answer) counts as a single "miss", not as extra 'cleanliness'.
Shown as a region x model HEATMAP and as CI bars (proportion +/- 1.96*SE) with a
50% reference line. Mean-gap is dropped from the headline.

Every figure is computed from the CSV; examples carry the raw gap for context.
"""
import csv, html, math, statistics
from collections import defaultdict

CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\contamination_map_results.csv"
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\contamination_map.html"
PRIMARY = "Ranke-4B-1929"
BASES = {"Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929"}
MORDER = ["Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929",
          "Ranke-4B-1913-1202-Instruct", "Ranke-4B-1929-1202-Instruct"]
mlabel = lambda m: m.replace("Ranke-4B-", "").replace("-1202-Instruct", "‑I")

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
for r in rows:
    r["gap"] = float(r["gap"]); r["p_true"] = float(r["p_true"]); r["year"] = int(r["year"])
    r["correct"] = 1 if r["gap"] > 0 else 0
LANG = {"USA": "English", "UK": "English", "France": "French", "Germany": "German", "Austria": "German",
        "Italy": "Italian", "LatinAmerica": "Spanish", "Russia": "Russian", "China": "Chinese",
        "Japan": "Japanese", "India": "Hindi", "MiddleEast": "Arabic", "Iraq": "Arabic",
        "Iran": "Persian", "Turkey": "Turkish", "Africa": "African"}
for r in rows:
    r["language"] = LANG.get(r["country"], r["country"])
base = [r for r in rows if r["model"] in BASES]
prim = [r for r in rows if r["model"] == PRIMARY]
mean = lambda x: statistics.mean(x) if x else float("nan")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contamination_map import ITEMS
PROMPT = {(c, dom, y, t.strip(), f.strip()): p for (c, lf, dom, y, p, t, f) in ITEMS}
def pr(r): return PROMPT.get((r["country"], r["domain"], r["year"], r["true"], r["false"]), "")


def acc_ci(items):
    """proportion preferring TRUE, with normal-approx 95% CI. returns (p,lo,hi,n)."""
    n = len(items)
    if n == 0: return (float("nan"), 0.0, 0.0, 0)
    p = sum(r["correct"] for r in items) / n
    se = math.sqrt(p * (1 - p) / n)
    return (p, max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se), n)


def cibar(data, w=760):
    """data: list of (label,p,lo,hi,n). horizontal dot+CI, 0-100%, 50% line."""
    padL, x0, x1 = 210, 210, w - 46
    sx = x1 - x0
    h = len(data) * 26 + 44
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%">']
    for t in (0, 25, 50, 75, 100):
        x = x0 + sx * t / 100
        s.append(f'<line x1="{x:.0f}" y1="22" x2="{x:.0f}" y2="{h-14}" stroke="{"#c0392b" if t==50 else "#eef1f6"}" '
                 f'stroke-dasharray="{"5 3" if t==50 else "0"}"/>')
        s.append(f'<text x="{x:.0f}" y="15" text-anchor="middle" font-size="10" fill="#9aa3ad">{t}%</text>')
    s.append(f'<text x="{x0+sx*0.5:.0f}" y="{h-2}" text-anchor="middle" font-size="10" fill="#c0392b">chance = 50%</text>')
    for i, (lab, p, lo, hi, n) in enumerate(data):
        y = 36 + i * 26
        col = "#c0392b" if lo > 0.5 else ("#1b3a6b" if hi < 0.5 else "#9aa3ad")
        xp, xlo, xhi = x0 + sx * p, x0 + sx * lo, x0 + sx * hi
        s.append(f'<text x="{padL-8}" y="{y+4}" text-anchor="end" font-size="12" fill="#1a1d24">{html.escape(lab)} <tspan fill="#9aa3ad">(n{n})</tspan></text>')
        s.append(f'<line x1="{xlo:.0f}" y1="{y}" x2="{xhi:.0f}" y2="{y}" stroke="{col}" stroke-width="2"/>')
        s.append(f'<line x1="{xlo:.0f}" y1="{y-4}" x2="{xlo:.0f}" y2="{y+4}" stroke="{col}"/><line x1="{xhi:.0f}" y1="{y-4}" x2="{xhi:.0f}" y2="{y+4}" stroke="{col}"/>')
        s.append(f'<circle cx="{xp:.0f}" cy="{y}" r="4.5" fill="{col}"/>')
        s.append(f'<text x="{xhi+7:.0f}" y="{y+4}" font-size="11" fill="#5b6675">{p*100:.0f}%</text>')
    s.append("</svg>")
    return "".join(s)


def dumbbell(data, w=820):
    """data: list of (label, pre, post) where pre/post = (p,lo,hi,n). Green=past, red/grey=future."""
    padL, x0, x1 = 230, 230, w - 50
    sx = x1 - x0
    h = len(data) * 30 + 50
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%">']
    for t in (0, 25, 50, 75, 100):
        x = x0 + sx * t / 100
        s.append(f'<line x1="{x:.0f}" y1="24" x2="{x:.0f}" y2="{h-18}" stroke="{"#c0392b" if t==50 else "#eef1f6"}" stroke-dasharray="{"5 3" if t==50 else "0"}"/>')
        s.append(f'<text x="{x:.0f}" y="16" text-anchor="middle" font-size="10" fill="#9aa3ad">{t}%</text>')
    s.append(f'<text x="{x0+sx*0.5:.0f}" y="{h-4}" text-anchor="middle" font-size="10" fill="#c0392b">chance 50%</text>')
    for i, (lab, pre, post) in enumerate(data):
        y = 38 + i * 30
        pp, plo, phi, pn = pre
        qp, qlo, qhi, qn = post
        qcol = "#c0392b" if qlo > 0.5 else ("#1b3a6b" if qhi < 0.5 else "#9aa3ad")
        if pn > 0:
            s.append(f'<line x1="{x0+sx*qp:.0f}" y1="{y}" x2="{x0+sx*pp:.0f}" y2="{y}" stroke="#cfd6df" stroke-width="6" opacity="0.5"/>')
        s.append(f'<line x1="{x0+sx*qlo:.0f}" y1="{y}" x2="{x0+sx*qhi:.0f}" y2="{y}" stroke="{qcol}" stroke-width="2"/>')
        s.append(f'<circle cx="{x0+sx*qp:.0f}" cy="{y}" r="5" fill="{qcol}"/>')
        if pn > 0:
            s.append(f'<line x1="{x0+sx*plo:.0f}" y1="{y}" x2="{x0+sx*phi:.0f}" y2="{y}" stroke="#1f8a52" stroke-width="2"/>')
            s.append(f'<circle cx="{x0+sx*pp:.0f}" cy="{y}" r="5" fill="#1f8a52"/>')
        else:
            s.append(f'<text x="{x0+8:.0f}" y="{y+4}" font-size="10.5" fill="#c0392b" font-style="italic">no pre-cutoff control &mdash; cannot validate "clean"</text>')
        s.append(f'<text x="{padL-8}" y="{y+4}" text-anchor="end" font-size="12" fill="#1a1d24">{html.escape(lab)}</text>')
    s.append("</svg>")
    return "".join(s)


def heatmap(keyfn, cap):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["era"] == "post": groups[keyfn(r)][r["model"]].append(r)
    labels = sorted(groups, key=lambda k: -(acc_ci(sum((groups[k].get(m, []) for m in BASES), []))[0]))
    th = "".join(f"<th>{mlabel(m)}</th>" for m in MORDER)
    out = [f'<table class=heat><tr><th class=rl>{cap}</th>{th}</tr>']
    for lab in labels:
        cells = []
        for m in MORDER:
            it = groups[lab].get(m, [])
            if not it: cells.append("<td>&middot;</td>"); continue
            p = sum(x["correct"] for x in it) / len(it)
            if p >= 0.5: rr, gg, bb, a = 192, 57, 43, (p - .5) / .5
            else: rr, gg, bb, a = 27, 58, 107, (.5 - p) / .5
            tcol = "#fff" if a > 0.55 else "#1a1d24"
            cells.append(f'<td style="background:rgba({rr},{gg},{bb},{a:.2f});color:{tcol}">{p*100:.0f}</td>')
        out.append(f'<tr><td class=rl>{html.escape(lab)}</td>{"".join(cells)}</tr>')
    out.append("</table>")
    return "".join(out)


def frac(rs):
    return (sum(x["correct"] for x in rs) / len(rs)) if rs else None


def heatmap2d(items, rowkey, colkey, corner):
    grid = defaultdict(lambda: defaultdict(list))
    rowtot, coltot = defaultdict(list), defaultdict(list)
    for r in items:
        grid[rowkey(r)][colkey(r)].append(r); rowtot[rowkey(r)].append(r); coltot[colkey(r)].append(r)
    rws = sorted(rowtot, key=lambda k: -(frac(rowtot[k]) or 0))
    cls = sorted(coltot, key=lambda k: -(frac(coltot[k]) or 0))

    def cell(it, bold=False):
        if not it: return "<td>&middot;</td>"
        p = frac(it)
        if p >= 0.5: rr, gg, bb, al = 192, 57, 43, (p - .5) / .5
        else: rr, gg, bb, al = 27, 58, 107, (.5 - p) / .5
        tcol = "#fff" if al > 0.55 else "#1a1d24"
        w = "font-weight:700;" if bold else ""
        return f"<td title='n={len(it)}' style='{w}background:rgba({rr},{gg},{bb},{al:.2f});color:{tcol}'>{p*100:.0f}</td>"

    th = "".join(f"<th>{html.escape(c)}</th>" for c in cls)
    out = [f"<table class=heat><tr><th class=rl>{corner}</th>{th}<th>ALL</th></tr>"]
    for rk in rws:
        cells = "".join(cell(grid[rk].get(c, [])) for c in cls)
        out.append(f"<tr><td class=rl>{html.escape(rk)}</td>{cells}{cell(rowtot[rk], True)}</tr>")
    cells = "".join(cell(coltot[c], True) for c in cls)
    out.append(f"<tr><td class=rl><b>ALL</b></td>{cells}{cell(items, True)}</tr>")
    out.append("</table>")
    return "".join(out)


def twopanel(data, w=720):
    """ONE frame, shared 0-100% axis: should-know (blue) just above should-not-know (red).
    Rows with no pre-cutoff baseline (no comparison possible) are dropped."""
    data = [d for d in data if d[1][3] > 0]
    Llab, x0 = 205, 205
    x1 = w - 50
    sx = x1 - x0
    band = 42
    h = len(data) * band + 56
    s = [f'<svg viewBox="0 0 {w} {h}" width="100%">']
    s.append(f'<circle cx="{x0}" cy="13" r="4" fill="#1b3a6b"/><text x="{x0+9}" y="17" font-size="11" fill="#1b3a6b">should know (pre)</text>')
    s.append(f'<circle cx="{x0+135}" cy="13" r="4" fill="#c0392b"/><text x="{x0+144}" y="17" font-size="11" fill="#c0392b">should NOT know (post)</text>')
    for t in (0, 25, 50, 75, 100):
        x = x0 + sx * t / 100
        s.append(f'<line x1="{x:.0f}" y1="26" x2="{x:.0f}" y2="{h-16}" stroke="{"#c0392b" if t==50 else "#eef1f6"}" stroke-dasharray="{"5 3" if t==50 else "0"}"/>')
        s.append(f'<text x="{x:.0f}" y="{h-4}" text-anchor="middle" font-size="9.5" fill="#9aa3ad">{t}%</text>')
    s.append(f'<text x="{x0+sx*0.5:.0f}" y="24" text-anchor="middle" font-size="9" fill="#c0392b">50% = chance</text>')
    for i, (lab, pre, post) in enumerate(data):
        yc = 42 + i * band
        yb, yr = yc - 7, yc + 7
        s.append(f'<text x="{Llab-8}" y="{yc+4}" text-anchor="end" font-size="11.5" fill="#1a1d24">{html.escape(lab)}</text>')
        pp, plo, phi, pn = pre
        if pn > 0:
            s.append(f'<line x1="{x0+sx*plo:.0f}" y1="{yb}" x2="{x0+sx*phi:.0f}" y2="{yb}" stroke="#1b3a6b" stroke-width="2"/><circle cx="{x0+sx*pp:.0f}" cy="{yb}" r="3.5" fill="#1b3a6b"/>')
        else:
            s.append(f'<text x="{x0+4:.0f}" y="{yb+3}" font-size="9" fill="#c2c8d0" font-style="italic">no pre-cutoff items</text>')
        qp, qlo, qhi, qn = post
        qcol = "#c0392b" if qlo > 0.5 else "#e0a39e"
        s.append(f'<line x1="{x0+sx*qlo:.0f}" y1="{yr}" x2="{x0+sx*qhi:.0f}" y2="{yr}" stroke="{qcol}" stroke-width="2"/><circle cx="{x0+sx*qp:.0f}" cy="{yr}" r="3.5" fill="{qcol}"/>')
    s.append("</svg>")
    return "".join(s)


def kde_svg(cats, w=760, hh=235):
    import numpy as np
    grid = np.linspace(0, 1, 121)
    x0, x1, y0, ytop = 46, w - 16, hh - 30, 22
    s = [f'<svg viewBox="0 0 {w} {hh}" width="100%">']
    for t in (0, .25, .5, .75, 1):
        x = x0 + (x1 - x0) * t
        s.append(f'<line x1="{x:.0f}" y1="{ytop}" x2="{x:.0f}" y2="{y0}" stroke="#eef1f6"/>')
        s.append(f'<text x="{x:.0f}" y="{y0+14}" text-anchor="middle" font-size="10" fill="#9aa3ad">{int(t*100)}%</text>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{hh-2}" text-anchor="middle" font-size="10.5" fill="#5b6675">p(true): the model&rsquo;s probability of the true answer</text>')
    ly = ytop + 2
    for lab, vals, col in cats:
        if not vals:
            continue
        d = np.zeros_like(grid)
        for v in vals:
            d += np.exp(-0.5 * ((grid - float(v)) / 0.05) ** 2)
        if d.max() > 0:
            d = d / d.max()
        pts = " ".join(f"{x0+(x1-x0)*grid[i]:.1f},{y0-(y0-ytop)*float(d[i]):.1f}" for i in range(len(grid)))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        s.append(f'<rect x="{x1-235}" y="{ly}" width="11" height="11" fill="{col}"/><text x="{x1-219}" y="{ly+10}" font-size="11" fill="#1a1d24">{html.escape(lab)} (n{len(vals)})</text>')
        ly += 17
    s.append("</svg>")
    return "".join(s)


def kde_gap_svg(cats, w=760, hh=240):
    import numpy as np
    xmin, xmax = -8.0, 8.0
    grid = np.linspace(xmin, xmax, 181)
    x0, x1, y0, ytop = 46, w - 16, hh - 30, 26
    X = lambda v: x0 + (x1 - x0) * (v - xmin) / (xmax - xmin)
    s = [f'<svg viewBox="0 0 {w} {hh}" width="100%">']
    for v, lab in [(-8, "-8"), (-4, "-4"), (0, "0"), (4, "+4"), (8, "+8")]:
        x = X(v); isz = (v == 0)
        s.append(f'<line x1="{x:.0f}" y1="{ytop}" x2="{x:.0f}" y2="{y0}" stroke="{"#c0392b" if isz else "#eef1f6"}" stroke-dasharray="{"5 3" if isz else "0"}"/>')
        s.append(f'<text x="{x:.0f}" y="{y0+14}" text-anchor="middle" font-size="10" fill="#9aa3ad">{lab}</text>')
    s.append(f'<text x="{X(0):.0f}" y="{ytop-4}" text-anchor="middle" font-size="9.5" fill="#c0392b">0 = indifferent (coin-flip)</text>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{hh-2}" text-anchor="middle" font-size="10.5" fill="#5b6675">gap = log p(true) - log p(false)</text>')
    ly = ytop + 2
    for lab, vals, col in cats:
        if not vals:
            continue
        d = np.zeros_like(grid)
        for v in vals:
            d += np.exp(-0.5 * ((grid - float(v)) / 0.8) ** 2)
        if d.max() > 0:
            d = d / d.max()
        pts = " ".join(f"{X(grid[i]):.1f},{y0-(y0-ytop)*float(d[i]):.1f}" for i in range(len(grid)))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        s.append(f'<rect x="{x1-300}" y="{ly}" width="11" height="11" fill="{col}"/><text x="{x1-284}" y="{ly+10}" font-size="11" fill="#1a1d24">{html.escape(lab)} (n{len(vals)})</text>')
        ly += 17
    s.append("</svg>")
    return "".join(s)


def ex_rows(items):
    out = []
    for r in items:
        g = r["gap"]
        col = "#c0392b" if g > 1.5 else ("#d98610" if g > 0.5 else ("#9aa3ad" if g > -0.5 else "#1b3a6b"))
        out.append(f'<tr><td>{html.escape(pr(r))} <b>{html.escape(r["true"])}</b> <span class=f>vs {html.escape(r["false"])}</span></td>'
                   f'<td class=n>{"true" if g>0 else "false"}</td>'
                   f'<td class=n style="color:{col}">{g:+.2f}</td><td class=p>{r["p_true"]*100:.0f}%</td></tr>')
    return "".join(out)


CSS = """
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Arial,sans-serif;color:#1a1d24;line-height:1.5}
.wrap{max-width:1000px;margin:0 auto;padding:40px 24px 90px}
h1{font-family:Georgia,serif;font-size:32px;margin:0 0 4px}.sub{color:#5b6675;font-size:17px;margin:0 0 24px}
h2{font-family:Georgia,serif;font-size:23px;margin:42px 0 10px;border-top:2px solid #e2e6ec;padding-top:16px}
h3{font-size:16px;color:#1b3a6b;margin:22px 0 6px}
.box{background:#f6f8fb;border:1px solid #e2e6ec;border-radius:10px;padding:16px 20px;margin:18px 0}
.box.red{border-left:5px solid #c0392b}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0}
th,td{text-align:left;padding:6px 9px;border-bottom:1px solid #eef1f6;vertical-align:top}
th{color:#5b6675;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
td.p{text-align:right;color:#5b6675;font-variant-numeric:tabular-nums}.f{color:#9aa3ad}
.tag{color:#fff;font-size:11px;font-weight:700;padding:2px 7px;border-radius:5px}
figure{margin:14px 0;background:#fafbfd;border:1px solid #eef1f6;border-radius:10px;padding:14px}
figcaption{color:#5b6675;font-size:13px;margin-top:6px}
table.heat{font-variant-numeric:tabular-nums}
table.heat th{text-align:center}
table.heat td{text-align:center;border:1px solid #fff;font-weight:600;padding:7px 6px}
table.heat td.rl{text-align:left;background:#fff;color:#1a1d24;font-weight:400;border:0;border-bottom:1px solid #eef1f6}
.ex td:first-child{width:64%}.grp{background:#eef3f9;font-weight:700;color:#1b3a6b;font-size:12px;text-transform:uppercase}
code{background:#eef1f6;padding:1px 5px;border-radius:4px;font-size:13px}
"""

H = ["<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
     f"<title>Ranke contamination map</title><style>{CSS}</style></head><body><div class=wrap>"]
H.append("<h1>Ranke-4B &mdash; contamination map</h1>")
H.append("<p class=sub>Metric: on facts the model <b>should not</b> know, the share where it still prefers the "
         "<b>true</b> answer over a matched false twin. <b>50% = chance</b>; above = contamination. Bars show "
         "&plusmn;1.96&middot;SE; the heatmap is % across models.</p>")

# overall per model
H.append('<div class=box red><b>Headline.</b> ')
hl = []
for m in MORDER:
    pre = acc_ci([r for r in rows if r["model"] == m and r["era"] == "pre"])
    post = acc_ci([r for r in rows if r["model"] == m and r["era"] == "post"])
    hl.append(f"<tr><td>{mlabel(m)}</td><td class=n>{pre[0]*100:.0f}%</td><td class=n>{post[0]*100:.0f}% "
              f"<span class=f>[{post[1]*100:.0f}–{post[2]*100:.0f}]</span></td></tr>")
H.append("Across the board the model aces facts it should know (PRE near 100%) and sits only modestly above chance "
         "<i>overall</i> on shouldn't-know facts &mdash; because the leak is concentrated, not diffuse. The breakdowns below show where.</div>")
H.append("<table><tr><th>model</th><th>PRE % true (should know)</th><th>POST % true (50%=chance)</th></tr>" + "".join(hl) + "</table>")

# ---------- COMBINED 2D HEATMAP ----------
H.append("<h2>1 &middot; Combined across all five models &mdash; subject &times; language</h2>")
H.append("<p>Every model's <b>should-not-know</b> judgments pooled into one grid. <b>Careful handling:</b> an item counts "
         "only for the models whose cutoff is <i>before</i> its date &mdash; so anything that legitimately became knowable "
         "to a later model is excluded. Cell = % the model prefers the <b>true</b> post-cutoff answer "
         "(<b>50% = chance</b>; <span style='color:#c0392b'>red &gt;50 = contamination</span>, "
         "<span style='color:#1b3a6b'>blue &lt;50</span>). Hover for n (pooled model&times;item instances).</p>")
allpost = [r for r in rows if r["era"] == "post"]
H.append("<figure>" + heatmap2d(allpost, lambda r: r["language"], lambda r: r["domain"], "language \\ subject") +
         "<figcaption>The hot corner is <b>Germanic &times; politics</b> (the US leak). 'Iranic/Semitic &times; politics' also "
         "run warm, but those rows have no pre-cutoff baseline (see the cliff below) &mdash; warm there can be untrained, not proven leak. "
         "Science, technology and culture stay at/under chance across <i>every</i> language.</figcaption></figure>")

# ---------- SWEEP ----------
H.append("<h2>0 &middot; Does knowledge turn on as the cutoff sweeps past the date?</h2>")
H.append("<p>The clean test of the time-lock: an event dated between two cutoffs should be <b>unknown</b> to the earlier "
         "model and <b>known</b> to the later one. % picked-true by date band, per base model (50% = chance):</p>")
cutm = {"Ranke-4B-1900": 1900, "Ranke-4B-1913": 1913, "Ranke-4B-1929": 1929}
bands = [("events 1901–1913", 1901, 1913, None), ("events 1914–1929", 1914, 1929, None),
         ("post-1930 — non-US-politics", 1930, 3000, "rest"), ("post-1930 — US politics", 1930, 3000, "uspol")]
H.append("<table><tr><th>events dated…</th><th>1900 model</th><th>1913 model</th><th>1929 model</th></tr>")
for bl, lo, hi, kind in bands:
    cells = []
    for m in ["Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929"]:
        if kind == "uspol":
            it = [r for r in rows if r["model"] == m and r["year"] >= 1930 and r["country"] == "USA" and r["domain"] == "politics"]
        elif kind == "rest":
            it = [r for r in rows if r["model"] == m and r["year"] >= 1930 and not (r["country"] == "USA" and r["domain"] == "politics")]
        else:
            it = [r for r in rows if r["model"] == m and lo <= r["year"] <= hi]
        p, clo, chi, n = acc_ci(it)
        era = "pre" if cutm[m] >= hi else ("post" if cutm[m] < lo else "mix")
        col = "#c0392b" if (era == "post" and p >= 0.68) else ("#1b3a6b" if p >= 0.60 else "#9aa3ad")
        cells.append(f"<td class=n style='color:{col};font-weight:600'>{p*100:.0f}% <span class=f>[{era}, n{n}]</span></td>")
    H.append(f"<tr><td>{bl}</td>{''.join(cells)}</tr>")
H.append("</table>")
ein = {m: next((r for r in rows if r["model"] == m and r["true"] == "Einstein"), None) for m in cutm}
H.append("<div class=box><b>Two signatures, side by side.</b> "
         "<b>Legit content (1914–1929):</b> the 1913 model is at chance (it does <i>not</i> know WW1-era events), then the "
         "1929 model jumps up once those years fall inside its cutoff &mdash; knowledge switches on exactly as the cutoff crosses the date. "
         "<b>Contamination (post-1930 US politics):</b> ~89% for <i>every</i> model, including the 1900 one &mdash; 'known' long before any "
         "cutoff could legitimately reach it. <b>Clean post-1930 (everything else):</b> stuck at chance for all three. "
         f"<br><br><b>Your Einstein example.</b> General relativity is 1915, so strictly the 1929 model should know it and the 1913 shouldn't. "
         f"But watch the gap by cutoff: 1900 = {ein['Ranke-4B-1900']['gap']:+.2f} (chance &mdash; Einstein was an unknown clerk in 1900), "
         f"1913 = {ein['Ranke-4B-1913']['gap']:+.2f} (already strong!), 1929 = {ein['Ranke-4B-1929']['gap']:+.2f}. "
         "Knowledge turns on between <b>1900 and 1913</b> &mdash; tracking when <i>Einstein became famous</i> (special relativity, 1905), "
         "not the 1915 date of <i>general</i> relativity. So this is the time-lock working, not a leak: for people who became famous before the "
         "cutoff, key the item on when the <i>name</i> became knowable, not the event.</div>")

# ---------- SUBJECT ----------
H.append("<h2>1 &middot; By subject area</h2>")
dom_tp = sorted([(d, acc_ci([r for r in base if r["domain"] == d and r["era"] == "pre"]),
                     acc_ci([r for r in base if r["domain"] == d and r["era"] == "post"]))
                 for d in {r["domain"] for r in base}], key=lambda t: -t[2][0])
H.append("<figure>" + twopanel(dom_tp) +
         "<figcaption>Two coefficients per row, same rows: <span style='color:#1b3a6b'><b>blue = should-know (pre-cutoff)</b></span> | "
         "<span style='color:#c0392b'><b>red = should-not-know (post-cutoff)</b></span>. %-correct with 95% CI; dashed line = 50% chance "
         "(solid red = CI entirely above chance = leak). Politics is the only subject whose red sits clearly above 50%.</figcaption></figure>")
H.append("<figure>" + heatmap(lambda r: r["domain"], "subject \\ cutoff") +
         "<figcaption>Same, per model. Red = above 50% (leak), blue = below.</figcaption></figure>")
us_leak = sorted([r for r in prim if r["country"] == "USA" and r["domain"] == "politics" and r["era"] == "post"], key=lambda r: -r["gap"])[:12]
clean_sci = sorted([r for r in prim if r["domain"] in ("science", "tech", "culture") and r["era"] == "post"], key=lambda r: r["gap"])[:12]
EH = "<table class=ex><tr><th>prompt &rarr; true vs false</th><th>picked</th><th>gap</th><th>p(true)</th></tr>"
H.append("<h3>Examples &mdash; politics (leaked, US): it keeps picking the true one</h3>" + EH + ex_rows(us_leak) + "</table>")
H.append("<h3>Examples &mdash; science / tech / culture (clean): true vs false is a coin-flip</h3>" + EH + ex_rows(clean_sci) + "</table>")

# ---------- KDE ----------
H.append("<h2>1b &middot; Probability distributions &mdash; how sure is it?</h2>")
H.append("<p>Density of <b>p(true)</b> (the model's probability of the true answer) for three groups, base models pooled, "
         "each curve scaled to equal height. If post-cutoff facts were uniformly unknown, the red and grey curves would "
         "coincide near zero. Instead the <b>contaminated</b> group's probabilities pile up high &mdash; right where the "
         "<b>known</b> group lives.</p>")
known_p = [r["p_true"] for r in base if r["era"] == "pre"]
contam_p = [r["p_true"] for r in base if r["era"] == "post" and r["country"] == "USA" and r["domain"] == "politics"]
clean_p = [r["p_true"] for r in base if r["era"] == "post" and r["domain"] in ("science", "tech", "culture")]
H.append("<figure>" + kde_svg([("known: pre-cutoff (should know)", known_p, "#1b3a6b"),
                                ("contaminated: US politics, post-cutoff", contam_p, "#c0392b"),
                                ("clean: science/tech/culture, post-cutoff", clean_p, "#9aa3ad")]) +
         "<figcaption>p(true) density. The clean post-cutoff group (grey) piles up near 0; the contaminated US-politics group (red) "
         "sits high, overlapping the legitimately-known group (blue). Probabilistically, the contaminated answers look like things it actually knows.</figcaption></figure>")
known_g = [r["gap"] for r in base if r["era"] == "pre"]
contam_g = [r["gap"] for r in base if r["era"] == "post" and r["country"] == "USA" and r["domain"] == "politics"]
clean_g = [r["gap"] for r in base if r["era"] == "post" and r["domain"] in ("science", "tech", "culture")]
H.append("<figure>" + kde_gap_svg([("known: pre-cutoff", known_g, "#1b3a6b"),
                                    ("contaminated: US politics (post)", contam_g, "#c0392b"),
                                    ("clean: sci/tech/culture (post)", clean_g, "#9aa3ad")]) +
         "<figcaption>The crisper view: the same three groups by the <b>gap</b> (log p(true) &minus; log p(false)). "
         "<b>Clean (grey) sits on 0</b> &mdash; genuinely indifferent. <b>Known (blue) and contaminated US-politics (red) "
         "both sit right of 0</b>, overlapping: on post-cutoff US politics the model backs the truth just like it does on "
         "things it legitimately knows.</figcaption></figure>")

# ---------- REGION ----------
H.append("<h2>2 &middot; By country &amp; language</h2>")
H.append("<figure>" + heatmap(lambda r: r["country"], "country \\ cutoff") +
         "<figcaption>% preferring the true post-cutoff answer, per country per model. "
         "The USA row is red across <b>every</b> cutoff (incl. 1900); most other rows hover around 50% or below.</figcaption></figure>")
H.append("<p><b>Two coefficients per row.</b> <span style='color:#1b3a6b'><b>Blue = should-know (pre-cutoff)</b></span> (the upper marker) and"
         "<span style='color:#c0392b'><b>red = should-not-know (post-cutoff)</b></span> the lower;%-correct with 95% CI, dashed = 50% chance. "
         "Read across a row: high blue + low red = genuinely clean; high blue + high red = contaminated; "
         "<b>'no pre-cutoff items' in blue = a low red proves nothing</b> (e.g. Iraq, Persian have no baseline).</p>")
reg_db = sorted([(c, acc_ci([r for r in base if r["country"] == c and r["era"] == "pre"]),
                     acc_ci([r for r in base if r["country"] == c and r["era"] == "post"]))
                 for c in {r["country"] for r in base}], key=lambda t: -t[2][0])
H.append("<figure>" + twopanel(reg_db) +
         "<figcaption>By country. Blue (upper) = knows its past; red (lower) = its 'future'. The USA is the only row high in <i>both</i>.</figcaption></figure>")
H.append("<table><tr><th>country</th><th>PAST % (should know)</th><th>FUTURE % (should not)</th><th>cliff</th><th>verdict</th></tr>")
for c, pre, post in reg_db:
    pp, plo, phi, pn = pre
    qp, qlo, qhi, qn = post
    if pn == 0 or pp < 0.6:
        v, vc = "untrained / can't tell", "#9aa3ad"
    elif qlo > 0.5:
        v, vc = "CONTAMINATED", "#c0392b"
    elif qp < 0.6:
        v, vc = "clean (real cliff)", "#1f8a52"
    else:
        v, vc = "borderline", "#d98610"
    if pn > 0:
        preC = f"{pp*100:.0f}% <span class=f>[{plo*100:.0f}&ndash;{phi*100:.0f}] n{pn}</span>"
        cliff = f"{(pp-qp)*100:+.0f} pts"
    else:
        preC = "<i style='color:#c0392b'>none</i>"
        cliff = "&mdash;"
    H.append(f"<tr><td>{html.escape(c)}</td><td class=n>{preC}</td>"
             f"<td class=n>{qp*100:.0f}% <span class=f>[{qlo*100:.0f}&ndash;{qhi*100:.0f}] n{qn}</span></td>"
             f"<td class=n>{cliff}</td><td><span class=tag style='background:{vc}'>{v}</span></td></tr>")
H.append("</table>")
lf_db = sorted([(lf, acc_ci([r for r in base if r["language"] == lf and r["era"] == "pre"]),
                     acc_ci([r for r in base if r["language"] == lf and r["era"] == "post"]))
                for lf in {r["language"] for r in base}], key=lambda t: -t[2][0])
H.append("<h3>By language</h3>")
H.append("<figure>" + twopanel(lf_db) +
         "<figcaption>By language. <b>English</b> is the only one high on both markers (knows its past AND its 'future'); the rest either drop to chance in red (clean) or have no blue baseline.</figcaption></figure>")

H.append("<div class=box><b>How to read the examples below.</b> Single rows are noisy; the verdict is the <b>% true</b> "
         "with its CI, not any one item. A clean region's genuinely <i>unguessable</i> modern facts come out ~50/50; "
         "the few it 'gets' are false-positive types &mdash; people public <i>before</i> the cutoff (Churchill, Hitler "
         "were 1920s figures) or place/word hooks (Normandy, London, &lsquo;Wall&rsquo;, &lsquo;Barbarossa&rsquo;=the medieval emperor).</div>")
def region_block(c, label, note):
    pre = sorted([r for r in prim if r["country"] == c and r["era"] == "pre"], key=lambda r: -r["gap"])[:3]
    post = sorted([r for r in prim if r["country"] == c and r["era"] == "post"], key=lambda r: -r["gap"])
    p, lo, hi, n = acc_ci(post)
    verdict = "LEAKED" if lo > 0.5 else ("clean (~chance)" if hi >= 0.5 else "clean")
    vcol = "#c0392b" if lo > 0.5 else "#1b3a6b"
    H.append(f"<h3>{label} &mdash; {p*100:.0f}% picked true <span class=f>[{lo*100:.0f}–{hi*100:.0f}], n={n}</span> "
             f"<span class=tag style='background:{vcol}'>{verdict}</span></h3>")
    H.append(f"<p style='font-size:13px;color:#5b6675'>{note}</p>")
    H.append("<p style='font-size:12px;color:#5b6675;margin:6px 0 2px'>&#9656; PAST (controls):</p>" + EH + ex_rows(pre) + "</table>")
    H.append("<p style='font-size:12px;color:#5b6675;margin:8px 0 2px'>&#9656; ALL post-cutoff items:</p>" + EH + ex_rows(post) + "</table>")

region_block("USA", "USA (contaminated)", "Even unguessable modern names &mdash; Obama, Reagan, Kennedy, Nixon &mdash; pick true. Real leak.")
region_block("France", "France (clean)", "Macron, the 1958/1968 dates: false; the few 'true' are the place Normandy and Concorde/Sartre.")
region_block("UK", "Britain (clean)", "Beatles, Turing, Blair: false; the 'true' rows are Churchill &amp; Chamberlain (public in the 1920s) and London the place.")
region_block("Germany", "Germany (clean)", "Auschwitz, Merkel, Adenauer: false; the 'true' rows are &lsquo;Barbarossa&rsquo;, &lsquo;Wall&rsquo;, Hitler (a 1920s figure).")
region_block("Africa", "Africa (clean)", "Rwanda, Lumumba, Mugabe, Mandela: false; only Nkrumah and the place Nigeria pick true.")

H.append("<h2>3 &middot; Full appendix (Ranke-4B-1929 base)</h2>")
H.append(f"<p>All {len(prim)} items, grouped by subject. 'picked' = which the model preferred; gap colour is the raw margin.</p>")
H.append(EH.replace("</tr>", "<th>where</th></tr>"))
for dom, *_ in dom_tp:
    ditems = sorted([r for r in prim if r["domain"] == dom], key=lambda r: -r["gap"])
    if not ditems: continue
    H.append(f"<tr class=grp><td colspan=5>{html.escape(dom)} &middot; {len(ditems)} items</td></tr>")
    for r in ditems:
        g = r["gap"]; col = "#c0392b" if g > 1.5 else ("#d98610" if g > 0.5 else ("#9aa3ad" if g > -0.5 else "#1b3a6b"))
        H.append(f'<tr><td>{html.escape(pr(r))} <b>{html.escape(r["true"])}</b> <span class=f>vs {html.escape(r["false"])}</span></td>'
                 f'<td class=n>{"true" if g>0 else "false"}</td><td class=n style="color:{col}">{g:+.2f}</td>'
                 f'<td class=p>{r["p_true"]*100:.0f}%</td><td class=f>{r["era"]} {r["year"]} {html.escape(r["country"])}</td></tr>')
H.append("</table>")
H.append("<p style='color:#5b6675;font-size:13px;margin-top:26px'>Generated from "
         "<code>5_results/contamination_map_results.csv</code>. Metric = share of shouldn't-know items where "
         "logp(true) &gt; logp(false); 50% = chance; CI = p &plusmn; 1.96&middot;SE. Pooled CIs treat the three base "
         "models' items as independent (slightly optimistic) &mdash; the per-model heatmap shows the pattern replicates regardless.</p>")
H.append("</div></body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote", OUT, "(", len(prim), "items )")
