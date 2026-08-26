"""
Standalone supplementary KDE page (does NOT touch contamination_map.html).
  Chart 1: KDE of the GAP = logp(true) - logp(false), by category (sharper separation).
  Chart 2: KDE of p(true), with the contaminated group split into 'got it right' vs 'got it
           wrong' (by gap sign, not by p — avoids circularity), to show the bimodality.
Pooled over the three base models.
"""
import csv, html
import numpy as np

CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\contamination_map_results.csv"
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\kde_extras.html"
BASES = {"Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929"}
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["model"] in BASES]
for r in rows:
    r["gap"] = float(r["gap"]); r["p"] = float(r["p_true"])

pre = lambda r: r["era"] == "pre"
uspol = lambda r: r["era"] == "post" and r["country"] == "USA" and r["domain"] == "politics"
clean = lambda r: r["era"] == "post" and r["domain"] in ("science", "tech", "culture")

known_g = [r["gap"] for r in rows if pre(r)]
contam_g = [r["gap"] for r in rows if uspol(r)]
clean_g = [r["gap"] for r in rows if clean(r)]
known_p = [r["p"] for r in rows if pre(r)]
clean_p = [r["p"] for r in rows if clean(r)]
us_right = [r["p"] for r in rows if uspol(r) and r["gap"] > 0]
us_wrong = [r["p"] for r in rows if uspol(r) and r["gap"] <= 0]


def kde(vals, grid, bw):
    d = np.zeros_like(grid)
    for v in vals:
        d += np.exp(-0.5 * ((grid - float(v)) / bw) ** 2)
    return d / d.max() if (len(vals) and d.max() > 0) else d


def chart(cats, xmin, xmax, ticks, zero, xlabel, bw, w=780, hh=265):
    grid = np.linspace(xmin, xmax, 181)
    x0, x1, y0, ytop = 46, w - 16, hh - 30, 24
    X = lambda v: x0 + (x1 - x0) * (v - xmin) / (xmax - xmin)
    s = [f'<svg viewBox="0 0 {w} {hh}" width="100%">']
    for v, lab in ticks:
        x = X(v); isz = abs(v - zero) < 1e-9
        s.append(f'<line x1="{x:.0f}" y1="{ytop}" x2="{x:.0f}" y2="{y0}" stroke="{"#c0392b" if isz else "#eef1f6"}" stroke-dasharray="{"5 3" if isz else "0"}"/>')
        s.append(f'<text x="{x:.0f}" y="{y0+14}" text-anchor="middle" font-size="10" fill="#9aa3ad">{lab}</text>')
    s.append(f'<text x="{X(zero):.0f}" y="{ytop-4}" text-anchor="middle" font-size="9.5" fill="#c0392b">{"indifferent (gap 0)" if xmin<0 else "50% = chance"}</text>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{hh-2}" text-anchor="middle" font-size="10.5" fill="#5b6675">{xlabel}</text>')
    ly = ytop + 2
    for lab, vals, col in cats:
        if not vals:
            continue
        d = kde(vals, grid, bw)
        pts = " ".join(f"{X(grid[i]):.1f},{y0-(y0-ytop)*float(d[i]):.1f}" for i in range(len(grid)))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        s.append(f'<rect x="{x1-300}" y="{ly}" width="11" height="11" fill="{col}"/><text x="{x1-284}" y="{ly+10}" font-size="11" fill="#1a1d24">{html.escape(lab)} (n{len(vals)})</text>')
        ly += 17
    s.append("</svg>")
    return "".join(s)


CSS = ("body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1a1d24;line-height:1.5}"
       ".wrap{max-width:880px;margin:0 auto;padding:40px 24px 80px}h1{font-family:Georgia,serif;font-size:28px;margin:0 0 4px}"
       ".sub{color:#5b6675;font-size:16px;margin:0 0 22px}h2{font-family:Georgia,serif;font-size:21px;margin:36px 0 8px;border-top:2px solid #e2e6ec;padding-top:14px}"
       "figure{margin:14px 0;background:#fafbfd;border:1px solid #eef1f6;border-radius:10px;padding:14px}"
       "figcaption{color:#5b6675;font-size:13.5px;margin-top:8px}.box{background:#f6f8fb;border:1px solid #e2e6ec;border-left:5px solid #1b3a6b;border-radius:10px;padding:14px 18px;margin:16px 0}code{background:#eef1f6;padding:1px 5px;border-radius:4px}")

H = [f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
     f"<title>Ranke KDE — supplementary</title><style>{CSS}</style></head><body><div class=wrap>"]
H.append("<h1>Probability distributions — supplementary</h1>")
H.append("<p class=sub>Companion to <code>contamination_map.html</code> (which is unchanged). Base models pooled; "
         "each curve scaled to equal height (shape comparison).</p>")

H.append("<h2>1 &middot; The gap, log p(true) &minus; log p(false)</h2>")
H.append("<figure>" + chart(
    [("known: pre-cutoff (should know)", known_g, "#1b3a6b"),
     ("contaminated: US politics, post", contam_g, "#c0392b"),
     ("clean: science/tech/culture, post", clean_g, "#9aa3ad")],
    -8, 8, [(-8, "-8"), (-4, "-4"), (0, "0"), (4, "+4"), (8, "+8")], 0,
    "gap = log p(true) - log p(false)   (0 = indifferent / coin-flip)", 0.8) +
    "<figcaption>The gap separates the three groups far more sharply than raw probability. "
    "<b>Clean (grey) sits squarely on 0</b> &mdash; the model genuinely can't choose true from false. "
    "<b>Known (blue) and contaminated US-politics (red) both sit well to the right of 0</b>, overlapping each other: "
    "on post-cutoff US politics the model backs the truth just like it does on things it legitimately knows.</figcaption></figure>")

H.append("<h2>2 &middot; p(true), with the contaminated group split</h2>")
H.append("<figure>" + chart(
    [("known: pre-cutoff", known_p, "#1b3a6b"),
     ("US politics it gets RIGHT (gap>0)", us_right, "#c0392b"),
     ("US politics it gets WRONG (gap<=0)", us_wrong, "#e0a39e"),
     ("clean: sci/tech/culture, post", clean_p, "#9aa3ad")],
    0, 1, [(0, "0%"), (.25, "25%"), (.5, "50%"), (.75, "75%"), (1, "100%")], 0.5,
    "p(true): the model's probability of the true answer", 0.05) +
    "<figcaption>The pooled 'contaminated' curve in the main doc had a long tail because it mixes two populations. "
    "Split by whether the model actually prefers the truth (gap&gt;0 vs not &mdash; not by p, which would be circular): "
    "the items it <b>gets right</b> (dark red) pile up high alongside the <b>known</b> group (blue); the ones it "
    "<b>gets wrong</b> (light red) sit down near the <b>clean</b> group (grey). So 'contaminated US politics' is really "
    "'partly strongly leaked, partly not yet' &mdash; and the strongly-leaked part is probabilistically identical to genuine knowledge.</figcaption></figure>")

H.append("<div class=box><b>Takeaway.</b> Whether you look at the <i>gap</i> (chart 1) or split p(true) (chart 2), the "
         "story is the same: a clean post-cutoff fact is a coin-flip (gap 0 / p near 0), whereas the leaked US-politics "
         "facts behave exactly like the model's legitimate pre-cutoff knowledge. The gap view is the crisper one for a referee.</div>")
H.append("</div></body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote", OUT)
