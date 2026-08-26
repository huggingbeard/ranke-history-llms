"""
Updated KDE set on the CORRECTED length-matched metric (lm_battery.csv, matched pairs,
pooled over the 3 base models with each model's own era). New file -> does NOT touch
kde_extras.html / contamination_map.html.
Per category, two views:
  (a) diff from 50%  = P(true vs false) - 50%  in PROBABILITY space  (0 = chance)
  (b) delta          = logp(true) - logp(false)  in LOG-PROB space   (0 = indifferent)
Two curves each: should-know (PRE, blue) vs should-not-know (POST, red).
"""
import csv, html, math
import numpy as np

CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\lm_battery.csv"
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\kde_lengthmatched.html"
BASE = {"Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929"}
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["matched"] == "1" and r["model"] in BASE]
for r in rows:
    g = float(r["gap"]); r["gap"] = g
    r["d50"] = (1.0 / (1.0 + math.exp(-g)) - 0.5) * 100.0   # % points from chance

CATS = [
    ("USA", "politics", "&#128308; CONTAMINATED &mdash; post-cutoff (red) sits right on the should-know pile; significant (pooled p&#8776;0.02 vs chance)"),
    ("UK", "politics", "&#128994; clean &mdash; post-cutoff collapses to chance"),
    ("France", "politics", "&#128994; clean &mdash; post-cutoff at/below chance"),
    ("Germany", "politics", "&#128993; mostly clean, but post-cutoff leans slightly positive (not significant, n.s.)"),
    ("Italy", "politics", "&#128993; bimodal &mdash; some Fascist-era facts (Mussolini) leak for the 1900/1913 models; small n, not significant"),
    ("Spain", "politics", "&#128994; clean (small n)"),
    ("Russia", "politics", "&#128994; clean (small n)"),
    ("West", "culture", "&#128994; clean &mdash; post-cutoff at/below chance"),
    ("West", "science", "&#128993; bimodal &mdash; ubiquitous facts (Sputnik, DNA, Moon landing) leak; net above chance but NOT significant"),
]
sub = lambda cat, era: [r for r in rows if (r["country"], r["domain"]) == cat and r["era"] == era]


def kde(vals, grid, bw):
    d = np.zeros_like(grid)
    for v in vals:
        d += np.exp(-0.5 * ((grid - float(v)) / bw) ** 2)
    return d / d.max() if (len(vals) and d.max() > 0) else d


def chart(curves, xmin, xmax, ticks, zero, xlabel, bw, w=430, hh=210):
    grid = np.linspace(xmin, xmax, 161)
    x0, x1, y0, ytop = 40, w - 12, hh - 28, 22
    X = lambda v: x0 + (x1 - x0) * (v - xmin) / (xmax - xmin)
    s = [f'<svg viewBox="0 0 {w} {hh}" width="100%">']
    for v, lab in ticks:
        x = X(v); isz = abs(v - zero) < 1e-9
        s.append(f'<line x1="{x:.0f}" y1="{ytop}" x2="{x:.0f}" y2="{y0}" stroke="{"#c0392b" if isz else "#eef1f6"}" stroke-dasharray="{"5 3" if isz else "0"}"/>')
        s.append(f'<text x="{x:.0f}" y="{y0+13}" text-anchor="middle" font-size="9.5" fill="#9aa3ad">{lab}</text>')
    s.append(f'<text x="{X(zero):.0f}" y="{ytop-3}" text-anchor="middle" font-size="9" fill="#c0392b">{"chance" if xmin<0 and zero==0 and xmax>40 else "indifferent"}</text>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{hh-2}" text-anchor="middle" font-size="10" fill="#5b6675">{xlabel}</text>')
    ly = ytop + 1
    for lab, vals, col in curves:
        if not vals:
            continue
        d = kde(vals, grid, bw)
        pts = " ".join(f"{X(grid[i]):.1f},{y0-(y0-ytop)*float(d[i]):.1f}" for i in range(len(grid)))
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        s.append(f'<rect x="{x0+4}" y="{ly}" width="10" height="10" fill="{col}"/>'
                 f'<text x="{x0+18}" y="{ly+9}" font-size="10" fill="#1a1d24">{html.escape(lab)} (n{len(vals)})</text>')
        ly += 15
    s.append("</svg>")
    return "".join(s)


def pair(pre_d, post_d, pre_g, post_g):
    BLUE, RED = "#1b3a6b", "#c0392b"
    a = chart([("should-know (pre)", pre_d, BLUE), ("should-NOT-know (post)", post_d, RED)],
              -50, 50, [(-50, "-50"), (-25, "-25"), (0, "0"), (25, "+25"), (50, "+50")], 0,
              "P(true vs false) - 50%  (percentage points from chance)", 4.5)
    b = chart([("should-know (pre)", pre_g, BLUE), ("should-NOT-know (post)", post_g, RED)],
              -8, 8, [(-8, "-8"), (-4, "-4"), (0, "0"), (4, "+4"), (8, "+8")], 0,
              "delta = log p(true) - log p(false)", 0.8)
    return f'<div class=row><div class=cell><div class=t>(a) distance from 50% chance</div>{a}</div>' \
           f'<div class=cell><div class=t>(b) delta (log-prob gap)</div>{b}</div></div>'


CSS = ("body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1a1d24;line-height:1.45}"
       ".wrap{max-width:1000px;margin:0 auto;padding:34px 24px 80px}h1{font-family:Georgia,serif;font-size:26px;margin:0 0 4px}"
       ".sub{color:#5b6675;font-size:15px;margin:0 0 20px}h2{font-family:Georgia,serif;font-size:19px;margin:30px 0 2px;border-top:2px solid #e2e6ec;padding-top:12px}"
       ".cap{color:#5b6675;font-size:13px;margin:2px 0 6px}.row{display:flex;gap:14px;flex-wrap:wrap}"
       ".cell{flex:1;min-width:330px;background:#fafbfd;border:1px solid #eef1f6;border-radius:9px;padding:8px 10px}"
       ".t{font-size:12px;color:#5b6675;font-weight:600;margin:2px 0 2px}"
       ".box{background:#f6f8fb;border:1px solid #e2e6ec;border-left:5px solid #1b3a6b;border-radius:9px;padding:12px 16px;margin:14px 0}code{background:#eef1f6;padding:1px 5px;border-radius:4px}")

H = ["<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>",
     f"<title>Ranke KDE - length-matched</title><style>{CSS}</style></head><body><div class=wrap>"]
H.append("<h1>Probability-difference KDE &mdash; corrected (length-matched)</h1>")
H.append("<p class=sub>Minimal pairs scored only where true & false answers have the <b>same token length</b> "
         "(the one bias-free comparison), whole answer teacher-forced, pooled over the three base models. "
         "<span style='color:#1b3a6b'><b>Blue = should-know (pre-cutoff)</b></span>, "
         "<span style='color:#c0392b'><b>red = should-NOT-know (post-cutoff)</b></span>. Each curve scaled to equal height.</p>")

# Overview: should-not-know, US politics vs the clean rest
us_post_d = [r["d50"] for r in sub(("USA", "politics"), "post")]
us_post_g = [r["gap"] for r in sub(("USA", "politics"), "post")]
rest_post_d = [r["d50"] for r in rows if r["era"] == "post" and not (r["country"] == "USA" and r["domain"] == "politics")]
rest_post_g = [r["gap"] for r in rows if r["era"] == "post" and not (r["country"] == "USA" and r["domain"] == "politics")]
pre_all_d = [r["d50"] for r in rows if r["era"] == "pre"]
pre_all_g = [r["gap"] for r in rows if r["era"] == "pre"]
GREY = "#9aa3ad"
ov_a = chart([("should-know (pre, all)", pre_all_d, "#1b3a6b"),
              ("should-NOT-know: US politics", us_post_d, "#c0392b"),
              ("should-NOT-know: rest of West", rest_post_d, GREY)],
             -50, 50, [(-50, "-50"), (-25, "-25"), (0, "0"), (25, "+25"), (50, "+50")], 0,
             "P(true vs false) - 50%", 4.5, w=470, hh=230)
ov_b = chart([("should-know (pre, all)", pre_all_g, "#1b3a6b"),
              ("should-NOT-know: US politics", us_post_g, "#c0392b"),
              ("should-NOT-know: rest of West", rest_post_g, GREY)],
             -8, 8, [(-8, "-8"), (-4, "-4"), (0, "0"), (4, "+4"), (8, "+8")], 0,
             "delta = log p(true) - log p(false)", 0.8, w=470, hh=230)
H.append("<h2>Overview &mdash; the localized leak</h2>")
H.append("<p class=cap>Pre-cutoff (blue) sits far right of chance everywhere. Post-cutoff splits: "
         "<b>US politics (red) is shifted right toward the pre-cutoff pile</b>, while the rest of the West "
         "(grey) sits on chance / 0.</p>")
H.append(f'<div class=row><div class=cell><div class=t>(a) distance from 50% chance</div>{ov_a}</div>'
         f'<div class=cell><div class=t>(b) delta (log-prob gap)</div>{ov_b}</div></div>')

for c, d, cap in CATS:
    cat = (c, d)
    H.append(f"<h2>{c} &middot; {d}</h2>")
    H.append(f"<p class=cap>{cap}</p>")
    H.append(pair([r["d50"] for r in sub(cat, "pre")], [r["d50"] for r in sub(cat, "post")],
                  [r["gap"] for r in sub(cat, "pre")], [r["gap"] for r in sub(cat, "post")]))

H.append("<div class=box><b>How to read it.</b> In (a), 0 = a coin-flip between true and false; +50 = the model is "
         "certain of the truth. In (b), 0 = indifferent; positive = it backs the truth. A <b>clean</b> time-locked "
         "model shows red sitting on 0 while blue sits to the right (a real cliff). <b>US politics</b> is the only "
         "category whose red curve clearly climbs onto the blue should-know pile &mdash; it knows what it shouldn't "
         "(significant). <b>Science and Italy</b> are <i>bimodal</i>: a cluster of strongly-leaked, globally-ubiquitous "
         "facts (Sputnik, DNA, the Moon landing; Mussolini) plus a cluster at chance &mdash; net above 50% but not "
         "statistically significant at this N, so suggestive rather than proven. UK, France, Spain, Russia and culture "
         "sit cleanly at or below chance.</div>")
H.append("</div></body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote", OUT)
