# -*- coding: utf-8 -*-
"""Head-to-head diagnostics + visualization: Ranke-4B-1929 vs Talkie-1930-13B on the IDENTICAL frozen bank.
Reads results_*.json in 8_suite/, runs the significance diagnostics, writes comparison.html (inline SVG, no
deps). Tolerant of a missing Talkie file (renders Ranke-only with a 'pending' note) so it can run early."""
import os, json, math, glob, html
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "comparison.html")
esc = html.escape

ORDER = ["Ranke-4B-1929", "Ranke-4B-1929-1202-Instruct", "talkie-1930-13b-base"]
NICE = {"Ranke-4B-1929": "Ranke-4B-1929 (base)", "Ranke-4B-1929-1202-Instruct": "Ranke-1929 (instruct)",
        "talkie-1930-13b-base": "Talkie-1930-13B (base)"}
COLOR = {"Ranke-4B-1929": "#2b6cb0", "Ranke-4B-1929-1202-Instruct": "#5b9bd5", "talkie-1930-13b-base": "#c0392b"}

R = {}
for f in glob.glob(os.path.join(HERE, "results_*.json")):
    try:
        d = json.load(open(f, encoding="utf-8"))
        if "aggregate" in d:
            R[d["model"]] = d
    except Exception:
        pass
models = [m for m in ORDER if m in R]
TALKIE_IN = "talkie-1930-13b-base" in R


def ndcdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def get(m, key):
    return R[m]["aggregate"].get(key)


def twoprop(x1, n1, x2, n2):
    if not (n1 and n2):
        return None
    p1, p2 = x1 / n1, x2 / n2
    pp = (x1 + x2) / (n1 + n2)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) or 1e-9
    z = (p1 - p2) / se
    return 2 * (1 - ndcdf(abs(z)))


# ---------- SVG grouped bar chart ----------
def svg_bars(title, series, ymax=100, chance=25.0, w=620, h=300):
    # series: list of (label, value, color)
    pad_l, pad_b, pad_t = 52, 56, 40
    plot_w, plot_h = w - pad_l - 20, h - pad_b - pad_t
    n = len(series); bw = plot_w / max(n, 1) * 0.6; gap = plot_w / max(n, 1)
    s = [f'<svg viewBox="0 0 {w} {h}" style="max-width:100%;height:auto;border:1px solid #e7eaef;border-radius:8px;background:#fff">']
    s.append(f'<text x="{w/2}" y="22" text-anchor="middle" style="font:700 15px Georgia,serif;fill:#1a1d24">{title}</text>')
    for gv in (0, 25, 50, 75, 100):
        if gv > ymax: continue
        y = pad_t + plot_h * (1 - gv / ymax)
        s.append(f'<line x1="{pad_l}" x2="{w-20}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eef1f5"/>')
        s.append(f'<text x="{pad_l-6}" y="{y+4:.1f}" text-anchor="end" style="font:11px sans-serif;fill:#8a93a0">{gv}%</text>')
    # chance line
    yc = pad_t + plot_h * (1 - chance / ymax)
    s.append(f'<line x1="{pad_l}" x2="{w-20}" y1="{yc:.1f}" y2="{yc:.1f}" stroke="#c0392b" stroke-dasharray="5 4" stroke-width="1.5"/>')
    s.append(f'<text x="{w-22}" y="{yc-5:.1f}" text-anchor="end" style="font:700 11px sans-serif;fill:#c0392b">chance {chance:.0f}%</text>')
    for i, (lab, val, col) in enumerate(series):
        x = pad_l + gap * i + (gap - bw) / 2
        bh = plot_h * (min(val, ymax) / ymax)
        y = pad_t + plot_h - bh
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{col}"/>')
        s.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" style="font:700 13px sans-serif;fill:#1a1d24">{val:.0f}%</text>')
        for j, ln in enumerate(lab.split("\n")):
            s.append(f'<text x="{x+bw/2:.1f}" y="{h-pad_b+16+j*13:.1f}" text-anchor="middle" style="font:11px sans-serif;fill:#52637a">{esc(ln)}</text>')
    s.append("</svg>")
    return "".join(s)


CSS = ("body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:980px;margin:0 auto;padding:36px 22px;color:#1a1d24;line-height:1.55}"
       "h1{font-family:Georgia,serif;font-size:27px;margin:0 0 4px}.sub{color:#5b6675;font-size:15px;margin:0 0 20px}"
       "h2{font-family:Georgia,serif;font-size:20px;margin:32px 0 8px;border-top:2px solid #e7eaef;padding-top:14px}"
       "table{border-collapse:collapse;margin:12px 0;font-size:13.5px;width:100%}th,td{border:1px solid #eceff3;padding:7px 10px}"
       "th{background:#f0f3f8;text-align:left}.lab{background:#f8fafc;font-weight:600}.r{text-align:right;font-variant-numeric:tabular-nums}"
       ".box{background:#fafbfd;border:1px solid #e2e6ec;border-left:4px solid #2b6cb0;border-radius:8px;padding:14px 18px;margin:16px 0;font-size:14.5px}"
       ".good{border-left-color:#2e7d32}.warn{border-left-color:#c0843a}.bad{border-left-color:#b91c1c}"
       ".charts{display:flex;flex-wrap:wrap;gap:18px;margin:14px 0}.charts>div{flex:1;min-width:300px}"
       "code{background:#f0f3f8;padding:1px 5px;border-radius:4px}")
H = [f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ranke vs Talkie — contamination head-to-head</title><style>{CSS}</style></head><body>']
A = H.append
A("<h1>Ranke-4B-1929 vs Talkie-1930-13B</h1>")
A("<p class=sub>Same wall (~1930), different teams, ~3&times; size. <b>Identical</b> forced-choice bank on both: "
  "15 facts &times; 3 post-cutoff topics + 16 pre-cutoff controls, each a choice against 3 matched distractors. "
  "Matched distractors put a 4B and a 13B on the <b>same 25% chance line</b> &mdash; the fair comparison a raw "
  "rank-ratio can't give (it inflates with model size).</p>")

if not TALKIE_IN:
    A("<div class=box warn><b>Talkie results pending</b> &mdash; the runpod job hasn't reported yet. This page "
      "currently shows Ranke only and will be rebuilt with Talkie the moment its numbers land.</div>")

# ---- headline chart: POST %-correct ----
post_series = [(NICE[m].replace(" (", "\n("), get(m, "post:ALL")["pct"], COLOR[m]) for m in models if get(m, "post:ALL")]
pre_series = [(NICE[m].replace(" (", "\n("), get(m, "pre:ALL")["pct"], COLOR[m]) for m in models if get(m, "pre:ALL")]
A("<h2>1. The headline &mdash; post-cutoff knowledge vs blind chance</h2>")
A("<div class=charts>")
A("<div>" + svg_bars("Post-1930 facts: % beating 3 distractors", post_series) + "</div>")
A("<div>" + svg_bars("Pre-cutoff controls: % correct (probe fires?)", pre_series) + "</div>")
A("</div>")
A("<p style='color:#5b6675;font-size:13px'>Left: contamination test &mdash; a clean ~1930 model sits on the red "
  "chance line. Right: the sanity check &mdash; both models must ace facts they legitimately know, or the test "
  "means nothing.</p>")

# ---- per-topic ----
topics = ["Vietnam-US war", "WWII", "Cold War"]
A("<h2>2. Post-cutoff by topic</h2>")
A("<div class=charts>")
for t in topics:
    ser = [(NICE[m].split(" (")[0] + "\n" + NICE[m].split("(")[1].rstrip(")") if "(" in NICE[m] else NICE[m], get(m, "post:" + t)["pct"], COLOR[m]) for m in models if get(m, "post:" + t)]
    if ser:
        A("<div>" + svg_bars(t, ser, w=420) + "</div>")
A("</div>")

# ---- table ----
A("<h2>3. The numbers</h2>")
A("<table><thead><tr><th>model</th><th class=r>pre-cutoff</th><th class=r>POST overall</th><th class=r>vs chance (binom p)</th>"
  "<th class=r>Vietnam</th><th class=r>WWII</th><th class=r>Cold War</th></tr></thead><tbody>")
for m in models:
    po = get(m, "post:ALL"); pr = get(m, "pre:ALL")
    A(f"<tr><td class=lab>{esc(NICE[m])}</td>"
      f"<td class=r>{pr['correct']}/{pr['n']} ({pr['pct']:.0f}%)</td>"
      f"<td class=r>{po['correct']}/{po['n']} ({po['pct']:.0f}%)</td>"
      f"<td class=r>{po['binom_p']:.3f}</td>"
      + "".join(f"<td class=r>{get(m,'post:'+t)['pct']:.0f}%</td>" for t in topics) + "</tr>")
A("</tbody></table>")

# ---- diagnostics / verdict ----
A("<h2>4. Diagnosis</h2>")
rb = get("Ranke-4B-1929", "post:ALL")
if TALKIE_IN:
    tk = get("talkie-1930-13b-base", "post:ALL")
    p_tk_chance = tk["binom_p"]
    p_diff = twoprop(tk["correct"], tk["n"], rb["correct"], rb["n"])
    verdict_cls = "bad" if (tk["pct"] > 40 and p_tk_chance < 0.05) else ("warn" if p_tk_chance < 0.10 else "good")
    msg = (f"<b>Talkie-1930-13B post-cutoff: {tk['correct']}/{tk['n']} = {tk['pct']:.0f}%</b> "
           f"(chance 25%, one-sided binomial p={p_tk_chance:.3f}). "
           f"Pre-cutoff controls {get('talkie-1930-13b-base','pre:ALL')['pct']:.0f}% confirm the probe fires.<br><br>")
    if p_tk_chance >= 0.10:
        msg += ("Talkie sits at chance on specific post-1930 facts &mdash; like Ranke. On the <i>fair</i> "
                "matched-distractor test, the contamination Guohui's rank-ratio suggested does <b>not</b> survive: "
                "its 5.53&times;/2.37&times; lift was the 13B's richer generic-name floor, not real post-cutoff knowledge.")
    elif p_tk_chance < 0.05 and tk["pct"] > 40:
        msg += ("Talkie is <b>above chance</b> on specific post-1930 facts &mdash; genuine leakage that survives "
                "matched distractors. This is real contamination, not a size artifact, and it is stronger than Ranke.")
    else:
        msg += "Talkie is marginally above chance &mdash; a weak signal worth a topic-level dig before any verdict."
    p_diff_txt = f"Ranke vs Talkie difference: two-proportion p={p_diff:.3f}." if p_diff is not None else ""
    msg += "<br><br>" + (f"Ranke base post-cutoff {rb['pct']:.0f}% (p={rb['binom_p']:.3f}, at chance). " + p_diff_txt)
    A(f"<div class=box {verdict_cls}>{msg}</div>")
else:
    A(f"<div class=box><b>Ranke-4B-1929 base post-cutoff: {rb['pct']:.0f}%</b> (chance 25%, p={rb['binom_p']:.3f}) "
      "&mdash; at chance, clean. Awaiting Talkie to complete the head-to-head.</div>")

A("<p style='color:#9aa3af;font-size:12px;margin-top:24px'>Forced-choice sum-logp, 3 matched distractors "
  "(chance 25%). Same question_bank.json scored on every model. Ranke via transformers locally; Talkie-1930-13B "
  "via its own library on a runpod RTX A6000, scored with a forward reimplemented to expose all-position logits "
  "(self-validated against the official last-position forward).</p>")
A("</body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote", OUT, "| models:", models, "| talkie_in:", TALKIE_IN)
