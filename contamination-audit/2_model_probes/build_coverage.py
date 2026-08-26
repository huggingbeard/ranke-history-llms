# -*- coding: utf-8 -*-
"""Analyse the dense probes: era x region recognition HEATMAP (coverage_results.csv) + association-
lift group summary incl. the PRE_STRONG vs PRE_WEAK 'Hindenburg-zone' contrast (assoc_results2.csv).
Writes coverage.html and prints the key numbers."""
import csv, os, statistics as st, html
R = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\\"
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\coverage.html"
esc = html.escape
cov = list(csv.DictReader(open(R + "coverage_results.csv", encoding="utf-8")))
asc = list(csv.DictReader(open(R + "assoc_results2.csv", encoding="utf-8")))
ERAS = ["E1<=1789", "E2 1789-1815", "E3 1815-1865", "E4 1865-1900", "E5 1900-1914", "E6 1914-1929"]
REGS = ["BR", "US", "FR", "DE", "RU"]
RNAME = {"BR": "British", "US": "American", "FR": "French", "DE": "German", "RU": "Russian"}


def pct(era, reg):
    rows = [int(r["correct"]) for r in cov if r["era"] == era and r["region"] == reg]
    return (100 * sum(rows) / len(rows), len(rows)) if rows else (None, 0)


def color(p):
    if p is None:
        return "#eee", "#999"
    # red(0) -> yellow(50) -> green(100)
    if p <= 50:
        f = p / 50; r, g, b = 192 + int((224 - 192) * f), 57 + int((192 - 57) * f), 47 + int((64 - 47) * f)
    else:
        f = (p - 50) / 50; r, g, b = 224 - int((224 - 46) * f), 192 - int((192 - 125) * f), 64 - int((64 - 50) * f)
    return f"rgb({r},{g},{b})", ("#fff" if p > 70 or p < 30 else "#1a1d24")


print("ERA x REGION recognition %-correct (pooled over all 4 models):")
print(f'{"era":<14}' + "".join(f"{RNAME[x]:>10}" for x in REGS))
for e in ERAS:
    line = f"{e:<14}"
    for reg in REGS:
        p, n = pct(e, reg)
        line += f"{(f'{p:.0f}%' if p is not None else '-'):>10}"
    print(line)

# association groups
print("\nASSOCIATION-LIFT by group (mean over triples), per model:")
mods = sorted(set(r["model"] for r in asc), key=lambda m: ("Instruct" in m, m))
for grp in ["POST", "PRE_STRONG", "PRE_WEAK", "NULL"]:
    line = f"  {grp:<11}"
    for m in mods:
        v = [float(r["lift"]) for r in asc if r["model"] == m and r["kind"] == grp]
        line += f"{m.replace('Ranke-4B-',''):>6}={st.mean(v):+.2f} " if v else ""
    print(line)
print("\nPRE_WEAK (Hindenburg-zone) per-triple lift, 1929 base [should be high if known, near 0 if thin]:")
for r in [x for x in asc if x["model"] == "Ranke-4B-1929" and x["kind"] == "PRE_WEAK"]:
    print(f'  {r["label"]:<28} {float(r["lift"]):+.2f}')

# ---------- HTML ----------
def cell(era, reg):
    p, n = pct(era, reg)
    bg, fg = color(p)
    return f'<td style="background:{bg};color:{fg};text-align:center">{(f"{p:.0f}%" if p is not None else "&ndash;")}<br><span style="font-size:10px;opacity:.7">n{n}</span></td>'

CSS = ("body{font-family:-apple-system,Segoe UI,Arial,sans-serif;max-width:920px;margin:0 auto;padding:36px 22px;color:#1a1d24}"
       "h1{font-family:Georgia,serif;font-size:25px;margin:0 0 4px}.sub{color:#5b6675;font-size:15px;margin:0 0 18px}"
       "h2{font-family:Georgia,serif;font-size:19px;margin:28px 0 6px;border-top:2px solid #e7eaef;padding-top:12px}"
       "table{border-collapse:collapse;margin:10px 0;font-size:13px}th,td{border:1px solid #fff;padding:7px 10px}"
       "th{background:#f0f3f8;color:#1a1d24}.lab{text-align:left;background:#f6f8fb;font-weight:600}"
       ".box{background:#fafbfd;border:1px solid #e2e6ec;border-radius:9px;padding:12px 16px;margin:12px 0;font-size:14px}")
H = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
     f"<title>Ranke knowledge envelope</title><style>{CSS}</style></head><body>"]
A = H.append
A("<h1>The knowledge envelope &mdash; era &times; region</h1>")
A("<p class=sub>Recognition (forced-choice, length-matched) of <b>should-know</b> facts, pooled over all four models. "
  "Green = solid, red = at chance (50%). This is what a 1929 synthetic respondent reliably knows &mdash; and why it "
  "nails Nelson but fumbles Hindenburg.</p>")
A("<table><thead><tr><th></th>" + "".join(f"<th>{RNAME[x]}</th>" for x in REGS) + "</tr></thead><tbody>")
for e in ERAS:
    A(f'<tr><td class=lab>{esc(e)}</td>' + "".join(cell(e, reg) for reg in REGS) + "</tr>")
A("</tbody></table>")
A("<div class=box><b>Read:</b> knowledge is strong in the well-documented 19th century and the Anglo world, and "
  "thins toward the <b>1914&ndash;1929</b> row and the German/Russian columns &mdash; the model's corpus is older and "
  "English-skewed. Hindenburg (German, 1914&ndash;29) sits in the weakest corner; Nelson (British, 1805) in the strongest.</div>")

A("<h2>Association-lift by group</h2>")
A("<table><thead><tr><th class=lab>group</th>" + "".join(f"<th>{m.replace('Ranke-4B-','')}</th>" for m in mods) + "</tr></thead><tbody>")
GL = {"POST": "post-cutoff events (leak test)", "PRE_STRONG": "19th-c / Anglo (control, should be high)",
      "PRE_WEAK": "1914-29 / Continental (Hindenburg zone)", "NULL": "matched null (~0)"}
for grp in ["POST", "PRE_STRONG", "PRE_WEAK", "NULL"]:
    row = f'<tr><td class=lab>{GL[grp]}</td>'
    for m in mods:
        v = [float(r["lift"]) for r in asc if r["model"] == m and r["kind"] == grp]
        row += f"<td style='text-align:center'>{(f'{st.mean(v):+.2f}' if v else '&ndash;')}</td>"
    A(row + "</tr>")
A("</tbody></table>")
A("<div class=box>If <b>PRE_STRONG &gg; PRE_WEAK</b>, the Nelson/Hindenburg gap is a coverage effect: the model holds "
  "19th-c associations strongly but barely holds the 1914&ndash;29 / Continental ones &mdash; even though both are "
  "pre-cutoff and &lsquo;should-know&rsquo;.</div>")
A("</body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("\nwrote", OUT)
