# -*- coding: utf-8 -*-
"""Standalone QUESTION LIST document: every facet in the comprehensive battery (prompt, true
answer, twins), grouped by category and POST/PRE, each marked KEPT or DROPPED by the DeepSeek
gate with the reason. This is the transparency artifact -- separate from the verdict doc."""
import os, sys, json, re, html, csv
sys.path.insert(0, os.path.dirname(__file__))
from dig_candidates_full import CANDIDATES
RAW = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\facet_validation_raw_full.json"
RES = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\dig_validated_results_full.csv"
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\questions.html"
esc = html.escape
raw = json.load(open(RAW, encoding="utf-8")) if os.path.exists(RAW) else {}
MODS = ["Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929", "Ranke-4B-1929-1202-Instruct"]
results = {}  # (model, category, true) -> (gap, correct)
if os.path.exists(RES):
    for x in csv.DictReader(open(RES, encoding="utf-8")):
        results[(x["model"], x["category"], x["true"])] = (float(x["gap"]), int(x["correct"]))


def parse_json(txt):
    txt = re.sub(r"^```(json)?|```$", "", txt.strip(), flags=re.MULTILINE).strip()
    a = txt.find("["); b = txt.rfind("]")
    s = txt[a:b + 1] if a >= 0 and b > a else txt
    try:
        return json.loads(re.sub(r",(\s*[}\]])", r"\1", s))
    except Exception:
        out = []
        for m in re.finditer(r"\{[^{}]*\}", s):
            try:
                out.append(json.loads(re.sub(r",(\s*})", r"\1", m.group(0))))
            except Exception:
                pass
        return out


# verdict map per event key
verds = {}
for idx, (cat, role, yr, facets) in enumerate(CANDIDATES):
    key = f"{idx}|{role}"
    vmap = {}
    if key in raw:
        try:
            for v in parse_json(raw[key]["text"]):
                vmap[v["i"]] = v
        except Exception:
            pass
    verds[idx] = vmap

CATS = []
for cat, role, yr, facets in CANDIDATES:
    if cat not in CATS:
        CATS.append(cat)

CSS = ("body{margin:0;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;color:#1a1d24;line-height:1.5}"
       ".wrap{max-width:1040px;margin:0 auto;padding:40px 22px 90px}h1{font-family:Georgia,serif;font-size:27px;margin:0 0 4px}"
       ".sub{color:#5b6675;font-size:15px;margin:0 0 18px}h2{font-family:Georgia,serif;font-size:20px;margin:30px 0 4px;border-top:2px solid #e7eaef;padding-top:14px}"
       "h3{font-size:14px;color:#5b6675;margin:14px 0 4px;text-transform:uppercase;letter-spacing:.05em}"
       "table{border-collapse:collapse;width:100%;margin:6px 0 14px;font-size:13px}"
       "th,td{border:1px solid #e6e9ef;padding:5px 8px;vertical-align:top;text-align:left}th{background:#f0f3f8}"
       ".q{width:48%}.tw{color:#5b6675}.keep{color:#2e7d32;font-weight:700}.drop{color:#b4341f;font-weight:700}"
       "tr.dropped{background:#fdf4f3}.rsn{color:#8a4b44;font-size:12px}"
       ".legend{background:#f6f8fb;border:1px solid #e2e6ec;border-radius:8px;padding:10px 14px;font-size:13px;margin:10px 0}"
       "code{background:#eef1f6;padding:1px 4px;border-radius:3px}")
H = ['<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
     f"<title>Ranke diagnostic — the question list</title><style>{CSS}</style></head><body><div class=wrap>"]
A = H.append
A("<h1>The question list</h1>")
A("<p class=sub>Every probe in the comprehensive diagnostic, verbatim &mdash; prompt, true answer, and the "
  "candidate false &lsquo;twins&rsquo; &mdash; grouped by category and by <b>should-NOT-know (post-cutoff)</b> vs "
  "<b>should-know (pre-cutoff baseline)</b>. Each row is marked KEPT or DROPPED by the DeepSeek-V4-Pro gate, with its "
  "reason. The scorer length-matches: it picks, from the twins, the one with the same token count as the true answer.</p>")
tot = kept = 0
# summary table
sumrows = {}
for idx, (cat, role, yr, facets) in enumerate(CANDIDATES):
    for i, fac in enumerate(facets):
        v = verds[idx].get(i, {})
        drop = (v.get("verdict") == "drop") or (v.get("true_correct") is False)
        sumrows.setdefault(cat, [0, 0])
        sumrows[cat][0] += 1
        if not drop:
            sumrows[cat][1] += 1
        tot += 1; kept += (0 if drop else 1)
A('<div class=legend><b>Summary.</b> ' + str(kept) + " of " + str(tot) + " facets kept by the gate. "
  + " &nbsp;|&nbsp; ".join(f"{esc(c)}: {k}/{n}" for c, (n, k) in sumrows.items()) + "</div>")


def render(cat):
    for role in ("POST", "PRE"):
        for idx, (c, r, yr, facets) in enumerate(CANDIDATES):
            if c != cat or r != role:
                continue
            A("<h3>" + ("Should-NOT-know &mdash; post-cutoff (the contamination test)" if role == "POST"
                        else "Should-know &mdash; pre-cutoff baseline (the ceiling)") + "</h3>")
            rows = []
            for i, fac in enumerate(facets):
                prompt, true, tw = fac[0], fac[1], fac[2]
                v = verds[idx].get(i, {})
                drop = (v.get("verdict") == "drop") or (v.get("true_correct") is False)
                mark = (f'<span class=drop>DROPPED</span><div class=rsn>{esc(v.get("reason",""))}</div>' if drop
                        else '<span class=keep>kept</span>')
                if drop:
                    res = '<span class=tw>&mdash;</span>'
                else:
                    g29 = results.get(("Ranke-4B-1929", c, true.strip()))
                    if g29:
                        hits = "".join("&#10003;" if results.get((m, c, true.strip()), (0, 0))[1] else "&#10007;" for m in MODS)
                        res = f'{g29[0]:+.1f} <span class=tw>[{hits}]</span>'
                    else:
                        res = '<span class=tw>(scored)</span>'
                rows.append(f'<tr class="{"dropped" if drop else ""}"><td>{i}</td>'
                            f'<td class=q>{esc(prompt)} <b>___</b></td>'
                            f'<td><b>{esc(true.strip())}</b></td>'
                            f'<td class=tw>{esc(", ".join(x.strip() for x in tw))}</td>'
                            f'<td>{mark}</td><td>{res}</td></tr>')
            A('<table><thead><tr><th>#</th><th>prompt</th><th>true answer</th><th>false twins</th><th>gate</th>'
              '<th>model: gap [1900&middot;1913&middot;1929&middot;Instr]</th></tr></thead>'
              '<tbody>' + "".join(rows) + '</tbody></table>')


for cat in CATS:
    A(f"<h2>{esc(cat)}</h2>")
    render(cat)
A("</div></body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(H))
print("wrote", OUT, "(", kept, "of", tot, "facets kept )")
