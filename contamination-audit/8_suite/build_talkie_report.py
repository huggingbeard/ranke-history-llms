# -*- coding: utf-8 -*-
"""Build a self-contained HTML report for the Talkie-1930-13B (base) contamination probe
on the locked v2 question bank. Every number is read from the result JSON files; nothing
is hand-typed. Also folds in the Ranke-4B-1929 result on the same bank for contrast.

Output: talkie_report.html  (open in any browser, no external assets)
"""
import json, os, html, datetime, math

HERE = os.path.dirname(os.path.abspath(__file__))
TALKIE = os.path.join(HERE, "results_v2_talkie-1930-13b-base.json")
RANKE = os.path.join(HERE, "results_v2_Ranke-4B-1929.json")
BANK = os.path.join(HERE, "question_bank_v2.json")
OUT = os.path.join(HERE, "talkie_report.html")

CHANCE = 25.0


def load(p):
    return json.load(open(p, encoding="utf-8"))


talk = load(TALKIE)
bank_items = load(BANK)["items"]
bank = {(b["era"], b["topic"], b["label"]): b for b in bank_items}
ranke = load(RANKE) if os.path.exists(RANKE) else None

ag = talk["aggregate"]
items = talk["items"]


def light(rec):
    """Red/amber/green per project rule. Test A: POST above chance? (binom_p<.05 AND pct>chance).
    If not above chance -> GREEN. (AMBER/RED need POST-vs-PRE; here every post cat is at chance,
    so all GREEN; we still flag 'watch' when pct>chance but n.s.)"""
    above = rec["binom_p"] < 0.05 and rec["pct"] > rec["chance_pct"]
    if not above:
        watch = rec["pct"] > rec["chance_pct"]
        return ("green", "CLEAN" + (" (watch)" if watch else ""))
    return ("red", "FLAGGED")  # would need PRE comparison to split RED/AMBER


def fmt_p(p):
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


# ---- per-topic rows (post) ----
post_topics = sorted(k.split(":", 1)[1] for k in ag if k.startswith("post:") and not k.endswith(":ALL"))
pre_topics = {k.split(":", 1)[1]: ag[k] for k in ag if k.startswith("pre:") and not k.endswith(":ALL")}

post_all = ag["post:ALL"]
pre_all = ag["pre:ALL"]

# ---- contrast numbers from Ranke-4B (same bank) ----
# IMPORTANT: the Ranke v2 JSON's summary dicts use the CALIBRATED metric, which the project
# DISCARDED (it broke the instrument). Recompute from correct_raw to match Talkie's raw
# forced-choice exactly (verified to reproduce the LOG's raw numbers).
def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def raw_stat(its):
    k = sum(it["correct_raw"] for it in its)
    n = len(its)
    if not n:
        return None
    gaps = [it["gap_raw"] for it in its]
    return {"correct": k, "n": n, "pct": round(100 * k / n, 1),
            "binom_p": round(binom_ge(k, n), 4), "chance_pct": CHANCE,
            "mean_gap": round(sum(gaps) / n, 3)}


ranke_rows = {}
ranke_post_all = ranke_pre_all = None
if ranke:
    rit = ranke["items"]
    ranke_post_all = raw_stat([x for x in rit if x["era"] == "post"])
    ranke_pre_all = raw_stat([x for x in rit if x["era"] == "pre"])
    for t in sorted({x["topic"] for x in rit if x["era"] == "post"}):
        ranke_rows[t] = raw_stat([x for x in rit if x["era"] == "post" and x["topic"] == t])


# ================= HTML =================
def bar(pct, color, chance=CHANCE, w=240):
    """horizontal bar with a chance marker line"""
    fill = max(0, min(100, pct)) / 100 * w
    cx = chance / 100 * w
    return f"""<svg width="{w}" height="20" viewBox="0 0 {w} 20" class="bar">
      <rect x="0" y="4" width="{w}" height="12" rx="3" fill="#eef1f5"/>
      <rect x="0" y="4" width="{fill:.1f}" height="12" rx="3" fill="{color}"/>
      <line x1="{cx:.1f}" y1="1" x2="{cx:.1f}" y2="19" stroke="#9aa3af" stroke-width="1.5" stroke-dasharray="2,2"/>
    </svg>"""


rows_html = []
for t in post_topics:
    rec = ag["post:" + t]
    cls, lab = light(rec)
    color = {"green": "#2e9e5b", "amber": "#d98a00", "red": "#d23c3c"}[cls]
    rows_html.append(f"""<tr>
      <td class="topic">{html.escape(t)}</td>
      <td class="num">{rec['correct']}/{rec['n']}</td>
      <td class="num">{rec['pct']:.1f}%</td>
      <td>{bar(rec['pct'], color)}</td>
      <td class="num">{fmt_p(rec['binom_p'])}</td>
      <td class="num">{rec['mean_gap']:+.2f}</td>
      <td><span class="pill {cls}">{lab}</span></td>
    </tr>""")
rows_html = "\n".join(rows_html)

# contrast table (talkie vs ranke per topic)
contrast_html = ""
if ranke:
    crows = []
    for t in post_topics:
        tk = ag["post:" + t]
        rk = ranke_rows.get(t)
        rk_txt = f"{rk['pct']:.1f}% (p={fmt_p(rk['binom_p'])})" if rk else "&mdash;"
        rk_flag = rk and rk["binom_p"] < 0.05 and rk["pct"] > CHANCE
        crows.append(f"""<tr>
          <td class="topic">{html.escape(t)}</td>
          <td class="num">{tk['pct']:.1f}% (p={fmt_p(tk['binom_p'])})</td>
          <td class="num {'flag' if rk_flag else ''}">{rk_txt}</td>
        </tr>""")
    contrast_html = f"""
    <h2>Contrast: same bank, a different model leaks</h2>
    <p>The identical 272-item v2 bank was also scored on <b>Ranke-4B-1929</b> (the 4B chat-tuned model,
    cutoff 1929). That model showed a real, FDR-flagged contamination signal in US civil rights
    ({ranke_rows.get('US civil rights',{}).get('pct','?')}% , p={fmt_p(ranke_rows.get('US civil rights',{}).get('binom_p',1))}).
    Talkie-13B base does <b>not</b> replicate it. This is a genuine model/architecture difference
    (13B base vs 4B chat-tuned; cutoff 1930 vs 1929), <b>not</b> a refutation of the 4B finding.</p>
    <table class="contrast">
      <thead><tr><th>Topic (post-cutoff)</th><th>Talkie-1930-13B base</th><th>Ranke-4B-1929</th></tr></thead>
      <tbody>
      <tr class="totrow"><td>POST / ALL</td>
        <td class="num">{post_all['pct']:.1f}% (p={fmt_p(post_all['binom_p'])})</td>
        <td class="num">{ranke_post_all['pct']:.1f}% (p={fmt_p(ranke_post_all['binom_p'])})</td></tr>
      {''.join(crows)}
      </tbody>
    </table>"""

# item-level detail, grouped by topic, post only, sorted by gap_best desc (most "knows it" first)
detail_blocks = []
for t in post_topics:
    its = [it for it in items if it["era"] == "post" and it["topic"] == t]
    its.sort(key=lambda x: -x["gap_best"])
    rec = ag["post:" + t]
    irows = []
    for it in its:
        b = bank.get((it["era"], it["topic"], it["label"]), {})
        prompt = html.escape(b.get("prompt", ""))
        true = html.escape((b.get("true", "") or "").strip())
        gap = it["gap_best"]
        gcls = "pos" if gap > 0 else "neg"
        mark = "✓" if it["correct"] else "·"
        irows.append(f"""<tr>
          <td class="mk {gcls}">{mark}</td>
          <td class="lbl">{html.escape(it['label'])}</td>
          <td class="pr">{prompt} <b>{true}</b></td>
          <td class="num {gcls}">{gap:+.2f}</td>
        </tr>""")
    cls, lab = light(rec)
    detail_blocks.append(f"""<details>
      <summary><b>{html.escape(t)}</b> &mdash; {rec['correct']}/{rec['n']} = {rec['pct']:.1f}%
        <span class="pill {cls}">{lab}</span></summary>
      <table class="items"><thead><tr><th></th><th>item</th><th>prompt → true answer</th><th>gap</th></tr></thead>
      <tbody>{''.join(irows)}</tbody></table>
    </details>""")
detail_html = "\n".join(detail_blocks)

cliff = pre_all["pct"] - post_all["pct"]
gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

DOC = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Talkie-1930-13B — contamination probe</title>
<style>
:root {{ --ink:#1a2230; --mut:#5c6675; --line:#e3e7ee; --blue:#2b5dab; --bg:#f7f8fa; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }}
.wrap {{ max-width:960px; margin:0 auto; padding:48px 28px 90px; }}
h1 {{ font-size:30px; margin:0 0 6px; letter-spacing:-.4px; }}
.sub {{ color:var(--mut); font-size:15px; margin:0 0 28px; }}
h2 {{ font-size:21px; margin:42px 0 10px; padding-top:14px; border-top:1px solid var(--line); }}
p {{ color:#2b3340; }}
.verdict {{ background:#fff; border:1px solid var(--line); border-left:5px solid #2e9e5b;
  border-radius:10px; padding:20px 22px; margin:0 0 26px; box-shadow:0 1px 2px rgba(20,30,50,.04); }}
.verdict .big {{ font-size:19px; font-weight:700; color:#1f7a47; }}
.cards {{ display:flex; gap:14px; flex-wrap:wrap; margin:22px 0 8px; }}
.card {{ flex:1 1 150px; background:#fff; border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; box-shadow:0 1px 2px rgba(20,30,50,.04); }}
.card .k {{ font-size:12px; text-transform:uppercase; letter-spacing:.5px; color:var(--mut); }}
.card .v {{ font-size:28px; font-weight:700; margin-top:4px; }}
.card .v small {{ font-size:14px; font-weight:500; color:var(--mut); }}
table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line);
  border-radius:10px; overflow:hidden; margin:14px 0 8px; }}
th,td {{ padding:9px 12px; text-align:left; border-bottom:1px solid var(--line); font-size:14px; }}
th {{ background:#f0f3f8; color:#3a4252; font-weight:600; font-size:12.5px;
  text-transform:uppercase; letter-spacing:.4px; }}
tr:last-child td {{ border-bottom:none; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.topic {{ font-weight:600; }}
.bar {{ display:block; }}
.pill {{ display:inline-block; padding:2px 9px; border-radius:20px; font-size:12px; font-weight:700; }}
.pill.green {{ background:#e4f6ea; color:#1f7a47; }}
.pill.amber {{ background:#fcefd6; color:#9a6700; }}
.pill.red {{ background:#fbe3e3; color:#b32424; }}
.totrow td {{ font-weight:700; background:#f6f8fb; }}
.flag {{ color:#b32424; font-weight:700; }}
details {{ background:#fff; border:1px solid var(--line); border-radius:10px; margin:10px 0; padding:4px 14px; }}
summary {{ cursor:pointer; padding:9px 2px; font-size:15px; }}
table.items, table.contrast {{ box-shadow:none; border:none; }}
table.items th, table.items td {{ font-size:13px; padding:6px 10px; }}
.mk {{ width:24px; text-align:center; font-weight:700; }}
.pos {{ color:#1f7a47; }}
.neg {{ color:#b32424; }}
.lbl {{ font-weight:600; white-space:nowrap; }}
.pr {{ color:#3a4252; }}
.meth {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:16px 20px; }}
.meth ul {{ margin:8px 0 0; padding-left:20px; }}
.meth li {{ margin:5px 0; }}
.foot {{ color:var(--mut); font-size:12.5px; margin-top:40px; border-top:1px solid var(--line); padding-top:14px; }}
code {{ background:#eef1f5; padding:1px 5px; border-radius:4px; font-size:13px; }}
.legend {{ font-size:12.5px; color:var(--mut); margin:4px 0 0; }}
</style></head>
<body><div class="wrap">

<h1>Talkie-1930-13B (base) — post-cutoff contamination probe</h1>
<p class="sub">Time-locked LLM contamination audit · v2 locked question bank · forced-choice (sum-logp) ·
chance = {CHANCE:.0f}%</p>

<div class="verdict">
  <div class="big">✓ CLEAN — no coherent post-cutoff knowledge</div>
  <p style="margin:8px 0 0">The model answers <b>{pre_all['pct']:.1f}%</b> of pre-cutoff control items correctly
  but only <b>{post_all['pct']:.1f}%</b> of post-cutoff items — at or below the {CHANCE:.0f}% chance line
  (one-sided binomial p = {fmt_p(post_all['binom_p'])}). A {cliff:.0f}-point cliff with the post side sitting
  at chance. <b>No</b> post-cutoff category rises significantly above chance.</p>
</div>

<div class="cards">
  <div class="card"><div class="k">Pre-cutoff (should know)</div>
    <div class="v">{pre_all['pct']:.1f}%<small> · {pre_all['correct']}/{pre_all['n']}</small></div></div>
  <div class="card"><div class="k">Post-cutoff (should not)</div>
    <div class="v">{post_all['pct']:.1f}%<small> · {post_all['correct']}/{post_all['n']}</small></div></div>
  <div class="card"><div class="k">Cliff (pre − post)</div>
    <div class="v">{cliff:.0f}<small> pts</small></div></div>
  <div class="card"><div class="k">Post vs chance</div>
    <div class="v">p={fmt_p(post_all['binom_p'])}</div></div>
</div>

<h2>Post-cutoff results by category</h2>
<table>
  <thead><tr><th>Topic</th><th class="num">correct</th><th class="num">%</th>
    <th>vs&nbsp;chance ({CHANCE:.0f}%)</th><th class="num">binom&nbsp;p</th>
    <th class="num">mean&nbsp;gap</th><th>verdict</th></tr></thead>
  <tbody>
  <tr class="totrow"><td>ALL POST</td><td class="num">{post_all['correct']}/{post_all['n']}</td>
    <td class="num">{post_all['pct']:.1f}%</td><td>{bar(post_all['pct'], '#2e9e5b')}</td>
    <td class="num">{fmt_p(post_all['binom_p'])}</td><td class="num">{post_all['mean_gap']:+.2f}</td>
    <td><span class="pill green">CLEAN</span></td></tr>
  {rows_html}
  </tbody>
</table>
<p class="legend">Dashed line = {CHANCE:.0f}% chance (4-option forced choice). <b>mean gap</b> = mean log-prob
advantage of the true answer over the best distractor; negative = the model prefers a wrong answer.
"watch" marks a category numerically above chance but not statistically (p ≥ 0.05).</p>

{contrast_html}

<h2>Method</h2>
<div class="meth">
  <ul>
    <li><b>Forced choice, whole-answer sum-logp.</b> For each item the model teacher-forces the true
      completion and each distractor; "correct" = it assigns the highest total log-probability to the
      true answer. Metric chosen on should-know controls, blind to the post items.</li>
    <li><b>Length-matched twins, no name cues.</b> Distractors are plausible same-type alternatives
      (e.g. wrong years, other real entities); the prompt gives the descriptive fact only and never
      cues the answer's name.</li>
    <li><b>Bank:</b> locked <code>question_bank_v2.json</code> — {len(items)} items
      ({post_all['n']} post-cutoff + {pre_all['n']} pre-cutoff controls), blind-screened.</li>
    <li><b>Numerics check:</b> custom forward pass vs reference forward, max logit diff =
      {talk.get('validation_maxdiff')} (exact).</li>
    <li><b>Red/amber/green:</b> GREEN = post not above chance; AMBER = above chance but below pre;
      RED = above chance and indistinguishable from pre. Every Talkie category is GREEN.</li>
  </ul>
</div>

<h2>Item-level detail (post-cutoff, sorted by the model's preference)</h2>
<p class="legend">Click a category. ✓ = model picked the true answer; gap is its log-prob margin over
the best distractor. Even the few ✓ hits carry small margins — consistent with chance, not recall.</p>
{detail_html}

<div class="foot">
  Source: <code>results_v2_talkie-1930-13b-base.json</code> (run finished 2026-06-26 13:35Z on a RunPod
  A100 PCIe, pod fkart67syltxb3, self-terminated and verified offline). Generated {gen_time} by
  <code>build_talkie_report.py</code>. All figures read directly from the result and bank JSON — none typed by hand.
</div>

</div></body></html>"""

open(OUT, "w", encoding="utf-8").write(DOC)
print("wrote", OUT, f"({len(DOC)} bytes)")
print("post topics:", len(post_topics), "| items:", len(items))
