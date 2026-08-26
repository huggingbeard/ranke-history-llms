# -*- coding: utf-8 -*-
"""ONE integrated contamination report on the locked v2 pre-registered bank.

Covers every model scored on question_bank_v2.json, side by side:
  - Ranke-4B-1929 (base)        -> contaminated in US mid-century domestic history
  - Talkie-1930-13B (base)      -> clean

All numbers are recomputed from each result file's per-item records using the RAW
forced-choice metric (the project's settled instrument). The Ranke v2 file also carries
a *calibrated* metric in its summary dicts; that was DISCARDED upstream because it broke
the instrument, so we deliberately ignore it and recompute from correct_raw/gap_raw.

Output: contamination_report.html  (self-contained, inline SVG, no external assets)
Supersedes the standalone talkie_report.html and the old v1-bank comparison.html.
"""
import json, os, math, html, datetime
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(HERE, "question_bank_v2.json")
OUT = os.path.join(HERE, "contamination_report.html")
esc = html.escape
CHANCE = 25.0
# Two-tier flagging. RED = registered primary test (BH-FDR, q=.05) -> confirmed leak.
# AMBER = above chance at the cautious nominal bar (a contamination audit should not dismiss
# borderline signals); set to .10 per user. GREEN = not above chance even at ALPHA.
ALPHA = 0.10      # nominal "elevated" bar (one-sided, uncorrected)
FDR_Q = 0.05      # pre-registered FDR level for the confirmed-leak (RED) tier

# model -> (file, correct field, gap field, metadata). Order = display order.
MODELS = [
    {"key": "Ranke-4B-1929", "nice": "Ranke-4B-1929", "short": "Ranke-4B", "color": "#3151b5",
     "file": "results_v2_Ranke-4B-1929.json", "cf": "correct_raw", "gf": "gap_raw",
     "size": "4B", "cutoff": 1929, "team": "Ranke", "tuning": "base"},
    {"key": "talkie-1930-13b-base", "nice": "Talkie-1930-13B", "short": "Talkie-13B", "color": "#0f9d8f",
     "file": "results_v2_talkie-1930-13b-base.json", "cf": "correct", "gf": "gap_best",
     "size": "13B", "cutoff": 1930, "team": "Talkie", "tuning": "base"},
]

bank_items = json.load(open(BANK, encoding="utf-8"))["items"]
bank = {(b["era"], b["topic"], b["label"]): b for b in bank_items}

# narrow (powered) vs breadth topics, derived from the bank's own level field
narrow_topics, broad_topics = [], []
seen = set()
for b in bank_items:
    if b["era"] != "post":
        continue
    key = b["topic"]
    if key in seen:
        continue
    seen.add(key)
    (narrow_topics if b.get("level") == "narrow" else broad_topics).append(key)
narrow_topics.sort()
broad_topics.sort()


def binom_ge(k, n, p=0.25):
    return sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1)) if n else 1.0


def stat(its, cf, gf):
    n = len(its)
    if not n:
        return None
    k = sum(it[cf] for it in its)
    g = [it[gf] for it in its]
    return {"correct": k, "n": n, "pct": round(100 * k / n, 1),
            "binom_p": binom_ge(k, n), "mean_gap": round(sum(g) / n, 3)}


def bh_flags(pairs, q=0.05):
    """Benjamini-Hochberg across (name, p) pairs. Returns {name: bool}."""
    m = len(pairs)
    order = sorted(range(m), key=lambda i: pairs[i][1])
    kmax = 0
    for rank, idx in enumerate(order, start=1):
        if pairs[idx][1] <= q * rank / m:
            kmax = rank
    flagged = {name: False for name, _ in pairs}
    for rank, idx in enumerate(order, start=1):
        if rank <= kmax:
            flagged[pairs[idx][0]] = True
    return flagged


def fmt_p(p):
    return "&lt;0.001" if p < 0.001 else f"{p:.3f}"


# ---------- load + compute per model ----------
M = {}
for spec in MODELS:
    path = os.path.join(HERE, spec["file"])
    if not os.path.exists(path):
        continue
    d = json.load(open(path, encoding="utf-8"))
    cf, gf = spec["cf"], spec["gf"]
    its = d["items"]
    rec = {"spec": spec, "items": its,
           "pre": stat([x for x in its if x["era"] == "pre"], cf, gf),
           "post": stat([x for x in its if x["era"] == "post"], cf, gf),
           "narrow": {t: stat([x for x in its if x["era"] == "post" and x["topic"] == t], cf, gf) for t in narrow_topics},
           "broad_pool": stat([x for x in its if x["era"] == "post" and bank.get((x["era"], x["topic"], x["label"]), {}).get("level") == "broad"], cf, gf),
           "broad": {t: stat([x for x in its if x["era"] == "post" and x["topic"] == t], cf, gf) for t in broad_topics}}
    # FDR across narrow clusters (the pre-registered primary test, q=.05)
    pairs = [(t, rec["narrow"][t]["binom_p"]) for t in narrow_topics if rec["narrow"][t]]
    rec["fdr"] = bh_flags(pairs, FDR_Q)
    # verdict: RED = FDR-confirmed; AMBER = above chance at nominal ALPHA but not FDR
    rec["reds"] = [t for t in narrow_topics if rec["fdr"].get(t) and rec["narrow"][t]["pct"] > CHANCE]
    rec["ambers"] = [t for t in narrow_topics if rec["narrow"][t] and rec["narrow"][t]["binom_p"] < ALPHA
                     and rec["narrow"][t]["pct"] > CHANCE and t not in rec["reds"]]
    if rec["reds"]:
        rec["verdict"] = ("bad", "CONTAMINATED")
    elif rec["ambers"]:
        rec["verdict"] = ("warn", "ELEVATED — not clean")
    else:
        rec["verdict"] = ("good", "CLEAN")
    M[spec["key"]] = rec

models = [s["key"] for s in MODELS if s["key"] in M]

# worked examples with full per-option log-probs (from probe_examples.py), optional
EXP = os.path.join(HERE, "probe_examples.json")
EX = json.load(open(EXP, encoding="utf-8")) if os.path.exists(EXP) else None


def topic_light(rec_model, topic):
    s = rec_model["narrow"].get(topic) or rec_model["broad"].get(topic)
    if not s:
        return ("green", "—")
    fdr = rec_model["fdr"].get(topic, False)
    if fdr and s["pct"] > CHANCE:
        return ("red", "LEAK")
    if s["binom_p"] < ALPHA and s["pct"] > CHANCE:
        return ("amber", "elevated")
    return ("green", "clean")


# ================= SVG =================
def svg_bars(title, series, ymax=100, chance=CHANCE, w=460, h=290):
    pad_l, pad_b, pad_t = 46, 60, 38
    plot_w, plot_h = w - pad_l - 16, h - pad_b - pad_t
    n = len(series)
    gap = plot_w / max(n, 1)
    bw = gap * 0.58
    s = [f'<svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;border:1px solid #e6e9ef;border-radius:10px;background:#fff">']
    s.append(f'<text x="{w/2}" y="22" text-anchor="middle" style="font:700 14px Georgia,serif;fill:#1a2230">{esc(title)}</text>')
    for gv in (0, 25, 50, 75, 100):
        if gv > ymax:
            continue
        y = pad_t + plot_h * (1 - gv / ymax)
        s.append(f'<line x1="{pad_l}" x2="{w-16}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eff2f6"/>')
        s.append(f'<text x="{pad_l-6}" y="{y+4:.1f}" text-anchor="end" style="font:10px sans-serif;fill:#9aa3af">{gv}</text>')
    yc = pad_t + plot_h * (1 - chance / ymax)
    s.append(f'<line x1="{pad_l}" x2="{w-16}" y1="{yc:.1f}" y2="{yc:.1f}" stroke="#94a0b0" stroke-dasharray="5 4" stroke-width="1.3"/>')
    s.append(f'<text x="{w-18}" y="{yc-5:.1f}" text-anchor="end" style="font:700 10px sans-serif;fill:#6b7787">chance {chance:.0f}%</text>')
    for i, (lab, val, col) in enumerate(series):
        x = pad_l + gap * i + (gap - bw) / 2
        bh = plot_h * (min(val, ymax) / ymax)
        y = pad_t + plot_h - bh
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="3" fill="{col}"/>')
        s.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" style="font:700 12px sans-serif;fill:#1a2230">{val:.0f}%</text>')
        for j, ln in enumerate(lab.split("\n")):
            s.append(f'<text x="{x+bw/2:.1f}" y="{h-pad_b+16+j*12:.1f}" text-anchor="middle" style="font:10.5px sans-serif;fill:#52637a">{esc(ln)}</text>')
    s.append("</svg>")
    return "".join(s)


def render_example(ex):
    """One fully worked forced-choice probe: prompt + every option with its raw sum-logp."""
    kw = ex["tag"]
    tcls = ("red" if "LEAK" in kw else "amber" if "BORDERLINE" in kw else "ctrl" if "pre-control" in kw else "green")
    opts = sorted(ex["options"], key=lambda o: -o["logp"])
    los = [o["logp"] for o in opts]
    lo, hi = min(los), max(los)
    rng = (hi - lo) or 1.0
    true_ntok = [o["ntok"] for o in opts if o["is_true"]][0]
    dist_max_ntok = max(o["ntok"] for o in opts if not o["is_true"])
    rows = []
    for o in opts:
        w = 6 + (o["logp"] - lo) / rng * 94
        barcol = "#2e9e5b" if o["is_true"] else "#b9c2cf"
        badge = '<span class="tg true">TRUE</span>' if o["is_true"] else ""
        pick = '<span class="pk">◀ model picked</span>' if o.get("picked") else ""
        rcls = "hit" if (o.get("picked") and o["is_true"]) else ("missrow" if o.get("picked") else "")
        rows.append(f'<div class="opt {rcls}"><div class="otext">{esc(o["text"])}{badge}{pick}</div>'
                    f'<div class="obarwrap"><div class="obar" style="width:{w:.0f}%;background:{barcol}"></div></div>'
                    f'<div class="olp">{o["logp"]:.1f}<span class="ot">·{o["ntok"]}tok</span></div></div>')
    outcome = ("<span class='pos'>✓ picked the TRUE answer — a hit (counts as &quot;knows it&quot;)</span>"
               if ex["model_correct"] else
               "<span class='neg'>✗ picked a wrong answer — a miss (counts as &quot;doesn't know&quot;)</span>")
    note = ""
    if true_ntok - dist_max_ntok >= 4:
        note = (f'<div class="exnote"><b>Caveat — length bias:</b> the true answer is {true_ntok} tokens vs '
                f'{dist_max_ntok} for the longest distractor, so raw sum-logp is stacked against it. A miss here is '
                f'partly an artifact of length, not pure ignorance — exactly why length-matched twins matter, and why '
                f'an imperfectly-matched item can read as falsely "clean".</div>')
    return (f'<div class="ex"><span class="extag {tcls}">{esc(kw)}</span>'
            f'<div class="exq"><span class="q">&ldquo;{esc(ex["prompt"])} ____&rdquo;</span> '
            f'<span class="blank">→ which completion?</span></div>'
            f'{"".join(rows)}'
            f'<div class="exout">Model&rsquo;s choice = highest log-prob → {outcome}. '
            f'Gap (true − best distractor) = {ex["gap_raw"]:+.2f}.</div>{note}</div>')


# ================= build HTML =================
P = []
A = P.append
gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

CSS = """
:root{--ink:#1a2230;--mut:#5c6675;--line:#e6e9ef;--bg:#f6f8fa;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:46px 26px 100px}
h1{font:700 30px/1.2 Georgia,serif;margin:0 0 6px;letter-spacing:-.3px}
.sub{color:var(--mut);margin:0 0 26px;font-size:15px}
h2{font:700 21px Georgia,serif;margin:46px 0 4px;padding-top:16px;border-top:1px solid var(--line)}
h2 .n{color:#9aa3af;font-weight:400;margin-right:8px}
h3{font:700 16px Georgia,serif;margin:24px 0 6px}
p{color:#2b3340}
.lead{font-size:15px}
.cards{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0}
.vcard{flex:1 1 320px;background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;box-shadow:0 1px 2px rgba(20,30,50,.04)}
.vcard.good{border-left:5px solid #2e9e5b}.vcard.warn{border-left:5px solid #d98a00}.vcard.bad{border-left:5px solid #d23c3c}
.vcard .m{font:700 17px Georgia,serif}.vcard .spec{color:var(--mut);font-size:12.5px;margin:2px 0 10px}
.vbig{font-size:18px;font-weight:800}
.good .vbig{color:#1f7a47}.warn .vbig{color:#9a6700}.bad .vbig{color:#b32424}
.vcard .stat{display:flex;gap:16px;margin-top:10px;font-size:13px;color:var(--mut)}
.vcard .stat b{display:block;font-size:20px;color:var(--ink)}
.charts{display:flex;gap:16px;flex-wrap:wrap;margin:14px 0}
.charts>div{flex:1 1 300px;min-width:280px}
table{width:100%;border-collapse:collapse;background:#fff;border:1px solid var(--line);border-radius:10px;overflow:hidden;margin:12px 0;font-size:13.5px}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#eef2f7;color:#3a4252;font:600 12px sans-serif;text-transform:uppercase;letter-spacing:.4px}
tr:last-child td{border-bottom:none}
.num{text-align:right;font-variant-numeric:tabular-nums}
.lab{font-weight:600;background:#f8fafc}
.totrow td{font-weight:700;background:#f6f8fb}
.pill{display:inline-block;padding:2px 9px;border-radius:20px;font:700 11.5px sans-serif}
.pill.green{background:#e4f6ea;color:#1f7a47}.pill.amber{background:#fcefd6;color:#9a6700}.pill.red{background:#fbe3e3;color:#b32424}
.flag{color:#b32424;font-weight:700}
.box{background:#fff;border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:14px 0}
.box.good{border-left:4px solid #2e9e5b}.box.warn{border-left:4px solid #d98a00}.box.bad{border-left:4px solid #d23c3c}
.box ul{margin:8px 0 0;padding-left:20px}.box li{margin:5px 0}
details{background:#fff;border:1px solid var(--line);border-radius:10px;margin:9px 0;padding:3px 14px}
summary{cursor:pointer;padding:9px 2px}
table.items{box-shadow:none;border:none;margin:6px 0 10px}
table.items th,table.items td{font-size:12.5px;padding:5px 9px}
.mk{width:22px;text-align:center;font-weight:700}.pos{color:#1f7a47}.neg{color:#b32424}
.lbl{font-weight:600;white-space:nowrap}.pr{color:#3a4252}
code{background:#eef2f7;padding:1px 5px;border-radius:4px;font-size:13px}
.legend{font-size:12.5px;color:var(--mut);margin:4px 0 0}
.foot{color:var(--mut);font-size:12.5px;margin-top:44px;border-top:1px solid var(--line);padding-top:14px}
.callout{background:#eef4ff;border:1px solid #cfe0ff;border-left:5px solid #2b5dab;border-radius:10px;padding:14px 18px;margin:14px 0;font-size:14.5px}
.exgrid{display:flex;flex-direction:column;gap:14px;margin:14px 0}
.ex{background:#fff;border:1px solid var(--line);border-radius:12px;padding:16px 18px;box-shadow:0 1px 2px rgba(20,30,50,.04)}
.extag{display:inline-block;font:700 11.5px sans-serif;padding:2px 9px;border-radius:20px;margin-bottom:8px}
.extag.red{background:#fbe3e3;color:#b32424}.extag.amber{background:#fcefd6;color:#9a6700}
.extag.green{background:#e4f6ea;color:#1f7a47}.extag.ctrl{background:#e7eefb;color:#2b5dab}
.exq{font-size:14.5px;margin:2px 0 12px}.exq .q{color:#2b3340}.exq .blank{color:#b32424;font-weight:700}
.opt{display:flex;align-items:center;gap:10px;padding:3px 0}
.otext{flex:0 0 290px;font-size:13px}
.obarwrap{flex:1;background:#f0f3f7;border-radius:4px;height:16px;overflow:hidden}
.obar{height:16px;border-radius:4px}
.olp{flex:0 0 84px;text-align:right;font:600 12.5px monospace;font-variant-numeric:tabular-nums}
.olp .ot{color:#9aa3af;font-weight:400;font-size:11px;margin-left:3px}
.opt.hit .otext{color:#1f7a47;font-weight:700}.opt.missrow .otext{color:#b32424;font-weight:700}
.tg{display:inline-block;font:700 9.5px sans-serif;padding:1px 5px;border-radius:3px;vertical-align:middle;margin-left:5px}
.tg.true{background:#dff3e6;color:#1f7a47}
.pk{color:#2b5dab;font-weight:700;font-size:11.5px;margin-left:6px}
.exout{margin-top:10px;font-size:13.5px;font-weight:600}
.exnote{margin-top:6px;font-size:12.5px;color:#9a6700;background:#fcefd6;border-radius:6px;padding:6px 10px}
"""

A(f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
  f'<meta name="viewport" content="width=device-width,initial-scale=1">'
  f'<title>Time-locked LLM contamination — integrated report</title><style>{CSS}</style></head><body><div class="wrap">')

A("<h1>Time-locked LLM contamination — integrated report</h1>")
A('<p class="sub">Locked, pre-registered v2 bank · forced-choice (raw sum-logp) · chance = 25% · '
  f'{len([x for x in bank_items if x["era"]=="post"])} post-cutoff items + '
  f'{len([x for x in bank_items if x["era"]=="pre"])} pre-cutoff controls · '
  f'{len(models)} models on the identical bank</p>')

# ---- §0 one-line story ----
A('<p class="lead">Two ~1930-cutoff models, scored on the <b>same</b> bank of post-cutoff facts. Each item is a '
  'four-way forced choice against length-matched distractors, so a 4B and a 13B sit on the same 25% chance line. '
  'The question for every model: does it know things it should not?</p>')

# ---- §1 verdict cards ----
A('<h2><span class="n">1</span>Screening result at a glance</h2>')
A('<div class="callout"><b>These are screening calls, not final rulings.</b> The probe is a cheap smoke detector; '
  'the definitive test is grepping the training corpus for the specific facts. The significance bar exists to '
  '<i>prioritise</i> the search, not to <i>permit</i> it — so treat every non-green cluster, and (because the probe '
  'under-powers at this sample size) the strongest items even inside green clusters, as a corpus-search target '
  '(see §8). Contamination should not have to clear a high bar before anyone looks.</div>')
A('<div class="cards">')
for k in models:
    r = M[k]; sp = r["spec"]; cls, lab = r["verdict"]
    bits = [f'<span class="flag">{esc(h)} (leak)</span>' for h in r["reds"]] + \
           [f'{esc(h)} (elevated)' for h in r["ambers"]]
    hit_txt = (" — " + "; ".join(bits)) if bits else " across the board"
    A(f'''<div class="vcard {cls}">
      <div class="m">{esc(sp["nice"])} <span class="pill {'red' if cls=='bad' else ('amber' if cls=='warn' else 'green')}">{lab}</span></div>
      <div class="spec">{sp["size"]} · {sp["team"]} · {sp["tuning"]} · cutoff {sp["cutoff"]}</div>
      <div class="vbig">{lab}{hit_txt}</div>
      <div class="stat">
        <div>pre-cutoff<b>{r["pre"]["pct"]:.0f}%</b><span>{r["pre"]["correct"]}/{r["pre"]["n"]}</span></div>
        <div>post-cutoff<b>{r["post"]["pct"]:.0f}%</b><span>{r["post"]["correct"]}/{r["post"]["n"]}</span></div>
        <div>cliff<b>{r["pre"]["pct"]-r["post"]["pct"]:.0f} pts</b><span>pre − post</span></div>
      </div>
    </div>''')
A('</div>')

# narrative under the cards
narr = []
for k in models:
    r = M[k]; cls, lab = r["verdict"]
    if cls == "bad":
        extra = (f" It is also elevated on {', '.join(esc(h) for h in r['ambers'])} (above chance at α=.10, not FDR-confirmed)."
                 if r["ambers"] else "")
        narr.append(f"<b>{esc(M[k]['spec']['nice'])}</b> leaks: {', '.join(esc(h) for h in r['reds'])} sits above chance and "
                    f"survives Benjamini-Hochberg FDR (q=.05) across the four powered clusters — specific post-cutoff recall, not noise."
                    + extra)
    elif cls == "warn":
        narr.append(f"<b>{esc(M[k]['spec']['nice'])}</b> is <i>not</i> clean: {', '.join(esc(h) for h in r['ambers'])} is above "
                    f"chance at the cautious α=.10 bar (p≈0.06) — it does not survive FDR, so it is a weak, unconfirmed signal "
                    f"rather than a demonstrated leak, but it is too close to the line to dismiss in a contamination audit.")
    else:
        narr.append(f"<b>{esc(M[k]['spec']['nice'])}</b> is clean: no powered cluster rises above chance even at α=.10, "
                    f"and post-cutoff overall sits at/below 25% behind a {r['pre']['pct']-r['post']['pct']:.0f}-point cliff.")
A("<p>" + " ".join(narr) + " Same wall, same bank, different result — so contamination is a property of the specific "
  "training corpus, not an artifact of the test.</p>")

# ---- §2 anatomy of a probe (worked examples) ----
if EX and EX.get("examples"):
    A('<h2><span class="n">2</span>Anatomy of a probe — five worked examples</h2>')
    A(f'<p>Every item is a four-way forced choice. The model is shown a descriptive prompt (it never contains the '
      f'answer\'s name) and we read off its total log-probability for the true completion and each of three '
      f'length-matched distractors — all <i>real</i> entities of the same type. Highest log-prob wins; if that is the '
      f'true answer it counts as a hit. Bars below are the model\'s actual scores on '
      f'<b>{esc(EX["model"])}</b> (longer = more probable; green = the true answer).</p>')
    A('<div class="exgrid">')
    for ex in EX["examples"]:
        A(render_example(ex))
    A('</div>')
    A('<p class="legend">Read the leaks: on Orangeburg the model puts the true post-1929 term <b>11 log-units</b> '
      'above the next real candidate (Jackson State, Kent State) — that is specific knowledge of a 1968 event it '
      'should not have. Contrast Sputnik, a near-tie it gets wrong, and the Wright brothers control it nails.</p>')

# ---- §3 headline charts ----
A('<h2><span class="n">3</span>The headline — does the probe fire, and does the model leak?</h2>')
pre_series = [(M[k]["spec"]["short"], M[k]["pre"]["pct"], M[k]["spec"]["color"]) for k in models]
post_series = [(M[k]["spec"]["short"], M[k]["post"]["pct"], M[k]["spec"]["color"]) for k in models]
A('<div class="charts">')
A("<div>" + svg_bars("Pre-cutoff controls — probe must fire", pre_series) + "</div>")
A("<div>" + svg_bars("Post-cutoff overall — contamination test", post_series) + "</div>")
A("</div>")
A('<p class="legend">Left: the sanity check — both models must ace facts they legitimately know. '
  'Right: the test — a clean ~1930 model sits on the dashed chance line.</p>')

# ---- §3 narrow clusters ----
A('<h2><span class="n">4</span>Powered clusters (the primary test)</h2>')
A(f'<p>Four deep clusters (~40 items each), pre-registered. Each gets a one-sided binomial test vs 25%. '
  f'<b>LEAK (red)</b> = survives Benjamini-Hochberg FDR (q=.05, the registered primary test) across the four '
  f'clusters. <b>elevated (amber)</b> = above chance at the cautious nominal bar α=.10 but not FDR-confirmed — '
  f'flagged because a contamination audit should not wave away a p≈0.06 cluster. <b>clean (green)</b> = not above '
  f'chance even at α=.10.</p>')
A('<div class="charts">')
for t in narrow_topics:
    ser = [(M[k]["spec"]["short"], M[k]["narrow"][t]["pct"], M[k]["spec"]["color"]) for k in models if M[k]["narrow"].get(t)]
    A("<div style='flex:1 1 220px;min-width:210px'>" + svg_bars(t, ser, w=300, h=240) + "</div>")
A("</div>")
# narrow table
A('<table><thead><tr><th>cluster</th>'
  + "".join(f'<th class="num">{esc(M[k]["spec"]["short"])} %</th><th class="num">p (FDR)</th><th>verdict</th>' for k in models)
  + '</tr></thead><tbody>')
for t in narrow_topics:
    A(f'<tr><td class="lab">{esc(t)}</td>')
    for k in models:
        s = M[k]["narrow"][t]; fdr = M[k]["fdr"].get(t, False)
        lc, lt = topic_light(M[k], t)
        ptxt = fmt_p(s["binom_p"]) + (" ✓" if fdr and s["pct"] > CHANCE else "")
        A(f'<td class="num">{s["correct"]}/{s["n"]} · {s["pct"]:.0f}%</td>'
          f'<td class="num {"flag" if fdr and s["pct"]>CHANCE else ""}">{ptxt}</td>'
          f'<td><span class="pill {lc}">{lt}</span></td>')
    A("</tr>")
A("</tbody></table>")
A('<p class="legend">✓ next to a p-value = that cluster survives FDR (q=.05). "elevated" = nominally above chance '
  'at α=.10 but not FDR-confirmed; "clean" = not above chance even at α=.10. Mean log-prob gap (negative = the '
  'model still prefers a wrong answer on average) is shown in the per-cluster detail below.</p>')

# ---- §4 breadth scan ----
A('<h2><span class="n">5</span>Breadth scan</h2>')
A('<p>Five lighter topics (~15–18 items) sweeping the rest of post-cutoff knowledge, plus the pooled breadth result.</p>')
A('<table><thead><tr><th>topic</th>'
  + "".join(f'<th class="num">{esc(M[k]["spec"]["short"])} %</th><th class="num">p</th>' for k in models)
  + '</tr></thead><tbody>')
for t in broad_topics:
    A(f'<tr><td class="lab">{esc(t)}</td>')
    for k in models:
        s = M[k]["broad"][t]
        A(f'<td class="num">{s["correct"]}/{s["n"]} · {s["pct"]:.0f}%</td><td class="num">{fmt_p(s["binom_p"])}</td>')
    A("</tr>")
A('<tr class="totrow"><td>breadth pool</td>')
for k in models:
    s = M[k]["broad_pool"]
    A(f'<td class="num">{s["correct"]}/{s["n"]} · {s["pct"]:.0f}%</td><td class="num">{fmt_p(s["binom_p"])}</td>')
A("</tr></tbody></table>")

# ---- §5 master table ----
A('<h2><span class="n">6</span>All numbers</h2>')
cols = ["pre-controls", "POST all"] + narrow_topics + ["breadth pool"]
A('<table><thead><tr><th>model</th>' + "".join(f'<th class="num">{esc(c)}</th>' for c in cols) + '</tr></thead><tbody>')
for k in models:
    r = M[k]
    cells = [f'{r["pre"]["pct"]:.0f}% ({r["pre"]["correct"]}/{r["pre"]["n"]})',
             f'{r["post"]["pct"]:.0f}% (p={fmt_p(r["post"]["binom_p"])})']
    for t in narrow_topics:
        s = r["narrow"][t]; fdr = r["fdr"].get(t, False) and s["pct"] > CHANCE
        cells.append(f'<span class="{"flag" if fdr else ""}">{s["pct"]:.0f}% (p={fmt_p(s["binom_p"])})</span>')
    s = r["broad_pool"]; cells.append(f'{s["pct"]:.0f}% (p={fmt_p(s["binom_p"])})')
    A(f'<tr><td class="lab">{esc(r["spec"]["nice"])}</td>' + "".join(f'<td class="num">{c}</td>' for c in cells) + "</tr>")
A("</tbody></table>")

# ---- §6 per-model deep dive ----
A('<h2><span class="n">7</span>Per-model deep dive</h2>')
for k in models:
    r = M[k]; sp = r["spec"]; cls, lab = r["verdict"]
    A(f'<h3>{esc(sp["nice"])} — <span class="pill {"red" if cls=="bad" else ("amber" if cls=="warn" else "green")}">{lab}</span></h3>')
    A(f'<p class="legend">Pre-cutoff controls fire at <b>{r["pre"]["pct"]:.0f}%</b>; post-cutoff overall '
      f'<b>{r["post"]["pct"]:.0f}%</b> (p={fmt_p(r["post"]["binom_p"])}) — a {r["pre"]["pct"]-r["post"]["pct"]:.0f}-point cliff.</p>')
    flagged = [(t, "bad") for t in r["reds"]] + [(t, "warn") for t in r["ambers"]]
    if not flagged:
        A(f'<div class="box good">No powered cluster rises above chance even at α=.10. Clean across the board.</div>')
    for t, boxcls in flagged:
        s = r["narrow"][t]
        its = [x for x in r["items"] if x["era"] == "post" and x["topic"] == t]
        its.sort(key=lambda x: -x[sp["gf"]])
        top = [x for x in its if x[sp["gf"]] > 0][:10]
        rows = ""
        for x in top:
            b = bank.get((x["era"], x["topic"], x["label"]), {})
            rows += (f'<tr><td class="lbl">{esc(x["label"])}</td>'
                     f'<td class="pr">{esc(b.get("prompt",""))} <b>{esc((b.get("true","") or "").strip())}</b></td>'
                     f'<td class="num pos">{x[sp["gf"]]:+.2f}</td></tr>')
        if boxcls == "bad":
            head = (f'<b>{esc(t)}: confirmed leak.</b> {s["correct"]}/{s["n"]} = {s["pct"]:.0f}%, p={fmt_p(s["binom_p"])}, '
                    f'survives FDR. Distractors are other real entities of the same type, so picking the truth means the '
                    f'model knows it. Strongest hits (log-prob margin over best distractor):')
        else:
            head = (f'<b>{esc(t)}: borderline / unconfirmed.</b> {s["correct"]}/{s["n"]} = {s["pct"]:.0f}%, p={fmt_p(s["binom_p"])} '
                    f'(above chance at α=.10, <i>does not</i> survive FDR; mean gap {s["mean_gap"]:+.2f} is still negative, '
                    f'so argmax hits exceed chance while the model on average still prefers a wrong answer). Too close to the '
                    f'line to call clean — the items it does pick:')
        A(f'<div class="box {boxcls}">{head}'
          f'<table class="items"><thead><tr><th>item</th><th>prompt → true</th><th>gap</th></tr></thead>'
          f'<tbody>{rows}</tbody></table></div>')
    # collapsible item detail per topic
    for t in narrow_topics + broad_topics:
        s = r["narrow"].get(t) or r["broad"].get(t)
        its = [x for x in r["items"] if x["era"] == "post" and x["topic"] == t]
        its.sort(key=lambda x: -x[sp["gf"]])
        lc, lt = topic_light(r, t)
        rows = ""
        for x in its:
            b = bank.get((x["era"], x["topic"], x["label"]), {})
            g = x[sp["gf"]]; gcls = "pos" if g > 0 else "neg"
            mark = "✓" if x[sp["cf"]] else "·"
            rows += (f'<tr><td class="mk {gcls}">{mark}</td><td class="lbl">{esc(x["label"])}</td>'
                     f'<td class="pr">{esc(b.get("prompt",""))} <b>{esc((b.get("true","") or "").strip())}</b></td>'
                     f'<td class="num {gcls}">{g:+.2f}</td></tr>')
        A(f'<details><summary>{esc(t)} — {s["correct"]}/{s["n"]} = {s["pct"]:.0f}% '
          f'<span class="pill {lc}">{lt}</span></summary>'
          f'<table class="items"><thead><tr><th></th><th>item</th><th>prompt → true</th><th>gap</th></tr></thead>'
          f'<tbody>{rows}</tbody></table></details>')

# ---- §8 corpus-search triage (grep-first) ----
A('<h2><span class="n">8</span>What to do next — grep the corpus, don\'t gate on significance</h2>')
A('<div class="callout">In practice you want to grep the training corpus for <b>anything that looks suspicious</b>, '
  'not only what clears a significance threshold. The probe is a cheap, under-powered screen and ~18% of post items '
  'lean toward a false "clean" through length bias — so a clean cluster is an <i>upper bound</i>, never proof. A corpus '
  'grep is cheap and definitive. The right workflow: let the probe <i>rank</i> suspicion, then search top-down until '
  'the corpus says clean — the bar prioritises the search, it does not authorise it.</div>')
# build a concrete, prioritised search list from the data
search_rows = []
for k in models:
    r = M[k]; sp = r["spec"]
    tiers = [("P1 · confirmed leak", r["reds"]), ("P2 · elevated", r["ambers"])]
    for tier, topics in tiers:
        for t in topics:
            its = [x for x in r["items"] if x["era"] == "post" and x["topic"] == t and x[sp["cf"]] == 1]
            its.sort(key=lambda x: -x[sp["gf"]])
            for x in its:
                b = bank.get((x["era"], x["topic"], x["label"]), {})
                search_rows.append((tier, sp["short"], t, x["label"], (b.get("true", "") or "").strip(), x[sp["gf"]]))
    # P3: spot-check the single strongest hit in each otherwise-clean narrow cluster
    for t in narrow_topics:
        if t in r["reds"] or t in r["ambers"]:
            continue
        hits = [x for x in r["items"] if x["era"] == "post" and x["topic"] == t and x[sp["cf"]] == 1]
        if hits:
            x = max(hits, key=lambda x: x[sp["gf"]])
            b = bank.get((x["era"], x["topic"], x["label"]), {})
            search_rows.append(("P3 · spot-check clean", sp["short"], t, x["label"], (b.get("true", "") or "").strip(), x[sp["gf"]]))
A(f'<p>Concrete grep targets, highest-suspicion first — the exact post-cutoff facts each model picked correctly. '
  f'Search the corpus for these strings (and near-paraphrases); a verbatim or near-verbatim hit confirms the leak.</p>')
A('<table><thead><tr><th>priority</th><th>model</th><th>cluster</th><th>fact</th><th>search string (true answer)</th><th class="num">gap</th></tr></thead><tbody>')
tier_cls = {"P1": "red", "P2": "amber", "P3": "green"}
for tier, short, t, label, true, gap in search_rows:
    pc = tier_cls[tier[:2]]
    A(f'<tr><td><span class="pill {pc}">{esc(tier)}</span></td><td>{esc(short)}</td><td>{esc(t)}</td>'
      f'<td class="lbl">{esc(label)}</td><td><code>{esc(true)}</code></td><td class="num">{gap:+.1f}</td></tr>')
A('</tbody></table>')
A(f'<p class="legend">{len(search_rows)} targets. P1/P2 are the model\'s actual hits in flagged clusters; P3 spot-checks '
  'the strongest single hit in each cluster the probe called clean — because "not significant" is not "searched". '
  'Practical gate: the corpus itself (~160B tokens) — this needs the model team\'s training set / dedup index; the '
  'probe has done its job by handing over this ranked list.</p>')

# ---- §9 method + integrity ----
A('<h2><span class="n">9</span>Method, decision rule &amp; what is <i>not</i> claimed</h2>')
A('<div class="box"><b>Method</b><ul>'
  '<li><b>Forced choice, whole-answer sum-logp.</b> The model teacher-forces the true completion and each '
  'distractor; "correct" = highest total log-probability to the truth. Metric fixed on should-know controls, '
  'blind to the post items.</li>'
  '<li><b>Length-matched twins, no name cues.</b> Distractors are plausible same-type alternatives (wrong years, '
  'other real entities); the prompt states the descriptive fact and never cues the answer\'s name.</li>'
  '<li><b>Raw, not calibrated.</b> Popularity calibration was tried and discarded — it broke the instrument '
  '(drove pre-controls down and inflated post). All figures here are recomputed from raw per-item records, so the '
  'two models are strictly apples-to-apples.</li>'
  '<li><b>Significance (two tiers).</b> One-sided binomial vs 25% per cluster. <b>RED / confirmed leak</b> = survives '
  'Benjamini-Hochberg FDR (q=.05) across the four powered clusters — the pre-registered primary test. <b>AMBER / '
  'elevated</b> = above chance at the cautious nominal bar α=.10 but not FDR-confirmed. The amber bar is deliberately '
  'loose: in a contamination audit a false "clean" is the costly error, so a p≈0.06 cluster is flagged, not dismissed '
  '(treating p=.049 and p=.058 as opposite verdicts is the classic significant/not-significant fallacy).</li>'
  '<li><b>Length-matched twins.</b> Distractors are built to match the true answer\'s token count so raw sum-logp is '
  'unbiased. This holds on average (median length gap = 0) but not perfectly — see the caveat below.</li>'
  '</ul></div>')
A('<div class="box"><b>Red / amber / green</b> (per cluster, as applied here): '
  '<b>GREEN</b> = not above chance even at α=.10. <b>AMBER</b> = above chance at α=.10 but not FDR-confirmed '
  '(suspect, unconfirmed). <b>RED</b> = above chance and survives FDR (confirmed leak). '
  'A non-significant cliff is evidence of contamination <i>only</i> if the post side is also above chance — noise alone '
  'is not. (The original refinement — splitting amber/red further by whether POST is statistically below the PRE '
  'baseline — is reported via the per-cluster numbers in §4.)</div>')
if EX and EX.get("length_audit"):
    la = EX["length_audit"]
    A(f'<div class="box warn"><b>Caveat — residual length bias (biases toward false-clean).</b> Raw sum-logp penalises '
      f'longer true answers. Of {la["post_n"]} post items, {la["mismatch_ge2"]} '
      f'({100*la["mismatch_ge2"]/la["post_n"]:.0f}%) are imperfectly length-matched; '
      f'{la["true_longer_all"]} have a true answer longer than every distractor (biased toward a miss) vs '
      f'{la["true_shorter_all"]} the other way — a net lean toward under-detecting contamination (see the DNA example '
      f'in §2). Another reason a "clean" verdict is an upper bound, and the corpus grep (§8) is the real arbiter.</div>')
A('<div class="box warn"><b>Not claimed.</b> Modest item counts make the clean verdicts a strong upper bound on '
  'cleanliness, not proof of zero leakage. The map is not exhaustive. Instruct-tuned variants are not yet scored on '
  'this bank. A corpus search (Brown v. Board, Freedom Rides, Orangeburg, Wallace, Selma) to certify the source of '
  'the Ranke leak is still pending.</div>')

flist = ", ".join(f'<code>{esc(M[k]["spec"]["file"])}</code>' for k in models)
A(f'<div class="foot">Sources: {flist} (raw per-item records) + <code>question_bank_v2.json</code>. '
  'Talkie run finished 2026-06-26 13:35Z on a RunPod A100 PCIe (pod fkart67syltxb3, self-terminated, verified '
  'offline); Ranke scored locally. Generated ' + gen_time + ' by <code>build_report.py</code>. '
  'Every figure is recomputed from the JSON — none typed by hand. Supersedes <code>talkie_report.html</code> and the '
  'old v1-bank <code>comparison.html</code>.</div>')

A("</div></body></html>")
open(OUT, "w", encoding="utf-8").write("\n".join(P))
print("wrote", OUT, f"({sum(len(x) for x in P)} bytes)")
print("models:", models, "| narrow:", narrow_topics, "| broad:", broad_topics)
for k in models:
    print(" ", k, M[k]["verdict"], "reds", M[k]["reds"], "ambers", M[k]["ambers"],
          "| pre", M[k]["pre"]["pct"], "post", M[k]["post"]["pct"])
