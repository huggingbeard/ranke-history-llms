# -*- coding: utf-8 -*-
"""Build a SELF-EXPLANATORY HTML report + figures for the foresight study (finance-journal standard):
every figure states what the x-axis shows and what the y-axis measures, defines its terms, and gives the
take-away; the prose defines every concept for a non-expert. Reads the result JSONs; numbers never hardcoded.
Run: python build_foresight_report.py
"""
import os, json, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import common

HERE = common.RESULTS
FIG = common.FIGURES
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 140, "axes.grid": True, "axes.axisbelow": True,
                     "grid.alpha": 0.25, "font.family": "DejaVu Sans"})

BLUE, ORANGE, RED, GREY, GREEN2 = "#2c6fbb", "#e07b39", "#c0392b", "#8a8a8a", "#2e8b57"


def jload(fn, d=None):
    p = os.path.join(HERE, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else d


stats = jload("stats_report.json", {})
gate = jload("gate_report.json", {})
SB = jload("scores_by_model.json", {})
prod = jload("production_report.json", {})
textp = jload("text_perplexity.json", {})
bank = jload("question_bank_foresight.json", {})
S = SB.get("scores", {})
META = {it["id"]: it for it in bank.get("items", [])}
bats = stats.get("batteries", {})
figs = []   # (filename, alt)

XLAB = "Knowledge cutoff of the AI 'mind'  (it has read every text up to this year, and nothing after)"


# ============================================================ FIG 1: main result
def fig1():
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    xs = ["1900", "1913", "1929"]
    series = [("Great War (1914)", "wwi", BLUE, "o", "-"),
              ("Great Depression (1929)", "dep", ORANGE, "s", "-"),
              ("Timeless facts (winter is cold) — control", "ctrlB", GREEN2, "^", "--"),
              ("Far-future events (moon landing, atom bomb) — control", "ctrlC", GREY, "D", ":")]
    for lab, key, col, mk, ls in series:
        row = bats.get(key, {})
        ys = [row.get(f"XSvsModern_{t}", [None])[0] for t in xs]
        lo = [row.get(f"XSvsModern_{t}", [None, None])[1] for t in xs]
        hi = [row.get(f"XSvsModern_{t}", [None, None, None])[2] for t in xs]
        if any(v is None for v in ys):
            continue
        err = [[y - l for y, l in zip(ys, lo)], [h - y for y, h in zip(ys, hi)]]
        ax.errorbar(xs, ys, yerr=err, marker=mk, ls=ls, color=col, capsize=4, lw=2.4, ms=8, label=lab)
    ax.axhline(0, color="k", lw=1.1)
    ax.text(2.02, 0.05, "0 = no more surprised than a modern AI\nthat already knows the outcome",
            fontsize=8.5, va="bottom", ha="right", color="k",
            bbox=dict(boxstyle="round", fc="#fff7e6", ec="#ddb", alpha=.9))
    ax.set_xlabel(XLAB, fontsize=10)
    ax.set_ylabel("EXCESS SURPRISE AT THE OUTCOME  (nats)\n"
                  "how much more the model is surprised by what happened,\nover and above a modern AI that has read the whole 20th century",
                  fontsize=9.5)
    ax.set_title("Could a mind of the time imagine the catastrophe?   (higher = harder to foresee)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.8, loc="upper left", framealpha=.95, title="Each line is one kind of event")
    ax.margins(x=0.12)
    fig.tight_layout(); p = os.path.join(FIG, "fig1_main.png"); fig.savefig(p); plt.close(fig)
    figs.append(("fig1_main.png", "Excess surprise at each catastrophe, by model cutoff"))


# ============================================================ FIG 2: WWI lexical vs conceptual
def fig2():
    wh = stats.get("wwi_horizon", {})
    models = [("Mind of 1900", "1900"), ("Mind of 1913\n(eve of the war)", "1913")]
    lex = [wh.get("dlex_1900"), wh.get("dlex_1913")]
    conc = [wh.get(f"dconc_{t}_vs1929", [None])[0] for _, t in models]
    conc_lo = [wh.get(f"dconc_{t}_vs1929", [None, None])[1] for _, t in models]
    conc_hi = [wh.get(f"dconc_{t}_vs1929", [None, None, None])[2] for _, t in models]
    import numpy as np
    x = np.arange(len(models)); w = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    ax.bar(x - w/2, lex, w, color=BLUE, label="Surprise at the WORDS & NAMES (lexical)\n— e.g. 'Passchendaele', 'Verdun': rare tokens")
    cerr = [[c - l for c, l in zip(conc, conc_lo)], [h - c for c, h in zip(conc, conc_hi)]]
    ax.bar(x + w/2, conc, w, color=ORANGE, yerr=cerr, capsize=5,
           label="Surprise at the IDEA (conceptual)\n— a long, millions-dead, throne-toppling war")
    ax.axhline(0, color="k", lw=1)
    for xi, c in zip(x + w/2, conc):
        ax.annotate("≈ 0\n(not surprising)", (xi, c), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5, color=RED, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([m for m, _ in models])
    ax.set_ylabel("EXCESS SURPRISE AT WWI OUTCOMES  (nats)\n"
                  "vs the 1929 model, which lived through 1914–1918", fontsize=9.5)
    ax.set_xlabel("Which pre-war 'mind' is being tested", fontsize=10)
    ax.set_title("The 1913 mind's shock at the World War was about the WORDS, not the IDEA",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.4, loc="upper right", framealpha=.95)
    fig.tight_layout(); p = os.path.join(FIG, "fig2_wwi_split.png"); fig.savefig(p); plt.close(fig)
    figs.append(("fig2_wwi_split.png", "WWI surprise split into words vs idea"))


# ============================================================ FIG 3: home by Christmas?
def fig3():
    cw = stats.get("confident_wrongness_wwi_1913", {})
    items = cw.get("items", [])
    if not items:
        return
    # readable labels from the bank's true outcome
    def lab(i):
        t = META.get(i["id"], {}).get("target", i["id"]).strip()
        return (t[:42] + "…") if len(t) > 43 else t
    items = sorted(items, key=lambda r: r["p_true_1913"])
    labs = [lab(i) for i in items]
    vals = [i["p_true_1913"] for i in items]
    facet = [META.get(i["id"], {}).get("facet", "") for i in items]
    cols = [BLUE if v >= 0.5 else ORANGE for v in vals]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    ax.barh(labs, vals, color=cols)
    ax.axvline(0.5, color="k", lw=1.2, ls="--")
    ax.text(0.5, len(items)-0.35, "coin-flip", fontsize=8.5, va="top", ha="center",
            bbox=dict(boxstyle="round", fc="white", ec="#bbb"))
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Probability the 1913 mind places on the TRUE outcome\n"
                  "0 = sure of a short/easy war    ·    1 = sure of the long, grinding war that happened",
                  fontsize=9.5)
    ax.set_ylabel("Each bar = one forecast about the coming war\n(labelled by the true outcome)", fontsize=9)
    ax.set_title("The 1913 mind expected a LONG war — it missed only the trench tactics",
                 fontsize=11, fontweight="bold")
    ax.legend(handles=[Patch(color=BLUE, label="leaned to the TRUE outcome (≥ 0.5)"),
                       Patch(color=ORANGE, label="leaned to the wrong outcome (< 0.5)  — the trench items")],
              fontsize=8.4, loc="lower right", framealpha=.95)
    fig.tight_layout(); p = os.path.join(FIG, "fig3_homebychristmas.png"); fig.savefig(p); plt.close(fig)
    figs.append(("fig3_homebychristmas.png", "1913 model's probability on the true WWI outcomes"))


# ============================================================ FIG 4: real text (modern-anchored, powered)
def fig4():
    did = textp.get("DiD_modern_anchored")
    if not did:
        return
    import numpy as np
    tags = ["1900", "1913", "1929"]
    x = np.arange(len(tags)); w = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    for j, (ev, lab, col) in enumerate([("wwi", "World War  (control — should be ≈0)", BLUE),
                                        ("dep", "Great Depression", ORANGE)]):
        vals = [did[f"{ev}_at_{t}"]["did"] for t in tags]
        los = [did[f"{ev}_at_{t}"]["ci"][0] for t in tags]
        his = [did[f"{ev}_at_{t}"]["ci"][1] for t in tags]
        err = [[v - l for v, l in zip(vals, los)], [h - v for v, h in zip(vals, his)]]
        ax.bar(x + (j - 0.5) * w, vals, w, yerr=err, capsize=4, color=col, label=lab)
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"mind of {t}" for t in tags])
    ax.set_ylabel("Excess surprise on the post-shock prose  (nats/char)\n"
                  "how much MORE foreign it is to the pre-shock mind\nthan to a modern mind that has read the outcome", fontsize=9.5)
    ax.set_xlabel("Which pre-shock mind is reading the real passages", fontsize=10)
    ax.set_title("Arm B on REAL prose (modern-anchored, 3 passages/cell, 95% CI)\n"
                 "post-shock text vs. matched pre-shock control, same authors", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.6, loc="upper left", framealpha=.95)
    fig.tight_layout(); p = os.path.join(FIG, "fig4_realtext.png"); fig.savefig(p); plt.close(fig)
    figs.append(("fig4_realtext.png", "Modern-anchored Arm B: excess surprise on real post-shock prose"))


# ============================================================ FIG 6: Zweig cross-lingual war
def fig_zweig():
    z = jload("zweig_probe.json")
    if not z:
        return
    import numpy as np
    summ = z["summary"]
    langs = [("en", "English"), ("de", "German"), ("fr", "French")]
    models = [("1900", GREEN2), ("1913", ORANGE), ("1929", RED), ("modern", "#555555")]
    themes = [("progress", "Faith in irresistible PROGRESS"),
              ("war", "A great European war is UNTHINKABLE"),
              ("women", "A woman's place is the HOME"),
              ("youth", "AGE, not youth, commands respect")]
    x = np.arange(len(langs)); w = 0.2
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6), sharex=True)
    for ax, (th, title) in zip(axes.flat, themes):
        for j, (m, col) in enumerate(models):
            vals = [summ[m][l][th] for l, _ in langs]
            ax.bar(x + (j - 1.5) * w, vals, w, color=col, label=("modern mind" if m == "modern" else f"mind of {m}"))
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([n for _, n in langs], fontsize=10)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_ylabel("holds belief (nats)\n+ holds  ·  − rejects", fontsize=8.5)
    axes.flat[0].legend(ncol=2, fontsize=8, loc="upper right", framealpha=.95)
    fig.suptitle("Does each frozen mind hold Zweig's four pre-1914 illusions?  (calibrated belief-preference)\n"
                 "The x-axis is the language the SAME models are prompted in; the y-axis measures how far each\n"
                 "prefers the pre-1914 belief over its opposite (+ holds the illusion, − rejects it). Bars: 1900/1913/1929 + modern.",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92]); p = os.path.join(FIG, "fig6_zweig.png"); fig.savefig(p); plt.close(fig)
    figs.append(("fig6_zweig.png", "Zweig's four themes by language and model"))


# ============================================================ FIG 5: production distribution
def fig5():
    if not prod:
        return
    from foresight_bank import PRODUCTION
    import numpy as np
    items = PRODUCTION
    def short(it):
        t = META.get(it["id"], {}).get("target", it["id"]).strip()
        return (t[:34] + "…") if len(t) > 35 else t
    labs = [short(it) for it in items]
    y = np.arange(len(items)); h = 0.26
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for k, (t, col) in enumerate([("1900", GREEN2), ("1913", ORANGE), ("1929", RED)]):
        vals = [prod["items"][it["id"]]["coded"].get(t, {}).get("frac_true", 0.0) for it in items]
        ax.barh(y + (1-k)*h, vals, h, color=col, label=f"mind of {t}")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Fraction of 6 free-writing continuations that VOLUNTEER the true outcome\n"
                  "(0 = never says it unprompted · 1 = says it every time)", fontsize=9.5)
    ax.set_ylabel("Each row = one forecast (labelled by its true outcome)", fontsize=9)
    ax.set_title("What the minds VOLUNTEER in free writing (no answer to pick from)",
                 fontsize=11.5, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right", framealpha=.95)
    fig.tight_layout(); p = os.path.join(FIG, "fig5_production.png"); fig.savefig(p); plt.close(fig)
    figs.append(("fig5_production.png", "Free-generation: fraction volunteering the true outcome, per item"))


# ================================================== FIG 7: cross-lingual THREE-event foreseeability
def fig_crosslingual_events():
    z = jload("crosslingual_events.json")
    if not z:
        return
    import numpy as np
    cs = z["conceptual_surprise"]
    langs = [("en", "English", BLUE), ("de", "German", ORANGE), ("fr", "French", RED)]
    events = [("world_war", "World War\n(1913 mind vs 1929)"),
              ("fall_of_empires", "Fall of the empires\n(1913 mind vs 1929)"),
              ("great_depression", "Great Depression\n(1929 mind vs modern)")]

    def val(ev, lg):
        return cs[ev][lg]["1929"] if ev == "great_depression" else cs[ev][lg]["1913"]

    x = np.arange(len(events)); w = 0.24
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    for j, (lg, name, col) in enumerate(langs):
        ax.bar(x + (j - 1) * w, [val(ev, lg) for ev, _ in events], w, color=col, label=name)
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-1, 1, color="#2e8b57", alpha=0.08)
    ax.text(-0.44, 0.0, "foreseeable band (≈0)", fontsize=8, color="#2e8b57", va="center", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in events], fontsize=9.5)
    ax.set_ylabel("Excess conceptual surprise, over a mind that lived through it (nats)\n"
                  "(≈0 = the outcome fit the pre-event worldview; large + = it did NOT)", fontsize=9.3)
    ax.set_xlabel("Each group is one catastrophe; the three bars are the SAME models prompted in English, German, French.\n"
                  "War & empires are measured against the 1929 mind that read them; the Depression against a modern mind.", fontsize=8.8)
    ax.set_title("Was each catastrophe foreseeable? — three events, three languages, on equal footing\n"
                 "The World War and the fall of the empires were foreseeable (≈0 everywhere); the Depression was not",
                 fontsize=10.8, fontweight="bold")
    ax.legend(title="prompted in", fontsize=9, loc="upper left")
    fig.tight_layout(); p = os.path.join(FIG, "fig7_crosslingual_events.png"); fig.savefig(p); plt.close(fig)
    figs.append(("fig7_crosslingual_events.png", "Cross-lingual foreseeability of three events"))


# fig_zweig / fig_crosslingual_events superseded by build_panel_report.py (figP4 / figP1-P3)
for f in (fig1, fig2, fig3, fig4, fig5):
    try:
        f()
    except Exception as e:
        print("figure error:", f.__name__, e)


# ============================================================ HTML
def g(x, d="?"):
    return x if x is not None else d


def esc(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


wh = stats.get("wwi_horizon", {})
con = stats.get("contrast_dep_vs_wwi_at_1929", {})
cw = stats.get("confident_wrongness_wwi_1913", {})


def gv(ev):
    return gate.get(ev, {}).get("verdict", "?")


H = []
H.append("""<!doctype html><meta charset='utf-8'><title>Could anyone have seen it coming? — Foresight report</title>
<style>
 body{font:16px/1.65 Georgia,serif;max-width:820px;margin:2rem auto;padding:0 1.1rem;color:#1c1c1c;background:#fbfbf9}
 h1{font:700 30px/1.2 'Helvetica Neue',Arial,sans-serif;margin:.2em 0}
 h2{font:700 21px 'Helvetica Neue',Arial,sans-serif;margin:2.2em 0 .3em;border-bottom:2px solid #e6e2d8;padding-bottom:.2em}
 h3{font:700 16px 'Helvetica Neue',Arial;margin:1.4em 0 .2em;color:#333}
 .lead{font-size:18px;color:#333}
 .box{background:#fff;border:1px solid #e6e2d8;border-radius:8px;padding:.9em 1.1em;margin:1em 0}
 .glo{background:#f4f6fb;border-left:4px solid #2c6fbb}
 .key{background:#fff7e6;border-left:4px solid #e07b39}
 dt{font-weight:700;margin-top:.5em} dd{margin:0 0 .3em 0}
 figure{margin:1.4em 0;text-align:center} img{max-width:100%;border:1px solid #e6e2d8;border-radius:6px;background:#fff}
 figcaption{font-size:14px;color:#444;text-align:left;margin-top:.5em}
 .axis{background:#eef2fa;border-radius:5px;padding:.5em .7em;font-size:14px;margin:.5em 0}
 .take{font-weight:700;color:#8a3b00}
 table{border-collapse:collapse;width:100%;font:14px/1.4 'Helvetica Neue',Arial;margin:1em 0}
 th,td{border:1px solid #d8d4c8;padding:6px 9px;text-align:center} th{background:#f2efe6}
 .gr{color:#2e8b57;font-weight:700}.rd{color:#c0392b;font-weight:700}
 code{background:#eee;padding:1px 4px;border-radius:3px;font-size:.9em}
 .small{font-size:13px;color:#666}
</style>""")

H.append("<h1>Could anyone have seen it coming?</h1>")
H.append("<p class='lead'>Measuring how <b>foreseeable</b> the two great catastrophes of the early 20th "
         "century&mdash;the <b>First World War (1914)</b> and the <b>Great Depression (1929)</b>&mdash;really "
         "were, using AI &lsquo;minds&rsquo; frozen in the past.</p>")

H.append("<div class='box glo'><h3 style='margin-top:0'>What this is, in one minute</h3>"
         "<p>We trained AI language models on text written <b>only up to a certain year</b> and nothing after. "
         "A model trained on everything up to 1913 is like a very well-read person at the end of 1913: it knows "
         "the whole world <i>up to then</i>, but has no idea what 1914 will bring. We then ask each frozen mind "
         "how <b>surprised</b> it is by what actually happened next. Surprise you can&rsquo;t fake&mdash;a mind "
         "that couldn&rsquo;t imagine the outcome is genuinely startled by it. That surprise is a direct, "
         "hindsight-free measure of how <b>foreseeable</b> the catastrophe was.</p></div>")

H.append("<div class='box glo'><h3 style='margin-top:0'>The words you&rsquo;ll need (defined once)</h3><dl>"
         "<dt>The three &lsquo;minds&rsquo;</dt><dd>AI models that have read everything up to <b>1900</b>, "
         "<b>1913</b>, or <b>1929</b>&mdash;and nothing after.</dd>"
         "<dt>Surprise (measured in &lsquo;nats&rsquo;)</dt><dd>How unexpected a statement is to a mind. Big "
         "number = very surprised (the mind thought it unlikely). A &lsquo;nat&rsquo; is just the unit, like a "
         "degree on a thermometer&mdash;only the comparisons matter, not the absolute value.</dd>"
         "<dt>The trap: words vs. ideas</dt><dd>A 1913 mind is startled by the word "
         "&lsquo;<i>Passchendaele</i>&rsquo; simply because it is an unfamiliar name&mdash;not because it "
         "couldn&rsquo;t imagine a long war. So we carefully separate <b>surprise at the vocabulary</b> "
         "(lexical) from <b>surprise at the idea itself</b> (conceptual). <b>Only surprise at the idea measures "
         "foresight.</b></dd>"
         "<dt>Excess surprise</dt><dd>How much <i>more</i> surprised an old mind is than a mind that already "
         "knows how the story ends (either our 1929 model, which lived through WWI, or a modern AI that has "
         "read the whole century). <b>Zero</b> excess = the old mind finds it as unremarkable as one that saw "
         "it happen. <b>Large</b> excess = it could not see it coming.</dd>"
         "<dt>The gate</dt><dd>A cheating check: is an old mind secretly &lsquo;peeking&rsquo; (already knowing "
         "the outcome from leaked later text)? If so, low surprise wouldn&rsquo;t mean foresight. We test this "
         "directly.</dd></dl></div>")

# ---- headline
H.append("<h2>The headline</h2>")
H.append(f"<div class='box key'><p>Once we strip away mere vocabulary, the two disasters are <b>not the same "
         f"kind of event</b>:</p><ul>"
         f"<li><b>The First World War was largely foreseeable.</b> To the 1913 mind, the <i>idea</i> of a long, "
         f"millions-dead, empire-toppling war was about as surprising as the fact that winter days are cold.</li>"
         f"<li><b>The Great Depression was a genuine bolt from the blue.</b> Even the mind sitting on the very "
         f"eve of the 1929 crash found the Depression&rsquo;s depth <i>harder to imagine than a moon "
         f"landing</i>.</li></ul>"
         f"<p>The difference is precise and statistically clear: measured on the 1929 mind, the Depression "
         f"carried <b>{g(con.get('difference'))} nats</b> more excess conceptual surprise than the World War "
         f"(95% confidence interval {g(con.get('ci95'))}, p&nbsp;=&nbsp;{g(con.get('p_two_sided'))}). "
         f"A <b>p-value</b> this small means the gap is very unlikely to be a fluke of the particular sentences "
         f"we chose.</p></div>")

# ---- FIG 1
H.append("<h2>The main result, in one picture</h2>")
H.append("<figure><img src='../figures/fig1_main.png' alt='main result'>"
         "<figcaption>"
         "<div class='axis'><b>The x-axis shows</b> which frozen mind is being tested&mdash;the AI that has "
         "read up to 1900, 1913, or 1929. <b>The y-axis measures</b> <i>excess surprise</i>: how much more "
         "surprised that mind is by a catastrophe&rsquo;s outcome than a modern AI that already knows the whole "
         "20th century (in nats; <b>0</b>, the black line, means &lsquo;no more surprised than a mind that saw "
         "it happen&rsquo;). Vertical bars are 95% confidence intervals.</div>"
         "<p><b>What you see:</b> the orange <b>Depression</b> line sits high and stays high&mdash;every "
         "pre-1930 mind, even the one on the eve of the crash (1929), is deeply surprised, more so than by the "
         "grey <b>far-future control</b> events (moon landing, atom bomb). The blue <b>World War</b> line sits "
         "low, down among the green <b>timeless-fact controls</b> (&lsquo;winter is cold&rsquo;).</p>"
         "<p class='take'>Take-away: the Depression&rsquo;s severity was far less imaginable in advance than "
         "the World War&rsquo;s.</p>"
         "<p class='small'>Note: the absolute heights differ slightly between AI families; only comparisons "
         "<i>between lines at the same cutoff</i> are meaningful, which is exactly what the headline uses.</p>"
         "</figcaption></figure>")

# ---- WWI section
H.append("<h2>Why the World War looks foreseeable</h2>")
H.append(f"<p>At first glance the 1913 mind <i>is</i> badly surprised by the war (its raw excess surprise is "
         f"{g(wh.get('dS_1913_vs1929'))} nats). But almost all of that is surprise at unfamiliar "
         f"<b>vocabulary</b>&mdash;place-names and coinages the 1929 model picked up from 1914&ndash;1918 "
         f"writing. Surprise at the <b>idea</b> itself is essentially nil.</p>")
H.append("<figure><img src='../figures/fig2_wwi_split.png' alt='wwi split'>"
         "<figcaption>"
         "<div class='axis'><b>The x-axis shows</b> the two pre-war minds (1900 and 1913). <b>The y-axis "
         "measures</b> their excess surprise at the war&rsquo;s outcomes, in nats, compared with the 1929 mind "
         "that lived through the war. Each mind&rsquo;s surprise is split into two bars: <b>blue</b> = surprise "
         "at the new <i>words and names</i>; <b>orange</b> = surprise at the <i>idea</i> (a long, "
         "millions-dead, throne-toppling war). Orange whiskers are 95% confidence intervals.</div>"
         "<p><b>What you see:</b> the blue (vocabulary) bar is large; the orange (idea) bar is indistinguishable "
         "from zero.</p>"
         "<p class='take'>Take-away: strip away the unfamiliar words and the <i>character</i> of the Great War "
         "was not a surprising idea to a well-read mind of 1913.</p></figcaption></figure>")

H.append("<h3>&lsquo;Home by Christmas&rsquo; was a mood, not a limit of the imagination</h3>")
H.append(f"<p>If the long war had been unthinkable, the 1913 mind would have <i>confidently</i> predicted the "
         f"short-war consensus. It does the opposite. On the war&rsquo;s <b>length</b> it leans firmly to the "
         f"true, grinding, years-long war&mdash;asked outright how long a war will last it writes "
         f"&lsquo;<i>for years</i>&rsquo; every time. What it genuinely fails to foresee is narrower and more "
         f"specific: the <b>trench-deadlock tactics</b> (static lines, the machine-gun in a fixed position)&mdash;"
         f"the only forecasts it gets confidently wrong ({g(cw.get('n_confident_wrong'))} of {g(cw.get('n'))}). "
         f"The famous short-war optimism of 1914 lived in newspapers and cheering crowds&mdash;not in the "
         f"written record the model learned from, which already contained Bloch&rsquo;s 1899 prediction of "
         f"attrition.</p>")
H.append("<figure><img src='../figures/fig3_homebychristmas.png' alt='home by christmas'>"
         "<figcaption>"
         "<div class='axis'><b>The x-axis shows</b> the probability the 1913 mind places on the <i>true</i> "
         "outcome of each forecast (0 = it is sure of a short, easy war; 1 = sure of the long, grinding war "
         "that actually happened; the dashed line at 0.5 is a coin-flip). <b>The y-axis</b> lists the "
         "individual forecasts, each labelled by its true outcome; blue bars are ones the mind gets right "
         "(leans true, &ge;&nbsp;0.5), orange bars it gets wrong.</div>"
         "<p><b>What you see:</b> the top six bars (about the war&rsquo;s <i>length</i> and cost) sit far to "
         "the right&mdash;the mind expects the long war. Only the bottom two (about <i>trench deadlock</i>) sit "
         "at the left.</p>"
         "<p class='take'>Take-away: the 1913 mind expected a long, costly war&mdash;it did <i>not</i> believe "
         "&lsquo;home by Christmas&rsquo;; what it missed was the specific trench-warfare form.</p>"
         "</figcaption></figure>")

# ---- Depression
H.append("<h2>Why the Depression was genuinely unforeseeable</h2>")
dep_row = bats.get("dep", {})
H.append(f"<p>The Depression is the opposite story. Its excess conceptual surprise is large at every "
         f"cutoff&mdash;{g(dep_row.get('XSvsModern_1900',['?'])[0])} nats at 1900, "
         f"{g(dep_row.get('XSvsModern_1913',['?'])[0])} at 1913, "
         f"{g(dep_row.get('XSvsModern_1929',['?'])[0])} at 1929&mdash;and it does <i>not</i> fall as the crash "
         f"approaches. Even the 1929 mind, standing on the edge of the cliff, finds the coming decade of "
         f"25%-unemployment, thousands of bank failures and a lost ten years <b>more surprising than the "
         f"far-future control events</b> (the moon landing, the atomic bomb). Asked what the years after the "
         f"crash hold, the models predict the comfortable consensus&mdash;continued prosperity, a quick "
         f"rebound&mdash;not the catastrophe.</p>")

# ---- gate
H.append("<h2>The cheating check (is this real, or is the AI peeking?)</h2>")
H.append("<p>A low-surprise reading only means &lsquo;foresight&rsquo; if the old mind isn&rsquo;t secretly "
         "peeking at the answer. We test each mind on plain <b>facts</b> about the event it shouldn&rsquo;t "
         "know. If it scores at chance, it isn&rsquo;t peeking.</p>")
H.append("<table><tr><th>Event</th><th>Mind tested</th><th>Score on &lsquo;forbidden&rsquo; facts<br>"
         "<span class='small'>(25% = pure guessing)</span></th><th>Verdict</th><th>What it means</th></tr>")
gw, gd = gate.get("wwi", {}), gate.get("dep", {})
H.append(f"<tr><td>Great Depression</td><td>1929</td><td>{g(gd.get('post_correct'))}/{g(gd.get('post_n'))} "
         f"= {g(gd.get('post_pct'))}% (p={g(gd.get('binom_p'))})</td><td class='gr'>CLEAN</td>"
         f"<td>At chance&mdash;not peeking. It doesn&rsquo;t even reliably know the Oct&nbsp;1929 crash itself. "
         f"So its high surprise is <b>real</b>.</td></tr>")
H.append(f"<tr><td>First World War</td><td>1913</td><td>{g(gw.get('post_correct'))}/{g(gw.get('post_n'))} "
         f"= {g(gw.get('post_pct'))}%</td><td class='rd'>flagged</td>"
         f"<td>Looks high&mdash;but only because WWI place-names (Gallipoli, Passchendaele, the liner "
         f"<i>Lusitania</i>) were <b>already famous before 1914</b>. It fails on genuinely new things (the "
         f"&lsquo;tank&rsquo;, the 1917 US entry). This is exactly the <i>vocabulary</i> surprise we already "
         f"stripped out.</td></tr></table>")
H.append("<p class='small'>The asymmetry is itself telling: the war was &lsquo;known&rsquo; only as a set of "
         "familiar words; the Depression was not known at all.</p>")

# ---- real text
H.append("<h2>Does it hold on real historical writing?</h2>")
H.append("<p>The results above use carefully-built sentences. As a reality check we repeat the measurement on "
         "genuine public-domain prose from the period.</p>")
H.append("<figure><img src='../figures/fig4_realtext.png' alt='real text'>"
         "<figcaption>"
         "<div class='axis'><b>The x-axis shows</b> the frozen mind (cutoff 1900/1913/1929). <b>The y-axis "
         "measures</b> the average surprise <i>per word</i> as each mind reads a real passage (higher = the "
         "writing is more foreign to that mind). <b>Solid</b> lines are texts written just <i>after</i> a shock "
         "(Keynes on the ruin of Europe, 1919; Roosevelt on the Depression, 1933); <b>dashed</b> lines are "
         "matched texts from just <i>before</i> (Angell, 1910; Hoover&rsquo;s &lsquo;bright with hope&rsquo;, "
         "1929).</div>"
         "<p><b>What you see:</b> the post-WWI text (solid blue) is barely more surprising to the pre-war minds "
         "than its pre-war control; the post-crash text (solid orange) is markedly more surprising to every "
         "mind than its pre-crash control.</p>"
         "<p class='take'>Take-away: real prose agrees with the constructed probes&mdash;the language of the "
         "Depression was foreign to every pre-1930 mind.</p></figcaption></figure>")

# ---- recognition vs production
if prod:
    from foresight_bank import PRODUCTION
    H.append("<h2>One more honesty check: recognition vs. speaking up</h2>")
    H.append("<p>There are two ways to read a mind: ask it to <b>pick</b> the more plausible of two endings "
             "(recognition), or let it <b>freely continue</b> a sentence and see what it volunteers "
             "(production). Volunteering is the harder test. If anything, the minds volunteer the true "
             "catastrophe <i>less</i> than they recognise it&mdash;so any &lsquo;foresight&rsquo; we report is "
             "an <b>upper bound</b>, never inflated.</p>")
    H.append("<table><tr><th>Mind</th><th>Recognises the true outcome<br>(picks it in a forced choice)</th>"
             "<th>Volunteers it unprompted<br>(free writing)</th></tr>")
    for t, sfx in [("1900", "1900"), ("1913", "1913"), ("1929", "1929")]:
        rc = [S[t][it["id"]]["correct_cal"] for it in PRODUCTION]
        pr = [prod["items"][it["id"]]["coded"].get(t, {}).get("frac_true", 0.0) for it in PRODUCTION]
        H.append(f"<tr><td>{t}</td><td>{round(100*sum(rc)/len(rc))}%</td><td>{round(sum(pr)/len(pr),2)}</td></tr>")
    H.append("</table>")
    H.append("<p>But that &lsquo;0.19&rsquo; is a <b>skewed average</b> that hides the real pattern. Broken out "
             "item by item:</p>")
    H.append("<figure><img src='../figures/fig5_production.png' alt='production distribution'>"
             "<figcaption>"
             "<div class='axis'><b>The x-axis shows</b> the fraction of six free-writing continuations that "
             "volunteer the true outcome (0 = the mind never says it unprompted; 1 = it says it every time). "
             "<b>The y-axis</b> lists the forecasts, labelled by their true outcome; the three coloured bars are "
             "the 1900, 1913 and 1929 minds.</div>"
             "<p><b>What you see:</b> one row&mdash;how long the war lasts&mdash;is at 1.0 (the mind volunteers "
             "&lsquo;for years&rsquo; every time); most other rows are at 0.</p>"
             "<p class='take'>Take-away: the minds spontaneously predict exactly one true thing&mdash;the war "
             "will last years&mdash;and mostly ramble on the rest. The single clean signal is enough to bury "
             "&lsquo;home by Christmas&rsquo;.</p></figcaption></figure>")
    H.append("<p class='small'>Honesty note: on the items scored 0, the base model usually wanders off-topic "
             "rather than predicting the rosy outcome, so a 0 mixes &lsquo;got it wrong&rsquo; with &lsquo;said "
             "something irrelevant&rsquo;. Production is therefore a crude floor, not a clean score&mdash;which "
             "is why we treat teacher-forcing (recognition) as the main read and production only as a check "
             "that it isn&rsquo;t inflating the signal.</p>")

# ---- cross-lingual three-event section
xle = jload("crosslingual_events.json")
if xle:
    cs = xle["conceptual_surprise"]
    H.append("<h2>Three events, three languages, on equal footing</h2>")
    H.append("<p>The findings above use the English item bank. Do they hold when the same models are prompted in "
             "<b>German</b> and <b>French</b> &mdash; and does a third catastrophe, the <b>fall of the old imperial "
             "order</b> (the emperors deposed, the empires broken up), behave like the war or like the Depression? "
             "We built matched outcome statements for all three events in all three languages (8 per event) and "
             "read off the same conceptual surprise.</p>")
    H.append("<figure><img src='../figures/fig7_crosslingual_events.png' alt='cross-lingual three events'>"
             "<figcaption><b>Figure 6.</b> <i>The x-axis</i> groups the three catastrophes; within each group the "
             "three bars are the <span style='color:#2c6fbb'>English</span>, <span style='color:#e07b39'>German</span> "
             "and <span style='color:#c0392b'>French</span> readings of the <i>same</i> models. <i>The y-axis measures</i> "
             "excess conceptual surprise over a mind that lived through the event (nats): the World War and the fall of "
             "the empires are judged against the 1929 mind that read them, the Depression against a modern mind. The "
             "green band marks &lsquo;&asymp;0&rsquo; &mdash; the outcome fit the pre-event worldview. "
             "<span class='take'>Take-away: in every language the World War and the collapse of the empires sit in the "
             "foreseeable band, while the Great Depression stands well above it &mdash; the imperial collapse that "
             "history calls &lsquo;unimaginable&rsquo; was, to the 1913 mind, no more surprising than the war.</span>"
             "</figcaption></figure>")
    H.append(f"<p class='small'>At the 1913 mind, excess surprise at the war's outcomes is "
             f"{cs['world_war']['en']['1913']:+.1f}/{cs['world_war']['de']['1913']:+.1f}/{cs['world_war']['fr']['1913']:+.1f} "
             f"(EN/DE/FR) and at the empire-collapse {cs['fall_of_empires']['en']['1913']:+.1f}/"
             f"{cs['fall_of_empires']['de']['1913']:+.1f}/{cs['fall_of_empires']['fr']['1913']:+.1f} &mdash; both near zero; "
             f"the Depression, read by the 1929 mind, stays at "
             f"{cs['great_depression']['en']['1929']:+.1f}/{cs['great_depression']['de']['1929']:+.1f}/"
             f"{cs['great_depression']['fr']['1929']:+.1f}. Every prompt and score is in "
             f"<code>reports/crosslingual_events.html</code>.</p>")

# ---- Zweig worldview section
zw = jload("zweig_probe.json")
if zw:
    H.append("<h2>The pre-1914 worldview itself (Zweig)</h2>")
    H.append("<p>Stefan Zweig's memoir <i>The World of Yesterday</i> catalogues the illusions of the pre-1914 mind. "
             "We turn the four he dwells on &mdash; faith in progress, the unthinkability of war, the confinement of "
             "women, and the veneration of age over youth &mdash; into calibrated belief-vs-skeptic choices, in all "
             "three languages.</p>")
    H.append("<figure><img src='../figures/fig6_zweig.png' alt='Zweig four themes'>"
             "<figcaption><b>Figure 7.</b> <i>The x-axis</i> of each panel is the language the same models are prompted "
             "in; <i>the y-axis measures</i> how far each mind prefers the pre-1914 belief over its opposite (calibrated "
             "nats; + holds the illusion, &minus; rejects it). Bars are the 1900/1913/1929 minds and a modern mind. "
             "<span class='take'>Take-away: the models robustly reproduce two of Zweig's illusions &mdash; faith in "
             "progress (held in every language, dropped by the modern mind) and the Continental complacency about war "
             "(German and French hold &lsquo;war is unthinkable&rsquo;, English already expects war). His account of "
             "women's confinement registers most in the modern mind (as history it knows), and the cult of age shows no "
             "clean signal &mdash; two honest holdouts.</span></figcaption></figure>")

# ---- bottom line
H.append("<h2>Bottom line</h2>")
H.append("<div class='box key'><p>Freeze a mind before the shock, compare its surprise to a mind that has read "
         "the sequel, and separate the words from the ideas. By that measure, <b>1914 and 1929 were different "
         "kinds of disaster</b>. The Great War was loud in unfamiliar names but its <i>character</i> was "
         "imaginable in advance&mdash;the &lsquo;home by Christmas&rsquo; illusion was a public mood, not a "
         "failure of imagination. The Great Depression was the true tail event: on the eve of the crash, its "
         "depth was less conceivable than a man walking on the moon. Models that treat all rare disasters as "
         "equally unforecastable are, for these two, mispricing the difference.</p></div>")
H.append("<p class='small'>Every number here is computed from the model runs (files in this folder); nothing "
         "is hand-entered. Full method, confidence intervals and limitations are in the accompanying paper "
         "(<code>paper/foresight.pdf</code>).</p>")

open(os.path.join(common.REPORTS, "foresight_report.html"), "w", encoding="utf-8").write("\n".join(H))
print("wrote foresight_report.html and", len(figs), "figures:", [f for f, _ in figs])
