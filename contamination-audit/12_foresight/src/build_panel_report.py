# -*- coding: utf-8 -*-
"""Panel figures + the transparent panel HTML (finance-journal standard: every figure states what
the x-axis shows and the y-axis measures, and gives the take-away).

Figures (figures/):
  figP1_ranking.png    the headline: eight events ranked by pooled Dconc, floor & ceiling bands
  figP2_bylang.png     the same per language (three panels)
  figP3_truefalse.png  true-vs-false discrimination by model (pooled)
  figP4_zweig.png      Zweig v3 (four themes x four models x three languages)
Report: reports/panel_events.html — every stem, target, neutral and score, color-coded.
Run: python src/build_panel_report.py
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common
from panel_events import EVENTS_META, ITEMS, TIER_B, TIER_C, LANGS
from panel_checks import TRUEFALSE

plt.rcParams.update({"font.size": 11, "figure.dpi": 140, "axes.grid": True, "axes.axisbelow": True,
                     "grid.alpha": 0.25, "font.family": "DejaVu Sans"})
BLUE, ORANGE, RED, GREY, GREEN2 = "#2c6fbb", "#e07b39", "#c0392b", "#8a8a8a", "#2e8b57"


def jload(fn):
    p = os.path.join(common.RESULTS, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


ST = jload("stats_panel.json")
SC = jload("panel_scores.json")
GT = jload("gate_panel.json")
LN = {"en": "English", "de": "German", "fr": "French"}


def fig_ranking():
    rank = ST["ranking"]
    labels = [r["label"] for r in rank]
    means = [r["dconc_pooled"] for r in rank]
    lo = [r["dconc_pooled"] - r["ci"][0] for r in rank]
    hi = [r["ci"][1] - r["dconc_pooled"] for r in rank]
    colors = [BLUE if r["anchor"] != "modern" else ORANGE for r in rank]
    # floor / ceiling bands (pooled across languages, at the 1913 mind)
    fB = [ST["tiers"][lg]["tierB_vs1929_1913"][0] for lg in LANGS]
    fC = [ST["tiers"][lg]["tierC_bridged_1913"][0] for lg in LANGS]
    y = range(len(rank))
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    ax.axvspan(min(fB) - 0.3, max(fB) + 0.3, color=GREEN2, alpha=0.13)
    ax.text((min(fB) + max(fB)) / 2, len(rank) - 0.45, "timeless-truth floor\n(foreseeable)",
            ha="center", fontsize=8, color=GREEN2)
    ax.axvspan(min(fC) - 0.3, max(fC) + 0.3, color=GREY, alpha=0.18)
    ax.text((min(fC) + max(fC)) / 2, len(rank) - 0.45, "moon-landing ceiling\n(unforeseeable to all)",
            ha="center", fontsize=8, color="#555")
    ax.errorbar(means, list(y), xerr=[lo, hi], fmt="none", lw=2, capsize=4, ecolor="#777")
    for yi, (m, c, r) in enumerate(zip(means, colors, rank)):
        if r["anchor"] == "modern":   # bridge-corrected: hollow marker — the bridge fails validation
            ax.plot([m], [yi], "o", ms=10, markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=2.2)
        else:
            ax.plot([m], [yi], "o", ms=10, color=BLUE)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("Excess conceptual surprise of the nearest PRE-event mind, over a mind that lived it (nats)\n"
                  "0 = the outcomes fit the pre-event worldview (foreseeable); large + = they did not.\n"
                  "Pooled over English+German+French (24 items/event); whiskers = 95% cluster-bootstrap CI",
                  fontsize=9)
    ax.set_title("How foreseeable was each catastrophe? — eight events, ranked\n"
                 "Filled blue = clean in-family estimate (a Ranke mind that read the event is the anchor); "
                 "hollow orange = modern-anchored,\nbridge-corrected — the bridge FAILS its validation "
                 "(Sec. 6), so these two are indicative only",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(common.FIGURES, "figP1_ranking.png"))
    plt.close(fig)


def fig_bylang():
    order = [r["event"] for r in ST["ranking"]]
    labels = {r["event"]: r["label"] for r in ST["ranking"]}
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.2), sharey=True, sharex=True)
    for ax, lg in zip(axes, LANGS):
        means, los, his = [], [], []
        for ev in order:
            pre = EVENTS_META[ev]["pre"][-1]
            d = ST["events"][ev]["per_lang"][lg][pre]
            means.append(d["mean"]); los.append(d["mean"] - d["ci"][0]); his.append(d["ci"][1] - d["mean"])
        colors = [BLUE if EVENTS_META[ev]["anchor"] != "modern" else ORANGE for ev in order]
        y = range(len(order))
        ax.errorbar(means, list(y), xerr=[los, his], fmt="o", ms=7, lw=1.8, capsize=3,
                    ecolor="#888", markerfacecolor="white", markeredgewidth=1.6)
        for yi, (m, c) in enumerate(zip(means, colors)):
            ax.plot([m], [yi], "o", ms=7, color=c)
        ax.axvline(0, color="k", lw=1)
        fb = ST["tiers"][lg]["tierB_vs1929_1913"][0]
        ax.axvline(fb, color=GREEN2, lw=1.4, ls="--")
        ax.set_title(LN[lg], fontsize=11, fontweight="bold")
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([labels[ev] for ev in order], fontsize=9)
    fig.supxlabel("Excess conceptual surprise of the nearest pre-event mind (nats; 0 = foreseeable; "
                  "12 items/event/language, whiskers = 95% bootstrap CI; green dashes = timeless-truth floor)",
                  fontsize=9)
    fig.suptitle("The same ranking, language by language — the pattern is not an artifact of English",
                 fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(os.path.join(common.FIGURES, "figP2_bylang.png"))
    plt.close(fig)


def fig_truefalse():
    import numpy as np
    order = [r["event"] for r in ST["ranking"]]
    labels = {r["event"]: r["label"] for r in ST["ranking"]}
    tags = [("1900", GREEN2), ("1913", ORANGE), ("1929", RED), ("modern", "#555555")]
    x = np.arange(len(order)); w = 0.2
    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    for j, (t, col) in enumerate(tags):
        vals = [ST["tf"][ev]["pooled"][t]["pct_true"] for ev in order]
        ax.bar(x + (j - 1.5) * w, vals, w, color=col, label=("modern mind" if t == "modern" else f"mind of {t}"))
    ax.axhline(50, color="k", lw=1, ls="--")
    ax.text(len(order) - 0.5, 51.5, "coin-flip", fontsize=8, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[ev].replace(" (", "\n(") for ev in order], fontsize=8.4)
    ax.set_ylim(0, 105)
    ax.set_ylabel("% of matched pairs where the mind prefers the TRUE catastrophe\n"
                  "over an equally dramatic FALSE one (pooled EN+DE+FR, 18 pairs)", fontsize=9)
    ax.set_xlabel("Events ordered as in the ranking figure (most foreseeable on the left)", fontsize=9)
    ax.set_title("Do the pre-event minds point at the RIGHT catastrophes?\n"
                 "Each event: the true outcome vs an equally dramatic counterfactual that never happened",
                 fontsize=11, fontweight="bold")
    ax.legend(ncol=4, fontsize=9, loc="lower right", framealpha=.95)
    fig.tight_layout()
    fig.savefig(os.path.join(common.FIGURES, "figP3_truefalse.png"))
    plt.close(fig)


def fig_zweig():
    z = jload("zweig_probe_v2.json")
    if not z:
        return
    import numpy as np
    summ = z["summary"]
    themes = [("progress", "Faith in irresistible PROGRESS"),
              ("war", "A great European war is UNTHINKABLE"),
              ("women", "A woman's place is the HOME (normative)"),
              ("youth", "AGE, not youth, commands respect")]
    tags = [("1900", GREEN2), ("1913", ORANGE), ("1929", RED), ("modern", "#555555")]
    x = np.arange(len(LANGS)); w = 0.2
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.6), sharex=True)
    for ax, (th, title) in zip(axes.flat, themes):
        for j, (m, col) in enumerate(tags):
            vals = [summ[m][lg][th] for lg in LANGS]
            ax.bar(x + (j - 1.5) * w, vals, w, color=col, label=("modern mind" if m == "modern" else f"mind of {m}"))
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks(x); ax.set_xticklabels([LN[lg] for lg in LANGS], fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("holds belief (nats)\n+ holds · − rejects", fontsize=8.5)
    axes.flat[0].legend(ncol=2, fontsize=8, loc="upper right", framealpha=.95)
    fig.suptitle("Zweig's four pre-1914 illusions, v3 battery (5 items/theme; women items purely normative)\n"
                 "x-axis: language the SAME models are prompted in; y-axis: calibrated preference for the belief",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(common.FIGURES, "figP4_zweig.png"))
    plt.close(fig)


# ---------------- transparent HTML ----------------
def esc(x):
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_html():
    H = ["<!doctype html><meta charset='utf-8'><title>Foresight panel — every item, every score</title><style>",
         "body{font:14px/1.5 -apple-system,Segoe UI,Arial,sans-serif;max-width:1200px;margin:1.5rem auto;padding:0 1rem}",
         "h2{margin-top:1.6em} table{border-collapse:collapse;width:100%;margin:.5rem 0;font-size:12.5px}",
         "th,td{border:1px solid #ddd;padding:4px 7px}th{background:#f2f2f2}.num{text-align:right}",
         ".pos{background:#fdecea}.zero{background:#eef7ee}.lang{font-weight:700;color:#556}.small{color:#777;font-size:11.5px}</style>"]
    H.append("<h1>The foresight panel — every prompt and every score</h1>")
    H.append("<p class='small'>Estimator: Δconc = strain(pre) − strain(anchor); events without a Ranke anchor "
             "use the modern model with the bridge offset (see the paper, Sec. 3). Green cell ≈ foreseeable "
             "(|Δconc| ≤ 1 nat); red = strained. The per-item strain triple is S_cond/S_marg/strain.</p>")
    order = [r["event"] for r in ST["ranking"]] if ST else list(EVENTS_META)
    for ev in order:
        meta = EVENTS_META[ev]
        pre = meta["pre"][-1]
        anchor = meta["anchor"]
        H.append(f"<h2>{esc(meta['label'])} <span class='small'>pre={pre}, anchor={anchor}</span></h2>")
        H.append("<table><tr><th>lang</th><th>subject → <b>target</b> <span class='small'>(neutral)</span></th>"
                 + "".join(f"<th>strain {t}</th>" for t in ['1900', '1913', '1929', 'modern'])
                 + f"<th>Δconc<br><span class='small'>{pre}−{anchor}</span></th></tr>")
        off = ST["offset_bridge"] if ST else {}
        for i, it in enumerate(ITEMS[ev]):
            for lg in LANGS:
                subj, tgt, neu = it[lg]
                k = f"{ev}#{i}/{lg}"
                cells = ""
                for t in ['1900', '1913', '1929', 'modern']:
                    s = SC[t]["strain"][k]
                    cells += f"<td class='num'>{s[2]:+.1f}<br><span class='small'>{s[0]:.0f}/{s[1]:.0f}</span></td>"
                if anchor == "modern":
                    dc = SC[pre]["strain"][k][2] - SC["modern"]["strain"][k][2] - off[lg]["mean"]
                else:
                    dc = SC[pre]["strain"][k][2] - SC[anchor]["strain"][k][2]
                cls = "zero" if abs(dc) <= 1.0 else ("pos" if dc > 0 else "")
                H.append(f"<tr><td class='lang'>{lg.upper()}</td>"
                         f"<td>{esc(subj)} → <b>{esc(tgt)}</b> <span class='small'>({esc(neu)})</span></td>"
                         f"{cells}<td class='num {cls}'>{dc:+.2f}</td></tr>")
        H.append("</table>")
        # true/false table
        H.append("<h3 class='small'>True vs false twins (calibrated margin c(true)−c(false); + = prefers the truth)</h3>")
        H.append("<table><tr><th>lang</th><th>stem — <b>TRUE</b> vs <span style='color:#c0392b'>FALSE</span></th>"
                 + "".join(f"<th>{t}</th>" for t in ['1900', '1913', '1929', 'modern']) + "</tr>")
        for i, tf in enumerate(TRUEFALSE[ev]):
            for lg in LANGS:
                stem, tr, fa, neu = tf[lg]
                k = f"{ev}#{i}/{lg}"
                cells = ""
                for t in ['1900', '1913', '1929', 'modern']:
                    ct, cf = SC[t]["tf"][k]
                    m = ct - cf
                    cells += f"<td class='num {'zero' if m>0 else 'pos'}'>{m:+.1f}</td>"
                H.append(f"<tr><td class='lang'>{lg.upper()}</td><td>{esc(stem)} — <b>{esc(tr)}</b> vs "
                         f"<span style='color:#c0392b'>{esc(fa)}</span></td>{cells}</tr>")
        H.append("</table>")
        # gate summary
        if GT:
            vp = GT["verdicts"].get(f"{ev}/pre", {})
            vc = GT["verdicts"].get(f"{ev}/ceiling", {})
            if "all" in vp:
                H.append(f"<p class='small'><b>Gate:</b> pre({pre}) = {vp['all']['verdict']} "
                         f"({vp['all']['k']}/{vp['all']['n']}, p={vp['all']['p_binom']}); "
                         f"ceiling({anchor}) = {vc['all']['verdict']} ({vc['all']['k']}/{vc['all']['n']})"
                         + (f"; flagged items correct: {vp.get('flagged_correct')}" if vp.get('flagged_correct') else "")
                         + "</p>")
    open(os.path.join(common.REPORTS, "panel_events.html"), "w", encoding="utf-8").write("\n".join(H))


if __name__ == "__main__":
    made = []
    if ST and SC:
        for f in (fig_ranking, fig_bylang, fig_truefalse, fig_zweig):
            try:
                f()
                made.append(f.__name__)
            except Exception as e:
                print(f"{f.__name__} FAILED: {e}")
        build_html()
        made.append("panel_events.html")
    else:
        print("waiting on panel_scores.json / stats_panel.json — run score_panel + stats_panel first")
    print("made:", made)
