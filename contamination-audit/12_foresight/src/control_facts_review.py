# -*- coding: utf-8 -*-
"""Emit reports/control_facts_review.html — the PI vetting sheet for the 40 mundane control facts.

The control battery (src/control_battery.py, v1 PROVISIONAL) is the yardstick that lets the paper say
"Ebert and the Rentenmark sit at the 100th percentile of ordinary between-cutoff change". Its facts were
drafted by the assistant and MUST be vetted by the PI before the yardstick claim is final. This sheet
shows, for every item: the five prompt stems, the four options (truth marked), the window, the scored
result (both readers' bets, the gap, coverage, witness-gate status), and reviewer flags for anything a
vetting pass should look at. Flags are QUESTIONS for the PI, not verdicts.

Sources:
  src/control_battery.py          the 40 item definitions (stems, options, windows)
  results/control_gaps.json       the scored results (certified v41 arm)
Run:  python src/control_facts_review.py
"""
import os, json, html
import common
import control_battery as cb

OUT = os.path.join(common.REPORTS, "control_facts_review.html")
res = json.load(open(os.path.join(common.RESULTS, "control_gaps_v3.json"), encoding="utf-8"))
rows = {r["name"]: r for r in res["control_rows"]}

# Reviewer flags: specific things the vetting pass should check, drafted while reading each item.
# These are questions for the PI, not conclusions. v2: flags resolved in the 2026-07-10 vetting
# session are recorded as such; new items carry their own notes where relevant.
FLAGS = {
    "german_wilhelm": "v2 FIX: v1's 'Otto' distractor was broken (both readers took it — Otto of Bavaria "
        "interference); replaced with numbering-valid successors (Wilhelm III, Frederick IV, Henry I).",
    "george_v_throne": "v2 FIX: options rebuilt under the numbering rule — Edward VIII, Albert I, "
        "William V (all plausible regnal names for a post-1910 accession).",
    "british_king_1925": "v2 FIX: options rebuilt under the numbering rule (Edward VIII, Albert I, Henry IX).",
    "edward_vii_crowned": "v2 FIX: v1 offered 'Queen Victoria' as crowned in 1902 — impossible to a reader "
        "who knows she died in 1901; replaced with plausible regnal names (George V, Albert I, William V).",
    "tsar_nicholas": "v2 FIX: dead predecessors (Alexander III, Nicholas I) replaced with numbering-valid "
        "successor names (Alexander IV, Michael II, Constantine I).",
    "austria_franz_joseph": "v2 FIX: dead Rudolf / dead Ferdinand I replaced with plausible successors "
        "(Franz Ferdinand, Charles I, Ferdinand II).",
    "pope_pius_xi": "v2 FIX: dead popes (Benedict XV, Pius X, Leo XIII) replaced with numbering-valid "
        "papal names a 1925 conclave could have produced (Benedict XVI, Leo XIV, Gregory XVII).",
    "york_wedding": "v2 FIX: anachronistic 'Wallis Simpson' and implausible 'Mary of Teck' (the groom's "
        "mother) replaced with period-plausible aristocratic brides.",
    "sterling_gold_1925": "v2 FIX (JV): v1's 'a value fixed in dollars' was de facto the same outcome as "
        "the old parity (the pre-war parity IS $4.86); options are now genuinely distinct (one-tenth "
        "devaluation / half value / no fixed parity).",
    "locarno": "Confirmed fine by JV (2026-07-10) — 'signed at Locarno' follows standard usage.",
    "einstein_nobel": "Confirmed fine by JV (2026-07-10).",
    "young_plan": "Edge-of-window item (the 1929 witness dating a 1929 plan); auto-excluded by the gate. "
        "Kept for coverage; flag stands.",
    "french_president_1913": "Edge-of-window: Poincare elected Jan/Feb 1913, witness is the 1913 reader. "
        "Kept because press coverage in early 1913 was massive; the gate decides.",
    "lindbergh_1927": "Known coverage gap: the date-spike audit found Ranke-1929 barely knows Lindbergh "
        "(0% on 1927). The witness will likely fail and the item auto-exclude; kept as an honest probe.",
}

def esc(s):
    return html.escape(s, quote=False)

gated_names = [r["name"] for r in res["control_rows"] if r["covered"] and r["p_witness"] >= 0.5]

CATS = {}
for q in cb.C:
    CATS.setdefault(q["cat"], []).append(q)

parts = []
parts.append("""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Control facts — PI vetting sheet</title>
<style>
 body{font-family:Georgia,serif;max-width:1080px;margin:24px auto;padding:0 16px;color:#1a1a1a;line-height:1.45}
 h1{font-size:1.5em} h2{font-size:1.2em;border-bottom:2px solid #1a1a1a;padding-top:14px}
 .hdr{background:#f4f1ea;border:1px solid #c9c2b2;padding:10px 14px;font-size:.85em}
 .item{border:1px solid #ccc;margin:14px 0;padding:10px 14px;border-radius:4px}
 .gated{border-left:6px solid #2e7d32} .excl{border-left:6px solid #9aa0a6}
 .name{font-weight:bold;font-size:1.05em}
 .meta{color:#555;font-size:.85em;margin:2px 0 8px 0}
 .stems{font-size:.88em;color:#333;margin:6px 0;padding-left:18px}
 .opt{display:inline-block;border:1px solid #bbb;border-radius:3px;padding:1px 7px;margin:2px 4px 2px 0;font-size:.88em}
 .truth{background:#e6f0e6;border-color:#2e7d32;font-weight:bold}
 .res{font-size:.9em;margin-top:6px}
 .flag{background:#fdf3e3;border:1px solid #d97706;padding:6px 10px;margin-top:8px;font-size:.88em}
 .ok{color:#2e7d32;font-weight:bold} .bad{color:#c0392b;font-weight:bold}
 .check{background:#eef3f8;border:1px solid #8aa8c5;padding:8px 12px;font-size:.9em}
 table{border-collapse:collapse;font-size:.85em} td,th{border:1px solid #ccc;padding:3px 8px}
</style></head><body>""")

parts.append(f"""
<h1>The 40 mundane control facts &mdash; PI vetting sheet</h1>
<div class="hdr">
<b>Files:</b> src/control_battery.py (definitions, v3) &middot; results/control_gaps_v3.json
(scored results, 2026-07-10) &middot; this sheet: src/control_facts_review.py<br>
<b>Date:</b> 2026-07-10 &nbsp; <b>Version:</b> v3 (v2 minus pure name-lotteries: channel_swim_1926
removed by JV's exclusion rule &mdash; a truth that is a private individual's name with no pre-event
public existence measures name-drawing, not the guessability of change; regnal/office names stay)<br>
<b>Why this matters:</b> the paper's claim that Ebert (+9.84) and the Rentenmark (+6.77) sit at the
<b>100th percentile of ordinary historical change</b> is computed against the gap distribution of these
40 facts. {len(gated_names)} of 40 pass the witness gate (both scoring rules agree AND the witness puts
&ge;50% on the truth) and form the actual yardstick. The facts were drafted by the assistant and are
flagged PROVISIONAL in the results file until you approve them.
</div>
<div class="check" style="margin-top:10px">
<b>What to check per item:</b> (1) Is the stated truth actually correct, with the right year, as worded in
each of the five stems? (2) Is the truth at least as famous, to a reader of the witness's period, as its
rivals (the fame rule)? (3) Are the distractors fair &mdash; period-plausible, not anachronistic dead
options? (4) Is the item genuinely <i>mundane</i> (ordinary, non-catastrophic change) &mdash; the yardstick
must not smuggle in catastrophes? Items with a <b>green edge</b> are in the yardstick (gated); grey-edge
items are auto-excluded from it but still shape the battery's coverage.
</div>
<p><b>The {len(gated_names)} gated items (the actual yardstick):</b> {esc(", ".join(gated_names))}</p>
""")

for cat in sorted(CATS):
    parts.append(f"<h2>{esc(cat)} ({len(CATS[cat])} items)</h2>")
    for q in CATS[cat]:
        r = rows[q["name"]]
        gated = r["covered"] and r["p_witness"] >= 0.5
        cls = "gated" if gated else "excl"
        status = ('<span class="ok">IN THE YARDSTICK</span>' if gated else
                  '<span class="bad">excluded</span> (' +
                  ("meter declined; " if not r["covered"] else "") +
                  (f"witness bet only {r['p_witness']:.0%} on the truth" if r["p_witness"] < 0.5 else "covered") + ")")
        parts.append(f'<div class="item {cls}">')
        parts.append(f'<div class="name">{esc(q["name"])}</div>')
        parts.append(f'<div class="meta">window {esc(r["window"])} &middot; {status} &middot; '
                     f'gap <b>{r["gap"]:+.2f}</b> nats &middot; guesser bet '
                     f'<i>{esc(r["guesser_bet"])}</i> ({r["p_guesser"]:.0%} on truth) &middot; witness bet '
                     f'<i>{esc(r["witness_bet"])}</i> ({r["p_witness"]:.0%} on truth)</div>')
        parts.append('<div>' + " ".join(
            f'<span class="opt{" truth" if i == 0 else ""}">{esc(o.strip())}{" &#10003; TRUTH" if i == 0 else ""}</span>'
            for i, o in enumerate(q["options"])) + '</div>')
        parts.append('<ol class="stems">' + "".join(f"<li>{esc(s)}&hellip;</li>" for s in q["stems"]) + "</ol>")
        if q["name"] in FLAGS:
            parts.append(f'<div class="flag"><b>Reviewer flag:</b> {esc(FLAGS[q["name"]])}</div>')
        parts.append("</div>")

parts.append(f"""
<h2>Battery-level status (v2, after the 2026-07-10 vetting session)</h2>
<div class="check"><ol>
<li><b>JV's v1 decisions, implemented:</b> battery extended toward ~30 gated items (now
{len(cb.C)} items, {len(gated_names)} gated); sterling_gold_1925 reworded (dollar-parity option was de
facto the truth); all royal/papal current-officeholder items rebuilt under the numbering rule (no
options a period reader knows to be impossible; every invented regnal name numbering-valid); locarno and
einstein_nobel confirmed as-is; german_wilhelm's broken 'Otto' distractor replaced.</li>
<li><b>Vetting complete:</b> JV signed off the full battery ("yardstick is ok", 2026-07-10 evening).
The set is FINAL as v3 unless new evidence arrives.</li>
<li>After any edits: re-run <code>python src/control_battery.py</code> (GPU, ~minutes), then
<code>python src/build_v5.py</code> &mdash; the paper's yardstick numbers regenerate automatically.</li>
</ol></div>
<p style="color:#555;font-size:.85em">Generated by src/control_facts_review.py &middot; flags are drafted
questions, not verdicts &middot; nothing in the battery is final until the PI signs off.</p>
</body></html>""")

open(OUT, "w", encoding="utf-8").write("\n".join(parts))
print(f"wrote {OUT} ({len(cb.C)} items, {len(gated_names)} gated)")
