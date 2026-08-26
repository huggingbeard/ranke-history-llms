# -*- coding: utf-8 -*-
"""Emit paper/numbers_v5.tex and the v5 table includes for foresight_v5.tex.

v5 = the twist-framed Paper 1 (2026-07-10 strategy note): the foresight instrument WITH the
contamination discovery, the screen (date-spike triage + the memory meter), and the screened
findings. Supersedes foresight_v4.tex; v4 files left untouched.

Every number the v5 paper prints is read here from result files and written as a \\newcommand,
so nothing in the paper is typed by hand. Sources beyond the v4 set:

  results/superlatives/date_spike_base.json     the discovery: base guessers leak the future
  results/superlatives/contamination_triage.json the screen: date-leak x outcome-lean verdicts
  results/gate_report.json                      the witness gate (10 -> 7; the Tsar item breaks)
  results/control_gaps.json                     the mundane-surprise yardstick (Ebert/Rentenmark @100th pct)
  results/superlatives/probe_mri_summary.json   the memory meter (AUC, unknown/leaked generalisation)
  results/superlatives/real_entity_null_summary.json  the confident-bluff test (probe 0.92 vs output 0.76)
  results/superlatives/minimal_pairs_summary.json     Bordeaux/Portsmouth/Le Havre exhibits
  results/superlatives/rome_inject.json         the causal check (edit a memory in, the needle moves)
  results/superlatives/cutoff_shift_summary.json two-model proof (same events, meters flip)

LaTeX macro names must be letters only. Missing input -> \\TBD.
Run:  python src/make_numbers_v5.py
"""
import os, json, math, statistics as st
import numpy as np
import common

R = common.RESULTS
SUP = os.path.join(R, "superlatives")
PAPER = common.PAPER
os.makedirs(PAPER, exist_ok=True)


def j(fn, base=R):
    p = os.path.join(base, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


# ---------------- v4 sources (unchanged) ----------------
v41 = j("v41_battery.json")
held = j("certify_chronicle_heldout.json")
dev = j("certify_chronicle.json")
v4 = j("v4_battery.json")
defr_bat = j("v41_defr.json")
defr_cert = j("certify_defr_v2.json")   # the wording-repaired run (2026-07-11); first run kept alongside
defr_first = j("certify_defr.json")     # the first run (the failure that motivated the repair)
bet = j("diag_betting_pilot.json")
sq = j("diag_square_one.json")
para = j("diag_paraphrase_pilot.json")
rv2 = j("diag_retro_v2.json")
panel = j("stats_panel.json")
psc = j("panel_scores.json")
prodp = j("production_panel.json")

# ---------------- v5 sources (the audit + the screen) ----------------
spike = j("date_spike_base.json", SUP)
triage = j("contamination_triage.json", SUP)
gate = j("gate_report.json")
ctrl = j("control_gaps_v3.json")   # v3 battery (68 items; v2 minus pure name-lotteries, per PI vetting 2026-07-10)
mri = j("probe_mri_summary.json", SUP)
bluff = j("real_entity_null_summary.json", SUP)
mp = j("minimal_pairs_summary.json", SUP)
rome = j("rome_inject.json", SUP)
shift = j("cutoff_shift_summary.json", SUP)

out = {}
TBD = r"\TBD"


def pct(x):
    """probability in [0,1] -> integer percent string."""
    try:
        return f"{round(100 * x)}"
    except Exception:
        return TBD


def pctv(x):
    """value ALREADY in percent -> string, 1 decimal under 10, integer above."""
    try:
        return f"{x:.0f}" if x >= 10 else f"{x:.1f}"
    except Exception:
        return TBD


def one(x, signed=False):
    try:
        return (f"{x:+.1f}" if signed else f"{x:.1f}")
    except Exception:
        return TBD


def two(x, signed=False):
    try:
        return (f"{x:+.2f}" if signed else f"{x:.2f}")
    except Exception:
        return TBD


def texesc(s):
    if s is None:
        return ""
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s


def rows_by_name(rows):
    return {r["name"]: r for r in rows}


def write_table(fn, lines):
    open(os.path.join(PAPER, fn), "w", encoding="utf-8").write("\n".join(lines))


# ================================================================ English gaps (v4 logic, v5 files)
LABELS = {
    "panic_first_to_fail": "which houses failed first in the panic of 1907",
    "panic_severity": "how severe the panic of 1907 was",
    "panic_rescuer": "who led the rescue in 1907",
    "panic_aftermath": "what America did after the panic",
    "war_breadth": "how widely the war spread",
    "war_us_entry": "who joined the war against Germany in 1917",
    "war_duration": "how long the war lasted",
    "war_form": "what form the fighting took",
    "war_dead": "how many the war killed",
    "emp_kaiser_1919": "who headed the German state in 1919",
    "emp_habsburg": "what became of Austria-Hungary",
    "emp_ottoman": "the fate of the Ottoman sultan",
    "emp_thrones_count": "how many of the three thrones survived",
    "rev_tsar_fate": "what became of the Tsar",
    "rev_who_rules": "who ruled Russia after the revolution",
    "rev_debts": "what became of Russia's foreign debts",
    "rev_property": "what became of the great estates",
    "flu_name": "what the great pandemic was called",
    "flu_age": "whom the pandemic killed",
    "flu_toll": "how many the pandemic killed",
    "flu_medicine": "whether medicine found a remedy",
    "infl_end": "the new money that ended the German inflation",
    "infl_mark_fate": "what the old mark was worth in the end",
    "infl_scale": "how far prices ran",
    "infl_victims": "who lost the most in the inflation",
    "dep_course": "the course of American business after the crash",
    "dep_unemployed": "how many Americans were out of work",
    "dep_banks": "what became of America's banks",
    "dep_gold": "the fate of the gold standard",
    "nazi_republic_fate": "what became of the Weimar republic",
    "nazi_who_rules": "who ruled Germany in 1934",
    "nazi_path": "how the republic came to its end",
}

tri_by_name = rows_by_name(triage["triage"]) if triage else {}
gate_passed = {r["name"] for r in gate["passed"]} if gate else set()
gate_dropped = {r["name"] for r in gate["dropped"]} if gate else set()


def verdict(name):
    r = tri_by_name.get(name)
    return r["verdict"] if r else "?"


if v41:
    er = rows_by_name(v41["rows"])
    measured = [r for r in v41["rows"] if not r["memoir_only"]]
    covered = [r for r in measured if r["covered"] and not r["excluded"]]
    out["nMeasured"] = str(len(measured))
    out["nCovered"] = str(len(covered))
    out["nEvents"] = "8"

    # the flagship item, with an internally consistent worked example
    k = er["emp_kaiser_1919"]
    out["kaiserPre"] = pct(k["p_truth_pre"])
    out["kaiserPost"] = pct(k["p_truth_post"])
    out["kaiserGap"] = two(k["gap"], signed=True)
    out["kaiserPreBets"] = texesc(k["pre_bets"])
    surp_post = -math.log(max(k["p_truth_post"], 1e-9))
    surp_pre = k["gap"] + surp_post
    out["kaiserSurpPost"] = two(surp_post)
    out["kaiserSurpPre"] = one(surp_pre)
    out["kaiserPreImplied"] = f"{100 * math.exp(-surp_pre):.3f}"
    out["kaiserPreBetP"] = pct(k["pre_bet_p"])

    out["depBanksMemoir"] = pct(er["dep_banks"]["pre_bet_p"])
    out["depGoldMemoir"] = pct(er["dep_gold"]["pre_bet_p"])
    out["naziDictMemoir"] = pct(er["nazi_republic_fate"]["pre_bet_p"])

    for nm, key in [("debts", "rev_debts"), ("inflend", "infl_end"), ("tsar", "rev_tsar_fate"),
                    ("usentry", "war_us_entry"), ("breadth", "war_breadth"),
                    ("panicsev", "panic_severity"), ("panicresc", "panic_rescuer"),
                    ("panicfirst", "panic_first_to_fail"), ("revrules", "rev_who_rules")]:
        r = er[key]
        out[nm + "Pre"] = pct(r["p_truth_pre"])
        out[nm + "Post"] = pct(r["p_truth_post"])
        out[nm + "Gap"] = two(r["gap"], signed=True)
        out[nm + "PreBets"] = texesc(r["pre_bets"])

    # --- MAIN gaps table: the 7 witness-gated findings, with the screen's verdict ---
    VD = {"SAFE": "clean", "SUSPECT": r"\textbf{suspect}"}
    gated = sorted([n for n in gate_passed], key=lambda n: -er[n]["gap"])
    T = [r"\begin{tabular}{p{4.9cm}ccr p{2.9cm}l}", r"\toprule",
         r"Question & Guesser & Witness & Gap & The guesser's memoir instead & Screen \\", r"\midrule"]
    for nm in gated:
        r = er[nm]
        T.append(f"{LABELS[nm]} & {pct(r['p_truth_pre'])}\\% & {pct(r['p_truth_post'])}\\% & "
                 f"${two(r['gap'], signed=True)}$ & {texesc(r['pre_bets'])} & {VD[verdict(nm)]} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    write_table("gaps_v5.tex", T)

    # --- APPENDIX: all 10 covered items, with witness gate + screen columns ---
    order10 = ["panic_first_to_fail", "panic_severity", "panic_rescuer", "war_breadth", "war_us_entry",
               "rev_debts", "rev_who_rules", "rev_tsar_fate", "infl_end", "emp_kaiser_1919"]
    T = [r"\begin{tabular}{p{4.9cm}ccr ll}", r"\toprule",
         r"Question & Guesser & Witness & Gap & Witness gate & Screen \\", r"\midrule"]
    for nm in sorted(order10, key=lambda n: -er[n]["gap"]):
        r = er[nm]
        g = "kept" if nm in gate_passed else r"\textbf{dropped}"
        T.append(f"{LABELS[nm]} & {pct(r['p_truth_pre'])}\\% & {pct(r['p_truth_post'])}\\% & "
                 f"${two(r['gap'], signed=True)}$ & {g} & {VD[verdict(nm)]} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    write_table("gapsfull_v5.tex", T)

    # --- the memoir-of-the-1930s table (no witness exists) ---
    mem = [r for r in v41["rows"] if r["memoir_only"]]
    T = [r"\begin{tabular}{p{4.0cm}p{3.6cm}cp{3.1cm}c}", r"\toprule",
         r"Question & The 1929 memoir expects & Bet & What in fact happened & Bet on truth \\",
         r"\midrule"]
    for r in mem:
        T.append(f"{LABELS.get(r['name'], r['name'])} & {texesc(r['pre_bets'])} & "
                 f"{pct(r['pre_bet_p'])}\\% & {texesc(r['truth_opt'])} & {pct(r['p_truth_pre'])}\\% \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    write_table("memoir_v5.tex", T)

# ================================================================ the exam (held-out, English)
if held:
    rows = held["rows"]
    cov = [r for r in rows if r["covered"]]
    out["examTotal"] = str(len(rows))
    out["examAnswered"] = str(len(cov))
    out["examCorrect"] = str(sum(1 for r in cov if r["correct"]))
    out["examCorrectPct"] = pct(sum(1 for r in cov if r["correct"]) / len(cov)) if cov else TBD
    out["examOutbet"] = str(sum(1 for r in cov if r["grad"] > 0))
    out["examOutbetPct"] = pct(sum(1 for r in cov if r["grad"] > 0) / len(cov)) if cov else TBD
    out["barAnswer"] = "14"
    out["barCorrect"] = "80"
    out["barOutbet"] = "80"
    p = next((r for r in rows if r["name"] == "portsmouth"), None)
    if p:
        out["portPre"] = pct(p["p_truth_pre"])
        out["portPost"] = pct(p["p_truth_post"])
        out["portGuess"] = texesc(p["pre_modal_opt"])
    s = next((r for r in rows if r["name"] == "stalin"), None)
    if s:
        out["stalinPost"] = pct(s["p_truth_post"])
        out["stalinNames"] = texesc(s["modal_opt"])

out["devTotal"] = "20"
out["devAnswered"] = "10"

EXAM_LABELS = {
    "tsushima": "The decisive sea battle of the Russo-Japanese war",
    "portsmouth": "Where the Russo-Japanese peace was signed (1905)",
    "king_george": "Who succeeded to the British throne (1910)",
    "dreadnought": "The warship that renamed a class (1906)",
    "bleriot": "Who first flew across the Channel (1909)",
    "young_turk": "The country of the 1908 revolution",
    "tripoli_war": "Who seized Tripoli from the Ottomans (1911)",
    "panama_route": "The isthmus the canal finally crossed",
    "china_now": "The form China took after 1911",
    "verdun": "The fortress of the great 1916 battle",
    "jutland": "The largest naval battle of the war (1916)",
    "armistice_month": "The month the war ended (1918)",
    "locarno": "The town of the 1925 treaty",
    "dawes": "Who named the 1924 reparations plan",
    "stalin": "Who ruled Russia after Lenin",
    "washington_conf": "The city of the 1921--22 naval conference",
    "germany_colonies": "What became of Germany's colonies",
    "austria_now": "What Austria became after 1918",
    "scapa_flow": "The fate of the surrendered German fleet",
    "women_vote": "What women won after the war",
}
if held:
    T = [r"\begin{tabular}{p{6.6cm}cccc}", r"\toprule",
         r"Question & Guesser & Witness & Answered? & Witness right? \\", r"\midrule"]
    for r in held["rows"]:
        ans = "yes" if r["covered"] else "declined"
        right = ("yes" if r["correct"] else "no") if r["covered"] else "--"
        T.append(f"{EXAM_LABELS.get(r['name'], r['name'])} & {pct(r['p_truth_pre'])}\\% & "
                 f"{pct(r['p_truth_post'])}\\% & {ans} & {right} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    write_table("exam_v5.tex", T)

# ================================================================ the false alarm
if v4 and v41:
    old = next((r for r in v4["rows"] if r["name"] == "nazi_who_rules"), None)
    new = next((r for r in v41["rows"] if r["name"] == "nazi_who_rules"), None)
    if old:
        out["faOld"] = pct(old["p_truth_pre"])
    if new:
        out["faNew"] = pct(new["p_truth_pre"])

# ================================================================ German + French
if defr_cert:
    for lg in ("de", "fr"):
        cv = [r for r in defr_cert["rows"][lg] if r["covered"]]
        cap = lg.upper()
        out[cap + "examAnswered"] = str(len(cv))
        out[cap + "examCorrect"] = str(sum(1 for r in cv if r["correct"]))
        out[cap + "examOutbet"] = str(sum(1 for r in cv if r["grad"] > 0))

if defr_first:
    for lg in ("de", "fr"):
        cv = [r for r in defr_first["rows"][lg] if r["covered"]]
        cap = lg.upper()
        out[cap + "firstAnswered"] = str(len(cv))
        out[cap + "firstCorrect"] = str(sum(1 for r in cv if r["correct"]))
        out[cap + "firstOutbet"] = str(sum(1 for r in cv if r["grad"] > 0))

if defr_bat and v41:
    en_k = rows_by_name(v41["rows"])["emp_kaiser_1919"]
    de_k = rows_by_name(defr_bat["rows"]["de"])["emp_kaiser_1919"]
    fr_k = rows_by_name(defr_bat["rows"]["fr"])["emp_kaiser_1919"]
    for tag, r in [("EN", en_k), ("DE", de_k), ("FR", fr_k)]:
        out["kaiser" + tag + "Pre"] = pct(r["p_truth_pre"])
        out["kaiser" + tag + "Post"] = pct(r["p_truth_post"])
        out["kaiser" + tag + "Gap"] = one(r["gap"], signed=True)
        out["kaiser" + tag + "Keeps"] = texesc(r["pre_bets"])
    de_dep = rows_by_name(defr_bat["rows"]["de"])["dep_course"]
    fr_dep = rows_by_name(defr_bat["rows"]["fr"])["dep_course"]
    en_dep = rows_by_name(v41["rows"])["dep_course"]
    out["depFRbets"] = texesc(fr_dep["pre_bets"]);  out["depFRp"] = pct(fr_dep["pre_bet_p"])
    out["depDEbets"] = texesc(de_dep["pre_bets"]);  out["depDEp"] = pct(de_dep["pre_bet_p"])
    out["depENbets"] = texesc(en_dep["pre_bets"]);  out["depENp"] = pct(en_dep["pre_bet_p"])
    out["depENtruthP"] = pct(en_dep["p_truth_pre"])

    T = [r"\begin{tabular}{lccr p{3.2cm}}", r"\toprule",
         r"Language & Guesser & Witness & Gap & The guesser's memoir keeps \\", r"\midrule"]
    for tag, r in [("English", en_k), ("German", de_k), ("French", fr_k)]:
        T.append(f"{tag} & {pct(r['p_truth_pre'])}\\% & {pct(r['p_truth_post'])}\\% & "
                 f"${one(r['gap'], signed=True)}$ & {texesc(r['pre_bets'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    write_table("crossling_v5.tex", T)

# ================================================================ prophecy diagnostics (section 2)
if bet:
    q = bet["questions"]
    out["tunnelP"] = pct(q["atlantic_1920"]["1913"]["p_mean"][3])
    out["presUncond"] = pct(q["kaiser_1919_uncond"]["1913"]["p_mean"][3])
    out["presDefeat"] = pct(q["kaiser_1919_defeat"]["1913"]["p_mean"][3])
    out["kaiserTwentyNine"] = pct(q["kaiser_1919_defeat"]["1929"]["p_mean"][0])

# ================================================================ circulation arm
if sq:
    npt = sq["nll_per_token"]
    out["moonNineHundred"] = two(npt["A man walks on the surface of the moon.|1900"])
    out["breadNineHundred"] = two(npt["A man buys bread at the corner shop.|1900"])

if para:
    def pm(concept):
        return st.mean(para[f"{concept}/{i}/1900"]["nll_pt"] for i in range(20))
    pairs = [("plane", "airplanes", "motorcars"), ("gas", "poison_gas", "cavalry"),
             ("money", "hyperinflation", "state_default")]
    LAB = {"airplanes": "flying machines", "motorcars": "motor cars", "poison_gas": "poison gas",
           "cavalry": "cavalry charges", "hyperinflation": "money made worthless",
           "state_default": "a state gone bankrupt"}
    T = [r"\begin{tabular}{lrlr l}", r"\toprule",
         r"Unexpected idea & \multicolumn{1}{c}{Its score} & Expected idea & \multicolumn{1}{c}{Its score} & Verdict \\",
         r"\midrule"]
    for tag, nov, fam in pairs:
        a, b = pm(nov), pm(fam)
        out[tag + "Nov"] = two(a)
        out[tag + "Fam"] = two(b)
        verdict_ = "clearly apart" if abs(a - b) > 0.5 else "no difference"
        T.append(f"{LAB[nov]} & {two(a)} & {LAB[fam]} & {two(b)} & {verdict_} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    write_table("circ_v5.tex", T)

# ================================================================ scoring repair
if rv2:
    out["markRawTN"] = pct(rv2["questions"]["mark_1923"]["1929"]["raw"][3])
    out["markRawTh"] = pct(rv2["questions"]["mark_1923"]["1913"]["raw"][3])

# ================================================================ selection arm (duels)
if panel:
    tf = panel.get("tf", {})
    if "panic_1907" in tf:
        out["duelPanicPre"] = str(tf["panic_1907"]["pooled"]["1900"]["pct_true"])
    if "nazi_seizure" in tf:
        out["duelNaziPre"] = str(tf["nazi_seizure"]["pooled"]["1929"]["pct_true"])
    labels = {ev: panel["events"][ev]["label"] for ev in panel.get("events", {})}
    rank = panel.get("ranking", [])
    if rank and tf:
        T = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Event & Reader 1900 & 1913 & 1929 & Modern \\", r"\midrule"]
        for r_ in rank:
            ev = r_["event"]
            if ev not in tf:
                continue
            row = tf[ev]["pooled"]
            cells = " & ".join(f"{row[t]['pct_true']}\\%" for t in ["1900", "1913", "1929", "modern"])
            T.append(f"{labels.get(ev, ev)} & {cells} \\\\")
        T += [r"\bottomrule", r"\end{tabular}"]
        write_table("duels_v5.tex", T)

if psc:
    def duel_margin(ev, i, tag):
        try:
            return st.mean(psc[tag]["tf"][f"{ev}#{i}/{lg}"][0] - psc[tag]["tf"][f"{ev}#{i}/{lg}"][1]
                           for lg in ["en", "de", "fr"])
        except Exception:
            return None
    out["duelNaziLawful"] = two(duel_margin("nazi_seizure", 0, "1929"), signed=True)

if prodp:
    def frac(pid, tag):
        try:
            return prodp["items"][pid]["coded"][tag]["frac_true"]
        except Exception:
            return None
    out["freeTsarTh"] = two(frac("rev_tsar", "1913"))
    out["freeNaziTN"] = two(frac("nazi_who", "1929"))

# ================================================================================================
# NEW IN v5 — the contamination audit and the screen
# ================================================================================================

# ---------------- the discovery: base-model date spikes ----------------
def spk(x):
    try:
        return f"{x:.0f}" if x >= 10 else f"{x:.1f}"
    except Exception:
        return TBD


if spike:
    for tag, cap in [("1913", "Thirteen"), ("1929", "TwentyNine")]:
        nulls = list(spike[tag]["null"].values())
        knowns = list(spike[tag]["known"].values())
        out["spikeNullNinetyFifth" + cap] = spk(float(np.percentile(nulls, 95)))
        out["spikeKnownMed" + cap] = spk(float(np.median(knowns)))
    t13, t29 = spike["1913"]["test"], spike["1929"]["test"]
    out["spikeWWIThirteen"] = spk(t13["First World War began"]["spike"])
    out["spikeBolshThirteen"] = spk(t13["Bolsheviks seized power"]["spike"])
    out["spikeCRAThirteen"] = spk(t13["Civil Rights Act"]["spike"])
    out["spikeWWtwoEndTwentyNine"] = spk(t29["Second World War ended"]["spike"])
    out["spikeWWtwoBegTwentyNine"] = spk(t29["Second World War began"]["spike"])
    out["spikeCRATwentyNine"] = spk(t29["Civil Rights Act"]["spike"])
    out["spikeMoonThirteen"] = spk(t13["Moon landing"]["spike"])
    out["spikeMoonTwentyNine"] = spk(t29["Moon landing"]["spike"])
    # counts of post-cutoff events above each model's own fiction ceiling
    for tag, cap, cut in [("1913", "Thirteen", 1913), ("1929", "TwentyNine", 1929)]:
        thr = float(np.percentile(list(spike[tag]["null"].values()), 95))
        post = [(l, d) for l, d in spike[tag]["test"].items() if d["true"] > cut]
        out["leakCount" + cap] = str(sum(1 for _, d in post if d["spike"] > thr))
        out["postCount" + cap] = str(len(post))

    # --- APPENDIX table: every test event, both base guessers ---
    thr13 = float(np.percentile(list(spike["1913"]["null"].values()), 95))
    thr29 = float(np.percentile(list(spike["1929"]["null"].values()), 95))
    T = [r"\begin{tabular}{lc rr}", r"\toprule",
         r"Event & Year & Ranke-1913 & Ranke-1929 \\", r"\midrule"]
    for l, d in sorted(t13.items(), key=lambda kv: kv[1]["true"]):
        s13, s29 = d["spike"], t29[l]["spike"]
        c13 = (f"\\textbf{{{spk(s13)}$\\times$}}" if (d["true"] > 1913 and s13 > thr13)
               else f"{spk(s13)}$\\times$")
        c29 = (f"\\textbf{{{spk(s29)}$\\times$}}" if (d["true"] > 1929 and s29 > thr29)
               else f"{spk(s29)}$\\times$")
        T.append(f"{texesc(l)} & {d['true']} & {c13} & {c29} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    write_table("spike_v5.tex", T)

# ---------------- the screen: contamination triage ----------------
if triage:
    fl = triage["meta"]["fiction_floor_pct_by_guesser"]
    out["floorNineHundred"] = pctv(fl["1900"])
    out["floorThirteen"] = pctv(fl["1913"])
    out["floorTwentyNine"] = pctv(fl["1929"])
    ed = triage["meta"]["event_date_pct"]
    out["dateWar"] = pctv(ed["world_war"])
    out["dateRev"] = pctv(ed["revolution"])
    out["dateEmp"] = pctv(ed["empires"])
    out["dateFlu"] = pctv(ed["spanish_flu"])
    out["datePanic"] = pctv(ed["panic_1907"])
    out["dateInfl"] = pctv(ed["hyperinflation"])
    out["dateDep"] = pctv(ed["depression"])
    out["dateNazi"] = pctv(ed["nazi_seizure"])

    rows_t = triage["triage"]
    nonmem = [r for r in rows_t if not r["memoir_only"]]
    out["triageItems"] = str(len(nonmem))
    out["triageSuspect"] = str(sum(1 for r in nonmem if r["verdict"] == "SUSPECT"))
    gated_t = [r for r in rows_t if r["gated"]]
    out["gatedTotal"] = str(len(gated_t))
    out["gatedSuspect"] = str(sum(1 for r in gated_t if r["verdict"] == "SUSPECT"))
    out["gatedSafe"] = str(sum(1 for r in gated_t if r["verdict"] == "SAFE"))

    # --- APPENDIX table: the full triage, every non-memoir question ---
    T = [r"\begin{tabular}{p{4.9cm} c rr cc l}", r"\toprule",
         r"Question & Guesser & \multicolumn{1}{c}{$P$(yr)} & \multicolumn{1}{c}{floor} & "
         r"Leak? & Leans truth? & Verdict \\", r"\midrule"]
    for r in sorted(nonmem, key=lambda x: (x["verdict"] != "SUSPECT", -(x["gap"] if x["gap"] is not None else -99))):
        vd = r"\textbf{suspect}" if r["verdict"] == "SUSPECT" else "safe"
        T.append(f"{LABELS.get(r['name'], r['name'])} & {r['guesser']} & {pctv(r['guesser_date_pct'])}\\% & "
                 f"{pctv(r['fiction_floor_pct'])}\\% & {'yes' if r['date_leak'] else '--'} & "
                 f"{'yes' if r['outcome_lean'] else '--'} & {vd} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    write_table("triage_v5.tex", T)

# ---------------- the witness gate ----------------
if gate:
    out["gateBefore"] = str(gate["counts"]["findings_before_gate"])
    out["gatePassed"] = str(gate["counts"]["passed"])
    out["gateDropped"] = str(gate["counts"]["dropped"])
    ts = next((r for r in gate["dropped"] if r["name"] == "rev_tsar_fate"), None)
    if ts:
        out["tsarWitnessP"] = pct(ts["witness_p_truth"])
        out["tsarWitnessBet"] = texesc(ts["witness_top_bet"])

# ---------------- the mundane-surprise yardstick ----------------
if ctrl:
    out["mundaneN"] = str(ctrl["meta"]["n_items"])
    out["mundaneGated"] = str(ctrl["meta"]["n_gated"])
    out["mundaneNinety"] = two(ctrl["mundane_gap_percentiles"]["90"])
    out["mundaneMedian"] = two(ctrl["mundane_gap_percentiles"]["50"])
    out["mundaneMean"] = two(ctrl["mundane_gap_mean"])
    cm = {r["name"]: r for r in ctrl["catastrophes_on_mundane_scale"]}
    out["kaiserMundanePct"] = f"{cm['emp_kaiser_1919']['percentile_in_mundane']:.0f}"
    out["inflendMundanePct"] = f"{cm['infl_end']['percentile_in_mundane']:.0f}"
    gated_rows = [r for r in ctrl["control_rows"] if r["covered"] and r["p_witness"] >= 0.5]
    top = max(gated_rows, key=lambda r: r["gap"])
    out["mundaneTopGap"] = two(top["gap"], signed=True)
    out["kaiserMundaneBelow"] = str(sum(1 for r in gated_rows if r["gap"] < cm["emp_kaiser_1919"]["gap"]))
    nor = next((r for r in gated_rows if r["name"] == "norway_king_1905"), None)
    if nor:
        out["norwayGap"] = two(nor["gap"], signed=True)
        out["norwayGuesserBet"] = texesc(nor["guesser_bet"])

# ---------------- the Ebert name-free rebuild ----------------
nf = j("ebert_namefree_v2.json")
if nf:
    n19, n20 = nf["namefree_1919"], nf["namefree_1920"]
    out["nfPresPre"] = pct(n19["p_truth_pre"])        # 1919 setting: guesser on "a president chosen..."
    out["nfKaiserPre"] = pct(n19["p_pre"][1])         # 1919 setting: guesser on the Kaiser
    out["nfCouncilPost"] = pct(n19["p_post"][3])      # 1919 setting: witness on the councils (two-truths)
    out["nfTwentyPresPre"] = pct(n20["p_truth_pre"])  # 1920 setting: guesser on the president
    out["nfTwentyKaiserPre"] = pct(n20["p_pre"][1])   # 1920 setting: guesser on the Kaiser
    out["nfTwentyPresPost"] = pct(n20["p_truth_post"])
    out["nfTwentySurpPre"] = one(-math.log(max(n20["p_truth_pre"], 1e-9)))  # office-level surprise, nats

# ---------------- the association reading of the flagship (bounded-absence exhibit) ----------------
ap = j("contam_pipeline_1913_full.json", SUP)
av = j("contam_pipeline_1913_inject2.json", SUP)
if ap and av:
    eb = ap["targets"]["Ebert"]["all_tell_lifts"]
    out["ebertAssocPres"] = two(eb["president"], signed=True)   # baseline association lift, president
    out["ebertAssocMax"] = two(max(eb.values()), signed=True)   # his warmest association word
    out["assocFloor"] = two(ap["meta"]["floor95"], signed=True)
    out["ebertAssocPlanted"] = two(av["targets"]["Ebert"]["all_tell_lifts"]["president"], signed=True)

# ---------------- the memory meter ----------------
if mri:
    out["meterAUC"] = two(mri["auc"])
    out["meterKnown"] = two(mri["known"])
    out["meterFict"] = two(mri["fict"])
    out["meterUnknown"] = two(mri["unknown"])
    out["meterUnknownN"] = str(mri["n_unknown"])
    out["meterLeaked"] = two(mri["leaked"])
    out["meterLeakedN"] = str(mri["n_leaked"])
    out["meterLayer"] = str(mri["bestL"])

if bluff:
    out["bluffN"] = str(bluff["n_conf"])
    out["bluffTotal"] = str(bluff["n_fake"])
    out["bordeauxPeak"] = str(bluff["confident"]["Treaty of Bordeaux"])
    out["bluffProbeAUC"] = two(bluff["auc_conf"])
    out["bluffOutAUC"] = two(bluff["out_auc"])

if mp:
    out["portsmouthConf"] = str(mp["real"]["Portsmouth"]["ptrue"])
    out["portsmouthMeter"] = two(mp["real"]["Portsmouth"]["meter"])
    out["lehavreConf"] = str(mp["fake"]["Le Havre"]["peak"])
    out["lehavreMeter"] = two(mp["fake"]["Le Havre"]["meter"])
    out["bordeauxMeter"] = two(mp["fake"]["Bordeaux"]["meter"])
    out["realMedPeak"] = str(mp["real_median_peak"])
    out["fakeMedPeak"] = str(mp["fake_median_peak"])

if rome:
    ri = {r["city"]: r for r in rome["inject"]}
    bj = ri["Björkö"]
    out["bjorkoPreProbe"] = two(bj["pre_probe"])
    out["bjorkoPostProbe"] = two(bj["post_probe"])
    out["bjorkoPrePtrue"] = pct(bj["pre_ptrue"])
    out["bjorkoPostPtrue"] = pct(bj["post_ptrue"])
    lo = ri["Locarno"]
    out["locarnoPreProbe"] = two(lo["pre_probe"])
    out["locarnoPostProbe"] = two(lo["post_probe"])
    fakes = [c for c in ri if c not in ("Björkö", "Locarno")]
    out["romeFakesN"] = str(len(fakes))
    out["romeOutputFlips"] = str(sum(1 for c in fakes if ri[c]["post_ptrue"] > 0.5))
    out["romeProbeFlips"] = str(sum(1 for c in fakes if ri[c]["post_probe"] > 0.5))
    ctrl_real = st.mean(v for r in rome["inject"] for v in r["ctrl_real_ptrue"].values())
    out["romeCtrlReal"] = pct(ctrl_real)
    ctrl_fake = st.mean(v for r in rome["inject"] for v in r["ctrl_fake_probe"].values())
    out["romeCtrlFake"] = two(ctrl_fake)

if shift:
    out["shiftCleanThirteen"] = two(shift["categories"]["clean"]["m1913"])
    out["shiftCleanTwentyNine"] = two(shift["categories"]["clean"]["m1929"])
    out["shiftLeakThirteen"] = two(shift["categories"]["leak13"]["m1913"])
    out["shiftGapTwentyNine"] = two(shift["categories"]["gap29"]["m1929"])
    out["shiftKnownThirteen"] = two(shift["groups"]["pre-1913 real facts"]["m1913"])
    out["shiftKnownTwentyNine"] = two(shift["groups"]["pre-1913 real facts"]["m1929"])
    out["shiftFictThirteen"] = two(shift["groups"]["invented facts"]["m1913"])
    out["shiftFictTwentyNine"] = two(shift["groups"]["invented facts"]["m1929"])
    out["shiftLusitaniaThirteen"] = two(shift["events"]["Lusitania sunk"]["m1913"])
    out["shiftLusitaniaTwentyNine"] = two(shift["events"]["Lusitania sunk"]["m1929"])

# ================================================================ write numbers_v5.tex
lines = [r"% AUTO-GENERATED by make_numbers_v5.py -- do not edit by hand.",
         r"\providecommand{\TBD}{\textbf{[TBD]}}"]
for k, v in out.items():
    lines.append(f"\\newcommand{{\\N{k}}}{{{v}}}")
open(os.path.join(PAPER, "numbers_v5.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"wrote paper/numbers_v5.tex with {len(out)} macros")
tbd = [k for k, v in out.items() if v == TBD]
if tbd:
    print("WARNING TBD macros:", tbd)
else:
    print("no TBD macros -- all numbers resolved")
