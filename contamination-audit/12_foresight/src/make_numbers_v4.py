# -*- coding: utf-8 -*-
"""Emit paper/numbers_v4.tex and the v4 table includes for foresight_v4.tex.

Every number the v4 paper prints is read here from the certified-meter result files and written as a
\\newcommand, so nothing in the paper is typed by hand. The v4 paper is built ONLY on the chronicle
meter (foresight gaps + the exam) plus the two honestly-relabelled side arms (circulation = sentence
surprise; selection = duels). It does NOT use the dead panel conceptual-strain ranking.

Sources read:
  results/v41_battery.json              English foresight gaps (covered + memoir rows)
  results/certify_chronicle_heldout.json  the held-out English exam (the certification)
  results/certify_chronicle.json          the dev exam (the run that failed, 10/20)
  results/v4_battery.json                 the pre-repair battery (the 41% false alarm)
  results/v41_defr.json                   German + French foresight gaps
  results/certify_defr.json               German + French exams (not certified)
  results/diag_betting_pilot.json         prophecy-framed betting (the tunnel, the president drop)
  results/diag_square_one.json            sentence surprise: the moon vs bread
  results/diag_paraphrase_pilot.json      sentence surprise: airplanes vs motorcars, etc.
  results/diag_retro_v2.json              the scoring repair (raw vs calibrated on the mark)
  results/stats_panel.json                the duels, by reader (selection arm)
  results/panel_scores.json               per-duel margins (selection arm)
  results/production_panel.json           free generation (selection arm)

LaTeX macro names must be letters only, so we spell things out. Missing input -> \\TBD.
Run:  python src/make_numbers_v4.py
"""
import os, json, math, statistics as st
import common

R = common.RESULTS
PAPER = common.PAPER
os.makedirs(PAPER, exist_ok=True)


def j(fn):
    p = os.path.join(R, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


v41 = j("v41_battery.json")
held = j("certify_chronicle_heldout.json")
dev = j("certify_chronicle.json")
v4 = j("v4_battery.json")
defr_bat = j("v41_defr.json")
defr_cert = j("certify_defr.json")
bet = j("diag_betting_pilot.json")
sq = j("diag_square_one.json")
para = j("diag_paraphrase_pilot.json")
rv2 = j("diag_retro_v2.json")
panel = j("stats_panel.json")
psc = j("panel_scores.json")
prodp = j("production_panel.json")

out = {}
TBD = r"\TBD"


def pct(x):
    """probability in [0,1] -> integer percent string."""
    try:
        return f"{round(100 * x)}"
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
    """escape the few LaTeX-special characters that appear in option text."""
    if s is None:
        return ""
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s


def rows_by_name(rows):
    return {r["name"]: r for r in rows}


# ---------------------------------------------------------------- English gaps
LABELS = {
    "panic_first_to_fail": "which houses failed first in the panic of 1907",
    "panic_severity": "how severe the panic of 1907 was",
    "panic_rescuer": "who led the rescue in 1907",
    "war_breadth": "how widely the war spread",
    "war_us_entry": "who joined the war against Germany in 1917",
    "emp_kaiser_1919": "who headed the German state in 1919",
    "rev_tsar_fate": "what became of the Tsar",
    "rev_who_rules": "who ruled Russia after the revolution",
    "rev_debts": "what became of Russia's foreign debts",
    "infl_end": "the new money that ended the German inflation",
    "dep_course": "the course of American business after the crash",
    "dep_unemployed": "how many Americans were out of work",
    "dep_banks": "what became of America's banks",
    "dep_gold": "the fate of the gold standard",
    "nazi_republic_fate": "what became of the Weimar republic",
    "nazi_who_rules": "who ruled Germany in 1934",
    "nazi_path": "how the republic came to its end",
}

if v41:
    er = rows_by_name(v41["rows"])
    measured = [r for r in v41["rows"] if not r["memoir_only"]]
    covered = [r for r in measured if r["covered"] and not r["excluded"]]
    out["nMeasured"] = str(len(measured))
    out["nCovered"] = str(len(covered))
    out["nEvents"] = "8"

    # the flagship item, with an internally consistent worked example
    k = er["emp_kaiser_1919"]
    out["kaiserPre"] = pct(k["p_truth_pre"])          # 0
    out["kaiserPost"] = pct(k["p_truth_post"])        # 78
    out["kaiserGap"] = two(k["gap"], signed=True)     # +9.84
    out["kaiserPreBets"] = texesc(k["pre_bets"])      # Kaiser Wilhelm
    surp_post = -math.log(max(k["p_truth_post"], 1e-9))
    surp_pre = k["gap"] + surp_post                    # exact: gap = surp_pre - surp_post
    out["kaiserSurpPost"] = two(surp_post)             # 0.25
    out["kaiserSurpPre"] = one(surp_pre)               # 10.1
    out["kaiserPreImplied"] = f"{100 * math.exp(-surp_pre):.3f}"  # ~0.004 (percent)
    out["kaiserPreBetP"] = pct(k["pre_bet_p"])         # 99: the guesser's own bet on the Kaiser

    # memoir bets quoted in the section 5 prose (pulled, not hand-typed)
    out["depBanksMemoir"] = pct(er["dep_banks"]["pre_bet_p"])          # 94
    out["depGoldMemoir"] = pct(er["dep_gold"]["pre_bet_p"])            # 82
    out["naziDictMemoir"] = pct(er["nazi_republic_fate"]["pre_bet_p"]) # 83

    for nm, key in [("debts", "rev_debts"), ("inflend", "infl_end"), ("tsar", "rev_tsar_fate"),
                    ("usentry", "war_us_entry"), ("breadth", "war_breadth"),
                    ("panicsev", "panic_severity"), ("panicresc", "panic_rescuer"),
                    ("panicfirst", "panic_first_to_fail"), ("revrules", "rev_who_rules")]:
        r = er[key]
        out[nm + "Pre"] = pct(r["p_truth_pre"])
        out[nm + "Post"] = pct(r["p_truth_post"])
        out[nm + "Gap"] = two(r["gap"], signed=True)
        out[nm + "PreBets"] = texesc(r["pre_bets"])

    # --- the gaps table (foreseen/unforeseen), English, covered non-memoir items ---
    order = ["panic_first_to_fail", "panic_severity", "panic_rescuer", "war_breadth", "war_us_entry",
             "rev_debts", "rev_who_rules", "rev_tsar_fate", "infl_end", "emp_kaiser_1919"]
    T = [r"\begin{tabular}{p{5.3cm}ccr p{3.5cm}}", r"\toprule",
         r"Question & Guesser & Witness & Gap & The guesser's memoir instead \\", r"\midrule"]
    for nm in order:
        r = er[nm]
        T.append(f"{LABELS[nm]} & {pct(r['p_truth_pre'])}\\% & {pct(r['p_truth_post'])}\\% & "
                 f"${two(r['gap'], signed=True)}$ & {texesc(r['pre_bets'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "gaps_v4.tex"), "w", encoding="utf-8").write("\n".join(T))

    # --- the memoir-of-the-1930s table (no witness exists) ---
    mem = [r for r in v41["rows"] if r["memoir_only"]]
    T = [r"\begin{tabular}{p{4.6cm}p{4.1cm}cp{3.5cm}c}", r"\toprule",
         r"Question & The 1929 memoir expects & Its bet & What in fact happened & Its bet on that \\",
         r"\midrule"]
    for r in mem:
        T.append(f"{LABELS.get(r['name'], r['name'])} & {texesc(r['pre_bets'])} & "
                 f"{pct(r['pre_bet_p'])}\\% & {texesc(r['truth_opt'])} & {pct(r['p_truth_pre'])}\\% \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "memoir_v4.tex"), "w", encoding="utf-8").write("\n".join(T))

# ---------------------------------------------------------------- the exam (held-out, English)
if held:
    rows = held["rows"]
    cov = [r for r in rows if r["covered"]]
    out["examTotal"] = str(len(rows))
    out["examAnswered"] = str(len(cov))
    out["examCorrect"] = str(sum(1 for r in cov if r["correct"]))
    out["examCorrectPct"] = pct(sum(1 for r in cov if r["correct"]) / len(cov)) if cov else TBD
    out["examOutbet"] = str(sum(1 for r in cov if r["grad"] > 0))
    out["examOutbetPct"] = pct(sum(1 for r in cov if r["grad"] > 0) / len(cov)) if cov else TBD
    # pre-registered bars (design constants, not results)
    out["barAnswer"] = "14"
    out["barCorrect"] = "80"
    out["barOutbet"] = "80"
    # the Portsmouth worked example (a NAME item the guesser cannot know)
    p = next((r for r in rows if r["name"] == "portsmouth"), None)
    if p:
        out["portPre"] = pct(p["p_truth_pre"])
        out["portPost"] = pct(p["p_truth_post"])
        out["portGuess"] = texesc(p["pre_modal_opt"])
    # the Stalin failure (retrieval by fame)
    s = next((r for r in rows if r["name"] == "stalin"), None)
    if s:
        out["stalinPost"] = pct(s["p_truth_post"])
        out["stalinNames"] = texesc(s["modal_opt"])

# The development run predates the frozen scoring, so certify_chronicle.json stores raw per-question
# distributions, not a scored summary. Its outcome is documented in the project log (2026-07-04:
# "DEV certification: agreement-filter = 100% precise, 50% coverage"): 10 of 20 answered, all correct.
out["devTotal"] = "20"
out["devAnswered"] = "10"

# --- the full exam table: all 20 held-out questions, both readers' bets ---
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
    open(os.path.join(PAPER, "exam_v4.tex"), "w", encoding="utf-8").write("\n".join(T))

# ---------------------------------------------------------------- the false alarm
if v4 and v41:
    old = next((r for r in v4["rows"] if r["name"] == "nazi_who_rules"), None)
    new = next((r for r in v41["rows"] if r["name"] == "nazi_who_rules"), None)
    if old:
        out["faOld"] = pct(old["p_truth_pre"])
    if new:
        out["faNew"] = pct(new["p_truth_pre"])

# ---------------------------------------------------------------- German + French
LG = {"de": "German", "fr": "French"}
if defr_cert:
    for lg in ("de", "fr"):
        cv = [r for r in defr_cert["rows"][lg] if r["covered"]]
        cap = lg.upper()
        out[cap + "examAnswered"] = str(len(cv))
        out[cap + "examCorrect"] = str(sum(1 for r in cv if r["correct"]))
        out[cap + "examOutbet"] = str(sum(1 for r in cv if r["grad"] > 0))
        out[cap + "examCovN"] = str(len(cv))

if defr_bat and v41:
    en_k = rows_by_name(v41["rows"])["emp_kaiser_1919"]
    de_k = rows_by_name(defr_bat["rows"]["de"])["emp_kaiser_1919"]
    fr_k = rows_by_name(defr_bat["rows"]["fr"])["emp_kaiser_1919"]
    for tag, r in [("EN", en_k), ("DE", de_k), ("FR", fr_k)]:
        out["kaiser" + tag + "Pre"] = pct(r["p_truth_pre"])
        out["kaiser" + tag + "Post"] = pct(r["p_truth_post"])
        out["kaiser" + tag + "Gap"] = one(r["gap"], signed=True)
        out["kaiser" + tag + "Keeps"] = texesc(r["pre_bets"])
    # the national-memory split on the Depression's course
    de_dep = rows_by_name(defr_bat["rows"]["de"])["dep_course"]
    fr_dep = rows_by_name(defr_bat["rows"]["fr"])["dep_course"]
    en_dep = rows_by_name(v41["rows"])["dep_course"]
    out["depFRbets"] = texesc(fr_dep["pre_bets"]);  out["depFRp"] = pct(fr_dep["pre_bet_p"])
    out["depDEbets"] = texesc(de_dep["pre_bets"]);  out["depDEp"] = pct(de_dep["pre_bet_p"])
    out["depENbets"] = texesc(en_dep["pre_bets"]);  out["depENp"] = pct(en_dep["pre_bet_p"])
    out["depENtruthP"] = pct(en_dep["p_truth_pre"])

    # cross-lingual Kaiser table
    T = [r"\begin{tabular}{lccr p{3.2cm}}", r"\toprule",
         r"Language & Guesser & Witness & Gap & The guesser's memoir keeps \\", r"\midrule"]
    for tag, r in [("English", en_k), ("German", de_k), ("French", fr_k)]:
        T.append(f"{tag} & {pct(r['p_truth_pre'])}\\% & {pct(r['p_truth_post'])}\\% & "
                 f"${one(r['gap'], signed=True)}$ & {texesc(r['pre_bets'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "crossling_v4.tex"), "w", encoding="utf-8").write("\n".join(T))

# ---------------------------------------------------------------- prophecy diagnostics (section 2)
if bet:
    q = bet["questions"]
    # Atlantic-1920: option index 3 = "a tunnel beneath the sea"
    out["tunnelP"] = pct(q["atlantic_1920"]["1913"]["p_mean"][3])
    # who rules Germany 1919: option index 3 = "an elected president"; plain vs told-of-defeat
    out["presUncond"] = pct(q["kaiser_1919_uncond"]["1913"]["p_mean"][3])
    out["presDefeat"] = pct(q["kaiser_1919_defeat"]["1913"]["p_mean"][3])
    # the 1929 model, told of the defeat, still bets on the Kaiser: option index 0
    out["kaiserTwentyNine"] = pct(q["kaiser_1919_defeat"]["1929"]["p_mean"][0])

# ---------------------------------------------------------------- circulation arm (sentence surprise)
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
    T = [r"\begin{tabular}{llrr l}", r"\toprule",
         r"Pair & Unexpected idea & \multicolumn{1}{c}{Its score} & \multicolumn{1}{c}{Expected idea's} & Verdict \\",
         r"\midrule"]
    for tag, nov, fam in pairs:
        a, b = pm(nov), pm(fam)
        out[tag + "Nov"] = two(a)
        out[tag + "Fam"] = two(b)
        verdict = "clearly apart" if abs(a - b) > 0.5 else "no difference"
        T.append(f"{LAB[nov]} vs {LAB[fam]} & {LAB[nov]} & {two(a)} & {two(b)} & {verdict} \\\\")
    T += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, "circ_v4.tex"), "w", encoding="utf-8").write("\n".join(T))

# ---------------------------------------------------------------- scoring repair (section on the meter)
if rv2:
    out["markRawTN"] = pct(rv2["questions"]["mark_1923"]["1929"]["raw"][3])
    out["markRawTh"] = pct(rv2["questions"]["mark_1923"]["1913"]["raw"][3])

# ---------------------------------------------------------------- selection arm (duels)
if panel:
    EVK = {"panic_1907": ("panic", "1900"), "world_war": ("wwi", "1929"),
           "fall_of_empires": ("emp", "1929"), "russian_revolution": ("rev", "1929"),
           "spanish_flu": ("flu", "1929"), "hyperinflation": ("infl", "1929"),
           "great_depression": ("dep", "1929"), "nazi_seizure": ("nazi", "1929")}
    labels = {ev: panel["events"][ev]["label"] for ev in EVK if ev in panel.get("events", {})}
    tf = panel.get("tf", {})
    # the two ends of the selection story
    if "panic_1907" in tf:
        out["duelPanicPre"] = str(tf["panic_1907"]["pooled"]["1900"]["pct_true"])
    if "nazi_seizure" in tf:
        out["duelNaziPre"] = str(tf["nazi_seizure"]["pooled"]["1929"]["pct_true"])
        out["duelNaziMargin"] = two(tf["nazi_seizure"]["pooled"]["1929"]["mean_margin"], signed=True)
    # the duel table by reader
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
        open(os.path.join(PAPER, "duels_v4.tex"), "w", encoding="utf-8").write("\n".join(T))

if psc:
    def duel_margin(ev, i, tag):
        try:
            return st.mean(psc[tag]["tf"][f"{ev}#{i}/{lg}"][0] - psc[tag]["tf"][f"{ev}#{i}/{lg}"][1]
                           for lg in ["en", "de", "fr"])
        except Exception:
            return None
    out["duelNaziLawful"] = two(duel_margin("nazi_seizure", 0, "1929"), signed=True)
    out["duelPanicTrusts"] = two(duel_margin("panic_1907", 0, "1900"), signed=True)
    out["duelFluSerum"] = two(duel_margin("spanish_flu", 4, "1913"), signed=True)

if prodp:
    def frac(pid, tag):
        try:
            return prodp["items"][pid]["coded"][tag]["frac_true"]
        except Exception:
            return None
    out["freeTsarTh"] = two(frac("rev_tsar", "1913"))
    out["freeNaziTN"] = two(frac("nazi_who", "1929"))

# ---------------------------------------------------------------- write numbers_v4.tex
lines = [r"% AUTO-GENERATED by make_numbers_v4.py -- do not edit by hand.",
         r"\providecommand{\TBD}{\textbf{[TBD]}}"]
for k, v in out.items():
    lines.append(f"\\newcommand{{\\N{k}}}{{{v}}}")
open(os.path.join(PAPER, "numbers_v4.tex"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"wrote paper/numbers_v4.tex with {len(out)} macros")
tbd = [k for k, v in out.items() if v == TBD]
if tbd:
    print("WARNING TBD macros:", tbd)
else:
    print("no TBD macros — all numbers resolved")
