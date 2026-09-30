#!/usr/bin/env python3
"""Plan v2.11 analysis: unblind the two modern judges' grades and compute every pre-set statistic.

Inputs (local run folder RUN):
  judging/secret_label_map.jsonl, judging/calibration_key.jsonl
  judging/judge_deepseek_flash/<item>/judgment.json, judging/judge_sonnet/<item>.json
  eval/judge_matrix.jsonl
Output: RUN/analysis/results.json and RUN/analysis/per_prompt.csv (one row per prompt x condition x judge).
"""
import csv, json, random, sys
from collections import defaultdict
from pathlib import Path

RUN = Path(sys.argv[1])
BATTERY = Path(r"G:\detritus\ranke1929_talkie_sft_20260928\eval\frozen_eval_84.jsonl")
AX = ["directness", "substance", "coherence", "naturalness", "period_fit"]
CONDS = ["base", "qa_300", "qa_1000", "qa_3000"]
TRAINED = CONDS[1:]
rng = random.Random(1929)
B = 10000

battery = {json.loads(l)["id"]: json.loads(l) for l in open(BATTERY, encoding="utf-8")}
secret = {json.loads(l)["item_id"]: json.loads(l)["labels"] for l in open(RUN / "judging" / "secret_label_map.jsonl", encoding="utf-8")}
matrix = {(r["prompt_id"], r["condition"]): r for r in map(json.loads, open(RUN / "eval" / "judge_matrix.jsonl", encoding="utf-8"))}


def load_judge(name, pattern):
    grades = {}
    for iid in secret:
        f = pattern(iid)
        if not f.exists():
            continue
        j = json.loads(f.read_text(encoding="utf-8"))
        for g in j["grades"]:
            grades[(iid, secret[iid][g["label"]])] = g
    return grades


J = {"deepseek_flash": load_judge("deepseek_flash", lambda i: RUN / "judging" / "judge_deepseek_flash" / i / "judgment.json"),
     "sonnet": load_judge("sonnet", lambda i: RUN / "judging" / "judge_sonnet" / f"{i}.json")}
if len(sys.argv) > 2:  # optional: restrict to named judges, e.g. "sonnet" (preliminary run)
    J = {k: v for k, v in J.items() if k in sys.argv[2].split(",")}
OUTNAME = "results.json" if len(sys.argv) <= 2 else f"results_{sys.argv[2].replace(',', '_')}.json"
coverage = {k: len({i for i, _ in v}) for k, v in J.items()}

# calibration
cal_key = {json.loads(l)["item_id"]: json.loads(l)["labels"] for l in open(RUN / "judging" / "calibration_key.jsonl", encoding="utf-8")}
calib = {}
for jn, pat in [("deepseek_flash", lambda i: RUN / "judging" / "judge_deepseek_flash_cal" / i / "judgment.json"),
                ("sonnet", lambda i: RUN / "judging" / "judge_sonnet_cal" / f"{i}.json")]:
    ok = n = 0
    for iid, lab in cal_key.items():
        f = pat(iid)
        if not f.exists():
            continue
        g = {x["label"]: x for x in json.loads(f.read_text(encoding="utf-8"))["grades"]}
        clean = next(k for k, v in lab.items() if v == "clean"); inj = next(k for k, v in lab.items() if v == "injected")
        n += 1; ok += g[inj]["period_fit"] < g[clean]["period_fit"]
    calib[jn] = {"n": n, "injected_lower_period_fit": ok, "passed": n == 20 and ok >= 16}
fit_judges = [j for j in J if calib.get(j, {}).get("passed")]

per = defaultdict(dict)   # (iid, cond) -> judge -> grade
for jn, gr in J.items():
    for (iid, cond), g in gr.items():
        per[(iid, cond)][jn] = g
(RUN / "analysis").mkdir(exist_ok=True)
with open(RUN / "analysis" / "per_prompt.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["item_id", "suite", "category", "condition", "judge"] + AX + ["fatal_failure", "overall_0_100", "n_anachronisms", "new_tokens", "termination", "cut_at_new_question"])
    for (iid, cond), d in sorted(per.items()):
        m = matrix[(iid, cond)]
        for jn, g in d.items():
            w.writerow([iid, battery[iid]["suite"], battery[iid]["category"], cond, jn] + [g[a] for a in AX] +
                       [g["fatal_failure"], g["overall_0_100"], len(g.get("anachronisms", [])), m["new_tokens"], m["termination"], m["cut_at_new_question"]])


def score(iid, cond, fn, judges=None):
    d = per.get((iid, cond), {})
    vals = [fn(g) for jn, g in d.items() if judges is None or jn in judges]
    return sum(vals) / len(vals) if vals else None


substance = lambda g: g["substance"] * 25
fluency = lambda g: (g["coherence"] + g["naturalness"] + g["period_fit"]) / 3 * 25
overall = lambda g: g["overall_0_100"]
pfit = lambda g: g["period_fit"]
hist_ids = [i for i in battery if battery[i]["suite"] == "history_reasoning_36"]
all_ids = list(battery)


def vec(ids, cond, fn, judges=None):
    return [score(i, cond, fn, judges) for i in ids]


def mean(x):
    x = [v for v in x if v is not None]
    return sum(x) / len(x) if x else None


def boot_ci(x):
    x = [v for v in x if v is not None]
    ms = sorted(sum(rng.choice(x) for _ in x) / len(x) for _ in range(2000))
    return [ms[50], ms[1949]]


def paired(ids, a, b, fn, judges=None):
    d = [(score(i, a, fn, judges), score(i, b, fn, judges)) for i in ids]
    d = [x - y for x, y in d if x is not None and y is not None]
    n = len(d)
    md = sum(d) / n
    bs = sorted(sum(rng.choice(d) for _ in d) / n for _ in range(B))
    perm = sum(1 for _ in range(B) if sum(v if rng.random() < .5 else -v for v in d) / n >= md) / B
    return {"n": n, "mean_diff": md, "ci95": [bs[int(.025 * B)], bs[int(.975 * B)]], "p_one_sided": perm,
            "wins": sum(v > 0 for v in d), "losses": sum(v < 0 for v in d), "ties": sum(v == 0 for v in d)}


res = {"coverage_items_per_judge": coverage, "calibration": calib, "judges_used_for_period_fit": fit_judges}
res["means"] = {c: {"substance_history36": mean(vec(hist_ids, c, substance)), "substance_ci": boot_ci(vec(hist_ids, c, substance)),
                    "fluency_all84": mean(vec(all_ids, c, fluency)), "fluency_ci": boot_ci(vec(all_ids, c, fluency)),
                    "overall_all84": mean(vec(all_ids, c, overall)),
                    "period_fit_all84": mean(vec(all_ids, c, pfit, fit_judges or None)),
                    "anachronism_flagged_replies": sum(1 for i in all_ids if any(len(g.get("anachronisms", [])) for g in per.get((i, c), {}).values())),
                    "mean_new_tokens": mean([matrix[(i, c)]["new_tokens"] for i in all_ids]),
                    "hit_token_ceiling": sum(matrix[(i, c)]["termination"] == "length" for i in all_ids),
                    "invented_new_question": sum(matrix[(i, c)]["cut_at_new_question"] for i in all_ids)}
                for c in CONDS}
eff = {c: paired(hist_ids, c, "base", substance) for c in TRAINED}
ps = sorted((eff[c]["p_one_sided"], c) for c in TRAINED)
holm = {}
for k, (p, c) in enumerate(ps):
    holm[c] = min(1.0, max(p * (len(ps) - k), max([holm[x] for x in holm] or [0])))
nonin = {c: paired(all_ids, c, "base", pfit, fit_judges or None) for c in TRAINED}
res["effect_vs_base_substance"] = {c: {**eff[c], "p_holm": holm[c]} for c in TRAINED}
res["period_fit_noninferiority"] = {c: {**nonin[c], "margin": -0.25, "noninferior": nonin[c]["ci95"][0] > -0.25,
                                        "flags_ok": res["means"][c]["anachronism_flagged_replies"] <= res["means"]["base"]["anachronism_flagged_replies"] + 3}
                                    for c in TRAINED}
res["effect_rule"] = {c: res["effect_vs_base_substance"][c]["mean_diff"] > 0 and res["effect_vs_base_substance"][c]["p_holm"] < 0.05
                      and res["period_fit_noninferiority"][c]["noninferior"] and res["period_fit_noninferiority"][c]["flags_ok"] for c in TRAINED}
groups = sorted({battery[i]["category"] for i in hist_ids})
slope = paired(hist_ids, "qa_3000", "qa_1000", substance)
g_imp = {g: mean(vec([i for i in hist_ids if battery[i]["category"] == g], "qa_3000", substance)) >
            mean(vec([i for i in hist_ids if battery[i]["category"] == g], "qa_1000", substance)) for g in groups}
res["slope_top_vs_1000"] = {**slope, "groups_improved": sum(g_imp.values()), "groups": g_imp,
                            "passes": slope["mean_diff"] >= 5 and sum(g_imp.values()) >= 4 and slope["wins"] > slope["losses"]}
res["group_means_substance"] = {g: {c: mean(vec([i for i in hist_ids if battery[i]["category"] == g], c, substance)) for c in CONDS} for g in groups}
res["suite_means_overall"] = {s: {c: mean(vec([i for i in all_ids if battery[i]["suite"] == s], c, overall)) for c in CONDS}
                              for s in sorted({b["suite"] for b in battery.values()})}
res["category_means_overall"] = {g: {c: mean(vec([i for i in all_ids if battery[i]["category"] == g], c, overall)) for c in CONDS}
                                 for g in sorted({b["category"] for b in battery.values()})}
# judge agreement
pairs = [(per[k]["deepseek_flash"], per[k]["sonnet"]) for k in per if "deepseek_flash" in per[k] and "sonnet" in per[k]]


def spearman(a, b):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v); i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            for k in range(i, j + 1):
                r[s[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = rk(a), rk(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** .5
    return num / den if den else None


def wkappa(a, b, k=5):
    n = len(a); O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[x][y] += 1
    ra = [sum(r) for r in O]; cb = [sum(O[i][j] for i in range(k)) for j in range(k)]
    num = sum(((i - j) ** 2) * O[i][j] for i in range(k) for j in range(k))
    den = sum(((i - j) ** 2) * ra[i] * cb[j] / n for i in range(k) for j in range(k))
    return 1 - num / den if den else None


res["judge_agreement"] = {} if not pairs else {"n_pairs": len(pairs),
                          "spearman_overall": spearman([a["overall_0_100"] for a, _ in pairs], [b["overall_0_100"] for _, b in pairs]),
                          "weighted_kappa": {ax: wkappa([a[ax] for a, _ in pairs], [b[ax] for _, b in pairs]) for ax in AX}}
same_best = 0; nb = 0
for i in all_ids:
    best = []
    for jn in J:
        sc = {c: per.get((i, c), {}).get(jn, {}).get("overall_0_100") for c in CONDS}
        if None in sc.values():
            break
        best.append(max(sc, key=sc.get))
    if len(best) == 2:
        nb += 1; same_best += best[0] == best[1]
res["judge_agreement"]["same_best_condition_share"] = same_best / nb if nb else None
res["per_judge_means"] = {jn: {c: {"substance_history36": mean(vec(hist_ids, c, substance, [jn])),
                                   "overall_all84": mean(vec(all_ids, c, overall, [jn]))} for c in CONDS} for jn in J}
(RUN / "analysis" / OUTNAME).write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps({"coverage": coverage, "calibration": calib, "means": {c: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in res["means"][c].items() if not k.endswith("ci")} for c in CONDS},
                  "effect": {c: {k: round(v, 4) if isinstance(v, float) else v for k, v in res["effect_vs_base_substance"][c].items()} for c in TRAINED},
                  "effect_rule": res["effect_rule"], "slope": {k: v for k, v in res["slope_top_vs_1000"].items() if k != "groups"},
                  "agreement": res["judge_agreement"]}, indent=1, default=str))
