#!/usr/bin/env python3
"""Unblind + score the night grading (runs/night_20260929/judging_night). Written BEFORE any grade was seen.

Scores (same definitions as scripts/v2/analyze_v2.py):
  substance_history36 = substance*25 on the 36 history-reasoning prompts (main outcome)
  overall_all84       = judge overall_0_100 (mean of 5 axes*25, capped at 25 if fatal) on all 84 prompts
  period_fit_all84    = period_fit 0-4; anachronism_flagged = replies with >=1 listed anachronism by any judge
Per prompt x version: mean over the two judges. Anchors (base, talkie13b) appear in both calls of a prompt:
  averaged over the two calls; the g1-vs-g2 difference is reported as a consistency check.
Paired tests over prompts: bootstrap 95% CI of the mean difference + one-sided sign-flip permutation p (10,000).
Pre-set contrasts (each isolates one change):
  C0 cpt_raw - base                         continued pretraining alone (no Q+A training)
  C1 cpt_bookqa ep3 - bookqa(base Ranke) ep3 continued pretraining, same book Q+A data
  C2 cpt_no_encyclopedia ep3 - cpt_bookqa ep3 adding Talkie answers to book questions
  C3 cpt_everything ep3 - cpt_no_encyclopedia ep3 adding Talkie answers to encyclopaedia questions
  C4 everything_plus ep3 - cpt_everything ep3 adding 47 more books + Talkie answers to their questions
  C5 everything_plus ep5 - everything_plus ep3 longer training
  C6 cpt_talkie_all ep3 - cpt_bookqa ep3     Talkie answers only vs book pairs only
Plus every version vs base (Holm over the 11) and every version vs talkie13b (gap to teacher).
Output: judging_night/analysis/results.json, per_grade.csv (one row per call x reply x judge).
"""
import csv, json, random, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
JN = ROOT / "runs" / "night_20260929" / "judging_night"
BATTERY = ROOT / "eval" / "frozen_eval_84.jsonl"
AX = ["directness", "substance", "coherence", "naturalness", "period_fit"]
EP = "cpt_everything_plus_r64_5ep_ep"
VERSIONS = ["cpt_raw", "bookqa_all_r64c_ep3", "cpt_bookqa_all_r64_ep3", "cpt_talkie_all_r64_ep3",
            "cpt_no_encyclopedia_r64_ep3", "cpt_everything_r64_ep3"] + [f"{EP}{e}" for e in range(1, 6)]
ANCH = ["base", "talkie13b"]
ALL = ANCH + VERSIONS
CONTRASTS = {"C0_cpt_alone": ("cpt_raw", "base"),
             "C1_cpt_same_bookqa": ("cpt_bookqa_all_r64_ep3", "bookqa_all_r64c_ep3"),
             "C2_add_talkie_book_answers": ("cpt_no_encyclopedia_r64_ep3", "cpt_bookqa_all_r64_ep3"),
             "C3_add_talkie_encyclopaedia": ("cpt_everything_r64_ep3", "cpt_no_encyclopedia_r64_ep3"),
             "C4_add_47_books_plus_talkie": (f"{EP}3", "cpt_everything_r64_ep3"),
             "C5_longer_training_ep5_vs_ep3": (f"{EP}5", f"{EP}3"),
             "C6_talkie_only_vs_books_only": ("cpt_talkie_all_r64_ep3", "cpt_bookqa_all_r64_ep3")}
B = 10000
rng = random.Random(1929)

battery = {json.loads(l)["id"]: json.loads(l) for l in open(BATTERY, encoding="utf-8")}
secret = [json.loads(l) for l in open(JN / "secret_label_map.jsonl", encoding="utf-8")]
matrix = {(r["prompt_id"], r["condition"]): r for r in map(json.loads, open(JN / "judge_matrix.jsonl", encoding="utf-8"))}
sys.path.insert(0, str(ROOT / "scripts"))
from judge_blind_outputs import validate  # noqa: E402

judge_files = {"deepseek_flash": lambda i: JN / "judge_deepseek_flash" / i / "judgment.json",
               "sonnet": lambda i: JN / "judge_sonnet" / f"{i}.json"}
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 else None   # e.g. "sonnet" for a preliminary one-judge result
if ONLY:
    judge_files = {k: v for k, v in judge_files.items() if k in ONLY}
SUFFIX = "" if not ONLY else "_" + "_".join(ONLY)
raw = defaultdict(list)          # (pid, cond, judge) -> list of grades (anchors: 2 calls)
cover, invalid, rows = defaultdict(int), [], []
for s in secret:
    iid, pid, lab = s["item_id"], s["prompt_id"], s["labels"]
    for jn, pat in judge_files.items():
        f = pat(iid)
        if not f.exists():
            continue
        try:
            j = validate(json.loads(f.read_text(encoding="utf-8")), iid, list(lab))
        except Exception as e:  # keep a record; excluded from scores
            invalid.append({"judge": jn, "item_id": iid, "error": str(e)[:200]}); continue
        cover[jn] += 1
        for g in j["grades"]:
            c = lab[g["label"]]
            raw[(pid, c, jn)].append({**g, "call": iid[-2:]})
            rows.append([iid, pid, battery[pid]["suite"], battery[pid]["category"], c, jn, g["label"]] + [g[a] for a in AX] +
                        [g["fatal_failure"], g["overall_0_100"], len(g.get("anachronisms") or [])])

sub = lambda g: g["substance"] * 25
ovr = lambda g: g["overall_0_100"]
pft = lambda g: g["period_fit"]
hist = [i for i in battery if battery[i]["suite"] == "history_reasoning_36"]
allp = list(battery)
judges = list(judge_files)


def score(pid, c, fn, js=None, call=None):
    v = []
    for jn in (js or judges):
        gs = [g for g in raw.get((pid, c, jn), []) if call is None or g["call"] == call]
        if gs:
            v.append(sum(fn(g) for g in gs) / len(gs))
    return sum(v) / len(v) if v else None


def mean(x):
    x = [v for v in x if v is not None]
    return sum(x) / len(x) if x else None


def boot(x):
    x = [v for v in x if v is not None]
    ms = sorted(sum(rng.choice(x) for _ in x) / len(x) for _ in range(2000))
    return [ms[50], ms[1949]]


def paired(ids, a, b, fn, js=None):
    d = [(score(i, a, fn, js), score(i, b, fn, js)) for i in ids]
    d = [x - y for x, y in d if x is not None and y is not None]
    n = len(d); md = sum(d) / n
    bs = sorted(sum(rng.choice(d) for _ in d) / n for _ in range(B))
    p = sum(1 for _ in range(B) if sum(v if rng.random() < .5 else -v for v in d) / n >= md) / B
    return {"n": n, "mean_diff": md, "ci95": [bs[int(.025 * B)], bs[int(.975 * B)]], "p_one_sided": p,
            "wins": sum(v > 0 for v in d), "losses": sum(v < 0 for v in d), "ties": sum(v == 0 for v in d)}


res = {"coverage_calls_per_judge": dict(cover), "calls_expected": len(secret), "invalid": invalid}
res["means"] = {c: {"substance_history36": mean([score(i, c, sub) for i in hist]),
                    "substance_ci": boot([score(i, c, sub) for i in hist]),
                    "overall_all84": mean([score(i, c, ovr) for i in allp]),
                    "overall_ci": boot([score(i, c, ovr) for i in allp]),
                    "period_fit_all84": mean([score(i, c, pft) for i in allp]),
                    "anachronism_flagged": sum(1 for i in allp if any(len(g.get("anachronisms") or []) for jn in judges for g in raw.get((i, c, jn), []))),
                    "fatal_share": mean([mean([float(g["fatal_failure"]) for jn in judges for g in raw.get((i, c, jn), [])]) for i in allp]),
                    "per_judge_substance": {jn: mean([score(i, c, sub, [jn]) for i in hist]) for jn in judges},
                    "mean_new_tokens": mean([matrix[(i, c)].get("new_tokens") for i in allp]),
                    "hit_token_ceiling": sum(matrix[(i, c)].get("termination") == "length" for i in allp)}
                for c in ALL}
vb = {c: paired(hist, c, "base", sub) for c in VERSIONS}
ps = sorted((vb[c]["p_one_sided"], c) for c in VERSIONS); holm, run = {}, 0
for k, (p, c) in enumerate(ps):
    run = max(run, min(1.0, p * (len(ps) - k))); holm[c] = run
res["vs_base_substance"] = {c: {**vb[c], "p_holm": holm[c]} for c in VERSIONS}
res["vs_talkie_substance"] = {c: paired(hist, c, "talkie13b", sub) for c in VERSIONS}
res["contrasts"] = {k: {"a": a, "b": b, "substance_history36": paired(hist, a, b, sub), "overall_all84": paired(allp, a, b, ovr),
                        "period_fit_all84": paired(allp, a, b, pft)} for k, (a, b) in CONTRASTS.items()}
res["epoch_curve_substance"] = {e: res["means"][f"{EP}{e}"]["substance_history36"] for e in range(1, 6)}
res["anchor_consistency_g1_vs_g2"] = {a: {"mean_abs_diff_substance": mean([abs(score(i, a, sub, call="g1") - score(i, a, sub, call="g2"))
                                                                           for i in hist if score(i, a, sub, call="g1") is not None and score(i, a, sub, call="g2") is not None]),
                                           "mean_g1": mean([score(i, a, sub, call="g1") for i in hist]),
                                           "mean_g2": mean([score(i, a, sub, call="g2") for i in hist])}
                                       for a in ANCH}
# judge agreement on overall (same reply, both judges)
pairs = [(raw[(p, c, "deepseek_flash")][k]["overall_0_100"], raw[(p, c, "sonnet")][k]["overall_0_100"])
         for (p, c, jn) in list(raw) if jn == "deepseek_flash" and (p, c, "sonnet") in raw
         for k in range(min(len(raw[(p, c, "deepseek_flash")]), len(raw[(p, c, "sonnet")])))]


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


res["judge_agreement"] = {"n_pairs": len(pairs), "spearman_overall": spearman([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None}
out = JN / "analysis"; out.mkdir(exist_ok=True)
with open(out / f"per_grade{SUFFIX}.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["item_id", "prompt_id", "suite", "category", "condition", "judge", "label"] + AX + ["fatal_failure", "overall_0_100", "n_anachronisms"])
    w.writerows(rows)
(out / f"results{SUFFIX}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
r2 = lambda v: round(v, 1) if isinstance(v, float) else v
print(json.dumps({"coverage": dict(cover), "invalid": len(invalid),
                  "means": {c: {k: r2(res["means"][c][k]) for k in ("substance_history36", "overall_all84", "period_fit_all84", "anachronism_flagged")} for c in ALL},
                  "contrasts": {k: {m: [r2(v["mean_diff"]), [r2(x) for x in v["ci95"]], v["p_one_sided"]] for m, v in d.items() if m not in ("a", "b")}
                                for k, d in res["contrasts"].items()},
                  "agreement": res["judge_agreement"]}, indent=1))
