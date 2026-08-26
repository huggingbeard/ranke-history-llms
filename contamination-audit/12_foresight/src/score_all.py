# -*- coding: utf-8 -*-
"""CORE SCORER (Arm A). For each model in {Ranke-1900, -1913, -1929, Qwen2.5-3B modern anchor}, score every
frozen expectation item ONCE:
  S_cond = -slp(subject, target)     surprise of the outcome given the loaded pre-shock subject
  S_marg = -slp(neutral, target)     lexical base rate of the outcome (de-loaded frame)
  strain = S_cond - S_marg           PMI conceptual strain (tokenizer-internal; comparable across tokenizers)
  forced choice over [target]+twins, calibrated c(a)=slp(subject,a)-slp(neutral,a): correct_cal, gap_cal
  raw forced choice (no calibration): correct_raw, gap_raw
  length-matched-twin forced choice (bias-free subset): correct_matched, gap_matched
  entropy over outcome_set [true, consensus-twin]: p_true, H (nats)

Writes scores_long.csv (one row per model x item) + scores_by_model.json. The horizon differencing ΔS, the
gradient, the PMI decomposition and Δstrain are computed downstream in stats.py from these per-model scores.
Loads one model at a time (bf16 GPU, CPU fallback on OOM); frees between. Run: python score_all.py
"""
import os, json, csv, math, statistics as st
import torch
import common
from foresight_bank import RANKE, MODERN

BANK = json.load(open(os.path.join(common.RESULTS, "question_bank_foresight.json"), encoding="utf-8"))
ITEMS = BANK["items"]

# model tag -> hf id ; cutoff for era logic / labelling
MODELS = [("1900", RANKE["1900"], 1900), ("1913", RANKE["1913"], 1913),
          ("1929", RANKE["1929"], 1929), ("modern", MODERN, 9999)]


def softmax2(a, b):
    m = max(a, b); ea, eb = math.exp(a - m), math.exp(b - m); Z = ea + eb
    return ea / Z, eb / Z


scores = {}   # tag -> id -> dict
for tag, mid, cut in MODELS:
    print(f"\n=== loading {mid} ({tag}) ===", flush=True)
    try:
        tok, model = common.load(mid)
    except torch.cuda.OutOfMemoryError:
        print("  CUDA OOM -> retrying on CPU", flush=True)
        old = common.DEV; common.DEV = "cpu"; common.DT = torch.float32
        tok, model = common.load(mid); common.DEV = old
    scores[tag] = {}
    for it in ITEMS:
        subj, neu, tgt = it["subject"], it["neutral"], it["target"]
        twins = it["twins"]; mtwin = it["matched_twin"]
        # base surprises
        slp_subj_t = common.slp(model, tok, subj, tgt)
        slp_neu_t = common.slp(model, tok, neu, tgt)
        S_cond = -slp_subj_t; S_marg = -slp_neu_t; strain = S_cond - S_marg
        # calibrated forced choice over all options
        c_true = slp_subj_t - slp_neu_t
        c_tw = []
        raw_tw = []
        for t in twins:
            c_tw.append(common.slp(model, tok, subj, t) - common.slp(model, tok, neu, t))
            raw_tw.append(common.slp(model, tok, subj, t))
        c_best = max(c_tw)
        raw_true = slp_subj_t; raw_best = max(raw_tw)
        # matched-twin (bias-free) forced choice
        c_match = common.slp(model, tok, subj, mtwin) - common.slp(model, tok, neu, mtwin)
        # entropy over outcome_set [true, matched consensus twin], calibrated
        p_true, p_false = softmax2(c_true, c_match)
        H = -(p_true * math.log(p_true + 1e-12) + p_false * math.log(p_false + 1e-12))
        scores[tag][it["id"]] = dict(
            S_cond=round(S_cond, 4), S_marg=round(S_marg, 4), strain=round(strain, 4),
            c_true=round(c_true, 4), c_best_twin=round(c_best, 4), c_matched_twin=round(c_match, 4),
            gap_cal=round(c_true - c_best, 4), gap_matched=round(c_true - c_match, 4),
            gap_raw=round(raw_true - raw_best, 4),
            correct_cal=int(c_true > c_best), correct_matched=int(c_true > c_match),
            correct_raw=int(raw_true > raw_best),
            p_true=round(p_true, 4), H=round(H, 4))
    # quick per-model sanity on controls
    b = [scores[tag][it["id"]] for it in ITEMS if it["battery"] == "ctrlB"]
    c = [scores[tag][it["id"]] for it in ITEMS if it["battery"] == "ctrlC"]
    print(f"  Tier-B controls correct_cal: {sum(x['correct_cal'] for x in b)}/{len(b)}  "
          f"(mean strain {st.mean(x['strain'] for x in b):+.2f})", flush=True)
    print(f"  Tier-C future-to-all correct_cal: {sum(x['correct_cal'] for x in c)}/{len(c)}  "
          f"(mean S_cond {st.mean(x['S_cond'] for x in c):+.2f})", flush=True)
    common.free(model)

# ---- write long CSV ----
meta_fields = ["id", "battery", "facet", "event_year", "true_valence", "period_consensus",
               "specificity", "length_matched"]
score_fields = ["S_cond", "S_marg", "strain", "c_true", "c_best_twin", "c_matched_twin",
                "gap_cal", "gap_matched", "gap_raw", "correct_cal", "correct_matched", "correct_raw", "p_true", "H"]
meta_by_id = {it["id"]: it for it in ITEMS}
out_csv = os.path.join(common.RESULTS, "scores_long.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["model", "cutoff"] + meta_fields + score_fields)
    for tag, mid, cut in MODELS:
        for it in ITEMS:
            s = scores[tag][it["id"]]
            w.writerow([tag, cut] + [it.get(k) for k in meta_fields] + [s[k] for k in score_fields])

json.dump({"models": {t: m for t, m, _ in MODELS}, "scores": scores},
          open(os.path.join(common.RESULTS, "scores_by_model.json"), "w"), indent=1)
print(f"\nwrote {out_csv} and scores_by_model.json")
