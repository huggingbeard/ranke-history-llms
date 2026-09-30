#!/usr/bin/env python3
"""Blind grading inputs for the night 2026-09-29/30 training versions (same judges + JUDGE_PROMPT.md as round 2).

Design (fixed before any grade is seen):
- 11 graded versions: cpt_raw (continued pretraining, no Q+A training), the FINAL epoch of each 3-epoch arm
  (bookqa_all_r64c, cpt_talkie_all_r64, cpt_bookqa_all_r64, cpt_no_encyclopedia_r64, cpt_everything_r64),
  and all 5 epochs of cpt_everything_plus_r64_5ep.
- Two anchors in EVERY grading call: base Ranke-1929 and Talkie-13B direct answers, the exact texts graded in
  round 2's 5-way run (taken from judging5 blinded items + secret map). Anchors keep the grading context equal
  across calls and link this round to round 2.
- Per item, the 11 versions are shuffled (seed 1929) into two calls of 6 and 5, each + the 2 anchors
  (8 and 7 replies). Labels shuffled per call. Item ids <prompt_id>__g1 / __g2.
- Answers cut at the first new 'Question:'/'Q:' turn, as in prepare_judging.py (raw kept).
Outputs in runs/night_20260929/judging_night/: judge_matrix.jsonl, blinded_items.jsonl, secret_label_map.jsonl,
manifest.json (input files + sha256).
"""
import hashlib, json, random, re
from pathlib import Path

ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
N = ROOT / "runs" / "night_20260929"
BATTERY = ROOT / "eval" / "frozen_eval_84.jsonl"
J5 = ROOT / "runs" / "v2_20260929_0015" / "cluster_final" / "out" / "judging5"
OUT = N / "judging_night"
SOURCES = {
    "cpt_raw": N / "cpt_bookqa_all_r64" / "eval_answers.jsonl",
    "bookqa_all_r64c_ep3": N / "bookqa_all_r64c" / "eval_answers.jsonl",
    "cpt_talkie_all_r64_ep3": N / "cpt_talkie_all_r64" / "eval_answers.jsonl",
    "cpt_bookqa_all_r64_ep3": N / "cpt_bookqa_all_r64" / "eval_answers.jsonl",
    "cpt_no_encyclopedia_r64_ep3": N / "cluster_cpt_no_encyclopedia_r64" / "eval_answers.jsonl",
    "cpt_everything_r64_ep3": N / "cluster_cpt_everything_r64" / "eval_answers.jsonl",
    **{f"cpt_everything_plus_r64_5ep_ep{e}": N / "cluster_cpt_everything_plus_r64_5ep" / "eval_answers.jsonl"
       for e in range(1, 6)},
}
ANCHORS = ["base", "talkie13b"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


battery = jl(BATTERY)
ids = [b["id"] for b in battery]
cut_re = re.compile(r"\n\s*(Question|Q)\s*:")
matrix = {}
for cond, path in SOURCES.items():
    rows = [r for r in jl(path) if r["condition"] == cond]
    assert len(rows) == 84 and {r["prompt_id"] for r in rows} == set(ids), (cond, len(rows))
    for r in rows:
        m = cut_re.search(r["answer"])
        matrix[(r["prompt_id"], cond)] = {
            "prompt_id": r["prompt_id"], "condition": cond, "answer_raw": r["answer"],
            "answer": r["answer"][:m.start()].strip() if m else r["answer"], "cut_at_new_question": bool(m),
            "termination": r.get("termination"), "new_tokens": r.get("new_tokens"), "source": str(path)}

# anchors: exact round-2 texts
j5_items = {x["item_id"]: x for x in jl(J5 / "blinded_items.jsonl")}
j5_map = {x["item_id"]: x["labels"] for x in jl(J5 / "secret_label_map.jsonl")}
for pid in ids:
    rep = {j5_map[pid][r["label"]]: r["reply"] for r in j5_items[pid]["replies"]}
    for a in ANCHORS:
        matrix[(pid, a)] = {"prompt_id": pid, "condition": a, "answer": rep[a], "answer_raw": None,
                            "cut_at_new_question": None, "source": str(J5 / "blinded_items.jsonl")}

rng = random.Random(1929)
graded = list(SOURCES)
blinded, secret = [], []
for b in battery:
    pid = b["id"]
    order = graded.copy(); rng.shuffle(order)
    for g, part in (("g1", order[:6]), ("g2", order[6:])):
        conds = part + ANCHORS
        rng.shuffle(conds)
        mapping = {f"M{i + 1}": c for i, c in enumerate(conds)}
        iid = f"{pid}__{g}"
        blinded.append({"item_id": iid, "prompt": b["prompt"], "date_limit": b["date_limit"], "checks": b["criteria"],
                        "replies": [{"label": l, "reply": matrix[(pid, c)]["answer"]} for l, c in mapping.items()]})
        secret.append({"item_id": iid, "prompt_id": pid, "labels": mapping})

OUT.mkdir(exist_ok=True)
w = lambda p, rows: p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
w(OUT / "judge_matrix.jsonl", list(matrix.values()))
w(OUT / "blinded_items.jsonl", blinded)
w(OUT / "secret_label_map.jsonl", secret)
man = {"battery": [str(BATTERY), sha(BATTERY)], "judging5_items": [str(J5 / "blinded_items.jsonl"), sha(J5 / "blinded_items.jsonl")],
       "sources": {c: [str(p), sha(p)] for c, p in SOURCES.items()}, "seed": 1929,
       "matrix_rows": len(matrix), "calls": len(blinded),
       "cut_at_new_question": sum(1 for v in matrix.values() if v["cut_at_new_question"])}
(OUT / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
print(json.dumps({k: man[k] for k in ("matrix_rows", "calls", "cut_at_new_question")}))
