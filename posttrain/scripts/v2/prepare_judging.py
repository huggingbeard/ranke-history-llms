#!/usr/bin/env python3
"""Plan v2.11 evaluation prep (runs locally after the cluster job).

1. Cut each eval answer at the first line that starts a new 'Question:' turn (plan section 5);
   keep the raw answer and a cut flag. Writes eval/judge_matrix.jsonl.
2. Blind the 4 conditions per item (shuffled labels, seed 1929) with the existing
   scripts/build_blind_judge_inputs.py -> judging/blinded_items.jsonl + secret_label_map.jsonl.
3. Judge calibration set (plan v2.5): 20 accepted Talkie answers, clean vs one fixed post-1913
   template sentence inserted mid-answer -> judging/calibration_items.jsonl + calibration_key.jsonl.
"""
import hashlib, json, random, re, subprocess, sys
from pathlib import Path

RUN = Path(sys.argv[1])            # local run folder with eval/ and stageB/ fetched from the cluster
ROOT = Path(r"G:\detritus\ranke1929_talkie_sft_20260928")
BATTERY = ROOT / "eval" / "frozen_eval_84.jsonl"
sys.path.insert(0, str(ROOT / "scripts" / "v2"))
from ranke_judge import INJECT, sentences  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


rows = [json.loads(l) for l in open(RUN / "eval" / "blind_output_matrix.jsonl", encoding="utf-8")]
out = []
for r in rows:
    raw = r["answer"]
    m = re.search(r"\n\s*(Question|Q)\s*:", raw)
    cut = raw[:m.start()].strip() if m else raw
    out.append({**r, "answer_raw": raw, "answer": cut, "cut_at_new_question": bool(m)})
jm = RUN / "eval" / "judge_matrix.jsonl"
jm.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
print("judge matrix", len(out), "rows; cut", sum(x["cut_at_new_question"] for x in out))

J = RUN / "judging"
J.mkdir(exist_ok=True)
subprocess.run([sys.executable, str(ROOT / "scripts" / "build_blind_judge_inputs.py"),
                "--battery", str(BATTERY), "--battery-sha256", sha(BATTERY),
                "--matrix", str(jm), "--matrix-sha256", sha(jm), "--out", str(J)], check=True)

acc = [json.loads(l) for l in open(RUN / "stageB" / "stageB_accepted.jsonl", encoding="utf-8")]
acc = [a for a in acc if len(sentences(a["answer"])) >= 2]
rng = random.Random(1929)
pick = rng.sample(acc, 20)
items, key = [], []
for i, a in enumerate(pick):
    ss = sentences(a["answer"]); mid = max(1, len(ss) // 2)
    inj = " ".join(ss[:mid] + [INJECT[i % len(INJECT)]] + ss[mid:])
    clean_first = rng.random() < 0.5
    labels = {"M1": "clean" if clean_first else "injected", "M2": "injected" if clean_first else "clean"}
    text = {"clean": a["answer"], "injected": inj}
    iid = f"CAL{i + 1:02d}"
    items.append({"item_id": iid, "prompt": a["question"], "date_limit": "1913-12-31",
                  "checks": ["Answers the question directly.",
                             "Uses no knowledge or terminology that depends on events after 31 December 1913.",
                             "Explains rather than merely naming facts."],
                  "replies": [{"label": k, "reply": text[v]} for k, v in labels.items()]})
    key.append({"item_id": iid, "labels": labels, "candidate_id": a["candidate_id"], "injected": INJECT[i % len(INJECT)]})
(J / "calibration_items.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
(J / "calibration_key.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in key), encoding="utf-8")
print("calibration items", len(items))
