import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATTERY = ROOT / "eval" / "frozen_eval_84.jsonl"
FIXTURE = ROOT / "scratch" / "report_fixture"
ORDER = ["base", "qa_300", "qa_1000", "qa_3000"]
LABELS = ["M1", "M2", "M3", "M4"]


def dump_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def main():
    if FIXTURE.exists():
        shutil.rmtree(FIXTURE)
    (FIXTURE / "judgments").mkdir(parents=True)
    items = [json.loads(line) for line in BATTERY.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    matrix, mapping = [], []
    for item in items:
        mapping.append({"item_id": item["id"], "labels": dict(zip(LABELS, ORDER))})
        for index, condition in enumerate(ORDER):
            matrix.append({"prompt_id": item["id"], "condition": condition,
                           "answer": f"Synthetic {condition} answer for {item['id']}.",
                           "termination": "eos", "new_tokens": 80 + index})
        grades = []
        for index, label in enumerate(LABELS):
            score = 35 + (0, 8, 16, 24)[index]
            grades.append({"label": label, "directness": 2 + index // 2, "substance": 2 + index // 2,
                           "coherence": 2 + index // 2, "naturalness": 2 + index // 2,
                           "period_fit": 2 + index // 2, "fatal_failure": False,
                           "overall_0_100": score, "evidence": "Synthetic fixture.", "anachronisms": []})
        result = {"item_id": item["id"], "grades": grades,
                  "ranking": [["M4"], ["M3"], ["M2"], ["M1"]], "ranking_reason": "Synthetic fixture."}
        target = FIXTURE / "judgments" / item["id"]
        target.mkdir()
        (target / "judgment.json").write_text(json.dumps(result), encoding="utf-8")
    matrix_path, mapping_path = FIXTURE / "matrix.jsonl", FIXTURE / "mapping.jsonl"
    dump_jsonl(matrix_path, matrix)
    dump_jsonl(mapping_path, mapping)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    command = [sys.executable, str(ROOT / "scripts" / "analyze_and_report.py"),
               "--battery", str(BATTERY), "--battery-sha256", sha(BATTERY),
               "--matrix", str(matrix_path), "--matrix-sha256", sha(matrix_path),
               "--mapping", str(mapping_path), "--judgments", str(FIXTURE / "judgments"),
               "--out", str(FIXTURE / "report")]
    subprocess.run(command, check=True)
    summary = json.loads((FIXTURE / "report" / "summary.json").read_text(encoding="utf-8"))
    assert summary["design"]["n_prompts"] == 84
    assert summary["slope_gate"]["pass"] is True
    assert len(summary["gate_history_group_gains"]) == 6
    assert (FIXTURE / "report" / "report.html").stat().st_size > 10_000
    print(json.dumps({"fixture": "passed", "report": str(FIXTURE / "report" / "report.html")}))


if __name__ == "__main__":
    main()
