#!/usr/bin/env python3
"""Stage C data (plan v2.4): nested SFT sets from Ranke-accepted Talkie records.

Same ordering as v1 builder (hash round-robin over source x form, unique passages
first) minus the modern-selector quality tiers, which v2 removed.
Dose rule: 300/1000/3000; if pool < 3000 top dose = all; if pool < 1000 -> 100/300/all.
"""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_nested_talkie_datasets import interleave  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accepted", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.accepted, encoding="utf-8") if l.strip()]
    assert len({r["candidate_id"] for r in rows}) == len(rows)
    n = len(rows)
    if n >= 3000:
        sizes = [300, 1000, 3000]
    elif n >= 1000:
        sizes = [300, 1000, n]
    else:
        sizes = [100, 300, n]
    first, repeats, seen = [], [], set()
    for r in interleave(rows):
        (repeats if r["passage_id"] in seen else first).append(r)
        seen.add(r["passage_id"])
    master = first + repeats
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    man = {"input": a.accepted, "input_rows": n, "sizes": sizes,
           "ordering": "v1 hash round-robin over source x form (seed tag 1929), unique passages first; no quality tiers",
           "sets": {}}
    prev = set()
    for label, size in zip(["qa_300", "qa_1000", "qa_3000"], sizes):
        sel = master[:size]
        ids = {r["candidate_id"] for r in sel}
        assert prev <= ids
        prev = ids
        p = out / f"{label}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sel), encoding="utf-8")
        man["sets"][label] = {"path": str(p), "rows": len(sel), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                              "unique_passages": len({r["passage_id"] for r in sel}),
                              "sources": dict(Counter(r["source_id"] for r in sel)),
                              "forms": dict(Counter(r["form"] for r in sel))}
    (out / "manifest.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in man.items() if k != "sets"} | {"rows": {k: v["rows"] for k, v in man["sets"].items()}}))


if __name__ == "__main__":
    main()
