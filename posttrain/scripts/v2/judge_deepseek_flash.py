#!/usr/bin/env python3
"""Modern eval judge 1 (plan v2.4): DeepSeek-V4 Flash via OpenRouter, unchanged JUDGE_PROMPT.md.
One call per item (all 4 anonymized replies). Saves request, raw reply, parsed judgment per item; resume-safe.
Hard cap: stops if cumulative cost exceeds --max-usd."""
import argparse, concurrent.futures as cf, json, os, re, sys, threading
from pathlib import Path

API = Path(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api")
os.environ["OPENROUTER_API_KEY"] = (API / "openrouter-secundo.txt").read_text().strip()
sys.path.insert(0, str(API))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openrouter_safe_call import or_chat  # noqa: E402
from judge_blind_outputs import validate  # noqa: E402

MODEL = "deepseek/deepseek-v4-flash"
MAX_TOKENS = 16000
lock = threading.Lock()
spent = [0.0]


def one(item, out_root, judge_prompt, max_usd):
    out = out_root / item["item_id"]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "judgment.json").exists():
        return "cached"
    with lock:
        if spent[0] > max_usd:
            return "cap"
    req = judge_prompt.rstrip() + "\n\nITEM AND ANONYMIZED REPLIES:\n" + json.dumps(item, ensure_ascii=False)
    (out / "request.txt").write_text(req, encoding="utf-8")
    labels = [r["label"] for r in item["replies"]]
    last = None
    for attempt in range(3):
        txt, meta = or_chat([{"role": "user", "content": req}], model=MODEL, max_tokens=MAX_TOKENS, reasoning_max=6000)
        with lock:
            spent[0] += meta.get("cost") or 0
        (out / f"raw_attempt{attempt}.txt").write_text(txt or "", encoding="utf-8")
        (out / f"meta_attempt{attempt}.json").write_text(json.dumps(meta, default=str), encoding="utf-8")
        try:
            m = re.search(r"\{.*\}", txt, re.S)
            res = validate(json.loads(m.group(0)), item["item_id"], labels)
            res["judge_model"] = MODEL
            (out / "judgment.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
            return "ok"
        except Exception as e:
            last = e
    (out / "error.txt").write_text(str(last), encoding="utf-8")
    return "error"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blinded", required=True)
    ap.add_argument("--judge-prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-usd", type=float, default=3.0)
    # 2026-09-30: provider ignores the 6,000 reasoning cap; 7-8-reply calls hit 16,000 total tokens in 32% of
    # attempts (paid, no grades). Defaults unchanged; the night grading passes --max-tokens 32000 --workers 24.
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--api-threads", dest="workers", type=int, default=12)  # network-bound API calls, not local compute (JV OK for 12, 2026-09-30 09:21)
    a = ap.parse_args()
    global MAX_TOKENS
    MAX_TOKENS = a.max_tokens
    items = [json.loads(l) for l in open(a.blinded, encoding="utf-8") if l.strip()]
    jp = Path(a.judge_prompt).read_text(encoding="utf-8")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from collections import Counter
    with cf.ThreadPoolExecutor(a.workers) as ex:
        res = Counter(ex.map(lambda it: one(it, out, jp, a.max_usd), items))
    print(json.dumps({"results": res, "usd": round(spent[0], 4)}))


if __name__ == "__main__":
    main()
