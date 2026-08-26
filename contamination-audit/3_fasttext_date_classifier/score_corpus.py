"""
Score the big corpus and flag likely post-cutoff (contaminated) documents.

    python score_corpus.py --model data/model.bin --input corpus.jsonl --out scores.jsonl

Input : JSONL, one document per line: {"id": str (optional), "text": str, ...}
Output: JSONL, one line per document:
        {"id":..., "n_chunks":N, "mean":m, "max":mx, "frac_above":f, "flag":bool}

A document is flagged if its strongest passage looks post-cutoff (max >= --flag-max)
OR a meaningful share of passages do (frac_above >= --flag-frac). Tune on the
known-clean vs known-dirty piles before trusting it. Resumable: re-run to continue.

fastText predict is fast; the bottleneck is I/O. This scales to your 160B tokens
on CPU in hours, no LLM needed.
"""
import argparse, os, json
import fasttext
from common import normalize, chunk_words


def prob_post(model, text):
    labels, probs = model.predict(normalize(text), k=-1)
    d = {lab: p for lab, p in zip(labels, probs)}
    if "__label__post" in d:
        return float(d["__label__post"])
    if "__label__pre" in d:
        return 1.0 - float(d["__label__pre"])
    return float(probs[0]) if labels[0].endswith("post") else 1.0 - float(probs[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", type=int, default=200)
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--flag-max", type=float, default=0.90)
    ap.add_argument("--flag-frac", type=float, default=0.10)
    ap.add_argument("--above", type=float, default=0.5, help="passage threshold for frac_above")
    ap.add_argument("--checkpoint", type=int, default=2000, help="flush every N docs")
    args = ap.parse_args()

    # resume: collect ids already scored
    done = set()
    if os.path.exists(args.out):
        for ln in open(args.out, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                try:
                    done.add(json.loads(ln)["id"])
                except Exception:
                    pass
        print(f"resume: {len(done)} docs already scored")

    model = fasttext.load_model(args.model)
    out = open(args.out, "a", encoding="utf-8")

    n, flagged = 0, 0
    for i, ln in enumerate(open(args.input, encoding="utf-8")):
        ln = ln.strip()
        if not ln:
            continue
        d = json.loads(ln)
        did = d.get("id", i)
        if did in done:
            continue
        ps = [prob_post(model, p) for p in chunk_words(d["text"], args.words, args.min_words)]
        if not ps:
            rec = {"id": did, "n_chunks": 0, "mean": None, "max": None, "frac_above": None, "flag": False}
        else:
            mx = max(ps)
            frac = sum(p >= args.above for p in ps) / len(ps)
            flag = (mx >= args.flag_max) or (frac >= args.flag_frac)
            rec = {"id": did, "n_chunks": len(ps), "mean": round(sum(ps) / len(ps), 4),
                   "max": round(mx, 4), "frac_above": round(frac, 4), "flag": flag}
            flagged += flag
        out.write(json.dumps(rec) + "\n")
        n += 1
        if n % args.checkpoint == 0:
            out.flush()
            print(f"  scored {n} docs, flagged {flagged} ...", flush=True)

    out.close()
    print(f"done. scored {n} new docs, flagged {flagged}. -> {args.out}")


if __name__ == "__main__":
    main()
