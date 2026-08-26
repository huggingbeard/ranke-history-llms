"""
Stream a balanced sample from institutional/institutional-books-1.0 into the
pipeline's data contract: {"text":..., "year":..., "genre":...}.

We balance by (genre, side-of-1913) so eval.py's per-genre date check is meaningful.
Text is truncated per book (we don't need whole books -- a chunk gives plenty of
passages). Resumable: re-run to top up without duplicating barcodes.

    python pull_institutional.py --out dated.jsonl --per-side 200 --cell-cap 60
"""
import os, re, json, argparse
from collections import defaultdict

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from datasets import load_dataset

REPO = "institutional/institutional-books-1.0"
TOK = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip()
os.environ["HF_TOKEN"] = TOK
YEAR_RE = re.compile(r"(1[5-9]\d{2}|20[0-2]\d)")


def parse_year(rec):
    for f in ("date1_src", "date2_src"):
        v = rec.get(f)
        if v:
            m = YEAR_RE.search(str(v))
            if m:
                return int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Users\hvoth\timelock-contamination\dated.jsonl")
    ap.add_argument("--cutoff", type=int, default=1913)
    ap.add_argument("--per-side", type=int, default=200, help="target books per side of cutoff")
    ap.add_argument("--cell-cap", type=int, default=60, help="max books per (genre, side)")
    ap.add_argument("--max-words", type=int, default=6000, help="truncate each book to this many words")
    ap.add_argument("--min-words", type=int, default=300)
    ap.add_argument("--scan-cap", type=int, default=20000, help="stop scanning after this many rows")
    args = ap.parse_args()

    # resume: remember barcodes already written, and current counts
    seen, side_count, cell_count = set(), defaultdict(int), defaultdict(int)
    if os.path.exists(args.out):
        for ln in open(args.out, encoding="utf-8"):
            try:
                d = json.loads(ln)
                seen.add(d.get("_barcode"))
                side = "post" if d["year"] > args.cutoff else "pre"
                side_count[side] += 1
                cell_count[(d["genre"], side)] += 1
            except Exception:
                pass
        print(f"resume: {len(seen)} books already in {args.out}")

    ds = load_dataset(REPO, "default", split="train", streaming=True, token=TOK)
    ds = ds.select_columns(["barcode_src", "date1_src", "date2_src",
                            "language_gen", "topic_or_subject_gen", "text_by_page_gen"])

    out = open(args.out, "a", encoding="utf-8")
    scanned, kept = 0, 0
    for rec in ds:
        scanned += 1
        if scanned % 500 == 0:
            print(f"  scanned {scanned}  kept {kept}  (pre={side_count['pre']} post={side_count['post']})", flush=True)
        if scanned > args.scan_cap:
            print("hit scan cap."); break
        if side_count["pre"] >= args.per_side and side_count["post"] >= args.per_side:
            print("both sides full."); break

        if (rec.get("language_gen") or "").lower() != "eng":
            continue
        bc = rec.get("barcode_src")
        if bc in seen:
            continue
        year = parse_year(rec)
        if not year:
            continue
        side = "post" if year > args.cutoff else "pre"
        if side_count[side] >= args.per_side:
            continue
        genre = (rec.get("topic_or_subject_gen") or "UNKNOWN").strip() or "UNKNOWN"
        if cell_count[(genre, side)] >= args.cell_cap:
            continue
        pages = rec.get("text_by_page_gen") or []
        words = " ".join(p for p in pages if p).split()
        if len(words) < args.min_words:
            continue
        text = " ".join(words[:args.max_words])

        out.write(json.dumps({"text": text, "year": year, "genre": genre, "_barcode": bc}) + "\n")
        seen.add(bc); side_count[side] += 1; cell_count[(genre, side)] += 1; kept += 1
        if kept % 25 == 0:
            out.flush()
    out.close()

    print(f"\nDONE. scanned {scanned}, kept {kept}.  pre={side_count['pre']} post={side_count['post']}")
    print("\nper (genre, side):")
    for k in sorted(cell_count):
        print(f"   {k[0][:30]:<30} {k[1]:<5} {cell_count[k]}")


if __name__ == "__main__":
    main()
