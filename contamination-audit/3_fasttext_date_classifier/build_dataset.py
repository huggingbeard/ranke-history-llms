"""
Turn dated documents into fastText training data.

Input : JSONL of {"text": str, "year": int, "genre": str (optional)}
Output: train.txt / val.txt  (fastText format: "__label__pre1913 <passage>")
        test.jsonl           (rich test set: {text, year, genre} for eval.py)

Splits are stratified by (genre, label) so every genre appears in every split,
which lets eval.py prove the model learned DATE and not TOPIC.

Example:
    python build_dataset.py --input dated.jsonl --outdir data --cutoff 1913 --words 200
"""
import argparse, os, random, json
from collections import defaultdict
from common import read_jsonl, normalize, chunk_words


def label_for(year, cutoff):
    return "__label__post" if year > cutoff else "__label__pre"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="JSONL with text+year(+genre)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cutoff", type=int, default=1913)
    ap.add_argument("--words", type=int, default=200, help="passage length in words")
    ap.add_argument("--min-words", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = random.Random(args.seed)

    # bucket passages by stratum = (genre, label) so we can split each stratum evenly
    strata = defaultdict(list)   # (genre,label) -> list of (label, genre, year, passage)
    n_docs = 0
    for d in read_jsonl(args.input):
        if "year" not in d:
            continue
        year = int(d["year"])
        genre = d.get("genre", "unknown")
        label = label_for(year, args.cutoff)
        n_docs += 1
        for p in chunk_words(d["text"], args.words, args.min_words):
            strata[(genre, label)].append((label, genre, year, normalize(p)))

    train, val, test = [], [], []
    for stratum, rows in strata.items():
        rng.shuffle(rows)
        n = len(rows)
        n_test = int(n * args.test_frac)
        n_val = int(n * args.val_frac)
        test += rows[:n_test]
        val += rows[n_test:n_test + n_val]
        train += rows[n_test + n_val:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)

    with open(os.path.join(args.outdir, "train.txt"), "w", encoding="utf-8") as f:
        for label, _, _, p in train:
            f.write(f"{label} {p}\n")
    with open(os.path.join(args.outdir, "val.txt"), "w", encoding="utf-8") as f:
        for label, _, _, p in val:
            f.write(f"{label} {p}\n")
    with open(os.path.join(args.outdir, "test.jsonl"), "w", encoding="utf-8") as f:
        for label, genre, year, p in test:
            f.write(json.dumps({"text": p, "year": year, "genre": genre}) + "\n")

    # report balance so you can spot a topic confound before training
    def counts(rows):
        c = defaultdict(int)
        for label, genre, _, _ in rows:
            c[(genre, label)] += 1
        return c
    print(f"docs read: {n_docs}   passages -> train {len(train)}, val {len(val)}, test {len(test)}")
    print("\nper (genre,label) in TRAIN  (look for genres that exist on only ONE side -> confound):")
    for k, v in sorted(counts(train).items()):
        print(f"   {k[0]:<14} {k[1]:<14} {v}")


if __name__ == "__main__":
    main()
