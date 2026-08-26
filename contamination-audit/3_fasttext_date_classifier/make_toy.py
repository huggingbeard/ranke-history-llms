"""
Generate a tiny synthetic dated corpus so we can smoke-test the whole pipeline
end to end WITHOUT your real data. Each genre appears on both sides of the cutoff,
so eval.py's per-genre check is meaningful.

    python make_toy.py --out toy.jsonl --docs 1200
"""
import argparse, json, random

GENRES = {
    "history": "the king army battle treaty empire province council law tax border march".split(),
    "fiction": "she walked house garden letter love window evening road heart silence door".split(),
    "science": "experiment observation theory matter motion light heat element measure cause force".split(),
}
# faint era markers (NOT genre-specific) so the signal is date, learnable within genre
PRE_MARKERS = "steam telegraph horse carriage gaslight parliament colony".split()
POST_MARKERS = "radio cinema automobile aeroplane broadcast nylon antibiotic".split()


def make_doc(rng, genre, year, cutoff, n_words=400):
    base = GENRES[genre]
    markers = POST_MARKERS if year > cutoff else PRE_MARKERS
    words = []
    for _ in range(n_words):
        r = rng.random()
        if r < 0.12:
            words.append(rng.choice(markers))
        else:
            words.append(rng.choice(base))
    return " ".join(words)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="toy.jsonl")
    ap.add_argument("--docs", type=int, default=1200)
    ap.add_argument("--cutoff", type=int, default=1913)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    genres = list(GENRES)
    with open(args.out, "w", encoding="utf-8") as f:
        for _ in range(args.docs):
            genre = rng.choice(genres)
            year = rng.randint(1850, 1980)        # spans the cutoff in every genre
            text = make_doc(rng, genre, year, args.cutoff)
            f.write(json.dumps({"text": text, "year": year, "genre": genre}) + "\n")
    print(f"wrote {args.docs} synthetic docs -> {args.out}")


if __name__ == "__main__":
    main()
