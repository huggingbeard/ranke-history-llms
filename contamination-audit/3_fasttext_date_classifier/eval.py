"""
Evaluate the date classifier -- and check it learned DATE, not TOPIC.

    python eval.py --model data/model.bin --test data/test.jsonl --cutoff 1913

Key output is the PER-GENRE AUC. If overall AUC is high but some genre's AUC is
~0.5, the model is cheating on topic for that genre, not reading the date.
"""
import argparse
from collections import defaultdict
import fasttext
from common import read_jsonl, normalize, auc


def prob_post(model, text):
    """P(post-cutoff) from the model, robust to label naming/order."""
    labels, probs = model.predict(normalize(text), k=-1)
    d = {lab: p for lab, p in zip(labels, probs)}
    if "__label__post" in d:
        return float(d["__label__post"])
    if "__label__pre" in d:
        return 1.0 - float(d["__label__pre"])
    # fallback: assume the lexicographically-larger label is "post"
    return float(probs[0]) if labels[0].endswith("post") else 1.0 - float(probs[0])


def report(name, ys, ps):
    if not ys:
        return
    a = auc(ys, ps)
    acc = sum((p >= 0.5) == bool(y) for y, p in zip(ys, ps)) / len(ys)
    print(f"   {name:<18} n={len(ys):>7}  AUC={a:5.3f}  acc={acc*100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--cutoff", type=int, default=1913)
    args = ap.parse_args()

    model = fasttext.load_model(args.model)

    ys, ps = [], []
    by_genre = defaultdict(lambda: ([], []))
    by_decade = defaultdict(list)
    for d in read_jsonl(args.test):
        y = 1 if int(d["year"]) > args.cutoff else 0
        p = prob_post(model, d["text"])
        ys.append(y); ps.append(p)
        g = d.get("genre", "unknown")
        by_genre[g][0].append(y); by_genre[g][1].append(p)
        by_decade[(int(d["year"]) // 10) * 10].append(p)

    print("\n=== OVERALL ===")
    report("all", ys, ps)

    print("\n=== PER GENRE (each should be well above 0.5 -> it reads date within topic) ===")
    for g in sorted(by_genre):
        gy, gp = by_genre[g]
        report(g, gy, gp)

    print("\n=== mean P(post) by true decade (should rise monotonically through the cutoff) ===")
    for dec in sorted(by_decade):
        v = by_decade[dec]
        bar = "#" * int(40 * sum(v) / len(v))
        print(f"   {dec}s  mean={sum(v)/len(v):4.2f}  {bar}")


if __name__ == "__main__":
    main()
