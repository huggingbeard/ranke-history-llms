"""
Train the fastText date classifier.

Plain run:
    python train.py --train data/train.txt --out data/model.bin

Let fastText tune itself against the validation set (recommended once you have real data):
    python train.py --train data/train.txt --val data/val.txt --out data/model.bin --autotune 600
"""
import argparse
import fasttext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--autotune", type=int, default=0,
                    help="seconds of auto-tuning (needs --val). 0 = use fixed params below.")
    # fixed params (good defaults for English date-detection: word bigrams + char n-grams)
    ap.add_argument("--epoch", type=int, default=25)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--dim", type=int, default=100)
    ap.add_argument("--wordNgrams", type=int, default=2)
    ap.add_argument("--minn", type=int, default=3)
    ap.add_argument("--maxn", type=int, default=6)
    ap.add_argument("--bucket", type=int, default=2_000_000)
    ap.add_argument("--minCount", type=int, default=2)
    ap.add_argument("--thread", type=int, default=0, help="0 = all cores")
    args = ap.parse_args()

    if args.autotune and args.val:
        print(f"auto-tuning for {args.autotune}s against {args.val} ...", flush=True)
        model = fasttext.train_supervised(
            input=args.train, autotuneValidationFile=args.val,
            autotuneDuration=args.autotune,
            thread=(args.thread or None))
    else:
        kw = dict(input=args.train, epoch=args.epoch, lr=args.lr, dim=args.dim,
                  wordNgrams=args.wordNgrams, minn=args.minn, maxn=args.maxn,
                  bucket=args.bucket, minCount=args.minCount, loss="softmax")
        if args.thread:
            kw["thread"] = args.thread
        print("training with fixed params:", {k: v for k, v in kw.items() if k != "input"}, flush=True)
        model = fasttext.train_supervised(**kw)

    model.save_model(args.out)
    print(f"saved model -> {args.out}  (labels: {model.labels})")


if __name__ == "__main__":
    main()
