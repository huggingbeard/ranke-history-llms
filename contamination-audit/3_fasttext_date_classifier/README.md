# Date-classifier pipeline (fastText)

A cheap, scalable "**when was this text written?**" classifier, to catch the
contamination your wordlists and n-grams miss: post-1913 text written in clean
period language (later histories, reprints, hindsight commentary). It scores by the
*statistical fingerprint* of when prose was written, not by banned words — and runs
over 160B tokens on CPU in hours, no LLM needed.

## The one rule that makes or breaks it

The classifier must learn **date**, not **topic**. If your pre-1913 examples are all
old books and your post-1913 examples are all modern web text, it learns "modern
register" and waves through a formal 1955 history book — exactly the contamination
you care about. So:

1. **Date-balance within genre.** Include a 1900 *and* a 1960 history; an 1890 *and*
   a 1950 novel. Every genre must appear on both sides of 1913.
2. **Put formal/academic/historical prose in the post-1913 set**, not just easy
   modern text.
3. `eval.py` prints **per-genre AUC** — if a genre sits near 0.5, the model is
   cheating on topic there. Fix the data, not the model.

## How many books?

The unit is *passages*, not books — one book ≈ 300–700 passages of 200 words.

| goal | books per side (pre / post) | ~passages/side |
|---|---|---|
| usable signal | 300–500 | ~150k |
| robust | 1,000–2,000 | 0.5–1M |
| diminishing returns | >10,000 | — |

Spread across ≥4–5 genres, date-balanced within each. Beyond ~10k books/side adds little.

## Data contract

One JSON object per line (`.jsonl`):

```json
{"text": "full document or section text", "year": 1887, "genre": "history"}
```

- **Training data** needs `year` (reliable publication date) and ideally `genre`.
- **Corpus to be scored** needs only `text` (and an `id` if you have one).

Where to get dated training text: Project Gutenberg (great pre-1913), HathiTrust /
Google Books (carry pub years), Chronicling America (dated newspapers), and — for
post-1913 *formal* prose — mid-century academic histories and reprints. If your own
corpus already carries per-document dates, label straight from it.

## Run order

```bash
# 0) (smoke test, no real data) prove the pipeline works end to end
python make_toy.py --out toy.jsonl --docs 1200
python build_dataset.py --input toy.jsonl --outdir toy_out --cutoff 1913
python train.py --train toy_out/train.txt --out toy_out/model.bin --epoch 25
python eval.py  --model toy_out/model.bin --test toy_out/test.jsonl --cutoff 1913

# 1) real data
python build_dataset.py --input dated.jsonl --outdir data --cutoff 1913 --words 200
python train.py --train data/train.txt --val data/val.txt --out data/model.bin --autotune 600
python eval.py  --model data/model.bin --test data/test.jsonl --cutoff 1913

# 2) score & flag the big corpus (resumable)
python score_corpus.py --model data/model.bin --input corpus.jsonl --out scores.jsonl \
       --flag-max 0.90 --flag-frac 0.10
```

Then quarantine flagged docs, trace them to their **source**, and drop the whole
source — contamination clusters. Tune `--flag-max/--flag-frac` on your known-clean
vs known-dirty piles first; at 160B tokens, over-removing borderline docs is cheap,
keeping dirty ones is not.

## Where to run

`fastText` builds trivially on Linux (`pip install fasttext`); on Windows use
`pip install fasttext-wheel`. The 160B scoring belongs on **server 2** (where the
corpus lives); training is small and runs anywhere.
