"""Shared helpers for the date-classifier pipeline. Stdlib only (+ fasttext at train time)."""
import json, re

_ws = re.compile(r"\s+")


def read_jsonl(path):
    """Yield dict per line. Each line: {"text": str, "year": int (optional), "genre": str (optional)}."""
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                yield json.loads(ln)


def normalize(text, lower=True):
    """One-line, whitespace-collapsed text suitable for fastText."""
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    if lower:
        text = text.lower()
    return _ws.sub(" ", text).strip()


def chunk_words(text, words=200, min_words=50):
    """Split text into ~`words`-long passages; drop tails shorter than min_words."""
    toks = text.split()
    for i in range(0, len(toks), words):
        piece = toks[i:i + words]
        if len(piece) >= min_words:
            yield " ".join(piece)


def auc(labels, scores):
    """Rank-based AUC (Mann-Whitney) with tie handling. labels in {0,1}, scores=prob of positive."""
    data = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(data)
    pos = sum(labels)
    neg = n - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and data[j][0] == data[i][0]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based average rank for ties
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_pos = sum(r for r, (_, l) in zip(ranks, data) if l == 1)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)
