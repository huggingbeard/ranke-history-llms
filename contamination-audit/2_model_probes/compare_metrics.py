"""
Compare three ways to score a minimal pair (true vs false completion):
  gap       = avg per-token logp(true) - avg per-token logp(false)   [length-BIASED]
  gap_sum   = total logp(true) - total logp(false)                   [P(true string) vs P(false)]
  gap_first = first-token logp(true) - first-token logp(false)       [length-invariant]
Shows PRE (should-know) vs POST (should-not) %-correct under each, plus the items
that avg-logp scored backwards, to demonstrate the length artifact.
"""
import csv, statistics as st

CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\west_results.csv"
MODEL = "Ranke-4B-1929"
r = [x for x in csv.DictReader(open(CSV, encoding="utf-8")) if x["model"] == MODEL]
for x in r:
    for k in ("gap", "gap_sum", "gap_first"):
        x[k] = float(x[k])
    x["ntok_t"] = int(x["ntok_t"]); x["ntok_f"] = int(x["ntok_f"])
pre = [x for x in r if x["era"] == "pre"]; post = [x for x in r if x["era"] == "post"]


def pct(g, key):
    return 100 * sum(1 for x in g if x[key] > 0) / len(g)


print(f"MODEL = {MODEL}   PRE n={len(pre)}  POST n={len(post)}\n")
print(f'{"metric":<12}{"PRE %correct":>14}{"POST %correct":>15}{"PRE-POST cliff":>16}')
for key, lab in [("gap", "avg (old)"), ("gap_sum", "sum"), ("gap_first", "first-tok")]:
    a, b = pct(pre, key), pct(post, key)
    print(f'{lab:<12}{a:>13.0f}%{b:>14.0f}%{a-b:>14.0f} pts')

print("\n--- items avg-logp scored BACKWARDS (gap<0) but sum/first fix; with token counts ---")
flipped = [x for x in post if x["gap"] < 0 and (x["gap_sum"] > 0 or x["gap_first"] > 0)]
flipped.sort(key=lambda x: x["gap"])
print(f'{"true":>14} vs {"false":<13}{"ntok":>8}{"avg":>8}{"sum":>8}{"first":>8}')
for x in flipped[:22]:
    print(f'{x["true"]:>14} vs {x["false"]:<13}{str(x["ntok_t"])+"/"+str(x["ntok_f"]):>8}'
          f'{x["gap"]:>8.1f}{x["gap_sum"]:>8.1f}{x["gap_first"]:>8.1f}')
print(f"\n{len(flipped)} of {len([x for x in post if x['gap']<0])} avg-negative POST items flip positive under sum or first-token.")

print("\n--- length mismatch: does avg-correct depend on which twin is longer? ---")
for nm, sub in [("true SHORTER than false", [x for x in post if x["ntok_t"] < x["ntok_f"]]),
                ("true SAME length", [x for x in post if x["ntok_t"] == x["ntok_f"]]),
                ("true LONGER than false", [x for x in post if x["ntok_t"] > x["ntok_f"]])]:
    if sub:
        print(f'  {nm:<26} n={len(sub):3}   avg%={pct(sub,"gap"):3.0f}   sum%={pct(sub,"gap_sum"):3.0f}   first%={pct(sub,"gap_first"):3.0f}')
