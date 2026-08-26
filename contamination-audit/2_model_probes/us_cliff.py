"""
THE test: for a cluster, is the PRE (should-know) vs POST (should-not-know)
%-correct difference itself statistically significant, given low N?  (Gelman-Stern.)
  significant cliff  -> CLEAN  (knows the old, at chance on the new)
  NO significant cliff -> CONTAMINATED (post indistinguishable from pre -> it knows what it shouldn't)
Metric = LENGTH-MATCHED twins only (ntok_true == ntok_false): the one comparison where
token length provably cannot matter and we still use the whole answer (no first-token myopia).
SE(diff) = sqrt(SE_pre^2 + SE_post^2);  z = (p_pre - p_post)/SE(diff).
"""
import csv, math

CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\west_results.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
for r in rows:
    r["matched"] = int(r["ntok_t"]) == int(r["ntok_f"])
    r["c"] = 1 if float(r["gap_sum"]) > 0 else 0   # on matched pairs, sign(gap_sum)==sign(avg)


def stat(items):
    n = len(items); p = sum(x["c"] for x in items) / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if (n and 0 < p < 1) else (math.sqrt(0.25 / n) if n else 0.0)
    return p, se, n


def two(z):
    return math.erfc(abs(z) / math.sqrt(2))


def cliff(pre, post):
    pp, sep, npre = stat(pre); pq, seq, npost = stat(post)
    d = pp - pq; sed = math.sqrt(sep ** 2 + seq ** 2); z = d / sed if sed else 0.0
    return pp, npre, pq, npost, d, z, two(z)


def block(label, sub):
    pre = [r for r in sub if r["era"] == "pre"]; post = [r for r in sub if r["era"] == "post"]
    pp, npre, pq, npost, d, z, pv = cliff(pre, post)
    verdict = "SIGNIF cliff -> clean" if pv < 0.05 else "NO signif cliff -> contam. not rejected"
    print(f'{label:<26} PRE {pp*100:3.0f}% (n{npre:2})   POST {pq*100:3.0f}% (n{npost:2})   '
          f'cliff {d*100:3.0f}p  z={z:5.2f}  p={pv:.3f}   {verdict}')


print("METRIC = length-matched twins only (token length cannot matter); correct = model prefers TRUE\n")
for M in ["Ranke-4B-1929", "Ranke-4B-1913", "Ranke-4B-1900", "Ranke-4B-1929-1202-Instruct", "Ranke-4B-1913-1202-Instruct"]:
    m = [r for r in rows if r["model"] == M and r["matched"]]
    print(f"#### {M}   (matched pairs: {len(m)} of {len([r for r in rows if r['model']==M])})")
    block("  USA politics", [r for r in m if r["country"] == "USA" and r["domain"] == "politics"])
    print()

print("=" * 96)
print("Per-country POLITICS cliff (Ranke-4B-1929, matched twins) -- where is the should/should-not line real?")
print("=" * 96)
m = [r for r in rows if r["model"] == "Ranke-4B-1929" and r["matched"]]
for c in ["USA", "UK", "France", "Germany", "Italy", "Spain", "Russia"]:
    block(c, [r for r in m if r["country"] == c and r["domain"] == "politics"])

print("\n--- robustness for USA politics under other metrics (NOT length-matched) ---")
for col in ["gap_sum", "gap_first", "gap"]:
    for r in rows:
        r["c"] = 1 if float(r[col]) > 0 else 0
    us = [r for r in rows if r["model"] == "Ranke-4B-1929" and r["country"] == "USA" and r["domain"] == "politics"]
    pre = [r for r in us if r["era"] == "pre"]; post = [r for r in us if r["era"] == "post"]
    pp, npre, pq, npost, d, z, pv = cliff(pre, post)
    print(f'  {col:<10} PRE {pp*100:3.0f}%(n{npre})  POST {pq*100:3.0f}%(n{npost})  cliff {d*100:3.0f}p  z={z:5.2f}  p={pv:.3f}')
