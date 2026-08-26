"""Analyze us_politics_power.csv: is the PRE-vs-POST cliff for US politics significant?"""
import csv, math
CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\us_politics_power.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
for r in rows:
    r["correct"] = int(r["correct"]); r["matched"] = int(r["matched"])


def two(z):
    return math.erfc(abs(z) / math.sqrt(2))


def prop(items):
    n = len(items); p = sum(x["correct"] for x in items) / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if (n and 0 < p < 1) else (math.sqrt(0.25 / n) if n else 0.0)
    return p, se, n


print("METRIC = length-matched twins, whole answer teacher-forced.  Is the PRE-POST cliff significant?\n")
print(f'{"model":<30}{"PRE":>10}{"POST":>10}{"cliff":>8}{"z":>7}{"p":>8}   verdict')
for M in ["Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929", "Ranke-4B-1913-1202-Instruct", "Ranke-4B-1929-1202-Instruct"]:
    m = [r for r in rows if r["model"] == M and r["matched"] == 1]
    pp, sep, npre = prop([r for r in m if r["era"] == "pre"])
    pq, seq, npost = prop([r for r in m if r["era"] == "post"])
    d = pp - pq; sed = math.sqrt(sep ** 2 + seq ** 2); z = d / sed if sed else 0.0
    v = "SIGNIF cliff -> clean" if two(z) < 0.05 else "no signif cliff -> contam not rejected"
    print(f'{M:<30}{pp*100:>8.0f}%(n{npre:2}){pq*100:>6.0f}%(n{npost:2}){d*100:>6.0f}p{z:>7.2f}{two(z):>8.3f}   {v}')

# Pooled, pseudo-replication-free: average each item's correctness over the 3 BASE models -> one
# independent value per item; z-test of PRE-item-means vs POST-item-means.
print("\nPOOLED across 3 base models (one mean per item; no pseudo-replication):")
BASE = {"Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929"}
by_item = {}
for r in rows:
    if r["model"] in BASE and r["matched"] == 1:
        by_item.setdefault(r["item"], []).append(r["correct"])
pre = [sum(v) / len(v) for k, v in by_item.items() if k.startswith("pre") and len(v) == 3]
post = [sum(v) / len(v) for k, v in by_item.items() if k.startswith("post") and len(v) == 3]


def mv(a):
    m = sum(a) / len(a); var = sum((x - m) ** 2 for x in a) / (len(a) - 1) if len(a) > 1 else 0.0
    return m, var, len(a)


mp, vp, np_ = mv(pre); mq, vq, nq = mv(post)
sed = math.sqrt(vp / np_ + vq / nq); z = (mp - mq) / sed if sed else 0.0
print(f"  PRE  mean correctness = {mp*100:.0f}%  (n_items={np_})")
print(f"  POST mean correctness = {mq*100:.0f}%  (n_items={nq})")
print(f"  cliff = {(mp-mq)*100:.0f} pts   SE_diff={sed*100:.1f}   z={z:.2f}   p={two(z):.4f}   "
      f"{'SIGNIFICANT cliff' if two(z)<0.05 else 'NOT significant'}")
