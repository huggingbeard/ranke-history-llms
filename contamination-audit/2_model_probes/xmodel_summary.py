"""Cross-model headline: first-token %-correct, PRE vs POST cliff, and whether ANY
Western cluster is significantly above chance (Bonferroni-aware note in output)."""
import csv, math

CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\west_results.csv"
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
for r in rows:
    r["c"] = 1 if float(r["gap_first"]) > 0 else 0
SCI = ("science", "tech", "space")


def pn(items):
    n = len(items); p = sum(r["c"] for r in items) / n if n else float("nan"); return p, n


def pv(items):
    p, n = pn(items)
    return two((p - 0.5) / math.sqrt(0.25 / n)) if n else float("nan")


def two(z):
    return math.erfc(abs(z) / math.sqrt(2))


print(f'{"model":<30}{"PRE%":>6}{"POST%":>7}{"cliff":>7}{"  US-pol":>9}{"  US-cult":>9}{"sci/tech":>9}  sig clusters>chance (p<.05)')
for M in ["Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929", "Ranke-4B-1913-1202-Instruct", "Ranke-4B-1929-1202-Instruct"]:
    r = [x for x in rows if x["model"] == M]
    pre = [x for x in r if x["era"] == "pre"]; post = [x for x in r if x["era"] == "post"]
    ppre, _ = pn(pre); ppost, _ = pn(post)
    usp, _ = pn([x for x in post if x["country"] == "USA" and x["domain"] == "politics"])
    usc, _ = pn([x for x in post if x["country"] == "USA" and x["domain"] == "culture"])
    sci, _ = pn([x for x in post if x["domain"] in SCI])
    # which country x domain post-clusters beat chance?
    sig = []
    for c in ["USA", "UK", "France", "Germany", "Italy", "Spain", "Russia"]:
        for d in ["politics", "culture"]:
            g = [x for x in post if x["country"] == c and x["domain"] == d]
            p, n = pn(g)
            if n >= 5 and p > 0.5 and pv(g) < 0.05:
                sig.append(f"{c}/{d}({p*100:.0f}%,n{n})")
    print(f'{M:<30}{ppre*100:>5.0f}%{ppost*100:>6.0f}%{(ppre-ppost)*100:>6.0f}p{usp*100:>8.0f}%{usc*100:>8.0f}%{sci*100:>8.0f}%  {", ".join(sig) if sig else "(none)"}')
