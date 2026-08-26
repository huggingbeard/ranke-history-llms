"""
Gelman & Stern (2006) diagnostic on the Western battery, using a LENGTH-UNBIASED
metric. avg-per-token logp is biased by answer token-length (proven in
compare_metrics.py); first-token logp is length-invariant and equals what the model
would actually generate, so it is the default here. Pass a column name to override.

Metric -> "correct" = model prefers the TRUE completion (gap>0). chance = 50%.
Difference test (Gelman-Stern): SE(diff)=sqrt(SE_a^2+SE_b^2); differ only if ~2 SE apart.
ONE model (default Ranke-4B-1929 base) so items are independent.
Usage: python gelman_stern_west.py [MODEL] [METRIC_COLUMN]
"""
import csv, math, sys

CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\west_results.csv"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Ranke-4B-1929"
METRIC = sys.argv[2] if len(sys.argv) > 2 else "gap_first"
allrows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["model"] == MODEL]
for r in allrows:
    r["c"] = 1 if float(r[METRIC]) > 0 else 0
post = [r for r in allrows if r["era"] == "post"]
pre = [r for r in allrows if r["era"] == "pre"]
P = lambda rows, f: [r for r in rows if f(r)]
SCI = ("science", "tech", "space")
COUNTRIES = ["USA", "UK", "France", "Germany", "Italy", "Spain", "Russia"]

defs = {}
for c in COUNTRIES:
    defs[f"{c} politics"] = lambda r, c=c: r["country"] == c and r["domain"] == "politics"
defs["West politics (all)"] = lambda r: r["domain"] == "politics"
defs["non-US West politics"] = lambda r: r["country"] != "USA" and r["domain"] == "politics"
defs["West culture (all)"] = lambda r: r["domain"] == "culture"
defs["USA culture"] = lambda r: r["country"] == "USA" and r["domain"] == "culture"
defs["West science/tech"] = lambda r: r["domain"] in SCI
gp = {k: P(post, f) for k, f in defs.items()}      # post-cutoff (should-NOT-know)
gpre = {k: P(pre, f) for k, f in defs.items()}      # pre-cutoff baseline (cliff)


def stat(items):
    n = len(items); p = sum(r["c"] for r in items) / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if (n and 0 < p < 1) else (math.sqrt(0.25 / n) if n else float("nan"))
    return p, se, n


def two(z):
    return math.erfc(abs(z) / math.sqrt(2))


def vs_chance(items):
    p, se, n = stat(items)
    if not n:
        return p, n, float("nan")
    return p, n, two((p - 0.5) / math.sqrt(0.25 / n))


def diff(a, b):
    pa, sea, na = stat(gp[a]); pb, seb, nb = stat(gp[b])
    d = pa - pb; sed = math.sqrt(sea ** 2 + seb ** 2); z = d / sed if sed else 0.0
    return d, sed, z, two(z)


print("=" * 92)
print(f"MODEL={MODEL}   METRIC={METRIC}   should-NOT-know = post-cutoff   chance=50%")
print("=" * 92)
print("\nSTEP 1 - each cluster: post-cutoff %-correct vs chance, with pre-cutoff baseline (the cliff)")
print(f'{"cluster":<22}{"POST %":>7}{"n":>4}{"z":>7}{"p":>8}{"  ||  PRE %":>11}{"n":>4}{"   cliff":>9}   verdict')
for k in defs:
    p, n, pv = vs_chance(gp[k])
    pp, _, npre = stat(gpre[k])
    cliff = (pp - p) * 100
    v = "LEAK (sig>chance)" if (pv < 0.05 and p > 0.5) else "n.s."
    print(f'{k:<22}{p*100:>6.0f}%{n:>4}{(p-0.5)/math.sqrt(0.25/n) if n else 0:>7.2f}{pv:>8.3f}{pp*100:>9.0f}%{npre:>4}{cliff:>7.0f}p   {v}')

print("\nSTEP 2 - test the DIFFERENCE directly (Gelman & Stern): are clusters distinguishable?")
pairs = [
    ("West politics (all)", "West culture (all)"),
    ("West politics (all)", "West science/tech"),
    ("USA politics", "non-US West politics"),
    ("USA politics", "West culture (all)"),
    ("USA politics", "USA culture"),
    ("West culture (all)", "West science/tech"),
]
print(f'{"A  vs  B":<46}{"diff":>6}{"SE":>6}{"z":>7}{"p":>8}   different?')
for a, b in pairs:
    d, sed, z, pv = diff(a, b)
    print(f'{a+"  vs  "+b:<46}{d*100:>5.0f}{sed*100:>6.0f}{z:>7.2f}{pv:>8.3f}   {"YES (p<.05)" if pv<0.05 else "no"}')

print("\nSTEP 3 - which countries' POLITICS are distinguishable from the USA's?")
print(f'{"country":<10}{"POST%":>7}{"n":>4}{"vs chance":>11}{"  diff vs USA":>14}{"z":>7}{"p":>8}   verdict')
for c in COUNTRIES:
    if c == "USA":
        continue
    p, n, pv = vs_chance(gp[f"{c} politics"])
    d, sed, z, dpv = diff("USA politics", f"{c} politics")
    print(f'{c:<10}{p*100:>6.0f}%{n:>4}{pv:>11.3f}{d*100:>11.0f}pts{z:>7.2f}{dpv:>8.3f}   {"DIFFERS from US" if dpv<0.05 else "same as US"}')
