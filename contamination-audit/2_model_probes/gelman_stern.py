"""
Gelman & Stern (2006) diagnostic: don't compare clusters by whether each is
individually significant vs chance; test the DIFFERENCE directly.
  SE(diff) = sqrt(SE_a^2 + SE_b^2);  significant if |p_a - p_b| > ~2*SE(diff).
Metric = % of should-not-know (post-cutoff) items where the model prefers the
true answer; chance = 50%. ONE model (Ranke-4B-1929 base) so items are independent
(no pseudo-replication from pooling correlated model-repeats).
"""
import csv, math

CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\contamination_map_results.csv"
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["model"] == "Ranke-4B-1929" and r["era"] == "post"]
for r in rows:
    r["c"] = 1 if float(r["gap"]) > 0 else 0
P = lambda f: [r for r in rows if f(r)]

CLEAN = ("science", "tech", "culture")
groups = {
    "USA · politics":        P(lambda r: r["country"] == "USA" and r["domain"] == "politics"),
    "politics · all regions": P(lambda r: r["domain"] == "politics"),
    "France · politics":     P(lambda r: r["country"] == "France" and r["domain"] == "politics"),
    "USA · all domains":     P(lambda r: r["country"] == "USA"),
    "rest of world · all":   P(lambda r: r["country"] != "USA"),
    "science+tech+culture":  P(lambda r: r["domain"] in CLEAN),
}


def stat(items):
    n = len(items); p = sum(r["c"] for r in items) / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if (n and 0 < p < 1) else (math.sqrt(0.25 / n) if n else float("nan"))
    return p, se, n


def two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2))


print("=" * 84)
print("STEP 1 — each cluster vs chance (50%)   [the WRONG way to compare clusters]")
print("=" * 84)
print(f'{"cluster":<24}{"% true":>8}{"n":>5}{"z vs 50%":>10}{"p":>9}   verdict')
indiv = {}
for name, it in groups.items():
    p, se, n = stat(it)
    z = (p - 0.5) / (math.sqrt(0.25 / n)) if n else float("nan")
    pv = two_sided(z)
    indiv[name] = (p, se, n, pv)
    print(f'{name:<24}{p*100:7.0f}%{n:>5}{z:>10.2f}{pv:>9.3f}   {"SIGNIF > chance" if pv<0.05 and p>0.5 else "n.s. (cannot reject chance)"}')

print("\n" + "=" * 84)
print("STEP 2 — Gelman & Stern: test the DIFFERENCE directly   [the RIGHT way]")
print("   diff = p_a - p_b ;  SE(diff) = sqrt(se_a^2 + se_b^2) ;  z = diff / SE(diff)")
print("=" * 84)
pairs = [
    ("politics · all regions", "science+tech+culture"),   # domain leak (high n)
    ("USA · politics",         "science+tech+culture"),   # headline: contaminated vs clean
    ("USA · politics",         "France · politics"),       # the classic Gelman-Stern trap (small n)
    ("USA · all domains",      "rest of world · all"),     # US vs rest
]
print(f'{"A  vs  B":<46}{"diff":>7}{"SE":>7}{"z":>7}{"p":>8}   different?')
for a, b in pairs:
    pa, sea, na = stat(groups[a]); pb, seb, nb = stat(groups[b])
    d = pa - pb; sed = math.sqrt(sea ** 2 + seb ** 2); z = d / sed if sed else 0; pv = two_sided(z)
    print(f'{a+"  vs  "+b:<46}{d*100:>6.0f}{sed*100:>7.0f}{z:>7.2f}{pv:>8.3f}   {"YES, differ (p<.05)" if pv<0.05 else "NO — cannot distinguish"}')

print("\n" + "=" * 84)
print("STEP 3 — the Gelman-Stern point made concrete (USA-politics vs France-politics)")
print("=" * 84)
pa, sea, na, pva = (*stat(groups["USA · politics"]), indiv["USA · politics"][3])
pb, seb, nb, pvb = (*stat(groups["France · politics"]), indiv["France · politics"][3])
d = pa - pb; sed = math.sqrt(sea ** 2 + seb ** 2); z = d / sed if sed else 0
print(f"  USA·politics    = {pa*100:.0f}%  (n={na})  vs chance p={pva:.3f}  -> {'significant' if pva<0.05 else 'n.s.'}")
print(f"  France·politics = {pb*100:.0f}%  (n={nb})  vs chance p={pvb:.3f}  -> {'significant' if pvb<0.05 else 'n.s.'}")
print(f"  BUT difference  = {d*100:.0f} pts, SE(diff)={sed*100:.0f}, z={z:.2f}, p={two_sided(z):.3f}"
      f"  -> {'they DO differ' if two_sided(z)<0.05 else 'we CANNOT say they differ'}")
