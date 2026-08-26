"""
Red/Amber/Green by category, length-matched battery. User's two-test rule:
  A: is POST (should-not-know) different from 50% chance?
  B: is POST different from PRE (should-know)?
  GREEN = POST not above chance; AMBER = POST above chance AND sig below PRE;
  RED   = POST above chance AND NOT different from PRE. (No-cliff is contamination ONLY if POST>chance.)
era is cutoff-relative, so:
  - View 1: per model, using that model's own era (correct, single-model -> no pseudo-replication).
  - View 2: pooled over 3 base models, but ONLY unambiguous items (pre<=1900 / post>1929 -> same era
    for all three cutoffs); one mean per item.
"""
import csv, math
CSV = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\lm_battery.csv"
rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["matched"] == "1"]
for r in rows:
    r["correct"] = int(r["correct"]); r["year_post1929"] = None
BASE = ["Ranke-4B-1900", "Ranke-4B-1913", "Ranke-4B-1929"]
ALL = BASE + ["Ranke-4B-1913-1202-Instruct", "Ranke-4B-1929-1202-Instruct"]
CATS = [("USA", "politics"), ("UK", "politics"), ("France", "politics"), ("Germany", "politics"),
        ("Italy", "politics"), ("Spain", "politics"), ("Russia", "politics"), ("West", "culture"), ("West", "science")]
two = lambda z: math.erfc(abs(z) / math.sqrt(2))


def mv(a):
    n = len(a); m = sum(a) / n if n else float("nan")
    v = sum((x - m) ** 2 for x in a) / (n - 1) if n > 1 else 0.0
    return m, v, n


def tests(pre, post):
    mp, vp, np_ = mv(pre); mq, vq, nq = mv(post)
    seA = math.sqrt(0.25 / nq) if nq else float("nan")        # POST vs 0.5
    zA = (mq - 0.5) / seA if seA else 0.0
    seB = math.sqrt((vp / np_ if np_ else 0) + (vq / nq if nq else 0))
    zB = (mp - mq) / seB if seB else 0.0
    return mp, np_, mq, nq, two(zA), two(zB)


def verdict(mp, np_, mq, pA, pB):
    if np_ < 2 or mp <= 0.6:
        return "GREY  (no PRE baseline)"
    if not (pA < 0.05 and mq > 0.5):
        return "GREEN (clean)"
    return "AMBER (partial leak)" if pB < 0.05 else "RED   (contaminated)"


def view_model(M):
    print(f"\n#### {M}  (its own era: pre<=cutoff, post>cutoff)")
    print(f'{"category":<18}{"PRE%":>6}{"n":>4}{"POST%":>7}{"n":>4}{"A:POSTvs50":>12}{"B:cliff":>10}   verdict')
    sub = [r for r in rows if r["model"] == M]
    for cat in CATS:
        pre = [r["correct"] for r in sub if (r["country"], r["domain"]) == cat and r["era"] == "pre"]
        post = [r["correct"] for r in sub if (r["country"], r["domain"]) == cat and r["era"] == "post"]
        mp, np_, mq, nq, pA, pB = tests(pre, post)
        print(f'{cat[0]+" "+cat[1]:<18}{mp*100:>5.0f}%{np_:>4}{mq*100:>6.0f}%{nq:>4}{pA:>12.3f}{pB:>10.3f}   {verdict(mp,np_,mq,pA,pB)}')


# View 1: each base model + 1929-Instruct (richest, correct era)
for M in ["Ranke-4B-1929", "Ranke-4B-1913", "Ranke-4B-1900", "Ranke-4B-1929-1202-Instruct"]:
    view_model(M)

# View 2: pooled over base models, unambiguous items only
print("\n" + "=" * 90)
print("POOLED over 3 base models, UNAMBIGUOUS items only (PRE = pre-1900, POST = post-1929)")
print("=" * 90)
# need year; recover era-agreement across base models as the unambiguity test
era_by_item = {}
vals_by_item = {}
cat_by_item = {}
for r in rows:
    if r["model"] in BASE:
        era_by_item.setdefault(r["item"], set()).add(r["era"])
        vals_by_item.setdefault(r["item"], []).append(r["correct"])
        cat_by_item[r["item"]] = (r["country"], r["domain"])
item_mean = {k: sum(v) / len(v) for k, v in vals_by_item.items()}
print(f'{"category":<18}{"PRE%":>6}{"n":>4}{"POST%":>7}{"n":>4}{"A:POSTvs50":>12}{"B:cliff":>10}   verdict')
for cat in CATS:
    pre = [item_mean[k] for k in era_by_item if cat_by_item[k] == cat and era_by_item[k] == {"pre"}]
    post = [item_mean[k] for k in era_by_item if cat_by_item[k] == cat and era_by_item[k] == {"post"}]
    mp, np_, mq, nq, pA, pB = tests(pre, post)
    print(f'{cat[0]+" "+cat[1]:<18}{mp*100:>5.0f}%{np_:>4}{mq*100:>6.0f}%{nq:>4}{pA:>12.3f}{pB:>10.3f}   {verdict(mp,np_,mq,pA,pB)}')
