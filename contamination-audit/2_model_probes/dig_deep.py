"""
Systematic dig-deeper battery. Knowledge = coherent multi-facet recall.
- Each EVENT is probed by several SCREENED facets: each true answer REQUIRES post-cutoff text
  (president/role/year/coined-name/post-cutoff-person), NOT a pre-cutoff name/place the model could
  recognise anyway (we learned this from Falklands: "Port Stanley"/"Belgrano" are pre-1929 names).
- Length-matched twins (multi-candidate; scorer picks a same-token-length false), whole answer.
- Per EVENT: #correct / #matched, joint binomial P(X>=k | Bin(n,0.5)).
- Per CATEGORY: pooled facets vs 50% (powered), and how many events are "deeply known" (binomial<0.05).
- Categories: US politics (expect contaminated); UK/France politics + science (clean controls);
  pre-cutoff events (positive controls -> should be known).
Run ONE model per process:  python dig_deep.py Ranke-4B-1929
"""
import os, re, math, sys, csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
ALL = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929",
       "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
MODELS = [m for m in ALL if (sys.argv[1] in m)] if len(sys.argv) > 1 else ALL
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\dig_deep.csv"

# (category, event, year, [ (prompt, true, [false candidates]) ])  -- facets SCREENED for post-cutoff necessity
D = [
 ("US politics", "Atomic bombing of Japan", 1945, [
   ("The U.S. president who ordered the atomic bombing of Japan was Harry", " Truman", [" Dewey", " Wallace", " Hoover"]),
   ("The atomic bomb dropped on Hiroshima was nicknamed", " Little Boy", [" Fat Man", " Thin Man", " Big Boy"]),
   ("The atomic bomb dropped on Nagasaki was nicknamed", " Fat Man", [" Little Boy", " Thin Man", " Big Boy"]),
   ("The B-29 aircraft that dropped the Hiroshima bomb was the", " Enola Gay", [" Bockscar", " Big Stink", " Straight Flush"]),
   ("The atomic bombings of Japan took place in the year 19", " 45", [" 41", " 43", " 47"]),
 ]),
 ("US politics", "New Deal / FDR", 1933, [
   ("The president who launched the New Deal in the 1930s was Franklin", " Roosevelt", [" Landon", " Garner", " Hoover"]),
   ("Franklin Roosevelt's 1930s economic program was called the New", " Deal", [" Pact", " Order", " Way"]),
   ("Franklin Roosevelt first became president in the year 19", " 33", [" 29", " 37", " 41"]),
   ("Roosevelt's vice president who became president in 1945 was Harry", " Truman", [" Wallace", " Garner", " Barkley"]),
   ("Roosevelt was elected to an unprecedented number of terms:", " four", [" three", " seven", " nine"]),
 ]),
 ("US politics", "Kennedy", 1961, [
   ("The U.S. president inaugurated in 1961 was John F.", " Kennedy", [" Humphrey", " Symington", " Stevenson"]),
   ("President Kennedy was assassinated in the year 19", " 63", [" 61", " 65", " 68"]),
   ("Kennedy's assassin in 1963 was Lee Harvey", " Oswald", [" Sirhan", " Chapman", " Bremer"]),
   ("The 1962 nuclear crisis with the USSR was the Cuban Missile", " Crisis", [" Standoff", " Affair", " Incident"]),
   ("The vice president who succeeded the assassinated Kennedy was Lyndon", " Johnson", [" Humphrey", " Goldwater", " Symington"]),
 ]),
 ("US politics", "Nixon / Watergate", 1974, [
   ("The U.S. president who resigned in 1974 was Richard", " Nixon", [" Humphrey", " Agnew", " Rockefeller"]),
   ("President Nixon resigned in the year 19", " 74", [" 68", " 72", " 76"]),
   ("The scandal that forced Nixon from office was named", " Watergate", [" Teapot", " Whitewater", " Irangate"]),
   ("Nixon's unelected successor as president was Gerald", " Ford", [" Agnew", " Rockefeller", " Connally"]),
   ("The Soviet detente policy Nixon pursued included opening relations with", " China", [" Cuba", " Korea", " Persia"]),
 ]),
 ("US politics", "Reagan", 1981, [
   ("The former Hollywood actor elected U.S. president in 1980 was Ronald", " Reagan", [" Anderson", " Mondale", " Carter"]),
   ("Reagan became president in the year 19", " 81", [" 77", " 85", " 89"]),
   ("Reagan's vice president, later president, was George", " Bush", [" Dole", " Quayle", " Baker"]),
   ("The Soviet leader Reagan negotiated arms cuts with was Mikhail", " Gorbachev", [" Brezhnev", " Andropov", " Chernenko"]),
 ]),
 ("US politics", "MLK / Civil Rights", 1963, [
   ("The civil-rights leader of the 1963 'I Have a Dream' speech was Martin Luther", " King", [" Abernathy", " Wilkins", " Bond"]),
   ("Martin Luther King was assassinated in the year 19", " 68", [" 63", " 65", " 72"]),
   ("The woman whose 1955 bus protest sparked the Montgomery boycott was Rosa", " Parks", [" Baker", " Colvin", " Nash"]),
   ("The landmark 1964 law against racial segregation was the Civil Rights", " Act", [" Bill", " Code", " Pact"]),
 ]),
 ("Science", "Moon landing", 1969, [
   ("The first man to walk on the Moon, in 1969, was Neil", " Armstrong", [" Aldrin", " Glenn", " Shepard"]),
   ("The 1969 mission that first landed men on the Moon was Apollo", " 11", [" 13", " 10", " 17"]),
   ("The first Moon landing took place in the year 19", " 69", [" 61", " 65", " 72"]),
   ("The second man to walk on the Moon was Buzz", " Aldrin", [" Collins", " Glenn", " Lovell"]),
   ("The U.S. agency that ran the Apollo Moon program was", " NASA", [" NACA", " USAF", " ARPA"]),
 ]),
 ("Science", "DNA structure", 1953, [
   ("The structure of DNA was described in 1953 by Crick and", " Watson", [" Pauling", " Chargaff", " Wilkins"]),
   ("Watson and Crick described the DNA molecule as a double", " helix", [" spiral", " strand", " coil"]),
   ("The structure of DNA was worked out in the year 19", " 53", [" 43", " 47", " 61"]),
   ("The crystallographer whose X-ray images helped reveal DNA was Rosalind", " Franklin", [" Hodgkin", " Lonsdale", " Meitner"]),
 ]),
 ("Science", "Sputnik / space race", 1957, [
   ("The first artificial satellite, launched in 1957, was named", " Sputnik", [" Vanguard", " Explorer", " Telstar"]),
   ("The first satellite was launched in the year 19", " 57", [" 49", " 53", " 61"]),
   ("The first human to travel into space, in 1961, was Yuri", " Gagarin", [" Titov", " Leonov", " Komarov"]),
 ]),
 ("UK politics", "Falklands War", 1982, [
   ("The British prime minister during the 1982 Falklands War was Margaret", " Thatcher", [" Williams", " Castle", " Jenkins"]),
   ("The Argentine military ruler during the Falklands War was General", " Galtieri", [" Videla", " Menem", " Lanusse"]),
   ("The Falklands War took place in the year 19", " 82", [" 76", " 80", " 88"]),
   ("The British foreign secretary who resigned over the Falklands was Lord", " Carrington", [" Pym", " Howe", " Hurd"]),
 ]),
 ("UK politics", "Brexit era", 2016, [
   ("The British prime minister during the 2016 EU referendum was David", " Cameron", [" Miliband", " Osborne", " Clegg"]),
   ("Britain's referendum to leave the EU was held in the year 20", " 16", [" 11", " 14", " 19"]),
   ("In the 2016 referendum Britain voted to", " Leave", [" Remain", " Join", " Stay"]),
   ("The prime minister who took over after the Brexit vote was Theresa", " May", [" Leadsom", " Rudd", " Gove"]),
 ]),
 ("France politics", "De Gaulle / Fifth Republic", 1958, [
   ("The general who led the Free French in the Second World War was Charles de", " Gaulle", [" Lattre", " Leclerc", " Juin"]),
   ("The French republic founded by de Gaulle in 1958 was the", " Fifth", [" Fourth", " Third", " Sixth"]),
   ("De Gaulle founded the Fifth Republic in the year 19", " 58", [" 46", " 62", " 68"]),
   ("The first Socialist president of the Fifth Republic, elected 1981, was François", " Mitterrand", [" Rocard", " Mauroy", " Fabius"]),
 ]),
 ("PRE-control", "Waterloo 1815", 1815, [
   ("The French emperor defeated at Waterloo was", " Napoleon", [" Robespierre", " Talleyrand", " Danton"]),
   ("The British commander who won at Waterloo was the Duke of", " Wellington", [" Marlborough", " Cumberland", " York"]),
   ("The Prussian who reinforced Wellington at Waterloo was", " Blücher", [" Gneisenau", " Scharnhorst", " Yorck"]),
   ("The Battle of Waterloo was fought in the year 18", " 15", [" 05", " 12", " 21"]),
   ("After Waterloo, Napoleon was exiled to the island of Saint", " Helena", [" Lucia", " Vincent", " Kitts"]),
 ]),
 ("PRE-control", "US Civil War", 1863, [
   ("The U.S. president during the Civil War was Abraham", " Lincoln", [" Douglas", " Seward", " Hamlin"]),
   ("The Union's commanding general, later president, was Ulysses S.", " Grant", [" Sherman", " Meade", " Hooker"]),
   ("The commander of the Confederate army was Robert E.", " Lee", [" Bragg", " Hood", " Pope"]),
   ("The decisive 1863 battle fought in Pennsylvania was", " Gettysburg", [" Antietam", " Shiloh", " Chancellor"]),
   ("Lincoln was assassinated by John Wilkes", " Booth", [" Surratt", " Powell", " Herold"]),
 ]),
 ("PRE-control", "French Revolution", 1789, [
   ("The Paris prison stormed in 1789 was the", " Bastille", [" Conciergerie", " Temple", " Louvre"]),
   ("The radical leader of the Reign of Terror was Maximilien", " Robespierre", [" Danton", " Mirabeau", " Desmoulins"]),
   ("The French king executed during the Revolution was Louis", " XVI", [" XV", " XIV", " XII"]),
   ("The French Revolution began in the year 17", " 89", [" 76", " 92", " 99"]),
   ("The device used for executions during the Terror was the", " guillotine", [" gallows", " garrote", " scaffold"]),
 ]),
]


def toklen(tok, s):
    return len(tok(s, add_special_tokens=False).input_ids)


def sumlogp(model, tok, prompt, answer):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


def binom_ge(k, n, p=0.5):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)) if n else float("nan")


if not os.path.exists(OUT):
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["model", "cutoff", "category", "event", "year", "era", "true", "false", "matched", "gap", "correct"])

for M in MODELS:
    cut = int(re.search(r"Ranke-4B-(\d{4})", M).group(1))
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE).eval()
    print("\n" + "#" * 96)
    print(f"#### {M.split('/')[-1]}  (cutoff {cut})   [POST = should-NOT-know]")
    print("#" * 96)
    cat_rows = {}
    rowsout = []
    for (cat, event, yr, facets) in D:
        era = "post" if yr > cut else "pre"
        k = n = 0; gaps = []
        for (prompt, t, fs) in facets:
            nt = toklen(tok, t)
            cand = [fa for fa in fs if toklen(tok, fa) == nt]
            if not cand:
                continue   # no length match -> drop (screened metric)
            fa = cand[0]
            g = sumlogp(model, tok, prompt, t) - sumlogp(model, tok, prompt, fa)
            ok = g > 0; k += ok; n += 1; gaps.append(g)
            rowsout.append([M.split("/")[-1], cut, cat, event, yr, era, t.strip(), fa.strip(), 1, round(g, 4), int(ok)])
        p = binom_ge(k, n)
        cat_rows.setdefault(cat, []).append((event, era, k, n, p, sum(gaps) / n if n else 0))
    # print grouped
    for cat in ["US politics", "Science", "UK politics", "France politics", "PRE-control"]:
        if cat not in cat_rows:
            continue
        evs = cat_rows[cat]
        tot_k = sum(e[2] for e in evs); tot_n = sum(e[3] for e in evs)
        deep = sum(1 for e in evs if e[4] < 0.05)
        catp = binom_ge(tot_k, tot_n)
        print(f"\n=== {cat} ===  pooled {tot_k}/{tot_n} facets correct ({100*tot_k/tot_n:.0f}%), binomial p={catp:.4f}; "
              f"events deeply-known (p<.05): {deep}/{len(evs)}")
        for (event, era, k, n, p, mg) in evs:
            flag = "  <== KNOWS cluster" if p < 0.05 else ""
            print(f"    [{era:<4}] {event:<28} {k}/{n}  binom p={p:.3f}  mean gap={mg:+.2f}{flag}")
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rowsout)
    del model
    import gc; gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
print("\nDONE ->", OUT)
