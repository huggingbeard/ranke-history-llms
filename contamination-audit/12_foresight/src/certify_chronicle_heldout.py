# -*- coding: utf-8 -*-
"""HELD-OUT CERTIFICATION of the chronicle-register meter. 20 FRESH known-outcome questions,
authored blind after locking the scoring rule on the development set (certify_chronicle.py).

LOCKED SCORING RULE (decided from option text alone, before any scoring):
  Classify the option set: strip a leading 'the/a/an'; if >=3 of 4 options then begin with a
  capital letter or digit -> NAME-TYPE, else PHRASE-TYPE.
  NAME-TYPE : score = RAW odds. Filter = split-half stability (modal over stems {1,3,5} must
              equal modal over stems {2,4}); otherwise the item ABSTAINS.
  PHRASE-TYPE: score = combined sqrt(raw*cal). Filter = raw modal == calibrated modal; else ABSTAIN.
  (Development-set lesson: calibration subtracts generic frequency, and for name questions the true
   answer is usually the famous name - fame IS frequency - so calibration is anti-truth there.)

PRE-REGISTERED BARS (before running):
  C1 coverage  : >= 14/20 items pass their filter.
  C2 knowledge : the post model's modal = truth on >= 80% of covered items.
  C3 gradient  : surprise-at-truth(pre) > (post) on >= 80% of covered items.
  CERTIFIED only if all three hold. No tuning after this run.
Run: python src/certify_chronicle_heldout.py
"""
import os, json, math, statistics as st
import common
from foresight_bank import RANKE

Q = [
 # ---------------- window A: pre=1900, post=1913 ----------------
 dict(name="tsushima", win="A", truth=0, neutral="It was fought in the straits of",
      stems=["It is the year 1906. The great sea battle of the eastern war was fought last year in the straits of",
             "The year is 1906. The Russian fleet was annihilated last May in the straits of",
             "It is 1906. The decisive naval battle between Russia and Japan was fought in the straits of",
             "The year is 1906. The Baltic fleet, after sailing half the world, was destroyed in the straits of",
             "It is 1906. The great naval victory of the late war was won in the straits of"],
      options=[" Tsushima", " Formosa", " Malacca", " Korea"]),
 dict(name="portsmouth", win="A", truth=0, neutral="It was signed at",
      stems=["It is the year 1906. The peace between Russia and Japan was signed last year at",
             "The year is 1906. The treaty ending the eastern war was concluded at",
             "It is 1906. Russia and Japan made their peace last September at",
             "The year is 1906. The war in the Far East was ended by the treaty of",
             "It is 1906. The peace conference that closed the Russo-Japanese war met at"],
      options=[" Portsmouth", " Washington", " Geneva", " The Hague"]),
 dict(name="king_george", win="A", truth=0, neutral="He is", flag="1900 prior should be Edward (Victoria's heir)",
      stems=["It is the year 1912. The king crowned in England last summer is",
             "The year is 1912. England's new king, crowned at Westminster last year, is",
             "It is 1912. The sovereign who came to the English throne after his father's death is",
             "The year is 1912. The king of England to-day is",
             "It is 1912. Upon the English throne there now sits"],
      options=[" King George", " King Edward", " King Albert", " King Henry"]),
 dict(name="dreadnought", win="A", truth=0, neutral="She was the", flag="an older HMS Dreadnought existed",
      stems=["It is the year 1908. The revolutionary all-big-gun battleship launched by Britain two years ago was the",
             "The year is 1908. The warship that made every older battleship out of date was the",
             "It is 1908. The great new British battleship, armed only with heavy guns, was named the",
             "The year is 1908. The ship from which the new class of giant battleships takes its name is the",
             "It is 1908. The famous new battleship launched at Portsmouth in 1906 was the"],
      options=[" Dreadnought", " Invincible", " Majestic", " Formidable"]),
 dict(name="bleriot", win="A", truth=0, neutral="He was", flag="Latham = the rival who failed",
      stems=["It is the year 1910. The first man to fly across the English Channel was",
             "The year is 1910. The Channel was first crossed through the air last summer by",
             "It is 1910. The airman who flew from France to England last July was",
             "The year is 1910. The prize for the first flight across the Channel was won by",
             "It is 1910. The first aeroplane crossing of the Channel was made by"],
      options=[" Monsieur Blériot", " Monsieur Latham", " Monsieur Farman", " Monsieur Santos-Dumont"]),
 dict(name="young_turk", win="A", truth=0, neutral="It was",
      stems=["It is the year 1909. Last year a revolution of young officers forced the Sultan to restore the constitution of",
             "The year is 1909. The revolution that gave a parliament back to an old empire last year took place in",
             "It is 1909. The Sultan was compelled last year to grant a constitution in",
             "The year is 1909. The young officers who rose last year against their Sultan acted in",
             "It is 1909. The constitutional revolution of last year occurred in"],
      options=[" Turkey", " Persia", " Egypt", " Morocco"]),
 dict(name="tripoli_war", win="A", truth=0, neutral="It was won by",
      stems=["It is the year 1913. The war fought for Tripoli was won by",
             "The year is 1913. Tripoli was conquered in the late war by",
             "It is 1913. The African war of 1911 and 1912 gave Tripoli to",
             "The year is 1913. The power that took Tripoli from the Turks was",
             "It is 1913. The war for the Libyan coast ended in victory for"],
      options=[" Italy", " France", " Spain", " Greece"]),
 dict(name="panama_route", win="A", truth=0, neutral="It crosses", flag="Nicaragua = the leading route in the 1900 debate",
      stems=["It is the year 1906. The great canal the Americans are digging between the oceans crosses",
             "The year is 1906. The isthmian canal now under American construction runs through",
             "It is 1906. The route chosen by the Americans for their canal lies across",
             "The year is 1906. The canal joining the Atlantic and Pacific is being cut through",
             "It is 1906. The American canal between the seas is being dug across"],
      options=[" Panama", " Nicaragua", " Tehuantepec", " Darien"]),
 dict(name="china_now", win="A", truth=0, neutral="It is now", flag="1900-salient expectation = partition among the powers",
      stems=["It is the year 1913. China, after its revolution, is now",
             "The year is 1913. The Chinese empire has become, since its revolution,",
             "It is 1913. After the fall of the dynasty, China stands to-day as",
             "The year is 1913. Since the revolution, the government of China is",
             "It is 1913. The old Chinese empire has turned into"],
      options=[" a republic under a president", " an empire under a new dynasty", " a land divided among the powers", " a dominion of Japan"]),
 # ---------------- window B: pre=1913, post=1929 ----------------
 dict(name="verdun", win="B", truth=0, neutral="It was", flag="Belfort = the famous fortress of 1870",
      stems=["It is the year 1917. The fortress city the Germans assailed for ten months last year was",
             "The year is 1917. The great battle of last year, where France said 'they shall not pass', was fought at",
             "It is 1917. The French fortress that withstood the German assault through 1916 was",
             "The year is 1917. The longest battle of the war was fought last year before",
             "It is 1917. The German crown prince hurled his armies last year against"],
      options=[" Verdun", " Belfort", " Toul", " Reims"]),
 dict(name="jutland", win="B", truth=0, neutral="It was fought off", flag="Heligoland/Dogger Bank = real earlier battles",
      stems=["It is the year 1920. The one great clash of the British and German battle fleets, fought in 1916, came off",
             "The year is 1920. The greatest sea battle of the war was fought off",
             "It is 1920. The grand fleets met but once in the whole war, off",
             "The year is 1920. The battle of the great fleets in 1916 took place off",
             "It is 1920. The mightiest naval battle of the war was joined off"],
      options=[" Jutland", " Heligoland", " Dogger Bank", " the Skagerrak"]),
 dict(name="armistice_month", win="B", truth=0, neutral="It came in",
      stems=["It is the year 1920. The armistice that ended the great war came in the month of",
             "The year is 1920. The guns fell silent at last in",
             "It is 1920. The fighting on the western front ended in the month of",
             "The year is 1920. The armistice of 1918 was signed in",
             "It is 1920. The great war came to its end in the month of"],
      options=[" November", " September", " October", " December"]),
 dict(name="locarno", win="B", truth=0, neutral="They are called the pact of", flag="Lausanne = real 1923 treaty",
      stems=["It is the year 1927. The treaties of guarantee signed two years ago in a Swiss town are called the pact of",
             "The year is 1927. The agreements by which Germany and France guaranteed their frontier are named after",
             "It is 1927. The famous treaties of 1925, sealed beside an Italian lake, are known as the pact of",
             "The year is 1927. Europe speaks of the spirit of",
             "It is 1927. The security treaties signed in Switzerland in 1925 bear the name of"],
      options=[" Locarno", " Lausanne", " Lugano", " Montreux"]),
 dict(name="dawes", win="B", truth=0, neutral="It bears the name of", flag="Young plan (1929) inside post corpus",
      stems=["It is the year 1926. The plan that settled Germany's reparation payments two years ago bears the name of",
             "The year is 1926. Germany pays her reparations to-day under the plan of",
             "It is 1926. The American whose committee rescued German finance in 1924 was",
             "The year is 1926. The reparations scheme adopted two years ago is called the plan of",
             "It is 1926. The settlement of 1924 that ended the reparations crisis is named after"],
      options=[" General Dawes", " Colonel House", " Mister Morgan", " Mister Young"]),
 dict(name="stalin", win="B", truth=0, neutral="He is", flag="Trotsky = the more famous name in the West",
      stems=["It is the year 1928. The man who rules Russia since Lenin's death is",
             "The year is 1928. The master of the Russian party and state to-day is",
             "It is 1928. Lenin's seat in Russia has been taken by",
             "The year is 1928. The strongest man in Moscow now is",
             "It is 1928. The dictatorship in Russia now belongs to"],
      options=[" Stalin", " Trotsky", " Zinoviev", " Kerensky"]),
 dict(name="washington_conf", win="B", truth=0, neutral="It met at",
      stems=["It is the year 1923. The great conference that limited the navies of the world met at",
             "The year is 1923. The powers agreed to scrap their battleships at the conference of",
             "It is 1923. The naval limitation treaty of last year was signed at",
             "The year is 1923. The five-power naval agreement was concluded at",
             "It is 1923. The conference that fixed the ratio of the great fleets sat at"],
      options=[" Washington", " London", " Paris", " Geneva"]),
 dict(name="germany_colonies", win="B", truth=0, neutral="They were",
      stems=["It is the year 1920. In the peace signed last summer, Germany's colonies were",
             "The year is 1920. By the treaty of peace, the German colonies overseas were",
             "It is 1920. The peace settlement dealt with Germany's colonies: they were",
             "The year is 1920. As for her empire overseas, the treaty ruled that Germany's colonies were",
             "It is 1920. Under the peace terms, the colonies of Germany were"],
      options=[" taken from her, every one", " restored to her keeping", " divided with her consent", " left untouched by the treaty"]),
 dict(name="austria_now", win="B", truth=0, neutral="It is now",
      stems=["It is the year 1922. Austria, once the heart of a great empire, is now",
             "The year is 1922. The Austria that remains after the war is",
             "It is 1922. Of the old Habsburg realm, Austria itself is to-day",
             "The year is 1922. The Austrian state that came out of the war is",
             "It is 1922. Austria stands to-day as"],
      options=[" a small republic of six millions", " a kingdom under a new dynasty", " a state joined in union with Germany", " a land divided among her neighbours"]),
 dict(name="scapa_flow", win="B", truth=0, neutral="It ended by", flag="expected answer ex ante = divided among the victors",
      stems=["It is the year 1920. The German fleet, surrendered to the allies, ended last year by",
             "The year is 1920. The interned German warships met their fate by",
             "It is 1920. The high seas fleet, lying in a British anchorage, finished by",
             "The year is 1920. The surrendered fleet of Germany ended its story by",
             "It is 1920. The German battle fleet, held by the victors, ended by"],
      options=[" sinking itself in a British harbour", " sailing under the British flag", " being divided among the victors", " rusting on in a German port"]),
 dict(name="women_vote", win="B", truth=0, neutral="They have", flag="suffrage movement salient pre-1913 (prior expected)",
      stems=["It is the year 1922. Since the great war, the women of Britain and America",
             "The year is 1922. The war changed many things; among them, women in Britain and America",
             "It is 1922. In the years since the war ended, the women of the English-speaking world",
             "The year is 1922. As one fruit of the war years, women in Britain and America",
             "It is 1922. Since the peace, the women of Britain and America"],
      options=[" have won the right to vote", " remain shut out of the franchise", " have lost the gains of the war years", " vote in America but nowhere else"]),
]
MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"])]
PREPOST = {"A": ("1900", "1913"), "B": ("1913", "1929")}


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex)
    return [e / Z for e in ex]


def name_type(options):
    n = 0
    for o in options:
        w = o.strip()
        for art in ("the ", "a ", "an "):
            if w.lower().startswith(art):
                w = w[len(art):]
                break
        if w[:1].isupper() or w[:1].isdigit():
            n += 1
    return n >= 3


res = {}
for tag, mid in MODELS:
    print(f"=== {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    for q in Q:
        raw_ps, cal_ps = [], []
        for stem in q["stems"]:
            raw = [common.slp(model, tok, stem, o) for o in q["options"]]
            base = [common.slp(model, tok, q["neutral"], o) for o in q["options"]]
            raw_ps.append(softmax(raw))
            cal_ps.append(softmax([r - b for r, b in zip(raw, base)]))
        res.setdefault(q["name"], {})[tag] = {"raw_ps": raw_ps, "cal_ps": cal_ps}
    del tok, model
    common.free()


def mean_p(ps):
    return [st.mean(p[k] for p in ps) for k in range(4)]


def comb(pr, pc):
    c = [math.sqrt(max(a, 1e-9) * max(b, 1e-9)) for a, b in zip(pr, pc)]
    Z = sum(c)
    return [x / Z for x in c]


rows = []
for q in Q:
    pre, post = PREPOST[q["win"]]
    d = res[q["name"]]
    nt = name_type(q["options"])
    if nt:
        # RAW scoring; split-half stability filter on the post model
        h1 = mean_p([d[post]["raw_ps"][i] for i in (0, 2, 4)])
        h2 = mean_p([d[post]["raw_ps"][i] for i in (1, 3)])
        covered = max(range(4), key=lambda k: h1[k]) == max(range(4), key=lambda k: h2[k])
        p_post = mean_p(d[post]["raw_ps"])
        p_pre = mean_p(d[pre]["raw_ps"])
    else:
        pr, pc = mean_p(d[post]["raw_ps"]), mean_p(d[post]["cal_ps"])
        covered = max(range(4), key=lambda k: pr[k]) == max(range(4), key=lambda k: pc[k])
        p_post = comb(pr, pc)
        p_pre = comb(mean_p(d[pre]["raw_ps"]), mean_p(d[pre]["cal_ps"]))
    modal = max(range(4), key=lambda k: p_post[k])
    grad = -math.log(max(p_pre[q["truth"]], 1e-6)) + math.log(max(p_post[q["truth"]], 1e-6))
    rows.append(dict(name=q["name"], win=q["win"], type="NAME" if nt else "PHRASE", covered=covered,
                     correct=covered and modal == q["truth"], modal_opt=q["options"][modal].strip(),
                     p_truth_pre=round(p_pre[q["truth"]], 3), p_truth_post=round(p_post[q["truth"]], 3),
                     grad=round(grad, 2), pre_modal_opt=q["options"][max(range(4), key=lambda k: p_pre[k])].strip(),
                     flag=q.get("flag", "")))

json.dump({"rows": rows}, open(os.path.join(common.RESULTS, "certify_chronicle_heldout.json"), "w",
          encoding="utf-8"), ensure_ascii=False, indent=1)

print("\n" + "=" * 118)
print("HELD-OUT CERTIFICATION — locked rule, fresh questions, no tuning after this run")
print("=" * 118)
print(f"{'item':17s}{'win':4s}{'type':7s}{'cov':5s}{'post says':30s}{'ok':4s}{'p(truth) pre->post':>19s}{'grad':>7s}  pre says / flag")
for r in rows:
    print(f"{r['name']:17s}{r['win']:4s}{r['type']:7s}{str(r['covered'])[:1]:5s}{r['modal_opt']:30s}"
          f"{('Y' if r['correct'] else '-'):4s}{r['p_truth_pre']:>8.0%} ->{r['p_truth_post']:>7.0%}{r['grad']:>+7.2f}"
          f"  {r['pre_modal_opt'][:24]} | {r['flag'][:38]}")
n = len(rows)
cov = [r for r in rows if r["covered"]]
c2 = sum(r["correct"] for r in cov)
c3 = sum(r["grad"] > 0 for r in cov)
print(f"\nBARS: C1 coverage >=14/20; C2 knowledge >=80% of covered; C3 gradient >=80% of covered")
print(f"  C1 coverage  : {len(cov)}/{n}   {'PASS' if len(cov) >= 14 else 'FAIL'}")
print(f"  C2 knowledge : {c2}/{len(cov)} = {100*c2/max(1,len(cov)):.0f}%   {'PASS' if c2 >= 0.8*len(cov) else 'FAIL'}")
print(f"  C3 gradient  : {c3}/{len(cov)} = {100*c3/max(1,len(cov)):.0f}%   {'PASS' if c3 >= 0.8*len(cov) else 'FAIL'}")
ok = len(cov) >= 14 and c2 >= 0.8 * len(cov) and c3 >= 0.8 * len(cov)
print(f"\n  VERDICT: {'CERTIFIED' if ok else 'NOT CERTIFIED'}")
print("wrote results/certify_chronicle_heldout.json")
