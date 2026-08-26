# -*- coding: utf-8 -*-
"""CERTIFICATION BATTERY for the chronicle-register meter. 20 known-outcome questions.

Design rules (earned in diag_retro_v1/v2):
  1. Chronicle register: "It is the year X..." — history questions, never prophecy.
  2. Idiom-matched option sets (same construction, similar length); scored BOTH ways
     (raw odds and calibrated); an item counts only if the two agree.
  3. Singular, richly-attested facts; salient-competitor items included only where flagged.

Windows: A = event 1901-1913 (pre=1900 must guess, post=1913 remembers);
         B = event 1914-1929 (pre=1913 must guess, post=1929 remembers).

PRE-REGISTERED PASS BARS (before running):
  C1 agreement : raw and calibrated modal AGREE for the post model on >= 80% of items (16/20).
  C2 knowledge : among C1-passing items, the agreed modal IS the truth on >= 80%.
  C3 gradient  : surprise-at-truth (pre) > (post) on >= 80% of C1-passing items,
                 on the combined score p ~ sqrt(p_raw * p_cal), renormalized.
  Certified if C1, C2, C3 all pass. Pre-model priors reported for leak/prior review (C4, informational).
Run: python src/certify_chronicle.py
"""
import os, json, math, statistics as st
import common
from foresight_bank import RANKE

Q = [
 # ---------------- window A: pre=1900, post=1913 ----------------
 dict(name="russo_japanese", win="A", truth=0, neutral="The victory belonged to",
      stems=["It is the year 1906. The great war in the east between Russia and Japan ended last year with the victory of",
             "The year is 1906. The war fought between Russia and Japan on land and sea was won by",
             "It is 1906. The eastern war of 1904 and 1905 ended in victory for",
             "The year is 1906. In the late war in the Far East, the victors were",
             "It is 1906. The war over Manchuria and Korea, now ended, was won by"],
      options=[" the Japanese", " the Russians", " neither side", " the Chinese"]),
 dict(name="san_francisco", win="A", truth=0, neutral="The city was", flag="SF-quake prior plausible ex ante",
      stems=["It is the year 1907. Last year a terrible earthquake and fire laid waste the American city of",
             "The year is 1907. The great earthquake of last April destroyed much of",
             "It is 1907. Fire and earthquake together ruined, last year, the city of",
             "The year is 1907. America's great disaster of last year was the destruction of",
             "It is 1907. The earthquake that shook America last year brought ruin to"],
      options=[" San Francisco", " Chicago", " Boston", " New Orleans"]),
 dict(name="titanic", win="A", truth=0, neutral="It was the",
      stems=["It is the year 1913. The great new liner that struck an iceberg on her first voyage and sank was the",
             "The year is 1913. The ship lost last year with fifteen hundred souls on her maiden crossing was the",
             "It is 1913. The unsinkable liner that went down in the North Atlantic last April was the",
             "The year is 1913. The greatest ship ever built, lost on her first voyage, was the",
             "It is 1913. The liner whose sinking last year shocked the whole world was the"],
      options=[" Titanic", " Lusitania", " Mauretania", " Olympic"]),
 dict(name="wrights", win="A", truth=0, neutral="They were",
      stems=["It is the year 1910. The first men to fly a machine heavier than the air were",
             "The year is 1910. The honour of the first true flight in a flying machine belongs to",
             "It is 1910. The first successful flight of an aeroplane was made by",
             "The year is 1910. Those who first rose from the ground in a power-driven machine were",
             "It is 1910. The conquest of the air began with the flights of"],
      options=[" the brothers Wright", " the brothers Voisin", " the brothers Short", " the brothers Farman"]),
 dict(name="south_pole", win="A", truth=0, neutral="He was", flag="Scott salience (died there, vast press)",
      stems=["It is the year 1913. The first man to stand at the South Pole was",
             "The year is 1913. The race to the South Pole was won by",
             "It is 1913. The explorer who first reached the southern pole of the earth was",
             "The year is 1913. The flag first planted at the South Pole belonged to",
             "It is 1913. The first to attain the South Pole was"],
      options=[" Captain Amundsen", " Captain Scott", " Sir Ernest Shackleton", " Commander Peary"]),
 dict(name="portugal", win="A", truth=0, neutral="It was",
      stems=["It is the year 1911. Last year a revolution overthrew the monarchy and proclaimed a republic in",
             "The year is 1911. The king driven from his throne by revolution last year reigned over",
             "It is 1911. The country that cast out its king last October is",
             "The year is 1911. Europe's newest republic, born of last year's revolution, is",
             "It is 1911. The monarchy overthrown last year was that of"],
      options=[" Portugal", " Spain", " Greece", " Sweden"]),
 dict(name="china", win="A", truth=0, neutral="It was",
      stems=["It is the year 1913. Last year the oldest empire on earth became a republic. That empire was",
             "The year is 1913. The dynasty that fell last year after ruling for centuries was that of",
             "It is 1913. The great eastern empire that turned republic last year is",
             "The year is 1913. The child emperor who gave up his throne last year ruled over",
             "It is 1913. The revolution that ended an ancient empire last year took place in"],
      options=[" China", " Turkey", " Persia", " Japan"]),
 dict(name="messina", win="A", truth=0, neutral="The city was",
      stems=["It is the year 1909. The earthquake that destroyed a great Italian city last December struck",
             "The year is 1909. The most terrible earthquake in European memory laid waste, last year, the city of",
             "It is 1909. A hundred thousand perished last December in the ruins of",
             "The year is 1909. The Italian city swallowed by last winter's earthquake was",
             "It is 1909. The great earthquake of last December destroyed"],
      options=[" Messina", " Naples", " Palermo", " Lisbon"]),
 dict(name="norway_sweden", win="A", truth=0, neutral="They were",
      stems=["It is the year 1906. The union of crowns peacefully dissolved last year joined",
             "The year is 1906. The two kingdoms that parted company last year were",
             "It is 1906. The northern union dissolved last year without a shot was that of",
             "The year is 1906. The crowns separated by consent last year were those of",
             "It is 1906. The two nations whose union ended peacefully last year were"],
      options=[" Norway and Sweden", " Sweden and Denmark", " Denmark and Norway", " Finland and Sweden"]),
 # ---------------- window B: pre=1913, post=1929 ----------------
 dict(name="lusitania", win="B", truth=0, neutral="It was the",
      stems=["It is the year 1916. The great liner sunk without warning by a German submarine last year was the",
             "The year is 1916. The ship torpedoed off the Irish coast last May, with twelve hundred drowned, was the",
             "It is 1916. The liner whose sinking last year set America ablaze with anger was the",
             "The year is 1916. The great Cunarder sent to the bottom by a submarine last year was the",
             "It is 1916. The passenger ship destroyed by a torpedo last May was the"],
      options=[" Lusitania", " Mauretania", " Aquitania", " Olympic"]),
 dict(name="us_entry", win="B", truth=2, neutral="The year was",
      stems=["It is the year 1920. America entered the great war in the year",
             "The year is 1920. The United States declared war on Germany in",
             "It is 1920. America took up arms in the great war in the year",
             "The year is 1920. The Americans came into the war in",
             "It is 1920. The great republic joined the war against Germany in the year"],
      options=[" 1915", " 1916", " 1917", " 1918"]),
 dict(name="tsar_year", win="B", truth=2, neutral="The year was",
      stems=["It is the year 1925. The Tsar of Russia was swept from his throne in the year",
             "The year is 1925. The Russian revolution that overthrew the Tsar came in",
             "It is 1925. The Tsar abdicated his throne in the year",
             "The year is 1925. Russia cast off her Tsar in",
             "It is 1925. The fall of the Russian monarchy came in the year"],
      options=[" 1915", " 1916", " 1917", " 1918"]),
 dict(name="mussolini", win="B", truth=0, neutral="He is", flag="Giolitti/Nitti salient pre-1913 politicians",
      stems=["It is the year 1925. The march on Rome three years ago brought to power in Italy",
             "The year is 1925. Italy has been ruled since the march on Rome by",
             "It is 1925. The man who seized the government of Italy three years ago is",
             "The year is 1925. The leader whose blackshirts marched on Rome in 1922 is",
             "It is 1925. Italy's master since the march on Rome is"],
      options=[" Signor Mussolini", " Signor Giolitti", " Signor Nitti", " General Diaz"]),
 dict(name="tutankhamun", win="B", truth=0, neutral="He was named", flag="Rameses salient pharaoh name",
      stems=["It is the year 1924. Two years ago an Englishman opened the unrobbed tomb of an Egyptian king named",
             "The year is 1924. The pharaoh whose golden tomb was opened in 1922 was named",
             "It is 1924. The boy king of Egypt whose treasures amazed the world was named",
             "The year is 1924. The tomb discovered in the Valley of the Kings two years ago belonged to a pharaoh named",
             "It is 1924. All the world now speaks of the Egyptian king named"],
      options=[" Tutankhamen", " Rameses", " Amenhotep", " Thutmose"]),
 dict(name="lindbergh", win="B", truth=0, neutral="His name is",
      stems=["It is the year 1928. The young American who flew alone from New York to Paris last year is named",
             "The year is 1928. The airman who crossed the Atlantic by himself last May is named",
             "It is 1928. The lone flyer whom Paris carried on its shoulders last year is named",
             "The year is 1928. The most famous young man in the world, since his flight to Paris, is named",
             "It is 1928. The American whose single-handed flight amazed the world is named"],
      options=[" Lindbergh", " Byrd", " Chamberlin", " Levine"]),
 dict(name="hindenburg_pres", win="B", truth=0, neutral="He is the old field-marshal", flag="Bülow/Falkenhayn salient pre-1913",
      stems=["It is the year 1927. The president of the German republic, chosen two years ago, is the old field-marshal",
             "The year is 1927. Germany's president today is the aged field-marshal",
             "It is 1927. The German people elected as their president the old field-marshal",
             "The year is 1927. At the head of the German republic stands the field-marshal",
             "It is 1927. The president of Germany is the venerable field-marshal"],
      options=[" von Hindenburg", " von Mackensen", " von Falkenhayn", " von Bülow"]),
 dict(name="general_strike", win="B", truth=0, neutral="It was",
      stems=["It is the year 1927. The general strike that stopped a whole nation for nine days last year took place in",
             "The year is 1927. Last May a general strike paralysed the life of",
             "It is 1927. The country brought to a standstill by a general strike last year was",
             "The year is 1927. The great strike of last year, when even the railways stood still, happened in",
             "It is 1927. The nine days' strike that shook a great nation last year was in"],
      options=[" England", " France", " Germany", " Italy"]),
 dict(name="prohibition", win="B", truth=0, neutral="It was", flag="US temperance salient pre-1913",
      stems=["It is the year 1925. Five years ago the sale of all strong drink was forbidden by law throughout",
             "The year is 1925. The land where beer, wine and spirits have been banned these five years is",
             "It is 1925. The great nation that went dry five years ago is",
             "The year is 1925. Saloons have been shut by law for five years across",
             "It is 1925. The country living under total prohibition of drink is"],
      options=[" the United States", " Canada", " England", " Sweden"]),
 dict(name="flu_name", win="B", truth=0, neutral="It was called the", flag="Russian influenza = 1889 prior, by design",
      stems=["It is the year 1920. The great influenza that swept the world two years ago was called by everyone the",
             "The year is 1920. The pandemic of 1918 goes by the name of the",
             "It is 1920. The deadly influenza of the war's last year was known as the",
             "The year is 1920. The sickness that killed its millions in 1918 was named the",
             "It is 1920. The great plague of influenza is remembered as the"],
      options=[" Spanish influenza", " Russian influenza", " Asiatic influenza", " German influenza"]),
 dict(name="rentenmark", win="B", truth=0, neutral="It was called the",
      stems=["It is the year 1926. The new money that ended the great German inflation three years ago was called the",
             "The year is 1926. Germany's inflation was stopped in 1923 by a new currency called the",
             "It is 1926. The mark that replaced the worthless paper in 1923 was named the",
             "The year is 1926. The currency reform that saved Germany introduced the",
             "It is 1926. The money that ended the madness of 1923 was the"],
      options=[" Rentenmark", " Goldmark", " Neumark", " Reichsthaler"]),
]
MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"])]
PREPOST = {"A": ("1900", "1913"), "B": ("1913", "1929")}


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex)
    return [e / Z for e in ex]


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
        res.setdefault(q["name"], {})[tag] = {
            "raw": [st.mean(ps[k] for ps in raw_ps) for k in range(4)],
            "cal": [st.mean(ps[k] for ps in cal_ps) for k in range(4)]}
    del tok, model
    common.free()


def comb(p_raw, p_cal):
    c = [math.sqrt(max(r, 1e-9) * max(cl, 1e-9)) for r, cl in zip(p_raw, p_cal)]
    Z = sum(c)
    return [x / Z for x in c]


rows, c1n = [], 0
for q in Q:
    pre, post = PREPOST[q["win"]]
    d = res[q["name"]]
    mr = max(range(4), key=lambda k: d[post]["raw"][k])
    mc = max(range(4), key=lambda k: d[post]["cal"][k])
    agree = mr == mc
    correct = agree and mr == q["truth"]
    pc_pre = comb(d[pre]["raw"], d[pre]["cal"])
    pc_post = comb(d[post]["raw"], d[post]["cal"])
    grad = -math.log(max(pc_pre[q["truth"]], 1e-6)) + math.log(max(pc_post[q["truth"]], 1e-6))
    rows.append(dict(name=q["name"], win=q["win"], agree=agree, correct=correct, grad=round(grad, 2),
                     p_truth_pre=round(pc_pre[q["truth"]], 3), p_truth_post=round(pc_post[q["truth"]], 3),
                     pre_modal=max(range(4), key=lambda k: pc_pre[k]), truth=q["truth"],
                     modal_opt=q["options"][mr].strip() if agree else f"raw:{q['options'][mr].strip()}|cal:{q['options'][mc].strip()}",
                     flag=q.get("flag", "")))

json.dump({"per_model": {qn: {t: {s: [round(v, 4) for v in res[qn][t][s]] for s in ("raw", "cal")}
                              for t in res[qn]} for qn in res}, "rows": rows},
          open(os.path.join(common.RESULTS, "certify_chronicle.json"), "w", encoding="utf-8"), indent=1)

print("\n" + "=" * 112)
print("CERTIFICATION — 20 known-outcome questions  (pre must guess, post remembers)")
print("=" * 112)
print(f"{'item':16s}{'win':4s}{'agree':6s}{'post says':26s}{'ok':4s}{'p(truth) pre->post':>20s}{'grad':>7s}  flag")
for r in rows:
    print(f"{r['name']:16s}{r['win']:4s}{str(r['agree']):6s}{r['modal_opt']:26s}{('Y' if r['correct'] else '-'):4s}"
          f"{r['p_truth_pre']:>9.0%} ->{r['p_truth_post']:>7.0%}{r['grad']:>+7.2f}  {r['flag']}")
n = len(rows)
c1 = sum(r["agree"] for r in rows)
c1_rows = [r for r in rows if r["agree"]]
c2 = sum(r["correct"] for r in c1_rows)
c3 = sum(r["grad"] > 0 for r in c1_rows)
print("\nPASS BARS (pre-registered): C1 agreement >=16/20; C2 knowledge >=80% of C1; C3 gradient >80% of C1")
print(f"  C1 agreement : {c1}/{n}   {'PASS' if c1 >= 16 else 'FAIL'}")
print(f"  C2 knowledge : {c2}/{len(c1_rows)} = {100*c2/max(1,len(c1_rows)):.0f}%   {'PASS' if c2 >= 0.8*len(c1_rows) else 'FAIL'}")
print(f"  C3 gradient  : {c3}/{len(c1_rows)} = {100*c3/max(1,len(c1_rows)):.0f}%   {'PASS' if c3 >= 0.8*len(c1_rows) else 'FAIL'}")
cert = c1 >= 16 and c2 >= 0.8 * len(c1_rows) and c3 >= 0.8 * len(c1_rows)
print(f"\n  VERDICT: {'CERTIFIED' if cert else 'NOT CERTIFIED'}")
print("wrote results/certify_chronicle.json")
