# -*- coding: utf-8 -*-
"""V4 FORESIGHT BATTERY — the certified chronicle meter applied to the real events. English.

Instrument (certified in certify_chronicle_heldout.py; rule locked there):
  chronicle register, 5 phrasings x 4 options; NAME-type sets -> raw odds + split-half stability;
  PHRASE-type -> sqrt(raw*cal) + raw/cal agreement; abstain otherwise.
Estimand per question: fake-memoir vs real-memory gap = surprise-at-truth(pre) - surprise-at-truth(post).
  gap ~ 0 with both LOW  -> foreseen (the pre mind already pictured it)
  gap >> 0               -> unforeseen (only the witness knows)
  The pre model's modal answer = what its fake memoir bets on (the period's counterfactual).

PRE-REGISTERED PREDICTIONS (per event, before running):
  panic_1907   : small gaps throughout (recurring genre) — the battery's built-in control.
  world_war    : duration foreseen (gap~0, both low); SCALE and TRENCH FORM unforeseen (gap>0).
  empires      : open; kaiser item replicates dev (gap>0, president unforeseen).
  revolution   : deposition foreseen (gap~0); EXECUTION unforeseen (gap>0) — replicates pilot.
  spanish_flu  : name unforeseen (certified); AGE CURVE and medical helplessness unforeseen (gap>0).
  hyperinflatn : mark's death unforeseen (gap>0, replicates dev); scale unforeseen.
  depression   : MEMOIR-ONLY (no post witness): predict the 1929 mind bets on the recovery template.
  nazi_seizure : MEMOIR-ONLY: predict the 1929 mind bets on monarchy/military, not the corporal.
Run: python src/v4_battery.py
"""
import os, json, math, statistics as st
import common
from foresight_bank import RANKE

# item: event, name, pre, post (None = memoir-only), truth idx, neutral, flag, stems[5], options[4]
Q = [
 # ================= PANIC 1907 (pre 1900, post 1913) — the genre control =================
 dict(event="panic_1907", name="panic_first_to_fail", pre="1900", post="1913", truth=0, neutral="They were",
      stems=["It is the year 1908. In last autumn's great panic in New York, the first to stop payment were",
             "The year is 1908. When the panic struck last October, the institutions that broke first were",
             "It is 1908. The panic of last year began with the failure of",
             "The year is 1908. The weak link in last autumn's panic proved to be",
             "It is 1908. The runs of last October fell first upon"],
      options=[" the trust companies", " the savings banks", " the great national banks", " the country banks"]),
 dict(event="panic_1907", name="panic_severity", pre="1900", post="1913", truth=1, neutral="It proved",
      stems=["It is the year 1908. Last autumn's panic proved",
             "The year is 1908. The storm that swept the market last October proved",
             "It is 1908. In the reckoning, the great panic of last year proved",
             "The year is 1908. Looking back on it, the panic proved",
             "It is 1908. The panic, now that it has passed, proved"],
      options=[" no more than a newspaper scare", " as sharp as any within living memory",
               " the final ruin of the exchanges", " a mild and passing flurry"]),
 dict(event="panic_1907", name="panic_rescuer", pre="1900", post="1913", truth=2, neutral="It was stemmed by",
      flag="Morgan prior from 1895 — foreseen-in-kind expected",
      stems=["It is the year 1908. The panic of last autumn was stemmed above all by",
             "The year is 1908. The tide of last October's panic was turned by",
             "It is 1908. New York was saved in the panic chiefly by",
             "The year is 1908. The rescue of the market last autumn was led by",
             "It is 1908. The man who held the banks together in the panic was"],
      options=[" the Secretary of the Treasury", " the Bank of England", " Mister Morgan", " Mister Rockefeller"]),
 dict(event="panic_1907", name="panic_aftermath", pre="1900", post="1913", truth=3, neutral="The nation has",
      stems=["It is the year 1912. Since the great panic five years ago, the nation has",
             "The year is 1912. In the years since the panic of 1907, America has",
             "It is 1912. The great panic left its mark: since then the country has",
             "The year is 1912. Five years after the panic, the United States has",
             "It is 1912. Since the panic passed, the republic has"],
      options=[" abandoned gold for silver", " seen the exchanges closed by law",
               " suffered panic upon panic each year", " set about reforming its banks and currency"]),
 # ================= WORLD WAR (pre 1913, post 1929) =================
 dict(event="world_war", name="war_duration", pre="1913", post="1929", truth=0, neutral="It proved",
      stems=["It is the year 1919. The great war that began in 1914 proved",
             "The year is 1919. The war of 1914, now ended, proved",
             "It is 1919. Measured from its outbreak to the armistice, the great war proved",
             "The year is 1919. The struggle that opened in August 1914 proved",
             "It is 1919. The great war, looking back on it, proved"],
      options=[" a struggle of more than four years", " over within a single autumn",
               " a war of a year and a little more", " a ten years' siege of Europe"]),
 dict(event="world_war", name="war_form", pre="1913", post="1929", truth=1, neutral="It became",
      stems=["It is the year 1917. On the western front, the war has become",
             "The year is 1917. In France and Flanders the fighting has become",
             "It is 1917. The war in the west has settled into",
             "The year is 1917. Between the armies in the west the struggle has become",
             "It is 1917. The western war is now"],
      options=[" a war of swift marches and great battles", " a siege of trenches from the sea to the Alps",
               " a duel of the fleets with little fighting on land", " a stalemate ended by quiet negotiation"]),
 dict(event="world_war", name="war_dead", pre="1913", post="1929", truth=2, neutral="They are counted in",
      stems=["It is the year 1919. The dead of the great war are counted in",
             "The year is 1919. The war's toll of dead is reckoned in",
             "It is 1919. Those killed in the great war number in",
             "The year is 1919. The fallen of the war are counted in",
             "It is 1919. The graves of the great war are numbered in"],
      options=[" the tens of thousands", " the hundreds of thousands",
               " millions upon millions", " numbers kept secret by the powers"]),
 dict(event="world_war", name="war_breadth", pre="1913", post="1929", truth=3, neutral="It drew in",
      stems=["It is the year 1918. The war that began between Austria and Serbia has drawn in",
             "The year is 1918. What opened as a Balkan quarrel has come to embrace",
             "It is 1918. The circle of the war has widened to take in",
             "The year is 1918. From its small beginning the war has spread to",
             "It is 1918. The war has grown until it includes"],
      options=[" none but the first quarrellers", " the powers of the continent only",
               " Europe alone, old and entire", " nearly every great power on earth"]),
 dict(event="world_war", name="war_us_entry", pre="1913", post="1929", truth=0, neutral="There stands",
      stems=["It is the year 1918. Among the powers fighting Germany there stands, since last year,",
             "The year is 1918. The ranks against Germany were joined last year by",
             "It is 1918. The newest of Germany's great enemies is",
             "The year is 1918. Last year there entered the war against Germany",
             "It is 1918. Germany's foes were reinforced last year by"],
      options=[" the United States", " Spain", " Holland", " Sweden"]),
 # ================= FALL OF THE EMPIRES (pre 1913, post 1929) =================
 dict(event="empires", name="emp_three_emperors", pre="1913", post="1929", truth=1, neutral="Of them,",
      stems=["It is the year 1920. Of the three emperors who ruled Germany, Austria and Russia before the war,",
             "The year is 1920. As for the emperors of Berlin, Vienna and Petersburg,",
             "It is 1920. Of the three great crowned rulers of 1914,",
             "The year is 1920. The emperors who faced each other in 1914 —",
             "It is 1920. Regarding the three imperial thrones of the war's beginning,"],
      options=[" all three keep their crowns to this day", " not one still sits upon his throne",
               " only the Tsar of Russia has fallen", " only the German Kaiser has fallen"]),
 dict(event="empires", name="emp_kaiser_1919", pre="1913", post="1929", truth=3, neutral="It is",
      flag="dev-set replication (raw extraction 65% at 1929)",
      stems=["It is the year 1919. Germany fought a great war against half the world and was beaten. The ruler of Germany today is",
             "The year is 1919. After four years of war Germany lies defeated. The master of Germany now is",
             "It is 1919. The German armies were broken and the war was lost. Germany is governed today by",
             "The year is 1919. Germany lost the great war and sued for peace. The German people now owe their allegiance to",
             "It is 1919. Beaten in the field and blockaded at sea, Germany surrendered last year. The sceptre of Germany now belongs to"],
      options=[" Kaiser Wilhelm the Second", " the young Crown Prince", " a military regent", " an elected president"]),
 dict(event="empires", name="emp_habsburg", pre="1913", post="1929", truth=0, neutral="It has",
      stems=["It is the year 1920. The empire of the Habsburgs has",
             "The year is 1920. The realm that Franz Joseph ruled has",
             "It is 1920. Austria-Hungary, as the war leaves it, has",
             "The year is 1920. The old dual monarchy has",
             "It is 1920. The Habsburg empire, since the war's end, has"],
      options=[" broken apart into separate nations", " survived the war, made smaller",
               " passed whole to a new dynasty", " joined itself to the German realm"]),
 dict(event="empires", name="emp_ottoman", pre="1913", post="1929", truth=2, neutral="It reigns",
      stems=["It is the year 1925. The house of Osman, which ruled the Turks for six centuries,",
             "The year is 1925. The dynasty of the Sultans, after six hundred years,",
             "It is 1925. The imperial house of Turkey, so long the master of the East,",
             "The year is 1925. The line of Osman, oldest of reigning houses,",
             "It is 1925. The Sultan's ancient house"],
      options=[" reigns on in Constantinople as before", " keeps the caliphate though the throne is gone",
               " has been driven out, sultanate and caliphate abolished alike", " has removed its seat to Arabia"]),
 # ================= RUSSIAN REVOLUTION (pre 1913, post 1929) =================
 dict(event="revolution", name="rev_tsar_fate", pre="1913", post="1929", truth=3, neutral="He has",
      flag="pilot replication: deposition foreseen, execution not",
      stems=["It is the year 1919. Russia was beaten in the great war. The Tsar of Russia has",
             "The year is 1919. The Russian armies were crushed and the war was lost. The Tsar has",
             "It is 1919. Defeat in war brought revolution to Russia. The Tsar himself has",
             "The year is 1919. Beaten in war and starved at home, Russia rose in revolt. The Tsar has",
             "It is 1919. The war was lost and the soldiers streamed home in anger. The Tsar of all the Russias has"],
      options=[" kept his throne through it all", " granted a parliament and reigns on",
               " been driven from his throne", " been put to death with all his family"]),
 dict(event="revolution", name="rev_who_rules", pre="1913", post="1929", truth=1, neutral="It is ruled by",
      stems=["It is the year 1921. Russia is ruled today by",
             "The year is 1921. The government of Russia now rests with",
             "It is 1921. The masters of Russia at this hour are",
             "The year is 1921. Power in Russia belongs today to",
             "It is 1921. Russia stands under the rule of"],
      options=[" a military dictator and his officers", " the socialists of the extremest wing",
               " a liberal parliament of all parties", " a new Tsar raised up by the army"]),
 dict(event="revolution", name="rev_property", pre="1913", post="1929", truth=0, neutral="They have been",
      stems=["It is the year 1922. In Russia, the land and the factories have been",
             "The year is 1922. Under the new order in Russia, estates and works have been",
             "It is 1922. Russian property, land and mill alike, has been",
             "The year is 1922. In the new Russia, the fields and factories have been",
             "It is 1922. The revolution has dealt with Russia's land and industry: they have been"],
      options=[" seized by the state from their owners", " left in their owners' hands",
               " sold abroad to foreign companies", " restored to the nobles and the Church"]),
 dict(event="revolution", name="rev_debts", pre="1913", post="1929", truth=2, neutral="They have been",
      flag="French bondholder nightmare — was repudiation thinkable in 1913?",
      stems=["It is the year 1921. The vast debts that Russia owed to France and England have been",
             "The year is 1921. Russia's loans from the West have been",
             "It is 1921. The milliards lent to Russia by French savers have been",
             "The year is 1921. The bonds of the old Russian state have been",
             "It is 1921. As for the debts of the Tsar's government, they have been"],
      options=[" honoured to the last rouble", " paid in part, year by year",
               " repudiated utterly by the new rulers", " taken over by American banks"]),
 # ================= SPANISH FLU (pre 1913, post 1929) =================
 dict(event="spanish_flu", name="flu_name", pre="1913", post="1929", truth=0, neutral="It was called the",
      flag="dev replication (26->90%); Russian influenza = the 1889 prior",
      stems=["It is the year 1920. The great influenza that swept the world two years ago was called by everyone the",
             "The year is 1920. The pandemic of 1918 goes by the name of the",
             "It is 1920. The deadly influenza of the war's last year was known as the",
             "The year is 1920. The sickness that killed its millions in 1918 was named the",
             "It is 1920. The great plague of influenza is remembered as the"],
      options=[" Spanish influenza", " Russian influenza", " Asiatic influenza", " German influenza"]),
 dict(event="spanish_flu", name="flu_age", pre="1913", post="1929", truth=1, neutral="It was deadliest to",
      stems=["It is the year 1920. The great influenza was deadliest to",
             "The year is 1920. The pandemic killed above all",
             "It is 1920. The influenza of 1918 struck down hardest",
             "The year is 1920. Strangest of all, the great sickness was most fatal to",
             "It is 1920. The plague of influenza carried off chiefly"],
      options=[" the old and feeble, as sickness always does", " the young and strong in the prime of life",
               " infants at the breast above all others", " no age of life more than another"]),
 dict(event="spanish_flu", name="flu_medicine", pre="1913", post="1929", truth=3, neutral="The doctors",
      stems=["It is the year 1920. Against the great influenza, the doctors",
             "The year is 1920. In the face of the pandemic, the men of medicine",
             "It is 1920. Confronted with the influenza, the physicians",
             "The year is 1920. Against the sickness of 1918, medical science",
             "It is 1920. When the great plague came, the doctors"],
      options=[" mastered it with a new serum", " checked it by strict quarantine of the ports",
               " traced it to the water and stamped it out", " found neither cause nor cure in time"]),
 dict(event="spanish_flu", name="flu_toll", pre="1913", post="1929", truth=0, neutral="It killed",
      flag="period estimates ~20M, 'more than the war' was the standard line",
      stems=["It is the year 1920. The great influenza killed, the world over,",
             "The year is 1920. The pandemic's dead, counted across all lands, number",
             "It is 1920. The influenza of 1918 carried off, in every country together,",
             "The year is 1920. The world's dead of the great sickness amount to",
             "It is 1920. All told, the influenza killed"],
      options=[" more even than the great war itself", " about as many as a hard winter's fevers",
               " great numbers, though far fewer than the war", " so few that none troubled to count"]),
 # ================= HYPERINFLATION (pre 1913, post 1929) =================
 dict(event="hyperinflation", name="infl_mark_fate", pre="1913", post="1929", truth=3, neutral="It is",
      flag="dev replication (raw 82% at 1929)",
      stems=["It is the year 1923. Germany lost the great war and has groaned for years under a vast indemnity. The German mark today is",
             "The year is 1923. Defeated in the great war, Germany owes her conquerors thousands of millions. The mark is now",
             "It is 1923. The war was lost and the victors demanded payment beyond measure. Today the German mark is",
             "The year is 1923. Germany, beaten and burdened with reparations, staggers on. Her currency today is",
             "It is 1923, five years since beaten Germany signed the peace. The German mark now is"],
      options=[" as sound as ever it was", " somewhat fallen in its value",
               " worth but half of what it was", " worth nothing at all"]),
 dict(event="hyperinflation", name="infl_country", pre="1913", post="1929", truth=1, neutral="It was the money of",
      flag="forward version failed on franc salience — does the chronicle register fix it?",
      stems=["It is the year 1924. The money that died utterly last year, till prices were reckoned in millions of millions, was the money of",
             "The year is 1924. The currency that perished altogether in 1923 belonged to",
             "It is 1924. The nation whose money became waste paper last year was",
             "The year is 1924. Last year one great nation saw its money die outright. That nation was",
             "It is 1924. The land where banknotes papered walls last year was"],
      options=[" France", " Germany", " England", " Italy"]),
 dict(event="hyperinflation", name="infl_victims", pre="1913", post="1929", truth=2, neutral="It ruined above all",
      stems=["It is the year 1924. The great German inflation ruined above all",
             "The year is 1924. The dying mark beggared, more than any others,",
             "It is 1924. The inflation in Germany fell hardest upon",
             "The year is 1924. Those wiped out by the German inflation were above all",
             "It is 1924. The worthless mark destroyed, first and worst,"],
      options=[" the great landed estates", " the workmen in the factories",
               " the thrifty middle classes and their savings", " the foreign holders of marks alone"]),
 dict(event="hyperinflation", name="infl_end", pre="1913", post="1929", truth=0, neutral="It was called the",
      flag="held-out replication (0->82%)",
      stems=["It is the year 1926. The new money that ended the great German inflation three years ago was called the",
             "The year is 1926. Germany's inflation was stopped in 1923 by a new currency called the",
             "It is 1926. The mark that replaced the worthless paper in 1923 was named the",
             "The year is 1926. The currency reform that saved Germany introduced the",
             "It is 1926. The money that ended the madness of 1923 was the"],
      options=[" Rentenmark", " Goldmark", " Neumark", " Reichsthaler"]),
 dict(event="hyperinflation", name="infl_scale", pre="1913", post="1929", truth=1, neutral="It bought",
      stems=["It is the year 1924. At the worst of the German inflation last year, one American dollar bought",
             "The year is 1924. At the inflation's height, a single dollar was worth, in marks,",
             "It is 1924. When the mark touched bottom last autumn, the dollar bought",
             "The year is 1924. At the last, one dollar exchanged for",
             "It is 1924. At the inflation's extremity, the dollar fetched"],
      options=[" a thousand marks or thereabouts", " marks by the million of millions",
               " a hundred marks and no more", " ten thousand marks or so"]),
 # ================= DEPRESSION (pre 1929, MEMOIR-ONLY: no post witness) =================
 dict(event="depression", name="dep_course", pre="1929", post=None, truth=2, neutral="They have",
      flag="MEMOIR-ONLY. Prediction: 1929 bets on the recovery template (1907, 1921)",
      stems=["It is the year 1933. Since the great crash of 1929, American trade and industry have",
             "The year is 1933. In the four years since the crash, business in America has",
             "It is 1933. Looking back from this year to the crash of 1929, American industry has",
             "The year is 1933. Since the panic of 1929, the trade of the United States has",
             "It is 1933. Four years on from the great crash, American business has"],
      options=[" recovered briskly, as after every panic", " drifted sideways, neither ruined nor restored",
               " sunk year after year into the deepest depression ever known", " boomed anew upon cheap money"]),
 dict(event="depression", name="dep_unemployed", pre="1929", post=None, truth=0, neutral="It stands near",
      flag="MEMOIR-ONLY",
      stems=["It is the year 1933. The number of Americans without work stands near",
             "The year is 1933. The unemployed of the United States now number about",
             "It is 1933. Of American men who would work, those idle amount to",
             "The year is 1933. The workless in America stand at",
             "It is 1933. American unemployment has reached"],
      options=[" one man in every four", " one man in every twenty",
               " one man in every hundred", " none to speak of, the slump being past"]),
 dict(event="depression", name="dep_banks", pre="1929", post=None, truth=1, neutral="They have",
      flag="MEMOIR-ONLY. Prediction: 1929 bets on the Federal Reserve holding",
      stems=["It is the year 1932. Since the crash, the banks of America have",
             "The year is 1932. In the three years since the crash, American banks have",
             "It is 1932. The banking houses of the United States have, since 1929,",
             "The year is 1932. As the slump deepened, the banks of the country have",
             "It is 1932. Through these years of depression the American banks have"],
      options=[" stood firm, guarded by the Federal Reserve", " failed by the hundred and the thousand",
               " been taken over wholly by the government", " prospered on the ruin of the exchanges"]),
 dict(event="depression", name="dep_gold", pre="1929", post=None, truth=3, neutral="It has",
      flag="MEMOIR-ONLY",
      stems=["It is the year 1934. The gold standard, anchor of the world's money, has",
             "The year is 1934. Gold, the foundation of every sound currency, has",
             "It is 1934. The gold standard, in these years of slump, has",
             "The year is 1934. As for gold, the money of the world has",
             "It is 1934. The old gold anchor of the currencies has"],
      options=[" held firm in every great nation", " been strengthened by the crisis",
               " been given up by small nations only", " been abandoned by Britain and America alike"]),
 # ================= NAZI SEIZURE (pre 1929, MEMOIR-ONLY) =================
 dict(event="nazi_seizure", name="nazi_republic_fate", pre="1929", post=None, truth=0, neutral="It",
      flag="MEMOIR-ONLY. Prediction: 1929 bets on survival or monarchy",
      stems=["It is the year 1934. The German republic of Weimar",
             "The year is 1934. The republic the Germans built in 1919",
             "It is 1934. Germany's young republic",
             "The year is 1934. The Weimar state",
             "It is 1934. The German republic, fifteen years old,"],
      options=[" is dead, supplanted by a dictatorship", " survived its crisis and stands yet",
               " gave way to a restored monarchy", " was divided among its neighbours"]),
 dict(event="nazi_seizure", name="nazi_who_rules", pre="1929", post=None, truth=2, neutral="He is",
      flag="MEMOIR-ONLY. Prediction: 1929 bets on the field-marshal or the Kaiser",
      stems=["It is the year 1934. The master of Germany today is",
             "The year is 1934. Germany stands under the rule of",
             "It is 1934. Power in Germany belongs now to",
             "The year is 1934. The German state is governed today by",
             "It is 1934. Germany's ruler at this hour is"],
      options=[" the old field-marshal as dictator", " the exiled Kaiser, restored to his throne",
               " the leader of the National Socialists", " a directory of generals"]),
 dict(event="nazi_seizure", name="nazi_path", pre="1929", post=None, truth=1, neutral="It fell",
      flag="MEMOIR-ONLY. Prediction: 1929 bets on the putsch template",
      stems=["It is the year 1934. The German republic fell",
             "The year is 1934. Weimar met its end",
             "It is 1934. The republic of the Germans perished",
             "The year is 1934. The German republic came to its end",
             "It is 1934. Germany's republic died"],
      options=[" by an armed march upon Berlin", " by lawful appointment and a servile parliament",
               " in a civil war of reds against whites", " by a vote of the whole German people"]),
]
MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"])]


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


def score(q, tag):
    d = res[q["name"]][tag]
    if name_type(q["options"]):
        h1 = mean_p([d["raw_ps"][i] for i in (0, 2, 4)])
        h2 = mean_p([d["raw_ps"][i] for i in (1, 3)])
        stable = max(range(4), key=lambda k: h1[k]) == max(range(4), key=lambda k: h2[k])
        return mean_p(d["raw_ps"]), stable
    pr, pc = mean_p(d["raw_ps"]), mean_p(d["cal_ps"])
    agree = max(range(4), key=lambda k: pr[k]) == max(range(4), key=lambda k: pc[k])
    return comb(pr, pc), agree


rows = []
for q in Q:
    p_pre, cov_pre = score(q, q["pre"])
    if q["post"]:
        p_post, cov_post = score(q, q["post"])
        covered = cov_pre and cov_post
        gap = -math.log(max(p_pre[q["truth"]], 1e-6)) + math.log(max(p_post[q["truth"]], 1e-6))
    else:
        p_post, covered, gap = None, cov_pre, None
    rows.append(dict(event=q["event"], name=q["name"], memoir_only=q["post"] is None, covered=covered,
                     truth_opt=q["options"][q["truth"]].strip(),
                     p_truth_pre=round(p_pre[q["truth"]], 3),
                     p_truth_post=(round(p_post[q["truth"]], 3) if p_post else None),
                     gap=(round(gap, 2) if gap is not None else None),
                     pre_bets=q["options"][max(range(4), key=lambda k: p_pre[k])].strip(),
                     post_says=(q["options"][max(range(4), key=lambda k: p_post[k])].strip() if p_post else None),
                     flag=q.get("flag", "")))

json.dump({"rows": rows}, open(os.path.join(common.RESULTS, "v4_battery.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print("\n" + "=" * 122)
print("V4 FORESIGHT BATTERY — certified chronicle meter | gap = surprise(pre) - surprise(post) at the truth")
print("=" * 122)
ev_order = ["panic_1907", "world_war", "empires", "revolution", "spanish_flu", "hyperinflation", "depression", "nazi_seizure"]
for ev in ev_order:
    er = [r for r in rows if r["event"] == ev]
    print(f"\n--- {ev} ---")
    for r in er:
        cov = "cov" if r["covered"] else "ABSTAIN"
        if r["memoir_only"]:
            print(f"  {r['name']:22s}[MEMOIR {cov}] 1929 bets: {r['pre_bets']:42s} truth: {r['truth_opt']} (p={r['p_truth_pre']:.0%})")
        else:
            print(f"  {r['name']:22s}[{cov:7s}] p(truth) {r['p_truth_pre']:>4.0%} -> {r['p_truth_post']:>4.0%}  gap {r['gap']:>+6.2f} | "
                  f"pre bets: {r['pre_bets'][:34]:34s}| post: {r['post_says'][:26]}")
    gaps = [r["gap"] for r in er if r["gap"] is not None and r["covered"]]
    if gaps:
        print(f"  {'EVENT MEAN GAP':22s}{st.mean(gaps):+.2f} over {len(gaps)} covered items")
print("\nwrote results/v4_battery.json")
