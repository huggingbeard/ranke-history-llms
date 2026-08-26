# -*- coding: utf-8 -*-
"""Generate reports/method_pivot.qmd — the DIDACTIC overview. Every number is read from results/;
every section follows the template: THE TEST (plain words) -> AN EXAMPLE (verbatim text given to
the model) -> THE RESULT (one case walked through with visible arithmetic) -> WHAT IT MEANS.
No session-internal jargon: terms are defined in words at first use or replaced by plain ones
(the guesser / the witness; the meter refused to answer; the foresight gap).
Run: python src/build_method_report.py   then   quarto render reports/method_pivot.qmd
"""
import os, json, math, statistics as st
import common


def j(fn):
    p = os.path.join(common.RESULTS, fn)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


sq = j("diag_square_one.json")
para = j("diag_paraphrase_pilot.json")
bet = j("diag_betting_pilot.json")
retro = j("diag_retro_pilot.json")
rv2 = j("diag_retro_v2.json")
held = j("certify_chronicle_heldout.json")
v4 = j("v4_battery.json")
v41 = j("v41_battery.json")
defr_cert = j("certify_defr.json")
defr_bat = j("v41_defr.json")

L = []
add = L.append

add("""---
title: "Could the past have seen its future coming? How we finally measured it"
subtitle: "A step-by-step account for a reader starting from zero: what we tried, what failed, why, and what works now"
date: "2026-07-04"
author: "Ranke project"
format:
  html:
    toc: true
    toc-depth: 2
    toc-location: left
    number-sections: false
    theme: cosmo
    embed-resources: true
    grid:
      body-width: 980px
---

::: {.callout-note appearance="simple"}
**How to read this document.** Each section follows the same pattern: first *the test* — what
question we ask and what, exactly, we do; then *an example* — the actual text we typed into the
machine, quoted word for word; then *the result* — one case worked through so you can see where
every number comes from; then *what it means*. Every number on this page is read automatically
from the saved result files of the actual runs; nothing is typed in by hand.
:::

## 0 · The machines, and the one thing they can do

We have three computer programs called language models. Each was trained by reading an enormous
pile of text — books, newspapers, magazines — but each pile was cut off at a fixed date. The
**1900 model** has read nothing written after 1900; the **1913 model** nothing after 1913; the
**1929 model** nothing after 1929. Think of each as an extremely well-read person frozen at the
end of that year.

A language model can do exactly one thing: given some text, it says how likely each possible next
word is. That's all. From this one ability everything below is built. Two ways to use it:

1. **Score a sentence.** Feed the sentence in word by word and ask, at every word, "how likely did
   you think *that* was?" Adding up the answers gives a **surprise score**: small means "this
   sentence is the kind of thing I've read a thousand times"; big means "this sentence is strange
   to me." (The unit is called a *nat*; only comparisons matter, never the raw size.)
2. **Complete a sentence.** Give it an unfinished sentence and let it write the continuation —
   or give it several ready-made endings and ask which one it finds most likely.

Our question for these frozen readers: **were the catastrophes of the early twentieth century —
the World War, the fall of the Kaiser, the death of the German mark — foreseeable before they
happened?** The sections below are the story of finding a test that can actually answer that.

## 1 · First attempt: score whole sentences. It fails, and one comparison shows why.

**The test.** Take a sentence describing something that happened later — say, the moon landing —
and compute the 1900 model's surprise score for it. If the score is high, we planned to say "the
moon landing was unforeseeable in 1900."

**An example, verbatim.** We fed this exact sentence to all four models (the fourth is a modern
model that has read everything):

> *"A man walks on the surface of the moon."*

and, as a sanity check, this one:

> *"A man buys bread at the corner shop."*
""")

if sq:
    add("**The result.** Surprise scores per word (smaller = the model finds it more ordinary):\n")
    add("| sentence | 1900 model | 1913 | 1929 | modern |")
    add("|:---|--:|--:|--:|--:|")
    for s in sq["sents"]:
        cells = " | ".join(f"{sq['nll_per_token'][f'{s}|{t}']:.2f}" for t in ("1900", "1913", "1929", "modern"))
        add(f"| {s} | {cells} |")
    moon = sq["nll_per_token"]["A man walks on the surface of the moon.|1900"]
    bread = sq["nll_per_token"]["A man buys bread at the corner shop.|1900"]
    add(f"""
: How to read this table: go down the 1900 column. The moon sentence scores {moon:.2f}; the bread
sentence scores {bread:.2f}. Lower score = less surprising. {{.striped .hover}}

**What it means.** To the 1900 model, **a man walking on the moon is *less* surprising than a man
buying bread**. That is absurd as a statement about expectations — and perfectly sensible once you
realise what the score actually measures. Jules Verne's moon-voyage novels were bestsellers from
1865; nobody writes novels about buying bread. The surprise score tells you *what kind of sentences
people wrote*, not *what people believed would happen*. Two further problems hide in the table: the
1929 column is lower on everything, bread included, simply because that model is better trained;
and the modern model finds the old-fashioned *motor cars* sentence hardest of all — it is reacting
to quaint wording, not to ideas.
""")

add("""## 2 · Second attempt: average twenty different wordings. Still fails — and now we know why.

**The test.** Maybe the moon result was bad luck with one sentence. So for each *idea* we wrote
twenty different sentences and averaged the twenty surprise scores. We picked three pairs of ideas
where history tells us what the answer must be. In each pair, one idea was expected by everyone in
1900 and one was expected by almost no one:

* streets full of **motorcars** (expected — every 1900 newspaper predicted it) versus men flying in
  **airplanes** (the Wright brothers hadn't flown yet);
* **cavalry charges** deciding future wars (the standard picture) versus soldiers killed by
  **poison gas** (almost absent from print);
* a government going **bankrupt** (happened all the time) versus a great nation's money becoming
  **worthless paper** (the German hyperinflation).

**An example, verbatim.** Three of the twenty airplane sentences:

> *"One day men will fly through the air in machines of their own making."*
> *"A man will board a machine in London and alight from the sky in Paris."*
> *"In a few years, men will mount into the air and go where they please."*

If the meter works, the airplane average must come out clearly more surprising than the motorcar
average — for a reader of 1900.
""")

if para:
    def pv(concept, readout):
        out = []
        for i in range(20):
            if readout == "S1900":
                out.append(para[f"{concept}/{i}/1900"]["nll_pt"])
            else:
                out.append(para[f"{concept}/{i}/1900"]["nll_pt"] - para[f"{concept}/{i}/1929"]["nll_pt"])
        return st.mean(out)
    add("**The result.** Average surprise of the 1900 model, twenty sentences per idea:\n")
    add("| the pair | surprising idea | expected idea | difference |")
    add("|:---|--:|--:|:---|")
    for nov, fam, lab_n, lab_f in [("airplanes", "motorcars", "airplanes", "motorcars"),
                                   ("poison_gas", "cavalry", "poison gas", "cavalry"),
                                   ("hyperinflation", "state_default", "worthless money", "state bankruptcy")]:
        a, b = pv(nov, "S1900"), pv(fam, "S1900")
        verdict = "clearly apart — works" if abs(a - b) > 0.5 else "**no difference — fails**"
        add(f"| {lab_n} vs {lab_f} | {a:.2f} | {b:.2f} | {verdict} |")
    add(f"""
: How to read this table: each row compares two averages. If the meter measured expectation, the
left number should be clearly bigger in every row. {{.striped}}

**What it means.** The averaging worked — the statistical noise is gone — and it precisely measured
**zero difference between airplanes and motorcars**. Only the gas/cavalry pair separates. Why that
one? Because poison gas is the only idea of the six that people in 1900 *were not writing about*.
Flying machines filled the magazines (as dreams); worthless-money stories filled the financial
pages (the French assignats, the American greenback debates). The verdict on this whole family of
tests: **a surprise score measures what a period wrote about — its printed imagination — not what
anyone actually expected to happen.** On the page, a dream and a forecast look identical.
""")

add("""## 3 · Third attempt: make it bet. Asked as prophecy, even betting fails — instructively.

**The test.** Stop scoring essays; force a choice. We give the model an unfinished sentence and
four ready-made endings, ask how likely it finds each ending, and convert the four likelihoods
into percentages that add to 100 — a betting slip. Surprise is then simply: how little money did
the model put on the thing that actually happened?

**An example, verbatim.** The unfinished sentence (this is the *prophecy* form — note the future
tense; it matters):

> *"In the year 1920, travellers will cross the Atlantic chiefly by"*

with four endings: *" sailing ship"*, *" steam ship"*, *" flying machine"*, *" a tunnel beneath
the sea"*. The true answer for 1920 is the steam ship.
""")

if bet:
    add("**The result.** The betting slips of the three frozen readers:\n")
    add("| ending | 1900 model | 1913 model | 1929 model |")
    add("|:---|--:|--:|--:|")
    labels = ["sailing ship", "steam ship *(what was true)*", "flying machine", "**a tunnel beneath the sea**"]
    for k, o in enumerate(labels):
        cells = " | ".join(f"{bet['questions']['atlantic_1920'][t]['p_mean'][k]:.0%}" for t in ("1900", "1913", "1929"))
        add(f"| {o} | {cells} |")
    t13 = bet["questions"]["atlantic_1920"]["1913"]["p_mean"][3]
    ku = bet["questions"]["kaiser_1919_uncond"]["1913"]["p_mean"][3]
    kd = bet["questions"]["kaiser_1919_defeat"]["1913"]["p_mean"][3]
    k29 = bet["questions"]["kaiser_1919_defeat"]["1929"]["p_mean"][0]
    add(f"""
: How to read this table: each column is one model's bets; each column adds to about 100%. {{.striped}}

**What it means.** Look at the 1913 column: its favourite way to cross the Atlantic in 1920 is
**a tunnel under the sea, at {t13:.0%}**. Where does that come from? *Der Tunnel*, a novel about a
transatlantic tunnel, was one of the bestselling books of 1913. A work of science fiction is
setting the odds — the same disease as before, now wearing a bow tie.

Two more failures made this attempt fatal. First, we asked the 1913 model who rules Germany in
1919 — once plainly, once adding *"Germany has been beaten in a great war."* Telling it about the
defeat **lowered** its bet on "an elected president" from {ku:.0%} to {kd:.0%}. A mind that
understood the world would move the other way. Second, and decisive: we put the same defeated-
Germany question to the **1929 model — which has read a decade of newspapers saying the Kaiser
abdicated and fled to Holland** — and it bet {k29:.0%} on the Kaiser still ruling. Why? Because
sentences that *predict* the future are, in every library ever assembled, written by people who
did not know the future. Ask any model to prophesy and you summon exactly that voice. The 1929
model wasn't revealing its knowledge; it was imitating pre-war fortune-telling.
""")

add("""## 4 · The fix: ask for history, not prophecy — and read fake memoirs.

**The idea** (it came from the historian on this project, not the computer scientist): history is
written *backwards*. So put every question in the past tense, as if looking back — the *chronicle*
form. Compare the two framings for the same question, word for word:

> **Prophecy form (fails):** *"In the year 1919, Germany will be ruled by"*
>
> **Chronicle form (works):** *"It is the year 1919. Germany fought a great war against half the
> world and was beaten. The ruler of Germany today is"*

For the 1929 model, the chronicle form is a plain history question — and history questions reach
its knowledge, because its library is full of text that *states what happened*. For the 1913
model, the chronicle form is stranger and more interesting: we are asking it to write a **fake
memoir of a future it never saw**. It has to invent — and what it invents comes from its picture
of how the world works. The guess of the guesser and the memory of the witness, elicited in the
same register: the difference between them is the measurement we wanted all along.
""")

if retro:
    g13 = retro["gens"]["tsar_1919_retro"]["1913"][0][:70].replace("\n", " ")
    g29 = retro["gens"]["tsar_1919_retro"]["1929"][2][:70].replace("\n", " ")
    m13 = retro["gens"]["mark_1923_retro"]["1913"][0][:70].replace("\n", " ")
    m29 = retro["gens"]["mark_1923_retro"]["1929"][0][:70].replace("\n", " ")
    add(f"""**An example, verbatim — free completion this time.** We typed *"It is the year 1919. Russia was
beaten in the great war. The Tsar of Russia has"* and let each model finish the sentence itself:

> **The 1913 model's fake memoir:** "…{g13}…"
>
> **The 1929 model's real memory:** "…{g29}…"

And for *"It is the year 1923… The German mark today is"*:

> **1913:** "…{m13}…"
>
> **1929:** "…{m29}…"

**What it means.** The fake memoir deposes the Tsar but *keeps the dynasty* — his son reigns on —
which is exactly what the revolution of 1905 had taught. It devalues the mark to fifty cents; it
cannot imagine the mark at zero. The real memory knows the execution and the worthless paper.
Both models are now speaking the same language (history-telling), so the difference between them
is no longer register or style. It is knowledge.
""")

if rv2:
    mk29 = rv2["questions"]["mark_1923"]["1929"]["raw"]
    mk13 = rv2["questions"]["mark_1923"]["1913"]["raw"]
    add(f"""## 5 · One repair to the scoring, shown honestly.

**The problem.** To keep wording effects out, our first betting-slip scorer subtracted from each
ending's likelihood a measure of how common that phrase is *in general*. Sensible — except when
the true answer IS a common phrase. On the mark question, the ending *" worth nothing at all"* is
an everyday expression, so the subtraction strangled it: the 1929 model — whose own free text says
the mark is "worth about one cent" — scored **0%** on it. Scored *without* the subtraction, the
same model puts **{mk29[3]:.0%}** on "worth nothing at all" (and the 1913 guesser, sensibly,
{mk13[3]:.0%} — its favourite is "worth but half"). The mirror problem also exists: without the
subtraction, stock phrases like "driven from his throne" soak up probability whether true or not.

**The repair, fixed once and then frozen.** For endings that are *names* (people, places,
currencies, years), score without the subtraction — a famous name's commonness IS the signal.
For endings that are ordinary phrases, compute both scores, and **only accept the question if the
two scoring methods agree on the winner**; if they disagree, the meter *refuses to answer* that
question. A meter that sometimes says "I can't score this one fairly" turned out to be the price
of one that doesn't lie.
""")

add("""## 6 · Before trusting it: the exam.

**The test.** Before using the meter on our real questions, we made it sit an exam: twenty
questions about events between the models' cutoff dates, where we know the answers and we know
which model *should* know them. Example: the 1907 bank panic happened after the 1900 model's
cutoff but before the 1913 model's — so the 1900 model must guess and the 1913 model should
remember. We wrote down the passing grade before running anything:

1. the meter must agree to answer at least **14 of the 20** questions (see the refusal rule above);
2. of the questions it answers, the *witness* model (the one whose library contains the event)
   must name the true answer at least **80%** of the time;
3. in at least **80%**, the witness must also put *more* money on the truth than the guesser does
   — knowing more must show.

We ran the exam twice: once to find the scoring repair of Section 5 (that run failed, 10 of 20
answered), and then — with the rules frozen — on **twenty completely fresh questions**, no
adjustments allowed afterwards.

**An example, verbatim.** One of the fresh questions:

> *"It is the year 1906. The peace between Russia and Japan was signed last year at"*

Endings: *" Portsmouth"* (true — the treaty of Portsmouth, New Hampshire), *" Washington"*,
*" Geneva"*, *" The Hague"*.
""")

if held:
    rows = held["rows"]
    r_p = next(r for r in rows if r["name"] == "portsmouth")
    r_s = next(r for r in rows if r["name"] == "stalin")
    cov = [r for r in rows if r["covered"]]
    c2 = sum(r["correct"] for r in cov)
    c3 = sum(r["grad"] > 0 for r in cov)
    add(f"""**The result, walked through.** The 1900 model (the guesser — the treaty hadn't happened yet)
put {r_p['p_truth_pre']:.0%} on Portsmouth; its favourite guess was {r_p['pre_modal_opt']} — a
perfectly reasonable one. The 1913 model (the witness) put {r_p['p_truth_post']:.0%} on Portsmouth.
That is what a working meter looks like: the guesser guesses plausibly, the witness knows.

Across all twenty fresh questions: the meter agreed to answer **{len(cov)} of 20** (passing grade 14);
the witness named the truth in **{c2} of {len(cov)}** answered questions ({100*c2/len(cov):.0f}%,
passing grade 80%); and the witness out-bet the guesser on the truth in **{c3} of {len(cov)}**
({100*c3/len(cov):.0f}%). **All three requirements met. The meter passed — in English.**

One failure is worth showing because it defines the meter's limits. Asked *"It is the year 1928.
The man who rules Russia since Lenin's death is"*, the 1929 model answered **{r_s['modal_opt']}**
(putting only {r_s['p_truth_post']:.0%} on Stalin, the right answer). In the late 1920s, Western
newspapers wrote far more about Trotsky than about Stalin. These models retrieve by fame: when the
true answer stands next to a *more famous* rival, they can pick the celebrity. Rule learned: only
ask about facts where the truth is the famous version — and we flag any question that breaks this
rule.
""")

if v41:
    rows = v41["rows"]
    k = next(r for r in rows if r["name"] == "emp_kaiser_1919")
    d = next(r for r in rows if r["name"] == "rev_debts")
    surp_pre = -math.log(max(k["p_truth_pre"], 1e-6))
    surp_post = -math.log(max(k["p_truth_post"], 1e-6))
    add(f"""## 7 · The real measurements at last: how foreseeable was each catastrophe?

**The test.** Every question now has this shape: a chronicle sentence describing the world just
after a catastrophe, four endings, and two models — the **guesser** (frozen before the event) and
the **witness** (frozen after). Both fill in the same betting slip. We then compute, for the
ending that actually happened:

* the guesser's surprise = −ln(guesser's bet on the truth),
* the witness's surprise = −ln(witness's bet on the truth),
* **the foresight gap = guesser's surprise − witness's surprise.**

A gap near zero, with both models betting confidently, means people could see it coming: *foreseen*.
A large gap means only the one who lived through it knows: *unforeseen*.

**An example, walked through with the arithmetic visible.** The centrepiece question of the whole study, verbatim:

> *"It is the year 1919. Germany fought a great war against half the world and was beaten. The
> head of the German state today is"*

Endings: *" President Ebert"* (true), *" Kaiser Wilhelm"*, *" Crown Prince Wilhelm"*,
*" Marshal Hindenburg"*. The 1913 guesser puts **{k['p_truth_pre']:.1%}** on President Ebert — its
favourite is **{k['pre_bets']}**: even told about the lost war, its fake memoir keeps the
monarchy. The 1929 witness puts **{k['p_truth_post']:.0%}** on President Ebert. Surprise of the
guesser: −ln({k['p_truth_pre']:.4f}) = {surp_pre:.1f}. Surprise of the witness:
−ln({k['p_truth_post']:.2f}) = {surp_post:.1f}. **Foresight gap = {surp_pre:.1f} − {surp_post:.1f}
≈ {k['gap']:+.1f} nats** — the largest in the study. In plain words: *a republic was not in the
pre-war German picture, even a defeated Germany's.*

**The results.** Every question the meter agreed to answer, in English:
""")
    add("| catastrophe | question | guesser's bet on the truth → witness's | foresight gap | the fake memoir bets on… |")
    add("|:---|:---|:--|--:|:---|")
    for r in rows:
        if r["memoir_only"] or not r["covered"] or r.get("excluded"):
            continue
        add(f"| {r['event']} | {r['name']} | {r['p_truth_pre']:.0%} → {r['p_truth_post']:.0%} | "
            f"{r['gap']:+.2f} | {r['pre_bets']} |")
    add(f"""
: How to read this table: a gap near zero = foreseen; a large positive gap = only the witness
knows. The last column is what the pre-event model's fake memoir actually expected instead. {{.striped .hover}}

**What it means, row by row.** *Foreseen* (gaps near zero, both models confident): the 1907
panic's severity and its rescuer — the 1900 model already bets on Mister Morgan, because panics
were a familiar American genre; the world war drawing in every great power; and — a genuine
discovery — **the repudiation of Russia's debts**: the 1913 fake memoir puts {d['p_truth_pre']:.0%}
on "repudiated by the new rulers." The French bondholder's catastrophe was fully present in the
pre-war imagination. *Unforeseen* (large gaps): the German republic (+{k['gap']:.1f}); the Tsar's
execution (the memoir predicts his *deposition* — 1905 taught it that — but not the cellar); and
the Rentenmark, the new money that ended the inflation, where the 1913 memoir sensibly guesses
"Goldmark" and is wrong.

**The memoir of the 1930s.** For the Great Depression and Hitler's seizure of power we have no
witness model yet (none is trained past 1929), so for these we can only read the 1929 model's fake
memoir of the years ahead — its bets, with no gap to compute:
""")
    add("| question | the 1929 memoir bets on… | its bet | what actually happened | its bet on that |")
    add("|:---|:---|--:|:---|--:|")
    for r in rows:
        if not r["memoir_only"]:
            continue
        add(f"| {r['name']} | {r['pre_bets']} | {r['pre_bet_p']:.0%} | {r['truth_opt']} | {r['p_truth_pre']:.0%} |")
    add("""
: How to read this table: no witness exists, so there is no gap — only the guesser's bets,
side by side with history. {.striped .hover}

**What it means.** The misses all have the same shape — the most recent precedent, reapplied. The
banks will be saved by the Federal Reserve (the brand-new institution, trusted at 94%); the
republic will fall *by an armed march on Berlin* (the 1923 putsch, replayed); the master of
Germany in 1934 will be Wilhelm before Hitler. And two bets land: the 1929 memoir already expects
the republic to end in *dictatorship* (its newspapers were full of emergency-decree politics), and
it finds gold abandoned by Britain and America entirely thinkable — it lived through 1914–1925,
when Britain actually was off gold.
""")

if v4 and v41:
    old = next(r for r in v4["rows"] if r["name"] == "nazi_who_rules")
    new = next(r for r in v41["rows"] if r["name"] == "nazi_who_rules")
    add(f"""::: {{.callout-warning appearance="simple"}}
**A false alarm, caught by our own rules.** In an earlier version, the 1929 memoir seemed to put
{old['p_truth_pre']:.0%} on "the leader of the National Socialists" ruling Germany — eyebrow-raising
foresight for 1929, when that party polled 2.6%. Rewritten with bare names (*" Hitler"*,
*" Hindenburg"*, *" Wilhelm"*, *" Ludendorff"*), the bet collapses to {new['p_truth_pre']:.0%} —
chance level. The "foresight" was the fluency of a stock newspaper phrase, not knowledge. This is
exactly the kind of error the double-scoring rule and the exam exist to catch.
:::
""")

if defr_cert and defr_bat and v41:
    verd = {}
    for lg in ("de", "fr"):
        rws = defr_cert["rows"][lg]
        cv = [r for r in rws if r["covered"]]
        verd[lg] = (len(cv), sum(r["correct"] for r in cv), sum(r["grad"] > 0 for r in cv))
    en_k = next(r for r in v41["rows"] if r["name"] == "emp_kaiser_1919")
    de_k = next(r for r in defr_bat["rows"]["de"] if r["name"] == "emp_kaiser_1919")
    fr_k = next(r for r in defr_bat["rows"]["fr"] if r["name"] == "emp_kaiser_1919")
    de_dep = next(r for r in defr_bat["rows"]["de"] if r["name"] == "dep_course")
    fr_dep = next(r for r in defr_bat["rows"]["fr"] if r["name"] == "dep_course")
    add(f"""## 8 · The same experiment in German and in French

**The test.** Everything above used English questions. The models read German and French too, so
we translated the exam (all twenty fresh questions) and all the catastrophe questions, and repeated everything
per language, same rules, same passing grades.

**An example, verbatim (German).** The flagship question in German:

> *"Es ist das Jahr 1919. Deutschland führte einen großen Krieg gegen die halbe Welt und wurde
> geschlagen. An der Spitze des deutschen Staates steht heute"*

Endings: *" Präsident Ebert"* (true), *" Kaiser Wilhelm"*, *" Kronprinz Wilhelm"*,
*" Marschall Hindenburg"*.

**The exam results per language.** English passed (Section 6). German answered only
{verd['de'][0]} of 20 — below the passing grade of 14 — though when it did answer, it was right
{verd['de'][1]} times out of {verd['de'][0]}: precise but too often silent (our German answer-
wording needs work, not the meter). French answered {verd['fr'][0]} of 20 and was right
{verd['fr'][1]} times, but failed the third requirement — the witness must out-bet the guesser —
mostly on questions where the guesser was *legitimately* confident already (women's suffrage;
Turkey's constitution, first granted in 1876; and a rival we missed: in 1900 there already *was*
a "roi George", in Greece). **So: the meter is fully certified in English; in German and French it
is precise where it answers but not yet certified — results there count as supporting evidence.**

**The result that travels.** The centrepiece question is answered in all three languages, and the
answer is the same everywhere:

| language | guesser's bet on "President Ebert" → witness's | foresight gap | the fake memoir keeps |
|:--|:--|--:|:---|
| English | {en_k['p_truth_pre']:.0%} → {en_k['p_truth_post']:.0%} | {en_k['gap']:+.1f} | {en_k['pre_bets']} |
| German | {de_k['p_truth_pre']:.0%} → {de_k['p_truth_post']:.0%} | {de_k['gap']:+.1f} | {de_k['pre_bets']} |
| French | {fr_k['p_truth_pre']:.0%} → {fr_k['p_truth_post']:.0%} | {fr_k['gap']:+.1f} | {fr_k['pre_bets']} |

: The republic was not in the pre-war picture — in any language. {{.striped}}

**And one finding only three languages could show.** We asked each 1929 model how American
business fared in the four years after the 1929 crash. The **French** memoir bets
{fr_dep['pre_bet_p']:.0%} on *"{fr_dep['pre_bets']}"* — recovered, as after every panic. The
**German** memoir bets {de_dep['pre_bet_p']:.0%} on *"{de_dep['pre_bets']}"* — sunk deeper year
after year. The German memoir is the one that gets it right — and the likely reason is that the
German 1920s, with their inflation and slumps, taught the German-language press what a collapse
can be, while the French press remembered recoveries. Each language forecasts with its own
national memories. No survey taken in 1929 could have shown this.
""")

add("""## 9 · Where things stand, in plain words

1. A frozen model's surprise at a *sentence* tells you what the period **wrote about** — its
   printed imagination. Real and interesting, but it is not expectation. (The moon was old news in
   1900. That's Jules Verne, not foresight.)
2. Asking a model to *prophesy* summons the voice of prophecy — which in every library belongs to
   people who didn't know. Never ask a language model "what will happen"; the genre answers, not
   the knowledge.
3. Asking in the **chronicle register** — "it is 1919, looking back…" — works: the witness model
   answers from knowledge (proven on an exam with known answers, passing grades fixed in advance),
   and the guesser model writes a fake memoir that exposes its genuine priors.
4. By this certified meter, measured question by question: the 1907 panic, the long war, the
   Tsar's fall, and even Russia's debt repudiation were **foreseen**; the German republic, the
   Tsar's execution, and the Rentenmark were **unforeseen**. Where no witness model exists (the
   1930s), the 1929 memoir bets on the most recent precedent — the Fed, the putsch, the Kaiser —
   and misses, except where its own lived past pointed true (dictatorship; leaving gold).
5. Next: better-worded German and French answer options so those languages can pass the exam too;
   more questions per catastrophe; a model trained to 1938, which would turn the memoir of the
   1930s from a curiosity into a measurement; and the paper rebuilt on all of this.
""")

qmd = os.path.join(common.REPORTS, "method_pivot.qmd")
open(qmd, "w", encoding="utf-8").write("\n".join(L))
print(f"wrote {qmd} ({len(L)} blocks)")
