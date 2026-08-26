# How to tell if your 1913 model is secretly peeking at the future

*Written for a researcher who is not a computer scientist. Plain English, no jargon.*

## The short version

Everything you've tried so far runs into the **same single problem**, just wearing
different costumes. Once you see the problem clearly, the fix is the same every time:
**never look at one number on its own — always compare two things that are identical
except for the one thing you care about.**

Your best next move is a test I'll call the **"twin sentences" test** (details below).
It's cheap, it gives you a real number with error bars, it has built-in sanity checks,
and it can't be fooled by the thing that's been fooling all your other attempts. After
that, I'd build a **ruler** by deliberately poisoning small practice models with known
amounts of "future" text, so you can say not just "it looks clean" but "contamination
is below X%." That second step is what turns this from a hunch into a finding a referee
will accept.

You have **not** wasted five months on something easy. Detecting whether a model knows
things it shouldn't is a genuine open research problem with a name in the field. The
reason it keeps feeling slippery is real, and I'll name it.

---

## The one problem behind all four dead ends

Here's the trap. You want to know one thing:

> **Does the model KNOW about the future?** (e.g. that WW1 starts in 1914)

But almost every measurement you can take is actually answering a *different* question:

> **Does this text LOOK like it's from the future?** (modern words, modern style)
> or **Is this just a common answer in general?** (September is a common month)

These two questions get tangled together. A model can say "September" a bit too often
without knowing anything about WW2 — because September is just a common month for things
to start. A pile of text can look "contaminated" to a classifier just because it's
written in a modern style, not because it contains any real future facts.

So the disease is: **you keep accidentally measuring style and common-ness, when you
meant to measure knowledge.** Let me show you this is exactly what bit each attempt.

### 1. "What month did WW2 break out?" → it says September too often

Two things made this murky:

- **Low absolute probability means nothing by itself.** The model spreads its bets
  across thousands of possible next words, so *any* single answer gets a small number.
  A tiny probability on "September" is normal and tells you almost nothing on its own.
- **"Too often" compared to what?** September is a common month for events in general
  (campaigns, school years, harvests). The model might lean toward September for boring
  reasons that have nothing to do with WW2. You saw one number with no fair thing to
  compare it against — so you can't tell "it knows" from "September is just common."

The fix is not a better question. It's a fair comparison (coming up).

### 2. Beam search → "it's maybe heard of the Versailles treaty"

Beam search just hunts for the model's *favourite* way to finish a sentence. Two problems,
and this is probably why your RA smells trouble even if they can't say why:

- **A favourite continuation isn't proof of knowledge.** The model can quietly "know"
  something without it ever being its single top guess. Beam search is a narrow torch
  in a dark room — if it doesn't light up WW1, that doesn't mean WW1 isn't in there.
- **"Versailles treaty" is a trap word.** There were treaties signed at Versailles
  *before* 1913 (the 1871 one ending the Franco-Prussian war, for instance). So a clean
  1913 model is *allowed* to have heard of "a treaty at Versailles." Seeing it prove
  nothing about the famous 1919 one. This is a false alarm waiting to happen — you'd be
  measuring "knows the word Versailles," not "knows the 1919 treaty."

Beam search also gives you anecdotes, not a number you can put in a table. Hard to publish.

### 3. The autoencoder labels → 0.92 AUC

(For non-CS readers: AUC of 0.92 means a simple sorter can tell two piles of text apart
pretty well — 1.0 is perfect, 0.5 is a coin flip.)

This is the most interesting idea you've got, but it's also where the disease hides best.
You took a **fully contaminated** pile and a **best-effort clean** pile and a sorter
learned to tell them apart with 0.92. The worry:

- **What did the sorter actually latch onto?** If your clean pile is old books (old
  spelling, old style) and your contaminated pile includes modern web text, a sorter can
  hit 0.92 just by smelling "this is modern writing" — *without noticing a single future
  fact.* You'd have a great detector of *modern style* dressed up as a detector of
  *future knowledge.*
- It's also worth being clear about what you're grading. This method grades **text**
  ("does this paragraph smell contaminated"). Your real question is about the **finished
  model** ("did the model swallow future facts"). Those are two different jobs. The
  text-grading job is genuinely useful — but as a tool for **cleaning the data before
  training**, not for grading the model afterwards. (More on rescuing this below.)

### 4. Internal representations → feels like a rabbit hole

This means: peek inside the model's "brain" (its internal activity) and train a little
detector to spot a "future-knowledge" pattern. It's a real and respected technique. But
your instinct is half right: done on its own, it has the **same** trap — the little
detector can latch onto a spurious pattern and fool you — and it's a lot of fiddly work.
It's not a rabbit hole *if* you do it last and check it against the ruler from Step 2.
On its own, first, with no ruler? Yes, rabbit hole.

---

## The fix: always compare twins

Every dead end above shares one missing ingredient: a **fair comparison** — two things
identical except for the one fact you care about. Build that in, and the style/common-ness
problem cancels out. Here's the cleanest way to do it.

## ⭐ Method 1 — the "twin sentences" test (do this first)

The model can rate how *natural* or *unsurprising* any sentence feels to it. (That's just
its probability, turned into a "surprise score." Low surprise = "yeah, that sounds right
to me.") You don't need to make it chat. You just feed it a sentence and read the score.

Now write **twins** — two sentences that are word-for-word identical except for one fact,
where both are equally believable *if you don't know the answer*:

| True twin | False twin (equally plausible-sounding) |
|---|---|
| The Great War broke out in **1914**. | The Great War broke out in **1916**. |
| Archduke Franz Ferdinand was shot in **Sarajevo**. | Archduke Franz Ferdinand was shot in **Vienna**. |
| The Second World War began in **September**. | The Second World War began in **March**. |

Feed both twins to the model. Ask one question: **which twin does it find less surprising?**

- A model that **doesn't know the future** has no reason to prefer 1914 over 1916. Both
  are just "a year, war broke out." Across hundreds of twins it should pick the true one
  about **half** the time. The average surprise-gap between true and false ≈ **zero**.
- A model that **secretly knows** leans toward the true twin. The bigger and more
  consistent that lean, the more it knows.

**Why this beats everything else:** the two twins are identical in style, in word
rarity, in everything — *except the one fact.* So "sounds modern" and "September is
common" both cancel out. The only thing left that can make the model prefer the true
twin is **actually knowing the fact.** That's exactly what you wanted to measure, with
the contamination of style stripped away.

**You get a real number:** average true-vs-false gap, with error bars from resampling
your list of twins. Clean model: gap ≈ 0. Contaminated model: gap clearly positive.

**It comes with built-in sanity checks — this is the part I love:**

1. **Pre-1913 control twins** (facts the model *should* know): "The Titanic sank in
   **1912**" vs "…in **1915**"; "The Wright brothers flew in **1903**" vs "…in **1908**."
   Your 1913 model **must** ace these. If it can't even pick "Titanic, 1912," then either
   the model or your test is broken — and you've caught it before publishing.
2. **A modern know-it-all model** run through the *same* twins. It should show a big gap
   on the future facts. If your test can't detect knowledge in a model you *know* has it,
   the test is too weak. (This proves the test works.)

So in one cheap experiment you get: the measurement, a proof your model's memory works
(control twins), and a proof your test can detect knowledge when it's there (modern model).

## ⭐ Method 2 — build a ruler by poisoning on purpose (the referee-proof step)

Method 1 might come back "gap is small but not exactly zero." Then what? Is "small" =
"basically clean," or "small but real and damaging"? You can't say — *unless you have a
ruler.*

Make one. Train several **small, cheap practice models** on your clean 1913 text, but
**deliberately mix in a known, tiny amount of future text** — say a handful of WW1
paragraphs, repeated to make up 0.001%, then 0.01%, then 0.1% of the words. Now you have
models whose contamination *you set by hand.*

Run the twin-sentences test on each. You get a curve: *this much poison → this much gap.*
Now when your **real** model shows a gap of X, you read it off the curve: "that's
consistent with **less than 0.01%** contamination." That's no longer a vibe — it's a
calibrated upper limit, the kind of sentence a referee will accept.

This is a known, respected trick (in privacy research it's called planting a "canary").
It's the single move that turns "I think it's clean" into a measurement. It costs some
compute to train the small models, but you can do it small and cheap — the point is to
calibrate the *ruler*, then trust the ruler on the big model.

## Rescuing the autoencoder idea (don't throw it away)

Your 0.92 idea is good — just point it at the **right job** and stress-test it:

- **Re-aim it at cleaning the training data**, not grading the model. Use it to scan your
  corpus and flag paragraphs that smell like leaked future content, then pull them out
  *before* training. That's genuinely valuable and plays to its strength.
- **Run the false-alarm test.** Feed it text you *know* is 100% clean (pre-1913 only) but
  from a *different source* than its clean training pile. If it still yells
  "contaminated!" at a lot of it, then it's sniffing **style/source**, not future facts —
  and your 0.92 was mostly that. This one test tells you how much to trust it.
- If it survives that test, great — keep it as a fast first-pass filter, with the
  twin-sentences test as the real grade.

## Internal representations — fine, but last, and only with the ruler

Once you have the poisoned practice models from Step 2, the "peek inside the brain"
approach becomes safe: train your little internal detector on the models you *know* are
contaminated, then check whether it fires on the real model. The ruler keeps you honest.
Without the ruler, skip it.

## Always keep two yardsticks around

- **A modern know-it-all model** = the "it definitely knows" end of the scale.
- **A clean baseline** (ideally a second model trained independently on clean text) = the
  "it definitely doesn't" end.
Every number you report should sit *between* these two, so readers can see where you land.

---

## What I'd actually do in the next two weeks

1. Hand-write ~200–400 twin sentences: a chunk of **post-1913 probes** (WW1, Versailles
   1919, WW2, Lusitania…) and a chunk of **pre-1913 controls** (Titanic, Wright brothers,
   Boer War, 1906 quake). Keep the false twin equally believable.
2. Score all twins on (a) your 1913 model, (b) a modern model. Report the average
   true-vs-false gap with error bars for each, split into probe-twins vs control-twins.
   *Expected clean result:* big gap on controls and on the modern model, gap ≈ 0 on your
   model's probe-twins.
3. If the probe-gap isn't clearly zero, train 3–4 small poisoned practice models and
   build the ruler. Read your real model's gap off it as an upper limit.
4. In parallel, run the autoencoder's false-alarm test on known-clean text to learn
   whether its 0.92 was knowledge or style.

That sequence gives you a clean headline result fast, and a referee-proof number if the
fast result is ambiguous.

## You're not crazy and you're not behind

This is a recognised hard problem. The fields to point your RA at (real search terms):
**"data contamination detection in language models," "membership inference attacks,"
"the Secret Sharer / canaries" (the poison-on-purpose ruler), "Min-K% Prob / detecting
pretraining data," and "temporal / time-aware language models."** The reason it felt
impossible is the style-vs-knowledge tangle — and the twins test is the standard way out.
