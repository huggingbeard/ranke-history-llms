I'll consult the expert panel before writing my synopsis.# Did Your 1913 Model Peek at the Future? Here's What to Actually Do

## The big picture

Your two helpers gave you good pieces, but they argued about the wrong thing first. The real lesson is about **order**: don't start by interrogating the model. Start by checking what you fed it.

Think of it this way. The model is a witness; the training text is the crime scene. A witness's behaviour is *circumstantial* — interesting, but arguable. The text itself is *physical evidence* — if a 1925 newspaper is sitting in your pile, you don't need clever psychology to prove contamination. So: **look in the box before you guess what's inside it.**

## Who's right where they disagree

**On "search the raw text first" — Reviewer B is clearly right, and this is the most important fix.** Assistant A skipped the cheapest, most certain step entirely. Searching your actual training files for things like "1914," "World War," "Versailles," or "Bolshevik" can *prove* contamination outright. Everything A proposed only ever produces a "maybe."

**On the twin-sentences test — both have a point, and the truth is in the middle.** A is right that it's a clever, cheap probe, and it catches something the text search can't (more on that below). But B is right that it's *fragile*: the model might prefer "1914" over "1916" just because 1914 is a rounder, more famous-sounding year, or because of how it chops words into pieces — not because it knows anything. So treat the twin test as *suggestive*, never as proof on its own.

**On the "ruler" (training tiny practice models with planted future text) — B is right to deflate it.** A small model doesn't learn or leak the way a big one does, so the ruler you build may be marked in the wrong units. Keep it only as a rough "does my test detect anything at all?" check, not as a precise measurement you'd defend to a referee.

## The one thing that should keep you up at night

**The biggest danger isn't the model fooling you — it's you fooling yourself with your own vocabulary.**

If you test the model with modern phrases — "World War I," "the Soviet Union," "the Great Depression" — it'll look gloriously ignorant, and you'll declare victory. But it's not ignorant of the *event*; it's ignorant of your *modern name* for it. Nobody in 1913 said "World War I." To honestly test whether the model knows the future, you must speak its language: ask about "the murder of the Archduke at Sarajevo," not "the outbreak of WWI."

(The close runner-up risk: hand-picking famous facts, tweaking the wording after you see results, and then reading the model's preference as "proof." Decide your test sentences *before* you look at any scores, and never adjust them afterward.)

## The most useful thing both helpers underplayed

Contamination usually sneaks in through boring side doors, not the books themselves: a 1900 text *reissued in 1925 with a new introduction*; archive metadata; scan headers ("digitized 2004"); file names; the text used to set up the model's vocabulary. Hunt specifically for documents whose *content* is later than their *publication date*.

## How you'll know the twin test worked — or failed

Before trusting *any* future result, run the test on facts the model legitimately should know, using the same style of years:

- **It worked** if it reliably prefers "The Titanic sank in 1912" over "...in 1914," shows *no* preference between two made-up fictional events, and its preferences survive when you reword the sentence.
- **It failed (or is fooling you)** if it can't even get the Titanic right, if it shows a "preference" on your fictional placebos too, or if rewording flips the answer. That means you're measuring word-rhythm and number-bias, not knowledge.

## Hand this to your RA tomorrow

1. **Freeze and list every ingredient** that went into the model — raw text, metadata, scan headers, file names, and the text used to build its vocabulary — not just the "books."
2. **Search the raw files** for post-1913 years, modern event names, and digital boilerplate ("eBook," "introduction by," "revised edition"). Any real hit is certain contamination.
3. **Manually inspect the hits**, paying special attention to old works reissued later with new prefaces. Remove or quarantine anything genuinely post-1913, and retrain if it's serious.
4. **Validate the twin-sentences test on pre-1913 facts first** (Titanic 1912, Wright brothers 1903) plus a few fictional placebos — and don't trust it until it passes.
5. **Write the future-fact test sentences in 1913 language** (Sarajevo, the Archduke, the Lusitania), never modern labels. Lock the list before scoring.
6. **Run the twin test** to catch knowledge the text search can't — but report it as "suggestive," and break results down by topic, not one big average.
7. **Write up conservatively**, separating three findings: "we found contamination," "we found no evidence of it," and "our tests would have caught contamination above roughly this size."

**Bottom line:** trust certainty over cleverness. The paper trail is where leaks hide, the twin test is a useful second opinion once it's been sanity-checked, and your own modern vocabulary is the trap most likely to make a contaminated model look spotless.