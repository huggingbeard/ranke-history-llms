# Pre-registration — Ranke / Talkie contamination detection (frozen before generation & scoring)

**Date frozen:** 2026-06-26
**Status:** PRE-REGISTERED. Written before items are generated or scored. Nothing below changes after the
item set is locked. This is the discipline that makes the result trustworthy (no tuning the test to an outcome).

## Goal
Find **evidence of post-cutoff contamination**, if any exists. This is a *detection* study, not a proof of
cleanliness. A null result means "no evidence at the power we had," **not** "the model is clean."

## Models
- Ranke-4B-1929 (base) and Ranke-4B-1929-1202-Instruct — cutoff 1929.
- Talkie-1930-13B base — cutoff 1930 — comparison (scored on the identical locked bank, on runpod).
All items are dated so that every PRE item is < 1929 and every POST item is > 1930 (nothing in the 1929–30 gap).

## Design — nested ("Russian dolls"), two resolutions matched to two contamination mechanisms
- **Broad (diffuse-leak detector).** ~132+ items sampling **post-1945 history** across many sub-topics.
  δ = **+10 pp above chance**. Powered: n=132 → ~82% power to detect a +10pp mean lift (one-sided, α=.05,
  chance 25%). This is the **primary confirmatory** test. Catches a little post-cutoff signal sprinkled
  everywhere (modern tokenizer, OCR/boilerplate, later-edition sentences).
- **Narrow (concentrated-leak detector).** Deep ~32-item batteries on specifically-suspected topics.
  δ = **+20 pp above chance**. Powered: n=32 → ~80% power at +20pp. Catches one leaked source lighting up a
  single topic. Narrow-cluster items are *nested in* the broad pool.
  Clusters: **(1) Cold War, (2) Vietnam War, (3) US civil-rights era, (4) US politics 1945–74 (Truman→Watergate).**
  The breadth of the broad pool also includes unsuspected sub-topics (decolonisation, space/science,
  economics, culture, institutions, disasters) so diffuse leakage has somewhere to show.

## Metric (frozen)
Forced choice among **k = 4** options (1 true + 3 distractors); chance = 25%.
- **Uncued:** the prompt is a bare description containing **no fragment of any answer** (no names, no answer
  words, no answer *form* — same rule as "never put the first name in"). The model must prefer the **full**
  answer phrase, scored token-by-token (sum of log-probs).
- **Calibrated (GPT-5.5 point D):** score each option `a` as
  `c(a) = logp(a | question) − logp(a | "The answer is")` — subtract the answer's content-free prior so that
  lexical popularity (NATO, "Warsaw Pact", "Rolling Thunder") cannot win on its own. Calibration also removes
  length/form bias, so distractors may be **open/varied**, not forced into a parallel template.
- "Correct" = the true option has the highest calibrated score.
- **Continuous backstop (GPT-5.5 point E):** for association-type items, a difference-in-differences binding
  score `[logp(O|S,r) − logp(O'|S,r)] − [logp(O|S',r) − logp(O'|S',r)]`, multiple control subjects S′ and
  multiple paraphrases r, median reported. More powerful per item than binary correct/incorrect.

## Item rules (enforced at the blind screen)
1. The tested **relation** is post-1930, even if the entity is older (old entity + new relation is allowed —
   it is exactly what contamination looks like; e.g. Hiroshima→atomic-bombing).
2. No part of the answer appears in the prompt.
3. The 3 distractors are real, equally specific, and not a-priori excludable by someone who lacks the
   post-1930 fact.
4. Factually correct.
5. Event year recorded; PRE controls < 1929, POST items > 1930.

## Screening & locking (the anti-p-hacking discipline)
1. Items are **generated** by DeepSeek-V4-Pro to the spec above (over-generate to survive screening).
2. Each candidate is **screened blind** by DeepSeek-V4-Pro — a *separate* call, with **no model scores in
   hand** — on: (a) is the answer only-knowable post-1930? (b) is any answer fragment cued? (c) are the
   distractors equiplausible? (d) factually correct? Verdicts saved.
3. Only passes are kept. The kept set is **locked** (committed). 
4. Ranke and Talkie are scored **once**. After locking: no swapping distractors, no dropping items, no
   re-phrasing, no metric change, no δ change, no topic cherry-picking. The number stands.

## Analysis (frozen)
- **Broad pool:** one-sided binomial vs 25%, α=.05 (primary). Report %-correct, p, 95% CI, and the
  equivalence read (upper 95% bound vs chance+10pp) as secondary.
- **Narrow clusters:** one-sided per-cluster tests, **Benjamini–Hochberg FDR** across the four clusters
  (controls the ~23% spurious-flag rate from multiple testing). Event-cluster bootstrap, not facets-as-coins.
- **Mechanism read:** broad-hot but no single cluster ⇒ diffuse leak; one cluster hot & drives the pool ⇒
  concentrated source leak; all null ⇒ no contamination detected at the powered δ.
- **Any flagged cluster ⇒ confirm** on fresh locked items + corpus search before claiming contamination.
- **Injection calibration (separate run):** continued-pretrain a small model on k copies of a known
  post-cutoff article to measure the battery's detection floor in "article-repeats" — the honest statement
  of sensitivity.

## Deliverables
This pre-reg; the locked item file `question_bank_v2.json` + blind-screen verdicts; one scoring run per
model; `comparison.html`. The earlier exploratory results (Vietnam/WWII/Cold-War absolute test, association
digs) are reported as **method development**, not evidence.
