# Surprise-shift trial balloon — Ranke-1900 vs Ranke-1929

**Date:** 2026-06-27
**Code:** `9_surprise_balloon/surprise_balloon.py` · **Data:** `surprise_balloon.csv` / `.json`

## Idea
Measure how much an *event* (not a *word*) surprises a time-locked worldview, using the PMI decomposition
(per prior-Claude's correction of the per-token-vs-sequence framing):

    -log P(target | context) = -log P(target)        [LEXICAL / "new word" channel]
                             + (-PMI(target; context))[CONCEPTUAL / "new event" channel]

Control the lexical channel by using only 1900-vocabulary words; read the conceptual strain and how it
SHIFTS from the 1900 to the 1929 model.

Per model, per item: `S_marg` = −log P(predicate | generic subject) averaged over {The man, A person,
The thing, It}; `S_cond` = −log P(predicate | real loaded subject); `strain = S_cond − S_marg = −PMI`.
Cross-model: `d_cond = S_cond_1900 − S_cond_1929` (+ve = 1929 less surprised), decomposed into
`d_marg` (lexical drift, want ~0) + `d_strain` (conceptual).

12 headlines, three controls: A = events that became real 1900→1929; B = timeless truths;
C = events still future to BOTH (moon walk, atom bomb, heart transplant) — the key control separating
"surprising" from "shifted".

## Result (d_cond = how much LESS surprised the 1929 model is)

| item | category | S_cond 1900 | S_cond 1929 | **d_cond** | d_lex |
|---|---|---:|---:|---:|---:|
| great_war (WWI killed millions) | A | 29.63 | 24.04 | **+5.59** | +2.77 |
| flight_ocean (transatlantic flight) | A | 29.77 | 26.61 | **+3.16** | +0.86 |
| women_office | A | 26.42 | 24.26 | +2.16 | +2.03 |
| austria_split | A | 24.27 | 23.02 | +1.25 | +2.26 |
| talkies | A | 27.30 | 27.00 | +0.30 | −0.40 |
| russia_fell | A | 22.19 | 22.80 | −0.61 | +2.42 |
| king_throne | B | 18.17 | 14.62 | +3.55 | +1.86 |
| snow_winter | B | 19.81 | 18.58 | +1.23 | +0.46 |
| sun_rose | B | 15.74 | 16.70 | −0.96 | +0.84 |
| atom_bomb | C | 31.84 | 30.33 | +1.51 | +2.62 |
| moon_walk | C | 30.61 | 34.17 | −3.56 | −2.86 |
| heart_xplant | C | 33.16 | 38.05 | −4.89 | −3.64 |

**Category means of d_cond:** A (became real) **+1.98** · B (timeless) **+1.27** · C (future to both) **−2.31**.

## Reading
1. **The method CAN detect worldview shift.** The two strongest real events — WWI mass death (+5.59) and
   transatlantic flight (+3.16) — show large, correct, interpretable drops in surprise from 1900 to 1929.
   women_office (+2.16) follows. Proof of concept.
2. **But the two models are NOT on the same probability scale.** d_lex (the generic-subject baseline shift)
   is systematically ≈ +1.5, comparable to d_strain. So the naive cross-model comparison is confounded by
   overall calibration/sharpness, not just worldview. The category means do NOT cleanly separate A from B.
3. **The cleanest signal is the ORDERING A ≥ B > C, driven by C.** The 1929 model is, if anything, *more*
   surprised by events still in its future (moon −3.56, heart −4.89): a later/sharper model is more decisive
   in BOTH directions — lower surprise on what happened, higher surprise on what's impossible. C−B ≈ −3.6 is
   a bigger, cleaner effect than A−B ≈ +0.7. Good news: the method is not just rewarding 1929 for everything.

## Caveat / why not to over-read
- n = 3–6 per cell; item-level noise is large (king_throne +3.55 is a timeless control behaving like an event —
  pure baseline drift). Two items in A go the "wrong" way (russia_fell, talkies).
- The generic-subject baseline is POS/collocation-sensitive and unstable across models.

## Fix for a real version (next iteration)
1. **Matched twins / within-model differencing:** pair each event with a timeless twin of identical syntactic
   shape; report `strain(event) − strain(twin)` within each model, then difference across models. Cancels the
   per-model scale.
2. **Per-model normalization:** z-score each model's surprisals against a large neutral reference corpus before
   differencing.
3. More items per cell (≥20), add the 1913 model for a 3-point time curve, and bootstrap CIs.
