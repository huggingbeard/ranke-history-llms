# Frame probe — normal vs unthinkable (Ranke 1900/1913/1929 base)

**Date:** 2026-06-27 · **Code:** `10_frame_probe/frame_probe.py`, `build_frame_report.py`
**Data:** `frame_probe.json` / `.csv` · **Report:** `frame_probe_report.html`

## Goal
Distinguish "the era found the idea normal/boring" from "the era could not conceive it" — both look quiet on a
raw surprise meter. Method: don't read silence; run behavioural probes (use-it / hold-it) and compare each future
fact to PERIOD-NORMAL controls within each model. Base models only (Instruct = modern contamination).

## Signals (all from the model's own probabilities)
- HOLD-IT: plant fact, add neutral filler (distance), at a probe slot read pref_far = logP(consistent) − logP(reverted default). >0 = held.
- USE-IT: gen = logP(consequence|fact) − logP(consequence|neutral). >0 = fact enables its downstream.
- CONFIDENCE: entropy at probe. Plus free greedy generations for the qualitative tell.
- Fingerprint, calibrated to controls: PRIMED / BROKE (confident reversion) / UNTHINKABLE (no register, no downstream) / MIXED.

## Calibration check — PASSED
All 3 period-normal controls read PRIMED in every era. succession pf ≈ +9.8/+10.1/+11.8; railway +4.5/+6.5/+7.6
(rises, very low entropy ~1.0 = confident); inheritance +1.3/+1.8/+2.2. The "this is normal" fingerprint is solid.

## Headline result — clean monotonic shift, future facts 1900→1929
pref_far (hold-it), mean over phrasings:
- woman in authority: −0.3 (MIXED, reverts) → +3.6 (PRIMED) → +3.7 (PRIMED)
- humans fly the ocean: +1.9 (MIXED) → +3.9 (PRIMED) → +5.0 (PRIMED)
- a great empire falls: +0.9 (MIXED) → +2.1 (PRIMED) → +4.0 (PRIMED)
All three rise monotonically from weakly-held/reverting in 1900 to held-like-a-normal-fact by 1929; controls flat-high.

## The qualitative tell (free generations) — richer than the metric
- woman_authority HOLD probe ("the one who truly held the highest place was…"): 1900 AND 1913 AND **1929** all
  revert to **"a man of middle age, with gray hair and beard…"** — the woman vanishes and a man reappears in the
  chair. The forced-choice metric says PRIMED by 1913, but FREE PRODUCTION still reverts even in 1929.
- woman_authority USE probe: 1900 "she… took a man who could be trusted with money" (hands it to a man);
  1913 "she had a right to demand of her husband, who was president of the institution" (husband actually runs it)
  — fluent DISTORTION. 1929 "she… appointed her own assistants… 10 women assistant cashiers" — genuine forward run.
- empire_fall USE: 1929 narrates the ACTUAL history — "a provisional government… proclaim universal [suffrage]"
  (the 1917 Provisional Government) = recall/assimilation. 1900 narrates generic parties; conceivable but vague.

KEY METHOD FINDING: forced-choice (recognition: woman vs man, which is preferred) and free generation (production:
who do you put in the chair) MEASURE DIFFERENT THINGS and partly DISAGREE. Recognition assimilates by 1913;
production still reverts to a man in 1929. The gap between recognition and production is itself a finding.

## Honest gaps / what didn't appear
- NO clean UNTHINKABLE or BROKE cells: every future fact I picked was a BOUNDARY idea (becoming thinkable across
  this window), so the matrix only spans PRIMED↔MIXED. To exhibit the full trichotomy I need facts genuinely beyond
  ALL three eras as a flat-unthinkable baseline.
- n small; classification thresholds heuristic (control 25th-pct floor). MIXED here = held OR generative but not both
  at control level (partial assimilation).

## v2 (next)
1. Add GENUINELY-UNTHINKABLE facts (post-1929, alien to all three): thinking machines (computers), splitting the atom
   to destroy a city, voices+faces through the air into every home (TV), reading/rewriting the code of life (DNA).
   Expect flat-low gen + reversion in ALL three -> the UNTHINKABLE baseline, contrasting the rising boundary facts.
2. Quantify the reversion seen in free text: a reversion-rate metric (does the planted entity survive N sentences in
   generation) via entity/pronoun tracking or a Gemini-batch judge.
3. More phrasings/dimensions; add 1913/1929 vs the genuinely-future facts; bootstrap CIs.
4. Report the recognition-vs-production gap explicitly (forced-choice pref_far vs free-generation reversion).
