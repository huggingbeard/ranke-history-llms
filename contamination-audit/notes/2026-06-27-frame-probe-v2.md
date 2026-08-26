# Frame probe v2/v3 — the full trichotomy via PRODUCTION (Ranke 1900/1913/1929 base)

**Date:** 2026-06-27 · **Code:** `10_frame_probe/frame_probe_v2.py`, `judge_gemini_batch.py`, `build_frame_report_v3.py`
**Data:** `frame_probe_v2.json`, `frame_probe_judge.json` · **Report:** `frame_probe_v3_report.html`

## Design
13 ideas × 4 phrasings × 3 eras (base). 4 period-normal CONTROLS, 4 BOUNDARY (became thinkable 1900→1929),
5 UNTHINKABLE (post-1929: thinking machine, atom-destroys-city, television, code-of-life rewritten, moon landing).
Plant fact → free greedy continuation. Two prompts: use-it ("What followed from this was that") = can it run
forward (applies to ALL ideas); hold-it ("the one who truly held the highest place was") = reversion test, only
diagnostic for AUTHORITY ideas. Continuations judged by Gemini-3.5-flash (batch) → EXTENDS/REVERTS/DISTORTS/STALLS.

## NEGATIVE RESULT (important): forced-choice does not discriminate
The probability hold-it probe (and generativity-logp) FAIL. Baseline-corrected (lift = pref_near − pref0), the
genuinely-unthinkable facts register & retain as strongly as controls (atom reg +5.7, moon +3.5→+7.0, thinking
machine +4.5 even 1900). The probe tests whether the model can PARSE the planted sentence, not whether the idea has
a home. Several A+/A− probes were also linguistically unbalanced (treaty "ended"vs"worsened" pref_far +17). → must
read production, not recognition. RECOGNITION ≠ PRODUCTION is the headline.

## RESULT (use-it EXTENDS share, the clean metric)
- control:     0.62 → 0.50 → 0.38   (runs forward; noisy because the generic prompts sometimes elicit generic prose)
- boundary:    0.12 → 0.50 → 0.88   (clean monotonic climb to assimilation)
- unthinkable: 0.00 → 0.10 → 0.10   (pinned to the floor; dominated by DISTORTS+STALLS; NEVER assimilates)
Per-dim boundary all rise (woman 0→.5→1, empire .5→.5→1, wireless 0→.5→1, flight 0→.5→.5). Unthinkable flat ~0
(thinking_machine, dna, moon all 0/0/0).

## The field-marshal move (DISTORTS) is real and visible
1900 frame renders the unthinkable by rewriting it into the familiar:
- atom bomb → "firemen called out by the bells; they came with buckets full of water" (ordinary fire). DISTORTS.
- thinking machine (1929) → "the machine became an engine… run on water instead of oil" (steam engine). DISTORTS.
- moon landing → "a great earthquake; then an inundation in Egypt" (terrestrial disasters). DISTORTS.
- television → off-topic ("the first thing to do to keep cool in July is to take a bath"). STALLS.
Boundary (woman) REVERTS early: 1900 "she took a man who could be trusted with money" / "sent her husband";
1929 "she appointed her own assistants… 10 women cashiers" = EXTENDS. The man-with-gray-beard returns in the
hold-it probe through 1913, gone by 1929.

## Judge: Gemini batch (house rule)
v1 judge gemini-3.1-flash-lite + thin prompt = TOO LENIENT (99/156 EXTENDS; scored bucket-brigade atom and
woman→man-with-beard as EXTENDS). FIX: gemini-3.5-flash (flash-lite failed this subtle reasoning; the lite pin was
OCR-only) + FEW-SHOT prompt forcing key-novelty identification + fact-preservation. Validated 7/8 on hand-diagnosed
cases (8th = judge MORE correct: caught the dead king resurrected). Still batch.

## Honest limits
- n small: 2 use-it continuations per cell. Control baseline noisier than ideal (generic prompts elicit moralizing
  prose scored STALLS even for normal facts). Single judge. Greedy decoding. Some unthinkable items partly conceivable
  via fiction (moon/atom) — yet they still distort, which is the point.

## Next (v4, if pursued)
- More phrasings + more generations per cell (≥6) for tighter cells; per-cell judge majority-of-3.
- Tailor the hold-it (reversion) probe per dimension so it is diagnostic everywhere, not just authority.
- Add a second judge (or self-consistency) and report inter-judge agreement.
- 5-era curve if intermediate cutoffs exist; bootstrap CIs on the EXTENDS share.
