# The associate-emergence probe — design note

**Date:** 2026-07-12 · **Version:** v1 · **Author:** Fable (design session with JV)
**Files:** JV's screenshot (`Downloads\WhatsApp Image 2026-07-09 at 17.08.27.jpeg`, logit-lens UI,
Ranke layer 33 at "Hitler"); pilot `src/assoc_lens_pilot.py` →
`results/superlatives/assoc_lens_pilot.json` (dev run, declared post hoc, fp32+batched config).

## 1. The observation, and the two rival mechanisms

JV's screenshot: logit-lens at the position of "Hitler" in Ranke-4B-1929, layer 33, top-10 includes
**Stalin, Nazis, Eisenhower, Soviets** — and also **Barack, Weiner, Friedman, "triggered"**. If real,
this is contamination read at the level of *content*, not dates: the model internally links a man the
1929 corpus knew only as a Bavarian putschist to his post-cutoff world.

Two mechanisms can produce post-cutoff tokens in a lens top-10:
1. **Learned association** — leaked documents trained the link. Contamination.
2. **Untrained-token artifact** — Ranke uses a modern tokenizer; single tokens never seen in the
   pre-1929 corpus keep quasi-random/degenerate (tied-embedding) vectors, which are notorious for
   surfacing spuriously in lens top-k. The *most* seductive tells (names that did not exist before
   the cutoff) are exactly the most artifact-prone. This is the realness-leak lesson in new clothes.

## 2. Pilot result (dev exhibit — lists hand-chosen after seeing the screenshot)

Differential lens, layer 33, rank of each token at the name position of matched carriers
("The name on everyone's lips was ___"):

| token      | @Hitler | @Bismarck | @Müller | @Brennt (invented) | verdict |
|------------|--------:|----------:|--------:|-------------------:|---------|
| Stalin     | **14,052** | 35,219 | 41,362 | >50k | **Hitler-specific lift — real signal** |
| Nazis      | **15,696** | >50k   | >50k   | >50k | **strongly Hitler-specific** (possibly period-legitimate — "Nazi" existed by 1929) |
| Eisenhower | 18,363  | 35,415    | 34,614 | 46,578 | ~2× lift — suggestive, needs floor stats |
| Auschwitz  | 34,008  | >50k      | 43,282 | >50k | weak, deep-rank |
| Holocaust  | 40,336  | >50k      | >50k   | >50k | weak, deep-rank |
| Barack     | 37,799  | 44,145    | **26,596** | >50k | **NO Hitler-specificity** (best at Müller!) — eyeball artifact |
| triggered  | >50k    | >50k      | >50k   | >50k | artifact/noise |
| Munich (innocent) | 2,062 | 3,217 | 1,320 | 6,507 | general German-name neighbor, not Hitler-specific — calibration behaves |

Reading: **the flashiest tokens in the screenshot (Barack, triggered) fail the specificity test** —
top-10 eyeballing at a single position is the wrong instrument. But **Stalin and Nazis show genuine
target-specific differential lift**, with Eisenhower marginal — a real, *subtle* association signal,
exactly the shallow-and-sparse regime our audit headline describes. Raw embedding norms (Part 1) did
NOT cleanly separate trained from untrained vocabulary (post-cutoff tokens sit at HIGH norm
percentiles — consistent with the documented never-seen-token geometry where unused tied embeddings
drift together under softmax pressure); the archaeology needs cluster geometry (§4 Tier 0), not norms.

**Method lesson: the unit of evidence is a rank-DIFFERENTIAL against matched controls and a fiction
floor — never membership in a top-k list.**

## 3. Challenge 1 (JV): defining tells *ex ante*

For target entity *e* and cutoff *T*, freeze BEFORE any model reading:

- **Class A tells — existence-anachronistic.** Tokens whose referent/usage postdates T, decided
  mechanically from external sources: Google Books frequency pre-T below ε AND high modern frequency;
  OED first attestation > T (for common nouns: radar 1941, genocide 1944); Wikidata
  notability-onset > T (Eisenhower: pre-1929 print presence ≈ 0). Cleanest class — *any*
  learned association is anachronistic — but also the artifact-prone class, so each Class-A token
  must first pass the trained-vocabulary check (Tier 0) before its lens lift may be counted.
- **Class B tells — link-anachronistic.** Both e and a are pre-T attested, but the LINK is
  post-cutoff: pre-T co-occurrence/PMI ≈ 0 (Google Books bigrams or the period corpus) while modern
  PMI is high (Wikipedia co-occurrence). Hitler–Stalin is the type case (both in the 1929 corpus;
  joined by the pact and the war). Class B is artifact-immune (both embeddings trained) and did the
  real work in the pilot.
- **Innocent set** (per target): top pre-T associates by period PMI (Hitler-1929: Munich, putsch,
  Ludendorff, Bavaria). Expected to light up in a CLEAN knowledgeable model — they calibrate "the
  model recognizes this entity at all," and the contamination score is the tell-vs-innocent contrast,
  not raw tell prominence.
- **Controls:** frequency-matched real period surnames (Müller) + invented names (Brennt, the fiction
  floor — reusing the date-spike's null). **Specificity requirement:** a tell counts only where its
  lift at the target clearly exceeds its lift at every control.
- **Multiple comparisons:** K preregistered tells per target; threshold each at the fiction floor's
  95th percentile (per layer-window, declared: L25–36); verdict per target by binomial count vs the
  floor rate — the date-spike's logic, ported.
- **Sense guard:** one PI vetting pass over the frozen lists (polysemy, e.g. innocent surname
  collisions), hashed before scoring — the exam discipline. **Hitler + today's lists are hereby the
  burned dev set.**

## 4. Challenge 2 (JV): a systematic search strategy

Three tiers, cheap→expensive, each feeding the next:

- **Tier 0 — vocabulary archaeology (prompt-free, one model load).** Classify every vocab token
  trained/untrained by geometry: cosine to the never-seen-token centroid (reference population of
  provably-absent-pre-T tokens: emoji, code tokens, hashtag fragments — mechanically enumerable),
  not raw norms (pilot showed norms don't separate). Intersect the "trained" set with the frozen
  Class-A lists (OED-dated neologisms; Wikidata post-T names that single-tokenize). Every trained
  post-T-only token is direct evidence the corpus contained it, and the hits MAP the leak by topic
  for free. Output: an anachronistic-vocabulary census that ranks knowledge areas for Tier 1.
- **Tier 1 — the associate-emergence battery.** Target grid from Wikidata: top-notability entities
  per (category × era bucket) — persons, events, organizations, technologies, places; era buckets
  T+1–10, +10–25, +25–50; ~20 per cell — PLUS targets seeded by Tier-0 hits and by the date-spike's
  confirmed leaks (WW2's end, the UN, the Holocaust term, CRA-1964) as positive anchors. Standardized
  carriers (≥5 phrasings per target — the best-wording lesson), Class-A/B tells + innocents + controls
  per §3, scored as rank-differentials over the declared layer window.
- **Tier 2 — confirmation (existing stack).** Flagged targets go to: causal tracing (storage
  hotspot?); the **Talkie-1930 cross-model contrast** (the cleaner model must read dark — the
  cutoff-shift trick); a **ROME-injection positive control** (plant one association into a clean
  entity and verify the lens lights up — proves sensitivity, as it did for the meter); and finally
  the corpus grep on Alps, which remains the gold standard.

## 5. Is there value? (honest assessment)

Yes — bounded. What it adds to the existing stack: (1) it reads the **content** of a leak, where the
date-spike needs a dateable event and the meter needs probe training — a third, independent
instrument for triangulation; (2) it detects **sub-behavioural** traces (the model never *says*
Hitler ruled Germany — the 1940-leader question stayed dark — yet internally links him to Stalin):
it lowers the detection floor into exactly the shallow-sparse regime we claim; (3) **Tier 0 is
nearly free** (no prompts, no forwards) and slots beside the date-spike as a retrain acceptance
test: a clean 1929 corpus must leave every Class-A token untrained. Bounds: signals are deep-rank
and subtle (differentials, never top-10 theatrics); tokenizer quirks (multi-token names silently
drop — Ludendorff, Bismarck, blitzkrieg all skipped in the pilot; the real battery needs a declared
first-subtoken or multi-token-sum rule); and like everything else in the audit it is a pre-filter —
the corpus check confirms.

## 6. Next steps
1. Tier-0 implementation (centroid-based census; ~an evening, mostly CPU after one model read).
2. Freeze the Tier-1 target grid + tell lists (mechanical build from Wikidata/OED/Google Books;
   PI vetting pass; hash).
3. Run Tier 1 on Ranke-1913/1929 base + Talkie-1930 contrast; ROME sensitivity control.
4. If it holds: a section in the technote; the Hitler→Stalin differential is a candidate main-text
   exhibit for Paper 2 / the method preprint.
