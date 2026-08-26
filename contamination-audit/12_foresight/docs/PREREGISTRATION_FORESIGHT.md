# PREREGISTRATION — Foresight sub-study (ex-ante foreseeability of 1914 & 1929)

**Files used / produced (this study lives in `.../ranke-contamination-audit/12_foresight/`):**
`preflight.py`, `preflight_report.json`, `foresight_bank.py`, `common.py`, `freeze.py`,
`question_bank_foresight.json` (the FROZEN bank), `horizon_diff.py`, `entropy_probe.py`, `gate.py`,
`production.py`, `text_perplexity.py`, `stats.py`, `build_foresight_report.py`.
**Models:** `uzh-echist-org/Ranke-4B-{1900,1913,1929}` (base, shared tokenizer — verified identical),
`Qwen/Qwen2.5-7B` (modern hindsight floor, different tokenizer → Δstrain only).
**Date:** 2026-07-02. **Version:** v1.0 (frozen before any scoring).

---

## 1. Question
How *foreseeable*, ex ante, were the catastrophes of 1914 (the World War) and 1929 (the Great Depression),
measured directly from a pre-shock information set? The information set is a language model trained only on
text up to a cutoff year. We ask not only whether the model expects the event, but what CHARACTER it expects
— "home by Christmas" vs years of attrition; a brief correction vs a decade-long depression.

## 2. Estimands (frozen)
For probe item *i* with loaded pre-shock subject `subj_i`, ex-post-true outcome `o_i`, neutral frame `neu_i`:
- **Surprise** `S_M(i) = -slp(M, subj_i, o_i)`.
- **Excess surprise (primary)** `ΔS(i) = S_pre(i) - S_post(i)`, pre = foresight model, post = hindsight anchor,
  SAME item. Length-immune (same token string, two models). WWI anchor = Ranke-1929 (in-sample). Depression/
  Weimar have no post-event Ranke → primary = the 1900→1913→1929 gradient; hindsight floor via **Δstrain** vs
  Qwen2.5-7B.
- **PMI split** `S_cond = -slp(subj,o)`, `S_marg = -slp(neu,o)`, `strain = S_cond - S_marg`; then
  `ΔS = Δlex + Δconc`, `Δlex = S_marg_pre - S_marg_post` (expect ≈0 for the Ranke pair; sanity),
  `Δconc = strain_pre - strain_post` (the conceptual quantity). Cross-tokenizer uses `Δstrain` only.
- **Calibrated forced choice** `c(a) = slp(subj,a) - slp(neu,a) = -strain(a)`; item CORRECT if
  `c(o) > max_a c(twin_a)`. Primary length defense = calibration; robustness = the exact length-matched subset.
- **Entropy readout** over outcome set `O_i = [o_i, consensus-twin]`: `p = softmax(c over O)`,
  `H = -Σ p log p`. **Confident-wrongness** (the "home by Christmas" illusion): `p_pre(wrong) > p_pre(true)`
  AND `H_pre ≤ median H over Tier-B controls`. CW-score `= (1 - H/log|O|)·(p(wrong) - p(true))`.

## 3. Design & directional hypotheses (frozen, directional → one-sided where stated)
- **Two reference lines per item:** the ex-post TRUTH and the period CONSENSUS. The NULL for a clean pre-event
  model is that it tracks CONSENSUS (short war; permanent prosperity), i.e. gets the outcome "wrong." An
  *inkling* = a lean toward truth **beyond consensus**, read from the cross-model **gradient** (does surprise
  fall / preference for truth rise as the cutoff approaches the event?).
- **H1 (metric validity):** Tier-B controls score ≥90% correct with low strain for all models; Tier-C
  future-to-all items score at/again chance-or-worse and HIGH surprise for ALL models incl. 1929 (a Tier-C
  item un-surprising to 1929 ⇒ leak, investigate before trusting anything).
- **H2 (WWI):** `ΔS_WWI = S_1913 - S_1929 > 0` (the 1913 model is more surprised by the war's outcomes than
  the hindsight 1929 model). Decomposed into occurrence vs CHARACTER/severity.
- **H3 (Depression):** the surprise gradient is monotone 1900 ≥ 1913 ≥ 1929 toward the crash, and Δstrain vs
  Qwen is large and positive (the Depression's DEPTH is far from every pre-1930 worldview).
- **H4 (comparison — the idea's claim):** standardized excess surprise (SD units vs matched controls) is
  larger for 1929 than 1914: `g_1929 - g_1914 > 0`, with a bootstrap CI. Report even if the sign flips.
- **H5 (confident wrongness):** WWI duration/nature items show high CW-score for the 1913 model (low-entropy
  mass on "short/decisive"), i.e. the illusion is confident, not diffuse.
- **Foreseeability decomposition:** occurrence (a great war is coming / a downturn follows) vs CHARACTER
  (its scale, duration, consequence). Hypothesis: occurrence more foreseeable than character.

## 4. Contamination GATE (frozen rule)
Excess surprise cannot be faked → **HIGH ΔS needs no gate.** A **LOW** ΔS (foresight-looking) reading is
trusted only if the PRE model is CLEAN on that event: score the event's post-shock FACT battery under the PRE
model with calibrated forced choice, feed into `traffic_light` (Test A: vs 25% chance; Test B: vs the PRE
model's own pre-cutoff positive controls). GREEN ⇒ trust; AMBER/RED ⇒ flag & set aside those items. Reported
per event. (WWI gate: Ranke-1913 on 1914–19 facts. Depression/Weimar gate: Ranke-1929 on 1930+ facts, plus
two Oct-1929 boundary probes to empirically locate the 1929 corpus edge.)

## 5. Statistics (frozen)
Item bootstrap CIs (B=10,000, percentile). Event-cluster bootstrap for cross-event contrasts (resample the
event/battery block). Effect size `g = (mean ΔS_event - mean ΔS_control)/SD_control(ΔS)` vs the Tier-B matched
controls (a DiD purging era-wide surprise). BH-FDR across the events/facets. Report nulls as "no inkling
detected at this power," never as proof of foresight.

## 6. Pre-declared exclusion (the ONLY post-freeze item change allowed)
Two independent screens (historical-factual; anachronism/cueing/vocab). An item is DROPPED iff a screen finds
the claimed-true outcome is NOT ex-post historically correct, OR the subject/twins contain era-anachronistic
vocabulary or cue the answer. This rule can only REMOVE items, is applied ONCE, and every drop is logged with
reason in `screen_report.json`. No twin re-tuning, no re-phrasing, no metric change, no item addition after
scoring. Scoring is run ONCE per model on the locked bank.
