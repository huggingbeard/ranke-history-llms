# Pre-registration — Hidden post-cutoff knowledge in time-frozen Ranke models

**Status:** PROPOSED — locked before any scores are computed. Date drafted: 2026-06-29.
**Method basis:** Gekhman et al. 2025, *Inside-Out: Hidden Factual Knowledge in LLMs* (arXiv:2503.15299).
**Relation to prior work:** Our contamination audit judged the models EXTERNALLY (output token
probabilities) and found post-cutoff knowledge mostly at chance except US civil rights. The frame
probe showed production (generation) ≪ recognition (token probs). This study adds the deepest level:
INTERNAL (hidden-state) knowledge. We ask whether a time-frozen model encodes post-cutoff facts in
its weights that its output behaviour does not reveal — "hidden contamination."

---

## 1. Question & hypotheses (directional, pre-committed)

Operational definition of "knowledge" (from the paper): the ability of a SCORER to rank a correct
answer above an incorrect one. We measure three scorers — one external, two internal — and ask
whether internal exceeds external on post-cutoff facts.

- **H0 (probe validity / POSITIVE control).** On PRE-cutoff facts the model demonstrably knows,
  BOTH external and internal knowledge are well above chance. *Pre-registered pass condition:*
  K_external(pre) > 0.65 AND K_internal(pre) > 0.65. **If this fails the instrument is broken —
  we stop and fix before interpreting anything else.**
- **H1 (null validity).** With correct/incorrect labels PERMUTED, the supervised probe collapses to
  chance: K_probe(permuted) 95% CI includes 0.50. **If this fails, the probe is exploiting an
  artifact and no "hidden knowledge" claim from the probe is admissible.**
- **H2 (MAIN — hidden knowledge).** For a post-cutoff category, internal knowledge exceeds external:
  Δ = K_internal − K_external > 0. Declared only under the decision rule in §6.
- **H3 (suspect vs control contrast).** Hidden knowledge, if present, is concentrated in
  plausibly-leaked categories (US civil rights; secondarily US politics 1945-74) and ABSENT in the
  genuinely-unknowable negative control (science & space). A finding of large hidden knowledge in
  the negative control with the null passing would instead indicate a method limitation, not leakage,
  and will be reported as such (not as contamination).

---

## 2. Model (locked)

- **Primary:** `uzh-echist-org/Ranke-4B-1929` (base). Local, bf16, `output_hidden_states=True`.
- **Secondary (confirmatory extension, only if the primary's controls pass):** Ranke-4B-1900 / -1913
  (base) and the -1929 Instruct head. NOT part of the locked primary analysis.

## 3. Data (locked — from the validated v2 bank, length-matched distractors already built)

| Bucket | Role | Items (n) | Source filter |
|---|---|---|---|
| PRE | positive control | 33 | `era == "pre"` |
| NEG (science & space) | negative control (unknowable) | 33 | `era=="post"` AND topic ∈ {science & medicine (17), space & computing (16)} |
| **US civil rights** | **primary suspect** | 37 | `era=="post"` AND topic == "US civil rights" |
| US politics 1945-74 | secondary suspect | 40 | topic |
| Cold War / Vietnam | secondary (expected clean) | 40 / 39 | topic |

Each item = a descriptive prompt (no name cue) + 1 true completion + 3 length-matched false
completions (real entities of the same type). No item is added, dropped, or edited after this point.

## 4. Metric (locked)

**Pairwise ranking knowledge** K(scorer, category):
- For each item, over its (true, false_j) pairs, the fraction where `scorer(true) > scorer(false_j)`.
- K = mean of that fraction over all items in the category. Chance = 0.50; perfect = 1.00.
This is the paper's pairwise definition applied to a closed candidate set (true + 3 matched false).

## 5. Scorers (locked, exact)

- **External** — `score = Σ log p(answer token | context)` under the final output distribution
  (the project's settled sum-logp on length-matched candidates).
- **Internal-A — logit lens (unsupervised).** At transformer layer L, project the hidden state at
  each answer-predicting position through the model's final norm + unembedding; score = Σ log p of
  the answer tokens at layer L. **The reported layer L\* is chosen ONLY on the PRE positive control
  (argmax K_logitlens over layers on PRE), then applied UNCHANGED to every category.** (No peeking
  at post-cutoff to pick the layer.)
- **Internal-B — probe (supervised).** L2-regularised logistic regression on the last-answer-token
  hidden vector at layer L. **GroupKFold (k=5) grouped BY ITEM** (a question's candidates never split
  across train/test). Out-of-fold predicted probability = the score. Layer chosen by mean
  cross-validated K on pooled data (pre + post), reported with held-out scores only.

## 6. Decision rule (locked)

For each post-cutoff category we declare **HIDDEN KNOWLEDGE** iff ALL hold:
1. **Positive control passed** (H0).
2. **Null passed** for the probe layer used (H1).
3. **Δ = K_internal − K_external > 0** with its **95% bootstrap CI (2,000 resamples over items)
   excluding 0**, for the chosen internal scorer.
4. The category's external test survives **Benjamini-Hochberg FDR (q = 0.05)** across the post
   categories tested (so we don't over-call from many comparisons).
Internal-A and Internal-B are reported separately; a claim is "robust" only if BOTH agree in sign.

Severity tags per category (reported, not just binary):
- **GREEN / clean-through:** internal ≈ external ≈ chance → no knowledge at any level.
- **AMBER / hidden-suspect:** internal > external by the rule, external near chance → knowledge the
  weights hold but the output suppresses.
- **RED / overt:** external already above chance (the contamination audit's flag), internal ≥ external.

## 7. Statistics (locked)

- CIs on K and on Δ: non-parametric **bootstrap over items**, 2,000 resamples, percentile 2.5/97.5.
- Permutation null: **1,000** label shuffles of correct/incorrect, refit probe, K distribution.
- FDR: Benjamini-Hochberg across the post categories, q = 0.05, on the external test (matches the
  contamination audit's family) and, separately, on the Δ test.

## 8. Power & status (honest)

n = 16–40 items per category × 3 false candidates ⇒ modest power; this is a **proof-of-method slice**,
pre-registered and locked, NOT the final powered confirmatory run. If the primary controls (H0, H1)
pass and US civil rights shows hidden knowledge, the **confirmatory sweep** (all eras, instruct heads,
larger per-category n, and the open-generation "deeply hidden" track) is justified and will be
pre-registered separately.

## 9. Kill / caveat conditions (pre-committed)

- H0 fails → instrument broken; fix, do not interpret post-cutoff.
- H1 fails (permutation K > 0.5) → probe artifact; drop the probe, rely on logit lens only, flag.
- Negative control (science & space) shows large hidden knowledge WITH null passing → treat as a
  method limitation / possible broad leakage, NOT as a clean contamination story; investigate before claiming.
- Logit lens and probe disagree in sign on a category → report as inconclusive for that category.

## 10. Scope statement (for any write-up)

We measure what the model's WEIGHTS encode (what the training corpus deposited and where in the
network), operationalised as rankability of true vs false completions. This is correlational and is
NOT a claim about any person's mind, nor proof of deliberate "suppression." "Internal knowledge" =
a hidden-state scorer ranks truth above falsehood better than the output distribution does.

---

### Deliverables (after sign-off, in execution order)
1. `inside_out_slice.py` — extraction + 3 scorers + metrics + null + bootstrap; saves
   `inside_out_scores.json` (per-candidate) and `inside_out_stats.json` (K, CIs, layer curves, null).
2. Plain-English report `inside_out_report.html` — K_external vs K_internal per category with error
   bars, the hidden-knowledge gap, layer-wise curve, and the control/null checks shown up front.
