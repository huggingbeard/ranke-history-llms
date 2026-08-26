# Pre-registration — Foresight Panel v2 (eight events, three languages)

**Files:** `src/panel_events.py`, `src/panel_checks.py` (bank);
`src/panel_bank.py` (screen + freeze); this document.
**Date:** 2026-07-04 (written BEFORE any panel scoring).
**Version:** 2.0 (supersedes `PREREGISTRATION_FORESIGHT.md` for the panel analyses; the v1 English
batteries and their frozen bank remain as registered).

## Question

How foreseeable, ex ante, was each major catastrophe of 1907–1933 — measured directly from frozen
pre-event information sets (time-locked LMs at 1900/1913/1929), in English, German and French?

## Estimators (locked)

1. **In-family conceptual surprise** (events with a Ranke hindsight anchor):
   `Δconc(item) = strain_pre(item) − strain_anchor(item)`, with
   `strain = −slp(target|subject) + slp(target|neutral)` under one model. Same tokenizer ⇒ the
   anchor's surprise is the exact lexical floor. Event-level = mean over the event's 12 items.
2. **Bridge-corrected excess strain** (Depression, Nazi seizure — no post-event Ranke):
   `XS(item) = strain_pre(item) − strain_modern(item)`; corrected by the **bridge offset**
   `offset_lang = mean over anchor-1929 event items of [strain_1929 − strain_modern]`
   (the 1929↔modern scale gap measured on outcomes the 1929 model has read).
   `Δconc^bridge = XS − offset_lang`. Assumption (stated in the paper): the scale gap on
   known-outcome items transfers to unknown-outcome items.
3. **True-vs-false discrimination**: calibrated preference `c(true) − c(false)` under the PRE model,
   `c(a) = slp(a|stem) − slp(a|neutral)`. Reported as % items with preference > 0 and mean margin;
   the anchor/modern models give the hindsight ceiling.

## Hypotheses (locked before scoring)

- **H1 (ranking):** Δconc(Depression, bridge) > Δconc(each in-family event), in each language.
- **H2 (validation):** Panic of 1907 ≈ Tier-B floor (Δconc ≈ 0). If violated, reported as a finding
  about the measure, not hidden.
- **H3 (cross-lingual stability):** the in-family events (war, empires, revolution) read ≈ 0 in all
  three languages.
- **H4 (discrimination):** pre models prefer the true catastrophe to the matched false twin at a
  rate above 50% for events read as foreseeable (Δconc ≈ 0); hindsight models near 100%.
- **Open (no directional prediction):** Spanish flu and hyperinflation — genuinely uncertain ex
  ante whether the pre-1914 record contains them; whatever they read is the finding. Nazi seizure
  after bridging: exploratory.

## Controls & gate (locked)

- Tier-B timeless items = the foreseeable floor; permutation baseline per language.
- Tier-C future-to-all items = unforeseeable ceiling AND leak tripwire: no pre-1930 model may show
  low XS on any Tier-C item in any language; violation triggers a leak investigation before any
  panel result is trusted.
- **Gate:** 6 post-event fact probes per event per language, 4-way calibrated forced choice, under
  the event's designated PRE model. GREEN = accuracy consistent with chance (25%), binomial test.
  RED items flagged in the bank with known period-prior caveats (Morgan, Versailles, Franz
  Ferdinand, Roosevelt) are excluded from the R/A/G verdict and reported separately.
- The main foresight read of an event×language is trusted only if its gate is GREEN/AMBER-explained.

## Exclusion rules (locked)

- Items failing the automated screen (target text leaked into stem; extreme cross-language length
  disparity > 2× in characters) are dropped before freezing and listed in the screen report.
- After scoring: no item-level exclusions except gate-triggered ones; leave-one-out influence is
  reported for every event-level mean.

## Statistics (locked)

Item bootstrap (B=10,000, seeded) per event×language; pooled across languages with cluster
bootstrap (cluster = translated item triplet); permutation p vs the Tier-B floor (same language);
Benjamini–Hochberg across the 8 events within each language family of tests; effect sizes in
Tier-B SD units. All numbers emitted to `numbers.tex` by `make_numbers.py`; nothing hand-typed.

## Bank freeze

`panel_bank.py` writes `results/question_bank_panel.json` with a SHA-256 hash; the scorers refuse
to run if the live bank hash differs from the frozen hash. Any post-freeze edit requires a new
freeze and is logged in LOG.md.
