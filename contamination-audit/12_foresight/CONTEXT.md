# Ranke-Foresight + Contamination Audit — compiled truth

**Last updated:** 2026-07-12 (NEW THREAD C: the association probe — piloted, certified, grid approved)
**Status:** PAPER 1 EXISTS: `paper/foresight_v5.pdf` (26pp, twist-framed, every number machine-generated
and independently verified 101/101). Yardstick FINAL (v3). Ebert caveat RESOLVED. DE/FR CLOSED: French
repaired-run certified, German fails honestly on coverage (`certify_defr_v2.json`; no second repair
pass by protocol). **THREAD C: the CONTAMINATION PIPELINE v1 — DELIVERED AND VALIDATED (2026-07-12 night).** Battery
(`src/battery_assoc_v1.json` v1.1) + runner (`contam_pipeline.py`: lift-vs-controls on certified slp,
fiction floor from the 5 invented persons, cluster-count verdicts) + INSERTED-MEMORY VALIDATION
(`pipeline_validate.py`: 3-sentence ROME plants; sensitivity PASS x2 — Ebert & Truman DARK→WHISPER
with planted cluster; specificity PASS x2, zero collateral) + TALKIE-1930 DARK CONTROL on RunPod
(triple-lock, pod verified terminated, ~$0.6): darker on Eisenhower/Obama/Franco, equal Hitler–Stalin
whisper in both (period co-occurrence check decides), brighter Kennedy→Dallas/Mao→China (v2 caveat:
add real-name floor population — invented-name floors are lenient for place-tells), radar FLAGGED in
both (shared-leak candidate, fits Talkie's known 1940 leak). SCOPE: person-targets certified;
term-targets provisional (compound-subword channel; smartphone anchor). Natural findings: Trotsky-1913
FLAGGED (Stalin/Red Army/exile + innocents = full signature), radar-1929 FLAGGED. Report:
`reports/contam_pipeline_v1.html`. v2 queue: real-name floors, matched invented-compound controls,
period co-occurrence checks, Google-Books tell hardening. Corpus-grep flags: robot, radar,
1913-Eisenhower, Trotsky-1913 cluster. Open tracks: corpus check + retrain (Alps), fake-name
presidents, venue formatting. **Version:** v9.4

---

## PAPER 1 — foresight_v5.tex (the twist-framed general-science paper)

**Decision locked (2026-07-10, JV):** write Paper 1 NOW on the dirty models with the screen folded in — do
NOT wait for the clean retrain + 1938. Reasons: the twist only works on dirty models; the screen already
protects the claims (leakage can fake foresight, not blindness); retrain = hostage risk; priority vs the
contamination-detection literature; 1938 feeds Paper 2 (QJE, disasters), not Paper 1. Clean-confirmation
coda slot reserved — when clean models land, re-run batteries + `python src/build_v5.py` and everything
regenerates.

**The arc (supersedes v4, whose "models are clean" premise was tested 7/8 and found FALSE):**
instrument → exam (certified EN 16/20) → THE TWIST (Sec 5: date-spike vs fiction floor; 1913 base leaks
WWI 13x / Bolsheviks 17x / CRA-1964 4.1x vs ceiling 1.4x; 9/22 and 5/13 post-cutoff events breach; leak is
shallow/selective/asymmetric) → THE SCREEN (Sec 6: date-spike primary; memory meter = bluff-catcher, two
killer exhibits: Treaty of Bordeaux 80%-confident-never-existed w/ probe 0.92 vs output 0.76, and ROME
causal flip Björkö 0.00→1.00; cutoff-shift 0.32→0.85; triage rule = date-leak × outcome-lean) → SCREENED
RESULTS (Sec 7 funnel: 26 questions → 10 scored → 7 witness-gated → 4 clean + 3 suspect) → memoir
(date-clean, certified wholesale) → implications (support: unrefuted-not-vindicated; weighting: HARDENED)
→ retrain plan w/ date-spike acceptance gate.

**Explicit retraction in-text:** war-breadth, US-entry, rev-debts (v4's headline "foreseen" trio)
reclassified as UNMEASURABLE — the 1913 guesser leaks the war (44% vs 8.1% floor) / revolution (51%) AND
leans to truth. Tsar moved to witness-gate drops (two-truths item; witness itself bets "driven from his
throne", 2%). Ebert's case STRENGTHENED by contamination: guesser leaked the war's date and still bet
everything on the Kaiser — whatever leaked did not contain the future's content.

**Pipeline (one command: `python src/build_v5.py`):** make_numbers_v5.py (201 macros, all from result
files, 0 hand-typed) → make_figures_v5.py (fig_gaps_v5 verdict-coloured; fig_leak_v5; fig_bluff_v5;
fig_rome_v5 with corrected real/invented legend) → verify_v5.py (89 independent re-derivations) → 2×
pdflatex (latexmk dropped: MiKTeX lacks perl). New refs: Shi, Cheng/Marone, Golchin, Kadavath, Meng,
Alain-Bengio, Ravfogel. Appendix B = full audit tables (spike, 25-item triage, gate).

## THE MUNDANE YARDSTICK — control battery v3 (PI-vetted)

The scale for "how big is a gap": ordinary between-cutoff changes scored by the same certified arm.
`src/control_battery.py` v3 → `results/control_gaps_v3.json` (v1/v2 results preserved alongside).
- **v1→v2 (JV vetting):** sterling reworded (dollar-parity option was de facto the truth, $4.86); ROYAL
  NUMBERING RULE — officeholder questions must never offer options the witness-period model knows
  impossible (no Queen Victoria crowned 1902, no dead popes reigning 1925), and invented regnal names must
  respect numbering (think Charles III; 1925 papal options = Benedict XVI / Leo XIV / Gregory XVII);
  german_wilhelm's 'Otto' distractor was BROKEN (Otto-of-Bavaria interference) — all 8 royal/papal items
  rebuilt and now GATE; battery extended 40→69 aiming at ~30 gated; locarno + einstein confirmed fine.
- **v2→v3 (JV):** NAME-LOTTERY EXCLUSION RULE — truths that are private individuals' names with no
  pre-event public existence (channel_swim_1926/Ederle, +12.1 = question-shape ceiling artifact) are
  excluded BY RULE; regnal/office names stay (constrained public choices). Documented in battery header,
  paper Sec 7, and the review sheet — a referee diffing versions finds a principle, not a cherry-pick.
- **v3 numbers:** 68 items, 29 gated; median 0.67, 90th pct 3.50, max +6.32 (norway_king_1905).
  **Ebert +9.84 and Rentenmark +6.77 = 100.0th percentile.** The Norway comparator carries the prose: the
  1900 guesser names Oscar II for the new Norway exactly as the 1913 guesser names the Kaiser — the
  republic sits 3.5 nats above even the naming of a new country's new king.
- **PI review artifact:** `reports/control_facts_review.html` (v3; per-item stems, options, results, gate
  status, flags). The ~28 new v2 facts still await JV's final item-by-item sign-off.

## THREAD C — The association probe (new instrument, 07-09..07-12)

Born from JV's lens screenshot (top-10 tokens at "Hitler" in Ranke-1929: Stalin/Nazis/Eisenhower but
also Barack/"triggered"). Disciplined into: **free-association prompts ("The name on everyone's lips
was X") + name-swapped controls + invented-name floor, scored with the certified batched slp machinery
at the OUTPUT layer** — JV's "isn't this just log-probing?" was right; questions hide shallow leaks
(must out-compete period priors), free association exposes them (Stalin rank 1,658/150k behind Hitler
at the 1929 mouth vs ~13k behind controls). Layer profile = secondary diagnostic only.
- **Validated**: cross-model contrast (1913 cold on Hitler→Stalin, prediction registered first) +
  SENSITIVITY CERTIFICATE (ROME inject Hitler→Stalin into 1913: held-out carriers 11,357→106,
  key-specific, collateral-free; `assoc_inject_1913.json`). Blank-card artifact identified (untrained
  tokens pollute lens top-k; Barack unmasked — ranks better behind Müller).
- **Tier-0 census** (`vocab_census_{1913,1929}.json`): never-seen-centroid geometry works at extremes;
  cross-cutoff deltas self-validate (Bolshevik/Soviet/fascist/jazz/insulin more trained in 1929);
  TRAP caught: "email" = German/French enamel → trilingual sense rule; SUSPECT: "robot" trained-ish in
  both models incl. 1913 (word coined 1920) → corpus-grep flag; middle zone needs freq calibration.
- **Binding rules**: doubly-anachronistic tells (JV's genocide correction); EN+DE+FR sense guard;
  cluster-counting not token-counting (inject showed neighbour-warming); summed-slp multi-token rule;
  ≥5 phrasings; fiction-floor multiplicity control. Hitler = burned dev case.
- **Grid v2 JV-APPROVED in full** (`notes/2026-07-12-tier1-target-grid-v2.md`; Lenin+U-boat out,
  Trotsky+poison-gas in — gas ties to the circulation arm's own "absent from print" finding).
- **Narrative doc** for humans: `reports/assoc_narrative/assoc_probe_narrative.pdf` (v1.1, 5pp).
- **Flagged, unclaimed**: Eisenhower whisper in the 1913 model (23,951 vs 39k+; coheres with the
  date-spike's faint WW2 traces); 1929 Hitler–Stalin still owes the period co-occurrence check.
- **NEXT (the approved block)**: author full per-target tell/innocent lists → mechanical Google Books
  EN/DE/FR pre-cutoff pass (reuse ngram_exposure.py fetch pattern) + tokenizer forms → JV final vet →
  hash → one run incl. Talkie-1930.

## THREAD B — Contamination audit / memory meter (stable since morning checkpoint)

Meter = linear probe on Ranke-4B-1929 BASE residual stream, retrieval-vs-confabulation. AUC 0.99
known-vs-invented; generalises leaked 0.70 / unknown 0.25; causally validated (ROME injection — reads
natural storage, not forced answers); two-model cutoff-shift proof. THE FLAW: realness leak — largely
fixed by per-category orthogonalisation (INLP; exposure corr +0.33→−0.13, memory AUC 0.86→0.90; 278-item
5-category battery: memory AUC 0.95, leak small + category-specific, per-category null closes it; cliff
past ~18 nulled dirs). Technote v1.1 (`reports/technote/memory_meter_technote.pdf`) = replication-grade;
claims verified real. Convergent headline: post-cutoff contamination is SHALLOW and SPARSE, rooted in
PRETRAINING (base leaks; self-tuned instruct amplifies). Operational rules: year-spike (best-worded, MAX
over ≥5 phrasings) = clean PRIMARY; meter = narrow specialist (confident-bluff catcher, 0.92 vs 0.76);
combined OR rule recall 0.91; corpus check = gold standard. Talkie-1930-13B head-to-head: Talkie cleaner
AND more complete — Ranke-1929 both leaks 1939-45 and under-covers its own late 1920s.

## Active questions
1. **Corpus check + retrain (the gold standard, NOT YET RUN)** — Alps/cloud greps
   (`6_corpus_contamination_finder/`), source-level purge, fix the date filter (root cause), retrain
   1913/1929/1938; date-spike = acceptance gate (leaks must fall to the fiction floor). 1938 unlocks
   Paper 2 (Depression + Hitler as measured gaps).
2. Fake-name "presidents" to close the meter's last realness residue (small GPU extraction; technote
   polish, not paper-critical).
3. DE/FR answer wording (neither language passes the exam; supporting evidence only).
4. Venue formatting (Science/NHB style) once content settles; optional arXiv method preprint (priority).

RESOLVED 2026-07-10 late: yardstick sign-off (v3 FINAL) and the Ebert name-free rebuild
(`src/ebert_namefree.py` → `results/ebert_namefree_v2.json`): witness breaks on office-phrases in BOTH
1919 and 1920 settings (1929 reader puts ~90% on the workers' councils; meter declines) → no certified
name-free GAP; but the guesser side is unambiguous and setting-independent — 90% Kaiser / 9--10% elected
president. Direction of the flagship NOT a name artifact (office = 9:1 underdog); magnitude partly name
(office-miss ~2.4 nats vs name-miss ~10.1). Written into paper Sec 7 (the "double statement") + Sec 9
(reported as guesser-side robustness check, not a certified gap).

## Decisions made (do not relitigate)
1. ONE general-science paper folding the detector in as trust mechanism; ML footprint = optional arXiv
   preprint only. Paper 2 (QJE) waits for the 1938 retrain.
2. Write Paper 1 NOW on dirty models + screen; retrain is a parallel track with a coda slot.
3. Suspect trio reported openly as unmeasurable (not silently dropped); Tsar to appendix as a broken item.
4. Year-spike primary / meter specialist / corpus gold. Realness correction per-entity-type.
5. Royal numbering rule + name-lottery exclusion rule for the yardstick (JV, 2026-07-10).
6. No Jacobian lens; no Min-K% (frequency-confounded); ROME-to-confirm-contamination is NULL.
7. Results never overwritten (control_gaps v1/v2/v3 all kept); log aggressively.

## Approaches tried and ABANDONED (unchanged from v8 — see LOG for detail)
Jacobian/tuned lens; ROME-editing-to-confirm; subject-token/multi-layer/trajectory probes (leak at every
depth — orthogonalisation is the fix); global realness null (too blunt); >18 nulled dirs (cliff); Min-K%
and thin-context forced choice (frequency-dominated); meter-alone as detector (over-flags); foresight
graveyard (sentence-surprise, forward betting slips, calibrated-only/raw-only, Δconc panel). NEW: v1
control battery's dead-option distractors (Queen Victoria, dead popes, 'Otto') — broken, fixed by the
numbering rule; uk_pm_1926 as a prose comparator (never passed the witness gate — my error, removed);
office-phrase questions as certified gaps (witness collapses to the councils two-truths trap in 1919 AND
1920 settings — office abstractions lose their witness in this design; guesser-side probe only).

## Next steps
1. Corpus check + decontamination/retrain on Alps (separate workflow; the proof spine).
2. JV sign-off on new yardstick facts; Ebert name-free rebuild; fake-name presidents.
3. Optional polish: DE/FR wording; arXiv method preprint; venue formatting pass on v5.

## Key files
- **Paper 1:** `paper/foresight_v5.{tex,pdf}` + numbers_v5.tex + *_v5.tex tables; build via
  `src/build_v5.py` (make_numbers_v5 → make_figures_v5 → verify_v5 → pdflatex ×2). v4 untouched.
- **Yardstick:** `src/control_battery.py` (v3, 68 items) → `results/control_gaps_v3.json`;
  PI sheet `reports/control_facts_review.html` (regen: `src/control_facts_review.py`).
- **Audit/meter (B):** `reports/technote/memory_meter_technote.pdf` (v1.1);
  `reports/remembering_meter.html`; results/superlatives/{contamination_triage, date_spike_base,
  probe_mri_summary, real_entity_null_summary, minimal_pairs_summary, rome_inject, cutoff_shift_summary,
  memory_detector_v3_summary, bignull_v3_summary}.json.
- **Foresight (A):** START `reports/method_pivot.html`; `reports/validity_pass.html`;
  results/{v41_battery, certify_chronicle_heldout, gate_report}.json.
- **Strategy:** `notes/2026-07-10-paper-and-retrain-strategy.md`.
- **Runtime:** venv `C:\Users\hvoth\timelock-contamination\.venv`, PYTHONUTF8=1; GPU = RTX 4090 24GB
  (since ~2026-07-10; old 12GB-VRAM infeasibility notes are obsolete). **SCORING STANDARD since
  2026-07-11: fp32 + batched** (`common.load` defaults to float32, `common.slp_batch/slp_pairs`) —
  bf16 proved batch-shape-noisy (~0.04 nats); fp32-batched is exact (3e-5), replicable, and x4.3
  faster than the old bf16-sequential. Regression gate: `src/test_slp_batch.py` (rerun after torch/GPU
  changes); diagnosis archived in `test_slp_batch2.py`. Pre-2026-07-11 result files are
  bf16-sequential (kernel-noise-sized rerun differences, no verdict flips measured). common.py's VRAM
  guard waits politely for other GPU jobs (e.g. the WSL vLLM server). **Everything logged in LOG.md.**
