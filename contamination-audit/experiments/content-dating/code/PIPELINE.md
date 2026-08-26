# Production decontamination pipeline — authoritative spec

Version 1.0 — 2026-07-24 — office box. Validated 2026-07-22/24 (see `LOG.md` in the project root,
commits up to `59af3fe`). All parameters PROVISIONAL pending PI sign-off of the operating dials (§6).
Data root: `D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating\`.

## 0. Purpose and principle

Remove every book whose text was **written** after the cutoff (validated at 1929) from a ~2M-book
training corpus. Governing rule: **over-removing clean books is cheap; keeping contamination is not.**
Estimand per book = the date of the words, not the paperwork; reprints are split by content
(old-body re-set = keep; modern editorial matter = remove).

## 1. Stage 1 — deterministic paperwork rule (CPU, ~free)

`freeze_pipeline.py` v2: `remove ⇔ max(catalogue_year, stage2_docyear, imprint_year(text)) > cutoff`.

- `imprint_year`: newest year within 60 chars of a publication keyword in front (6,000 chars) +
  back (3,000 chars) matter, with the v2 library-stamp guard.
- `stage2_docyear`: **Guohui's corpus-build pass** — Qwen3-14B-FP8 reads the document and estimates
  a publication year. Measured behavior (2026-07-24, gold-audited books): content-derived, 88%
  within ±5y of truth on wrong-metadata books, fooled 6.9% [CI 2–12%] by archaic-style moderns,
  does NOT anchor on catalogue metadata. Exact script/prompt: with Guohui (archive request open).
  For fresh corpora this pass must be recomputed (~50–90 A100-h ≈ $80–130 at 2M books).

Validated: catches essentially **100% of honest-paperwork contamination** (in-sample 0.996–1.000;
OOS 250/250). Flags ~7% of the corpus, ~2/3 of them clean books recoverable by the existing
paratext pass ($75–110 at 2M per the repo README).

## 2. Stage 2b — reprint editorial pass (PI policy, 2026-07-23)

Books flagged as post-cutoff **editions of pre-cutoff works** get one gpt-oss-20b read of front
(12,000 chars) + back (3,000) matter: *"any later-date introduction/editor's/translator's note,
afterword, or commentary? → REMOVE; bare re-set of the old text → KEEP."*
Validated on 80 gold reprints: 48/50 with modern apparatus removed; **27/29 clean re-sets rescued**
(premium period text). Script: `reprint_editorial_pass.py`; verdicts: `reprint_verdicts.csv`.

## 3. Stage 2 — reader team on stage-1 survivors (GPU)

Per surviving book: **8 random 2,000-character body passages** (extractor v2, seed 1929, 5–95%
window), each judged BEFORE/AFTER-1929 by two models with the frozen prompt (verbatim in
`run_flotilla_v2.py`), temperature 0:

- **Qwen3.6-35B-A3B** (BF16 on A100; the accuracy champion) — flag at **≥2 of 8** AFTER votes;
- **gpt-oss-20b** — flag at **≥3 of 8**;
- book removed if **either** flags (OR rule; the split-half-validated operating point).

Validated (OOS, pre-registered): rescues **0.875 [0.805, 0.927]** of wrong-metadata contamination
that fully defeats stage 1; costs **13.2% [9.9, 17.1]** of certified-clean books. Strict variant
(35B k=4 / AND rules) halves the loss at ~8–10 points of rescue.
**Scaling is exhausted**: gpt-oss-120b ties the 35B on contaminants, runs 5pp WORSE on the binding
1930s stratum at matched FP, 4× slower (2026-07-24, $6.69 eval). Do not chase larger readers.

## 4. Stage 3 — exhaustive grep + targeted adjudication (kept pile)

- **Sieve** (CPU): word-boundary Aho-Corasick over the FULL text with `markers_1945.json` +
  `markers_1929.json` (447 terms, each validated zero-hit across 1,361 certified pre-1929 books;
  391 mined — top discriminators are post-1929 stylistic vocabulary: "basically", "contacted",
  "phoned", "sponsored") + strict year-in-dating-context regex. Hit rates: certified cleans ~1%,
  kept-pile expectation 3–8%.
- **Adjudication** (GPU): gpt-oss-20b reads ±800 chars around ≤6 hits, must QUOTE the proving
  phrase (raw untagged excerpts — v1's injected tags induced label-trust hallucinations; fixed).
- Validated: +30–40% of the readers' leftovers on honest-body books (B misses 39%, C 30%, OOS
  moderns 50%); ~0% on cut-front disguises (nothing for the year-sieve to read); ~0.5% extra clean
  loss; occasionally discovers real contamination in "certified clean" pools.
- Open dial (unvalidated): vocabulary-vote rule (≥3 distinct mined terms → remove without semantic
  proof) — would attack the disguised class; FP cost measurable on certified negatives first.

## 5. Validated end-to-end numbers (quote these)

| Stratum | Composite removal | Evidence |
|---|---|---|
| Honest-paperwork contamination | ~1.00 | OOS 250/250 [CI ≥0.985] |
| Wrong-metadata contamination | ~0.92–0.95 | doc-dater 0.93 + readers 0.88 of residual + grep top-up; OOS plants composite 0.991 under the measured fooled-rate |
| Reprints w/ modern apparatus | 0.96 | 48/50, editorial pass |
| Clean-book cost | ~13.5–14% | 13.2% readers + ~0.5% grep (+stage-1 FPs recoverable) |
| Residual (characterized) | archaic style + no imprint + wrong metadata + no modern vocabulary, all at once | the 2-of-8-votes near-misses; got-throughs 1/3 caught |

## 6. Operating dials (PI decisions)

1. Reader strictness: OR-2/3 (validated headline) vs strict (≈half the corpus loss, −8–10 rescue).
2. Stage-3 vocabulary-vote rule: off (validated) vs on (needs FP measurement).
3. Escalation tier (designed, unbuilt): uncertain band (1–2 votes) → whole-book read, ~$150–300/2M.

## 7. Cost at 2M books (A100-SXM @ $1.49/h, measured throughputs)

| Item | A100-h | $ |
|---|---|---|
| Stage 1 + grep scans (CPU) | — | ~5 |
| 35B reader (14.9M passages @37/s) | ~112 | ~170 |
| 20B reader (@~100/s) | ~41 | ~60 |
| Reprint pass + stage-3 adjudication | ~8 | ~15 |
| **Total (HTTPS-client throughput, conservative)** | **~160** | **~250** |
| Realistic with offline batched engine | ~90–110 | **~130–180** |
| (+ fresh-corpus 14B doc-dater, if needed) | +50–90 | +80–130 |

Wall-clock: shards by book — 1 pod ≈ a week, 4 pods ≈ 2 days, 8 pods ≈ 1 day. Teardown per pod:
terminate → verify zero pods → delayed re-verify (`runpod_a100_verify.py`).

## 8. Document map

`content_dating_report.html` (v2.2, technical) · `summary_nontechnical.pdf` (v3.0, plain language) ·
`PREREGISTRATION_OOS.md` + `score_oos.py` (the honesty test) · `results.csv` / `ensemble_results.csv` /
`combo35_results.csv` (all configurations) · `composite_results.csv` · `reprint_verdicts.csv` ·
`markers_1929.json` + `expand_markers_1929.py` · `stage3_grep.py` / `stage3_adjudicate.py` ·
`build_plants.py` + `test_plant_realism.py` (the disguise recipe + realism gate) ·
`review_queue_negatives.csv` (81 books awaiting adjudication) · project `LOG.md` (full trail).
