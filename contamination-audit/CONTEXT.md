# Ranke time-locked LLM — contamination audit

**Last updated:** 2026-07-24
**Status:** Active — probing side settled (models clean); decontamination side COMPLETE and validated OOS; unified repo restructured (external review grade B); next milestone = `production/` package for CSCS.

## Current understanding
The Ranke-4B models (Qwen3-4B architecture, ~160B tokens, cutoffs 1900/1913/1929) are trained only on
pre-cutoff text, for use as "synthetic respondents." Two workstreams:

**Probing (definitive 2026-06-25):** the models are essentially CLEAN of coherent post-cutoff
knowledge; residual = shallow name/term exposure (cued completion). Earlier "US politics RED" was a
metric/cue artifact — superseded. Metric = length-matched twins, no cues; R/A/G rule.

**Decontamination (complete 2026-07-22/24):** a validated multi-stage pipeline for removing
post-cutoff books from the ~2M-book corpus, all code unified in
`github.com/huggingbeard/ranke-edition-dating` (clone `C:\Users\hvoth\Documents\GitHub\ranke-edition-dating`),
restructured thematically (`pipeline/` frozen definitions · `validation/` evidence base · `reports/`
all documents · `docs/`), with a consistency gate binding harnesses to canon and a hostile
deepseek-v4-pro review (D → B after fixes; residual = the declared production/ gap).
Pipeline: **Stage 1** paperwork rule (catalogue year + Guohui's Qwen3-14B docyear + imprint regex) —
~100% of honest-metadata contamination; flagged reprints get the gpt-oss editorial pass (modern
apparatus out, 27/29 clean re-sets rescued). **Stage 2** reader team (Qwen3.6-35B ≥2/8 OR
gpt-oss-20b ≥3/8 on 8×2000-char random passages) — rescues 0.875 [0.805,0.927] of wrong-metadata
contamination OOS at 13.2% [9.9,17.1] clean loss. **Stage 3** exhaustive grep (markers_1945 +
markers_1929: 447 null-corpus-validated terms incl. mined stylistic fingerprints "basically/
contacted/phoned" + strict year-in-context sieve) with quote-the-proving-phrase adjudication —
+30–40% of reader leftovers at ~0.5% loss; ~0% on cut-front disguises. **Referee tier (designed,
unbuilt):** Guohui's agentic dater (92/94% blind-validated; code preserved in
`pipeline/referee-agentic-dater/`) on the ~5–10% suspicious band with FLIPPED burden of proof
(no positive pre-1929 evidence → remove) — the only mechanism touching the silent-book residual.
All headline numbers are OOS from a PRE-REGISTERED test (commit 5176fd8; 743 fresh books; all five
bands passed; the two shrunk numbers are the quoted ones). Production cost 2M books: ~$150–250
rented A100, 1–2 days on 4–8 pods. Validation campaign: ~$17 total.

Key negative results (do not revisit): reader scaling exhausted (gpt-oss-120b ties 35B on C, 5pp
WORSE on the 1930s stratum, 4× slower); grep-directed passage sampling ≤ random; keyword lists as
daters unusable (93% FP) though excellent as adjudicated screens; recall-0.95-at-FP≤10% on the
hardest stratum is unreachable by evidence-hunting (archaic-genre books carry no date in body text —
their dates live in front matter; the stages are complements).

## Active questions
1. Referee-tier adoption: measure the agentic dater's abstention rate on genuinely-clean band books
   (= the flipped burden's corpus cost) from its 2,000 released records × our certified negatives.
2. Operating dials (PI): reader strictness OR-2/3 vs strict; stage-3 vocabulary-vote rule (≥3 mined
   terms → remove; needs FP measurement first).
3. `production/` package for CSCS: externalized config, Slurm orchestration, environment spec;
   needs full-corpus path/format + partition/account + HF access mode from CSCS/Guohui.
4. Guohui (on vacation): confirm full-corpus records carry stage2_docyear (near-certain; sample was
   carved with fields intact; corpus NOT on HF — org history-llms hosts only 2 DPO sets); archive
   the 14B script for provenance.
5. Review queue: 81 certifier-flagged old books (Llama-70B-only flags; spot-checks = over-caution).
6. Probing side (older): US-politics power, length-matched batteries per category (see 2026-06-25 state).

## Decisions made
1. Recall headline on HARD strata only (gold contaminants + 1930-46 ordinary books); post-1947
   gov-doc stratum is diagnostic ("a gov-doc classifier by another name" — PI).
2. Stage-2 config frozen: 35B(2/8) OR 20B(3/8), 8×2000 random; certifiers deepseek-v4-flash +
   Llama-3.3-70B (gemini-flash-lite banned — over-flags old content).
3. Reprint policy: remove only with post-1929 editorial matter (gpt-oss pass); clean re-sets kept.
4. Every recall claim carries an FP bound (Llama-1B answers AFTER to everything — 99.9% FP).
5. Quote OOS-shrunk numbers everywhere (rescue 0.875 not 0.93; loss 13.2% not 10%).
6. stage2_docyear: no substitute generator needed for the existing union (per-record stored data at
   CSCS); rebuild only for genuinely new sources.
7. Repos unified: anachronism-decontaminator merged (full history) + archived read-only; Guohui's
   dater preserved from temp-guohui; one repo, thematic layout, consistency gate.
8. Pod discipline: every cloud session terminate → verify zero pods → delayed re-verify (practiced 5×).

## Approaches tried and abandoned
1. gpt-oss-120b as reader — negative result (see above), $6.69.
2. Qwen3.6-35B on the 24GB 4090 under vLLM — kernel hang / CPU-bound eager / incompatible quant;
   3-bit GGUF on llama.cpp works but 25× slow (within 1-3 pts of BF16 accuracy — quantization is not
   the limiter, the runtime is).
3. Grep-directed passage sampling as recall booster — no gain over random.
4. Stage-3 v1 — substring marker matches + adjudicator trusting injected [marker:X] tags: 60% FP on
   cleans; fixed with word boundaries + raw untagged excerpts + quote-the-proving-phrase.
5. Loose year-contexts in the sieve ("of 1930") — fire on OCR noise; strict publication-context only.
6. Hand-curated markers without null-corpus validation — "new deal"/"nazi"/"stalin" are pre-1929
   English; in-corpus zero-hit validation is mandatory.
7. README-only repo cleanup — rejected by PI; full git-mv restructure with re-run verification done.

## Next steps
1. Build `production/` package (needs CSCS specifics — see Active questions 3).
2. Referee-tier abstention measurement → adoption decision → implement local-Map variant (~$150/2M).
3. PI dial decisions; adjudicate the 81-book review queue (optional).
4. On Guohui's return: docyear confirmation + script archive.

## Key files
- Repo (all code/docs/reports): `C:\Users\hvoth\Documents\GitHub\ranke-edition-dating` — entry
  points `README.md` → `PIPELINE.md`; reports in `reports/`; evidence in `validation/`;
  LOG snapshot in `docs/`. Latest commit 39cdd92.
- Data (Dropbox): `experiments\content-dating\` (passages/, outputs/ incl. all judgment parquets,
  intermediates/, positives_text/, plants_text/); corpus sample:
  `temp-guohui\decontamination\datasets\history-llms\sample-10gb\`.
- Live trail: this folder's `LOG.md` (aggressively maintained, 22–24 July fully logged).

## Sub-studies
- **`12_foresight/`** — Paper 1 exists (`12_foresight/paper/foresight_v5.pdf`, machine-generated
  numbers, 89 verify checks); WWI trio retracted as unmeasurable; Ebert/Rentenmark stand. See
  `12_foresight/CONTEXT.md` (v9.0) before resuming.
