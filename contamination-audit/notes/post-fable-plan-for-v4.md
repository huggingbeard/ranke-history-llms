# Post-Fable plan for the foresight paper (v4 → v5 + the split)

**Date:** 2026-07-07
**Version:** v1
**Author:** Opus (planning session with JV)

**Source documents read for this plan (full paths):**
- `D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\notes\fable-second-round` — Fable's line-by-line critique of `foresight_v4`.
- `D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\notes\foresight_split_and_todos.md` — Fable's two-paper split + the `/goal` work list.
- `D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\12_foresight\CONTEXT.md`, `README.md`, `docs\PREREGISTRATION_FORESIGHT_v2.md`.
- Code: `12_foresight\src\v41_battery.py` (scoring engine; gap at line 360), `src\common.py` (`slp`, `cal_score`, `load`), `src\production.py` (free-generation arm, unused), `src\foresight_bank.py` (`RANKE`, `MODERN` model ids).
- Data: `12_foresight\results\v41_battery.json` (English certified battery, 26 questions, 10 answered).

**Environment:** venv `C:\Users\hvoth\timelock-contamination\.venv`; `PYTHONUTF8=1`; RTX 3060; models
`uzh-echist-org/Ranke-4B-{1900,1913,1929}` + `Qwen/Qwen2.5-3B` (modern reader); HF token at
`...\claudecode\api\HF.txt`. Full battery scores in minutes. **Rule: never overwrite `results/`; version
new files; append to `LOG.md`.**

---

## 1. The problem, in one sentence

Every finding in `foresight_v4` rests on one question we have not yet closed: **is the model's
preference for the true outcome real foresight, or is it surface — fame / familiarity / how often the
corpus printed the word?** The gap metric (`gap = -ln p_guesser(truth) + ln p_witness(truth)`) cannot,
on its own, tell those two apart.

## 2. Why Fable's reframe is superior to the v4 draft

Four holes in v4, each a version of that confound, each with a concrete fix Fable supplies:

1. **The witness can't read some of the questions we score.** The Tsar "finding" (+2.71 nats) is
   computed from a witness — the 1929 reader, for whom the execution was decade-old news — that puts
   **2%** on the truth. A gap built from two tiny unstable numbers is not a measurement. Fix: the
   **witness-gate** — a question enters the paper only if the witness scores ≥ 50% on the truth.
2. **The availability heuristic and a token counter are observationally identical.** v4's central
   behavioural claim (the pre-event mind over-weights the most recent precedent) looks exactly like "the
   model parrots whatever the corpus printed most." Fix: **frequency-control** — measure the surface
   driver, net it out, keep only what survives. (This is JV's fame-matching idea, made quantitative.)
3. **"Foreseen" has no yardstick.** A +0.2 gap is small compared to *nothing*. Fix: a
   **control-distribution** — the same instrument run on mundane, on-schedule outcomes, so "foreseen"
   means "gap indistinguishable from the gap on a peace-conference location."
4. **One instrument is doing two jobs.** Forced-choice betting cannot measure whether an outcome was
   *thinkable*, because the researcher writes the option into the model's head. Thinkability needs
   **free generation** (did the reader ever *volunteer* Ebert?), which already exists unused in
   `production.py`. Fix: two instruments — generation for thinkability, betting for weighting.

Net: v4 measures a real signal but cannot yet rule out that the signal is fame, and cannot say how big
"small" is. Fable closes both holes and cleaves the work into a Science methods paper and a QJE
economics paper.

## 3. The two-paper split (Fable, Part A)

- **Paper 1 — the instrument.** Target Science / Nature Human Behaviour. Claim = the method: you can
  read vanished states of knowledge out of frozen readers *if you ask the right way*. Ships as soon as
  Phase 1 + parts of Phase 2 are in; does **not** wait for the 1939 model. Carries the taxonomy of
  failed designs, the certification exam, the two-instrument principle, and showcase findings (Ebert
  +9.8 in three languages, the Rentenmark, the French–German Depression split).
- **Paper 2 — the disasters.** Target QJE. Claim = empirical: rare-disaster *support* mostly holds,
  *weighting* fails in a patterned way, and the pattern is the availability heuristic at the level of a
  society's written record — **or**, if frequency-control says so, it is print frequency, and the paper
  says that instead. **Waits for the 1939 witness** (the Depression and Hitler are the two disasters the
  calibration literature rests on; measured beats narrated). Cites Paper 1 for the instrument; no salami.

## 4. The plan

### Phase 1 — runs now, on existing data + models. The confound/validity core. (Fable's first tier.)

**1a. witness-gate** — `src/witness_gate.py`. Filter `v41_battery.json` to rows with `p_truth_post ≥ 0.50`;
set memoir-only rows (no witness) aside separately. Regenerate the findings table; write
`results/gate_report.json` naming every dropped question with both readers' full betting slips and the
reason. Then the **Tsar autopsy**: ≥20 paraphrases of the four endings (meaning fixed), re-scored under
the 1929 witness via `common.slp` + the `v41` softmax → decide ending-wording (fixable) vs genuine
ignorance (question dies). Output `results/tsar_autopsy.json`.
*Done when:* no surviving row has witness < 50%; every casualty explained; Tsar verdict recorded.

**1b. modern-reader-audit** — `src/modern_reader_audit.py`. The forced-choice arm is suspect (modern
reader scored 33% on lawful-appointment-vs-putsch, which it knows). Run Qwen2.5-3B on ~20 forced-choice
questions with undisputed famous answers. Near ceiling → arm reads knowledge, the Hitler item is broken
→ fix it. Near chance → arm cannot read knowledge → cut from Paper 2. Output `results/modern_audit.json`
+ keep/fix/cut memo.
*Done when:* ceiling result recorded and decision made.

**1c. frequency-control (model-internal first cut, no corpus needed)** — `src/frequency_control.py`. The
pipeline already separates the two channels: `raw = slp(stem, option)` is preference **with** the clue;
`base = slp(neutral, option)` (the `cal_score` base term) is context-free familiarity — a **print-share
proxy**. For each gated finding, decompose into (familiarity component, context-driven component
`cal = raw − base`); a finding "survives the confound" iff it holds on the context-driven part. For the
availability claim: does the pre-model's weight on the most-recent precedent **exceed** its base
familiarity with it? Output `results/freq_control_internal.json` + a one-page memo naming which
Section-7 claim (strong "diagnostic expectations" vs weak "the record weighs the future") the data
supports.
*Optional upgrade (Phase 2, needs corpus):* Fable's true version — retrieval counts of each precedent in
the Ranke corpus by 5-year window (separates recency from volume), regress bets on print share. Gated on
whether the ~160B-token corpus is reachable on disk/Alps; if not, the proxy stands with the limitation
stated.
*Done when:* every gated finding tagged survives/does-not; memo names the licensed claim.

**1d. control-distribution** — `src/control_battery.py` + `results/control_bank.json`. Author ≥40 mundane
between-cutoff (1900–1929) outcomes that arrived on schedule — treaty/exposition locations, election
results, expected successions, on-time technologies — in the same memoir format (neutral stem + 5
phrasings + 4 options + truth), obeying the famous-truth constraint. Score with the `v41` pipeline + the
gate. Output `results/control_gaps.json` (distribution + 10/50/90th pct); re-express every surviving
catastrophe gap as a percentile of it.
*Done when:* a figure places each catastrophe on the mundane-gap distribution; the draft redefines
"foreseen"/"unforeseen" against it.

**Phase 1 deliverable:** `reports/validity_pass.html` (clear-report style, define-first): gated table,
modern-reader verdict, familiarity decomposition, control yardstick — each with what-we-had / what-we-
establish-now. Append to `LOG.md` as results land.

### Phase 2 — design + larger runs (local, bigger; after Phase 1 decides what survives)

- **event-universe** — Barro–Ursúa disaster list 1900–1933 for covered countries + a written rule for
  political catastrophes + a question-generation recipe; generate mechanically; **freeze**
  `prereg/event_universe_v1.json` (dated, unrun). Kills "you picked the questions."
- **generation-arm** — scale up `production.py`: K≈200 memoir completions per event under the pre-event
  reader; hand-validated classifier (≥0.9 agreement on 100 labels) for "volunteers the truth."
  `results/volunteer_rates.json`. Support claims then cite generation, not betting.
- **error-bars** — bootstrap every gap over ≥20 ending + ≥5 stem paraphrases; CI on each; flag any
  overlapping the control median. No table ships a gap without an interval.
- **german-fix** — rewrite German endings under the recipe; re-run `certify_defr.py` against frozen
  marks. Pass → certified; fail → demotion paragraph + explicit caveat on the French–German Depression
  contrast.

### Phase 3 — paper surgery + the long pole

- **split-restructure** — two manuscript skeletons; move every v4 section/table/appendix per Part A;
  no empirical claim in both; mapping file recording where each paragraph went.
- **witness-1939-spec** — spec what Paper 2 needs from the 1939 model before it trains (measurable Table
  2 rows, new 1933–39 questions, the 1929/1939 certification exam). Write immediately — the Alps queue is
  the long pole.
- **seed-ensemble** (compute-permitting) — second seed of the 1913 model; recompute flagship gaps
  cross-seed for a model-level SE.

## 5. Integrity gates (before trusting any number)

- Witness-gate leaves **no** surviving row with witness < 0.50.
- Modern-reader ceiling test is reported **before** any keep/cut decision.
- The familiarity decomposition is shown for the **known/control** items too, so we can watch the
  baseline move — same discipline that caught the `11_inside_out` probe lighting up its negative control.
- Three flagship numbers cross-checked by hand against `v41_battery.json` after the gate (Ebert +9.8,
  rev_debts +0.2, panic +0.02) to confirm covered rows didn't shift.
- Every gap carries its interval by end of Phase 2. Dated `LOG.md` entry per goal; `CONTEXT.md`
  checkpoint at the end.

## 6. Scope note

Phase 1 is "the analysis as Fable sequences it," on assets already in hand — no 1939 model required.
Phases 2–3 are mapped but gated on Phase-1 results and JV's go-ahead (Paper 2 genuinely waits on the
1939 model). The same frequency-control logic back-applies to the contamination audit's one live positive
(US civil rights, `11_inside_out`) — worth a paragraph there, not part of this plan's execution.
