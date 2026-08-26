# Paper & retrain strategy — foresight + contamination detector

**Date:** 2026-07-10
**Version:** v1
**Author:** Opus (strategy session with JV)

**What this is:** the publication and decontamination strategy settled this session — one place to hand a
co-author. Compiled from the conversation; grounded in:
- `12_foresight/CONTEXT.md` (compiled truth), `12_foresight/LOG.md` (trail).
- `12_foresight/reports/validity_pass.html` (Fable Phase-1), `reports/technote/memory_meter_technote.pdf` (v1.1),
  `reports/remembering_meter.html`, `reports/superlatives/date_spike_full.html`.
- `12_foresight/results/superlatives/contamination_triage.json`, `bignull_v3_summary.json`,
  `memory_detector_v3_summary.json`, `date_spike_base_big.json`.
- `notes/post-fable-plan-for-v4.md`, `notes/foresight_split_and_todos.md` (Fable's two-paper sketch).

---

## 1. The pitch (one paper, one line)

> We built AI minds frozen in the past to measure what history could see coming — and caught them secretly
> remembering the future. So we built a way to tell machine memory from machine bluffing, proved it by editing
> a memory in and watching the needle move, and used it to separate real foresight from the models peeking.

This is the **twist framing**: foresight is the frame, the contamination discovery is the complication, the
detector is the rigour, the screened findings are the payoff. It reads to a general-science editor in one
sentence, it's surprising, it's rigorous, and it's broadly relevant (AI you can trust) *while staying a
science-question paper*.

## 2. One paper, not two (the decision)

**Paper 1 (target: Science / Nature / NHB)** = the foresight instrument **with the contamination detector
folded in** as its trust mechanism, plus the screened findings. **Paper 2 (target: QJE)** = the disasters /
economics, and it **waits for the decontaminated retrain**.

Do **not** spin the detector out as a separate ML/interp journal paper. Reasoning (fits JV's constraints):
- The detector is **load-bearing** for Paper 1 — a referee's first move is "aren't these models polluted?", and
  the screen is the answer; it must be *in* the paper, applied to *these* findings.
- JV has **no ML-venue network** — a paper he can't confidently place is a liability, not an asset.
- Separating risks a **gotcha**: a standalone "our models are contaminated" preprint, without the screen beside
  it, reads as an attack on the foresight paper.
- The ML footprint JV wants = an **optional arXiv preprint** of the *method* (probe + ROME + confound battery),
  posted for priority and as a citation target — **never on the critical path**, upgradeable later by someone
  with the network.

## 3. Division of labour — three instruments, honest bounds

Settled after all the stress-testing (this is the paper's methodological takeaway):
- **Year-spike (best-worded, MAX over ≥5 phrasings) = the clean PRIMARY signal** for "does the model know a
  date." It has **no realness bias** by construction. This is what the foresight screen actually runs on.
- **Memory meter (internal probe) = a narrow SPECIALIST** — its distinctive, causally-validated job is catching
  a **confident confabulation** the output cannot (0.92 vs 0.76 confidence). Its known liability, the realness
  leak, is category-specific and largely orthogonalisable (per entity-type), with a residual.
- **Corpus check = the gold standard.** Both signals are **pre-filters, not proof**; they flag candidates for
  the corpus grep (and, ultimately, the retrain).

Slogan: **trust the year-spike as primary; use the meter for what it uniquely does; confirm with the corpus.**

## 4. Framing discipline (keeps it a science paper)

Sell the **historical finding**, keep the meter as a **tool**. Novelty-of-method is *not* the claim, so an ML
reviewer can't kill it on method novelty; the *finding* is the claim. Two killer figures carry the meter in the
main text — the confident bluff (Treaty of Bordeaux, 80% on a year that never was) and the ROME flip (change
the memory, the needle moves, controls flat). All heavy mechanics → Methods + Extended Data + the arXiv
preprint. **Do not use the Jacobian lens**: a linear probe + causal ROME check is the right rigour for a tool;
the lens is a different question, 4B-scale-risky, and would pull the paper into AI-reviewer territory.

## 5. Novelty positioning (pre-empt "isn't this just X?")

Be honest up front. The **output** detectors (our date-spike, Min-K% Prob, Time Travel) are ~80% the existing
**training-data-contamination / effective-knowledge-cutoff / model-self-knowledge** literature — cite it
generously (Shi et al.; the "Dated Data" effective-cutoff work; Kadavath; Ravfogel INLP for the fix). The
genuinely-new tread is (a) the **fiction-null calibration** (invented events as a certain no-knowledge floor)
and (b) the **application** — purpose-built frozen models as scientific instruments + **contamination-aware
screening** that *salvages* inferences rather than just detecting leaks. Note it is **not "hallucination"**
(that's the mirror image — confidently *wrong*; this is confidently *right* about the forbidden).

## 6. What survives the screen (what the paper reports)

The **contamination triage** is the new witness-gate — run it first; it decides what belongs in the paper.
On the current (contaminated) models it sorts the findings cleanly:
- **SAFE (report as genuine foresight):** the German republic / **Ebert** (+9.84), the **Rentenmark** (+6.77,
  even date-clean), and the **1907 panic** (the 1900 guesser is genuinely pre-1907). Plus the memoir
  "availability" predictions (Fed-holds, putsch) — the 1929 guesser is date-clean on 1929/1933.
- **SUSPECT (drop or heavily caveat):** the **"foreseen" WWI / revolution findings** (war-breadth, US-entry,
  debt-repudiation) — the 1913 model *knows* those events; that "foresight" is the model peeking.

So the paper's honest claim narrows to the *unforeseen* flagships + the panic — which were the interesting
ones anyway — and drops the "everyone saw it coming" column. Note Ebert also needs a **name-free rebuild**
(monarch / republican president / dictator / council) to claim the *structure* was unforeseen rather than the
*name Ebert* being unknown.

## 7. The retrain plan (separate workflow)

The models are contaminated, so the "frozen mind" premise isn't literally true. To make it true — and to give
Paper 2 clean measurements — **retrain decontaminated models at cutoffs 1913 / 1929 / 1938.**
- **Why 1938:** it is a *witness* for 1929–38, turning the Depression and Hitler's rise (today memoir-only,
  unmeasurable) into **measured foresight gaps** — the two disasters the rare-events literature rests on. This
  is Paper 2's core new material. (Dropping 1900 costs the panic finding; a ~1946 model would later let a 1938
  *guesser* measure "was WWII foreseeable.")
- **Decontamination = grep, then purge sources, then fix the filter.** The tell (a 1913 model that knows "Civil
  Rights Act of 1964") means whole **modern documents** slipped the date filter — so it's source-level. The
  complete sieve is `year_sieve.py`: any doc in a bucket containing a **year after that bucket's cutoff** is
  contaminated; aggregate by **source**, purge the modern sources wholesale, and **fix the date filter that let
  them in** (the root cause — else the next retrain re-leaks). Complement with `find_contamination.py` (entity
  markers). Corpus lives on **Alps / cloud** (too big to bring local); the greps stream, so they run there as a
  CPU batch job.
- **Acceptance gate = the same date-spike.** After retrain, re-run it: the leaked events (1914, 1917, 1939,
  1945, 1964) must **collapse to the ~13% fiction floor.** The instrument that *found* the leak *certifies* the
  fix — a clean narrative loop.

## 8. The narrative arc (where the new outcomes inject)

Structure the story so the contamination is a **disclosed, screenable** limitation, not a hidden killer:
1. A frozen-model foresight instrument. 2. The twist — the models leak the future (date-spike). 3. The screen —
triage; some findings survive on the dirty models (Ebert, Rentenmark, panic). 4. Paper 1 ends pointing forward:
"to measure the disasters that matter cleanly, retrain decontaminated models." 5. **Paper 2 opens with the
clean 1913/29/38 models** (certified by the detector) and delivers the **new outcomes** — the Depression and
Hitler as gaps, plus clean re-runs of the WWI/revolution findings the screen flagged. The new outcomes are the
**payoff of the retrain**, so they land *late* in the arc — earned, not assumed. Optional: a "clean
confirmation" coda in Paper 1 (Ebert & Rentenmark re-measured on a retrained 1913) if the retrain lands in time.

## 9. Open decisions / next steps

1. **Corpus check + retrain** on Alps/cloud (the separate workflow above) — the real proof spine.
2. Draft **Paper 1** (twist-framed).
3. **PI to vet** the 40 mundane control facts; decide Tsar (drop vs clean rebuild); approve the **Ebert
   name-free rebuild**.
4. Optional polish: fake-name "presidents" to close the meter's last realness residue; extend the rich-context
   suite; the optional arXiv method preprint (priority).
5. Coordination: only one Claude stack should own `CONTEXT.md` (full-rewrite file); `LOG.md` is append-safe.
