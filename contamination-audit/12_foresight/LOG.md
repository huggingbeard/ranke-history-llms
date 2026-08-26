# Log — Foresight sub-study (12_foresight)

Append-only. Newest at the bottom. See CONTEXT.md for compiled truth.

### 2026-07-07 14:00
Started Fable-review Phase 1 (plan: notes/post-fable-plan-for-v4.md). **Witness-gate (1a) done.**
New `src/witness_gate.py` re-analyses `results/v41_battery.json` under the rule "keep a finding only
if the witness — the post-event reader that lived through the event — puts >= 0.50 on the truth."
Pure re-analysis, no model. Output `results/gate_report.json`.
Result: of 10 measured (covered) findings, **7 PASS, 3 DROP**.
- DROPPED: rev_tsar_fate (witness p=0.015 on "put to death", gap +2.71 — the witness bets "driven
  from his throne"); rev_who_rules (witness 0.043, +0.72, witness bets "the generals of the old army");
  panic_first_to_fail (witness 0.130, +0.92). All three: the 1929 witness itself cannot find the truth,
  so the gap was a ratio of two tiny numbers.
- The two flagship UNFORESEEN findings SURVIVE cleanly: **emp_kaiser_1919 / Ebert** (witness 0.776,
  gap +9.84) and **infl_end / Rentenmark** (witness 0.719, gap +6.77).
- Biggest structural change: **revolution's event-mean gap collapses +1.21 -> +0.20** (only rev_debts,
  "debts repudiated", +0.20 foreseen, survives). The Russian Revolution's apparent unforeseeability was
  entirely an artifact of two questions the witness couldn't read. This is exactly Fable's warning; the
  abstract's Tsar finding must go, Ebert + Rentenmark stay.
Next: Tsar autopsy — paraphrase the endings and re-score under the 1929 witness to decide whether
p=0.015 is fixable ending-wording or genuine model ignorance (question dies).

### 2026-07-07 14:15
**Tsar autopsy (1a) done** — `src/tsar_autopsy.py` -> `results/tsar_autopsy.json`. Loaded the 1929
witness, scored the original item (reproduced p=0.015), 22 meaning-preserving rewordings of the
execution ending, and a "clean duel" (execution vs only-FALSE fates, the competing-true "deposed"
option removed).
- Rewording does NOT rescue it: across 22 paraphrases the witness never exceeds **0.077** on the
  execution while " been driven from his throne" is on the ballot. So 0.015 is NOT phrasing.
- Clean duel: with the competing-true option removed, witness p(execution) = **0.768** — the 1929
  model KNOWS the Romanovs were executed.
- **VERDICT = BROKEN OPTION SET.** The item pit two TRUE outcomes (deposed AND executed) against each
  other; the model reasonably picked the more salient true answer. The +2.71 "unforeseen" gap was never
  a foresight measurement. Not model ignorance, not contamination (1918 << 1929). Item must be rebuilt
  with false-only distractors or cut.
- FOLLOW-UP (cheap, recommended): re-score the 1913 GUESSER on the clean-duel option set; that yields a
  VALID Tsar-execution foresight gap and may resurrect a real finding ("was the execution, cleanly
  posed, foreseeable in 1913?"). Deferred to the item-rebuild step.
Net for 1a: witness-gate + autopsy together confirm Fable — the Tsar finding collapses for a diagnosable
reason, while Ebert (+9.84) and the Rentenmark (+6.77) stand.

### 2026-07-07 20:30
Note on process: turns do NOT run in the background. When a reply ends I stop and wait; "continuing"
does not keep working unattended. No compute/cost was lost in the idle gap — each analysis is minutes on
the local 3060. Resumed and pushing through Phase 1.
Also: made `src/v41_battery.py` importable via a `__main__` guard (nothing imports it; `python
src/v41_battery.py` unchanged; verified: import loads NO models, all 33 items + funcs intact). This lets
the review scripts reuse the EXACT certified items + scoring instead of re-typing them.

**Modern-reader audit (1b) done** — two scripts.
1. `src/modern_reader_audit.py` -> `results/modern_audit.json`: ran Qwen2.5-3B over the whole battery via
   the certified arm. 71% on the 31 "clean" items (mean p(truth) 0.64). BUT confounded — a 3B reader is
   genuinely shaky on some 1907-33 specifics, several items are obscure (trust companies), and the
   archaic memoir wording pulls a modern model. Both pre-flagged ambiguous items missed at p=0.00
   (rev_tsar_fate -> "driven from throne"; nazi_path -> "armed march on Berlin").
2. `src/ceiling_facts_audit.py` -> `results/ceiling_facts_audit.json`: the clean test. 13
   universally-known facts (moon=Armstrong, WWII=Hitler, Mona Lisa=da Vinci...), plain register, SAME 3B
   reader, EXACT certified arm. **100% correct (13/13), mean p(truth)=0.89.**
CONCLUSION: the forced-choice ARM reads knowledge — it turns a known fact into the right pick. So the
battery misses (incl. the 33% Hitler item) are ITEM/framing difficulty, NOT an arm defect. **KEEP the
arm**; fix/flag ambiguous items; caveat that the arm needs a competent reader + clean options. nazi_path
pulls even a knowing reader to the putsch template — treat as either a misleading item OR a genuine
availability effect, to be separated in 1c (frequency-control).
Next: 1c frequency-control — decompose each guesser prediction into familiarity (context-free base) vs
context-driven, to test whether the "availability heuristic" is real reasoning or just print frequency.

### 2026-07-07 21:00
**Frequency-control (1c) done** — `src/frequency_control.py` -> `results/freq_control_internal.json`.
Scored every item under its OWN guesser (1900/1913/1929), split each prediction into familiarity
(base = softmax slp(neutral,option)) vs contextual (raw = mean_stems softmax slp(stem,option)); lift = raw-base.
- POOLED: regress p_raw on p_base over 132 option-instances -> R^2=0.51, slope 0.65, corr 0.71,
  mean|lift|=0.15. So context does ~half the work — the instrument is NOT just a token counter, but
  familiarity is a strong co-driver.
- The two flagship UNFORESEEN findings behave DIFFERENTLY (this is the key result):
  * **Rentenmark (infl_end) SURVIVES cleanly.** Truth "Rentenmark" base 0.00->raw 0.00; but the guesser's
    wrong bet "Goldmark" is base 0.02 -> raw 0.56 (+0.53). Both currency names start UNfamiliar, yet
    context actively BUILDS "Goldmark". So the Rentenmark being unforeseen is genuine conceptual foresight
    (the 1913 mind constructs the wrong specific expectation), NOT "Rentenmark is a rare word."
  * **Ebert (emp_kaiser_1919) is PARTLY CONFOUNDED.** Truth "Ebert" base 0.01->raw 0.00; the bet "Kaiser
    Wilhelm" is base 0.98 -> raw 0.99 (+0.01, i.e. context adds nothing). The Kaiser is simply the familiar
    DEFAULT; context does not build it. So the +9.84 gap rides substantially on NAME-familiarity (Ebert was
    an unknown name in 1913), not purely on structural unforeseeability. FIX: add a name-free reformulation
    (monarch / republican president / military dictator / council) to claim "the republic was unforeseen"
    rather than "the name Ebert was unfamiliar." Same lesson as the Tsar (two-truths) and the contamination
    audit (name != knowledge).
- Memoir PREDICTIONS show strong context-built (availability) signatures: dep_banks "Fed holds"
  base 0.22->raw 0.94 (+0.72); nazi_republic_fate "dictatorship" 0.00->0.52 (+0.51); dep_gold "gold
  abandoned" 0.02->0.68 (+0.66). Context, not frequency, builds these.
- Foreseen findings partly ride on truth-familiarity: Morgan base 0.93->0.77, US-entry 0.85->0.56 (context
  slightly LOWERS the already-familiar truth). Minor caveat to flag.
- Availability-vs-frequency verdict: **MIXED** on the internal proxy (only 9/19 wrong bets are context-
  lifted above familiarity). The strong "diagnostic expectations" claim needs the Phase-2 corpus
  print-share test (recency vs volume) to clinch; the internal proxy can't settle it alone.
Net 1c: the findings are not a token-counter artifact, but Ebert needs a name-free rebuild and the strong
availability claim is not yet earned. Rentenmark is the cleanest flagship.
Next: 1d control-distribution (mundane-outcome yardstick), then the Phase-1 validity report.

### 2026-07-07 22:30
**Control distribution (1d) done** — `src/control_battery.py` -> `results/control_gaps.json`. 40 mundane
1900-1929 outcomes (categories chosen by JV: coronations/successions, reparations conferences, ship
launches, royal marriages, elections, Nobel prizes, sterling's return to gold parity, on-schedule
canals/flights/expeditions), both windows, scored through the certified arm. Facts verified (web) for the
date/detail-sensitive ones. **PROVISIONAL — facts drafted for PI vetting.**
- Coverage: only **16/40** pass the witness-gate. WHY (not a bug): 13 the arm abstained on; 11 the WITNESS
  itself is wrong/unsure — the 1913 model thinks Roosevelt (not Taft) won 1908; the 1929 model bets Wilson
  (not Harding) for 1920, confuses Young/Dawes, and misses the 1924 Labour government. So the small 4B
  witnesses are factually SPOTTY even on events they "read." The gate correctly excludes them. **Honest
  limitation for the paper: these models are imperfect witnesses; only gate-passing findings are
  measurable, which caps breadth (not the validity of what passes).**
- The 16 clean mundane gaps span [-0.21, +4.28]; percentiles 10/25/50/75/90 = 0.00/0.15/0.25/1.58/2.73;
  mean +0.97. Hardest ordinary surprises: a conference named "Dawes" (+4.28), Genoa (+3.36), the 1924
  Coolidge election (+2.11).
- **Catastrophes placed on the mundane scale (the payoff):**
  * emp_kaiser_1919 / Ebert +9.84 -> **100th percentile** (above EVERY mundane gap)
  * infl_end / Rentenmark +6.77 -> **100th percentile**
  * war_us_entry +0.49 -> 56th; rev_debts +0.20 -> 37th; panic_rescuer/war_breadth ~+0.01 -> 19th;
    panic_severity -0.14 -> 6th.
  So "unforeseen" now means "beyond the 90th percentile of normal historical surprise," and the two
  flagships are literally off the scale. NB: the biggest mundane gaps (Dawes, Genoa) are ALSO
  name-familiarity-driven yet only reach ~+4, so Ebert/Rentenmark being far larger is NOT merely
  name-novelty — this partially rebuts the 1c name caveat on Ebert.
Caveat: 16 is a modest yardstick; expanding coverage (easier distractors / fewer options where the 4B
witness is unreliable) would firm the percentiles. Headline (flagships at 100th pct) is robust regardless.
Next: Phase-1 validity report (HTML), then checkpoint CONTEXT.md.

### 2026-07-07 23:00
**PHASE 1 COMPLETE.** Deliverable: `reports/validity_pass.html` (built by `src/build_validity_report.py`,
which reads the six result JSONs — no hand-typed numbers; didactic/define-first). All four Fable "first
tier" checks done, on existing data/models, in minutes of local GPU. Summary:
1. Witness-gate: 7/10 findings survive; Tsar/rev_who_rules/panic_first_to_fail drop; revolution mean gap
   +1.21->+0.20.
2. Tsar autopsy: broken option set (two true answers), not ignorance — model knows the execution (76.8%
   in a clean duel).
3. Arm validity: forced-choice arm reads knowledge (100% on 13 universal facts through the exact arm); the
   33% Hitler item is a bad question, not a broken arm.
4. Frequency-control (internal): Rentenmark survives cleanly (context builds "Goldmark"); Ebert partly a
   name artifact (Kaiser bet is pure familiarity) -> rebuild name-free; availability-vs-frequency MIXED,
   needs corpus test.
5. Control yardstick: Ebert (+9.84) & Rentenmark (+6.77) at the 100th percentile of 16 gated mundane gaps
   (range 0..+4.28); foreseen findings in the lower half. Only 16/40 gated — the 4B WITNESS models are
   factually spotty (honest limitation).
DECISIONS PENDING PI: (a) vet the 40 control facts; (b) Tsar — drop or rebuild clean; (c) approve Ebert
name-free rebuild; (d) Phase-2 go-ahead (corpus print-share test, event-universe, generation arm,
german-fix, error-bars, 1939-model spec). New scripts: witness_gate, tsar_autopsy, modern_reader_audit,
ceiling_facts_audit, frequency_control, control_battery, build_validity_report. v41_battery.py made
importable (guarded; verified unchanged). CONTEXT.md checkpoint still to do at session close.

### 2026-07-07 23:45
**INSTRUCT robustness re-run done** (PI asked). `src/instruct_rerun.py`; refactored `control_battery.py`
into a reusable `run(model_map,out_name)`. Instruct heads on the hub: **1913-1202-Instruct,
1929-1202-Instruct only — NO 1900 instruct**, so window-A guessers stay base-1900 (witnesses 1913-Instruct).
Scored raw (same teacher-forcing as base) for like-for-like; caveat: instruct adds a separate post-training
channel (robustness check, not ground truth). Outputs: results/{v41_battery_instruct, gate_report_instruct,
control_gaps_instruct}.json.
**Result — the "instruct is more confident" hypothesis did NOT pan out:**
- Coverage did NOT improve: catastrophe measured-coverage 10/26 -> **9/26**; control gated **16 -> 14 / 40**.
  Instruct causes MORE raw-vs-calibrated disagreement (arm abstentions), not fewer. So the 16/40 coverage
  limitation is NOT fixed by chat-tuning; it needs better item design (fewer options / easier distractors).
- **Ebert (flagship) is robust and STRONGER:** witness 0.776->0.748, gap **9.84 -> 12.34**, 100th percentile
  under both. Survives the model-variant swap cleanly.
- **Rentenmark: signal intact and LARGER, but the arm abstains under instruct.** guesser ~0.00, witness
  0.719->0.507, gap 6.77 -> **10.6** — MORE unforeseen. But under instruct raw & calibrated disagree on the
  modal, so `covered=False` and it drops from the gated set. This is a **coverage-rule technicality, not a
  reversal** — the finding is intact/stronger. FLAG: the coverage rule is fragile (drops a strong-signal
  item); consider relaxing it, and/or the Ebert-style name-free rebuild for infl_end too.
- **Tsar confirmed broken** under instruct too: witness 0.015 -> 0.004, also abstains. Consistent.
- Foreseen findings stable across variants (war_us_entry gap 0.49->0.39; rev_debts 0.20->0.24; panic_severity
  -0.14->-0.10). flu_name & infl_victims newly covered under instruct.
TAKEAWAY for tomorrow: instruct is a PASSED robustness check for Ebert and re-confirms the Tsar is broken,
but it does not buy more coverage. Two to-dos surfaced: (a) look at the arm's coverage rule (it abstains on
the strong Rentenmark signal under instruct); (b) coverage/breadth needs item redesign, not instruct models.
PI reviewing decisions 1(vet control facts)+2(Tsar)+3(Ebert) later. Picking up tomorrow.

### 2026-07-07 (first-name probe: the fair, frequency-invariant version)
Closed out the first+last-name knowledge probe after the forced-choice kept getting confounded by name frequency.
- Forced choice with frequency-matched distractors STILL failed: Sartre->Jean 96%, and Reagan->Ronald flipped to a
  false 88% with rare distractors (was 14% with common ones). That flip is the proof of no contamination — the
  score is pure base rate, not knowledge. Ruled the forced choice out entirely (src/firstname_freqmatch.py).
- Switched to a frequency-INVARIANT measure (src/firstname_pmi2.py):
  boost = log P(first | "the {role} {surname}'s first name was") - log P(first | "his/her name was").
  This puts rare (Igor) and common (Karl) names on equal footing. Clean for 26/29 people:
  KNOWN high — Bismarck 7.8, Lenin 6.3, Nietzsche 6.5, Marx 5.6, Stravinsky 6.8, Picasso 4.9, Hitler 5.1, Napoleon 4.7.
  POST-CUTOFF at/below 0 — Reagan -1.0, Parks -4.1, Pollock -1.1, Guevara -2.1, Orwell -0.1, Camus +0.2, Feynman -0.1.
  TIME-LOCK — Mussolini 0.6 (1913) -> 4.2 (1929); he took power 1922, between the cutoffs. Clean single-person flip.
  Fixes both old confounds: Igor no longer sub-chance (6.8), Picasso clearly known.
- Residual (honest): Sartre->Jean 4.4 and Beauvoir->Simone 3.7 stay positive post-cutoff because Jean/Simone are the
  role-default first names for a French intellectual (role leakage, not person knowledge).
- Tried role-controlled baseline (subtract "the {role}'s first name was") to kill Sartre: it works on Sartre (0.4)
  but is the WORSE measure — over-subtracts archetypes (Bismarck 1.7, Picasso -0.1, Lenin 1.1 collapse to ~0) and
  invents new false positives (Reagan +4.2, Guevara +4.0). Kept the generic boost, flagged the two French names.
- Graph: two stacked panels (1913, 1929), x = birth+25, green training window / grey after cutoff, segmented trends
  (within-window and post-cutoff drawn separately, no smoothing across the boundary), solid dots >=2 nats else hollow.
  figures/fig_firstname_boost.png ; results/prominence/firstname_boost.html (self-contained, table of all 29 with
  both baselines). Ranke instruct 1913 & 1929 (the meaningful models; 1900 has no instruct variant).

### 2026-07-08 (French swap + percent-change view)
Replaced the leaky French existentialists (Sartre->Jean, Beauvoir->Simone: role-default names) with Anatole France
and Charles de Gaulle; kept Camus. Result cleans the last residual:
  France->Anatole  +6.2 / +5.6  (pre-cutoff, distinctive name => real knowledge, +26,065% at 1929)
  de Gaulle->Charles +1.0 / +0.5 (famous only from 1940 => base rate; "Charles" is common, could have leaked, didn't)
  Camus->Albert    -0.6 / +0.2  (base rate, correctly unknown)
No post-cutoff false positives remain. Highest post-cutoff residuals now de Gaulle (+70%) and Presley (~+200%),
both below the boost>=2 knowledge threshold.
Added a percent-change view of the y-axis (per PI): y = (e^boost - 1)*100 on a log scale, 0% = no knowledge.
Both figures now written: figures/fig_firstname_boost.png (nats) and fig_firstname_pct.png (% change); the HTML
report embeds both. Panel title moved to top-right to clear the Bismarck label. src/firstname_pmi2.py.

### 2026-07-08 (data-driven knowledge line, strong-contrast zones, all-point labels, musician swap)
Three PI asks addressed on the first-name figure:
1. Knowledge line is now MEASURED, not eyeballed. Null = the 10 people who came of age >=1935 (both models
   provably cannot know them: Reagan, Parks, Pollock, Camus, Salk, Monk, Feynman, Berry, Guevara, Armstrong).
   Line = null mean + 2 s.d. -> 2.58 nats (+1,218%) for 1913, 2.59 nats (+1,227%) for 1929. Nearly identical.
   This resolves the "Elvis at +200% looks clean" confusion: a few-hundred-% point is far below the ~1,220% line,
   so it is floor noise, not knowledge. Solid dots clear the line, hollow do not.
2. Strong contrast: green fill above the line (knows the name) vs rose fill below (post-cutoff floor / no knowledge),
   replacing the faint grey band. Cutoff shown by the dashed vertical only.
3. Every point labeled via adjustText (installed into timelock-contamination venv) with thin leader lines.
Musician swap (PI): replaced Presley->Elvis (mild +1.1/+1.3 outlier) with Thelonious Monk and Chuck Berry.
   Monk->Thelonious -1.1/-2.0, Berry->Chuck -0.1/-0.0. Key result: the rarest-token first name in the set
   (Thelonious) is NOT inflated -- it sits BELOW the floor -- so the reverse worry (rare name self-inflates) is
   disproven. Neither a common name (Charles/de Gaulle +0.5) nor a rare one (Thelonious) fools the measure.
Files: src/firstname_pmi2.py; figures/fig_firstname_pct.png + fig_firstname_boost.png; results/prominence/
firstname_boost.html (updated captions explain the line, the zones, and the Monk/Berry test).

### 2026-07-08 (fixed circular threshold -> placebo null; "standout" statistic)
PI caught the knowledge line as circular: it was set from "post-1934 people = no knowledge", which assumes the very
time-lock the probe should DISCOVER (and would hide contamination). Rebuilt the null with no reference to time.
- For each person's frame, now score EVERY plausible wrong first name (same language + gender), not just the true one.
  POOLS[(lang,gender)] added; PEOPLE regained a lang field. ~844 wrong (surname->random name) boosts per model.
- Pooled wrong-name null is NOT centered at 0: mean +1.3 nats, sd 2.7, 95th pct ~5.8. DIAGNOSIS: the raw boost is
  mostly LANGUAGE/ROLE PRIMING -- "the German statesman X's first name was" lifts every German name several nats;
  person-specific knowledge is only the increment on top.
- Fix = per-frame recentering. STANDOUT(person) = boost(true) - mean(boost of wrong names in same frame). The null
  becomes the within-frame deviations of wrong names: centered at 0, sd ~1.6. Knowledge line = 95th pct = 2.38 (1913),
  2.34 (1929) nats. Non-circular, time-agnostic. Also auto-kills role leakage (Jean sits in the null -> bar rises).
- Result (stricter, honest): clears the line in BOTH models = Bismarck(+7.6), Napoleon, Marx, Nietzsche, A.France,
  Stanley -- all distinctively-named, all pre-cutoff. Time-lock now shows as a RISE not always a crossing:
  Mussolini -4.2->+0.4, Stravinsky -0.1->+1.2 (higher in 1929 but approach rather than vault the strict bar).
- KNOWN LIMIT (documented): a genuinely-known person whose first name is common in their language can't be credited --
  Einstein->Albert stays negative in both (Albert doesn't stand out among German names). Inherent to a first-name test.
- Figure: y = standout (nats) + a right-margin density strip showing the random-combo null (bell centered 0, 95th/99th
  marked) -- this literally shows "the variability of random combos" the PI asked for. adjustText labels all points.
  fig_firstname_boost.png (nats, primary) + fig_firstname_pct.png (% vs typical same-language name). HTML rewritten.

### 2026-07-08 (poison scatter; standout is conservative, not sharp)
PI: "does a common first name poison the standout test? show a scatterplot" + "we know it knows Hitler (free continuations)".
Built src/firstname_poison.py: x = true name's commonness (base-rate percentile among same-language names, from the
model), y = standout. Among people the model KNOWS (famous before cutoff, green; de Gaulle excluded as came-of-age-
pre-1929 but famous 1940), fit slope is negative but MODEST: r=-0.23 (1929), -0.25 (1913). So commonness does drag
the score (Einstein->Albert -1.9, Tolstoy->Leo, Curie->Marie, Wagner->Richard all low), but it is not the whole story:
base-rate commonness != confusability -- "Napoleon"/"Otto" are frequent tokens yet distinctive, so ride high.
The sharper point: HITLER is ground-truth known (free-continuation probe) yet standout = +2.0, right at the line, and
"Adolf" is only 36th-pct common. So the test under-credits a confirmed-known person.
VERDICT (in fig + html): the first-name standout is CONSERVATIVE/SPECIFIC, not sensitive -- essentially no false
positives (no post-cutoff person clears the line = no contamination signal) but many false negatives (misses knowns
with ordinary names, e.g. Hitler, Einstein). Clearing the line proves knowledge; sitting below does not prove ignorance.
=> good as a "no-contamination" check, poor as a sharp time-lock. Recommend leading the time-lock story with the
free-continuation / direct-knowledge evidence; keep first-names as the conservative specificity check.
Files: figures/fig_firstname_poison.png, results/prominence/firstname_poison.html.

### 2026-07-08 (self-correction probe -- PI's idea, sharpest signal yet)
Two PI directives: (a) raw boost is the sensible measure, add a permutation cutoff not a give-up; (b) NEW idea:
prime a WRONG first name + real surname + real claim-to-fame, see if the model reverts (Oswald Bismarck -> Otto?).
First checked the boost with a column-permutation (hold name, permute surname): it revealed the RAW boost is ALSO
priming-confounded -- colmean(Friedrich)=+6 across all German frames, colmean(Vladimir)=+6.1 across Russian frames,
so Nietzsche/Lenin's big raw boosts are mostly frame-priming, not person-specific. So no probability-shift metric is
clean. Pivoted to the self-correct idea.
Self-correct (src/firstname_selfcorrect.py): R = logP(true) - logP(wrong) in the wrong-primed frame; R>0 = reverts.
R=0 is a PRINCIPLED cutoff -- self-normalising (true vs the SPECIFIC lie given), so commonness & priming cannot enter,
and NO null needed (answers PI's "sharper cutoff" better than a permutation).
KEY RESULT -- clean control: every post-cutoff person ACCEPTS the wrong prime (1929 top completion: "Claude" de Gaulle
59%, "Donald" Reagan 20%, "Theodore" Monk 67%, "Ruth" Parks 22%) -- no knowledge, nothing to revert to. Iconic knowns
OVERRIDE it: told "Oswald Bismarck" -> Otto 66% (R+5.1); "Viktor Lenin" -> Vladimir 30% (+3.0); "Margot Curie" ->
Marie 26% (+2.8); Wagner +1.5, Napoleon +0.5. Time-lock: Wagner -1.6->+1.5, Napoleon -2.0->+0.5, Stravinsky -4.7->-0.7,
Hitler -1.9->-0.3 (rise across models).
HONEST limit: common-named / plausibly-primed knowns don't cleanly revert -- Einstein->Albert -3.1 (model outputs
generic Johann/Ernst, not Albert), Hitler->Adolf -0.3 (Adolf 8% is 2nd, just under the primed Alfred 10%), Tolstoy->Leo
(prefers Lev/Anton), Picasso, Mussolini. So still conservative for ordinary names, but the CONTROL is clean and R=0 is
principled. figures/fig_firstname_selfcorrect.png, results/prominence/firstname_selfcorrect.html (top-prediction column
shows the correction happen/fail verbatim).

### 2026-07-08 (within-lang permutation cutoff; robust self-correct + reverse control)
1. WITHIN-LANGUAGE permutation for the boost (src/firstname_perm.py, from cache). Null = wrong same-language pairs,
   per (lang,gender); cutoff = that language's own 95th pct. Figure SHOWS why per-language matters: German/Russian/
   Spanish clouds run past +6 (frame lifts any same-lang name), English ~+1.8. Clears p<=0.04 (1929): Bismarck 0.01,
   Napoleon 0.04, Nietzsche 0.03, France 0.01, Stanley 0.01. Does NOT clear: Lenin (0.37), Stravinsky (0.27), Hitler
   (0.09) -- their names sit inside their language cloud (not person-specific). False positives: only Salk (p=0.02),
   ~ the nominal 5% rate over ~13 post-cutoff people => cutoff is calibrated. Clean, honest, but governed by name
   DISTINCTIVENESS within language, not knowledge per se. figures/fig_firstname_perm.png.
2. ROBUST self-correct (4 decoys avg) + REVERSE control (src/firstname_selfcorrect2.py, GPU). Robust R weaker/honest:
   1929 reverts Napoleon +2.0, Lenin +1.8, Bismarck +1.2, Marx +0.7; most others near/below 0; Hitler -0.7 (up from
   1913 -2.7). The single-decoy version had overstated reversion.
   REVERSE CONTROL is the clean win = "parrot rate" P(echo the WRONG name | wrong-primed):
     post-cutoff PARROT the lie -- Parks echoes "Ruth" 36% vs "Rosa" 9% (prefers the lie!), Monk "Theodore" 32% vs
     "Thelonious" 0%, Reagan "Donald" 19% ~ "Ronald" 21% (indifferent), de Gaulle 24%.
     known REFUSE the lie -- Bismarck echoes "Oswald" 7% but keeps "Otto" 84%; Lenin wrong 3% / true 79%; Napoleon 8%.
   Parrot rate separates known (3-9%) from post-cutoff (12-36%); threshold ~12-15%. HITLER parrot 9% = on the KNOWS
   side (resists the lie) even though robust R=-0.7 -> reverse control vindicates the free-continuation ground truth
   where reversion alone missed it. Exceptions (known but higher parrot): Tolstoy 20% (Leo/Lev), Curie 16%.
   figures/fig_firstname_selfcorrect2.png, results/prominence/firstname_selfcorrect2.html.

### 2026-07-08 (Russia dig: the 1913 "1917 revolution" is mostly PREDICTION, not hard leakage)
Behavioral verification (src/russia_dig.py, both instruct models):
- DATE (pin it down): "in what year was the Tsar overthrown?" -> 1913 model peaks 1905 (73%), only 21% on 1917;
  1929 peaks 1917 (91%). CONTROL "Russo-Japanese war ended?" -> both peak 1905 (96-99%), so the method CAN localize
  a known date. So the 1913 model, pinned, defaults to the 1905 revolution it really knows; it only volunteers "1917"
  when primed for "most recent" (context-dependent) -> INCONSISTENT = confabulation. 1929 is consistent = knowledge.
- FATE (the sharp test): "the deposed Tsar Nicholas II was ___". 1913 top=exiled, execution-cluster only 4%; 1929
  execution-cluster 20% (5x). A model that ingested 1918 would know he was EXECUTED; the 1913 model predicts the
  historically-normal EXILE. Strong evidence AGAINST deep 1918 leakage.
- LEADER: both name Lenin (1913 71%, 1929 86%) -- not discriminating (Lenin led the Bolsheviks since 1903, pre-cutoff).
- SAMPLES: 1913 confabulates details ("Alexander II abdicated" -- wrong Tsar; "driven into exile", "deposed and
  imprisoned"); 1929 sharper ("ended the Romanov dynasty", "Romanov line extinct", "Soviet Government").
VERDICT: the 1913 model shows fuzzy ANTICIPATION (revolution + Tsar falls + ~1917 + Lenin -- all extrapolable from the
1905 revolution + Bolsheviks + instability) but lacks the CONTINGENT facts (the execution; consistent date). Prediction
with a faint prior, not memorized post-cutoff text. figures/fig_russia_dig.png. NEXT: Jacobian-lens on middle layers
(does 1917/executed surface internally then get suppressed?), then the 1000-sample judged probe.

### 2026-07-08 (CORRECTION: flat-vs-spike test shows the 1913 model IS contaminated, selectively)
PI's catch: if the 1913 model knew nothing, its year-guess would be FLAT over the future window (1916~1917~1920). It
is not. For "Tsar overthrown" the 1917 bar is 30x its neighbours (83% of all future-year mass). My earlier "prediction
not leakage" verdict was WRONG on the date. Built the systematic test (src/date_spike.py): score P(year) over +/-8
window for 13 dated events, spike = P(true)/mean(others).
CONTROLS (pre-1913, 1913 model should spike): Victoria 1901 83x, Russo-Jap 1905 248x, SF quake 1906 262x, Titanic
1912 48x -- all spike hard, method validated.
1913 MODEL ON POST-CUTOFF EVENTS (should be flat if clean):
  LEAKED (hard spikes on the exact year): "Bolsheviks seized power" 1917 = 124x (89%); "WWI began" 1914 = 41x (72%).
  PARTIAL: Armistice 1918 9x, Spanish flu 1918 9x.
  FLAT / clean: "Tsar overthrown" 1917 0x (argmax 1915 -- phrasing-dependent!), US-entered-war 1x, German hyperinflation
    1923 0x, Lindbergh 1927 2x, Wall St crash 1929 2x.
1929 model spikes on everything <=1918 (WWI 9811x, Bolsheviks 626x, flu 4175x) and weak on 1929-edge -- correct.
REVISED VERDICT: the 1913 model has SELECTIVE, PHRASE-KEYED, SHALLOW contamination. The two most canonical WWI-era
facts (WWI=1914, Bolsheviks-seized-power=1917) leaked as hard date spikes; less-canonical phrasings ("Tsar overthrown")
and 1920s facts (hyperinflation, Lindbergh, crash) did NOT. And the DEEP detail still absent (Tsar's fate = exiled not
executed, 4%). Pattern fits instruct-tuning leakage of common encyclopedic phrases, not deep integration. This is the
real contamination signal the audit exists to find. date_spike is now the primary detector. figures/fig_date_spike.png.
Next: relaunch jlens fit (does the leaked 1917 live mid-model?), then the 1000-sample judged probe.

### 2026-07-08 (BENCHMARKED contamination test + WW2/atomic/civil-rights expansion + write-up)
PI: "how do we know 2x is clean? show the distribution for true knowledge." Built the benchmark (src/date_spike_full.py):
per model, anchor the spike scale with KNOWN (19 famous pre-1913 facts) and a NO-KNOWLEDGE NULL (15 FICTIONAL events
with invented names + assigned year -> the model cannot know a date that never existed).
BENCHMARK: null median 0.8x/0.9x, 95th pct 1.9x (1913) / 2.3x (1929); KNOWN median 262x (1913) / 322x (1929), range up
to ~6600x. So the scale is anchored: 1x=noise, ~2x=indistinguishable from a fiction, ~300x=real memory. => 2x IS clean.
Leakage threshold = null 95th pct.
Expanded to 22 post-cutoff events (1914-29, WW2/atomic 1933-45, civil rights 1954-69). Findings:
- 1913 model LEAKS (>1.9x): WWI 1914 (41x), Bolsheviks 1917 (124x), Spanish flu 1918 (9x), and far-reach: WW2-ended
  1945 (8x), Civil Rights Act 1964 (27x!). CLEAN: US-entered-war, Versailles, hyperinflation, Lindbergh, WW2-began,
  Pearl Harbor, D-Day, Brown, Montgomery, MLK-Dream, JFK, MLK-assass, Moon.
- 1929 model correctly spikes on ALL 1914-1919 (WWI 15501x, flu 4175x, Versailles 1500x -- real knowledge, validates),
  then LEAKS past its cutoff: WW2-began 1939 (45x), WW2-ended 1945 (24x), Pearl Harbor (5x), Civil Rights Act 1964 (5x).
INTERPRETATION: contamination is (a) real, (b) SHALLOW -- every leaked spike 2-130x sits BELOW the known median ~300x,
(c) SELECTIVE/patchy -- leaks Civil Rights Act 1964 but clean on D-Day/Moon/MLK; tracks how date-canonical a fact is,
not distance, (d) PHRASE-KEYED (Bolsheviks-phrase leaks, Tsar-phrase doesn't). Fits instruct-tuning leakage of canonical
"Event: Year" one-liners over a period-locked base. Deep detail still absent (Tsar exiled not executed).
Deliverables: figures/fig_date_spike_full.png, results/superlatives/date_spike_full.html (full write-up, doc header,
method, benchmark, table, interpretation, caveats). jlens fit resumed in background.

### 2026-07-08 (VECTOR: contamination is in PRETRAINING; instruct is self-tune that AMPLIFIES)
PI: "instruct unlikely, it is self-tune; redo with base models." Base models were cached (Ranke-4B-1913/1929, no
-Instruct). Ran date_spike on BASE (src/date_spike_base.py, events in src/events_dates.py).
RESULT -- the BASE model leaks too (base null 95th 1.4x, base known median only 49x):
  1913 base: WWI 1914 = 13x, Bolsheviks 1917 = 17x, Civil Rights Act 1964 = 4x -- all ABOVE base threshold.
  instruct AMPLIFIES the same events: 13->41x, 17->124x, 4->27x (instruct known median 262x -- tuning sharpens ALL).
  1929 base leaks post-1929 too: WW2-began 5x, WW2-ended 13x. No event sits in an "instruct-only" band.
=> contamination originates in PRETRAINING (base), instruct sharpens it. figures/fig_date_spike_base.png.
SFT INSPECTION (cached ranke-1913-sft-dataset / -seedresponses): response_model = uzh-echist-org/Ranke-4B-1913 for
100% of rows -> the instruct data is SELF-GENERATED by the base (confirms PI's "self-tune"). Keyword hits split:
  "atomic" (36x) = atomic THEORY (Dalton 1807) -- benign; "civil rights" (11x) = pre-1913 (Nantes, 1792, 1829) -- benign.
  BUT "1914" (16x) includes the base's OWN leaked fact, in a response dated 1907: "In 1914 came the great war, which,
  beginning in the Balkans". So the self-generated SFT carries the base's contamination -> tuning on it is a
  self-distillation loop that AMPLIFIES the faint pretraining leak. Named revolution entities absent from SFT
  (bolshevik/soviet/lenin/hitler = 0), yet base still spikes Bolsheviks->1917 (17x) => that leak is pure pretraining.
VERDICT: root cause = imperfect pretraining time-lock (canonical post-cutoff dates leaked into the corpus); the
self-tuned instruct model amplifies by distilling the base's own leaked outputs. Instruct is amplifier, not source.

### 2026-07-08 (HEAD-TO-HEAD: Talkie-1930-13B base vs Ranke-1929-4B base, identical date-spike set)
Ran the identical 56-event date-spike set (events_dates.py) on Talkie-1930-13B base via RunPod (A100 80GB, ~14 min,
~$0.35, pod self-terminated; custom full_logits validated == forward, maxdiff 0.0). Scored Ranke-1929 base locally on
the same set (rescore_base_ptrue.py). Metric = P(true year) %, one shared threshold = 10% (95th pct of both models'
pooled fictional events). src/date_spike_headtohead.py -> Figure 3 + date_spike_headtohead.html.
VALIDATION (1914-19, inside both windows): both know them -- WWI Ranke 98%/Talkie 97%, Bolsheviks 85/96, US-entered
90/96, Armistice 69/92, flu 95/83, Versailles 96/78. Known-fact median Ranke 80%, Talkie 87%.
FINDING 1 -- RANKE LEAKS POST-CUTOFF, TALKIE IS CLEAN. Events after both cutoffs (>1930): Ranke-1929 leaks WW2-ended
1945 = 45%, WW2-began 1939 = 23%, Civil Rights Act 1964 = 22%, Pearl Harbor 1941 = 14%, D-Day 1944 = 13% (all >10%
threshold); Talkie-1930 sits at chance for ALL of them (3-10%), only a marginal Hiroshima 12%. So another lab's ~1930
base model is far cleaner than Ranke on post-cutoff contamination.
FINDING 2 -- TALKIE ALSO HAS BETTER IN-WINDOW COVERAGE. Late-1920s facts Ranke barely knows, Talkie does: German
hyperinflation 1923 Ranke 14%/Talkie 61%, Lindbergh 1927 0%/65%, Wall St Crash 1929 7%/76%. Ranke-1929-base has GAPS
in its own window (esp. Lindbergh 0%); Talkie-1930 is more complete.
CONCLUSION: Talkie-1930 is the cleaner and more complete time-lock on BOTH axes -- it knows its own window better and
leaks less past its cutoff. Ranke-1929's pretraining both under-covers the late 1920s AND over-leaks WW2/civil-rights.

### 2026-07-08 (200-item head-to-head: Talkie-1930 vs Ranke-1929 base — pattern CONFIRMED and sharpened)
Scaled the date-spike battery to 200 (events_dates_big.py: 50 known / 70 fictional-null / 80 test, WWI->COVID-2020).
Reran both: Talkie-1930-13B base on RunPod (200 events, maxdiff 0.0, pod self-terminated ~$0.4) + Ranke-1929 base local.
Shared threshold 12% (95th pct of 140 pooled fictional events -- tight). Known median Ranke 77%, Talkie 87%.
FINDING 1 (confirmed, stronger): RANKE LEAKS POST-CUTOFF, TALKIE CLEAN. Ranke-1929 leaks WW2-ends 1945=45%, UN-founded
  1945=26%, WW2-begins 1939=23%, Civil Rights Act 1964=15%, D-Day 1944=13%, Gagarin 1961=13%, Pearl Harbor 1941=12%
  (~7 leaks, concentrated 1939-45 + a couple 1960s). Talkie-1930 leaks only ONE: Fall of France 1940=24% (10 yrs past
  its cutoff). Everything post-~1945 (Cold War/space/modern) is clean for BOTH.
FINDING 2 (confirmed, much stronger with N): RANKE UNDER-COVERS ITS OWN 1920s. Late-window facts Ranke-1929-base fails
  but Talkie-1930 knows: Lindbergh 1927 Ranke 0%/Talkie 65%, Lenin dies 1924 2%/37%, Wall St Crash 1929 4%/56%, League
  of Nations 1920 7%/29%, March on Rome 1922 11%/57%, hyperinflation 1923 15%/60%. So Ranke's cutoff both LEAKS future
  (WW2) AND is LOSSY on its own final decade; Talkie is cleaner AND more complete.
CONCLUSION: robust across 200 items -- Talkie-1930 is the better time-lock on both axes. Ranke-1929's corpus both
under-covers the late 1920s and over-leaks 1939-45. figures/fig_date_spike_headtohead_big.png, date_spike_headtohead_big.html.

### 2026-07-08 (contamination triage — post date-spike)
Retrain/deconta parked as a separate track, so the guessers are KNOWN-contaminated. Built
`src/contamination_triage.py` -> `results/superlatives/contamination_triage.json`: screen every finding by
date_leak (guesser P(true year) > its own 95th-pct fiction floor, from events_dates.NULL) x outcome_lean
(top bet==truth or p_truth_pre>=.5). No corpus, no retrain. Base guessers; fiction floors 1900=9% 1913=8%
1929=9%.
**Result on the 7 GATED findings: 4 SAFE, 3 SUSPECT — and every SUSPECT is a "foreseen" WWI/revolution item.**
- SUSPECT (guesser knows the event AND leans to the truth -> likely leakage, not foresight):
  war_breadth (WWI, date 44% vs 8%, p 0.99), war_us_entry (44%, p 0.56), rev_debts (Bolsheviks 51%, p 0.80).
- SAFE:
  * emp_kaiser_1919 / Ebert (+9.84): guesser LEAKS the WWI/republic date (39%) but bet the WRONG outcome
    (Kaiser, p 0.00) -> the tree correctly KEEPS it. This is the key case (contaminated on date, clean on outcome).
  * infl_end / Rentenmark (+6.77): guesser DATE-CLEAN on the hyperinflation (3% < 8%) -> doesn't even know the
    event happened. Cleanest possible finding.
  * panic_rescuer (+0.02) & panic_severity (-0.14): 1900 guesser date-clean on the 1907 panic (2%) -> the one
    legitimate "foreseen-in-kind" result (a 1900 mind reasoning, not peeking).
- BONUS: every memoir "availability" prediction is SAFE — 1929 guesser date-clean on the 1929 crash (7%) and
  Hitler 1933 (1%). So the Fed-holds / putsch predictions are genuine, not leakage (validates Paper-2 material).
TAKEAWAY: the paper's genuine foresight = the UNFORESEEN flagships (Ebert, Rentenmark) + the panic (clean 1900
guesser). The WWI/revolution "foreseen" findings are largely CONTAMINATION, not foresight, and must be dropped
or heavily caveated. The triage is the new witness-gate (decides what survives) and is itself a Paper-1 method
("ex-ante history on models you can't fully trust"). Caveat: one canonical phrasing per event (phrasing-
sensitive) — but 44-51% vs 8% and 2-3% are far from the floor, so the calls are robust; a phrasing bootstrap
would firm the borderline flu items (15%). Reconciles with 1c: Ebert is (a) uncontaminated here AND (b) still
partly name-driven (1c) -> name-free rebuild still wanted.

### 2026-07-09 — Remembering-meter: reading contamination from inside the model (new sub-study)
Built a linear probe on residual-stream activations (last token, per layer) of Ranke-4B-1929 BASE that reads
"retrieval vs confabulation" — i.e. does the model genuinely remember a fact or make it up. Trained on KNOWN
(pre-1913 real) vs NULL (invented) events; output is a 0..1 "remembering-meter". Files (src/): probe_mri.py,
real_entity_null.py, minimal_pairs.py, rome_inject.py, rome_flip_by_layer.py, cutoff_shift.py, mri_report.py.
Numbers saved to results/superlatives/*_summary.json; report at reports/remembering_meter.html.

FINDINGS (all Ranke-4B-1929 BASE unless noted):
1. PILOT (probe_mri): known-vs-fictional separable from inside at AUC 0.99 (best layer 35/36); entropy-only
   baseline 0.71. Generalizes to UNSEEN classes: post-cutoff UNKNOWN reads 0.25 (n=64), post-cutoff LEAKED
   reads 0.70 (n=16). So the meter, trained only on known/invented, correctly flags contamination it never saw.
2. CONFOUND CONTROLS:
   a. Weird-word confound -> real, frequency-matched fake names (Treaty of Rennes etc.). Survives.
   b. Confidence confound (the serious one). Base model CONFIDENTLY confabulates fakes: Treaty of Bordeaux 80%,
      Rotterdam 77%, Siege of Glasgow 59% (16/40 real-name fakes are "confident"). NON-CIRCULAR test (ground
      truth = mine, confidence matched): real memory vs confident-lie -> probe 0.92 vs output-confidence 0.76.
      Probe reads truth BEYOND confidence. NB confident confabulation is in the BASE model, not an instruct artifact.
   c. Construction confound -> minimal pairs: real "Treaty of X" vs fake "Treaty of X", same template, real cities
      both. Probe 0.98 vs output 0.93. Backwards example: fake Treaty of Le Havre 72% sure vs real Treaty of
      Portsmouth (1905) 12% sure — confidence is no guide, probe still right.
   d. AIRTIGHT: ROME-style rank-1 weight injection (layer 9 down_proj, keyed on subject) writes a fake date in,
      surface byte-identical. Injection works 5/5 (P(true) 3-13% -> 84-96%, argmax = planted year), localized
      (control reals survive 97%, other fakes stay 0.01). Probe flips 2/5 fully at layer 28 (Bristol, Cardiff ->
      1.00) but 3/5 barely move (Rennes 0.09, Glasgow 0.00, Leeds 0.01) DESPITE all 5 outputting the year. KEY:
      not fooled by forced output -> reads NATURAL storage, not the answer. By-layer (rome_flip_by_layer): read
      far downstream (mid-layers OOD/unreliable); Bristol/Cardiff persist, others fade. Interpretation: probe
      detects naturally-stored memory (= what real contamination is), rejects a surgical bolt-on. Good for a
      contamination detector.
3. PROOF OF CONCEPT (cutoff_shift, 1913 vs 1929, each its own meter, both AUC ~0.99): identical 1914-29 events.
   CLEAN shift 0.32 -> 0.85 (Lusitania 0.20->1.00, Somme 0.03->0.72, Mussolini 0.21->0.67, Versailles 0.61->0.94,
   League 0.53->0.91). Controls flat (pre-1913 known 1.00/1.00, invented 0.00/0.00). Two honest non-shifters,
   both consistent w/ prior findings: (a) 6 events already leaked into 1913 (WW1 begins, Russian Rev, Armistice,
   Spanish flu, hyperinflation, Wall St Crash) read 0.90->0.97 — 1913 base leaks its own future; (b) 3 events are
   1929 gaps (Lenin dies, Lindbergh, penicillin) read 0.16->0.14 — Ranke-1929 undercovers late-1920s (matches the
   Talkie head-to-head gap finding).
PAYOFF: unlike the date-spike test, the meter needs no year — works on any post-cutoff claim (capitals, authorship,
CEOs). A general internal contamination scanner. NEXT: repeat on other model families; tighten confidence-matching;
non-dated battery; consider whether "natural storage" probe can be fooled by longer/MEMIT-style injection.

### 2026-07-09 (cont.) — Adversarial stress tests + v2 memory detector
FABLE'S TEST (obscure_real.py + obscure_real_parties.py): 20 GENUINE obscure pre-1929 treaties, meter = minimal-pairs
probe. Refutes the crude "reads realness" confound — deeply-obscure real treaties the model is blank on (Björkö, Zuhab,
Nemours ~0.00) sink to where invented ones sit. BUT the meter is genuinely FOOLED on some: Küçük Kaynarca meter 0.99 &
Hünkâr İskelesi 1.00 despite model knowing NEITHER date NOR signatories (independent parties forced-choice: both 0% vs
8% chance). Also false negatives (Roskilde/Belgrade: model knows them, meter ~0). On this hard OOD set the meter is only
~2/3 right (vs 98% in-distribution) — a strong IN-DISTRIBUTION detector that degrades OOD (likely residual distributional
tell on exotic names). NOT a bulletproof pillar. My earlier "(B) it really remembers them" claim was WRONG — retracted.
LESSON: meter reads "has a memory trace" (broader than date-recall) — Portsmouth (real, date 12%, meter 0.86) & Locarno
(real, date 23%, meter 1.00) confirm memory != date.

v2 MEMORY DETECTOR (treaties_battery.py + memory_detector.py): TWO-signal ground truth (known = nails date OR
signatories; unknown = fails both; invented = no-memory). PI grew the battery: +31 obscure early-modern treaties (Baltic,
Russo-Ottoman, Ottoman-Polish, Silesian) -> 76 real + 56 invented, 127 clean (58 known, 13 UNKNOWN, 56 invented).
v1 (trained real-vs-invented) vs v2 (trained on memory, real-unknown moved to negative): AUC vs memory truth 0.78 -> 0.86;
false positives 25 -> 14; real-unknown called remembered 10/13 -> 5/13. With only 7 unknowns (first pass) the gap was
negligible (0.91 vs 0.93); 13 unknowns made it real. v2 still leaves 14 FP — better, not bulletproof. Best layer 25.

INJECTION (rome_inject.py) now includes REAL treaties: Locarno 1925 (model already remembers: meter 1.00->1.00, no flip,
the 'memory != date' exhibit) and Björkö 1905 (model truly lacks it: meter 0.00->1.00, clean real-fact flip). Fakes:
Bristol/Cardiff flip to 1.0, Rennes/Glasgow/Leeds don't persist. TODO: fig_rome_inject legend still calls Locarno/Björkö
'fakes' (they're real) — fix. TODO: amend remembering_meter.html with the honest 'where it breaks' + v2 + Björkö/Locarno.

### 2026-07-09 (cont.) — Exposure ground-truth + combined rule + wide-net post-1930 scan
EXPOSURE (ngram_exposure.py, plot_exposure.py, make_exposure_figs.py): Google Books pre-1929 frequency as an
independent exposure proxy (robust fetch w/ backoff; caught+fixed throttling-zeros). Behaviourally-KNOWN treaties median
exposure 2.1e-9 vs UNKNOWN 1.3e-11 (~165x) -> the "unknown" label is fair (model barely saw them). Meter tracks exposure
(r=0.40 real). Among unknowns, meter's false positives have ~11x more exposure than the ones it gets right (5.9e-11 vs
5.5e-12) -> mistakes lean toward faint-trace-catching (b), but with exceptions (Polyanovka meter 0.99 @ ZERO exposure =
pure leak; Hünkâr İskelesi highest-exposure unknown yet meter 0.26 = miss). VIOLINS (fig_exposure_violins): meter median
invented 0.02 / real-unknown 0.25 / known 0.99; both behaviourally blank but real-unknown rides higher (16%->38% called
remembered) = the realness leak, visible; known towers over both so memory is the dominant signal. CSV: treaty_exposure.csv.

COMBINED RULE (combined_rule.py, fig_combined_rule): leak detector = flag if year-spike OR meter fires. On treaty battery
(58 known, 69 no-memory): year-spike alone recall 83% (10 missed, 1 FP); meter alone 78% (13 missed, 14 FP); OR 91%
(5 missed, 15 FP). OR halves missed leaks (two detectors miss different things: spike misses parties-only-known, meter
catches 5) at the cost of the meter's realness-leak false alarms — right trade for leak-hunting (cheap error vs expensive).

WIDE-NET POST-1930 SCAN (post1930_battery.py 82 items by topic/era, wide_net.py, fig_wide_net + fig_pearls): year-spike +
general date-event meter (probe_mri KNOWN-vs-NULL @L35) on post-cutoff facts. BOTH-light-up (robust): Germany surrenders
1945 (spike 36% meter 1.00), UN founded 1945 (26%/0.56) — matches prior head-to-head. Meter-only cluster (~13): Spanish
Civil War, FDR elected, Hungarian Revolution, Anschluss, Suez, Indian independence, Korean War, Pearl Harbor, First World
Cup, JFK-assassinated, PRC, Falklands — likely ENTITY/topic-familiarity leak (all hinge on pre-1929 entities; some have
earlier versions e.g. Hungarian Rev 1848), NOT specific-event contamination; year-spike is the specificity anchor.
Controls validate: invented + far-future (9/11, COVID, WWW, Chernobyl, Moon, Berlin-Wall-falls) all read DARK. CSV:
wide_net_scan.csv. NEXT: corpus check the candidates (esp. the meter-only cluster) to tell faint trace from realness leak.

### 2026-07-09 (cont.) — Best-wording, ROME-for-contamination (null), technical note
BEST-WORDING (best_wording.py): single arbitrary prompt UNDER-COUNTS both indicators (both phrasing-sensitive). Fix: per
event score >=5 specific phrasings; pick most-fluent (max mean per-token logprob = "most likely in a book") OR take max
indicator over the set. Recovers the band the PI remembered: WW2-begins meter 0.36->1.00, Hitler 0.27->0.97, Kristallnacht
0.00->0.94; Civil Rights Act year-spike 9%->30% (real leak under canonical phrasing). Controls stay dark at max (invented
0.00, Moon 0.04). => wide_net.py single-prompt run UNDERSTATES; re-run in max-over-phrasing mode. Formulation ideas: verb/
voice, entity- vs event-forward, model-mined canonical forms, register, multilingual, entity-cloze, distractor minimal pairs.

ROME-FOR-CONTAMINATION (rome_contamination.py): counterfactual-resistance pilot on treaties (known vs unknown ground
truth): plant a FALSE year on phrasing A, test on paraphrase B whether TRUE year survives. NULL: known lie-generalises
69%/true-survives 0% vs unknown 73%/0% — identical. ROME overwrites even a 94% memory to 0%; insertion success is
uninformative about prior. CONCLUSION: editing does NOT firm up contamination. Right tools instead: membership inference /
Min-K% Prob on the fact's canonical sentence (Shi 2024), causal tracing (Meng 2022, the diagnostic half of ROME, NOT yet
run), and the CORPUS CHECK (gold standard, next step).

TECHNICAL NOTE: reports/technote/memory_meter_technote.tex -> .pdf (9pp, MiKTeX pdflatex, 5 figs embedded). Replication-
grade: formal probe def, code (slp/extract/fit), validation+controls (AUC 0.99, injection causal, cutoff-shift), failure
modes (realness leak + exposure r=0.40 + v1->v2 0.78->0.86), combined OR leak rule (recall 0.91), best-wording, ROME
analysis, replication checklist, 14 refs. NEXT: corpus check the wide-net candidates; re-run wide-net max-over-phrasings.

### 2026-07-09 (cont.) — j-lens infeasible; trajectory null; CAUSAL TRACING wins
Q: does reading deeper (subject token / mid-layers / Jacobian lens) beat the single-late-layer meter and shrink the
realness leak? deep_probe.py (subject vs last token, all layers): NO — subject-token AUC 0.79 vs last 0.86, leak
unchanged; realness leak GROWS with depth but AUC does too (no free operating point). trajectory_probe.py (PI's Q2:
are false-positives late bloomers?): NO — known & real-unknown track together at ALL depths (both real); only invented
diverges down. Trajectory classifier 0.86->0.87 AUC, leak 0.24->0.21 (marginal). The realness leak is fundamental to a
generic 'memory-ness' direction (known & real-unknown are both real entities). JACOBIAN LENS: calibration = 31 min/prompt,
16.4GB (spills 12GB 3060) -> ~99h for the Gutenberg corpus = INFEASIBLE locally; and it's the wrong tool anyway.

CAUSAL TRACING (causal_trace.py, Fable's rec, ROME diagnostic) = THE WIN. Corrupt subject-token embedding (3x emb std),
restore clean residual at subject-last-token per layer, measure recovery of P(true-year sequence). Pre-registered bar:
known-vs-real-unknown AUC>=0.75 (meter is ~0.50 there), known-vs-invented>=0.85, invented~0. RESULT: storage-strength
medians known 0.42 / real-unknown 0.00 / invented 0.00; AUC known-vs-real-unknown 0.99, known-vs-invented 0.99, stored-vs-
not 0.99. VERDICT WIN. Known facts show a mid-layer (~L12) storage hotspot (ROME signature); real-unknown & invented show
NONE. Beats the realness leak decisively. Caveats: needs the model to produce the answer when clean (finds RETRIEVABLE
storage); ~37x costlier than meter -> confirmatory rung. Pipeline now: cheap scan (meter+year-spike, best-worded) ->
causal-tracing confirmation -> corpus check. NEXT: tuned lens (2nd opinion, local Gutenberg fit); note causal tracing
already answered the core Q.

### 2026-07-10 (realness-leak fix — memory meter v3)
Built the orthogonalisation fix for the meter's "realness leak" (meter partly reads presence-in-the-record,
not memory). GPU-FREE, entirely from cached activations. src/memory_detector_v3.py (+ _sweep.py) ->
results/superlatives/memory_detector_v3_summary.json (+ _sweep.json), figures/fig_memory_detector_v3.png.
METHOD: among the NO-MEMORY items (real-unknown + invented, both behaviourally blank, differ only in
realness) iteratively find+remove (INLP null-space projection) the direction separating real from fake,
until realness is undecodable there; then fit the memory probe on the nulled features. CV-honest (null dirs
+ probe fit on train fold, applied out-of-fold). Layer 25 (v2's). Also enriched labels already present in v2
(date+parties two-signal truth).
SWEEP finding (memory_detector_v3_sweep.py): there is a CLIFF. Nulling 10-18 directions closes the leak AND
keeps the external obscure treaties clean; nulling harder (>=19 dirs, stop<=0.55) destabilises them
(Kucuk Kaynarca blows up 0.19->0.77). Locked the safe operating point at stop=0.65 (~13-16 dirs); the win is
robust across the whole 0.80-0.58 plateau, not knife-edge.
RESULT v2 -> v3 (safe point):
- corr(meter, log book-exposure) among unknown reals: **+0.33 -> -0.13** (the leak, gone; the exact test I
  promised the PI).
- real-unknown vs invented meter gap: **+0.21 -> +0.05**; meter on real-unknown 0.35 -> 0.20 (~ invented 0.15).
- realness-leak AUC (real-unk vs invented via meter): 0.69 -> 0.57.
- memory AUC (real-known vs real-unknown): **0.86 -> 0.90** (preserved, slightly improved — not traded away).
- EXTERNAL obscure treaties (never trained): studied 0.92 -> 0.91 (kept high); unstudied 0.24 -> 0.16;
  **Kucuk Kaynarca 0.35 -> 0.19** (the tech-note poster-child FP, now correctly "not remembered").
- HONEST RESIDUAL: 1 stubborn external false positive remains (Wanghia 1844, 0.65) — the realness ceiling is
  reduced, not perfectly eliminated. Next lever if wanted: enrich Stack C with many more real-but-unknown
  treaty activations (a small GPU extraction), which orthogonalisation alone can't manufacture.
NET: the realness leak is closed on every in-distribution metric and improved out-of-distribution, with
memory preserved. The meter is now a materially cleaner pre-filter; still a pre-filter, confirmed by
best-worded year-spike + corpus check, per the tech note.

### 2026-07-10 (realness fix on a BIG diverse sample — 278 items)
PI asked to prove the realness fix on a big sample. Built src/bignull_battery.py: 278 items across 5
categories (wars, Nobel laureates, painters, novelists, presidents), 4 classes — famous / obscure /
invented / false-fact. REAL entities (216) verified against Wikipedia by research agents (each cited);
INVENTED (62) verified NON-existent by a research agent (it documented which candidates it REJECTED for
colliding with real wars/people). false-fact = real famous NON-presidents framed as president (real name,
no such memory). src/bignull_extract.py (GPU, Ranke-4B-1929 base) -> bignull_acts.npz (278x37x2560) +
date-spike labels. src/bignull_analyze.py (GPU-free) -> bignull_v3_summary.json + fig_bignull_v3.png.
Behavioural labels: 39 studied, 167 blank (117 real-but-blank + 50 invented), 72 ambiguous. Layer 30.
FINDINGS (all cross-validated, 3 meters compared):
1. **Memory reads cleanly and GENERALISES: AUC 0.95** (studied vs blank) across all 5 categories — far
   beyond treaties. The meter is a strong memory detector on a big diverse sample.
2. **The overall realness leak is SMALL (v2 leak AUC 0.55)** — the dramatic treaty-only leak (0.69) was
   partly treaty-specific / small-sample. Reassuring for the meter.
3. **The residual leak is CATEGORY-SPECIFIC — in PERSON-NAME categories**: painters 0.68, novelists 0.79
   (a real person-name carries familiarity); wars 0.45 and Nobel 0.49 show NO leak.
4. A **GLOBAL** realness null is too blunt for heterogeneous data: it OVER-corrects wars (0.45->0.32) and
   makes painters WORSE (0.68->0.76), because a real-painter-name direction != a real-war-name direction.
5. A **PER-CATEGORY** null (null realness within each entity type) CLOSES the leaks cleanly: painter
   **0.68->0.52**, novelist **0.79->0.54**, war->0.53, nobel->0.58, overall 0.55->0.48, with memory kept
   (0.95->0.94). This is the correct fix in a multi-category setting.
HONEST CAVEATS: (a) the per-category null nudges the false-fact class up (0.22->0.46) — a side-effect of
person-name-realness directions bleeding into the un-nulled president category; 0.46 is ~chance (not
"confidently remembered"), a hard ambiguous case, but a real residue. (b) The PRESIDENT category can't be
realness-nulled the standard way — it has no invented items to contrast; adding fake-name presidents (a
small GPU extraction) is the lever to close that.
LESSON FOR THE PAPER: "realness" is category-specific — a real name means different things for a war vs a
painter. The correction must be applied within entity type. On a big sample the meter is cleaner than the
treaty analysis implied, and its residual leak is closable per-category. Scripts: bignull_{battery,extract,
analyze}.py; data bignull_acts.npz / bignull_meta.json.

### 2026-07-10 (tech note updated to v1.1)
Updated reports/technote/memory_meter_technote.tex with the morning's work: new section
"Closing the realness leak by orthogonalisation, and a multi-category test" (the v3 treaty fix numbers +
the cliff; the 278-item five-category battery with the three findings — memory AUC 0.95, leak small &
category-specific, per-category null closes it; honest residuals + the meter's proper role); the updated
layer-plateau figure (fig_layer_curve_big) as an updated pilot Fig; the bignull per-category figure; INLP
citation (Ravfogel 2020); abstract rewritten to the honest "division of labour" framing — year-spike is the
clean PRIMARY for "does it know a date" (no realness bias), the meter's distinctive contribution is the
narrower confident-bluff case, both pre-filters feeding the corpus check. Forward-refs added from the
failure-modes section; files list + version/date bumped. Compiles clean (12 pp, no undefined refs/cites,
verified via scratch jobname build). NOTE: the live memory_meter_technote.pdf is LOCKED (open in a viewer),
so the final PDF wasn't regenerated — close the viewer and rebuild (pdflatex x2), or I can rebuild on ask.

### 2026-07-10 — checkpoint (memory-meter contamination sub-study)
Built + exhaustively stress-tested the memory meter (see the many 2026-07-09/10 entries above for per-experiment
detail + numbers). CONVERGENT HEADLINE: the 1929 model's post-cutoff knowledge is SHALLOW — vague word/association
traces, not coherent frequency-beating factual memory. Triangulated by (a) causal tracing (WW2 faint, indistinguishable
from unknown), (b) frequency-controlled rich-context probes (near chance; model defaults to its pre-1929 era —
Oppenheimer->Einstein, invaded->France), (c) earlier forced-choice at chance. Real leaks are rare/specific: WW2-end, UN,
"Holocaust" term, Paris Accords 1973, faint atomic bomb. Vietnam / post-45 presidents / MLK / moon = clean by the honest
(year-spike) measure. KEY LESSON re-confirmed: the meter has a fundamental REALNESS LEAK (presidents 8/8 meter-flagged
while year-spike at chance) — trust the best-worded year-spike for date knowledge; the meter's only distinctive value is
catching a confident confabulation. ABANDONED this session: Jacobian lens (infeasible + wrong tool), tuned lens, ROME-to-
confirm-contamination (null), subject-token/trajectory probe (no gain), Min-K% Prob (frequency-confounded, same trap as
the old Reagan-flip name probe). VIABLE non-date tool = rich-context completion (raw + calibrated lift), survives the
frequency stack 6/7. OPEN: corpus check of the leak candidates (not run); reconcile technote v1.1 (it externally gained
an 'orthogonalisation fix' + 278-item validation NOT in our results — verify). CONTEXT.md rewritten to v7.0 carrying both
threads (foresight instrument unchanged + this contamination sub-study).

### 2026-07-10 10:30 — checkpoint
/ctx-save from the orthogonalisation/paper-strategy stack (a second Claude stack checkpointed CONTEXT.md
in parallel earlier today — merged, not clobbered). This session's arc: (1) read the date-spike work and
argued it's a cleaner contamination instrument than the forced-choice arm; (2) built the contamination
TRIAGE — screening the foresight findings against the guessers' date-leaks: Ebert/Rentenmark/panic SAFE,
the "foreseen" WWI/revolution findings SUSPECT (contamination, not foresight); (3) settled paper strategy —
ONE general-science paper folding the detector in, ML as optional preprint, Paper 2 waits for a
decontaminated retrain (cutoffs 1913/1929/1938); (4) built the realness-leak FIX (v3 orthogonalisation) and
validated it on a 278-item, 5-category battery (memory AUC 0.95; leak category-specific in person-names;
per-category null closes it) — GPU-free from cached activations; (5) updated the layer figure (plateau
29-35); (6) updated the technote to v1.1 (compiles clean, 12pp) with the honest "division of labour" framing
(year-spike primary, meter = confident-bluff specialist, both pre-filters). Resolved the other stack's flag:
the technote v1.1 claims ARE real (run this session; memory_detector_v3 + bignull_*). Reconciled their
"realness leak is fundamental" with "orthogonalisable": the leak is NOT fixable by WHERE you read
(layer/token) but IS largely removable by nulling the realness DIRECTION, per entity-type; a residual remains
(president/false-fact 0.46). Honest nulls kept: ROME-editing-as-proof dead; causal tracing a narrow confirmer;
global null too blunt; the >18-direction cliff. OPEN: corpus check + retrain (on Alps), Paper 1 draft,
Ebert name-free rebuild, control-facts vetting, fake-name presidents to close the residue.


### 2026-07-10 (PAPER v5 — the twist-framed Paper 1, built and verified)
Rebuilt the paper per the strategy note (notes/2026-07-10-paper-and-retrain-strategy.md): foresight_v5.tex
supersedes v4 (v4 files untouched). v4's central premise claim ("companion project checked... models clean
of post-cutoff knowledge") was FALSE per the 7/8 date-spike audit; v5 makes the leak + screen the paper's
arc: instrument -> exam -> THE TWIST (Sec 5, date-spike vs fiction floor: 1913 base WWI 13x / Bolsheviks
17x / CRA-1964 4.1x vs ceiling 1.4x; 9/22 and 5/13 post-cutoff events breach) -> THE SCREEN (Sec 6: date
spike primary, memory meter as bluff-catcher w/ Bordeaux 80%-confident-never-existed + probe 0.92 vs
output 0.76, ROME causal check Bjorko 0.00->1.00, cutoff-shift 0.32->0.85; triage rule leak x lean) ->
SCREENED RESULTS (Sec 7: funnel 26->10->7->4 clean + 3 suspect; war-breadth/US-entry/rev-debts RETRACTED
as unmeasurable — explicit in-text retraction; Ebert/Rentenmark strengthened by yardstick 100th pct of 40
mundane controls; Tsar moved to witness-gate drops w/ two-truths autopsy) -> memoir certified date-clean
(6.8%/0.6% vs 8.9% floor) -> implications recast (support: unrefuted-not-vindicated, rests on panic+duels;
weighting: hardened) -> conclusion points at 1913/29/38 retrain w/ date-spike acceptance gate. New refs:
Shi/Cheng-Marone/Golchin/Kadavath/Meng/Alain-Bengio/Ravfogel. NEW appendix B: full audit (spike table,
25-item triage table, gate table).
PIPELINE: make_numbers_v5.py (197 macros, all from result files, 0 TBD) + make_figures_v5.py (fig_gaps_v5
verdict-coloured 7-row dumbbell; fig_leak_v5 spikes vs fiction ceiling both models; fig_bluff_v5
confidence-vs-meter scatter; fig_rome_v5 = ROME redraw with CORRECTED legend — Locarno/Bjorko labelled
real, closing the 7/9 TODO; also patched rome_inject.py's own legend) + verify_v5.py (85/85 independent
re-derivations agree) + build_v5.py (numbers -> figures -> verify -> 2x pdflatex; latexmk dropped — MiKTeX
lacks perl). Fixed two overfull tables inherited from v4 (memoir 48pt, circ 78pt; isolated per-table test
harness). OUTPUT: paper/foresight_v5.pdf, 25pp, zero overfull / zero undefined refs.
DECISION (user Q, answered): do NOT wait for the clean retrain + 1938 to write Paper 1 — the twist only
works on the dirty models, the screen already protects the claims (leak fakes foresight, not blindness),
retrain = hostage risk, priority matters, 1938 feeds Paper 2. Clean-confirmation coda slot reserved; when
clean models land, re-run batteries + build_v5.py regenerates everything.
OPEN (unchanged): corpus check on Alps; Ebert name-free rebuild (caveat is in Sec 9 limits); PI to vet the
40 mundane controls (caveat in Sec 7); DE/FR wording.


### 2026-07-10 (control battery v2 — PI vetting session on the 40 mundane facts)
Built the PI vetting sheet reports/control_facts_review.html (src/control_facts_review.py): all items w/
exact stems, options (truth marked), results, gate status, reviewer flags. JV vetted v1. DECISIONS:
(1) extend battery to ~30 GATED items; (2) locarno fine as-is; (3) sterling_gold_1925 badly worded — "a
value fixed in dollars" is de facto the pre-war parity ($4.86), replaced w/ distinct outcomes (one-tenth
devaluation / half value / no fixed parity); (4) ROYAL NUMBERING RULE: current-officeholder questions must
never offer options the witness-period model knows impossible (Queen Victoria crowned 1902; dead popes
reigning 1925) and invented regnal names must respect numbering (think Charles III — a new Edward after
Edward VII can only be Edward VIII); (5) einstein_nobel fine.
IMPLEMENTED in src/control_battery.py v2: rebuilt edward_vii_crowned, george_v_throne, british_king_1925,
tsar_nicholas, austria_franz_joseph, german_wilhelm (v1 'Otto' distractor was BROKEN — both readers took
it, Otto-of-Bavaria interference), pope_pius_xi (now Benedict XVI / Leo XIV / Gregory XVII — all
numbering-valid conclave outcomes), york_wedding (Wallis Simpson + groom's-mother distractors out).
EXTENDED 40 -> 69 items (29 new: A-window Roentgen-first-Nobel, Marconi-Atlantic, Entente-1904,
Haakon-VII, Model-T, Peary, Olympics-1908, Delhi-Durbar, Poincare-1913, Bethmann, Simplon; B-window
Hindenburg-1925 (all-real-1925-candidates options), Lindbergh (kept despite known Ranke-1929 gap — gate
decides), Kellogg, Lausanne, League-seat, insulin, Tutankhamen, BBC, Jazz-Singer, Bonar-Law-1922,
Poincare-franc-1926, Stresemann-Locarno, Churchill-Exchequer, Olympics-1924, Capablanca, Anatole-France,
Graf-Zeppelin, Ederle). Output -> results/control_gaps_v2.json (v1 kept untouched); make_numbers_v5 /
verify_v5 / review sheet re-pointed at v2.
GPU QUEUE: battery run launched but WAITING — a vLLM Qwen2.5-VL-7B server in WSL holds ~21/24 GB
(anno-family extraction). JV: leave queued; common.py's VRAM guard starts it when memory frees. Then:
build_v5.py regenerates yardstick numbers + review sheet.


### 2026-07-10 (control battery v2 RAN — 30/69 gated, target hit; yardstick claim revised 100th -> 97th pct)
vLLM server freed the GPU; battery ran clean (control_gaps_v2.json; v1 untouched). GATED 30/69 — exactly
the ~30 JV asked for, with broad category coverage. The royal numbering-rule fixes WORK: edward_vii_crowned
(+1.45), german_wilhelm (+1.05, guesser now sensibly on "Frederick the Fourth"), tsar_nicholas (+0.42),
british_king_1925 (+0.08) all gate now (v1: all broken/excluded). New yardstick: median 0.69, 90th pct
4.10, mean 1.53.
SUBSTANTIVE CHANGE: Ebert (+9.84) & Rentenmark (+6.77) drop from 100th -> 96.7th percentile. ONE gated
mundane item beats Ebert: channel_swim_1926 (+12.12) — a PURE NAME LOTTERY (guesser bets Annette
Kellerman; no 1913 reader can name a 1926 swimmer). Reading: it marks the natural ceiling of name-shaped
questions (~12 nats), flagships near but below it; strengthens the case for the Ebert name-free rebuild.
GIFT COMPARATOR: norway_king_1905 (+6.32) — naming the king of a country that didn't yet exist; guesser
bets Oscar II (the old union king) EXACTLY as the 1913 reader bets the Kaiser; Ebert sits 3.5 nats above
even that. Paper prose rewritten at 3 sites (intro, Sec 7 yardstick para w/ Norway+ceiling story,
implications "exceeded only by a pure name lottery"); dropped the ungated uk_pm_1926 comparator (v1 prose
had cited it though it never passed the witness gate — my error, now fixed). New macros mundaneTopGap /
kaiserMundaneBelow / norwayGap etc.; verify_v5 extended to 89 checks, 89/89 PASS. Rebuilt: foresight_v5.pdf
25pp clean. Review sheet regenerated (control_facts_review.html, v2): 29 NEW facts awaiting JV sign-off.


### 2026-07-10 (control battery v3 — name-lottery exclusion rule; flagships back at 100th pct)
JV: "lose the channel swim". Implemented as a PRINCIPLED EXCLUSION RULE, not an ad-hoc deletion (documented
in battery header + paper Sec 7 + review sheet, so a referee diffing v2->v3 finds a rule): NO PURE NAME
LOTTERIES — a truth that is a private individual's name with no pre-event public existence (Ederle)
measures name-drawing, not guessability of change; its ~12-nat gap is a question-shape ceiling artifact.
Regnal/office names STAY (constrained public choices) — norway_king_1905 kept as the boundary case, which
keeps the Ebert flagship benchmarked rather than conveniently unbenchmarked.
v3 RUN (control_gaps_v3.json; v1+v2 kept): 68 items, 29 gated. Median 0.67, 90th pct 3.50, mean 1.16, max
+6.32 (norway_king_1905). Ebert +9.84 & Rentenmark +6.77 -> **100.0th percentile** again — all 29 mundane
changes easier; Norway comparator carries the prose ("the republic was harder to reach than the naming of
a new country's new king", 3.5 nats above it). Paper prose updated at 3 sites; rebuilt: 201 macros, 89/89
verify PASS, 25pp, zero warnings. Review sheet regenerated (v3, 68 items) — the ~28 remaining new items
still await JV's final fact sign-off.


### 2026-07-10 (SESSION SUMMARY — Paper 1 v5 written; yardstick vetted to v3; CONTEXT.md -> v9.0)
Session arc: (1) Decided (with JV) to write Paper 1 NOW rather than after the clean retrain — twist needs
dirty models; screen protects claims; 1938 feeds Paper 2. (2) Built the full v5 pipeline + wrote
foresight_v5.tex: twist-framed, triage-screened, explicit retraction of the "foreseen" trio, Ebert/
Rentenmark strengthened, memoir certified date-clean; 201 macros / 89 verify checks / 25pp clean. Fixed
fig_rome_inject legend bug + two inherited overfull tables. (3) JV vetted the mundane yardstick facts:
sterling fixed, royal numbering rule implemented (8 items rebuilt — all now gate), battery 40->69->68
(name-lottery exclusion rule, channel swim out), two GPU runs (queued politely behind the WSL vLLM
server), final v3: 29 gated, flagships at the 100.0th percentile, Norway-king comparator in the prose.
OPEN: corpus check/retrain (Alps), JV sign-off on ~28 new facts, Ebert name-free rebuild, fake-name
presidents, DE/FR, venue formatting. CONTEXT.md rewritten to v9.0.


### 2026-07-10 (EBERT NAME-FREE REBUILD — done, folded into the paper; yardstick signed off)
JV: yardstick OK (= final sign-off on the v3 battery; review sheet + paper prose updated to "vetted") and
"finish the Ebert thing". Built src/ebert_namefree.py: stems byte-identical to emp_kaiser_1919, options =
OFFICES not men (elected president [truth] / Kaiser on his throne / general with dictatorial power /
council of workmen and soldiers), scored with the certified v41 engine (imported, no duplication).
RESULT (results/ebert_namefree{,_v2}.json):
- WITNESS BREAKS in both 1919 AND 1920 settings: the 1929 reader puts ~90% on "a council of workmen and
  soldiers" (genuinely true for early 1919 — the Tsar-style two-truths trap; moving to 1920 does NOT fix
  it) and the two-rule meter declines. => NO certified name-free gap exists; office-phrase questions lose
  their witness in this design. ABANDONED: the 1920-setting fix (tried, failed the same way).
- GUESSER SIDE (the caveat's actual target) is UNAMBIGUOUS and setting-independent: told the war is lost,
  the 1913 reader bets 90% Kaiser / 9-10% elected president. Direction of the flagship = NOT a name
  artifact (the office was a 9:1 underdog). Magnitude = partly name: office-miss ~2.4 nats vs name-miss
  ~10.1. Honest double statement now IN the paper (Sec 7): "the republic as a form of state was a
  nine-to-one underdog in the pre-war mind, and the republic as it actually arrived — with a saddler's
  son nobody had heard of at its head — was outside that mind entirely." Sec 9 limit rewritten from
  promise to result (reported as guesser-side robustness check, not a certified gap).
Rebuilt: 208 macros, 95/95 verify PASS, 26pp, zero warnings. Ebert item CLOSED.


### 2026-07-11 (DE/FR WORDING REPAIR + RE-CERTIFICATION — FRENCH CERTIFIES, GERMAN FAILS HONESTLY; item closed)
JV: stop deferring, finish DE/FR. Diagnosed certify_defr.json per item; found OUR bugs, not the meter's:
(1) DE young_turk GRAMMAR BUG — option "der Türkei" where stems need nominative "die Türkei";
(2) DE tsushima TWO-TRUTHS BUG — "in der Straße von Korea" is also true (Koreastraße = Tsushima Strait);
(3) FR panama ARTICLE BUG — "traverse Panama" vs "traverse le Nicaragua" grammar-penalised the truth;
(4) FR king_george bare-name prior — fixed with the ROYAL NUMBERING RULE options (George V / Édouard VIII /
Albert Ier / Guillaume V); plus canonicality repairs (bare surnames Blériot, Selbstversenkung/sabordage,
Compiègne stems, Generalsekretär stems, Mai-1916 Skagerrak, shortened NP truths).
PROTOCOL (declared in the script header BEFORE running): ONE repair pass, wordings only, bars untouched,
report either way; a pass = "repaired-run" certification, one grade below the EN held-out standard.
Built src/certify_defr_v2.py -> results/certify_defr_v2.json (first run kept in certify_defr.json).
RESULT: **FRENCH CERTIFIED (repaired run)** — C1 17/20, C2 14/17 (82%), C3 15/17 (88%); the numbering-rule
king_george now 34%->98% grad +1.06 (the rule works exactly as in the yardstick). **GERMAN STILL FAILS,
now purely on coverage** — C1 11/20 (even one fewer than v1: tsushima gained coverage+correct, jutland now
answers "Skagerrak" correctly but went split-half unstable), C2 11/11 (100%!), C3 10/11 (91%). Precise when
it speaks, too silent; per protocol we STOP (no second repair pass) and report the failure. HONEST MISS:
my FR panama "fix" backfired (truth p 22%->3%; witness genuinely prefers Nicaragua) — counted as the wrong
answer it is, disclosed in App D. PAPER updated: Sec 8 rewritten (two-run story + repaired-run grade), Sec 9
limit 5 rewritten (certification graded by language), App D rewritten (both runs, all repairs, the panama
lesson). make_numbers/verify repointed (+first-run macros; verify re-derives the FR-pass/DE-fail verdicts).
Rebuilt: 214 macros, 101/101 PASS, 26pp, zero warnings. DE/FR wording item CLOSED — remaining German path
is coverage (denser German print / retrain), not wording.


### 2026-07-11 (SCORING CONFIG CHANGE: batched + fp32 is now the standard default — gated, diagnosed, adopted)
JV: "make batching standard default". Implemented common.slp_pairs/slp_batch (right-padded, masked,
same tokenization+indexing as slp). PRE-DECLARED GATE (test_slp_batch.py, 640 real battery scores,
Ranke-4B-1913): bf16 batched vs bf16 sequential FAILED — max 0.23 nats / 0.038 prob drift (0 modal
flips, x4.6 speed). DIAGNOSIS (test_slp_batch2.py): batch=1 == sequential to 0.000000 (code exact);
fp32 batch-invariant to 3e-5; bf16-seq vs fp32-seq itself differs up to 0.10 nats => the drift is bf16
KERNEL-ACCUMULATION NOISE, present in all historical bf16 numbers too — bf16-sequential was never "the
truth", just one draw of the noise. ADOPTED: **fp32 + batched** as standard (common.load default dtype
= float32 since today; needs ~17GB — fine on the 4090; dtype=common.DT for bf16 on small cards).
fp32-batched is x4.3-4.5 FASTER than the old bf16-sequential AND exact/replicable (re-gated under fp32:
max diff 3e-5, prob diff 7e-6, 0 flips, PASS). Converted the live pipeline loops: v41_battery (rules
untouched), control_battery, certify_defr_v2, ebert_namefree, date_spike_base.spike (the retrain
acceptance gate — future acceptance runs are now exactly replicable). DISCLOSURE headers added to each:
results saved before 2026-07-11 were bf16-sequential; reruns will differ by bf16's own noise (~0.04
nats), no verdict flips measured. Published v5 numbers unaffected (built from saved JSONs).


### 2026-07-12 (ASSOCIATE-EMERGENCE PROBE — JV's lens observation piloted, disciplined, designed)
JV's WhatsApp screenshot (lens UI, Ranke L33 @ "Hitler"): top-10 shows Stalin/Nazis/Eisenhower/Soviets
but ALSO Barack/Weiner/Friedman/"triggered". Two rival mechanisms: learned association (leak) vs
UNTRAINED-TOKEN ARTIFACT (modern tokenizer; never-seen tokens keep degenerate vectors that pollute
lens top-k). PILOT (src/assoc_lens_pilot.py -> assoc_lens_pilot.json; dev, declared post hoc):
differential lens with matched carriers (Hitler / Bismarck / Mueller / invented Brennt), L33 ranks:
**Stalin 14,052@Hitler vs 35-41k@controls, >50k@invented — Hitler-SPECIFIC. Nazis 15,696 vs >50k
everywhere — strongly specific (possibly period-legit). Eisenhower ~2x lift (18k vs 35k) marginal.
Barack: NO specificity (best at Mueller!) — the screenshot's flashiest token is an eyeball artifact;
"triggered" >50k everywhere.** Lesson: unit of evidence = rank-DIFFERENTIAL vs controls + fiction
floor, never top-k membership. Raw embedding norms do NOT separate trained/untrained vocab (post-T
tokens at HIGH norm pctiles — never-seen-token geometry) -> archaeology needs centroid/cosine, not norms.
DESIGN (notes/2026-07-12-associate-emergence-probe.md): ex ante tells Class A (existence-anachronistic:
OED first-attestation/Wikidata notability/Google-Books-preT below eps — artifact-prone, gated by Tier-0
trainedness) + Class B (link-anachronistic: pre-T PMI~0, modern PMI high — artifact-immune, did the work
in the pilot: Hitler-Stalin) + innocent sets as within-target calibration + specificity requirement +
fiction-floor/binomial multiplicity control. SEARCH: Tier 0 vocabulary-archaeology census (prompt-free)
-> Tier 1 stratified Wikidata/OED target grid + date-spike-leak anchors -> Tier 2 confirmation (causal
tracing, Talkie-1930 dark-contrast, ROME sensitivity control, corpus grep gold). Hitler + today's lists
= BURNED DEV SET; held-out battery to be frozen+hashed. VALUE: third independent instrument (no dates
needed, no probe training), sub-behavioural detection floor, Tier 0 as near-free retrain acceptance
test. Next: Tier-0 implementation; freeze Tier-1 grid; run w/ Talkie contrast.


### 2026-07-12 (associate-emergence: 1913 cross-model test — PREDICTION CONFIRMED; genocide rule fixed)
JV corrections/asks: (1) "genocide" is a bad Class-A tell (word coined 1944 but the REFERENT — the
Armenian massacres — is pre-1929 knowledge): rule refined to DOUBLY-anachronistic tells only (word AND
referent postdate cutoff; radar yes, genocide no). (2) Clarified Pass-1 vs ROME in plain terms: Pass 1
= passive reading of what training filed (evidence about contamination); ROME = deliberate forgery to
prove the detector works (calibration, not evidence). (3) RAN THE 1913 TEST (assoc_lens_pilot.py 1913
-> assoc_lens_pilot_1913.json), prediction REGISTERED FIRST: 1913-Hitler & 1913-Stalin were print
nobodies -> clean model must show no association. RESULT: CONFIRMED — Stalin@Hitler 38,030 vs 42-46k
controls (flat; vs 14,052 strongly specific in 1929); Nazis 47k whisper (vs 15.7k in 1929); Munich
COLDER at Hitler than at Mueller in 1913 (surname, not the putsch man) but warmer in 1929. The
association TRACKS THE CORPUS = the probe passes the same two-model validation (cutoff-shift logic)
that certified the meter. HONEST RESIDUALS: (a) 1929 Hitler-Stalin link still needs the Class-B period
co-occurrence check (both famous by 1929; late-20s co-mention plausible) before "leak" is claimed;
(b) FLAGGED WHISPER: Eisenhower shows mild Hitler-specific lift IN THE 1913 MODEL (23,951 vs 39k+/>50k)
— no innocent 1913 reading exists; deep-rank single-carrier so NOT claimed, but coheres with the
date-spike's faint WW2 traces in 1913 (WW2-end 3.1x, Pearl 1.5x); goes into the held-out battery.
(c) Blank-card artifact profile (Barack 95.6/95.1 pctile, Weiner, triggered) near-identical across BOTH
models = tokenizer/never-seen geometry, corpus-independent — confirms artifact classification. Design
note updated implicitly; held-out battery next (freeze grid + tells, floors, multiple phrasings).


### 2026-07-12 (JV: "isn't the stethoscope just log-probing?" — YES; design simplified)
Pulled the full layer grid from the saved pilot: Stalin@Hitler (1929 model) emerges L30 (41.9k) ->
L33 14.1k -> **L36 (the mouth) 1,658** vs 12.5k @Bismarck / 13.3k @Mueller / 29.8k @invented — the
association is STRONGEST at the final layer, not suppressed. So (1) the probe IS ordinary log-probing;
the real novelty of JV's observation is the PROMPT SHAPE: free association ("the name on everyone's
lips was X") instead of a question — a question forces the leak to out-compete period priors (why
"leader of Germany in 1940" stayed dark); free association removes the competition and makes shallow
traces visible in plain output probabilities. (2) DESIGN SIMPLIFICATION: Tier-1 battery runs on the
existing certified slp machinery (batched, fp32) — carriers x frozen tells x control names x fiction
floor; no lens infrastructure on the critical path. (3) Layer profile demoted to secondary diagnostic:
early emergence (Nazis, L21-23 ~5k) = lexical/word-neighbour link (possibly period-legit); late
emergence (Stalin, L30+) = semantic association (the interesting kind); mouth-vs-inside disagreement
would flag suppression cases if they exist. My earlier "knowledge below the level of speech" framing
was WRONG for this case — corrected with data.


### 2026-07-12 (association probe: INJECT CERTIFICATE PASSED + Tier-0 census run + Tier-1 grid drafted)
1. SENSITIVITY CERTIFICATE (src/assoc_inject_1913.py -> assoc_inject_1913.json; criteria pre-registered
in header). Planted "The great rival of Hitler was -> Stalin" into the provably-cold 1913 model (ROME
rank-1 @L9, in-memory only). RESULTS: S1 PASS — on HELD-OUT carriers Stalin@Hitler rank 11,357->106
(carrier A) and 15,447->1,277 (carrier B, different phrasing): the free-association reading detects a
genuinely stored association and generalises across phrasings. S2 PASS — Stalin@Bismarck/@Mueller
unmoved (~15k->~15k): key-specific. S4 PASS — period facts (Portsmouth .31, Frankfurt .75) and
Bismarck-rival speech untouched. S3 QUALIFIED — within the Hitler key, embedding-NEIGHBOUR tells also
warmed (Eisenhower 3,811->334, Nazis 13,979->1,569): mechanistically expected (planted direction
overlaps similar name vectors). DESIGN LESSON -> similar-vector tells = ONE signal family; battery
stats must count CLUSTERS, not tokens.
2. TIER-0 CENSUS (src/vocab_census.py -> vocab_census_{1913,1929}.json). Method VALIDATED at extremes:
emoji never-seen cluster razor-tight (median cos .9985); period words 0-20th pctile (clearly trained);
far-modern controls clean (smartphone ~never-seen in both). INTERNAL VALIDATION: cross-cutoff deltas
move the historically right way — Bolshevik (.68->.51), Soviet (.51->.40), fascist (.86->.61), jazz
(.59->.47), insulin (.82->.76) all MORE trained in 1929 than 1913, exactly as legit 1913-29 vocabulary
should. FLAGS: (a) "email" reads TRAINED in both — trilingual polysemy trap (German Email / French
email = ENAMEL, period-legit) -> NEW RULE: every tell needs a DE/FR sense check (corpus is EN+DE+FR);
(b) "robot" reads trained-ish in BOTH incl. 1913 (word coined 1920) — genuine corpus-check flag;
(c) middle zone (radar/laser/NATO ~40-50th pctile) not yet interpretable — needs frequency-calibrated
reference; (d) many key tells are MULTI-TOKEN (Gestapo, gulag, Leningrad, Stalingrad, Politburo,
penicillin) -> battery must use the slp sum rule, not single-token filtering.
3. TIER-1 GRID DRAFTED (notes/2026-07-12-tier1-target-grid-draft.md): era x category targets for both
cutoffs, tells/innocents/controls per target, positive anchors (date-spike leaks), negative anchors
(smartphone-era + Talkie-1930 must read darker), sensitivity anchor (tonight's inject), 5 binding rules
incl. the new trilingual guard + cluster counting; freeze protocol (JV vet -> mechanical Google-Books
EN/DE/FR pass -> hash -> one run). Hitler rows dev-burned.


### 2026-07-12 (inject-logic confusion resolved + research narrative PDF written)
JV misread the inject table ("high rank after inject = contamination? Bismarck unchanged = confirmed?")
— clarified the column logic: BEFORE column = the house reading (silence = clean evidence); AFTER
column = the controlled match (WE lit the fire; the screaming alarm certifies the DETECTOR, says
nothing about contamination); Bismarck row = the match didn't torch the kitchen (edit specificity, so
the calibration is fair). Also flagged the rank-direction trap (low number = warm = loud). The genuine
contamination signal remains the UNTOUCHED 1929 model's Stalin@Hitler = 1,658.
DELIVERABLE (JV request): full research narrative, dual-register (plain main text + technical boxes),
from the WhatsApp screenshot to the preregistration-ready battery:
reports/assoc_narrative/assoc_probe_narrative.{tex,pdf} (v1.0, 5pp, pdflatex x2, zero warnings).
Structure: screenshot -> two rival mechanisms -> controlled comparison (Barack unmasked; Stalin/Nazis
real) -> 1913 test w/ registered prediction -> "just log-probing" simplification (question hides,
free-association reveals; L36 Stalin 1,658) -> the match test w/ the column logic spelled out +
cluster-counting lesson -> census (email/enamel trap, robot suspect, cross-cutoff self-validation) ->
where-we-landed summary + honest open items (robot + Eisenhower-whisper await corpus grep; census
middle zone needs frequency calibration; 1929 Hitler-Stalin still owes the period co-occurrence check).
Every number traced to a named result file in the header.


### 2026-07-12 (narrative v1.1 — "what is inserted / why not rank 1" passage added)
JV's follow-up confusion on the inject table ("insert Stalin? then rank 1, no? or Hitler? but it is in
the question?") answered and folded into the narrative PDF (Step 5): the edit inserts a LINK into the
wiring (Hitler = key/doorbell, Stalin = content); on the PLANTED sentence Stalin IS rank 1 post-edit
(spoken check, ~50%) but that only proves the optimizer; the table reports HELD-OUT carriers that ask
for nothing, where Stalin floats to 106/150k — generalisation to fresh contexts is the certification
(real contamination also enters via other sentences than the test ones). Hitler-in-the-prompt is the
mechanism, not a flaw; Stalin never appears in any prompt. Recompiled clean (v1.1).


### 2026-07-12 (grid vetting pass 1 — JV: drop Lenin + U-boat, add Trotsky + poison gas -> grid v2)
JV reviewed the Tier-1 grid draft. DROPPED: Lenin (too much legitimate pre-1913 print presence;
Bolshevik axis already covered by date-spike anchors) and U-boat (INVALID, not just risky — word and
thing both pre-1913, U-1 launched 1906). ADDED: Trotsky (target-side pre-1913 presence acceptable, as
with 1929-Hitler; tells = sharply post-cutoff Class-B links ->Stalin rivalry / ->Red Army / ->exile,
Mexico) and poison gas (sense-shift target; weapon sense 1915+; NICE TIE-IN: the paper's own
circulation arm measured poison gas as the one future absent from pre-1913 print, so gas->war
associates in the 1913 model are anachronistic by our own published yardstick; tells ->Ypres,
->trenches, ->Flanders, ->masks; chlorine excluded as period-legit chemistry).
Written as notes/2026-07-12-tier1-target-grid-v2.md (v1 kept, never-overwrite rule). Remaining before
freeze: JV ruling on the rest of the rows, then the mechanical EN/DE/FR Google-Books pass + hash.


### 2026-07-12 (SESSION SUMMARY — Thread C born, validated, certified, grid approved; CONTEXT -> v9.3)
Arc of 07-11..07-12: (1) DE/FR closed (French certified on the repaired run, German honest fail).
(2) Batching made standard default the hard way: bf16 gate FAILED -> diagnosis -> fp32+batched adopted
(exact, x4.3 faster, 101/101 verify). (3) THREAD C: JV's screenshot -> controlled-comparison pilot
(Stalin/Nazis real, Barack/triggered = blank-card artifact) -> 1913 cross-model test (prediction
registered, confirmed; Eisenhower whisper flagged) -> "just log-probing" simplification (free
association vs questions; L36 Stalin 1,658) -> ROME inject certificate PASSED (11,357->106 held-out,
key-specific) -> Tier-0 census (email/enamel trap -> trilingual rule; "robot" suspect; cross-cutoff
self-validation) -> narrative PDF v1.1 for humans -> grid v1 -> JV vetting (Lenin+U-boat out,
Trotsky+poison-gas in) -> grid v2 -> **JV approved ALL rows ("yes all goood")**.
NEXT SESSION OPENS WITH (the approved block, in order): author complete per-target tell/innocent lists
for grid v2 -> mechanical Google Books EN/DE/FR pre-cutoff verification (reuse ngram_exposure.py fetch
w/ backoff; gentle-scraping rules) + tokenizer forms via summed-slp rule -> JV final vet of tells ->
hash lists -> ONE run: Ranke-1913 + Ranke-1929 + Talkie-1930 dark-control (Talkie = RunPod ~$0.4, ask
before spending) -> cluster-count verdicts vs fiction floor -> report either way. Also pending: corpus
grep flags now include "robot" and the 1913 Eisenhower whisper.


### 2026-07-12 (CONTAMINATION PIPELINE built + inserted-memory validation PASSED; Talkie dark-control launched)
Goal (JV): fully validated pipeline using activations + inserted memories to identify contaminated
memories in time-locked models. BUILT: battery_assoc_v1.json (v1.1; grid-v2 targets, cluster-grouped
tells, 5 invented persons as floor) + contam_pipeline.py (lift metric = slp(carrier(T), tell) - control
-name baseline, averaged over 5 carriers; floor95 from invented targets; verdict = clusters fired:
>=2 FLAGGED / 1 WHISPER / 0 DARK; importable functions).
BASELINE RUNS (fp32+batched): 1913 floor +2.50 — Trotsky FLAGGED (Stalin+4.05, Red Army+4.47,
exile+2.79; innocents also fire = knows the man + anachronistic links = textbook signature; JV's pick
paid off), Hitler(dev) FLAGGED, fascist/insulin/poison-gas/Kemal/tank WHISPER, Ebert/Ludendorff/
Eisenhower DARK. 1929 floor +2.62 — radar FLAGGED (waves+6.3, station+6.2, aircraft+3.7 — new
candidate), Hitler(dev) WHISPER (Stalin+3.00, matches certified exhibit), Eisenhower/Franco/Obama
WHISPER, Truman/Mao/Kennedy/Gorbachev/internet DARK.
NEGATIVE-ANCHOR LESSONS (smartphone): v1 fired via subword overlap (telephone~smart-PHONE) AND
untrained-x-untrained blank-card affinity (internet) -> v1.1 rule: tells must be census-trained, no
shared lexical root. v1.1: smartphone STILL whispers in 1913 (wireless/machine) — deeper cause: the
compound's TRAINED subwords ('smart','phone') carry period meaning; term-targets need matched
invented-compound controls (v2 design item). SCOPE: pipeline v1 validated for PERSON targets;
term-target verdicts labelled provisional.
INSERTED-MEMORY VALIDATION: v1 single-sentence plant FAILED honestly (Ebert president lift -2.06 ->
+2.42, 0.09 under floor; mechanism radiates, plant too weak) -> DECLARED one protocol revision
(3-sentence plant = multi-document realism; criteria untouched). v2: **PASS BOTH MODELS** — Ebert
DARK->WHISPER (president +7.42, office cluster), Truman DARK->WHISPER (+6.71); V2 specificity PASS
(zero other-verdict changes). Files: contam_pipeline_{1913,1929}{,_inject,_inject2}.json.
TALKIE DARK CONTROL: launched on RunPod A100 80GB ($1.19/hr, pod p2xrp0zb7k83qq, 1,680 pairs, gist
fe642a...) with the TRIPLE LOCK: (1) pod-side sleep-4800 self-kill + trap-EXIT push+terminate,
(2) monitor terminates on completion, (3) monitor force-kill at 90-min deadline. Monitor running.
Success criterion: Talkie verdicts must read darker than Ranke-1929 (esp. Hitler/Eisenhower/radar).


### 2026-07-12 (TALKIE DARK CONTROL BACK; pipeline v1 CLOSED with scorecard; pod verified terminated)
Talkie-1930 leg completed (~30 min A100, ~$0.6); pod self-terminated on completion (lock 2) and the
account was API-VERIFIED at zero pods. RESULTS (contam_pipeline_talkie1930.json, floor +2.68):
DARKER than Ranke-1929 on Eisenhower (DARK vs WHISPER), Obama (DARK vs WHISPER), Franco (1 vs 3 tells);
EQUAL on the Hitler-Stalin whisper (+3.08 vs +3.00 — in BOTH corpora: plausibly period-legitimate
late-20s dictator co-mention; the period co-occurrence check now decides, NOT assumed contamination);
BRIGHTER on Kennedy->Dallas (+4.29) and Mao->China (+3.65) — NEW CAVEAT FOUND BY THE CONTROL: the
fiction floor is built from invented names, but real names co-occur with real PLACES innocently ->
place-tells behind real-name targets can clear an invented-name floor; v2 refinement = add
real-but-unlinkable period names (Gladstone->Dallas) to the floor population. radar FLAGGED in BOTH
models (Talkie even stronger) = shared-leak candidate (Talkie's one known date-leak is Fall-of-France
1940; radar fits 1940) or shared artifact -> corpus grep. smartphone whispers in Talkie too ->
compound-subword channel confirmed mechanical, term-target scoping confirmed.
FINAL SCORECARD (reports/contam_pipeline_v1.html, from saved JSONs only): inserted-memory sensitivity
PASS x2, specificity PASS x2, floors work, negative anchor productive (2 artifact rules), dark control
PARTIAL with two named v2 refinements. THE GOAL ("fully validated pipeline using activations and
inserted memories to identify contaminated memories") is delivered for PERSON-TARGETS with bounded,
honest claims; term-targets provisional. v2 queue: real-name floor population, matched invented-
compound controls for terms, period co-occurrence checks (Hitler-Stalin), corpus-grep flags now:
robot, radar, 1913-Eisenhower whisper, Trotsky-1913 cluster.


### 2026-07-13 (JV audit of the innocents — three new rules; battery v1.2; verdicts recomputed, none flip)
JV challenged the innocents ("Trotsky and revolution is innocent??? fumes and poison gas?"). CONCEDED,
and the audit found a third error he didn't name. NEW RULES: (R6) sense-AMBIGUOUS associates excluded
from every role — Trotsky-"revolution" cannot distinguish legit 1905 knowledge from leaked 1917 (an
"innocent" label would launder 1917 signal); (R7) innocents must be FACTUAL period associates, never
lexical near-synonyms of the target — gas->"fumes" is tautology and calibrates nothing; (R8, the one
his audit exposed) tells must be impossible for THIS target's own pre-cutoff BIOGRAPHY, not merely
word-anachronistic — "exile" as Trotsky tell was INVALID (exiled twice pre-1913, escaped twice) ->
reclassified innocent. Plus: Kemal marked provisional_name_culture — "Ankara"/"sultan" lifts behind a
Turkish name vs a EUROPEAN-name fiction floor may be name-culture semantics, not knowledge -> v2 floor
must include culture-matched invented names.
Battery -> v1.2 (roles only; lifts untouched); verdicts recomputed from saved JSONs (reclass_v12.py ->
contam_pipeline_1913_v12.json): NO verdict flips — Trotsky stays FLAGGED but now on the two clean
clusters (Stalin/rivals + Red Army/military), with exile+Siberia as fired innocents (model knows the
man); poison gas WHISPER unchanged (masks; the dropped fumes/mines were never load-bearing). HTML
report refresh pending (next session; v1.2 verdict file is the source of record for 1913).


### 2026-07-13 (report refreshed to v1.2 — no deferral)
contam_pipeline_report.py updated: 1913 section now reads contam_pipeline_1913_v12.json (source of
record), verdict tables gained the excluded-ambiguous column, and the R6-R8 innocents-audit section is
in the report body. Regenerated reports/contam_pipeline_v1.html and opened for JV. Nothing pending
from the audit.


### 2026-07-13 (JV skepticism -> verdict-rule CALIBRATION AUDIT; whisper FPR measured; term-target class quarantined)
JV: "criteria may not find the right thing — poison gas/tank/fascist look suspicious; N or level too
low?" STRUCTURAL CONFIRMATION: single-cluster targets (fascist, poison gas) can NEVER flag under the
>=2-cluster rule — strength was ignored. Patched pipeline to save full lift matrices (_full runs);
Talkie raw slps refetched from the gist. AUDIT (calib_verdict_rule.py -> calib_verdict_rule.json):
(1) EMPIRICAL FPR on pseudo-targets (5 invented names x 11 tell-sets x 3 models): WHISPER ~15% (8/55
per model) — whispers are LEADS not evidence; FLAGGED 0.6% (1/165) — clean. JV right on the level, for
the whisper tier. (2) STRENGTH RULE (S=top-2 mean vs pseudo-target null): promotes fascist (S 4.15,
3/3 tells), Kemal (4.49), Franco (3.02) — JV's suspects ARE strong — BUT also promotes smartphone (the
KNOWN mechanical artifact) to FLAGGED in 1913 AND Talkie: strength measures association, and the
compound-subword channel produces strong association; the cluster rule had been accidentally shielding
us. (3) DECISIVE PATTERN: every rule-DEPENDENT verdict is a term-target (or name-culture/polysemy
provisional); on certified person-targets the two rules AGREE in all three models (Trotsky FLAGGED,
Hitler FLAGGED/WHISPER, Ebert/Ludendorff/Truman/Kennedy/Gorbachev DARK) — person-target conclusions
are RULE-ROBUST. DECISIONS: (a) whisper tier relabelled "lead, not evidence" (15% FPR quoted wherever
whispers appear); (b) term-target verdicts stay provisional under EITHER rule until matched
invented-compound controls exist (v2: a "steamgas" floor for "poison gas"); (c) no threshold change —
relaxing N/level would flag the artifact with the same stroke; (d) poison gas / tank / fascist join
robot, radar, 1913-Eisenhower, Trotsky-cluster as NAMED corpus-grep leads. Files:
contam_pipeline_{1913,1929}_full.json, calib_verdict_rule.json.


### 2026-07-13 (STAFF-READY EXPLANATORY MATERIAL — narrative v2.0 + HTML companion refreshed)
JV: update all explanatory material for staff — clear, error-exposing, replicable, no tech speak but
precise. DONE: (1) reports/assoc_narrative/assoc_probe_narrative_v2.pdf (v2.0, 8pp, zero warnings) —
extended with Step 7 (the pipeline, Trotsky worked end-to-end: roster/tells/innocents/controls, lift
+4.05 explained as "57x likelier", fiction floor, cluster verdicts), Step 8 (inserted-memory proof
incl. the failed first attempt and the declared revision; Talkie dark control incl. the Kennedy-Dallas
floor lesson and pod hygiene), Step 9 (JV's two audits as method: biography rules R6-R8; calibration
audit w/ 15%-whisper / 0.6%-flag FPRs and the fascist-vs-smartphone tension), a staff replication
section (exact commands, all roster-extension rules consolidated), and a rewritten where-we-landed +
open-work list. NOTE: original PDF locked by an open viewer -> v2.0 saved as *_v2.pdf. (2)
reports/contam_pipeline_v1.html — added the calibration-audit section (measured FPR table, the
alternative-rule test, the quarantine decision). Both documents: every number from named saved files.


### 2026-07-13 (JV one-term catch -> FLAG-TIER INSERTION TEST: FAILED TWICE, STOPPED; scope stated precisely)
JV: "why is Ebert clean with a lift of 6? because it's only one term?" EXACTLY RIGHT — one planted
fact = one cluster = WHISPER ceiling by construction; the insertion validation had certified the
WHISPER tier only; the FLAGGED tier (the trusted one, 0.6% FPR) had never been shown reachable by
inserted memories. FLAG TEST (pipeline_validate_flag.py, criteria pre-registered): plant TWO facts
into baseline-DARK Ebert (president/office + Weimar/weimar). ATTEMPT 1 (both @L9): FAIL — the second
edit overwrote the first (president +7.42 solo -> below floor; only Weimar fired +4.55). DECLARED one
mechanistically-motivated revision: two layers (president @L9, Weimar @L12). ATTEMPT 2: FAIL — still
only Weimar fired (+3.07, itself weakened): the edits fight over the same token-position budget
regardless of layer. STOPPED per protocol. FINDING: one-position rank-1 surgery cannot superpose two
facts; real multi-document contamination stores distributed — the faithful flag-tier certificate is a
DOCUMENT-LEVEL PLANT (fine-tune a copy on synthetic period-style articles; v2 queue, feasible on the
4090). F2 specificity held in both attempts (zero collateral). STATUS STATED PRECISELY in both staff
documents (narrative Step 9 "one-term audit" + HTML scorecard row): flag tier = measured specificity
(0.6% FPR) + independently-corroborated natural positives (Hitler/Trotsky vs date-spike); insertion
sensitivity NOT yet demonstrated. Files: contam_pipeline_1913_flagtest.json, _flagtest2.json. PDF
naming note: viewers held old copies locked; current build = assoc_probe_narrative_v2_1.pdf
(consolidate names once viewers close).


### 2026-07-13 (JV: "proving a negative — anything clean enough for Paper 1?" — YES; bounded-absence paragraph added)
DECISIONS: (1) JV kills the fine-tune plant idea ("we are not retraining the model") — the flag-tier
insertion certificate stays open, stated as such. (2) The Paper-1 question answered: the unforeseen
claims are SAFE, and this week's pipeline work STRENGTHENS them, via the physics-style upper-limit
structure: (i) error direction — leakage can only fake foresight, so residual contamination makes the
gaps UNDERSTATEMENTS; cleanliness was only ever needed for the (already retracted) foreseen claims;
(ii) capability control — witness 78% + yardstick prove the guesser's blank is specific, not general;
(iii) NEW: bounded absence — the association probe is insertion-calibrated (planted 3-sentence
Ebert->president reads +7.42 vs floor) and reads the real Ebert at -2.06 (president), warmest word
-1.09, ALL below floor, while the same probe flags genuinely stored traces (Hitler/Trotsky) in the
same model. "Nothing, down to a detectable three-sentence trace" — a calibrated upper limit, not an
unprovable zero. PAPER EDIT: bounded-absence paragraph added to Sec 7 after the name-free passage;
4 new macros (ebertAssocPres -2.06 / ebertAssocMax -1.09 / assocFloor +2.46 / ebertAssocPlanted
+7.42) pulled from contam_pipeline_1913_full.json + _inject2.json. Rebuilt: 218 macros, 101/101
verify PASS, 26pp, zero warnings. Memoir claims unaffected (date-clean guesser). Paper 1 status:
strengthened, no claim weakened by the week.


### 2026-07-13 (HONEST REASSESSMENT of the contamination pipeline, at JV request -> FREEZE decision)
Verdict: good instrument, main value already extracted; freeze, do not develop. WORTH IT: (1) the
bounded-absence paragraph in Paper 1 Sec 7 (insertion-calibrated null on Ebert) — the pipeline's most
valuable product is a NULL reading; (2) Trotsky-1913 (one genuinely new, dateless finding —
corroborative of the known leak headline, not conclusion-changing); (3) the free-association insight
(reusable; explains why behavioural tests understate contamination). WORTH LESS THAN THE EFFORT:
verdict tiers narrow in practice (WHISPER 15% FPR near-useless; FLAGGED spec-clean but sensitivity
never insertion-proven; term-targets quarantined); the leads list is worthless until the corpus grep
runs; Hitler-Stalin and radar unresolved without it. COST: ~3 days + trivial cloud; opportunity cost:
grep + venue pass sat still. DECISION: pipeline FROZEN as-is (no v2 controls / term fix / culture
floors — revive only if the method preprint materializes). NEXT EFFORT: (a) corpus grep on Alps —
settles the entire shopping list in one pass; (b) Paper 1 venue pass. Everything banked, documented,
replicable.


### 2026-07-13 (JV novelty check vs apoorvumang.github.io/knowledge-cutoff — "nothing new?" mostly right, already conceded)
Fetched/read the page: live benchmark, 12 commercial LLMs, 330 unforecastable events 2024-26, QA+MCQ,
"effective cutoff = last month above 50%", fabricated-event controls. ASSESSMENT: confirms the
detection technique (effective-cutoff probing + fabrication controls) is COMMODITY — exactly the
territory Paper 1 Sec 10 already concedes w/ Cheng/Shi/Golchin cites. Still ours, honestly sized:
(1) the mirror-image object (future-leakage into time-locked instruments vs staleness of current
models), (2) insertion-CALIBRATED detection floors (their fabrications = sanity check, ours =
sensitivity bound -> the Ebert paragraph), (3) contamination-aware SALVAGE (the actual methods claim),
(4) the application (foresight measurement). Consequence: freeze decision reinforced; the method
preprint idea further deflated; Paper 1 untouched in substance. PROTECTIVE EDIT: footnote citing the
live benchmark added to Sec 10 next to cheng2024 so no referee discovers it for us; rebuilt clean.
GREP CONFIRMED COMING (JV) — the leads list awaits it.
