# Foresight project: the split, and the work list

v1 — July 2026. Companion to foresight_v4.pdf.

---

## A. The split

**Paper 1 — the instrument.** Target: Science / Nature Human Behaviour. Working title: *Measuring what the past could see coming.* This paper's claim is the method: you can read vanished states of knowledge out of frozen readers, but only if you ask the right way, and here is the proof. It ships as soon as the fixes below are in. It does not wait for the 1939 model.

What goes in:

- The problem: every gauge of foreseeability built after the fact knows the ending; here is one that doesn't.
- The taxonomy of failures, promoted from appendix to centerpiece. Surprise-at-a-sentence measures what an age printed, not what it expected (moon vs. bread). Future-tense prompts summon the genre of prophecy, not knowledge (the 1929 reader betting on the Kaiser it knows fled). This is the intellectual meat and it is what a general-science audience will actually learn.
- The repair: memoir framing; the two-rule scoring; the meter that declines when it cannot score fairly.
- The exam: pre-registered marks, development set burned, fresh set passed. Certification as the way one earns trust for an instrument like this.
- The two-instrument architecture, stated as a principle: free generation measures whether an outcome was *in the picture* (thinkability); betting slips measure how the picture was *weighted*. Multiple choice can never measure thinkability, because the researcher writes the option into the reader's head.
- Showcase findings, presented as demonstrations of the instrument, not as an empirical campaign: Ebert (the republic outside the pre-war picture, +9.8 nats, replicated in three languages), the Rentenmark (a genuinely new thing), and the French–German split on the Depression (each language forecasts from its own national memory — the single best general-interest result in the project).
- Validation battery: witness-confidence gate, paraphrase bootstrap with error bars on every gap, the modern-reader audit of the forced-choice arm, German re-certification or its honest demotion.
- One paragraph, no more, pointing at applications beyond economics: history of science, the 1918 pandemic, any field that argues about what contemporaries could have known. This is the sentence that makes it a Science paper rather than an economics paper in disguise.

What stays out: Barro, Rietz, Gabaix, Wachter, peso problems, diagnostic expectations, the event universe, the frequency-vs-availability decomposition. Paper 1 may gesture at economics in one closing sentence. Every economics *claim* belongs to Paper 2.

**Paper 2 — the disasters.** Target: QJE. Working title: *Rare disasters, ex ante.* This paper's claim is empirical: the support assumption behind rare-disaster asset pricing mostly holds, the weighting assumption fails in a patterned and measurable way, and the pattern is the availability heuristic operating at the level of a whole society's written record — or, if the frequency control says otherwise, the pattern is print frequency, and the paper says that instead. It waits for the 1939 witness. Do not fight this; the Depression and Hitler are the two disasters the entire calibration literature rests on, and a version of the paper in which they are measured rather than narrated is worth a year.

What goes in:

- The pre-registered event universe: Barro–Ursúa disasters 1900–1933 for the countries the corpora cover, plus a written rule for admitting political catastrophes, plus a written recipe that turns an event into questions. Every decline reported. This kills the "you picked the questions" rejection.
- The witness-confidence gate as an inclusion rule, applied without mercy.
- The control distribution: the same instrument run on mundane between-cutoff outcomes, so that "foreseen" means "gap indistinguishable from the gap on a peace-conference location" and not "gap that felt small."
- Support measured by generation (volunteer rates), weighting measured by betting. The Appendix C forced-choice arm moves here, subordinated to the generation arm, and only if the modern-reader audit clears it.
- The frequency-vs-availability decomposition: do the model's bets on precedents exceed what raw print share predicts? If yes, diagnostic expectations in the wild, a century before the surveys — the Bordalo–Gennaioli–Shleifer result. If no, the honest weaker claim about how the written record weighs the future. The paper commits in advance to reporting whichever comes out.
- The conditionality bridge, one careful section: the memoir measures the shape of disaster given disaster; the asset-pricing literature needs unconditional probability; here is the logical step that connects them, stated before a referee states it.
- The 1939 (and eventually 1946) witness turning Table 2 from a memoir into measurements: gaps for the Depression's depth, the bank failures, the lawful appointment.
- Section 7 of v4, rewritten around whatever the decomposition finds.

Division of credit between the papers: Paper 2 cites Paper 1 for the instrument and claims no methods novelty; Paper 1 cites Paper 2 as the application and claims no economics findings. That is a clean answer to any salami objection.

The decontamination preprint remains the third, separate leg, cited by both.

---

## B. The work list for Opus

Ordered by dependency. Each goal has a completion test; a goal without its test met is not done. Paths follow the existing results/ convention.

**/goal witness-gate** — Implement the inclusion rule p_witness(truth) ≥ 0.50 as a filter over all existing catastrophe questions. Regenerate Table 1 under the gate. Write results/gate_report.json listing every dropped question with both readers' full betting slips and the reason. Then autopsy the Tsar question specifically: generate ≥20 paraphrases of its endings, re-score, and determine whether the witness's 2% is ending-wording (fixable) or genuine model ignorance (question dies). Completion test: regenerated Table 1 contains no row with witness < 50%, and gate_report.json explains every casualty.

**/goal control-distribution** — Build a battery of ≥40 mundane between-cutoff outcomes: locations of treaties and expositions, election results, successions that went as expected, technologies adopted on schedule. Same memoir recipe, same two-rule scoring, same gate. Write results/control_gaps.json with the full gap distribution and its 10th/50th/90th percentiles. Re-express every catastrophe gap as a percentile of this distribution. Completion test: a figure exists placing each catastrophe on the mundane-gap distribution, and the words "foreseen" and "unforeseen" in the draft are redefined against it.

**/goal event-universe** — Extract the Barro–Ursúa disaster list for 1900–1933 restricted to countries with meaningful corpus coverage; draft a one-page written rule for admitting political catastrophes (regime falls, seizures of power, sovereign defaults) and apply it; draft the question-generation recipe (event → aftermath sentence → four endings, with the famous-truth rule from Section 3 as a hard constraint). Generate the full question set mechanically from the recipe. Freeze as prereg/event_universe_v1.json before any model sees any question. Completion test: the file exists, is dated, and contains every event, every question, and the recipe text; no question in it has been run.

**/goal generation-arm** — For every event in the universe, run free generation: K = 200 completions per prompt family under the pre-event reader, in the memoir register. Build a rubric (and a classifier applying it) that judges whether a completion volunteers the true outcome, with a 100-completion hand-labeled validation sample and agreement ≥ 0.9. Write results/volunteer_rates.json. Completion test: volunteer rates exist for every universe event, classifier agreement documented, and the draft's support claims cite generation, not betting.

**/goal frequency-control** — For each precedent the memoir reaches for (putsch, Fed, Goldmark, cholera, Wilhelm...) and each true outcome, measure print share in the training corpus: retrieval counts under a fixed query set per concept, normalized per million tokens, by five-year window so recency is separable from volume. Write results/print_share.json. Then regress betting-slip weights on print share and on recency-weighted print share; the question is whether the model's preference for the last crisis exceeds what raw frequency predicts. Completion test: the regression exists with both specifications, and a one-page memo states in plain words which of the two papers' Section-7 claims the data supports.

**/goal error-bars** — Bootstrap every reported gap over ≥20 ending paraphrases and ≥5 prompt paraphrases; report a confidence interval on each. Flag any gap whose interval overlaps the control distribution's median. Completion test: no table in either draft reports a gap without an interval.

**/goal german-fix** — Diagnose the German refusals: the hypothesis in v4 is answer wording. Rewrite German endings under the same recipe, re-run the certification exam under the frozen marks — the marks do not move. If it passes, German is certified; if it fails again, write the demotion paragraph and stop. Completion test: either results/certify_de_v2.json passes all three marks, or the draft says plainly that German is supporting evidence only, and the French–German Depression contrast in Paper 1 carries an explicit caveat.

**/goal modern-reader-audit** — The forced-choice arm has the modern reader at 33% on lawful-appointment-vs-putsch, an answer it certainly knows. Determine why: run the modern reader on 20 forced-choice questions with undisputed famous answers. If it scores near ceiling, the Hitler item is broken and gets fixed; if it scores near chance, the arm cannot read knowledge and is cut from Paper 2 entirely, whatever that costs Appendix C. Completion test: a memo with the ceiling-test result and the keep/fix/cut decision, and Table 5 either repaired or gone.

**/goal seed-ensemble** — If the Alps allocation permits: train a second seed of the 1913 model. Recompute the flagship gaps (Ebert, Rentenmark, Russian debts, war spread) cross-seed. Completion test: a table of gap agreement across seeds; if compute is denied, a written note in both drafts that inference is conditional on one training run, so the limitation is ours before it is a referee's.

**/goal split-restructure** — Create two manuscript skeletons, move every section, table, and appendix of v4 according to Part A above, and verify no empirical claim appears in both. The taxonomy of failures and Appendix E merge into Paper 1's main text; Section 7 moves whole into Paper 2 and is marked for rewrite pending /goal frequency-control. Completion test: two compilable drafts, a mapping file recording where every v4 paragraph went, and zero orphaned content.

**/goal witness-1939-spec** — Write the specification of what Paper 2 needs from the 1939 model before it trains: which Table 2 rows become measurable, which new questions the universe generates for 1933–39 events, what the certification exam for a 1929/1939 pair looks like. Completion test: a spec document sufficient for the training run to be scheduled with the right corpus cutoff audit, so the model that arrives is the model the paper needs.

Sequencing: witness-gate and modern-reader-audit first (they decide what survives); event-universe frozen before generation-arm or any new betting runs; frequency-control before anyone rewrites Section 7; german-fix and error-bars parallel to everything; split-restructure last among the paper tasks; witness-1939-spec immediately, because the training queue on Alps is the long pole.
