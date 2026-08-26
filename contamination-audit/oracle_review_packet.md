# Ranke Contamination Audit: Status-Quo Review Packet for GPT-5.5 Pro

Please act as a skeptical methodological reviewer. We are evaluating whether time-locked historical language models contain post-cutoff contamination, and whether the current folder provides credible evidence of "un-contamination" or only absence of detected contamination.

The working directory is `C:/Users/wmf043/Dropbox/ranke-contamination-audit`. Most of the numbered folders and root reports are Joachim's work. The `temp-guohui/` folder is Guohui's own related work and should be treated separately.

## Question for review

Given the evidence summarized below, what is the strongest defensible claim we can make right now about contamination in Ranke-4B-1929 and Talkie-1930-13B? Please identify:

1. The strongest evidence for cleanliness / non-contamination.
2. The strongest evidence for remaining contamination or shallow leakage.
3. The weakest methodological links and what a referee or coauthor would attack.
4. What should be done next before making a public or paper-level claim.
5. How to word the conclusion conservatively.

## Important warning about chronology

The project has evolved quickly. Some root files and older reports are stale:

- `README.md` still contains an older bottom line that says contamination is "real but localized to American politics."
- Later entries in `CONTEXT.md` and especially the latest entries in `LOG.md` reverse or soften that claim after stricter no-cue and matched-distractor tests.
- The current status quo is therefore not the old "US politics is contaminated" headline. The current strongest claim is closer to: "We detect shallow name/string/association traces, but the best current forced-choice tests do not show coherent post-cutoff fact knowledge above chance."

## What has been done

### 1. Initial advice and design review

Folder: `1_advice_and_panel_review/`

The initial methodological review emphasized that model probing is circumstantial and corpus inspection is more direct. The fused advice said:

- First, search the raw training text for post-cutoff years, modern event names, metadata, scan headers, later editions, introductions, and digital boilerplate.
- Then use model probes as a second line of evidence.
- Avoid modern labels that a pre-cutoff model could not know, such as "World War I" for a 1913 model; use period-appropriate facts.
- Pre-register/freeze test wording before seeing scores.
- Validate probes on pre-cutoff positive controls and fictional/null controls.

This design advice is still important because the current folder mostly probes models; it does not yet prove source-level cleanliness of the full 160B corpus.

### 2. Early model probes and metric corrections

Folders: `2_model_probes/`, `5_results/`

The early probes used teacher-forced minimal pairs and rank/log-prob comparisons. Several important mistakes were found and corrected:

- Average log-prob per token was length-biased: it inflated scores when the true answer had more tokens.
- Total log-prob has the opposite length problem when answer lengths differ.
- First-token scoring is also bad for names because it compares subword fragments, not full answers.
- The corrected approach is whole-answer teacher-forced scoring on length-matched alternatives, or forced choice with matched distractors.
- Prompts must not include first-name or other cues to the answer; otherwise the test becomes name completion rather than historical knowledge.

This led to repeated reversals. Earlier results suggested a US-politics leak; later strict no-cue and matched-distractor designs weakened or eliminated that conclusion.

### 3. Strict no-cue diagnostic

Key files mentioned in `LOG.md`: `diagnostic.html`, `questions.html`, `story.html`, `5_results/dig_validated_results_full.csv`, `5_results/pairs_results.csv`, `5_results/quad_results.csv`

The strict diagnostic applied one rule: the answer must require post-cutoff text, must not be the topic name, and must not be cued by the prompt.

Reported result from the latest `CONTEXT.md` / `LOG.md`:

- Pooled across three base Ranke models, every post-cutoff category was at chance.
- US politics fell to 56%, p = .62.
- Answer types were also at chance: names 44%, terms 49%, dates 45%.
- A paired A/B design showed shallow name exposure: post-1929 people scored 38% no-cue vs 55% with cues, a +18 percentage point cue effect.
- Quadruplet tests showed that marginal two-way hits often disappear when the true answer must beat all matched alternatives.

Interpretation: the model can sometimes complete famous names or strings when cued, but the strict test does not show coherent post-cutoff fact knowledge.

### 4. Association-lift and cluster probes

Key files/results: `5_results/assoc_results.csv`, `5_results/assoc_results2.csv`, `5_results/assoc_dig_results.csv`, `assoc_summary.html`, `assoc_dig.html`, `coverage.html`

The association-lift tests ask whether post-cutoff concepts become more likely when paired with related places/events than with matched controls. This detected some semantic traces:

- Positive associations appeared for some hyper-documented post-cutoff events such as Auschwitz, Vietnam, and Stalingrad.
- Many apparent positives decomposed into pre-cutoff place/name priors rather than event knowledge.
- Place-salience and old associations mattered: e.g. French Indochina, Warsaw, Pearl Harbor as a naval base, Crimean/Yalta-style priors.

Important correction: a cross-model argument that the 1900 model must be clean was retracted. The 1900 cutoff itself is under audit, so using it as an uncontaminated baseline is circular.

Current interpretation: association traces exist, but they are spotty and often lexical/place-based. They do not yet imply coherent post-cutoff knowledge.

### 5. Absolute forced-choice tests for Vietnam, WWII, and Cold War

Key files/results: `2_model_probes/absolute_probe.py`, `5_results/absolute_probe_results.csv`, `2_model_probes/cw_dig.py`, `5_results/cw_dig_results.csv`

The design changed from relative comparisons to absolute tests. For each post-cutoff topic, the model had to choose the true continuation against three matched distractors, so chance is 25%.

Reported results for Ranke:

- Vietnam-US war: 5/15 for base and 5/15 for instruct; p = .31, at chance.
- WWII: 4/15 for base and 5/15 for instruct; at chance.
- Cold War initially looked elevated because of NATO/Warsaw Pact.
- A follow-up Cold War dig showed those signals were lexical or pre-cutoff-prior artifacts: NATO as a word-like string, Warsaw Pact as a bigram plus Battle of Warsaw 1920 / Warsaw salience. The model failed core properties such as founding year, dissolution year, NATO opposition, and Berlin Wall facts.

Interpretation: the strongest absolute tests on these topics support cleanliness, with remaining hits explainable by pre-cutoff priors, lexical probability, or generic-place salience.

### 6. Integrated Ranke-vs-Talkie head-to-head suite

Key folder: `8_suite/`

This is the latest integrated comparison and probably the most important current artifact.

Files:

- `8_suite/batteries.py`: defines the frozen question bank.
- `8_suite/suite.py`: scores Ranke models through Hugging Face / transformers.
- `8_suite/talkie_probe.py`: scores Talkie with a custom all-position logits reimplementation.
- `8_suite/results_Ranke-4B-1929.json`
- `8_suite/results_Ranke-4B-1929-1202-Instruct.json`
- `8_suite/results_talkie-1930-13b-base.json`
- `8_suite/comparison.html`
- `8_suite/STATUS_OVERNIGHT.md`

Design:

- Same wall around 1930: Ranke-4B-1929 vs Talkie-1930-13B base.
- Same 61-item bank for both models.
- 45 post-cutoff forced-choice facts: 15 Vietnam-US war, 15 WWII, 15 Cold War.
- 16 pre-cutoff controls.
- Each post item has 3 matched distractors, so chance is 25%.
- This is intended to be fair across model sizes: a 4B and 13B model face the same chance line, unlike raw rank-ratio measures.

Latest results:

- Ranke-4B-1929 base: pre controls 16/16; post 14/45 = 31.1%, p = 0.216.
- Ranke-4B-1929-1202-Instruct: pre controls 16/16; post 16/45 = 35.6%, p = 0.075.
- Talkie-1930-13B base: pre controls 16/16; post 12/45 = 26.7%, p = 0.454.
- Topic-level Talkie: Vietnam 4/15, WWII 5/15, Cold War 3/15.
- Ranke-vs-Talkie post-cutoff difference: two-proportion p about .64.
- Talkie scoring validated: `validation_maxdiff = 0.0`, meaning the custom all-position logits path exactly matched Talkie's official last-position forward on validation.

Interpretation:

- Both models clearly know pre-cutoff controls.
- Both sit statistically at chance on the post-cutoff forced-choice bank.
- The earlier impression that Talkie was more contaminated from Guohui's raw rank-ratio disappears under matched distractors.
- The `LOG.md` says this suggests the raw rank-ratio was partly a model-size artifact: Talkie's 13B size gives it a richer generic-name floor.

Important caveat:

- `LOG.md` also says the 45-item post bank is still "Ranke-shaped" and should be broadened beyond Vietnam/WWII/Cold War before the final Talkie-vs-Ranke claim. It proposes adding neutral multi-domain post-1930 items in science/technology, medicine, space/computing, decolonization/institutions, culture, economics, and disasters, plus broader pre-1930 controls.

### 7. Guohui's `temp-guohui` work: name-prior and relationship-binding probes

Folder: `temp-guohui/`

This is Guohui's own independent work and should not be merged uncritically with Joachim's forced-choice suite.

#### `temp-guohui/name-prior-probe/`

Design:

- Model: `history-llms/ranke-0.6b-1900-3005`, not the 4B Ranke suite.
- 300 pre-1900 controls and 300 post-1900 probes.
- Tests event/person binding: relationship prompt -> given-name cue -> surname target.
- Controls include bare name and historical-prior prompts, length-matched.
- Uses next-token rank of the target first token; no generation.

Key summary:

- Pre-1900 controls: relationship binding improves rank strongly.
- Against historical-prior control, pre-1900 gain is 2.41x.
- Post-1900 gain is 1.38x.
- Top-5 success is 20.0% for pre-1900 vs only 1.3% for post-1900.
- Interpretation in the dashboard: exploratory evidence; not by itself proof of contamination. Pre-cutoff controls show much stronger relationship binding than post-cutoff probes.

Methodological issue:

- It uses first-token rank of the surname target. Earlier project logic warns this can be problematic for multi-token names and subword fragments.

#### `temp-guohui/name-prior-probe-ranke-talkie/`

Design:

- 1930 cutoff version for Ranke-4B-1929 and Talkie-1930-13B base.
- 300 pre-1930 controls and 300 post-1930 probes.
- Same event-person relationship-binding idea, with bare-name and historical-prior controls.

Key summary:

- Ranke pre-1930 relationship-vs-historical rank ratio: 14.15x.
- Ranke post-1930 relationship-vs-historical rank ratio: 1.74x.
- Talkie pre-1930 relationship-vs-historical rank ratio: 34.28x.
- Talkie post-1930 relationship-vs-historical rank ratio: 2.37x.
- Against bare name, Talkie post-1930 ratio is 5.53x; Ranke post-1930 is 1.73x.
- Dashboard interpretation: Ranke stays closer to the 1x no-help baseline; Talkie looks more contaminated under this lens.

Current reconciliation with Joachim's later suite:

- The forced-choice matched-distractor suite argues that Guohui's rank-ratio signal may partly reflect Talkie's larger size and richer generic-name floor.
- Raw rank improvement is not on a common chance scale across 4B and 13B models.
- Matched forced choice puts both models on the same 25% chance baseline and finds Talkie at chance.

This does not make Guohui's work useless. It provides a sensitive "relationship-binding lift" diagnostic, but the interpretation is weaker than matched forced choice unless it is calibrated against model-size effects, first-token artifacts, and source/category composition.

## Current status-quo conclusion

The best current evidence says:

- Ranke-4B-1929 and its instruct variant know pre-cutoff controls well.
- On the best current strict forced-choice post-cutoff tests, Ranke is at chance.
- Talkie-1930-13B base also knows pre-cutoff controls and is at chance post-cutoff on the same bank.
- There are shallow traces: name/string completion, association lift for some famous post-cutoff topics, and relationship-binding rank improvements.
- Those traces do not currently rise to coherent post-cutoff fact knowledge under no-cue matched-distractor tests.

A conservative wording would be:

"We find no evidence that Ranke-4B-1929 or Talkie-1930-13B possess coherent post-cutoff factual knowledge on our current matched forced-choice batteries. Both models pass pre-cutoff positive controls and perform at chance on post-cutoff items. However, we do observe shallow name/string and association traces, and the full pretraining corpus has not yet been audited directly, so this is evidence of no detected contamination under these probes rather than proof of corpus-level absence of contamination."

## Main unresolved issues

1. Direct corpus audit is still pending. The project has a `6_corpus_contamination_finder/find_contamination.py` plan, but the full 160B corpus has not yet been inspected in this folder.
2. The current integrated bank is only 45 post-cutoff items across Vietnam, WWII, and Cold War. It should be broadened before a final general claim.
3. Some generated reports are stale and can mislead readers; `README.md` and some HTML artifacts still reflect older conclusions.
4. Rank-ratio relationship-binding probes are sensitive but not yet calibrated enough to serve as standalone contamination proof.
5. Prompt/item selection may have evolved after seeing results, so the final bank should be frozen and versioned before any formal write-up.
6. Remaining positive traces may still matter for downstream synthetic-respondent use even if they are not coherent historical knowledge.

## Please review as if you are a hostile but fair coauthor

Please do not simply agree. Stress-test the logic and give a prioritized list of changes needed before we can say "the ranking model is uncontaminated" or even "there is no evidence of contamination."
