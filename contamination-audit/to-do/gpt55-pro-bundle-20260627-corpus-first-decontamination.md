# GPT-5.5 Pro Extended Reasoning Bundle

## Instruction To GPT-5.5 Pro

You are GPT-5.5 Pro in extended reasoning mode. This is a self-contained bundle. Do not assume access to files, links, repositories, previous chats, private messages, or attachments not included below. If the evidence is insufficient, state exactly what is missing.

Please act as a hostile but constructive methodological coauthor and ML/data-engineering architect. The project is trying to build historically time-locked language models while avoiding post-cutoff contamination in the training data. The user wants a direct, corpus-first decontamination strategy that avoids wasting GPU hours by repeatedly training a model, probing it, finding remaining contamination, going back to clean the data, and retraining again.

The key question is not "how do we prove the current model is clean?" The key question is:

> Given a very large historical pretraining corpus, can we decontaminate the training data directly, before expensive model training, using scalable corpus methods, source-level metadata, temporal classifiers, probes-as-search-seeds, audit sampling, and validation? What is the best practical pipeline?

## User Task

Design a rigorous, efficient corpus-first decontamination pipeline for a large historical-language-model training corpus, apparently around 160B tokens. The model cutoff of interest is around 1929/1930, though earlier cutoffs such as 1900 and 1913 also appear in the project.

The user believes the current loop is inefficient:

1. Start with polluted training data.
2. Train a model using scarce GPU resources.
3. Probe the trained model using log-probability / rank / forced-choice tests.
4. Infer clues about contamination.
5. Decontaminate the corpus.
6. Train again.
7. Possibly discover the data are still not clean.

They want to know whether there is a way to clean the training data directly, before training a model, and how to do it at large scale.

## Desired Output

Please produce a concrete technical memo with:

1. A diagnosis of why the current train-probe-clean-retrain loop is inefficient and where model probes still help.
2. A corpus-first architecture for direct decontamination of a 160B-token historical corpus before model training.
3. A staged pipeline, from raw source inventory through filtering, source-level purge, manual audit, and final training manifest.
4. Specific detectors to use and how to combine them:
   - exact and regex search,
   - disambiguated event/entity markers,
   - date-of-writing / temporal-style classifiers,
   - metadata and edition/reprint checks,
   - near-duplicate matching against known post-cutoff corpora,
   - source-level aggregation,
   - active learning or lightweight classifiers,
   - optional small LLM-assisted review only where it is cost-effective.
5. A validation plan that does not require training a full LLM:
   - stratified manual audit,
   - contamination prevalence estimates,
   - recall/precision targets,
   - seeded-injection tests,
   - historical cutoff holdout tests,
   - how to certify "below detectable threshold" honestly.
6. How to convert model-probe results into corpus search seeds without overfitting to probe artifacts.
7. A realistic plan for the suspicious domains in this project, especially post-1945 US politics, Vietnam, Cold War, WWII, civil rights, and postwar institutions/science.
8. A decision rule for when to train a 1.7B diagnostic model, if ever, and when to skip it and go straight to one large clean run.
9. A concrete implementation checklist that Daniel, Dominik, Guohui, and Joachim could execute.
10. A conservative answer to "Can we clean directly without training a model?" with caveats and what information is still missing.

Please distinguish:

- corpus contamination vs SFT/instruction contamination,
- coherent post-cutoff factual knowledge vs shallow name/string exposure,
- source-level contamination vs isolated snippets,
- metadata/date errors vs text-content errors,
- high-recall quarantine vs high-precision proof,
- evidence of contamination vs absence of detected contamination.

## Context Snapshot

This project is `ranke-contamination-audit`. It audits time-locked historical language models, especially `Ranke-4B-1929` and `Ranke-4B-1929-1202-Instruct`, and compares them with `Talkie-1930-13B base`.

The Ranke models are supposed to be trained only on text before specified cutoffs such as 1900, 1913, or 1929. The application is historical/political-economy research using time-locked models as "synthetic respondents" or historically situated reasoning engines.

The project has evolved quickly and some files disagree. Earlier notes say US mid-century political history is clearly contaminated. Later notes say many early alarms were artifacts and that the current honest claim is "no evidence of coherent post-cutoff factual knowledge at the tested power," not proof of clean data. A newer preregistered test appears to have generated some concerning Ranke-4B-1929 results again, especially for US politics 1945-74 and Vietnam, but its pre-cutoff control performance is unexpectedly weak under the calibrated metric, so interpretation is not settled.

The user and Joachim are now concerned that relying on trained-model probes to discover contamination may waste GPU hours. Joachim's recent methodological direction, summarized without raw private messages, is:

- The decontamination loop has been too "model-score first": search for contaminated areas, run a large judgment/probe battery, then decide what to zap.
- A better engineering direction may be: when even one or two probes fire, turn all associated terms/entities/relations into corpus search/decontamination seeds rather than waiting for a large definitive battery.
- The 1.7B run may be useful as a diagnostic gate, but the larger goal should be to clean well enough to train one large model from scratch.
- There is also a broader conceptual idea of separating "worldview" from facts: use the time-locked model for historically situated reasoning/worldview, and retrieve factual context separately via RAG. That is relevant to the final application but secondary to this bundle's data-cleaning question.

## File Inventory

Included materials are excerpts and summaries from these local files:

- `README.md`: high-level project overview, older/stale "US politics contaminated" headline, and current proposed corpus search next step.
- `CONTEXT.md`: current methodological state, metric corrections, and later "cleaner than first thought" interpretation.
- `LOG.md`: chronological trail of reversals, failed/retired methods, absolute probes, Talkie comparison, and preregistered detection design.
- `1_advice_and_panel_review/03_fusion_synopsis.md`: early advice emphasizing raw corpus inspection before clever model probing.
- `3_fasttext_date_classifier/README.md`: existing proposed pipeline for date-of-writing classifier over the large corpus.
- `3_fasttext_date_classifier/build_dataset.py`: genre-balanced passage construction for the date classifier.
- `3_fasttext_date_classifier/score_corpus.py`: streaming corpus scorer that flags documents by post-cutoff passage probability.
- `4_sft_data_scan/scan_sft.py` and `scan_sft2.py`: scans SFT datasets for post-cutoff markers with disambiguating context.
- `6_corpus_contamination_finder/find_contamination.py`: streaming disambiguated regex/grep scanner for post-cutoff leakage in the pretraining corpus.
- `8_suite/PREREGISTRATION.md`: preregistered detection design using broad and narrow forced-choice batteries.
- `8_suite/generate_items.py`: item-generation plan for post-1930 detection batteries.
- `8_suite/score_v2.py`: calibrated forced-choice scoring.
- `8_suite/results_v2_Ranke-4B-1929.json`: current v2 aggregate results for Ranke-4B-1929.
- `to-do/tasks.json`: team to-do structure derived from project discussion, without raw chat transcript.

Known exclusions:

- No raw WhatsApp transcript is included.
- No Hugging Face token, API key, credentials, cookies, or private auth files are included.
- Full generated HTML/PDF dashboards, giant JSON outputs, caches, `.pyc`, model files, and raw datasets are excluded.
- The actual 160B-token training corpus is not available in this local bundle, so the answer should specify what corpus metadata and access details are needed.

## Project Map

Major folders visible in the local project:

```text
1_advice_and_panel_review/
2_model_probes/
3_fasttext_date_classifier/
4_sft_data_scan/
5_results/
6_corpus_contamination_finder/
7_chat/
8_suite/
temp-guohui/
to-do/
```

`2_model_probes/` contains many exploratory model-probe scripts. `3_fasttext_date_classifier/` and `6_corpus_contamination_finder/` are the main existing corpus-first tools. `4_sft_data_scan/` scans instruction/SFT data. `8_suite/` contains the newer preregistered model-evaluation pipeline.

## Current Pipeline As Understood

The current or proposed loop, based on local notes and project discussion, is:

1. Train a time-locked model from a supposedly pre-cutoff corpus.
2. Probe it using forced-choice/logprob/rank tests to detect post-cutoff knowledge.
3. Identify suspicious domains, e.g. US politics after 1945, Vietnam, Cold War.
4. Use those model-probe signals as hints for corpus search.
5. Search the raw 160B-token pretraining corpus and SFT data for post-cutoff markers.
6. Remove suspicious documents or sources.
7. Retrain and re-test.

The user is asking whether steps 1-4 can be largely bypassed: do more direct corpus decontamination first, with model training used only as a later acceptance test rather than as the primary contamination detector.

## Important Tension In The Current Evidence

### Older `README.md` headline

`README.md` says:

```text
The Ranke models are mostly clean time-locked models: ~90%+ correct on pre-cutoff facts, at chance post-cutoff - across UK / France / Germany / Italy / Spain / Russia politics, Western culture, and science. The one contaminated area is US 20th-century political history: post-cutoff US politics sits at 73-78% correct (significantly above the 50% chance, pooled p≈0.02), statistically as good as - or only marginally below - pre-cutoff US politics.
```

The same file says the "actual fix" is:

```text
1. python 6_corpus_contamination_finder/find_contamination.py --input <160B corpus>.jsonl --out flagged.jsonl
2. Read the snippets, find the recurring source, drop that source wholesale, fix the date filter.
3. Retrain the base; re-run 2_model_probes/battery50.py as the acceptance test.
```

### Later `CONTEXT.md` correction

`CONTEXT.md` later says the strictest tests undermined the earlier alarm:

```text
under the SINGLE RULE applied in full (answer must require post-cutoff text, never the topic's name, never cued; famous-old topics KEPT but probed via post-cutoff facts), pooled over 3 base models, EVERY category sits AT CHANCE post-cutoff INCLUDING US politics ... The models are essentially CLEAN of coherent post-cutoff KNOWLEDGE; what remains is shallow name/term exposure. Caveat: modest n = strong cleanliness upper bound, not proof.
```

It also states:

```text
SUPERSEDED earlier reads ... avg-logp metric gave "US politics 89%" = a token-LENGTH artifact; v2/v3 no-cue dig gave "US politics 68% p=.044 contaminated" -- but the fuller comprehensive battery ... drops US to chance too.
```

### Later `LOG.md` methodological reckoning

`LOG.md` says:

```text
Voth's challenge: "after 3 days isolating contamination you now invent a method that says clean - how?" He's right; "clean" was an overclaim.
```

It reframes the project:

```text
goal = DETECT EVIDENCE of contamination (not prove clean); a null = "no evidence at power", not "clean".
```

It also emphasizes:

```text
corpus search+retrain is the only CERTIFY path and needs the training corpus.
```

### Newer v2 model score may be concerning but unsettled

`8_suite/results_v2_Ranke-4B-1929.json`, using the preregistered calibrated forced-choice metric, reports:

```json
{
  "model": "Ranke-4B-1929",
  "metric": "calibrated_forced_choice",
  "broad_post": {
    "correct": 92,
    "n": 239,
    "pct": 38.5,
    "ci95": [32.6, 44.8],
    "binom_p": 0.0,
    "upper95": 44.8,
    "mean_gap": -1.348
  },
  "pre_controls": {
    "correct": 16,
    "n": 33,
    "pct": 48.5,
    "ci95": [32.5, 64.8],
    "binom_p": 0.003,
    "upper95": 64.8,
    "mean_gap": 0.57
  },
  "narrow": {
    "Cold War": {"correct": 12, "n": 40, "pct": 30.0, "binom_p": 0.2849},
    "US civil rights": {"correct": 9, "n": 37, "pct": 24.3, "binom_p": 0.6004},
    "US politics 1945-74": {"correct": 22, "n": 40, "pct": 55.0, "binom_p": 0.0},
    "Vietnam War": {"correct": 18, "n": 39, "pct": 46.2, "binom_p": 0.0033}
  }
}
```

This result, if trusted, suggests post-cutoff signal in broad postwar items, especially US politics 1945-74 and Vietnam. But the same calibrated metric only gets 48.5% on pre-cutoff controls, which is surprisingly low for a model that earlier aced pre-controls under the raw forced-choice metric. This may mean the v2 question bank/metric needs investigation before using it as evidence. Please address how corpus-first cleaning should proceed under this uncertainty.

## Lessons Already Learned From Model Probes

The project learned several hard methodological lessons:

1. Average log probability per token is length-biased.
2. Total log probability is biased in the other direction if answer lengths differ.
3. First-token scoring can compare subword fragments rather than full answers, especially for names.
4. Prompts containing first names, answer fragments, or answer form words can turn the test into string completion rather than knowledge.
5. Matched forced-choice with full-answer teacher-forced sum log probability is preferred for coherent factual knowledge.
6. Calibrating answer priors may help with lexical popularity, but can introduce new issues if it breaks pre-cutoff controls.
7. A null model-probe result is not proof of clean data; it only means no evidence at the tested power.
8. Probes are still valuable as "where to look" instruments, especially if any suspicious term/entity/relation fires.

## Existing Corpus-First Tool 1: Disambiguated Corpus Grep

File: `6_corpus_contamination_finder/find_contamination.py`

The script's own docstring says:

```text
find_contamination.py - grep a (pre-cutoff) training corpus for post-cutoff leakage.

This is the tool to point at the 160B PRETRAINING corpus on server 2. It streams the corpus, flags documents that contain unambiguous post-1929 entities, and - crucially - uses CONTEXT DISAMBIGUATION so it does NOT drown in the false positives that fooled the first keyword pass (Normandy=1066, Israel=the Bible, Kennedy=a 19th-c. surname, Nagasaki=an old port). It then traces flagged docs back to their source so you can pull the contaminating source wholesale.
```

It bakes in lessons:

```text
A bare keyword count LIES. "Normandy" is a medieval duchy; "Israel" is biblical; "Reagan" was a Confederate cabinet member. Always require disambiguating context for hook-prone terms, and ALWAYS eyeball the snippet before believing a hit.
```

It also says:

```text
At 160B tokens, over-removing is cheap, keeping dirty docs is not. Flag a whole document (and ideally its whole source) on a single solid hit.
```

Example high-priority markers in the script include:

```python
CONFIRMED = [
    (r"\btruman\b", r"harry|president|1945|atomic|doctrine"),
    (r"\beisenhower\b", r"dwight|president|general|allied|194\d|195\d"),
    (r"\bkennedy\b", r"john f|j\. ?f\.|jfk|president|assassinat|196\d|cuban"),
    (r"\bnixon\b", r"richard|president|watergate|196\d|197\d"),
    (r"\breagan\b", r"ronald|president|198\d"),
    (r"ho chi minh", None),
    (r"\bmuhammad ali\b", None),
]
```

The broader marker list includes topic buckets such as:

```text
US politics, WWII, Cold War, France, China/Asia, Sci/Tech, Intl orgs, Other 20c
```

Its output is not the full contaminated text, but records like:

```json
{
  "id": "document id",
  "source": "source field",
  "confirmed": true,
  "entities": ["kennedy"],
  "buckets": ["US politics"],
  "snippet": "short excerpt"
}
```

The script aggregates by entity, bucket, and source and tells the user to:

```text
open flagged.jsonl, read snippets, trace the recurring source values, and drop the whole contaminating source; then fix the date filter and retrain.
```

Open question: how should this be expanded into a full decontamination system rather than a one-off grep pass?

## Existing Corpus-First Tool 2: Date-Of-Writing Classifier

File: `3_fasttext_date_classifier/README.md`

The pipeline is intended as a cheap scalable "when was this written?" classifier:

```text
A cheap, scalable "when was this text written?" classifier, to catch the contamination your wordlists and n-grams miss: post-1913 text written in clean period language (later histories, reprints, hindsight commentary). It scores by the statistical fingerprint of when prose was written, not by banned words - and runs over 160B tokens on CPU in hours, no LLM needed.
```

Critical caveat:

```text
The classifier must learn date, not topic. If your pre-1913 examples are all old books and your post-1913 examples are all modern web text, it learns "modern register" and waves through a formal 1955 history book - exactly the contamination you care about.
```

Rules listed:

```text
1. Date-balance within genre. Include a 1900 and a 1960 history; an 1890 and a 1950 novel. Every genre must appear on both sides of 1913.
2. Put formal/academic/historical prose in the post-1913 set, not just easy modern text.
3. eval.py prints per-genre AUC - if a genre sits near 0.5, the model is cheating on topic there.
```

The scoring script `3_fasttext_date_classifier/score_corpus.py` streams a corpus and flags documents if the strongest passage or a meaningful share of passages look post-cutoff:

```python
flag = (mx >= args.flag_max) or (frac >= args.flag_frac)
rec = {
    "id": did,
    "n_chunks": len(ps),
    "mean": round(sum(ps) / len(ps), 4),
    "max": round(mx, 4),
    "frac_above": round(frac, 4),
    "flag": flag
}
```

The README suggests:

```text
Then quarantine flagged docs, trace them to their source, and drop the whole source - contamination clusters. Tune --flag-max/--flag-frac on your known-clean vs known-dirty piles first; at 160B tokens, over-removing borderline docs is cheap, keeping dirty ones is not.
```

Open questions:

- Is fastText enough, or should this be char n-grams / linear models / transformer classifiers / perplexity ratios?
- How should it be trained for the 1929/1930 cutoff, not just 1913?
- How should it be calibrated to avoid topic confounds?
- How should its flags be combined with regex/metadata/source flags?

## Existing Corpus-First Tool 3: SFT Data Scan

Files: `4_sft_data_scan/scan_sft.py` and `scan_sft2.py`

These scripts scan instruction/SFT datasets, not the pretraining corpus. They matter because post-training contamination can create apparent knowledge even if the base pretraining data are clean.

The v2 script describes itself as:

```text
Disambiguated post-1929 contamination scan (v2): markers that are common as pre-1929 place/surnames (Normandy, Israel, Kennedy, Nagasaki, Reagan, Vichy...) require a CONTEXT term in the same text piece, killing the false positives that swamped the first pass. Also counts the LITERAL facts the Instruct model emitted, to tell leakage apart from confabulation.
```

It scans dataset records and counts markers by topic and by location, e.g. user/question vs assistant/answer fields.

Example marker logic includes:

```python
("US politics", r"\btruman\b", r"harry|president|1945|1948|1949|doctrine|atomic"),
("US politics", r"\bkennedy\b", r"john f|j\. f\.|jfk|president|assassinat|196\d|cuban"),
("WWII", r"\bnormandy\b", r"1944|d-day|allied|landing|invasion|second world war|nazi|overlord|eisenhower"),
("Other 20c", r"\bisrael\b", r"1948|state of israel|israeli|zionis|ben-gurion|netanyahu|six-day|arab")
```

Open question: how should pretraining-corpus cleaning and SFT-cleaning be separated and validated?

## Existing Evaluation Plan: Preregistered Detection Study

File: `8_suite/PREREGISTRATION.md`

This is a model-probe plan, but it clarifies what domains the team suspects and what "detect contamination" means.

Goal:

```text
Find evidence of post-cutoff contamination, if any exists. This is a detection study, not a proof of cleanliness. A null result means "no evidence at the power we had," not "the model is clean."
```

Design:

```text
Broad (diffuse-leak detector). ~132+ items sampling post-1945 history across many sub-topics. delta = +10 pp above chance. Powered: n=132 -> ~82% power to detect a +10pp mean lift.

Narrow (concentrated-leak detector). Deep ~32-item batteries on specifically-suspected topics. delta = +20 pp above chance. Clusters: Cold War, Vietnam War, US civil-rights era, US politics 1945-74.
```

Metric:

```text
Forced choice among k = 4 options (1 true + 3 distractors); chance = 25%.
Uncued: the prompt is a bare description containing no fragment of any answer.
Calibrated: score each option a as c(a) = logp(a | question) - logp(a | "The answer is").
```

The preregistration includes:

```text
Any flagged cluster => confirm on fresh locked items + corpus search before claiming contamination.
Injection calibration (separate run): continued-pretrain a small model on k copies of a known post-cutoff article to measure the battery's detection floor in "article-repeats" - the honest statement of sensitivity.
```

Question for GPT-5.5 Pro: can the injection-calibration idea be converted into corpus-filter validation without training LLMs? For example, inject known post-cutoff documents into a held-out copy of the corpus and measure detector recall at document/source level.

## Existing Item Generation Plan

File: `8_suite/generate_items.py`

The script over-generates forced-choice candidate items for these topics:

```text
Cold War
Vietnam War
US civil rights
US politics 1945-74
decolonisation
space & computing
science & medicine
economics & institutions
culture & society
disasters & conflicts
pre-controls
```

It instructs the item generator:

```text
The tested fact/relationship must be POST-1930. The entity may be older, but the specific relation/event must be post-1930 (e.g. the city of Hiroshima is old, but 'Hiroshima -> atomic bombing' is post-1930 and valid).

The PROMPT is a bare description that contains NO part of any answer -- no names, words, or forms from the answer. Do NOT cue the answer's shape.
```

This item bank can be mined for search seeds, but it was built for model probing. Please explain how to transform question items into corpus queries without overfitting to the exact prompt/answer forms or missing contamination written in different language.

## Earlier Advice: Search Raw Text First

File: `1_advice_and_panel_review/03_fusion_synopsis.md`

Earlier advice in the project said:

```text
The real lesson is about order: don't start by interrogating the model. Start by checking what you fed it.
```

And:

```text
The model is a witness; the training text is the crime scene. A witness's behaviour is circumstantial - interesting, but arguable. The text itself is physical evidence.
```

It recommended:

```text
1. Freeze and list every ingredient that went into the model - raw text, metadata, scan headers, file names, and the text used to build its vocabulary - not just the "books."
2. Search the raw files for post-cutoff years, modern event names, and digital boilerplate.
3. Manually inspect the hits, paying special attention to old works reissued later with new prefaces.
4. Remove or quarantine anything genuinely post-cutoff, and retrain if serious.
```

This aligns with the user's current question, but it needs to be upgraded to a 160B-token, production-quality data pipeline.

## Team To-Do Context

From `to-do/tasks.json`, the relevant tasks and owners are:

```text
Confirm contaminated model location
- Owner: Daniel
- Done: history-llms/ranke-0.6b-1913-1106-cont-1 identified as contaminated model.
- Note: use as reference artifact for calibration, not production target.

Map where training data live
- Owner: Daniel + Dominik
- Goal: Dominik and Guohui know where the corpus and training artifacts live, how to access them, and which paths are relevant on CSCS/ALPS.

Search for post-1945 US political-history contamination
- Owner: Daniel + Dominik
- Dependency: map training data
- Goal: reproducible search pass reports suspect sources/chunks for post-1945 US presidents, Cold War, Vietnam, Watergate, and related terms.
- Note: use boring scalable tools first: grep-like search, n-grams, full-text indices, and source-level inspection.

Compare dropped Xmas data against random corpus samples
- Owner: Guohui
- Goal: ranked term list identifies words or clusters over-represented in data dropped around the Xmas-cleaning step.
- Note: feed likely terms into corpus search.

Answer the 1.7B retraining question
- Owner: Daniel
- Goal: short decision memo states token budget, compute cost, ETA, expected usefulness, and whether the proposed 30B-token run is enough.

Define frozen Ranke/Talkie holdout and thresholds
- Owner: Guohui + Joachim
- Goal: prespecified domains, item counts, screening rules, metric, cutoffs, one-shot scoring rule, and acceptable contamination deviations.
```

Question for GPT-5.5 Pro: how should these tasks be reorganized if the priority becomes direct corpus decontamination before model training?

## Practical Constraints

Please assume:

- The corpus is very large, around 160B tokens.
- The full corpus is not present in this local bundle and may live on a server/cluster such as CSCS/ALPS.
- Compute for full LLM training is expensive and should not be used as the main cleaning loop.
- CPU-scale corpus processing is acceptable.
- Lightweight classifiers such as fastText, char n-gram linear models, MinHash/SimHash, regex/Aho-Corasick, and source-level aggregation are acceptable.
- Some manual audit by humans is acceptable, but it must be stratified and small enough to do.
- Over-removal is probably less costly than retaining post-cutoff contamination, but too much over-removal may distort the historical corpus or remove valuable legitimate pre-cutoff material.
- The team can use a known contaminated 0.6B model as a calibration artifact, but the user wants the primary answer to avoid depending on training new LLMs.

## What I Want You To Decide

Please answer these hard questions directly:

1. Can the team build a credible direct decontamination pipeline without training a diagnostic model first?
2. If yes, what are the exact stages and artifacts?
3. What should be the unit of removal: sentence, chunk, document, source, edition, dataset, or entire source family?
4. How should the pipeline avoid naive keyword-search failures such as `Normandy`, `Israel`, `Kennedy`, `Nagasaki`, `Reagan`, and other pre-cutoff names with post-cutoff meanings?
5. How should the pipeline detect later histories written in old-fashioned style that do not contain obvious banned markers?
6. How should the pipeline detect modern metadata, OCR boilerplate, scan headers, reprint introductions, revised editions, filenames, and vocabulary/tokenizer contamination?
7. How should the team combine high-precision markers, temporal classifiers, near-duplicate matching, and source-level aggregation into one decision rule?
8. How should the team validate recall and precision without training a full LLM?
9. When does it become rational to spend GPU on a 1.7B diagnostic model?
10. What should Daniel, Dominik, Guohui, and Joachim each do next?

## Preferred Format For Your Answer

Please structure the answer as:

1. **Short Verdict**
2. **Why The Current Loop Wastes GPU**
3. **Direct Decontamination Pipeline**
4. **Detector Stack**
5. **Source-Level Removal Policy**
6. **Validation Without Full Model Training**
7. **How To Use Existing Probes As Search Seeds**
8. **1.7B Diagnostic Model: When To Run Or Skip**
9. **Owner-Based Implementation Checklist**
10. **Risks, Failure Modes, And Missing Information**
11. **A One-Page Operational Plan**

Be concrete enough that a team could start implementing the plan tomorrow.

