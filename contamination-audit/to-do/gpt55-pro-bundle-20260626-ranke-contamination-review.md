# GPT-5.5 Pro Extended Reasoning Bundle

## Instruction To GPT-5.5 Pro

You are GPT-5.5 Pro in extended reasoning mode. This is a self-contained review bundle. Do not assume access to files, links, repositories, prior chats, dashboards, or attachments not included below. If the evidence is insufficient, say exactly what is missing.

Please act as a hostile but fair methodological coauthor. The goal is not to praise the current analysis. The goal is to stress-test whether our contamination audit of time-locked historical language models is credible, identify the weakest links, and design the next experiment.

## User Task

We are evaluating whether time-locked historical language models, especially Ranke-4B-1929 and its instruct variant, contain post-cutoff contamination. We also compare them with Talkie-1930-13B base. The current project has numbered folders `1` through `8` plus `temp-guohui/`.

Please review two things:

1. The current contamination-testing workflow and evidence from the project folders.
2. Guohui's proposed additional idea: use a clearly modern/open model, such as a current LLaMA/Qwen-style model, as a "clearly polluted water" positive-control benchmark. For post-cutoff questions, compute log probabilities or ranks in both the time-locked candidate model and the clearly modern model. If the modern model ranks the true answer much better than the time-locked model, maybe that difference can serve as a calibrated contamination benchmark. If the time-locked model approaches the modern model's rank/logprob signal, that may indicate contamination.

## Desired Output

Please produce a structured methodological review with:

1. Strongest defensible claim right now about Ranke-4B-1929, Ranke-4B-1929-1202-Instruct, and Talkie-1930-13B.
2. Strongest evidence for cleanliness or non-contamination.
3. Strongest evidence for remaining leakage, shallow traces, or unresolved contamination.
4. Weakest methodological links and likely referee/coauthor attacks.
5. Assessment of the "clearly polluted model as benchmark" idea: whether plausible, how to improve it, and what caveats matter.
6. A concrete next-experiment design, including item bank, controls, model roles, metrics, statistical tests, and decision rules.
7. Conservative write-up language for a paper or internal memo.

Please distinguish:

- coherent post-cutoff factual knowledge,
- shallow name/string exposure,
- generic historical/name priors,
- tokenizer/first-token artifacts,
- model-size and architecture effects,
- direct corpus contamination.

## Context Snapshot

The user's dictated prompt called the model something like "RENKEY historical large-ranging model." In this workspace the model is clearly `Ranke`, a time-locked historical LLM family. Ranke models use a Qwen3-4B architecture trained from random initialization on historical text up to cutoff years such as 1900, 1913, or 1929. The audit focuses mostly on Ranke-4B-1929, its `1202-Instruct` fine-tune, and a comparison model Talkie-1930-13B base.

The analysis evolved quickly, and some root files are stale. Earlier passes suggested localized US-politics contamination. Later stricter tests softened that to: no strong evidence of coherent post-cutoff fact knowledge on the best current forced-choice tests, but shallow name/string and association traces remain.

The main methodological distinction is:

- matched forced-choice tests with 3 distractors and whole-answer summed log probability are treated as the primary test for coherent post-cutoff fact knowledge;
- rank-lift / association-lift probes are treated as sensitive diagnostics for shallow traces, not standalone proof of factual contamination;
- direct corpus auditing is still pending and would be stronger evidence than model probes alone.

## File Inventory

Included as summaries or excerpts:

- `README.md`: high-level project overview, including an older/stale bottom line.
- `CONTEXT.md`: current project context and key methodological decisions.
- `LOG.md`: chronological trail, especially the later reversals after cue-free and stricter tests.
- `1_advice_and_panel_review/03_fusion_synopsis.md`: early advice emphasizing corpus inspection before model probing.
- `3_fasttext_date_classifier/README.md`: proposed date-classifier pipeline for corpus-level contamination screening.
- `4_sft_data_scan/scan_sft2.py`: disambiguated scan approach for SFT datasets.
- `6_corpus_contamination_finder/find_contamination.py`: planned corpus grep/disambiguation scanner.
- `8_suite/batteries.py`: integrated forced-choice question bank design.
- `8_suite/suite.py` and `8_suite/talkie_probe.py`: whole-answer summed log probability scoring for Ranke and Talkie.
- `8_suite/results_*.json`: aggregate forced-choice results.
- `temp-guohui/name-prior-probe/*summary.csv` and quality JSON: Guohui's 1900 name-prior and event-binding probes.
- `temp-guohui/name-prior-probe-ranke-talkie/*summary.csv` and quality JSON: Guohui's 1930 Ranke-vs-Talkie relationship-binding probe.

Known exclusions:

- Raw large JSON outputs, full HTML dashboards, PDFs, caches, and generated files are excluded.
- Actual Hugging Face tokens, API keys, credentials, cookies, and local auth files are not included.
- Some script excerpts are summarized rather than copied verbatim because local token-file paths appear in code, though no secret value was visible.

## Current Project Map

The workspace has these major folders:

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
```

The root also contains reports and dashboards such as `report.html`, `diagnostic.html`, `questions.html`, `story.html`, `coverage.html`, `assoc_summary.html`, and `assoc_dig.html`.

## Current Synthesis

### What the older README says

`README.md` currently still contains an older bottom line:

```text
The Ranke models are mostly clean time-locked models: ~90%+ correct on pre-cutoff facts, at chance post-cutoff - across UK / France / Germany / Italy / Spain / Russia politics, Western culture, and science. The one contaminated area is US 20th-century political history: post-cutoff US politics sits at 73-78% correct (significantly above the 50% chance, pooled p approx. 0.02), statistically as good as - or only marginally below - pre-cutoff US politics.
```

This statement is now partly stale. Later tests are stricter and less supportive of a coherent US-politics contamination claim.

### What later `CONTEXT.md` / `LOG.md` say

Later entries emphasize the correction:

```text
under the strictest/airtight test the models are essentially CLEAN -- no category (incl US politics) shows significant post-cutoff KNOWLEDGE. The earlier 'US politics RED (68-89%)' was length-bias metric (89%) + first-name cues / item mix (68%). What remains is SHALLOW name/term exposure (A/B proves it), not coherent contamination.
```

The same later trail also says association traces still exist:

```text
REFINED PICTURE (3 levels): (1) shallow NAME strings (can complete 'John F.->Kennedy' with cue); (2) a few strong ASSOCIATIONS (Auschwitz/Vietnam/Stalingrad, as strong as legit pre-cutoff ones); (3) NO coherent FACT knowledge (free generation confabulates who/when/what).
```

And for recognition/coverage:

```text
POST-cutoff leak: group mean approx. 0 (no broad leak). Per-item, a few modest positives for 1929 base: Vietnam +2.23, Nuremberg +2.17, Stalingrad +2.12, Auschwitz +1.75, Yalta +1.62 - but all are PLACE NAMES that existed pre-1929, so this is residual place-name prior, NOT leaked event knowledge. The PURE post-1929 EVENT tests come out NEGATIVE: Hiroshima -0.42, Pearl Harbor -1.12, Chernobyl -1.69, MLK/Memphis -1.60, Cuba missile crisis -2.46.
```

Please treat the later stricter tests as more current than the older README headline.

## Important Methodological Decisions Already Learned

The project went through several mistakes and corrections:

1. Average log probability per token is length-biased.
2. Total log probability is also biased if answer lengths differ.
3. First-token scoring can compare subword fragments rather than full answers, especially for names.
4. First-name or answer-name cues turn the task into name completion rather than historical knowledge.
5. The current preferred model-probe metric for coherent knowledge is whole-answer teacher-forced scoring against matched distractors, preferably length-matched.
6. The decision rule should compare post-cutoff performance both against chance and against pre-cutoff positive controls.
7. Corpus-level inspection remains more direct than model probing, but the full 160B corpus has not yet been audited in this folder.

## Evidence Layer 1: Integrated Forced-Choice Suite

The latest integrated comparison is in `8_suite/`.

### Design excerpt from `8_suite/batteries.py`

```text
Every item is a forced choice: (label, prompt, true_continuation, [3 matched distractors]). The scored continuation strings are designed so that ONLY post-cutoff knowledge (for POST) or genuine pre-cutoff knowledge (for PRE) can pick the true one over equally-plausible alternatives. Matched distractors net out the generic-name/answer floor -- which scales with model size -- so a 4B and a 13B are compared on the SAME chance line (1/4 = 25%). That is the whole point: it makes Ranke-4B vs Talkie-13B a FAIR comparison, unlike a raw rank-ratio that inflates with model size.
```

The bank has:

```text
45 post-cutoff forced-choice items:
- 15 Vietnam-US war
- 15 WWII
- 15 Cold War

16 pre-cutoff positive controls

Chance line: 25%, because each item has one true answer plus three matched distractors.
Metric: whole-answer summed log probability.
```

### Scoring method

`8_suite/suite.py` scores Ranke models with:

```text
metric = forced_choice_sumlogp
```

It computes the summed log probability of the true continuation and each distractor continuation. An item is correct if:

```text
sum_logp(true continuation) > max sum_logp(distractor continuations)
```

`8_suite/talkie_probe.py` adapts Talkie's model implementation to produce all-position logits for teacher forcing. The script validates that its custom all-position logits match Talkie's official last-position forward pass:

```text
validation_maxdiff = 0.0
```

### Aggregate results from `8_suite/results_*.json`

```json
{
  "Ranke-4B-1929": {
    "pre_ALL": {"correct": 16, "n": 16, "pct": 100.0, "chance_pct": 25.0, "binom_p": 0.0, "mean_gap": 4.816},
    "post_ALL": {"correct": 14, "n": 45, "pct": 31.1, "chance_pct": 25.0, "binom_p": 0.2159, "mean_gap": -2.364}
  },
  "Ranke-4B-1929-1202-Instruct": {
    "pre_ALL": {"correct": 16, "n": 16, "pct": 100.0, "chance_pct": 25.0, "binom_p": 0.0, "mean_gap": 6.349},
    "post_ALL": {"correct": 16, "n": 45, "pct": 35.6, "chance_pct": 25.0, "binom_p": 0.0753, "mean_gap": -1.958}
  },
  "talkie-1930-13b-base": {
    "pre_ALL": {"correct": 16, "n": 16, "pct": 100.0, "chance_pct": 25.0, "binom_p": 0.0, "mean_gap": 6.406},
    "post_ALL": {"correct": 12, "n": 45, "pct": 26.7, "chance_pct": 25.0, "binom_p": 0.4543, "mean_gap": -2.68}
  }
}
```

Topic-level post-cutoff results:

```text
Ranke-4B-1929:
- Cold War: 5/15 = 33.3%, p = 0.3135
- Vietnam-US war: 5/15 = 33.3%, p = 0.3135
- WWII: 4/15 = 26.7%, p = 0.5387

Ranke-4B-1929-1202-Instruct:
- Cold War: 6/15 = 40.0%, p = 0.1484
- Vietnam-US war: 5/15 = 33.3%, p = 0.3135
- WWII: 5/15 = 33.3%, p = 0.3135

Talkie-1930-13B base:
- Cold War: 3/15 = 20.0%, p = 0.7639
- Vietnam-US war: 4/15 = 26.7%, p = 0.5387
- WWII: 5/15 = 33.3%, p = 0.3135
```

Interpretation proposed by the local notes:

- All three models ace pre-cutoff controls.
- All three are statistically at chance on the current post-cutoff forced-choice suite.
- The integrated bank is still narrow: Vietnam, WWII, Cold War.
- This is stronger evidence than raw rank-ratio comparisons because all models face the same 25% chance baseline.

## Evidence Layer 2: Strict Cue-Free Diagnostic and Earlier Reversals

From `5_results/dig_validated_results_full.csv`, aggregated:

```text
Post-cutoff cue-free/strict results by model:
- Ranke-4B-1900: 32/76 = 42.1%
- Ranke-4B-1913: 37/76 = 48.7%
- Ranke-4B-1929: 36/76 = 47.4%
- Ranke-4B-1929-1202-Instruct: 33/76 = 43.4%
```

Post-cutoff category totals pooled across models:

```text
Culture:          11/28 = 39.3%
France politics:   7/28 = 25.0%
Germany politics: 16/40 = 40.0%
Italy politics:   12/24 = 50.0%
Russia politics:  17/36 = 47.2%
Science:          22/40 = 55.0%
Spain politics:    8/24 = 33.3%
UK politics:      16/32 = 50.0%
US politics:      29/52 = 55.8%
```

The later `LOG.md` summarizes this as:

```text
EVERY category at chance post-cutoff incl US politics: US 56% p=.62, UK 54% p=.80, France 29%, Germany 43%, Russia 48%, Italy 50%, Spain 28%, Culture 38%, Science 53%. Answer-type: name 44%, term 49%, date 45% (all chance).
```

### Cued versus no-cue paired A/B evidence

From `5_results/pairs_results.csv`, aggregate across 80 rows:

```text
Cued:   44/80 = 55.0%
No cue: 30/80 = 37.5%
Difference: +17.5 percentage points for cued prompts
```

By model:

```text
Ranke-4B-1900: cued 12/20 = 60.0%, no-cue 6/20 = 30.0%
Ranke-4B-1913: cued 12/20 = 60.0%, no-cue 7/20 = 35.0%
Ranke-4B-1929: cued 10/20 = 50.0%, no-cue 8/20 = 40.0%
Ranke-4B-1929-1202-Instruct: cued 10/20 = 50.0%, no-cue 9/20 = 45.0%
```

Interpretation in local notes: name cues create shallow name-completion signal. That signal should not be confused with coherent fact knowledge.

### Quadruplet / stricter distractor evidence

From `5_results/quad_results.csv`:

```text
Post-cutoff:
- 2-way: 138/304 = 45.4%
- k-way:  63/304 = 20.7%

Pre-cutoff:
- 2-way: 111/120 = 92.5%
- k-way: 104/120 = 86.7%
```

Interpretation in local notes: many marginal 2-way post-cutoff hits vanish when the model must beat all matched alternatives.

## Evidence Layer 3: Association-Lift and Shallow Traces

The project also has association-lift probes, for example `5_results/assoc_results2.csv`. Early rows include:

```text
model,cutoff,kind,label,target,control,lift
Ranke-4B-1900,1900,POST,Hiroshima->atomic bomb,Hiroshima,Kyoto,1.579
Ranke-4B-1900,1900,POST,Nagasaki->atomic bomb,Nagasaki,Osaka,0.192
Ranke-4B-1900,1900,POST,Auschwitz->death camp,Auschwitz,Krakow,1.641
Ranke-4B-1900,1900,POST,Pearl Harbor->attack,Pearl Harbor,San Diego,2.139
Ranke-4B-1900,1900,POST,Stalingrad->battle,Stalingrad,Kazan,1.688
Ranke-4B-1900,1900,POST,Vietnam->US war,Vietnam,Thailand,4.896
Ranke-4B-1900,1900,POST,Cuba->missile crisis,Cuba,Haiti,-4.154
```

The later local interpretation is:

```text
There are shallow name/string and association traces for some famous post-cutoff terms or old place names, but free generation confabulates and matched forced-choice tests do not currently show coherent post-cutoff fact knowledge.
```

Please evaluate whether this interpretation is too forgiving, too cautious, or appropriate.

## Evidence Layer 4: Corpus and SFT Audit Plans

The strongest possible evidence would be direct training-corpus auditing. This is not complete yet.

### Initial advice from `1_advice_and_panel_review/03_fusion_synopsis.md`

The early fused advice says to start with the data:

```text
The model is a witness; the training text is the crime scene. A witness's behaviour is circumstantial - interesting, but arguable. The text itself is physical evidence - if a 1925 newspaper is sitting in your pile, you don't need clever psychology to prove contamination. So: look in the box before you guess what's inside it.
```

It also warned:

```text
If you test the model with modern phrases - "World War I," "the Soviet Union," "the Great Depression" - it'll look gloriously ignorant, and you'll declare victory. But it's not ignorant of the event; it's ignorant of your modern name for it. Nobody in 1913 said "World War I." To honestly test whether the model knows the future, you must speak its language.
```

### Date-classifier pipeline from `3_fasttext_date_classifier/README.md`

The folder proposes a scalable "when was this written?" classifier:

```text
A cheap, scalable "when was this text written?" classifier, to catch the contamination your wordlists and n-grams miss: post-1913 text written in clean period language (later histories, reprints, hindsight commentary). It scores by the statistical fingerprint of when prose was written, not by banned words - and runs over 160B tokens on CPU in hours, no LLM needed.
```

Critical caveat:

```text
The classifier must learn date, not topic. If your pre-1913 examples are all old books and your post-1913 examples are all modern web text, it learns "modern register" and waves through a formal 1955 history book - exactly the contamination you care about.
```

### Corpus contamination finder from `6_corpus_contamination_finder/find_contamination.py`

This script is designed to grep/disambiguate the full pretraining corpus:

```text
This is the tool to point at the 160B PRETRAINING corpus on server 2. It streams the corpus, flags documents that contain unambiguous post-1929 entities, and - crucially - uses CONTEXT DISAMBIGUATION so it does NOT drown in the false positives that fooled the first keyword pass (Normandy=1066, Israel=the Bible, Kennedy=a 19th-c. surname, Nagasaki=an old port).
```

Local notes say this direct corpus audit is a next step, not yet done in the current folder.

## Evidence Layer 5: Guohui's `temp-guohui` Rank-Lift / Relationship-Binding Probes

These are independent from Joachim's forced-choice suite and should not be merged uncritically.

### `temp-guohui/name-prior-probe/` - 1900 event binding

Question quality JSON:

```json
{
  "cutoff_year": 1900,
  "row_unit": "independent_event_person_base_fact",
  "group_counts": {
    "pre_1900_control": 300,
    "post_1900_probe": 300
  },
  "automatic_checks": {
    "year_1900_rows": [],
    "pre_group_after_cutoff": [],
    "post_group_before_cutoff": [],
    "duplicate_event_person_facts": [],
    "duplicate_person_rows": [],
    "duplicate_binding_prompts": [],
    "target_leak_rows": [],
    "death_year_only_templates": [],
    "bad_nonperson_keywords": [],
    "non_keep_audit_rows": [],
    "missing_source_rows": []
  },
  "agent_audit": {
    "status": "pending_external_review"
  }
}
```

Event-binding summary:

```text
pre_1900_control, binding_vs_historical:
- n = 300
- historical_prior_lenmatched_geomean_rank = 144.32
- binding_geomean_rank = 59.96
- rank_ratio_control_over_treatment = 2.41
- treatment_better_share = 0.73

post_1900_probe, binding_vs_historical:
- n = 300
- historical_prior_lenmatched_geomean_rank = 540.76
- binding_geomean_rank = 392.04
- rank_ratio_control_over_treatment = 1.38
- treatment_better_share = 0.65
```

Interpretation: relationship prompts improve rank more for pre-1900 controls than for post-1900 probes, but post-1900 still has a positive rank-lift.

### `temp-guohui/name-prior-probe-ranke-talkie/` - 1930 Ranke-vs-Talkie event binding

Question quality JSON:

```json
{
  "cutoff_year": 1930,
  "row_unit": "independent_event_person_base_fact",
  "group_counts": {
    "pre_1930_control": 300,
    "post_1930_probe": 300
  },
  "automatic_checks": {
    "year_1930_rows": [],
    "pre_group_after_cutoff": [],
    "post_group_before_cutoff": [],
    "duplicate_event_person_facts": [],
    "duplicate_person_rows": [],
    "duplicate_binding_prompts": [],
    "target_leak_rows": [],
    "death_year_only_templates": [],
    "bad_nonperson_keywords": [],
    "non_keep_audit_rows": [],
    "missing_source_rows": []
  },
  "agent_audit": {
    "status": "passed_after_revisions",
    "revision_notes": [
      "Post-1930 rows now require event_year >= 1932 for automatically generated office-holder table rows.",
      "The builder rejects target surname/eponym leakage, particle-ending cues, non-person names, duplicate people, duplicate event labels, and the audited compound-surname exclusion list.",
      "Office-holder table prompts include party/affiliation context when available instead of the original bare took-office template.",
      "Remaining caveat: source clustering is substantial and confidence intervals are case-level rather than cluster-adjusted."
    ]
  }
}
```

The probe scores three same-length prompts for each base fact:

```text
- bare_name_lenmatched
- historical_prior_lenmatched
- relationship_binding
```

It reads only the next-token distribution. It does not generate text.

Key implementation features from `run_ranke_talkie_probe.py`:

```text
- The relationship prompt, bare-name prompt, and historical-prior prompt are length-matched at the prompt-token level.
- The measured quantity is the next-token rank of the target's first token.
- Lower numerical rank is better.
- Aggregation uses geometric mean rank and sign-flip p-values.
```

Ranke summary:

```text
Ranke, pre_1930_control, relationship_vs_historical:
- n = 300
- historical_prior_lenmatched_geomean_rank = 35.35
- relationship_binding_geomean_rank = 2.50
- rank_ratio_control_over_treatment = 14.15
- treatment_better_share = 0.803

Ranke, post_1930_probe, relationship_vs_historical:
- n = 300
- historical_prior_lenmatched_geomean_rank = 275.44
- relationship_binding_geomean_rank = 158.01
- rank_ratio_control_over_treatment = 1.74
- treatment_better_share = 0.587
```

Talkie summary:

```text
Talkie, pre_1930_control, relationship_vs_historical:
- n = 300
- historical_prior_lenmatched_geomean_rank = 41.00
- relationship_binding_geomean_rank = 1.20
- rank_ratio_control_over_treatment = 34.28
- treatment_better_share = 0.877

Talkie, post_1930_probe, relationship_vs_historical:
- n = 300
- historical_prior_lenmatched_geomean_rank = 370.11
- relationship_binding_geomean_rank = 156.30
- rank_ratio_control_over_treatment = 2.37
- treatment_better_share = 0.710
```

Against bare-name control, Talkie post-1930 looked especially large:

```text
Talkie, post_1930_probe, relationship_vs_bare:
- bare_name_lenmatched_geomean_rank = 863.77
- relationship_binding_geomean_rank = 156.30
- rank_ratio_control_over_treatment = 5.53
- treatment_better_share = 0.847
```

Local reconciliation with the forced-choice suite:

- Guohui's rank-lift probe is sensitive, but it is not on a common chance scale across different model sizes and tokenizers.
- The 13B Talkie model may have a richer generic name floor or better ranking behavior even without coherent post-cutoff knowledge.
- The matched forced-choice suite puts Ranke and Talkie on the same 25% chance line and finds both at chance post-cutoff.
- Therefore, raw rank-lift should probably be interpreted as a diagnostic for shallow relationship/name traces, not proof of coherent contamination.

## Guohui's Proposed New Benchmark Idea

Guohui's intuition is:

```text
We want to test pollution level in unknown water. A clearly modern open-source model is like clearly polluted water. For post-cutoff questions, compute the true-answer log probability or rank in the time-locked model. Also ask the same question to the clearly modern model, for example a current Qwen or LLaMA model. The true answer should rank much better in the clearly polluted model than in a clean time-locked model. If our model's ranking is close to the polluted model, that suggests more contamination; if the gap is large, that suggests more cleanliness.
```

Please review whether this is plausible and how to make it statistically and methodologically defensible.

Specific issues to address:

1. Should the benchmark use raw rank, log rank, summed log probability, log odds, or forced-choice accuracy?
2. Should the benchmark model be an open modern base model, an instruction model, or both?
3. How do we handle architecture, tokenizer, size, and data-mixture differences?
4. Is it better to compare raw candidate/modern ratios, or residualize against null/shuffled prompts first?
5. How should we choose items where the modern model itself shows reliable knowledge?
6. Can this estimate "pollution level," or only "fraction of modern-model relationship signal"?
7. What would be a valid negative control and positive control?
8. How should we integrate this benchmark with the existing forced-choice suite?

## Candidate Design Sketch To Critique

One possible improved version of Guohui's idea:

1. Freeze a post-cutoff item bank before scoring.
2. For each item, create:
   - a true relationship prompt,
   - several null or shuffled relationship prompts that look plausible but should not identify the target,
   - a bare-name control,
   - a historical-prior control,
   - a matched forced-choice question with 3 or more distractors.
3. Score the same items on:
   - Ranke-4B-1929 base,
   - Ranke-4B-1929-1202-Instruct,
   - Talkie-1930-13B base,
   - one or more modern open base models as positive controls.
4. Use log rank or summed log probability, not raw rank ratio when possible.
5. Define a residual relationship signal per model and item:

```text
L_mi = log(rank_historical_prior_mi) - log(rank_relationship_mi)
R_mi = L_true_mi - mean(L_null_mik)
```

where larger `R_mi` means the correct relationship prompt improves the target rank beyond generic/null relationship prompt effects.

6. Normalize against a modern positive-control model only on items where the modern model has a positive reliable signal:

```text
normalized_signal_m = mean_i(R_mi) / mean_i(R_modern,i)
```

7. Interpret the result as a calibrated signal index, not direct percent contaminated data.
8. Any high residual signal must be followed by no-cue matched forced-choice/property tests before claiming coherent post-cutoff factual knowledge.

Please feel free to reject, modify, or replace this design.

## Open Problems And Caveats Already Known

1. Direct corpus audit of the 160B pretraining corpus remains pending.
2. The integrated forced-choice suite has only 45 post-cutoff items across Vietnam, WWII, and Cold War.
3. Some dashboards and root files are stale and still reflect older conclusions.
4. Guohui's relationship-binding probe uses next-token rank of the target first token, which may inherit first-token/subword artifacts.
5. The relationship-binding bank is heavily shaped by office-holder tables and politics/history sources; source clustering is substantial.
6. Confidence intervals in some temp-guohui summaries are case-level, not cluster-adjusted.
7. Prompt/item selection may have evolved after seeing scores; a final paper-level bank should be frozen and versioned before scoring.
8. The modern positive-control model is not literally "100% contaminated water." It differs in size, tokenizer, architecture, data mixture, instruction tuning, and factual coverage.
9. Some post-cutoff targets are also old place names or old surnames, which can create false positives through pre-cutoff priors.
10. Free generation is unreliable and confabulatory; recognition/logprob tests appear to measure stored knowledge more cleanly than chat.

## Please Review

Please answer as if deciding what we should do next. Do not simply agree with the current local interpretation. If you think the current evidence is not enough to claim cleanliness, say so. If you think the modern/polluted benchmark idea is flawed, say exactly how and propose a better version.

The practical goal is to tell Guohui and Joachim what experiment to run next and what claim they can safely make now.
