# Decontamination pipeline design

**User:** Anonymous  
**Created:** 6/27/2026 12:15:55  
**Updated:** 6/27/2026 14:56:02  
**Exported:** 6/27/2026 14:58:19  
**Link:** [https://chatgpt.com/c/6a3fa2da-db34-83eb-93e1-3fcea336ea2b](https://chatgpt.com/c/6a3fa2da-db34-83eb-93e1-3fcea336ea2b)  

## Prompt:
6/27/2026, 12:15:54 PM

[Attachment: Pasted markdown.md]

## Response:
6/27/2026, 12:15:56 PM

I’ll use the uploaded bundle as the evidence base and frame the answer as an implementation memo focused on turning existing detection pieces into a corpus-first production gate.

The design is converging on auditing a frozen training manifest with source/edition-level quarantine, provenance tracing, deterministic filters, and model probes reserved for seeds or acceptance tests.

> **Designing a memo on corpus decontamination**
>
> I need to create a thorough memo with clear sections. The tough questions need answers, like whether cleaning can occur without model training—you can, but only with corpus access, metadata, and source IDs. I’ll also address technical setup such as detectors and validation. There's a focus on using 160B historical corpus, cutting off around 1929-1930.
>
> **Diagnosing inefficiency in the train-probe loop and designing pipeline**
>
> The train-probe loop is expensive, has low recall, and is unreliable. Probes don’t localize the corpus; they only target semantic regions. Despite inefficiencies, probes can still inform where to focus, help with acceptance tests, and aid in calibration. I’ll need to freeze raw data, normalize metadata, and apply strong filters.
>
> **Reviewing flagged snippets and defining detectors**
>
> I need to focus on flagged text snippets, not the entire corpus. I’ll create structured labels, distinguishing post-1930 text from older content with modern metadata. To detect post-cutoff items, I'll incorporate specific event markers, like dates or disambiguated names. I'll also target suspicious domains, especially post-1945 topics like the Cold War, WWII, and civil rights.
>
> **Handling term and entity markers for corpus filtering**
>
> I need to define event and entity markers based on how they relate; like “World War II
>
> **Managing contamination and validation**
>
> I need to ensure the tokenizer is only fed the filtered corpus and avoid post-cutoff data. I'll focus on keeping the base pretraining corpus separate from the SFT corpus by scanning prompts and responses for validation. Acceptance tests should not mix with the full model training, and I'll audit with stratified sampling across detectors.
>
> **Estimating prevalence and recall testing**
>
> I’ll estimate prevalence with weighted intervals, maybe using Wilson or beta intervals. For recall tests, I’ll inject known post-cutoff documents and sources in specific time ranges (e.g., 1931-1945, 1945-1974) into the clean corpus and measure recall. I’ll also test with marker recall and near-duplicate injections like reprints and paraphrased content.
>
> **Validating contamination and precision**
>
> I’ll test the pipeline's ability to detect contamination through temporal classifiers and historical cutoff tests, then manually audit flagged docs
>
> **Setting up contamination tiers and monitoring**
>
> I’m thinking of using a tiered system to classify sources based on contamination risk. Tier A would reject content with clear post-cutoff markers, near duplicates, or high contamination. Tier B quarantines sources with ambiguous markers or weak flags. Tier C keeps, but I’ll monitor these samples with periodic checks.
>
> **Building soft scoring and suspicious domain detection**
>
> I’m thinking about using a “soft score” approach instead of a strict classifier at first. Key rules like flagging documents based on combinations of weak evidence or setting thresholds for contamination might help. I also need to refine thresholds based on suspicious domains, like post-1945 US politics.
>
> Thought for 6m 48s

## 1. **Short Verdict**

Yes. The team can build a credible **direct, corpus-first decontamination pipeline before training another diagnostic model**. In fact, that should now be the default path.

The right claim is not “we can prove the corpus is clean.” The right claim is:

> We can freeze the corpus, run high-recall source-level decontamination, validate it with seeded injections and stratified audits, and produce a versioned training manifest whose remaining post-cutoff contamination is below a stated detectable threshold.

That is much stronger engineering than repeatedly training a model, discovering a suspicious domain, and then going back to search the corpus.

For this project, the highest-priority action is to stop treating model probes as the primary detector. Use them as **search-seed generators** and **post-cleaning acceptance tests**. The uploaded project record already points in this direction: the existing notes explicitly say the “training text is the crime scene,” identify `3_fasttext_date_classifier/` and `6_corpus_contamination_finder/` as the main corpus-first tools, and emphasize that corpus search plus retraining is the only credible certification path. `Pasted markdown.md`

---

## 2. **Why The Current Loop Wastes GPU**

The current loop is inefficient because the model is an expensive, lossy, indirect sensor of the training corpus.

A trained model can tell you that something might have leaked, but it usually cannot tell you:

- which shard contained the contamination,
- whether the source was a modern edition, OCR boilerplate, SFT data, or raw pretraining data,
- whether the signal is coherent post-cutoff factual knowledge or shallow exposure to names,
- whether the issue is one snippet, a whole document, a whole source family, or a broken date filter,
- whether a null result means clean data or merely an underpowered probe.

The project has already learned this the hard way. Average log probability was length-biased; total log probability had the opposite length problem; first-token scoring compared subword fragments; cued prompts turned tests into string completion; calibration may have broken pre-cutoff controls; and the current honest interpretation is “no evidence at tested power,” not “clean.” `Pasted markdown.md`

The better division of labor is:

| Tool | Proper role |
|---|---|
| Corpus scanners | Primary decontamination instrument |
| Metadata/source audit | Primary removal decision instrument |
| Temporal classifiers | Recall expansion and triage |
| Near-duplicate matching | Detection of copied/reprinted modern material |
| Manual audit | Calibration, precision estimate, failure discovery |
| Model probes | Search seeds and final acceptance tests, not the main cleaning loop |
| 1.7B model | Optional gate after corpus cleaning, not a prerequisite |

A model probe is still useful when it fires. But the correct reaction to “Vietnam/US politics/Cold War looks suspicious” is not “train more models to be sure.” It is:

> Convert the fired relation, answer, aliases, dates, and neighboring event vocabulary into corpus queries; search the raw corpus; aggregate hits by source; quarantine the source family if the hit is real.

---

## 3. **Direct Decontamination Pipeline**

The pipeline should produce a frozen, inspectable **training manifest**, not merely a cleaned folder. Nothing should be deleted silently. Every excluded document/source needs a reason code.

### Stage 0 - Freeze scope and definitions

Before scanning, define the contamination target.

For `Ranke-4B-1929`, I would define:

- **Allowed:** text plausibly written or published no later than 1929, including references to older entities.
- **Disallowed:** text written, revised, introduced, annotated, digitized with retained boilerplate, or factually updated after the cutoff.
- **Borderline:** pre-1930 original works in post-1930 editions with modern introductions, notes, headers, OCR artifacts, or editorial apparatus.
- **Default policy:** if modern apparatus cannot be separated reliably, quarantine the whole document or edition.

This is important because “Hiroshima” as a place name is not contamination; “Hiroshima atomic bombing” is. “Kennedy” as a surname is not contamination; “John F. Kennedy assassination” is. “United Nations” is ambiguous historically unless context establishes the post-1945 institution.

### Stage 1 - Immutable raw source inventory

Create a table like:

`raw_source_inventory.parquet`

with one row per source unit:

| Field | Purpose |
|---|---|
| `source_id` | Stable source identifier |
| `source_family_id` | Groups editions, mirrors, dataset families |
| `document_id` | Stable document id |
| `raw_path` | Corpus path/shard |
| `dataset_name` | Books, newspapers, web, IA, Gutenberg, etc. |
| `language` | Detected and metadata language |
| `genre` | Book, newspaper, government, encyclopedia, academic, OCR, etc. |
| `metadata_title` | Original title |
| `metadata_author` | Author/editor |
| `metadata_pub_year` | Claimed publication year |
| `metadata_original_year` | Original composition/publication year if available |
| `metadata_edition_year` | Edition/reprint/revision year |
| `scan_upload_year` | Digitization/upload year if present |
| `token_count` | Approximate training-token weight |
| `text_sha256` | Raw text hash |
| `normalized_sha256` | Normalized text hash |
| `source_confidence` | Metadata confidence |
| `preprocessing_version` | Exact extraction version |

Do not begin cleaning until Daniel/Dominik can answer: “What exactly went into the previous Ranke model, including tokenizer/vocab input, SFT data, and any dedup/intermediate caches?”

### Stage 2 - Normalize text, preserve offsets

Produce normalized text for scanning but keep offsets back to raw text.

Normalizations:

- lowercase copy,
- Unicode normalization,
- OCR hyphenation repair,
- whitespace collapse,
- punctuation normalization,
- optional ASCII-folded copy,
- language-specific stemming only for auxiliary matching.

Output:

`document_text_index/`

with:

- `document_id`,
- `source_id`,
- `source_family_id`,
- normalized text,
- raw offset map,
- chunk boundaries,
- paragraph boundaries,
- token count.

### Stage 3 - Run detector stack

Every detector writes to a common evidence schema:

```json
{
  "document_id": "...",
  "source_id": "...",
  "source_family_id": "...",
  "detector": "regex_disambiguated",
  "severity": "A_hard",
  "bucket": "US politics 1945-74",
  "marker": "kennedy",
  "evidence": "john f. kennedy ... assassinated ... 1963",
  "span_start": 123456,
  "span_end": 123612,
  "confidence": 0.98,
  "action_hint": "source_quarantine"
}
```

The detector stack should include:

1. high-precision disambiguated regex/Aho-Corasick markers,
2. broad year/event/entity scan,
3. temporal-style/date-of-writing classifier,
4. metadata/edition/reprint checks,
5. modern boilerplate/header/footer detector,
6. near-duplicate matching against post-cutoff reference corpora,
7. source-level aggregation,
8. active-learning classifier over audit labels,
9. optional LLM-assisted review for ambiguous snippets only.

Aho-Corasick-style multi-pattern matching is appropriate for large corpora because it was designed to locate many keywords in a single pass over text; fastText-style classifiers are also appropriate because they are simple CPU-scale text classifiers, not GPU-heavy LLMs. ([cr.yp.to](https://cr.yp.to/bib/1975/aho.pdf?utm_source=chatgpt.com)) ([ACL Anthology](https://aclanthology.org/E17-2068/))

### Stage 4 - Aggregate by source

The core mistake to avoid is treating contamination as a sentence-level property. In a historical cutoff corpus, a post-cutoff hit usually means the source pipeline admitted a bad edition, a bad collection, or a broken metadata rule.

Create:

`source_risk_ledger.parquet`

with:

| Field | Meaning |
|---|---|
| `source_id` | Source |
| `source_family_id` | Source family |
| `n_docs` | Documents |
| `tokens` | Token weight |
| `hard_hit_count` | High-precision contamination hits |
| `soft_hit_count` | Ambiguous hits |
| `temporal_max_p_post` | Max passage post-cutoff probability |
| `temporal_frac_p_post` | Fraction of passages above threshold |
| `near_dup_dirty_score` | Match to post-cutoff reference corpora |
| `metadata_conflict` | Pub/edition/original-date conflict |
| `boilerplate_score` | Modern scanner/header/footer probability |
| `audit_status` | unaudited / sampled-clean / sampled-dirty |
| `recommended_action` | keep / strip / document-drop / source-drop / family-drop |

### Stage 5 - Quarantine before final keep

Create two manifests:

`quarantine_manifest.jsonl`

High recall. Includes all hard drops and all suspicious source families.

`clean_training_manifest.jsonl`

High precision. Contains only items passing source-level rules.

Each excluded item should have stable reason codes:

- `POST_CUTOFF_METADATA`
- `POST_CUTOFF_EVENT_MARKER`
- `POST_CUTOFF_NEAR_DUP`
- `MODERN_BOILERPLATE_UNSTRIPPABLE`
- `POST_CUTOFF_PREFACE_OR_NOTES`
- `TEMPORAL_CLASSIFIER_HIGH`
- `SOURCE_FAMILY_CONTAMINATED`
- `SFT_POST_CUTOFF`
- `TOKENIZER_VOCAB_POST_CUTOFF`
- `UNKNOWN_METADATA_HIGH_RISK`

### Stage 6 - Audit, calibrate, rerun

Do not do one pass and declare victory. The loop should be:

1. scan corpus,
2. aggregate source risks,
3. manually audit stratified samples,
4. update detector dictionaries/classifiers,
5. rerun scan,
6. freeze manifest,
7. validate frozen manifest,
8. train once.

That loop is CPU/manual-audit heavy, not GPU heavy.

---

## 4. **Detector Stack**

### 4.1 Exact and regex search

Use exact search for unambiguous post-cutoff terms and disambiguated regex for ambiguous terms.

Hard markers for a 1929 cutoff include terms like:

- `Watergate`,
- `NATO`,
- `Warsaw Pact`,
- `Marshall Plan`,
- `Vietnam War`,
- `Dien Bien Phu`,
- `Tonkin Gulf`,
- `Tet Offensive`,
- `Brown v. Board`,
- `Civil Rights Act of 1964`,
- `Voting Rights Act of 1965`,
- `Sputnik`,
- `NASA`,
- `Hiroshima` near `atomic` / `bomb` / `1945`,
- `Nagasaki` near `atomic` / `bomb` / `1945`,
- `Normandy` near `D-Day` / `1944` / `Overlord`,
- `Kennedy` near `John F.` / `JFK` / `assassination` / `1963`,
- `Reagan` near `Ronald` / `president` / `1980`,
- `Israel` near `1948` / `State of Israel` / `Israeli government` / `Ben-Gurion`.

The current `find_contamination.py` already has the right instinct: bare keywords lie. `Normandy`, `Israel`, `Kennedy`, `Nagasaki`, and `Reagan` all need context windows. `Pasted markdown.md`

Use two marker tiers:

**Tier A: hard markers**

A single hit can quarantine the document or source.

Example:

```text
\bkennedy\b within 150 tokens of
    john f|j\. ?f\.|jfk|president|assassinat|1963|cuban missile
```

**Tier B: recall markers**

A hit sends the document to classifier/manual-review strata, not automatic purge.

Example:

```text
\bkennedy\b
```

alone should not drop anything.

### 4.2 Event/entity marker graph

Do not maintain only a flat keyword list. Maintain a structured marker graph:

```yaml
entity: John F. Kennedy
cutoff_status: post_1929_relation
aliases:
  - John F Kennedy
  - J. F. Kennedy
  - JFK
ambiguous_surface_forms:
  - Kennedy
required_context_any:
  - president
  - assassination
  - Cuban Missile Crisis
  - 1960
  - 1961
  - 1962
  - 1963
negative_context:
  - 19th-century surname
  - Joseph Kennedy pre-1929 only
bucket:
  - US politics 1945-74
severity:
  with_required_context: A_hard
  bare_surface: B_soft
```

This graph should cover the suspicious domains named in the project:

- post-1945 US politics,
- Vietnam,
- Cold War,
- WWII,
- civil rights,
- postwar institutions,
- science and technology,
- decolonization,
- postwar economics and institutions.

### 4.3 Date-of-writing classifiers

The existing fastText classifier direction is good, but it should be generalized for 1929/1930 and earlier cutoffs.

Use at least three cheap models:

1. fastText word/subword classifier,
2. character n-gram logistic regression or linear SVM,
3. perplexity-ratio or language-model-ratio classifier using small n-gram LMs trained on pre/post corpora.

fastText is a reasonable baseline for this role because it is explicitly designed as a fast CPU-scale text classification method; the original fastText paper reports billion-word-scale training on multicore CPU. ([ACL Anthology](https://aclanthology.org/E17-2068/))

Train not just binary `pre` / `post`, but bins:

```text
pre_1850
1850_1899
1900_1913
1914_1929
1930_1945
1946_1974
1975_plus
```

Then collapse bins according to the desired cutoff.

This matters because a binary classifier may learn “Victorian vs modern internet.” A binned classifier is more likely to learn temporal gradients.

Critical training rules:

- Balance by genre.
- Balance by topic where possible.
- Include formal post-cutoff histories, not just modern web text.
- Include newspapers, books, encyclopedias, government reports, academic prose, fiction, religious text, OCR text.
- Keep a blind test set of later histories written in old-fashioned style.
- Report per-genre and per-topic AUC.
- Use leave-topic-out validation: train without “US politics,” test on US politics; train without “history books,” test on history books.
- Use a gray zone around the cutoff. For a 1929 cutoff, do not train a hard binary on 1928 vs 1930 as if those are stylistically separable. Train with margins, then handle 1925-1935 through metadata and audit.

Classifier output should not be a drop decision by itself except at extreme scores. It is a triage signal.

Suggested document aggregation:

```python
flag_temporal = (
    passage_max_post_prob >= 0.98
    or fraction_passages_post_prob_ge_0_80 >= 0.20
    or source_mean_post_prob >= source_threshold_by_genre
)
```

### 4.4 Metadata, edition, and reprint checks

This is probably the highest-value detector after hard regex.

For every source, extract:

- claimed publication year,
- original publication year,
- edition year,
- reprint year,
- copyright year,
- renewal year,
- scanner/upload year,
- translator/editor year,
- preface/introduction year,
- series/collection year,
- filename years,
- MARC/metadata years if available,
- OCR header/footer years.

Flag patterns such as:

```text
revised edition
new edition
with a new introduction
edited by
foreword by
copyright renewed
first published ...
this edition ...
Project Gutenberg release date
Internet Archive
digitized by
OCR
scanned in
ISBN
Library of Congress Cataloging-in-Publication
```

For pre-1930 original works in post-1930 editions:

- If the modern preface/notes/header can be reliably stripped, strip and keep the original text.
- If segmentation is unreliable, drop the edition.
- If a source family repeatedly contains unstrippable modern apparatus, drop the source family.

### 4.5 Near-duplicate matching

Build a dirty reference library:

- post-1930 Wikipedia snapshots or topic pages,
- postwar encyclopedias,
- modern biographies,
- modern history textbooks,
- Project Gutenberg/Internet Archive boilerplate,
- post-1930 newspapers,
- known contaminated corpus slices,
- data dropped during the prior “Xmas” cleaning step,
- SFT/instruction datasets,
- modern public-domain mirrors with headers.

Then use:

- paragraph-level SimHash for cheap broad matching,
- 5-13 word-shingle MinHash/LSH for near duplicates,
- containment scoring for cases where a modern paragraph is embedded in an old document.

Broder’s resemblance/containment framework is the classic basis for this kind of near-duplicate document detection, and large web-corpus pipelines such as CCNet also emphasize deduplication and filtering as core preprocessing steps. ([Semantic Scholar](https://www.semanticscholar.org/paper/On-the-resemblance-and-containment-of-documents-Broder/8addb1718c2bc6bbb0d82cd1a57b41198bf65965)) ([ACL Anthology](https://aclanthology.org/2020.lrec-1.494/))

Use two thresholds:

```text
near_dup_dirty_high:
    automatic document/source quarantine

near_dup_dirty_medium:
    audit/manual-review stratum
```

Near-duplicate detection catches contamination that keyword search misses: modern histories written in formal prose, reprinted encyclopedia entries, copied web biographies, and boilerplate.

### 4.6 SFT and instruction data scan

Keep this separate from pretraining.

A base model can be clean while an instruct model is dirty. Conversely, SFT contamination can make an instruct model look contaminated even if the base corpus is acceptable.

Produce separate artifacts:

```text
pretraining_clean_manifest.jsonl
sft_clean_manifest.jsonl
tokenizer_clean_manifest.jsonl
```

For SFT data, scan both user and assistant fields. The existing `scan_sft.py` / `scan_sft2.py` design is directionally right because it distinguishes location of leakage and uses disambiguating context. `Pasted markdown.md`

SFT policy should be stricter than pretraining policy:

- Drop any instruction example requiring post-cutoff knowledge.
- Drop assistant answers containing post-cutoff facts.
- Drop modern evaluation or benchmark examples.
- Drop examples whose system/user framing encodes modern institutions, values, or factual assumptions if the goal is historically situated reasoning.
- Maintain a separate audit for instruction contamination.

### 4.7 Tokenizer/vocabulary contamination

Do not ignore the tokenizer.

A tokenizer trained on modern text may expose modern strings even if the model weights are trained on clean historical text. This is not the same as coherent factual knowledge, but it can create shallow name/string exposure.

Audit:

- tokenizer training corpus,
- vocabulary tokens,
- special tokens,
- merges containing modern entities,
- benchmark/eval artifacts included in tokenizer training,
- SFT formatting templates.

For a strict historical model, train tokenizer/vocab only from the clean pre-cutoff manifest, or at least report deviations.

---

## 5. **Source-Level Removal Policy**

The removal unit should usually be:

1. **source family**, when repeated contamination or bad metadata is found;
2. **source/edition**, when a specific edition is contaminated;
3. **document**, for mixed corpora with independent documents;
4. **passage/paratext strip**, only when modern boilerplate or preface sections are mechanically separable;
5. **sentence**, almost never, except for deterministic boilerplate removal.

I would not use sentence-level removal for substantive contamination. It leaves behind context, metadata errors, and adjacent modern material. At 160B tokens, losing some clean material is cheaper than retaining a modern source family that teaches the model post-cutoff relations.

Suggested decision table:

| Evidence | Action |
|---|---|
| Explicit metadata year > cutoff and not merely original-year metadata | Drop source/edition |
| Modern preface/notes in otherwise old work, cleanly segmentable | Strip modern section; keep original |
| Modern preface/notes, not segmentable | Drop edition |
| One high-precision post-cutoff event hit | Drop document; escalate source audit |
| Multiple high-precision hits in same source | Drop source |
| Multiple contaminated sources under same family/provider rule | Drop source family or fix provider-specific date filter |
| High temporal-classifier score only | Audit or quarantine depending on token weight |
| Near-duplicate to dirty reference | Drop document/source |
| Bare ambiguous name only | Keep unless other evidence |
| Unknown metadata + soft markers + high classifier score | Quarantine pending audit |
| SFT post-cutoff fact | Drop example |
| Tokenizer/vocab modern terms | Report separately; retrain tokenizer if strict |

For this project, I would use a conservative source rule:

```text
If a source family contributes >0.1% of total tokens and has any hard contamination hit,
do a source-family audit before deciding whether to drop the whole family.

If a small source family has any hard contamination hit,
drop it outright unless it is a clearly segmentable boilerplate case.
```

The exact threshold can move, but the principle should not: high-recall quarantine first, high-precision proof later.

---

## 6. **Validation Without Full Model Training**

You can validate the cleaning pipeline without training a full LLM. You cannot prove zero contamination, but you can estimate residual risk honestly.

### 6.1 Seeded-injection tests

Take a held-out copy of the corpus manifest and inject known dirty material at known rates.

Inject at several levels:

| Injection type | Purpose |
|---|---|
| Whole post-1945 article as document | Tests document-level recall |
| Modern preface attached to old book | Tests paratext detection |
| Modern paragraph inside old text | Tests containment and passage detectors |
| Ambiguous entity with post relation | Tests disambiguated regex |
| Ambiguous entity without post relation | Tests false positives |
| Formal 1950 history prose without obvious banned terms | Tests temporal classifier |
| OCR-corrupted modern text | Tests robustness |
| Foreign-language post-cutoff text | Tests multilingual gaps |
| Modern boilerplate only | Tests strip/drop logic |

Measure:

- document-level recall,
- source-level recall,
- token-weighted recall,
- false positive rate on known-clean pre-cutoff sources,
- recall by domain: US politics, Vietnam, Cold War, WWII, civil rights, science/institutions.

Target:

```text
Hard-marker detector:
    near-100% recall for seeded explicit post-cutoff relations in target domains.

Full pipeline:
    >99% source-level recall for injected contaminated sources in suspicious domains.

Temporal classifier:
    high recall on formal post-cutoff prose, but used mainly for triage.

Manual audit:
    calibrated precision estimate for quarantine decisions.
```

Do not overclaim. If the injected dirty material does not contain any known marker and is stylistically indistinguishable from pre-cutoff prose, no scalable detector can guarantee detection. That is exactly why metadata and source provenance matter.

### 6.2 Stratified manual audit

Sample from strata, not just randomly.

Suggested strata:

1. hard marker hits,
2. soft ambiguous marker hits,
3. high temporal-classifier score, no marker,
4. medium temporal-classifier score,
5. near-duplicate dirty hits,
6. metadata conflict,
7. modern boilerplate/header hits,
8. high-token source families,
9. random clean-manifest documents,
10. random quarantine-manifest documents.

Each audit item should ask:

```text
Is this text itself post-cutoff?
Is it original pre-cutoff text with modern apparatus?
Is the metadata wrong?
Is the hit a false positive?
Should the removal unit be passage, document, source, or family?
What exact phrase supports the judgment?
```

Use two human labels for borderline cases and adjudicate disagreements.

### 6.3 Contamination prevalence estimates

Estimate separately:

- document-level contamination,
- token-weighted contamination,
- source-family contamination,
- SFT contamination,
- modern boilerplate contamination.

For the final clean manifest, report:

```text
In a stratified token-weighted audit of N documents / M tokens,
we observed X confirmed post-cutoff contaminations.
The estimated residual prevalence is p_hat with a 95% upper confidence bound of U.
This is not proof of zero contamination; it is an upper bound at this audit design's power.
```

Be careful with “zero hits.” If you audit 1,000 clean-manifest documents and find zero contamination, the rough one-sided 95% upper bound is about 3/1000, or 0.3% at the sampled-document level. That may sound worse than people expect. The way to get stronger assurance is not rhetorical confidence; it is larger audits, token-weighted sampling, better source-level provenance, and seeded recall tests.

### 6.4 Historical cutoff holdout tests

Use earlier cutoffs as simulations.

For example:

- Pretend the cutoff is 1900.
- Treat 1901-1929 material as “future.”
- Run the detector.
- Measure whether the pipeline catches post-1900 sources without destroying legitimate pre-1900 material.

Then repeat for 1913.

This is valuable because you have real historical text on both sides of simulated cutoffs. It tests whether the pipeline can detect “future relative to cutoff” text without relying only on famous post-1945 markers.

### 6.5 Frozen acceptance battery

After the corpus is cleaned, keep a locked model-probe battery as an acceptance test, but do not use it as the main detector.

The acceptance claim should be:

```text
The cleaned corpus passed frozen corpus validation.
A model trained on the cleaned manifest showed no evidence of coherent post-cutoff factual knowledge at the tested power.
```

Not:

```text
The model is clean.
```

---

## 7. **How To Use Existing Probes As Search Seeds**

The preregistered item bank is useful, but not as a literal search list.

For every model-probe item, convert it into a relation tuple:

```json
{
  "domain": "Vietnam War",
  "post_cutoff_relation": "1968 Tet Offensive associated with Viet Cong/North Vietnamese attacks",
  "true_answer": "Tet Offensive",
  "aliases": ["Tet", "Tết", "Tet attacks"],
  "dates": ["1968"],
  "entities": ["Vietnam", "Viet Cong", "North Vietnam", "Saigon"],
  "context_terms": ["offensive", "attacks", "lunar new year", "turning point"],
  "distractors": ["..."],
  "search_patterns": [
    "tet within 100 tokens of vietnam|viet cong|1968|offensive",
    "1968 within 100 tokens of saigon|viet cong|offensive"
  ]
}
```

Do not search only for the prompt wording. That overfits to the probe artifact and misses contamination written differently.

Use expansions:

- aliases,
- dates,
- role descriptions,
- old/new place names,
- spelling variants,
- OCR variants,
- foreign-language variants where relevant,
- event-to-entity and entity-to-event reverse queries,
- co-occurrence windows,
- source-level recurrence.

Use distractors as negative controls. If both true answers and distractors appear equally often, the probe may have been measuring lexical familiarity. If true post-cutoff relations recur in corpus sources, that is real contamination evidence.

For the current uncertain v2 results, proceed as follows:

- Do not interpret the v2 model score as settled evidence, because the pre-cutoff controls underperformed.
- Still use its fired domains - US politics 1945-74 and Vietnam especially - as high-priority corpus-search buckets.
- Freeze a fresh probe set for later acceptance, but do not wait for probe certainty before cleaning.

---

## 8. **1.7B Diagnostic Model: When To Run Or Skip**

Skip the 1.7B diagnostic model for now.

Run corpus-first decontamination first. A 1.7B model becomes rational only under these conditions:

1. **The clean manifest is already frozen.**  
   Do not train a diagnostic model on a moving corpus.

2. **The 1.7B run uses the exact final preprocessing/tokenizer/training manifest.**  
   Otherwise it tests the wrong pipeline.

3. **The cost is small relative to the large run.**  
   It should be a gate, not a new cleaning loop.

4. **There is a locked acceptance battery.**  
   The team must know in advance what result would block the large run.

5. **A failure leads to corpus search, not another round of model tinkering.**  
   If the 1.7B fires on Vietnam, the next step is source search for Vietnam relations, not new probe design.

6. **It answers a question corpus tools cannot answer.**  
   For example: whether residual shallow exposure after cleaning becomes coherent factual knowledge under the final recipe.

Do not run 1.7B merely because the team is anxious. That repeats the original failure mode.

Suggested decision rule:

```text
Run 1.7B only after:
    - corpus inventory is frozen,
    - detector stack has run,
    - source-level quarantine has been applied,
    - clean manifest is versioned,
    - seeded-injection recall passes,
    - stratified audit residual upper bound is acceptable,
    - SFT/tokenizer scans are clean or separately quarantined,
    - acceptance probes are frozen.

Otherwise:
    spend effort on corpus access, scanners, metadata, source aggregation, and audit.
```

The known contaminated 0.6B model can be useful as a calibration artifact, but it should not determine the cleaning pipeline.

---

## 9. **Owner-Based Implementation Checklist**

### Daniel

Own the go/no-go decision and scope.

Concrete tasks:

- Freeze the cutoff policy for 1900, 1913, and 1929/1930.
- Decide whether the operative cutoff is `<=1929`, `<1930`, or another exact rule.
- Define what counts as contamination: original text, revised edition, preface, notes, metadata, boilerplate, tokenizer, SFT.
- Produce the decision memo for whether 1.7B is skipped until after corpus cleaning.
- Maintain the final `manifest_decision_log.md`.
- Approve the claim language: “below detectable threshold,” not “clean.”
- Make sure the contaminated 0.6B model is used only as calibration/reference, not as production evidence.

### Dominik

Own corpus access and scalable infrastructure.

Concrete tasks:

- Map the exact corpus paths on CSCS/ALPS or the relevant server.
- Produce `raw_source_inventory.parquet`.
- Implement streaming corpus readers.
- Ensure every document has `document_id`, `source_id`, `source_family_id`, token count, raw hash, normalized hash.
- Run the disambiguated marker scanner over the full 160B-token corpus.
- Implement source-level aggregation.
- Build the dirty-reference near-duplicate index.
- Produce `detector_hits.jsonl`, `source_risk_ledger.parquet`, `quarantine_manifest.jsonl`, and `clean_training_manifest.jsonl`.

### Guohui

Own classifier, audit statistics, and validation.

Concrete tasks:

- Build genre-balanced date-of-writing datasets for 1900, 1913, and 1929/1930 cutoffs.
- Train fastText and character n-gram classifiers.
- Report per-genre, per-topic, and leave-topic-out performance.
- Run the classifier over the corpus and produce passage/document/source scores.
- Compare the dropped Xmas data against random corpus samples.
- Convert overrepresented terms into scanner seeds.
- Design seeded-injection tests.
- Implement token-weighted stratified audit sampling.
- Estimate residual contamination prevalence and confidence bounds.

### Joachim

Own methodological discipline and audit interpretation.

Concrete tasks:

- Freeze the validation protocol before seeing final results.
- Define evidence grades: hard contamination, soft suspicion, false positive, modern boilerplate, metadata error.
- Review the item-bank-to-search-seed conversion.
- Ensure probes are used only as seed generators and final acceptance tests.
- Define the acceptable residual-risk claim.
- Adjudicate hard borderline cases: old works in modern editions, modern introductions, ambiguous historical names, post-cutoff commentary on pre-cutoff events.

### Shared team deliverables

The team should produce these versioned artifacts:

```text
01_raw_source_inventory.parquet
02_normalized_document_index.parquet
03_marker_dictionary.yaml
04_detector_hits.jsonl
05_temporal_classifier_scores.parquet
06_near_duplicate_hits.parquet
07_metadata_conflicts.parquet
08_source_risk_ledger.parquet
09_quarantine_manifest.jsonl
10_clean_training_manifest.jsonl
11_sft_scan_report.md
12_tokenizer_vocab_audit.md
13_seeded_injection_report.md
14_stratified_audit_labels.csv
15_validation_report.md
16_go_no_go_decision.md
```

---

## 10. **Risks, Failure Modes, And Missing Information**

### Main risks

**Hidden modern prose with no obvious markers.**  
Later histories can be formal, archaic, and marker-light. This is why metadata, near-duplicates, and source-family provenance are essential.

**Classifier learns topic, not date.**  
If post-cutoff examples are modern web and pre-cutoff examples are old books, the classifier will fail on formal 1955 histories. The project notes already flag this danger. `Pasted markdown.md`

**Over-removal distorts the historical corpus.**  
Dropping too much postwar-looking material may remove legitimate pre-cutoff text about older entities. Mitigate with source-level audit and negative-context rules.

**Ambiguous names create false positives.**  
`Normandy`, `Israel`, `Kennedy`, `Nagasaki`, and `Reagan` require context windows and relation-based matching.

**Modern boilerplate survives.**  
Scanner headers, Project Gutenberg release notes, Internet Archive metadata, OCR logs, ISBNs, and cataloging text can contaminate a model even if the book is old.

**SFT contamination is misattributed to pretraining.**  
Keep base, instruct, tokenizer, and SFT audits separate.

**Tokenizer contamination is dismissed too casually.**  
Tokenizer exposure is not coherent factual knowledge, but it can explain shallow string exposure.

**Source IDs are too coarse or too fine.**  
If `source_id` is just a shard path, aggregation is useless. If it is every tiny snippet, source-level contamination is hidden. The team needs `source_family_id`.

**Absence of detected contamination gets oversold.**  
A null audit is not proof. The deliverable should state the residual upper bound and the detector’s power.

### Missing information

The uploaded bundle does not include the actual 160B-token corpus, so the following are still needed:

- exact corpus paths,
- raw source list,
- metadata schema,
- document/source identifiers,
- preprocessing scripts used for the previous Ranke run,
- tokenizer/vocab training data,
- SFT/instruction datasets actually used,
- deduplication state,
- removed “Xmas” data,
- language composition,
- genre composition,
- source-family mapping,
- exact training manifest for existing models,
- exact cutoff convention.

Without those, no one can certify decontamination. The memo can define the pipeline, but the pipeline has to run against the real corpus.

---

## 11. **A One-Page Operational Plan**

**Objective:** produce a frozen clean manifest for the 1929/1930 historical model without training another model as the primary detector.

**Step 1 - Freeze definitions.**  
Daniel and Joachim define the exact cutoff rule, contamination categories, and acceptable claim language.

**Step 2 - Inventory the corpus.**  
Daniel and Dominik locate every raw input, including pretraining, SFT, tokenizer/vocab, caches, and prior dropped data. Dominik creates `raw_source_inventory.parquet`.

**Step 3 - Build source IDs.**  
Every document gets `document_id`, `source_id`, and `source_family_id`. No source-level cleaning is possible without this.

**Step 4 - Run hard marker scan.**  
Dominik expands `find_contamination.py` into a full detector using disambiguated event/entity markers for US politics, Vietnam, Cold War, WWII, civil rights, postwar institutions, and science.

**Step 5 - Run metadata and boilerplate scan.**  
Detect post-cutoff publication years, edition years, prefaces, introductions, revised editions, scanner headers, OCR boilerplate, upload dates, ISBNs, cataloging data, and modern file names.

**Step 6 - Run temporal classifiers.**  
Guohui trains genre-balanced fastText and character n-gram classifiers for the relevant cutoffs and scores the full corpus by passage and document.

**Step 7 - Run near-duplicate matching.**  
Dominik builds dirty-reference corpora and runs paragraph/document-level MinHash or SimHash matching against post-cutoff sources and known contaminated/dropped material.

**Step 8 - Aggregate to source risk.**  
All detector outputs feed `source_risk_ledger.parquet`. Removal decisions are source-level unless the contamination is cleanly segmentable boilerplate.

**Step 9 - Quarantine aggressively.**  
Hard post-cutoff hits, dirty near-duplicates, bad metadata, and recurring contaminated source families go to `quarantine_manifest.jsonl`.

**Step 10 - Audit strategically.**  
Guohui samples by risk stratum and token weight. Joachim adjudicates borderline cases. The team updates detectors and reruns until failure modes stop appearing.

**Step 11 - Validate without LLM training.**  
Run seeded-injection tests and historical cutoff simulations. Report document-level, source-level, and token-weighted recall/precision.

**Step 12 - Freeze clean manifest.**  
Only then produce `clean_training_manifest.jsonl`.

**Step 13 - Decide on 1.7B.**  
Run a 1.7B gate only if the manifest is frozen, validation passes, the acceptance probes are locked, and the cost is justified. Otherwise go straight to the large clean run.

**Bottom line:** yes, clean directly. The practical route is not another model-probe-first cycle. It is a reproducible corpus ledger, high-recall detector stack, source-level quarantine, seeded validation, stratified audit, and a final manifest whose limitations are stated honestly.

---

**Sources:**

- [Efficient String Matching: An Aid to Bibliographic Search](https://cr.yp.to/bib/1975/aho.pdf?utm_source=chatgpt.com)
- [Bag of Tricks for Efficient Text Classification - ACL Anthology](https://aclanthology.org/E17-2068/)
- [[PDF] On the resemblance and containment of documents | Semantic Scholar](https://www.semanticscholar.org/paper/On-the-resemblance-and-containment-of-documents-Broder/8addb1718c2bc6bbb0d82cd1a57b41198bf65965)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)