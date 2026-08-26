# Normalizing Rank-Based Contamination Tests

## Short verdict

Guohui's idea is feasible and useful, but it should be framed carefully.

The problem he identifies is real: a clean time-locked model can rank a post-cutoff answer surprisingly high for reasons that are not coherent post-cutoff knowledge. A surname, place, acronym, year, or historical-looking completion can be helped by generic frequency, tokenizer shape, model size, broad pre-cutoff familiarity, or a prompt that already points toward a class of answers. This is exactly why the existing raw rank-ratio results can make Talkie look more contaminated than Ranke while the matched forced-choice suite puts both models at chance.

The right use of a clearly modern or post-cutoff model is therefore not "divide the clean model's rank by the polluted model's rank and call that contamination." That would be too crude. The better use is as a positive-control ruler: ask how much of the modern model's relationship-binding signal is also present in the time-locked model after subtracting model-specific generic name and prompt channels.

So the recommendation is:

1. Keep matched-distractor forced choice as the primary test for coherent post-cutoff fact knowledge.
2. Add a calibrated rank-lift diagnostic as a secondary test for shallow name/string/relationship traces.
3. Normalize that diagnostic against null/shuffled prompts and a modern positive-control model.
4. Report the result as "fraction of modern-model relationship signal," not as direct proof of contamination.

## What the current folder already tells us

The current evidence has two strands that need to be reconciled.

First, the strict forced-choice evidence is reassuring. In `8_suite`, Ranke-4B-1929 gets 16/16 pre-cutoff controls but only 14/45 post-cutoff facts, 31.1% against a 25% chance line. Talkie-1930-13B gets 16/16 pre-cutoff controls and 12/45 post-cutoff facts, 26.7%. Both are statistically at chance on the current Vietnam, WWII, and Cold War bank. This is the strongest current evidence against coherent post-cutoff factual knowledge.

Second, Guohui's relationship-binding probes are picking up a real signal. In the 1930 Ranke-vs-Talkie version, Ranke's post-1930 relationship-vs-historical rank ratio is about 1.74x, while Talkie's is about 2.37x. Against bare name, Talkie's post-1930 ratio is about 5.53x while Ranke's is about 1.73x. That does not automatically mean Talkie knows post-1930 facts, because the forced-choice bank says Talkie is at chance. It means raw rank-lift is sensitive to channels that forced choice deliberately cancels: model size, generic name floor, prompt style, and historical-biographical priors.

This is exactly the design flaw Guohui is worrying about. The good news is that it can be turned into a better measurement rather than discarded.

## Why a modern/polluted benchmark helps

A modern open model gives a useful upper anchor because it should know post-cutoff facts. If the benchmark model does not show a strong relationship-binding signal on a row, then that row is probably weak, ambiguous, poorly worded, or not actually testing accessible knowledge. If it does show a strong signal, then the row has passed a positive-control test: the prompt can elicit the relationship from a model that has the relevant post-cutoff information.

That lets us separate three cases:

- The modern model has no signal and the time-locked model has no signal: the item is uninformative.
- The modern model has a strong signal but the time-locked model has no residual signal: good evidence of cleanliness on that item.
- Both modern and time-locked models show residual signal: possible post-cutoff trace, requiring a forced-choice or property dig.

But the modern model cannot be treated as "100% contaminated water" in a literal sense. It differs from Ranke and Talkie in size, tokenizer, architecture, data mixture, and instruction style. It may know some facts better than others, and it may have its own generic name priors. It is a ruler with calibration error, not ground truth.

## Recommended metric

Use log rank, not raw rank ratios, because rank is highly skewed.

For each model `m` and item `i`, compute:

```text
L_mi = log(rank_historical_prior_mi) - log(rank_relationship_mi)
```

`L_mi > 0` means the relationship prompt improves the target's rank relative to the historical-prior prompt. This is the same information currently reported as a rank ratio, but on an additive scale that is easier to average, bootstrap, and compare.

Then add shuffled/null relationship prompts. For each item, create 3-5 null prompts that look like relationship-binding prompts but should not identify the target. For example:

- Same target name cue, but a shuffled event from another person in the same broad domain.
- Same office/event template, but a wrong country or wrong year band.
- Same relationship wording, but paired with a matched distractor person.

The exact null prompt must still be length-matched per tokenizer, just as the existing Guohui code already does for bare and historical controls.

Compute the residual relationship lift:

```text
R_mi = L_true_mi - mean(L_null_mik)
```

This subtracts the generic boost from "a relationship-looking prompt with a name cue" and leaves the part specific to the correct relationship.

Now compare the time-locked model to a modern positive-control model:

```text
NormalizedSignal_m = mean_i(R_mi) / mean_i(R_modern,i)
```

Only include items where the modern model has positive residual signal and passes a basic sanity threshold. Otherwise the item is not a valid ruler.

Interpretation:

- Near 0: the time-locked model lacks the relationship signal that the modern model has.
- Between 0 and 1: the time-locked model has some fraction of the modern relationship signal.
- Near or above 1: the time-locked model has as much relationship signal as the modern model, which is serious and should trigger item-level digs.

Do not overinterpret the number as "percent contaminated data." It is a calibrated signal index.

## Concrete next experiment

### 1. Freeze an item bank

Start from `temp-guohui/name-prior-probe-ranke-talkie/ranke_talkie_probe_questions.json`.

Keep the 300 post-1930 probes and 300 pre-1930 controls, but add a version tag such as `rank_lift_calibration_v1`. Do not revise item wording after seeing scores. Keep the current automatic checks: no target-surname leakage, no 1930 rows, no duplicate person rows, no non-keep audit rows.

For a first pass, it is acceptable to use the existing politics-heavy bank because the purpose is to evaluate the normalization idea. For a paper-level claim, broaden the bank beyond office-holder tables into science/technology, medicine, space/computing, institutions/decolonization, culture, economics, and disasters.

### 2. Add null prompts

For every row, generate 3-5 null relationship prompts. The safest first version is within-domain shuffling:

- For a post-1930 politics row, borrow a relationship prompt from another post-1930 politics row, but keep the current row's name cue and target.
- Avoid reusing the same country, office, event year, or person family when possible.
- Preserve prompt length by using the existing length-matching filler machinery.

This creates prompts that have the same "historical relationship" style but should not point to the target.

### 3. Score four model roles

Score exactly the same rows and prompt types on:

- Candidate model: Ranke-4B-1929 base.
- Candidate variant: Ranke-4B-1929-1202-Instruct.
- Peer time-locked model: Talkie-1930-13B base.
- Modern positive-control model: a current open base model, preferably same tokenizer/family/size as closely as practical. If an exact same-family 4B modern base exists, use it first; otherwise use a strong modern base model and treat architecture differences as a caveat.

Use base models where possible. Instruction models can be a robustness check, but they add a separate post-training contamination channel.

### 4. Compute three outputs

For each model and group, report:

```text
raw_lift = mean(log rank historical - log rank relationship)
null_lift = mean(log rank historical - log rank shuffled/null relationship)
residual_lift = raw_lift - null_lift
```

Then report:

```text
modern_normalized_signal = residual_lift_candidate / residual_lift_modern
```

Use case-level bootstrap confidence intervals, clustered by source table or event family when possible. Report by topic/domain, not only pooled.

### 5. Add forced-choice confirmation

For any topic or item family with high residual normalized signal, run a matched forced-choice dig:

- The target must beat 3 matched distractors.
- Chance must be explicit, usually 25%.
- The prompt must not cue the answer name.
- The fact should require post-cutoff information, not just an old name or place.

This keeps the current hierarchy intact: rank-lift detects suspicious traces; forced choice decides whether they look like coherent fact knowledge.

## Decision rules

Use the calibrated rank diagnostic as a traffic-light layer, not as a final verdict.

Green:

- Candidate post-cutoff forced choice is at chance.
- Candidate residual rank-lift is near zero or less than about 20% of the modern positive-control signal.
- No topic cluster has a coherent residual signal.

Yellow:

- Candidate forced choice is at chance, but residual rank-lift is materially positive, about 20-50% of the modern signal.
- Interpretation: shallow name/string/relationship trace. Requires item-level dig before any contamination claim.

Red:

- Candidate forced choice is above chance, or residual rank-lift is large and topic-coherent, especially above about 50% of the modern signal.
- Interpretation: likely contamination or post-cutoff relationship knowledge. Run property/coherence tests and trace corpus/SFT sources.

These thresholds are deliberately heuristic for v1. They should be tightened after seeing the modern positive-control and null-prompt distributions.

## What can go wrong

The biggest risk is choosing a bad benchmark model. A modern model is not just a polluted version of Ranke. It may be larger, better trained, instruction-tuned, tokenized differently, and more familiar with Wikipedia-style office-holder prose. A ratio to that model may understate contamination if the modern model is extremely strong, or overstate contamination if it shares name-prior artifacts.

The second risk is that null prompts are too easy. If null prompts are obviously nonsense, they will not estimate the generic relationship-prompt channel. They need to be plausible historical-looking prompts that simply do not identify the target.

The third risk is first-token rank. Guohui's current relationship-binding metric uses next-token rank of the target first token. This is sensitive and cheap, but it inherits the earlier "Ken is not Kennedy" concern. For a stronger version, score whole surnames or full names with teacher-forced summed log probability where tokenization permits. At minimum, stratify by single-token vs multi-token target and show that results are not driven by tokenization artifacts.

The fourth risk is source composition. The current 600-row relationship bank is heavily shaped by office-holder tables. That is fine for a method pilot but too narrow for a general contamination claim.

The fifth risk is treating a continuous residual as proof. A positive residual says the relationship phrase helped. It does not prove the model knows the fact unless it survives matched distractors and property questions.

## How to word the result if this works

A conservative wording would be:

> We add a calibrated rank-lift diagnostic to separate generic name familiarity from relationship-specific post-cutoff signal. On each item, we subtract the lift produced by shuffled relationship prompts and scale the residual by the lift observed in a modern positive-control model. This estimates how much of the modern model's post-cutoff relationship signal is present in the time-locked model. We treat this as a sensitivity diagnostic for shallow traces, while matched forced-choice tests remain the primary evidence for coherent post-cutoff factual knowledge.

If Ranke stays near zero:

> Ranke shows little residual relationship-binding signal after subtracting null relationship prompts, despite the modern positive-control model showing strong signal on the same items. Together with chance-level matched forced-choice performance, this supports the claim that we do not detect coherent post-cutoff factual knowledge, though shallow name/string traces remain possible.

If Ranke has positive residual but forced choice stays at chance:

> Ranke exhibits a measurable shallow relationship-binding trace, but this trace does not currently rise to coherent post-cutoff factual knowledge under matched forced-choice tests. We therefore report it as a sensitivity flag, not as confirmed contamination.

## Bottom line

Guohui's normalization idea is good, but the unit should be residualized within item and then benchmarked against a modern positive-control model. The polluted model should be a ruler, not the denominator of a raw ratio. The cleanest next step is a small `rank_lift_calibration_v1` experiment that adds shuffled/null relationship prompts to the existing 1930 relationship-binding bank, scores Ranke, Talkie, and one modern positive-control model, and reports residual lift plus forced-choice confirmation for any suspicious clusters.
