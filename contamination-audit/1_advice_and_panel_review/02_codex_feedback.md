This is a strong document overall, and it correctly identifies a real problem: most tests mix up **style** (“sounds modern”) with **knowledge** (“knows post-1913 facts”). I would keep that core idea.

But several claims are too strong, and a few key risks are missing.

## 1) What is wrong or technically shaky

### “Twin sentences can’t be fooled” is not true
That is the biggest overclaim. Twin sentences are useful, but they can still be fooled by things unrelated to true knowledge.

Examples:

- **Wording effects:**  
  “Great War” may be more common with certain years in training-style text, even if the model does not “know” the fact.
- **Token splitting (how text is chopped into pieces):**  
  Some years or names are easier for the model to process than others, which can change scores.
- **Position and length effects:**  
  Even tiny punctuation or rhythm differences can move score differences.
- **Default year bias:**  
  The model may prefer round or famous years (1914, 1900, 1945) more often in general.

So twins are good, but not a proof by themselves.

### “If it prefers true twin, only knowledge can explain it” is too strong
Not only knowledge. It could reflect frequency patterns like “war broke out in 1914” being a familiar phrase shape. That still may indicate leakage, but it is not as clean as claimed.

## 2) Where it is oversold / too optimistic

### The twin test needs stricter design than described
If you do this, add these safeguards:

1. **Balance direction:** half the time put true fact first, half second.  
2. **Use multiple false twins per true twin:** not just one wrong year/month.  
3. **Match plausibility carefully:** 1914 vs 1916 is not always equally plausible to readers or model.  
4. **Pre-register scoring rule:** decide in advance how you turn “surprise” into one number.  
5. **Report by topic, not one pooled score:** WW1, WW2, Versailles, etc., separately. One big average can hide a leak in one topic.

### “Cheap and fast” may be optimistic
Writing 200–400 high-quality twin items is harder than it sounds. Bad items will drown your signal. You likely need 2–3 rounds of revision and at least one blinded reviewer (someone who doesn’t know which side is true) to check item quality.

### Error bars are trickier than implied
If many items use near-identical templates, they are not independent. Standard error bars can look tighter than truth. Group similar items together when calculating uncertainty.

## 3) Important missing ideas and risks

### Missing: direct corpus audit
Before model probing, do boring but high-value checks:

- Search raw training text for years >1913.
- Search for anchor phrases (“Treaty of Versailles 1919”, “World War II”, “Nazi”, etc.).
- Run near-duplicate matching against known post-1913 sources.

This catches blatant contamination quickly and cheaply.

### Missing: pipeline leakage checks
Was the model trained **from scratch**?  
If it started from any already-trained model, contamination is almost guaranteed.  
Also check tokenizer/vocabulary building data, evaluation sets, and any filtering models used in preprocessing.

### Missing: non-generative tests
Add a simpler forced-choice format (“Which is more likely, A or B?”) with strict templates. This reduces free-form language noise.

## 4) Is the two-week plan realistic?

Partly. I would adjust it:

- **Week 1:**  
  - corpus and pipeline audit (must-do)  
  - build 80–120 high-quality twins, not 400 yet  
  - pilot on your 1913 model + modern model  
- **Week 2:**  
  - revise items based on pilot failures  
  - expand to ~200 solid items  
  - run final scoring with topic-level breakdown  
  - only then decide if poisoning/ruler is needed

This is more realistic and less likely to produce junk numbers quickly.

## 5) “Poison on purpose / ruler” practical traps

Good idea in principle, but the document understates practical pain.

### Trap 1: small models may not behave like the big model
A tiny model may memorize differently. Your ruler curve may not transfer.  
Mitigation: use the **largest affordable practice models**, not toy ones.

### Trap 2: unrealistic poisoning recipe
Repeating a few paragraphs may create a fake “memorization fingerprint” unlike real contamination.  
Mitigation: test several contamination styles:
- repeated snippets
- diverse future text
- low-density spread across documents

### Trap 3: randomness and variance
Training runs vary by random seed. One run per contamination level is weak.  
Mitigation: at least 2–3 runs per level if possible, then report range.

### Trap 4: compute/time cost
Training “3–4 small models” can still be expensive and slow, especially with multiple runs and contamination styles. Budget this honestly before promising referee-proof bounds.

---

## Bottom line

Keep the central insight (compare matched pairs, don’t trust raw probabilities).  
But tone down claims of cleanliness and certainty.  
Add three things the current advice misses: **corpus/pipeline audit, stricter twin construction, and realistic poisoning calibration with variance checks**.

If you do those, you’ll move from “interesting test” to something a skeptical reviewer will trust.