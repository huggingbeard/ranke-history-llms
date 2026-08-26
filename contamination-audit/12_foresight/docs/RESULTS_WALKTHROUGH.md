# Foresight study — full results walkthrough (progress report)

**Files this reads from:** `scores_by_model.json`, `stats_report.json`, `gate_report.json`,
`text_perplexity.json`, `production_report.json`, `question_bank_foresight.json` (all in
`D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\12_foresight\`).
**Date:** 2026-07-03 · **Version:** v1.0 · Every number below is copied from those files (nothing hand-typed).

This is a step-by-step inventory of what we found and exactly how each number was produced, written for a
colleague who has not seen the project. The paper is the compressed version of this document.

---

## 0. The setup, in plain terms

We have three AI language models, each trained only on text written up to a cutoff year — **1900**, **1913**,
**1929** — and nothing after. Think of each as a very well-read person frozen at the end of that year. We also
use one **modern** model (Qwen2.5-3B) that has read the whole 20th century; it is our "knows-the-ending"
yardstick. All three time-locked models share the same tokenizer (verified byte-identical), which is what lets
us compare them cleanly.

**Surprise.** For any model M, context c, and continuation o, the model reports a probability P(o|c). We call
`S = −log P(o|c)` the model's **surprise** at o, measured in *nats* (natural-log units). Bigger = the model
thought o less likely. Surprise is summed over all the words of o, so numbers run to tens of nats; only
*differences* between conditions matter, never the absolute level.

---

## 1. The measuring stick: splitting "surprise at the words" from "surprise at the idea"

**The problem.** A 1913 mind is surprised by the outcome *"…millions of young men died"* for two different
reasons mixed together: (a) *"millions"* is a heavy, less-common word (lexical), and (b) maybe a war killing
millions doesn't fit its picture of the world (conceptual). Only (b) is foresight.

**The trick.** Score the *same* outcome twice — once after the **loaded** pre-shock subject `u`, once after a
**neutral** stub `n` that keeps the grammar but strips the scenario — and subtract:

```
S_cond = −log P(o | u)     (surprise given the loaded scenario)
S_marg = −log P(o | n)     (surprise given a bare frame  =  the word-cost / lexical base rate)
strain = S_cond − S_marg   (what's left after the word-cost cancels  =  the IDEA-fit)
```

Both terms score the identical words, so the cost of the rare words is the same in both and **cancels** in the
subtraction. What can't cancel is anything that depends on the *context* — and "does this outcome follow from
the scenario?" is exactly a context-dependent, meaning question. So `strain` is the pure conceptual channel.
Sign convention: **negative strain = the scenario anticipated the outcome** (context made it more likely than
the bare frame); **strain ≈ 0 = the scenario was no help** (the idea is disconnected from the worldview).

**Worked example (real numbers from `scores_by_model.json`).**

| Probe (loaded subject → outcome) | model | S_cond | S_marg | **strain** |
|---|---|---:|---:|---:|
| *"The learned economists judge that share prices have now reached"* → *"a peak from which they will presently crash"* | 1929 | 28.41 | 28.29 | **+0.12** |
| (Depression — Fisher's "plateau") | modern | 28.18 | 33.27 | **−5.10** |
| *"A war among the Great Powers"* → *"would cost the lives of many millions of young men"* | 1929 | 24.10 | 30.40 | **−6.30** |
| (World War — casualties) | modern | 20.51 | 28.28 | **−7.77** |

Read the Depression row: the **modern** mind hears "economists judge prices have peaked" and its world-model
shouts "…crash" (strain −5.10 → context makes the crash ~160× more likely than the bare frame). The **1929**
mind hears the same and does *not* connect it to a crash at all (strain +0.12 ≈ 0 — it expects Fisher's
"permanently high plateau"). Read the War row: **both** minds connect "a great war" → "millions dead" (both
strains sharply negative); the 1929 mind is nearly as good as the modern one.

---

## 2. Does the measuring stick work? (the controls)

Before trusting the stick on the catastrophes, we check it on two control sets, comparing each Ranke model's
`strain` to the modern model's: `XS = strain(Ranke) − strain(modern)`. (High XS = the time-locked mind finds
the idea far more strained than a modern mind that knows it.)

**Tier-C — "future to everyone" (moon landing, atom bomb, heart transplant, radio, chess computer).** These
lie after *every* cutoff, so *all* the minds, including 1929, should be surprised. They are — and this set is
our "how surprising is a genuinely unforeseeable thing?" benchmark. Item-level XS at the 1929 model:

| Tier-C item | outcome | XS (1929 vs modern) |
|---|---|---:|
| c_moon | "men will walk upon the face of the … **moon**" | **+2.36** |
| c_bomb | "a single bomb will destroy a whole … **city**" | +0.78 |
| c_heart | "…the living heart of … **another man**" | +2.08 |
| c_ether | "voices and pictures will fly through the air into every … **home**" | +5.11 |
| c_engine | "a machine will beat the finest masters at … **chess**" | +0.39 |
| **mean** | | **+2.14** |

So a moon landing scores **+2.36**, and the whole "unforeseeable-to-all" benchmark averages **+2.14**. Keep
these two numbers in mind — they are the yardstick for §4. (Note the spread, 0.39–5.11: individual items are
noisy; the benchmark is the *average*.)

**Tier-B — timeless "boring" facts (winter is cold; the sun rises in the east; a ship takes days to cross).**
These should carry *no* excess surprise (≈ 0). The mean XS is **+1.25**, but the item spread is enormous
(−4.98 to +10.89): with only 6 items this is a **noisy null centred loosely near zero**. Lesson we carry
forward: Tier-B is too small/noisy to use as a precision baseline, so we do **not** build effect sizes on it
(see §9).

---

## 3. Result 1 — the World War was conceptually foreseeable

### 3a. Raw surprise: the 1913 mind *looks* shocked
Excess surprise of a pre-war model over the 1929 model (which lived through the war), averaged over the 20
World-War items: `ΔS = S_cond(pre) − S_cond(1929)`.
- **ΔS(1913 vs 1929) = +2.30 nats**, 95% CI [+1.32, +3.41] → clearly positive.
- ΔS(1900 vs 1929) = +2.87 [+1.73, +3.98].
- Null check: on the Tier-C "future to all" items, ΔS(1913) = **−0.45 [−0.99, +0.11]** ≈ 0, exactly as the
  differencing is designed to give (the 1929 model is *also* surprised by the moon landing, so the difference
  vanishes). This tells us the +2.30 on WWI is real signal, not an artifact of the method.

### 3b. …but it's almost entirely about the WORDS
Split ΔS = Δlex + Δconc (see §1; `S_cond = S_marg + strain`, so the split is exact):
- **Δlex(1913) = +2.58** — the word-cost gap. Words like *"Verdun"*, *"revolution"*, *"Passchendaele"* became
  common in the 1929 model's training text but were rare/absent for the 1913 model.
- **Δconc(1913) = −0.28**, CI **[−1.25, +0.62]** — the idea-fit gap, **indistinguishable from zero**.

So the 1913 mind is not *conceptually* more strained by the war's outcomes than the 1929 mind; it has merely
not yet met the vocabulary. Benchmarked against the modern model, the same conclusion: WWI excess conceptual
strain at 1929 is **+1.10 [+0.13, +2.02]** — down at the level of the timeless Tier-B controls (+1.25), far
below the Depression (§4). (Figure `fig2_wwi_split.png` shows the blue word-bar large and the orange idea-bar
at zero.)

### 3c. "Home by Christmas" was a mood, not a limit of imagination
For the 8 duration/nature items we read the 1913 model's probability on the *true* (long-war) outcome,
`p_true` (from a two-way softmax of the calibrated scores). Sorted:

| forecast (true outcome) | p_true (1913) | leans… |
|---|---:|---|
| "…marching into four years of slaughter" | 1.00 | TRUE |
| "…drag on for years, and bleed the nations white" | 1.00 | TRUE |
| "…come home only after long and bitter years" | 0.98 | TRUE |
| "…decided only after years of ruinous stalemate" | 0.87 | TRUE |
| "…pass their days dug into ditches in the earth" | 0.70 | TRUE |
| "…grind on for years despite the ruin" | 0.57 | TRUE |
| "…one of siege and deadlock, neither side able to move" | 0.04 | wrong |
| "…the gun in a fixed line, and the spade" | 0.03 | wrong |

Six of eight sit far to the right: the 1913 mind **expects a long, costly war**. It gets only the two
**trench-deadlock tactics** items wrong (the "confidently wrong" count is 2/8; mean confident-wrongness
−0.23). So it did *not* believe "home by Christmas"; what it missed was the specific static-trench *form* of
the fighting. (Figure `fig3_homebychristmas.png`.) This fits the historical record it was trained on —
Bloch (1899) had predicted attrition and exhaustion.

### 3d. Is the low surprise just cheating? The contamination gate
A low-surprise reading only means foresight if the mind isn't secretly peeking. We test the 1913 model on
plain 1914–18 *facts* (chance = 25%). It scores **9/12 = 75%** → flagged "RED". But the item level
(`gate_report.json`) shows this is a **place-name prior**, not event knowledge:

- **Right** on things that were famous places/objects *before* 1914: Gallipoli, Jutland (a peninsula),
  Passchendaele and the Somme, Versailles (a palace), the *Lusitania* (a liner launched in 1906), Armistice
  year "1918".
- **Wrong** on genuinely new post-1914 things: the **"tank"** as a weapon (gap −7.52), the **1917 U.S. entry**,
  and even **Verdun** (its distractors were also fortress towns).

The model recognises 1914-era *words*, not the *events* — which is precisely the lexical channel (Δlex) we
already strip out. So the low *conceptual* surprise stands.

**Status: robust.** WWI raw surprise is lexical (Δconc≈0, CI crosses 0); the mind expected a long war; the
gate signal is place-priors.

---

## 4. Result 2 — the Great Depression was genuinely unforeseeable ("less likely than the moon")

### 4a. High conceptual surprise at every cutoff, not falling toward the crash
Depression excess conceptual strain vs the modern model (`XS`), by cutoff:
- 1900: **+3.60** [+2.40, +4.91]
- 1913: **+2.84** [+1.47, +4.28]
- 1929: **+3.53** [+2.23, +4.81]

Even the 1929 mind, on the eve of the crash, is deeply strained by the Depression's outcomes — and the number
does **not** fall as the cutoff approaches 1929. (Figure `fig1_main.png`: the orange line high and flat.)

### 4b. The "less likely than the moon" claim — shown, not asserted
From §2, the "unforeseeable-to-all" benchmark at the 1929 model is: **moon landing +2.36**, whole Tier-C mean
**+2.14**. The Depression's mean XS at 1929 is **+3.53 > +2.36**. Item by item, several Depression outcomes are
*far* above the moon:

| Depression item (outcome) | XS (1929) | vs moon (+2.36) |
|---|---:|---|
| dep_realwage_up ("buy rather more than before" — the *benign* one) | +8.30 | far above |
| dep_sev_family ("want, and hunger, and long idleness") | +7.42 | far above |
| dep_dur_prosperity ("return only after long and bitter years") | +6.89 | far above |
| dep_sev_fourth ("a fourth of all who labour stand idle") | +5.69 | far above |
| dep_opt_plateau ("a peak from which they will presently crash") | +5.21 | far above |
| dep_trade_walls ("choked by new walls of tariff") | +4.70 | above |
| dep_bank_fail ("fail by the thousand") | +4.40 | above |
| … (others lower; a few near zero: dep_dur_slump −0.04, dep_pol_govts −0.57) | | mixed |

**Plainly:** to a mind that had read everything up to 1929, the *average* coming-Depression outcome was more
conceptually surprising (+3.53) than a man walking on the moon (+2.36) — and its sharpest features (mass
idleness, a decade-long slump, the crash after the "plateau") were more than twice as surprising. *Honest
caveat:* individual items are noisy and a few Depression items are near zero; this is a statement about the
**averages**, and the moon benchmark (+2.36) is itself one noisy point. The statistically clean version is the
contrast in §5.

### 4c. The gate says this is real, not cheating
The 1929 model on 1930+ Depression facts: **2/8 = 25% = pure chance** (binomial p = 0.63) → **GREEN (clean)**.
It does not know the Depression. Even the two **boundary probes** on the October-1929 crash *itself* — "Black
Thursday", "the crash came in October" — are **both wrong (0/2)**. So the eve-of-crash mind is forecasting the
whole Depression, the crash included, from genuine ignorance of the outcome. (Weimar/1933 facts: 4/8 = 50%,
p = 0.11, also GREEN but noisier — see §9.)

### 4d. What the model expects instead
On the forced-choice probes the models lean to the *optimistic consensus*, not the catastrophe: Depression
calibrated accuracy is 38% (1929), 44% (1913), 25% (1900) — at or below chance for picking the true bad
outcome over the rosy twin (Fisher's "plateau", a quick rebound). It expected prosperity to continue.

**Status: robust for the mean/contrast; the "vs moon" item comparison is suggestive given item noise.**

---

## 5. The headline: Depression vs World War, and why it survives the noise

The cleanest single number is the **contrast** between the two events at the *same* model, which cancels any
"the modern model is just sharper" offset. Difference-in-differences at the 1929 model:

```
DiD = XS(Depression) − XS(World War)
    = [strain_1929(dep) − strain_1929(wwi)] − [strain_modern(dep) − strain_modern(wwi)]
```

Worked on the two example items from §1 (plateau vs millions-dead):
- 1929's own gap:  strain_1929(dep)=+0.12,  strain_1929(wwi)=−6.30 → **+6.42**
- modern's own gap: strain_mod(dep)=−5.10, strain_mod(wwi)=−7.77 → **+2.67**
- DiD (this pair) = 6.42 − 2.67 = **+3.75**

Averaged over *all* Depression and WWI items, the DiD is **+2.43 nats**, 95% CI **[+0.80, +4.08]**,
**p = 0.0026** (bootstrap, resampling items). A p-value this small means the gap is very unlikely to be an
accident of which sentences we picked. **The 1929 mind separates "the Depression is coming" from "a great war
is coming" by far more than a mind that has read both endings does — and that extra separation is the
foresight gap.**

**Status: robust — this is the paper's headline.**

---

## 6. Corroboration on real historical prose (Arm B)

Instead of our own sentences, feed the models real published paragraphs and measure **average surprise per
word** (higher = more foreign). Post-shock vs matched pre-shock, same register:

| passage (source) | 1900 | 1913 | 1929 |
|---|---:|---:|---:|
| WWI *after* — Keynes 1919 (ruin of Europe) | 3.30 | 3.14 | 3.09 |
| WWI *before* — Angell 1910 (war is futile) | 3.72 | 3.49 | 3.44 |
| Depression *after* — Roosevelt 1933 inaugural | 3.42 | 3.26 | 3.20 |
| Depression *before* — Hoover 1929 inaugural | 2.71 | 2.60 | 2.45 |

Absolute levels reflect each author's style, so we read *changes across models*:
- **WWI:** difference-in-differences (1913 vs 1929, Keynes minus Angell) = **−0.01 nats/word** ≈ 0 → the
  post-war prose is no more foreign to the pre-war mind than to the hindsight mind. Matches §3.
- **Depression:** no post-event model exists, so we ask whether the after−before gap *shrinks* toward the
  crash. It does not: the Roosevelt-minus-Hoover gap is **+0.70 / +0.65 / +0.75 nats/word** at 1900/1913/1929.
  The 1929 mind is no less surprised by depression-language than the 1900 mind fifty years removed. Matches §4.

(Figure `fig4_realtext.png`. Because these are single passages of unequal authorship, this is corroboration,
not the primary estimate.)

**Status: consistent corroboration.**

---

## 7. Recognition vs production (an honesty check, with its own caveat)

Two ways to read a mind: **recognition** = pick the more plausible of two endings (teacher-forcing);
**production** = let it *freely write* and see what it volunteers. Production is the harder test. For 8 items
we generate 6 free continuations each (1 greedy + 5 sampled), code each by keyword markers (more true-markers
than false-markers → "TRUE"), and take `frac_true` = fraction of the 6 that lean true. Averaged over items:

- recognition: 62% (1900), **75% (1913)**, 50% (1929)
- production:  0.23 (1900), **0.19 (1913)**, 0.38 (1929)

**But 0.19 is a skewed mean.** The 1913 per-item distribution is `[1.00, 0.17, 0, 0, 0, 0.33, 0, 0]` — one
item at 1.00, five at 0.00 (Figure `fig5_production.png`). The single clean signal is **war duration**: asked
"a great war will most likely last ___", all 6 continuations say *"for years / many years / some time"*
(`frac_true = 1.00`) — it never says "Christmas / months / short". On the other items the base model mostly
**rambles off-topic** (e.g. on "the banks will ___" it wanders into currency and deposits, hitting neither
"fail" nor "safe"), so `frac_true = 0` there conflates *"predicted wrong"* with *"said something irrelevant"*.

**Honest reading:** production confirms recognition ≫ production (so any teacher-forced inkling is an upper
bound, good for validity), and the one clean production signal — the war will last years — reinforces §3. The
rest of production is too noisy to interpret either way.

**Status: supportive but crude; report with the distribution, not the bald mean.**

---

## 8. Guards against artifacts

- **Drama bias:** 20 of the 57 items are valence-balanced — the *true* outcome is the *less* dramatic one
  (real wages of the still-employed rose; no Kaiser was restored; the great fleets sat idle). A measure that
  merely rewarded catastrophic language would fail these; the results hold.
- **Length bias:** the primary numbers are the horizon difference (same string scored by two models — length
  cancels exactly) and calibrated scores (each option's own baseline subtracted). Restricting the forced
  choice to the 28/57 exactly length-matched twins does not change the ordering.
- **Blind screen + preregistration:** the 57-item bank was checked by two independent referees for historical
  accuracy and period authenticity, and locked before scoring (`screen_report.json`,
  `PREREGISTRATION_FORESIGHT.md`).

---

## 9. What is noisy or unresolved (limitations)

1. **Tier-B controls are a noisy null** (n=6, XS spread −4.98 to +10.89). We therefore do *not* build SD-unit
   effect sizes on them; the headline is the item-bootstrap contrast in §5, not an effect size vs Tier-B.
2. **Item-level XS is noisy** for every battery; all headline claims are *averages* with bootstrap CIs, and
   the cleanest is the within-model DiD (§5). The "vs moon" comparison (§4b) is a mean-level statement.
3. **Weimar / 1933 is inconclusive.** Excess strain ≈ +0.5 with wide CIs spanning zero (n=10); it is a
   *compound* counterfactual (needs the Depression *and* its political sequel) four years past the nearest
   cutoff. A dedicated ~1932/1938 model is needed — **planned**.
4. **Base-model forced-choice is noisy** (even the timeless controls score only 33–60%). We lead with the
   continuous strain/XS measures; forced choice and production are secondary.
5. **Cross-family scale offset:** absolute XS mixes "the modern model is sharper" with real surprise; only
   contrasts at a fixed model (§5) are offset-free. We never interpret an absolute XS alone.
6. **Small models, modest batteries:** a null "no inkling detected" is an upper bound on foresight, not proof
   of its absence.

---

## 10. Inventory of results (one line each)

| # | Result | Number | Status |
|---|---|---|---|
| C1 | Differencing null holds (future-to-all) | ΔS(1913) Tier-C = −0.45 [−0.99,+0.11] ≈ 0 | robust |
| C2 | Tier-C = unforeseeable benchmark | mean XS@1929 +2.14 (moon +2.36) | robust (noisy items) |
| W1 | WWI raw excess surprise | ΔS(1913) +2.30 [1.32,3.41] | robust |
| W2 | …is lexical, not conceptual | Δlex +2.58; **Δconc −0.28 [−1.25,+0.62]** | robust |
| W3 | WWI conceptual ≈ timeless controls | XS@1929 +1.10 [0.13,2.02] vs Tier-B +1.25 | robust |
| W4 | 1913 expected a long war | 6/8 items p_true≥0.5; misses only trench tactics | robust |
| W5 | WWI gate "RED" = place-prior | 75% on facts, but fails "tank"/1917-entry | robust |
| D1 | Depression conceptual surprise high, flat | XS 3.60/2.84/3.53 across cutoffs | robust |
| D2 | Depression > moon-class (mean) | 3.53 > Tier-C 2.14 (moon 2.36) | suggestive (item noise) |
| D3 | 1929 clean on Depression | gate 2/8=25% p=0.63; crash itself 0/2 | robust |
| D4 | Models expect the optimistic consensus | forced-choice 25–44% | supportive |
| **H** | **Headline: Dep − WWI at 1929** | **DiD +2.43 [0.80,4.08], p=0.0026** | **robust** |
| B1 | Real text corroborates (WWI null) | DiD −0.01 nats/word | corroboration |
| B2 | Real text corroborates (Dep high, flat) | excess +0.70/0.65/0.75 nats/word | corroboration |
| P1 | Recognition ≫ production | 75%→0.19 (1913), skewed; war-duration 1.00 | supportive, crude |
| L1 | Weimar/1933 | XS ≈ +0.5, CIs cross 0 | inconclusive |

**One-sentence bottom line:** stripped of vocabulary, the World War's character was about as surprising to a
1913 mind as a timeless truth, but the Great Depression's depth was — to a mind on the very eve of the crash —
more surprising than a moon landing; the two disasters differ by +2.43 nats (p=0.003), and everything else
(gate, real text, production, controls) lines up behind that one contrast.
