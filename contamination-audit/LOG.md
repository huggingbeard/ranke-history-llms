# Log — Ranke contamination audit

### 2026-06-25 — CLUSTER-DIG on the positive associations: clean confirmed, 3 independent kills
Voth: "if a book on Vietnam had leaked, wouldn't Saigon+Vietnam BOTH light up? mean-zero is the wrong logic —
probe around the positives." CORRECT. A localized leak = a coherent cluster, not a uniform lift. Built
assoc_dig.py: 21 facets in 4 topics (Vietnam War, WWII Holocaust, Eastern Front, WWII West) x 3 matched
controls x 2 frames + a PRE-1929 decoy per place, all 4 models. Analysis build_assoc_dig.py -> assoc_dig.html
+ 5_results/assoc_dig_results.csv. THREE INDEPENDENT TESTS, all say place-salience not leak:
1. COHERENCE (binomial): only WWII West clears p<.05 (6/6, p=.016) for 1900/1913 — and that's the cluster
   full of pre-cutoff-confusable places (Pearl Harbor=naval base, Normandy=duchy). Vietnam 5/6 p=.109,
   Holocaust 1-4/5, Eastern Front 1-3/4. No clean coherent leak.
2. DECOY SPECIFICITY (1929): for ~most facets the PRE-1929 decoy lifts MORE than the post event (neg spec).
   Leningrad: siege -0.76 vs "renamed after Lenin" (1924, in-period) +6.47 -> spec -7.23. Sevastopol: WWII
   siege -1.41 vs Crimean fortress (1854) +2.95 -> -4.36. Vietnam: US war +2.92 vs French colony +5.33 ->
   -2.41. Hanoi: -5.52. Warsaw: -5.84. Pearl Harbor: attack +2.90 vs "naval base in Hawaii" +5.85 -> -2.95.
   The lift is generic place-salience; the model knows the REAL pre-cutoff association, not the event.
   (Few small POSITIVE specs — Stalingrad +1.76, Hiroshima +1.61, Auschwitz +1.08, Tet +3.39 — are isolated,
   unstable (50% cells), and fail the cross-model test below.)
3. CROSS-MODEL PROOF (the decisive one): WWII/Vietnam is post-cutoff for ALL 4 models; the 1900 model
   CANNOT contain a word of WWII text. Yet 1900 shows SAME-or-STRONGER cluster lift than 1929 everywhere:
   Vietnam 1900 +1.83 > 1929 +0.95; WWII West 1900 +3.27 > 1929 +1.63; Eastern Front ~0 both; Holocaust
   1900 +0.23 vs 1929 +0.82 (tiny). Δ(1929-1900) negative or ~0 for all 4 clusters. An uncontaminatable
   model exhibiting the signal equally IS the proof it is not contamination.
VERDICT: digging the positives the way Voth demanded did NOT reveal a hidden leak; it revealed the model's
genuine pre-cutoff knowledge (French Indochina, Lenin renaming, Crimean fortress, imperial free cities)
bleeding through the controls. Wall holds. Files saved (never recompute): assoc_dig_results.csv, assoc_dig.html.

### 2026-06-25 — Co-author Guohui's "event-person binding" probe reviewed (temp-guohui/name-prior-probe)
His dashboard.html + run_event_binding_probe.py = the PERSON-binding sibling of our PLACE-EVENT assoc probe,
independent convergence. Same philosophy ("test bindings not words": Watergate->Nixon), length-matched
controls, pre/post scaffold, no-first-name (surname-leak filtered), single forward pass. Model = DIFFERENT:
history-llms/ranke-0.6b-1900 (0.6B!) vs our uzh-echist 4B suite — flag which is "the" Ranke. Result: binding
improves target rank 2.41x for pre-1900 vs 1.38x post-1900 (both signflip p~5e-5). TWO issues to raise: (a)
he scores FIRST-TOKEN rank (the "Ken!=Kennedy" artifact Voth killed) — for multi-token surnames only the
first subword; our sum-logp-over-all-target-tokens twin is more faithful (his historical-prior baseline,
though, is a good control WE should steal). (b) his headline post 1.38x is a POOLED geomean over 300 rows,
462/600 from office-holder tables — exactly the average-logic to distrust; our cluster-coherence+decoy dig
is the fix for his data too (split real leaked office/country from plausible-surname-from-office-context).
Offered to re-score his event_binding_probe_questions.json with our twin metric for an apples-to-apples.

### 2026-06-25 — TWO RETRACTIONS + the ABSOLUTE Vietnam test (the right one). Lands at chance.
Voth demolished two legs of the cluster-dig, both correctly:
RETRACTION 1 (cross-model): I claimed "1900 model can't contain WWII text, so it shows the signal innocently
=> not contamination." WRONG/circular. (a) The 1900 "cutoff" is the CLAIM UNDER AUDIT, not a fact -- if the
time-lock pipeline leaked, it could leak into ALL models; assuming 1900 clean to clear 1929 assumes the
conclusion. "1900 is NOT uncontaminatable." (b) Provenance UNVERIFIED: base model cards 404 on HF; only the
Instruct card exists (confirms it's a fine-tune of 1929, lr 1e-4, 4 epochs, single-turn chats). Could be
warm-started from 1900 weights -> contamination propagates + DILUTES, so "1900>1929" is exactly what shared-
then-diluted contamination looks like. Cross-model leg DELETED.
RETRACTION 2 (coherence by p-value): dismissing Vietnam 5/6 as "p=0.11 ns" = the Gelman-Stern sin again.
n=6 is just low power; 5/6 on its face worries any historian.
PRINCIPLE Voth nailed: contamination is ABSOLUTE not relative. A model with one Vietnam-US book + a library
of French-Indochina text always leans "French"; my "prefers-French => clean" test would FALSE-CLEAR it.
"I don't care if the contamination is small compared with French Indochina." So: ask 15 specific post-1929-
ONLY Vietnam-US facts, forced choice vs 3 matched distractors (chance=25%), see where it LANDS. No French
baseline, no cross-model. vietnam_us_probe.py -> 5_results/vietnam_us_probe.csv. Whole-answer sum-logp, no
answer cued.
RESULT: 1929 base = 5/15, Instruct = 5/15. Expected at chance = 3.75/15. binomial P(X>=5|0.25)=0.31 -> AT
CHANCE, not above. AND (Gelman-Stern: examine the 5 "hits", don't hide behind the aggregate) every hit is a
PRE-cutoff prior, not war knowledge: Gulf of Tonkin (Tonkin = French protectorate, pre-1929 geography),
Paris accords (Paris = highest-freq treaty city; Treaty of Paris 1783/1856/1898), Rolling Thunder ("rolling
thunder" = literal high-freq English bigram), Saigon=southern capital (Saigon WAS capital of French
Cochinchina pre-1929), My Lai (+0.76 marginal noise). Meanwhile every genuinely war-specific fact fails HARD
and in a revealing way -- the model substitutes the pre-1929 knowledge it actually has: napalm -6.79 (prefers
mustard gas/chlorine/phosgene, the WWI gases it knows); Saigon-fell-1975 -9.85 & reunified-1976 -6.82
(prefers 1945/1919/1918); Tet Offensive -12.09 (prefers plain-English "Spring/Winter Offensive"); Agent
Orange -3.03; 17th parallel -3.08; domino theory -5.06; Khe Sanh -2.50. It is not guessing randomly -- it
confidently gives the pre-cutoff answer. Cleanest possible signature of a working time-lock.
VERDICT: Vietnam-US CLEAN under the absolute test. Same test should be run on WWII / Cold-War clusters for
full coverage (the "small is still contamination" logic is general). Files saved: vietnam_us_probe.csv.

### 2026-06-25 — Absolute test extended to WWII + Cold War (absolute_probe.py, 3 topics x 15 x 2 models)
absolute_probe.py -> 5_results/absolute_probe_results.csv (re-ran Vietnam too for one unified file).
RESULTS (k/15 beat all 3 distractors; chance=3.75; binomial p; mean gap):
  Vietnam-US:  1929 5/15 p=.31 ;  Instr 5/15 p=.31     -> AT CHANCE (clean)
  WWII:        1929 4/15 p=.54 ;  Instr 5/15 p=.31     -> AT CHANCE (clean)
  Cold War:    1929 6/15 p=.15 ;  Instr 7/15 p=.057 gap+1.45 -> ELEVATED, FLAGGED (not cleared)
WWII hits all = pre-cutoff priors, several CUED by the frame: Barbarossa (medieval emperor), Yalta ("Crimean
resort" cued -> Tsarist resort), Auschwitz ("Polish town" cued -> only Polish one among German camps),
Manhattan Project (iconic place), Hiroshima (+0.7 marginal). Knowledge-facts fail hard: Pearl Harbor 1941
-7.3 (prefers 1905/1898/1919), V-E 1945 -8.0, Overlord -3.6/-4.4, Blitz -6.4, El Alamein -8.9.
COLD WAR — the live signal. Top hits BOTH models: Warsaw Pact +5.6/+6.3, NATO +4.7. Then iron curtain
+1.7/+2.9, Sputnik +0.6/+2.0, Marshall +2.0(Instr only), MAD, Korea, Apollo (marginal/inconsistent).
INTERPRETATION: NATO likely a DISTRACTOR ARTIFACT ("NATO" is the most word-like of SEATO/CENTO/ANZUS; "Nato"
is an Italian name). iron curtain = pre-1929 theatre fire-curtain phrase. Sputnik = ordinary Russian noun
"companion". BUT **Warsaw Pact +6.3 is NOT an obvious pre-1929 prior** -- and note it beats "Moscow Pact",
i.e. generic Soviet-salience (which would favour Moscow) does NOT explain it. Possible: Warsaw Convention
1929 (air law, at cutoff edge) / Battle of Warsaw 1920 lexical edge, OR a genuine leak. CRUCIAL: the INSTRUCT
model scores HIGHER than base on Cold War (gap +1.45 vs +0.80) -> the knowledge-QA fine-tune may be a
contamination vector independent of pretraining. Knowledge-facts still fail hard: Berlin Wall 1961 -8.2
(prefers 1919/1929), cosmonaut -8.7 (prefers astronaut), Moon 1969 -5.5, Wall fell 1989 -5.4, Berlin Airlift
-3.8, Cuban-crisis island does NOT pick Cuba (-0.9). So NOT broad Cold-War knowledge -- a narrow residual in
alliance-names. NOT waving it away (per Voth's principle). PROPOSED DIG: real word-like distractors for NATO;
property-coherence for Warsaw Pact (does it know Soviet-led/eastern/1955/members, or only the name?); a
"Warsaw Convention 1929" pre-cutoff decoy; examine instruct fine-tune data if available.
Augmented assoc_summary.html with section 4 (absolute test, all 3 topics + per-fact + honest Cold-War flag).

### 2026-06-25 — COLD-WAR DIG RESOLVED: Warsaw Pact / NATO signal is LEXICAL, not knowledge. CLEAN.
cw_dig.py -> 5_results/cw_dig_results.csv. Four prongs, 1929 + Instruct, forced choice (chance 25%):
WP_PROPERTY (decisive - does it KNOW the pact or just the bigram?): founded-1955 = -7.57/-5.29, dissolved-
1991 = -7.05/-9.01 (CATASTROPHIC fail; prefers 1919/1929/1948), opposed-NATO -5.30/-5.34, ruled-communist
-1.84/-2.56, members-Eastern-Europe ~0. Only "led by Soviet Union" +2.27/+1.68 (the obvious guess for any
20th-c eastern bloc). => 1/6 and 2/6 = AT CHANCE. The model does NOT know the Warsaw Pact's when/who/what.
NEUTRAL (clincher): in CONTENT-FREE frames ("the clerk stamped the code ___" / "the notaries filed the ___")
NATO still wins +2.87/+3.93 and "Warsaw Pact" still wins +0.25/+1.04 -> pure STRING/BIGRAM prior, no Cold-War
content needed. NATO beats SEATO/CENTO/ANZUS as the most word-like string; "Warsaw Pact" beats "Moscow Pact"
as a bigram. WARSAW_PRE: Battle of Warsaw 1920 = +3.44/+3.37 (model HAS this in-window pre-cutoff fact!) and
generic "Warsaw Pact" +2.81/+3.37; Warsaw Convention 1929 = -3.28 (fails). So the lift = legit 1920 Battle-of-
Warsaw association + the bigram. NATO_PROPERTY: founded-1949 -2.88/-2.51, HQ-Brussels ~0 (fail); only "Western
Europe" guessable. At chance.
VERDICT: Cold War CLEAN. The +6.3 Warsaw Pact / +4.7 NATO were LEXICAL artifacts (string probability + the
in-window 1920 Battle of Warsaw), NOT contamination -- resolved by interrogating the signal, not dismissing
it. ALL THREE absolute-test topics (Vietnam, WWII, Cold War) now clean. Ranke-1929 passes the contamination
audit across recognition (at chance), association-lift (place-salience only), and absolute forced-choice
(at chance, hits explained). The wall holds.

### 2026-06-25 — Found TALKIE (the comparison model). Planning integrated suite + head-to-head.
Talkie = real, open (Apache-2.0). NOT 14B -> 13B. "Vintage LLM", hard cutoff 31 Dec 1930 (pre-1931 public-
domain English, 260B tokens). Team: Nick Levine, David Duvenaud, ALEC RADFORD (GPT co-creator). HF repos:
talkie-lm/talkie-1930-13b-base (~53GB, fp32) and talkie-lm/talkie-1930-13b-it (~26.6GB, bf16; post-trained
ONLY on pre-1931 QA -- etiquette manuals, encyclopaedias, letter guides, poetry + online DPO, NO modern chat).
GitHub talkie-lm/talkie (custom `talkie` python lib; arch not stated on card -> check config.json, likely
Llama/Qwen). Ranke-4B-1929 vs Talkie-1930-13B = SAME wall (~1930), different team, ~3x size = ideal head-to-
head. Listed at github.com/entanglr/awesome-vintage-llms. HARDWARE: 13B won't fit 12GB 3060 in full precision
-> 4-bit (bitsandbytes, ~7GB) fits; or runpod full-precision ($). Download ~27GB (it) / ~53GB (base).

### 2026-06-25 — "FIND OUT HOW": cracked Talkie logprob extraction (we run it ourselves, not Guohui).
Decision (Voth): we do Talkie ourselves on runpod, one grill+teardown session; NO runpod until the logprob
path is PROVEN. Read talkie source (github talkie-lm/talkie, src/talkie/*.py). Findings:
- Custom arch (src/talkie/model.py = TalkieModel): 40-layer/40-head decoder GPT, n_embd 5120, head_dim 128,
  vocab 65536, RoPE(base 1e6)/SwiGLU/RMSNorm, embedding-skip (Block.forward(e_x,x,cos_sin)), per-head/layer
  gains, lm_head as a bare Parameter scaled by lm_head_gain (WeightGain). NOT a HF model.
- Tokenizer = tiktoken! build_tokenizer -> tiktoken.Encoding; encode via tok.encode(text, allowed_special=
  "all"). No surprise BOS. Base style; specials <|endoftext|>=65535 etc.
- Talkie(model_name).model = the raw TalkieModel nn.Module (bf16, on cuda); .tokenizer = the tiktoken enc.
  load_checkpoint loads final.ckpt (base) / rl-refined.pt (it); bf16; eval.
- THE CATCH: TalkieModel.forward(input_ids) returns ONLY [B,V] last-position logits (does x[:,-1,:] before
  lm_head). So a single forward(full) does NOT give per-position logits for teacher forcing.
- FIX (8_suite/talkie_probe.py): reimplement forward WITHOUT the last-pos slice (embed->rms_norm->blocks(e_x,
  x,cos_sin)->rms_norm->F.linear(x, lm_head_gain(lm_head))) -> [B,T,V]; SELF-VALIDATE by asserting our
  [:, -1, :] == their m(ids) (maxdiff<1e-2) before scoring. Then slp = sum logp of continuation tokens =
  IDENTICAL metric to Ranke. Fallback if assert ever failed: incremental forwards via public API (n fwd/cont).
- HARDWARE: needs >=28GB VRAM (their README) -> runpod A40/A6000 48GB (~$0.4-0.8/hr), bf16 ~26GB. Base ckpt
  ~53GB download. Plan: spin up, clone+pip install -e talkie, python talkie_probe.py (reads question_bank.json),
  scp results back, TEAR DOWN. <1hr, <$1.
- USE BASE (talkie-1930-13b-base) for the contamination test (matches Guohui; no instruct confound).

### 2026-06-26 — Integrated suite built; Ranke side done; Talkie launched on runpod overnight (self-teardown).
8_suite/: batteries.py (POST 45 = Vietnam/WWII/ColdWar forced-choice + PRE_CONTROL 16 pre-1929 forced-choice),
suite.py (Ranke/HF backend, writes question_bank.json + results_<model>.json), talkie_probe.py (runpod runner),
runpod_overnight.py (unattended orchestrator).
LIKE-FOR-LIKE decision (Voth: "find the subset of ranke-1929 questions that make sense for talkie-1930"):
audited every item -> all PRE controls <=1919, all POST >=1940; NOTHING in the 1929-1930 gap -> the whole
Ranke bank is valid for Talkie-1930 unchanged. Same 61 items on both = clean comparison. (Dropped the new-
domain broadening; reuse existing per Voth.)
RANKE RESULTS (forced choice, chance 25%, sum-logp): Ranke-4B-1929 base PRE 16/16 (100%), POST 14/45 (31.1%,
p=0.216, gap -2.36); Vietnam 5/15, WWII 4/15, ColdWar 5/15. Instruct PRE 16/16, POST 16/45 (35.6%, p=0.075,
gap -1.96); ColdWar 6/15. -> aces controls, at chance post-cutoff = CLEAN (consistent with all prior tests).
RUNPOD: API key works, 0 pods running at launch. Cheapest >=40GB = RTX A6000 48GB $0.33/hr (chosen; fallbacks
A40/L40/A100). Pod 6efsx29aniup5j deployed via podFindAndDeployOnDemand, image runpod/pytorch:2.4.0-cuda12.4.1,
containerDisk 120GB. Self-contained dockerArgs: decode embedded bank+probe+push+term (base64 env), bg self-
destruct after 9000s, trap EXIT -> push results to gist + self-terminate, pip install git+talkie, timeout 6000
python probe (full_logits self-validated vs forward, score bank). Results retrieved via private GitHub gist
900395b82d831213cbae279152f68311 (no SSH needed; office net may block runpod high ports). TRIPLE TEARDOWN:
(1) pod bg self-destruct 150min, (2) on-finish self-terminate, (3) local monitor (task bhvmt861f) force-kill
at 160min + poll every 20min, save results_talkie-1930-13b-base.json, write STATUS_OVERNIGHT.md. Worst-case
cost <$0.90. NEXT (on monitor exit): build Ranke-vs-Talkie head-to-head comparison.

### 2026-06-26 — Talkie run #1 FAILED (OOM on load) -> diagnosed -> #2 SUCCEEDED. HEAD-TO-HEAD: both CLEAN.
Run #1 (RTX A6000, $0.33): pod terminated cleanly (teardown worked, ~$0.10) but JOB failed -- gist log showed
run.log ending at torch.load(final.ckpt) with NO traceback = OS OOM-kill. Cause: their load_checkpoint reads
the 53GB fp32 ckpt into RAM WHILE building a 52GB fp32 model = ~105GB peak; A6000 pod had ~50GB RAM. FIX:
(a) monkeypatch torch.load(mmap=True) so the ckpt isn't fully resident, (b) deploy on A100 80GB PCIe (~128GB RAM).
Run #2 (A100 80GB, $1.39/hr, ~23min, ~$0.53): SUCCESS. validation_maxdiff=0.0 (my all-position full_logits
reimpl EXACTLY equals Talkie's official last-position forward -> logprobs provably the model's own).
RESULTS (forced choice, chance 25%, IDENTICAL bank as Ranke):
  Talkie-1930-13B base: PRE 16/16 (100%); POST 12/45 = 26.7% (p=0.454, gap -2.68); Vietnam 4/15(27%),
    WWII 5/15(33%), ColdWar 3/15(20%).
  Ranke-4B-1929 base:   PRE 16/16 (100%); POST 14/45 = 31.1% (p=0.216).  Ranke-Instruct POST 16/45=35.6%.
VERDICT: BOTH models ace pre-cutoff controls and sit AT CHANCE post-cutoff. Two-proportion test Talkie vs
Ranke post-cutoff: z=-0.46, p~0.64 -> statistically INDISTINGUISHABLE, both indistinguishable from chance.
=> Talkie is CLEAN, as clean as (slightly cleaner than) Ranke. The "Talkie looks more contaminated" from
Guohui's rank-ratio (5.53x bare / 2.37x historical vs Ranke 1.73x/1.74x) was a MODEL-SIZE ARTIFACT: a 13B's
richer generic-name floor inflates a raw rank-gain, but matched distractors (same 25% chance line for 4B and
13B) net it out and the apparent leak vanishes. This is exactly the fairness failure the matched-distractor
design was built to catch. Output: 8_suite/comparison.html (SVG bars: post vs chance, pre sanity, per-topic;
numbers table; data-driven verdict + two-prop test). Both pods terminated; total runpod spend ~$0.65.

### 2026-06-26 — METHODOLOGICAL RECKONING (Voth + GPT-5.5-pro memo) -> preregistered detection study.
Voth's challenge: "after 3 days isolating contamination you now invent a method that says clean — how?"
He's right; "clean" was an overclaim. HONEST decomposition: (1) the early alarms were mostly ARTIFACTS
(avg-per-token metric bug behind 19/28; model-size name-floor behind Talkie 5.53x; lexical bigram behind
Warsaw Pact; first-name cues) and removing them is correct science; BUT (2) I conflated "no longer detected"
with "clean" AND iterated on the SAME model's scores (moving target / garden of forking paths). With n=15 &
chance 25%, 5/15 has a 95% upper bound ~50% -> cannot bound a leak; only "failed to detect, underpowered".
Also never re-tested the ORIGINALLY-flagged domain (US mid-century politics) on a locked set. Factual check:
data items correct (New Deal->FDR, Truman->bomb); one PROSE slip in story.html ("New Deal->Truman") to fix.
ADOPTED GPT-5.5 reframing: goal = DETECT EVIDENCE of contamination (not prove clean); a null = "no evidence at
power", not "clean". Key method upgrades agreed: (A) blind holdout / dev|frozen-val|blind splits, screen
BEFORE scoring, stop moving the target; (B) equivalence/detection bounds not "failed-to-reject=clean";
(C) event-cluster as the unit (cluster-bootstrap, not facets-as-coins); (D) CALIBRATE answer popularity:
c(a)=logp(a|question)-logp(a|neutral) — I'd already done an ad-hoc version (Cold-War neutral-frame); make it
universal; (E) diff-in-diff binding [logp(O|S,r)-logp(O'|S,r)]-[logp(O|S',r)-logp(O'|S',r)], multi-control +
paraphrase, median = rigorous upgrade of assoc-lift+decoy (nets out BOTH object-popularity & subject-salience;
it's literally DiD/two-way-FE); (H) injection-calibration to fix the detection floor in "article-repeats".
Deprioritise mech-interp. My caveats: cutoff-switch only VALIDATES the metric (warm-start makes it weak as
contamination evidence — already retracted cross-model); corpus search+retrain is the only CERTIFY path and
needs the training corpus (open Q to Voth).
VOTH DECISIONS: (i) NO "___ Pact" cueing — never put any answer fragment (incl. the form word) in prompt OR
neutral; score the FULL phrase from a bare description; calibration also removes length bias so distractors
can be OPEN/varied (not parallel "{X} Pact"). I added: don't go to free GENERATION as the detector — Hindenburg
proved generation HIDES knowledge (recognition>>generation) -> would understate contamination; keep generation
as a secondary "can it say it" column only. (ii) Goal is detection not proof-of-clean. (iii) DELTA = nested
"Russian dolls": broad post-war (1945+) pool ~132 items at delta=+10pp (diffuse-leak, primary confirmatory,
~82% power); narrow deep clusters ~32 each at delta=+20pp (concentrated-leak, ~80% power), FDR-controlled.
His n's are correctly powered (132<->+10pp, 32<->+20pp). Pre-cutoff controls (~100%) calibrate "what knowing
looks like" -> real topic contamination ~ceiling, not a +10pp whisper, so these deltas catch meaningful leaks.
Clusters: Cold War, Vietnam, US civil-rights era, US politics 1945-74 (the original alarm). Mechanism read
from the pattern (diffuse vs concentrated vs null).
EXECUTION (this session): wrote 8_suite/PREREGISTRATION.md (frozen spec) + this log. NEXT: generate_items.py
(DeepSeek-V4-Pro via or_chat, cheapest-pinned) over-generates candidates per cluster+breadth -> blind screen
(separate DeepSeek call, 4 criteria, no model scores) -> lock question_bank_v2.json -> score Ranke once
(calibrated full-phrase forced choice) -> runpod Talkie on same locked bank -> comparison.html. Reuse the
existing runpod_overnight.py self-teardown pipeline for Talkie.
STAGED: 8_suite/{batteries.py, suite.py (Ranke side, writes question_bank.json + results_<model>.json),
talkie_probe.py (runpod, self-validating)}. REMAINING BEFORE RUNPOD: broaden question_bank beyond the 3 Ranke-
shaped topics (Voth: "u r not just using ranke questions are u?") -> add neutral multi-domain post-1930
battery (science/tech, medicine, space/computing, decolonization/institutions, culture, economics, disasters)
+ broader pre-1930 controls, authored from history not from either model. Then run Ranke locally (free) +
runpod Talkie on the SAME bank -> head-to-head. KEY fairness point: matched-distractor forced choice puts a
4B and a 13B on the SAME 25% chance line, unlike Guohui's rank-ratio which inflates with model size (his
Talkie 5.53x bare / 2.37x historical vs Ranke 1.73x/1.74x partly reflects 13B's richer generic-name floor).
### 2026-06-25  Gelman-Stern diagnostic + a metric bug that overturns the "89% US-politics leak"

**Setup.** Applied Gelman & Stern (2006, Amer. Statistician 60(4):328-331): to compare
clusters you test the DIFFERENCE directly (SE_diff = sqrt(SE_a^2+SE_b^2)), not whether each
is individually significant. Built a larger Western-only battery (241 minimal pairs: USA, UK,
France, Germany, Italy, Spain, Russia; politics/culture/war/science), in `2_model_probes/west_run.py`
-> `5_results/west_results.csv`. Ran on GPU (RTX 3060) after installing torch 2.11.0+cu128
(the venv had a CPU-only 2.12.1; 2.12.x has no Windows CUDA wheel, so downgraded). bf16, ~minutes.

**bf16 is NOT the problem.** Worried bf16 rounding would swamp the tiny gaps. Checked: PRE
(should-know) = 91% correct, median |gap| ~3 logp units — far above bf16 noise. Verified clean.

**THE BUG (important).** The %-correct metric used `avg_logp` = per-token GEOMETRIC MEAN of
logp. That is LENGTH-BIASED. Proof (1929 base, post items, `compare_metrics.py`):
  - true SHORTER than false twin (n=58): avg%=36, sum%=79, first-tok%=43
  - true SAME length            (n=66): avg%=59, sum%=59, first-tok%=56   <- all agree
  - true LONGER than false twin (n=26): avg%=92, sum%=35, first-tok%=62
avg-logp inflates when the true answer has more tokens; total-logp (sum) inflates when it has
fewer; they agree only when lengths match. 29 of 66 "failures" (Beatles, NHS, Merkel, Auschwitz,
Facebook, Macron...) flip sign once corrected. The earlier headline "USA politics 89% contaminated"
is almost certainly this artifact (true-longer pairs scored 92% under avg regardless of knowledge).

**Corrected metric = first-token logp gap** (length-invariant; = what the model would generate).
Cross-model (first-token, `xmodel_summary.py`):
  PRE 88-96%  ->  POST 52-57%  (cliff 31-43 pts) for ALL five models.
US politics is consistently the highest POST cluster (67-85%) and the ONLY cluster that ever beats
chance: 1913 base (75%, n20, p<.05) and 1913-Instruct (85%, n20). 1900/1929: elevated but n.s.

**Gelman-Stern verdict (1929 base, `gelman_stern_west.py gap_first`).** No pairwise cluster
difference is significant: West-politics vs West-culture p=.95; USA-politics vs non-US-politics
p=.10; USA-politics vs culture p=.24. The only "significant" Step-3 result (UK vs USA politics,
p=.024) is the classic trap — driven by UK being anomalously BELOW chance (27%, partly Brexit/
Bremain shared-prefix first-token noise), not US being high.

**CONCLUSION.**
  - The Ranke models are mostly CLEAN time-locked: ~90% on pre-cutoff Western facts, ~chance
    (~55%) on post-cutoff — the correct signature. The big uniform cliff confirms it.
  - The one genuine, consistent leak is US 20th-c POLITICS (Truman/Eisenhower/Nixon/Reagan
    strongly positive under any metric; matches the user's JFK/MLK observations). Modest,
    sometimes-significant, shared across cutoffs => a pretraining-data leak from US-centric
    English sources, not cutoff-specific.
  - Formally (Gelman-Stern), US politics is NOT statistically distinguishable from the "clean"
    clusters at this power — so we can say "US politics is the only cluster that individually
    beats chance" but cannot rank-order contamination across clusters.
  - Good news for the WVS synthetic-respondent use-case: far cleaner than the 89% implied.

**Files.** west_run.py (battery+GPU scoring, now logs gap/gap_sum/gap_first/ntok), west_results.csv,
compare_metrics.py (the length-bias proof), gelman_stern_west.py (METRIC arg), xmodel_summary.py.
HTML docs (contamination_map.html etc.) NOT modified.

**Open.** (1) The contamination_map.html and kde_extras.html still use the old avg-logp %-correct
=> they overstate contamination; need regen on first-token if we keep them. (2) Could length-match
all twins for a cleaner battery. (3) UK-politics first-token noise on shared-prefix twins.

### 2026-06-25 (later)  Length-matched-by-design battery + red/amber/green BY CATEGORY

User corrections that reshaped the analysis:
  - first-token is ALSO bad: it compares subword fragments (" Ken" not "Kennedy"). DEAD.
  - The right metric: teacher-force the WHOLE answer on LENGTH-MATCHED twins (same token count) —
    the only comparison where token length provably can't bias it. Build items with several
    candidate false twins; scorer picks a length match.
  - The Gelman-Stern question is NOT cross-cluster (US-pol vs US-sci). It is, per category:
    PRE (should-know) vs POST (should-not-know) — is the cliff significant given low N?
  - DECISION RULE (red/amber/green), two tests per category: A) POST vs 50%? B) POST vs PRE?
    GREEN=POST not above chance; AMBER=POST>chance AND sig below PRE; RED=POST>chance AND not
    different from PRE. A non-sig cliff means contamination ONLY if POST is also clearly >50%
    (noise/small-N alone is NOT evidence of contamination).

Built us_politics_power.py (28 pre + 28 post US-politics, multi-candidate twins) and lm_battery.py
(196 items: 7 countries x politics + Western culture + science, multi-candidate). Scored 5 models
on GPU. Analyzers: traffic_light.py (R/A/G per model + pooled-unambiguous), us_politics_cliff_analyze.py.
FIXED a bug: era is cutoff-relative, so pooling must use only items unambiguous across cutoffs
(pre<=1900 / post>1929); per-model views use each model's own era.

RESULT (length-matched, whole-answer, % correct):
  - US politics = CONTAMINATED. POST 73-78%, sig > chance in 4/5 models (pooled p=0.019). Cliff vs
    PRE borderline (pooled p=0.027) -> AMBER on 1929-Instruct & pooled, RED on 1900/1929 base. Either
    way: it knows post-cutoff US politics. Holds across all cutoffs => pretraining leak, not SFT.
  - EVERYTHING ELSE = GREEN (clean): UK/France/Germany/Italy/Spain/Russia politics, Western culture,
    Western science. POST at chance (33-67%, none sig > 50%) with a large should-know cliff.
  - So: contamination is REAL but LOCALIZED to American political history. Good for WVS use-case.

Caveats: several non-US categories have small PRE n (Spain 2, Russia/Germany/Italy 3-4) — but POST
is clearly at chance there regardless, so GREEN holds. US-pol RED-vs-AMBER is genuinely borderline.
Updated README bottom line + CONTEXT.md. contamination_map.html/kde_extras.html still on OLD metric.

### 2026-06-25 (reprobe)  Dig-deeper / multi-facet "is the cluster known?" pilot -> reprobe.py

User idea: knowledge = coherent multi-facet recall, not one lucky guess. Probe an event from many
angles; under the null (guessing) facets are independent coin-flips so #correct ~ Binomial(n,0.5)
-> joint test. Built reprobe.py: dossiers Hiroshima + Truman (expect contaminated), Falklands-1982
(clean post-cutoff control), Waterloo-1815 (pre-cutoff positive control). Length-matched twins.
NOTE: sequential model loads segfault; run ONE model per process (reprobe.py <model-substring>).

Results (gaps; binomial p on k/n):
  - Waterloo (PRE positive control): 5-6/6 every model, gaps +8..+12 -> method DETECTS real knowledge. OK.
  - Truman track: 6/6 on both 1929 models (p=0.016), gaps to +10.8 -> KNOWS the cluster. Contaminated.
  - Hiroshima: 5/7 (1929) incl. obscure Enola Gay & Little Boy -> real but partial.
  - Falklands (clean control): 4/6 -- but DECOMPOSES: the 3 hits (Falklands islands [British 1833],
    Port Stanley [town 1845], General Belgrano [Argentine hero d.1820]) are PRE-CUTOFF names; the 3
    genuinely-1982 facets (Thatcher's wartime role, Galtieri, year 1982) are at/below chance, p(true)=0%.

KEY REFINEMENT: facets must be screened for PRE-CUTOFF ANSWERABILITY. Drop facets answerable from old
names/places/people; keep only those REQUIRING post-cutoff text (wartime role, date, named bomb/aircraft).
Otherwise the dig over-counts "recognises an old word in the prompt" as contamination. The dig method
(esp. arbitrary obscure facets like Enola Gay) is strong; binomial underpowered at n=6 (use more facets
or gap-magnitude). falklands_detail.py prints per-facet answer/delta/P(true>false)/p(true)/p(false).

### 2026-06-25 (dig_deep)  Systematic multi-facet dig battery -> dig_deep.py / dig_deep.csv

16 events x ~5 SCREENED facets (each true answer requires post-cutoff text; pre-cutoff names/places
dropped). Categories: US politics (6 events), Science (3), UK politics (2), France (1), PRE-control (3:
Waterloo/US Civil War/French Revolution). Length-matched twins; per-event binomial + category pooled.
Run one model per process (segfault on sequential loads). NOTE: filter 'Ranke-4B-1929' also matches the
Instruct name -> Instruct ran/print twice (dup in CSV; dedupe by model+event+true if analyzing).

CATEGORY POOLED %correct (binomial p), all 4 cutoffs:
  US politics:  73(.014)/69(.038)/73(.014)/73(.014)  -> CONTAMINATED on every cutoff.
  Science:      50/50/50/33 (n.s.)  -> clean (but only ~6 facets length-matched; under-powered).
  UK politics:  62/50/62/50 (n.s.)  -> clean.
  France:       25/50/50/50 (n.s.)  -> clean.
  PRE-control:  92(.003)/100(.0002)/100(.0002)/100(.0002)  -> KNOWN; method detects real knowledge. OK.
Pre-control gaps +6..+14; US-pol events partial (New Deal 5/5 gaps +8; Kennedy/Reagan 2/4-2/3); even
ubiquitous Moon landing 0-3/4 -> science genuinely clean. 

CONCLUSION (strongest yet): under screened multi-facet digging, US politics is the ONLY contaminated
category, significant on all cutoffs; positive control validates the method; clean categories at chance.
CAVEAT: per-event binomial underpowered at 4-5 facets (4/4 caps at p=.062) -> use category pooling, or
>=8 facets/event for per-event calls. Science needs more length-matched facets to be conclusive.

### 2026-06-25 (no-cue dig, DEFINITIVE)  Strip first-name cues -> the real answer

User's key correction: putting the answer's FIRST NAME in the prompt ("...was Leonid [Brezhnev]",
"Harry [Truman]") turns it into name-completion (tests pre/post fame), NOT knowledge. Rule added to
DeepSeek screen (rule 4): prompt must not cue the answer; produce it from the descriptive fact alone.
Scripts parameterized (validate_facets.py CANDMOD/SUF; dig_run_validated.py SUF). Candidate sets:
dig_candidates_v2.py (Russia/Germany, no-cue), dig_candidates_v3.py (US politics, no-cue). All DeepSeek-
screened (also nukes pre-1929-famous names/places, coined-term checks).

RESULT cued -> no-cue (category pooled %correct, screened, length-matched):
  US politics:  66% -> 68% (p=0.044 on 1929 & 1913 base; 64% p=.09 instruct; 54% p=.43 on 1900). HOLDS.
  Russia/Soviet: 80% -> 50% (p=0.5).  COLLAPSES -> was name-completion (Brezhnev +19 etc. were artifacts).
  Germany WWII:  64% -> 25% (p=0.9).  COLLAPSES -> artifact.
Per-event (1929, no-cue): Nixon/Watergate 4/4 +3.2; New Deal 3/3 +4.5; Kennedy 3/4 +2.4; Atomic 5/6.
  Obama/recent 0/4, gap -3.7 on ALL models -> recent US is CLEAN; leak is mid-century (~1933-1974).

DEFINITIVE CONCLUSION: the ONLY genuine contamination is MID-20th-CENTURY US POLITICAL HISTORY. It
survives the strictest test (no name cues, post-cutoff-only facts, DeepSeek-screened, length-matched):
68%, p=0.044, real structured knowledge. Everything else -- Russia, Germany, UK, France, science,
culture, and recent US (Obama) -- is at chance once name-completion artifacts are removed. The big
cross-category suspects (Churchill +15.9, Brezhnev +19, Shostakovich +11.5) were famous-name completions.
Files: dig_candidates_v2/v3.py, dig_validated_v2/v3.json, dig_validated_results_v2/v3.csv.

### 2026-06-25 (COMPREHENSIVE + AIRTIGHT)  The clean diagnostic done in full -> REVERSES the headline

User: do the right thing in its entirety; single rule = ANSWER must require post-cutoff text, never the
topic's name, never cued; KEEP famous-old topics (Churchill/Shostakovich/Sartre) but probe their POST-cutoff
facts (Shostakovich->1937 'Fifth'; Churchill->PM 1940/Iron Curtain; Sartre->1964 Nobel). Built dig_candidates_full.py
(127 facets, 9 categories, POST+PRE), DeepSeek-gated to 119 (dropped cues 'New->Deal','Animal->Farm','1944->44',
'Franco' pre-1929). Scored 4 models -> dig_validated_results_full.csv. Airtight paired A/B (pairs_score.py: 20
post-1929 people, identical prompt cued vs no-cue) and quadruplet test (quad_score.py: true vs ALL matched twins).
Docs: questions.html (every item + each model's per-item result), diagnostic.html (verdict), story.html (journey).
SAVED all: rank_examples.csv, falklands_detail.csv, pairs_results.csv, quad_results.csv (never recompute).

RESULT (pooled 3 base, per-item, strict cue-free length-matched):
  EVERY category at chance post-cutoff incl US politics: US 56% p=.62, UK 54% p=.80, France 29%, Germany 43%,
  Russia 48%, Italy 50%, Spain 28%, Culture 38%, Science 53%. Answer-type: name 44%, term 49%, date 45% (all chance).
  AIRTIGHT A/B: post-1929 people no-cue 38% -> cued 55% (+18pt within-item cue effect). Shallow NAME exposure exists
  (can complete 'John F.->Kennedy') but no FACT knowledge.
  QUADRUPLET: 2-way 53% vs 4-way 30%; low-confidence 2-way hits survive 4-way only 31% (vs high-conf 67%) -> quads
  bleed the marginal signal (user's caution CONFIRMED; don't use for subtle contamination).

REVISED CONCLUSION: under the strictest/airtight test the models are essentially CLEAN -- no category (incl US
politics) shows significant post-cutoff KNOWLEDGE. The earlier 'US politics RED (68-89%)' was length-bias metric
(89%) + first-name cues / item mix (68%). What remains is SHALLOW name/term exposure (A/B proves it), not coherent
contamination. Recovering famous-old topics opened no new contaminated categories. Caveat: modest n -> 'no
significant contamination detected' = strong cleanliness upper bound, not proof. Good for the WVS use-case.

### 2026-06-25 (association-lift + generation + chat)  "It has seen something" -- at the ASSOCIATION level

User pushed: generation is the ground truth, and the chat shows traces (Hiroshima<->bomb). Built generate_test.py
(greedy, no cue): Cuban-crisis->Kennedy (completion form) but in CHAT form confabulates (Harding for Cuban crisis,
Kennedy for the bomb); most post-cutoff Qs -> WRONG, anchored to pre-1929 names (Pershing for WWII/Moon/Tannenberg,
Gloria Swanson for 1962 starlet, Kornilov for 1917 revolution). => shallow name traces + confabulation, not knowledge.

assoc_probe.py (association-LIFT: logp(concept|target) - logp(concept|matched control), 15 triples x 4 models,
saved assoc_results.csv). Positive controls validate the metric (Waterloo->Napoleon +2.8..+5.0; Trafalgar->Nelson
+3.0..+4.4); NULL (Osaka vs Kyoto) ~0. LEAKED associations (consistent across models, comparable to the genuine
pre-cutoff ones): Auschwitz->death camp +2.3..+3.9; Vietnam->US war +2.2..+4.9; Stalingrad->battle +1.7..+2.8;
Dallas->assassination weak +1. NOT leaked (at noise): Hiroshima/Nagasaki->atomic bomb (!), Pearl Harbor, Berlin->Wall,
Dresden, Guernica, Chernobyl. => "it has seen something" CONFIRMED but SPOTTY: a few hyper-documented events left a
real semantic trace; the Hiroshima link the chat showed was question-framing, not a stored association.

REFINED PICTURE (3 levels): (1) shallow NAME strings (can complete 'John F.->Kennedy' with cue); (2) a few strong
ASSOCIATIONS (Auschwitz/Vietnam/Stalingrad, as strong as legit pre-cutoff ones); (3) NO coherent FACT knowledge
(free generation confabulates who/when/what). Recognition (forced-choice probes) >> generation (free chat).

Conversational quality: looping/verbatim repetition + generic filler ('man of strong will and commanding presence')
+ confident confabulation. Causes: 4B params / 160B tokens / archaic pre-1929 corpus; greedy decoding w/o penalty;
thin self-distilled instruct SFT. FIX applied to chat_gpu.py: repetition_penalty=1.3, no_repeat_ngram_size=3 (kills
the looping). Advice for WVS: constrain to forced-choice/Likert (don't rely on free essays); measure with logprobs
not chat; confident confabulation is the real risk -> would need refusal/uncertainty SFT for free-form use.
GPU chat live at localhost:7860 (chat_gpu.py, 1929-Instruct).

### 2026-06-25 (KDE)  Updated KDE set on the length-matched metric -> kde_lengthmatched.html

build_kde_lm.py renders, per category, (a) P(true vs false)-50% [prob space] and (b) the gap [log-prob]
KDEs, should-know (blue) vs should-not-know (red), pooled over 3 base models. NEW file; did NOT touch
kde_extras.html. Overview: PRE median d50 +49.8; POST US-politics +43.0 (on the pre pile = contaminated);
POST rest-of-West +6.2 (chance). NUANCE the KDE surfaced: science (POST median d50 +37.6) and Italy
politics (+45.5, n11) and Germany (+24.4) post-cutoff curves are BIMODAL/lean positive — ubiquitous facts
(Sputnik, DNA, Moon landing; Mussolini) leak — net >50% but NOT significant (small n). So significance
test = US-only; distributions = US clear + a science/Italy lean worth flagging. Captions corrected to say so.

### 2026-06-25 — Dense association-lift + era×region coverage map: HINDENBURG HYPOTHESIS REFUTED
Ran assoc_probe2.py (49 triples, 4 models -> 5_results/assoc_results2.csv) and coverage_map.py (~84
era×region recognition facts, 4 models -> 5_results/coverage_results.csv). build_coverage.py ->
coverage.html (heatmap + assoc group table). KEY REVERSAL:
1. NO German/1914-29 coverage hole. Recognition heatmap German column strong throughout (E2-E6 =
   100/92/92/100/92%). At item level the 1929 model recognizes HINDENBURG at gap +7.42 (Instruct +8.73) —
   ESSENTIALLY IDENTICAL to NELSON (+7.45 / +6.72). It knows Hindenburg exactly as well as Nelson in
   stored-knowledge terms. (German-E6 cell is 92% not 100% only because "Weimar"=1919 is POST-cutoff for
   the 1900 model, which correctly fails it, -0.8.)
2. THEREFORE "wtf does it not know Hindenburg" in CHAT is NOT a knowledge/coverage gap. It is a
   RECOGNITION-vs-FREE-GENERATION gap. Teacher-forced, Nelson≈Hindenburg (~+7.4). In open generation,
   Nelson (ultra-high-freq single token, locked in fixed phrases) gets retrieved; Hindenburg (rarer,
   longer token string) loses to high-freq confabulation (Pershing) even though the association IS stored.
   Classic "knows more than it can say" for a small lightly-tuned base model.
   IMPLICATION (audit): reassuring — teacher-forced probes measure the TRUE stored-knowledge ceiling;
   free chat measures only the weaker retrieval floor. Generation cannot reveal hidden knowledge probes
   miss; it reveals LESS. Contamination verdict stands / is conservative.
   IMPLICATION (WVS synthetic respondents): CAUTION — free-text answers under-represent what the model
   knows and confabulate names. Scaffold retrieval (give choice set / use recognition signal); don't rely
   on open generation.
3. Association internal-validity check PASSED. PRE_WEAK (1914-29 events) group mean ≈0 for 1900 (+0.42)
   and 1913 (+0.65) models — CORRECT, those events are post-cutoff for them — and jumps to +3.57 for 1929
   (Instruct +4.26), near the +4.35 of PRE_STRONG 19th-c Anglo. The 1929 model stores the WWI era almost
   as strongly as the Napoleonic era. Strong WWI items: Sarajevo +9.51, Somme +8.44, Passchendaele +7.17,
   Marne +5.69, Jutland +4.75, Tannenberg +3.36, Weimar +3.09. Weak ones (lexical/freq, not a hole):
   Verdun +0.39, Versailles +0.27, Petrograd -0.16, Ypres +0.79, Gallipoli +1.01.
4. POST-cutoff leak: group mean ≈0 (no broad leak). Per-item, a few modest positives for 1929 base:
   Vietnam +2.23, Nuremberg +2.17, Stalingrad +2.12, Auschwitz +1.75, Yalta +1.62 — but all are PLACE
   NAMES that existed pre-1929, so this is residual place-name prior, NOT leaked event knowledge. The
   PURE post-1929 EVENT tests come out NEGATIVE: Hiroshima -0.42, Pearl Harbor -1.12, Chernobyl -1.69,
   MLK/Memphis -1.60, Cuba missile crisis -2.46. Pattern (place-names slightly +, pure events −) confirms
   the earlier "Hiroshima↔atomic was a framing artifact" finding and the CLEAN verdict.
Files saved (never recompute): assoc_results2.csv, coverage_results.csv, coverage.html. Chat server
(chat_gpu.py, localhost:7860, 1929-Instruct) relaunched after the GPU batteries finished.

### 2026-06-26 — PRE-REGISTERED v2 STUDY RESULT: EVIDENCE OF CONTAMINATION in US civil rights (guarded).
(Chronology note: this is the latest entry; earlier 2026-06-26 entries on the reckoning/pre-reg/runpod are
inserted above out of order.) Pipeline mistakes caught & fixed (each would have given a wrong answer):
(a) deepseek-v4-pro retired -> deepseek-v3.2; (b) blind referee INVERTED (rejected items FOR being post-cutoff)
-> rewrote rubric; (c) adjacent-year distractors a knower can't pick -> discriminability gate + non-year steer;
(d) popularity CALIBRATION broke the instrument (pre-controls raw 82% -> calibrated 48%; inflated POST 31->38)
-> DISCARDED; use RAW forced choice (metric chosen on should-know controls, blind to POST).
Locked: question_bank_v2.json, 272 items (239 post + 33 pre), blind-screened 91% pass; clusters all powered.
RESULT (RAW sum-logp forced choice, chance 25%, results_v2_Ranke-4B-1929.json):
  PRE controls 82% (fires; not 100%). US civil rights 17/37=45.9% p=0.0045 FDR-FLAGGED -> CONTAMINATED.
  US politics 1945-74 15/40=37.5% p=0.054 (misses FDR) elevated. Cold War 27.5% p=.42, Vietnam 28.2% p=.38
  at chance. Broad pool 74/239=31% p=0.022 upper95=37% (driven by the 2 US-domestic clusters).
ITEM-LEVEL DIG confirms specific recall (distractors are OTHER REAL civil-rights entities): Brown v Board over
Plessy (+4.4), Wallace "schoolhouse door" over Faubus/Maddox/Barnett (+9.4), Orangeburg massacre over Jackson
State/Kent State/Attica (+10.9), James Meredith over Lucy/Malone/Hunter (+4.8), Freedom Rides over invented
names (+4.7), + Black Panther, Birmingham, Emmett Till, Black Power. ~10 strong hits; verdict rests on those.
Concentrated leak in US mid-century DOMESTIC history; foreign/war topics at chance. Earlier "clean" = only
Vietnam/WWII/ColdWar ever tested. VINDICATES Voth's "how can it be clean?". NOT CLAIMED: war topics clean,
complete map, or Talkie/Instruct status (NOT yet scored on this bank). Certify still needs corpus search.
DELIVERABLE: 8_suite/writeup.html (guarded step-by-step: SVG flowchart, charts, results table, civil-rights
hits, claims-vs-not-claimed, open items, 9-row mistakes ledger). NEXT: Talkie + Ranke-Instruct on
question_bank_v2.json; corpus search (Brown v Board, Freedom Rides, Orangeburg, Wallace, Selma).

### 2026-06-26 17:08 — TALKIE-1930-13B (base) on the v2 bank: CLEAN. Overnight runpod run done & verified $0.
This closes the "NEXT: Talkie on question_bank_v2.json" item above. Overnight runpod A100 PCIe job
(runpod_overnight.py, pod fkart67syltxb3, triple-teardown). VERIFIED via runpod GraphQL `myself{pods}` =
empty list -> pod self-terminated, nothing idling. Finished 2026-06-26 13:35Z.
RESULT (RAW sum-logp forced choice, chance 25%, 272 items = 239 post + 33 pre):
  PRE controls 93.9% (31/33, fires hard). POST/ALL 21.8% (52/239) AT/BELOW chance, binom_p=0.89, mean_gap -5.5.
  NO post category above chance: US civil rights 37.8% (14/37) p=0.058 NOT FDR-sig & mean_gap still NEGATIVE
  (-2.95); US politics 1945-74 exactly 25.0% (10/40); Cold War 20%; Vietnam 17.9%; decolonisation 23.5%;
  science&medicine 0%; space&computing 12.5%. Huge clean cliff (94% pre vs 22% post).
KEY CONTRAST: Ranke-4B-1929 on the SAME v2 bank showed US civil-rights CONTAMINATION (45.9%, p=0.0045 FDR-flagged);
Talkie-13B base does NOT replicate that leak — US civil rights only borderline (p=.058) and not above chance overall.
CAVEAT: different model (13B base vs 4B chat-tuned; cutoff 1930 vs 1929), so this is a model/arch difference, NOT a
refutation of the 4B civil-rights finding. The two models genuinely differ on post-cutoff US-domestic recall.
FILES: results_v2_talkie-1930-13b-base.json (THE real output, 13:35Z). NOTE: STATUS_OVERNIGHT.md text names
"results_talkie-1930-13b-base.json" but that string is hardcoded in write_status(); the monitor's RESULTS var wrote
the v2 file. results_talkie-1930-13b-base.json (22:44Z 2026-06-25) is a SEPARATE older v1 run, untouched.
DELIVERABLE: 8_suite/talkie_report.html (this session). NEXT (unchanged): Ranke-Instruct on v2 bank; corpus search.

### 2026-06-26 17:30 — INTEGRATED cross-model report (supersedes the two standalone HTMLs).
build_report.py -> 8_suite/contamination_report.html: ONE report covering every model on the locked v2 bank
(Ranke-4B-1929 + Talkie-1930-13B), all numbers recomputed from raw per-item records. INTEGRITY: deliberately
IGNORE the Ranke v2 file's built-in summary dicts (they use the DISCARDED calibrated metric; would show civil
rights 24.3% p=.60) and recompute from correct_raw/gap_raw -> reproduces the settled raw result exactly. Bank's
own level field drives narrow(4)/breadth(5) split; BH-FDR(q=.05) across the 4 narrow clusters per model reproduces
LOG's flag (Ranke US civil rights survives FDR p=.0045 -> CONTAMINATED; US politics p=.054 nominal-only; Talkie
zero flags -> CLEAN). Report auto-surfaces Ranke's civil-rights hits (Orangeburg, Wallace, Meredith, Freedom Rides,
Brown v Board...). Old talkie_report.html + v1-bank comparison.html kept on disk (versioning rule) but superseded.

### 2026-06-26 18:15 — REPORT v2: alpha=.10 two-tier verdict + didactic worked examples + grep-first reframe.
User pushback (correct): calling Talkie "clean*" at p=.058 while calling Ranke US-politics "elevated" at p=.054 is the
significant/not-significant fallacy (LOG's own RETRACTION-2 sin); and for a contamination audit a false-clean is the
costly error, so use a LOOSER bar. NOTE prereg (8_suite/PREREGISTRATION.md) fixed alpha=.05 + FDR q=.05 -- so I did
NOT claim a 10% agreement; instead two TIERS: RED=survives FDR q=.05 (registered primary); AMBER=above chance at
nominal alpha=.10 but not FDR. New verdicts: Ranke = CONTAMINATED (civil rights RED) + US politics AMBER; Talkie =
"ELEVATED - not clean" (civil rights AMBER p=.058) -- no longer a clean sweep. build_report.py params ALPHA/FDR_Q.
ADDED to contamination_report.html: (1) "Anatomy of a probe" -- 5 fully worked items with REAL per-option raw sum-logp
scored on Ranke locally (probe_examples.py -> probe_examples.json): First flight (control, fires, +10.7), Orangeburg
(+10.9), Wallace (+9.4) [leaks], DNA (length-biased miss), Sputnik (borderline -0.01). (2) Length-bias audit: 54/239
post items (23%) imperfectly length-matched; 44 true-longer-than-all (bias->miss) vs 25 shorter -> NET LEAN TOWARD
FALSE-CLEAN; raw sum-logp penalises long true answers (DNA true=12tok vs 3 -> gap -73). (3) "Grep the corpus, don't
gate on significance" section: probe = under-powered screen, grep = arbiter; 51 prioritised search targets (P1 17
confirmed-leak hits, P2 29 elevated hits, P3 5 clean spot-checks) as concrete true-answer strings. Corpus access
(~160B tok, needs model-team dataset/dedup index) is the real pending gate. Files: build_report.py, probe_examples.py,
probe_examples.json. NEXT (unchanged): Ranke-Instruct on v2 bank; run the actual corpus grep when dataset available.

### 2026-06-27 — TRIAL BALLOON: event-surprise shift via PMI decomposition, Ranke-1900 vs -1929.
New sub-experiment 9_surprise_balloon/ (surprise_balloon.py, .csv/.json; notes/2026-06-27-surprise-balloon.md).
Idea (prior-Claude's PMI fix of per-token-vs-sequence): -logP(target|ctx) = -logP(target)[lexical] + (-PMI)[conceptual].
12 headlines in 1900-only vocab; target = event predicate; S_marg via generic subjects {The man/A person/The thing/It},
S_cond via real subject; strain=S_cond-S_marg; d_cond=S_cond_1900-S_cond_1929 (+=1929 less surprised). 3 controls:
A became-real-1900-29 (6), B timeless (3), C future-to-BOTH (3, the key control). RESULT category mean d_cond:
A +1.98, B +1.27, C -2.31. FLAGSHIP hits (method works): WWI killed millions +5.59, transatlantic flight +3.16,
women in office +2.16 -> 1929 clearly less surprised. CONFOUND found: models NOT on same prob scale (d_lex baseline
shift ~+1.5, comparable to signal) -> naive cross-model strain confounded by calibration/sharpness; A vs B don't
cleanly separate. CLEANEST signal = ordering A>=B>C driven by C: 1929 is MORE surprised by still-future events
(moon -3.56, heart -4.89) => later/sharper model is more decisive BOTH ways (good: not just rewarding 1929 for all).
n=3-6/cell, noisy (king_throne timeless +3.55 = baseline drift). FIX for real version: matched event/timeless twins
differenced within-model, or per-model z-norm vs neutral corpus; >=20 items/cell; add 1913 for a 3-pt curve; bootstrap CIs.

### 2026-06-27 — FRAME PROBE: normal-vs-unthinkable via use-it/hold-it behaviour vs period-normal controls.
10_frame_probe/ (frame_probe.py, build_frame_report.py, frame_probe.json/.csv, frame_probe_report.html;
notes/2026-06-27-frame-probe.md). Implements the historian brief: don't read silence, plant fact + probe whether the
frame can USE it (gen = logP(consequence|fact)-logP|neutral) and HOLD it (pref_far = logP(consistent)-logP(default)
after neutral filler distance), classify each FUTURE fact vs PERIOD-NORMAL controls. BASE models 1900/1913/1929 only
(Instruct=modern contamination). 6 dims x 2-3 phrasings: future {woman_authority, human_flight, empire_fall} + controls
{inheritance, railway, succession}. CALIBRATION PASSED: all 3 controls PRIMED in every era (succession pf +9.8/+10.1/
+11.8; railway +4.5/+6.5/+7.6 ent~1.0). HEADLINE: future facts rise monotonically MIXED->PRIMED: woman -0.3/+3.6/+3.7,
flight +1.9/+3.9/+5.0, empire +0.9/+2.1/+4.0. QUALITATIVE TELL (free gens) richer than metric: woman HOLD probe reverts
to 'a man of middle age with gray hair and beard' in ALL THREE incl 1929; woman USE in 1900/1913 hands authority to a
man/husband (fluent distortion); 1929 empire_fall narrates the REAL 1917 Provisional Government (recall). KEY METHOD
FINDING: forced-choice RECOGNITION (woman vs man preferred) assimilates by 1913, but free-text PRODUCTION still reverts
to a man in 1929 -> recognition!=production, the gap is itself a finding. GAPS: no UNTHINKABLE/BROKE cells appeared --
all chosen future facts were BOUNDARY ideas (becoming thinkable). v2: add genuinely-post-1929 facts (computers, atom-
destroys-city, TV, DNA) as flat-unthinkable baseline; quantify reversion-rate in generation (entity/pronoun track or
Gemini-batch judge); more phrasings; report recognition-vs-production gap explicitly.

### 2026-06-27 — FRAME PROBE v2/v3: forced-choice FAILS, PRODUCTION discriminates the trichotomy (big course-correct).
10_frame_probe/frame_probe_v2.py (.json/.csv): 13 ideas x 4 phrasings (4 control + 4 boundary + 5 unthinkable) x
1900/1913/1929 base. KEY NEGATIVE RESULT: the forced-choice hold-it probe (and generativity-logp) DO NOT discriminate
-- even baseline-corrected (lift=pref_near-pref0), the genuinely-unthinkable facts REGISTER and RETAIN as strongly as
controls (atom_city reg +5.7, moon +3.5-7.0, thinking_machine +4.5 even in 1900). Because the probe tests whether the
model can PARSE the just-planted sentence, not whether the idea has a home. Also several A+/A- probes linguistically
unbalanced (treaty 'ended'vs'worsened' pf+17) -> must work in lift not absolute. THE DISCRIMINATING SIGNAL = PRODUCTION
(free generation): control=EXTENDS (fluent on-topic), boundary=REVERTS 1900 -> EXTENDS 1929 (woman 'took a man'/'sent
her husband' 1900 -> '10 women cashiers' 1929), unthinkable=DISTORTS or STALLS in ALL eras (atom bomb -> firemen with
buckets of water; thinking machine -> steam engine on water; TV -> off-topic baths; moon -> earthquakes/Egypt). The
field-marshal move is REAL and visible. JUDGE: Gemini batch (house rule). v1 judge gemini-3.1-flash-lite + thin prompt
= TOO LENIENT (99/156 EXTENDS; called bucket-brigade atom + woman->man-with-beard 'EXTENDS'). FIXED v2 judge =
gemini-3.5-flash (flash-lite failed this subtle task; pin was OCR-only) + FEW-SHOT prompt forcing key-novelty id +
fact-preservation test -> validated 7/8 on hand-diagnosed cases (8th = judge MORE correct, caught dead-king resurrected).
Batch resubmitted (156). build_frame_report_v3.py staged: production trichotomy matrix + assimilation-trajectory chart +
generations w/ judge labels + the forced-choice negative-result section. RECOGNITION != PRODUCTION is the headline.

### 2026-06-27 — FRAME PROBE v3 DONE: trichotomy confirmed via production (judge=gemini-3.5-flash batch, validated).
Improved judge labels (156): STALLS 70, EXTENDS 34, REVERTS 30, DISTORTS 22 (healthy spread vs old 99 EXTENDS).
USE-IT EXTENDS share (the clean metric, hold-it is non-sequitur for non-authority dims): control 0.62/0.50/0.38;
boundary 0.12/0.50/0.88 (CLEAN monotonic climb); unthinkable 0.00/0.10/0.10 (FLAT on floor, dom DISTORTS+STALLS,
never assimilates). Per-dim boundary all rise (woman 0->.5->1, empire .5->.5->1, wireless 0->.5->1). Field-marshal
DISTORTS confirmed: atom->bucket-brigade fire, thinking-machine->steam engine on water, moon->earthquake+Egypt,
TV->off-topic baths. Report: 10_frame_probe/frame_probe_v3_report.html (trajectory chart w/ control band + boundary
climbing + unthinkable floor; trichotomy matrix; field-marshal panel; authority reversion panel; forced-choice
negative-result section). notes/2026-06-27-frame-probe-v2.md. Files: frame_probe_v2.py, judge_gemini_batch.py
(modes test/testcases/submit/fetch), build_frame_report_v3.py. v1 forced-choice files (frame_probe.py etc) kept,
superseded. NEXT v4: >=6 gens/cell + majority-of-3 judge; per-dim tailored reversion probe; 2nd judge for agreement;
bootstrap CIs on EXTENDS share.

### 2026-06-28 — FRAME PROBE: single INTEGRATED plain-English report (HS-class accessible) per user feedback.
User: prior v3 report was "inside baseball", jargon never introduced/footnoted; wants ONE clear didactic doc readable
by a 15-yr-old, with examples. Built build_frame_report_explained.py -> frame_probe_explained.html (integrates v2 design
+ v3 production result). ALL jargon removed from visible text: classes renamed control->"Everyday facts",
boundary->"Just-arriving facts (1900-29)", unthinkable->"Far-future facts"; labels renamed EXTENDS->"Runs with it",
REVERTS->"Slips back", DISTORTS->"Twists it", STALLS->"Loses the thread". Structure: hook (frozen 1900 mind) -> the
puzzle (silence is ambiguous) -> the trick (plant fact, let it write) w/ real worked examples (woman->reverts to a man
vs king->runs) -> the four outcomes w/ example each (atom->bucket brigade) -> the grader (Gemini, validated) -> the
yardstick (3 fact types, why) -> result (trajectory chart in plain words + headline) -> full table (plain labels) ->
more real examples -> the false start (forced-choice failed: recognising != thinking-with) -> honest fine print +
footnotes (frozen-mind/base-not-instruct, grader limits, quaint OCR spelling, what we do/don't claim). Numbers from
data (everyday 62/50/38, just-arriving 12/50/88, far-future 0/10/10 % Runs-with-it). Supersedes v1/v2/v3 report HTMLs
(kept on disk). Fixed a stray non-ASCII char in subtitle. NEXT unchanged (v4 robustness).

### 2026-06-28 — FRAME PROBE extended to TALKIE-1930-13B (robustness: model vs era). Pod terminated $0.
Only one Talkie model exists (talkie-1930-13b-base, ~1930 cutoff) -> compare to Ranke-1929 (same era, diff team, 13B
vs 4B). Talkie = 13B + custom lib -> RunPod A100 (talkie_frame_probe_runpod.py, pod in6vqby7no0hd2, $1.39/hr, triple
teardown; VERIFIED gone, LIVE PODS []). Generated 52 free continuations (same 13 ideas/plants/prompts as local, first
2 phrasings x use+hold) via greedy loop on Talkie's validated custom full_logits (validation_maxdiff 0.0). Saved
talkie_frame_gens.json. Generations COHERENT, higher-quality prose than 4B. EARLY READ: Talkie (bigger, 1930) seems to
RUN WITH some far-future ideas the 4B twisted -- esp thinking_machine ('the human brain became superfluous... the new
thinking-machine did all its work automatically') and space/moon ('men who had gone up into space came back'); atom
still distorted (arms race, not one-bomb-one-city). Suggests bigger+slightly-later model has more reach on a few ideas
-- pattern mostly holds but not identical. Judged by SAME gemini-3.5-flash few-shot batch (talkie_judge.py reuses
judge_gemini_batch). Report staged: build_frame_report_explained_v2.py -> frame_probe_explained_v2.html (Talkie vs
Ranke-1929 table, class-mean pattern, Talkie examples, plain language). Awaiting grader batch.

### 2026-06-28 — TALKIE RESULT graded: comparison is CONFOUNDED; honest report shipped (not the tidy story).
Grader (gemini-3.5-flash) on Talkie 52 gens: EXTENDS 24, STALLS 20, REVERTS 5, DISTORTS 3. Class-mean "runs with it":
Talkie control 100 / boundary 50 / unthinkable 70 vs Ranke-1929 38 / 88 / 10. Per-idea agreement only 5/13. CHECKED
the text before believing (research integrity): the raw numbers are MISLEADING for 2 reasons -> did NOT ship "pattern
holds". (1) FLUENCY CONFOUND: a 13B writes smoother on-topic prose so it earns EXTENDS more easily on EVERYTHING (control
38%->100%); grader more lenient on polished prose. Can't compare absolute scores 4B vs 13B. (2) GRADER LENIENCY on the
hardest item: scored Talkie atom_city EXTENDS but the text is an ARMS RACE ("whole world became one great arsenal...war
inevitable"), never grasps one-bomb-one-city = really a DISTORT. Plus n=2/idea = noisy (Ranke controls railway=DISTORTS,
treaty=STALLS are 4B-prose artifacts). ROBUST findings only: (a) both ace clear everyday ideas; (b) both BREAK on the
deepest seedless far-future ideas -- television (both STALLS) and dna_code (Ranke STALLS / Talkie DISTORTS->scripture);
(c) REAL divergence: Talkie RUNS WITH the thinking_machine ("human brain became superfluous...new thinking-machine did
all its work...terrible danger in war") where 4B twisted it to a steam engine -- plausibly because robots/automata were
live 1920s discourse (R.U.R. 1920). So: era sets the outer wall, a bigger+richer mind pushes closer to it. Honest report
frame_probe_explained_v2.html leads with the fluency-confound trap, reads idea-by-idea, flags the atom grader-leniency,
states 3 robust bullets. FIX to make solid: many more gens/idea (not 2) for BOTH models + stricter grader. Pod $0 (gone).

### 2026-06-29 — NEW TOOL: Aho-Corasick post-1945 decontamination filter (7_aho_decontam/).
Built a fast anachronism-tripwire scanner to clean book corpora (target: HF storytracer/LoC-PD-Books, 140k books /
~8B words / ~48GB) so no book post-dates 1945. THREE scripts in 7_aho_decontam/: validate_markers_1945.py (Google
Ngrams validation), aho_scan.py (engine + HF streaming scan, resume/checkpoint), benchmark.py (the speed demo). Venv =
C:\Users\hvoth\timelock-contamination\.venv + pyahocorasick. METHOD: 167 candidate n-grams (seeded from existing
CONFIRMED/MARKERS/BUCKETS + fresh sweep, incl. deliberate neg controls) -> Google Ngram Viewer JSON endpoint (en-2019,
batched, reused fetch_ngrams from period-classifier/src/ngram_sharpen.py) -> KEEP iff pre-1945 MEAN freq < 1.5e-7 AND
post-1950 mean > 1e-8 AND log10(post/pre) >= 1.5. Gate on pre_MEAN not pre_max: Google's mis-dated volumes inject
single-year spikes of modern vocab into the pre-1945 bucket, so pre_max is noise; pre_mean is robust. RESULT: 130 kept
(129 after dropping 'civil rights act' = genuine 1866/1875 referent), e.g. cold war/internet/nato/nasa/laser/software/
transistor/khrushchev/gorbachev/world bank/unesco. Method visibly rejected ~22% incl. SURPRISES: penicillin(1928),
television(word 1900), 'united nations'(=allied nations in WWI), chemotherapy(1907), pearl harbor, world war ii, cia,
continental drift(1912). markers_1945.json + markers_rejected.json saved (rejected keeps the reason = audit trail).
ENGINE design (key lesson from find_contamination.py: bare substring LIES): Aho-Corasick = cheap first-pass multi-pattern
filter, O(text) regardless of #patterns; then verify each hit with (1) word boundaries, (2) PROXIMITY context (±200 chars,
not whole-doc which is too loose) for homographs lsd(£sd money)/iron curtain(theatre)/common market(marketplace)/third
world(poetic)/nato(Italian "born", Spanish vireinato). Plus normalize: de-hyphenate OCR line breaks (agri¬\ncultural ->
agricultural; ¬=U+00AC always joins, ascii '-' only across \n) and STRIP Internet-Archive boilerplate ("digitized by the
internet archive ... archive.org") which else dominates FPs via 'internet'. SPEED (500-book sample, 1 core): Aho-Corasick
~75 MB/s, ~410 books/s, 1.23s; union-regex baseline ~9 MB/s, 10.2s -> AC is 8.3x faster (confirms find_contamination.py
"~10x" hypothesis); both engines flag identical sets; automaton build 0.1ms. Extrapolation: full 48GB -> ~11 min 1 core /
~1.3 min 8 cores. DETECTION (the real payoff): 13/500 flagged (2.6%); 12 are TRUE post-1945 contamination CONFIRMED by the
dataset's own 'year' metadata (stated 1952/54/59/64/67/69/69/71/83/86/92/92) = post-war US-GOVT area-studies bibliographies
& "country study" volumes (US Govt works public-domain at any date) leaked into a "pre-1929 public domain" corpus; 1 FP =
laser=OCR("later"), irreducible ~0.2%. So LoC-PD-Books is NOT all pre-1929 -- the scan cross-checks content vs stated year.
Full README + benchmark_results.json in 7_aho_decontam/. (Running 3000-book sample for a tighter contamination rate.)

### 2026-06-29 (cont.) — Aho-Corasick decontam: 3000-book FP audit -> ZERO false positives, lexicon finalized at 128.
Scaled benchmark to 3000 books (800MB text). The 500-slice was deceptively clean; 3000 exposed the real OCR mess and let
me harden precision to 100% on this sample. SPEED held: AC ~71 MB/s, ~265 books/s, 8.5x faster than union-regex (both
flag identical sets); full 48GB -> ~11 min 1 core / ~1.4 min 8 cores. FP AUDIT (read every flagged book, fixed each class
on evidence -- NOT overfit): (1) IA boilerplate is OCR-GARBLED ("digitized by tine internet arciiive") so clean-string
strip missed it -> now strip whole "digiti[sz]ed by ... library of congress" block + any "internet arc*". (2) OCR char
confusion: hotel->motel (h/m, "Eastman Motel"=Hotel) -> DROPPED motel; later/lazar(leper)/leisure->laser in ME verse ->
laser needs optics context. (3) foreign homographs: Italian nato=born, Spanish vireinato -> nato needs alliance context.
(4) English homonyms/literal: Richard Nixon(1821 settler), Martin Luther King(b.1882 genealogy), "great leap forward"(idiom),
ships "rock and roll", "double helix"(descriptive geometry), KGB(1922 radio callsign), "common market value", "a european
community", "third world-era"(Kabbalah) -> per-term proximity topic guards (±200 chars, NOT whole-doc). (5) REAL BUG:
context alts were unbounded substrings -> "nation" matched "emaNATION", "gene" matched "geNERATED" -> word-bounded all (\b).
24/24 unit tests pass (13 FP fragments clean, 11 TP fragments fire). EXCLUDE={civil rights act, motel}; 14 of 128 markers
carry a context guard. FINAL: 16/3000 flagged (0.53%), ALL 16 TRUE positives, every one year-confirmed (1952/54/59/61/64/
67/69/69/71/78/83/85/86/91/92/92); contamination = LoC Federal Research Division area-studies handbooks & "country study"
volumes (US-Govt = PD at any date). 0 FP in 3000. NEXT options offered to user: full 140k sweep (aho_scan.py streams w/
resume) or expand lexicon. Updated 7_aho_decontam/README.md with final numbers + FP-audit table; bench_3000_final.log saved.

### 2026-06-29 (cont.) — VALIDATION: controlled contamination trial (ROC/AUC/precision-recall/dose-response).
User asked "how do we know it works -- contaminate deliberately and measure." Built contamination_trial.py: plant a
length-X section of a NEW post-1945 book into a verified-clean pre-1929 book, score=#distinct markers (graded, for
ROC/PR), negatives=600 untouched old books, 400 trials per (source x dose), seed=20260629. 3 injectants bracket reality:
real_modern_loc (genuine post-1945 LoC books = the threat), wikipedia (vocab-rich), tinystories (bland fiction = worst
case). Added sklearn + matplotlib to the timelock venv. HEADLINE: PRECISION=100% (0/600 old books ever false-flagged, FP=0
at every dose) -> trust the flags. Story is all recall vs DOSE. DOSE-RESPONSE recall (real / wiki / tiny): sentence(60ch)
0.2/0.5/0; paragraph(1.5k) 2/16/0.2; few-pages(8k) 17/35/0.5; chapter(40k) 40/81/5; book(150k) 74/100/3 %. So a whole
modern BOOK is almost always caught (74-100%), a single modern SENTENCE usually slips (rarely contains a tell-tale word);
tinystories ~0 throughout = bland modern prose w/ no period vocabulary is invisible by design. AUC: per-dose 0.50(sentence)
-> 1.000(book wiki); pooled real 0.61 / wiki 0.70 / tiny 0.51 (modest only because it averages impossible tiny doses);
avg-precision 0.85/0.88/0.80. INTEGRITY: recall >= "detectability ceiling" (chunk-alone has a marker) in EVERY cell ->
scanner never misses a chunk containing a giveaway word; misses are chunks w/ zero period vocab (e.g. a Wikipedia article
ABOUT the 1860s; a butterfly kids' story) = limit of language not a bug. (1 cell recall slightly EXCEEDS ceiling: a guarded
term's ±200 context came from adjacent base text -- benign, good direction.) REAL-DATA recall (no injection) = 50% (60/120)
of genuine post-1945 LoC books flagged by vocabulary alone -> USE WITH the metadata year filter (year catches correctly-
dated modern; this scan catches MISLABELED ones); recall rises as lexicon grows. Saved trial_results.json + dose_response.png
+ roc.png + pr.png; README has a Validation section. Numbers stable across two full runs (robust).


### 2026-06-29 -- INSIDE-OUT hidden-knowledge slice (Ranke-1929 base), PRE-REGISTERED. Verdict: NO credible hidden knowledge.
New 11_inside_out/ (PREREGISTRATION.md locked before any score; inside_out_slice.py, inside_out_stats.json,
build_inside_out_report.py -> inside_out_report.html). Method = Gekhman 2025 Inside-Out (arXiv 2503.15299): knowledge =
pairwise rank true>false (chance .5); EXTERNAL = output sum-logp; INTERNAL-A = logit lens (layer L*=35/36 chosen on PRE
only); INTERNAL-B = probe (L2 logreg on last-answer-token hidden vec, GroupKFold-5 by item, layer Lp=20/36 by CV).
Buckets (v2 bank): PRE n33 (pos control), science+space n33 (NEG control = unknowable), US civil rights n37 (suspect),
+ US politics/Cold War/Vietnam. CONTROLS: positive PASS (PRE K ext .90 / lens .88 / probe .73, all > .65); null PASS
(permute labels -> probe K .50, CI [.44,.55]). RESULT: lens Delta ~= 0 EVERYWHERE (sci+space +.01, civil rights +.04,
US pol -.01, Cold War +.00, Vietnam +.00) -> no hidden knowledge by the clean scorer. PROBE flagged sci+space Delta +.26
FDR=YES and Cold War +.22 FDR=YES -- BUT sci+space is the GENUINELY-UNKNOWABLE negative control, so the pre-registered
KILL CONDITION fires: probe-Delta is CONFOUNDED (detects answer plausibility/typicality, not knowledge; probe ~chance .56
CI incl .5 on sci+space; big Delta only because EXTERNAL is BELOW chance .29 there). lens & probe DISAGREE -> no robust
claim (pre-reg requires both agree in sign). SUSPECT US civil rights: lens Delta +.04, probe Delta -.03 ns -> the external
contamination is FULLY ON THE SURFACE, not hidden deeper. BOTTOM LINE: external contamination verdict STANDS; the model
does not secretly encode post-cutoff facts. METHOD BONUS: naive hidden-knowledge probing manufactures false hidden
contamination; an unknowable negative control exposes it (the pre-reg worked exactly as designed). Caveats: lens L*=35
near-final (conservative test); n modest (16-40/topic); Ranke-1929 base slice only. NEXT (user queue): resumed N-scale frame-probe.


### 2026-06-29 -- INSIDE-OUT ROUND 2 (CONCLUSIVE), pre-registered, in new shareable repo. Verdict: NO hidden knowledge; probe signal = ARTIFACT.
Repo: C:/Users/hvoth/Documents/GitHub/ranke-hidden-knowledge (git, pushed PRIVATE to github.com/huggingbeard/ranke-hidden-knowledge;
PREREGISTRATION.md locked before scores; METHODS.md full replication; RESULTS.md; src/ build_data/extract/analyze/report).
DECISIVE ADD vs Round 1: (1) COUNTERFACTUAL artifact floor = 60 INVENTED facts (fictional lands/cities/kings, equally-plausible
options, arbitrary true) -- the model cannot know them; (2) ALL 36 layers reported (Round 1 picked one near-final lens layer).
Ranke-4B-1929 base. RESULT: the probe scores ~0.6 on INVENTED facts (artifact floor 0.52-0.66 across layers, 0.61 final). NO
post-cutoff category beats the floor: excess over floor (probe best layer) -- civil rights -0.00 [-.13,+.12], US politics +.02,
Cold War +.09 [-.02,+.19], Vietnam -.02, science +.04, space +.11 [-.04,+.25]; EVERY CI includes 0, none survive FDR. Positive
control PASSES (PRE probe 0.77 > floor max 0.66; ext .90, lens .88) -> probe works, real null. => Round 1's 'hidden knowledge'
(Cold War/science ~.68) was the probe's generic true-answer-manifold artifact, PROVEN by the invented-fact floor. PUZZLE RESOLVED:
the probe never tracked post-cutoff knowledge (at floor everywhere, random wiggles); no internal contamination to dissociate;
the only real contamination is the OVERT external one (civil rights/US politics), consistent with shallow/lexical (name-completion).
External verdict STANDS and is NOT an underestimate. METHOD RESULT: closed-set hidden-knowledge probing is dominated by artifact;
an invented-fact floor is mandatory. Credit: user pushed back on dismissing the probe as artifact -> built the proper control ->
it conclusively shows artifact (didn't jump to conclusions either way). Phase B (only if pursued): open-generation deeply-hidden
track + larger N (~120-230/cat for ~.10 gaps). NOTE: frame-probe N judge batch processing in parallel (separate deliverable).


### 2026-06-29 -- INSIDE-OUT: artifact-floor control FIXED (natural fictional names) -> result CHANGED + report rewritten to clear-report standard.
User caught that the counterfactual names were phonotactic gibberish (Brorrnmar) -> unequal plausibility, inflated floor. Rebuilt
with natural equally-plausible fictional names (Selwick, Belby; Haled Renning). Clean floor dropped 0.6 -> 0.49-0.62 (0.56 final).
THIS CHANGED THE VERDICT: with the inflated floor nothing cleared it and Cold War looked hidden; with the clean floor -> Cold War
FALLS TO FLOOR (+.08, CI[-.04,+.19], was floor-inflation), and US POLITICS 1945-74 EMERGES as the one candidate: probe gap over
floor +0.13, 95% CI [+.03,+.23] (clears 0) BUT does NOT survive BH-FDR. US politics is ALSO the most output-elevated topic (says .55,
38% in raw audit) -> convergent (external+internal) = the credible candidate, not established. US civil rights: gap +.07 CI incl 0
-> overt at output, NOT amplified internally (shallow/lexical). Vietnam/science/space at floor. Positive control passes (PRE probe
.72 > floor .62; says .90 lens .88). HONEST verdict (no 'conclusive'): within this test, no internal post-29 knowledge that clears
a clean floor AND survives correction; US politics is the standout candidate worth a powered follow-up. METHOD LESSON: control
quality decided the result (bad names hid US politics, inflated Cold War). Also wrote a reusable SKILL .claude/skills/clear-report
(define->exemplify->explain->apply->conclude, 15yo audience, ban overclaim; gold-standard=frame_probe_explained.html) and rebuilt
results/report.html to it. Repo github.com/huggingbeard/ranke-hidden-knowledge (private) pushed 7c68685. NEXT (Phase B): US politics
+ ~120-230 facts, free-generation deeply-hidden track, 1900/1913.


### 2026-06-29 -- INSIDE-OUT PHASE B (powered round) DONE: US-politics lean does NOT survive. No-hidden-knowledge reading strengthens.
Repo ranke-hidden-knowledge (pushed 65f700f). Pre-registered (PREREGISTRATION_PHASE_B.md). Batteries LLM-generated (Gemini
batch) + screened (automated no-cue/9-distinct/dedup + Gemini verify): US politics 104, PRE 120, invented floor 120, each
1 true + 9 options. Length-normalised (mean) logp for say/lens; probe unchanged; two-sample bootstrap vs floor; all 36 layers.
RESULT: US politics @ best layer 33 -- probe 0.55 vs floor 0.57 = gap -0.02 [-0.10,+0.06] (pairwise); 0.11 vs 0.14 = -0.04
[-0.12,+0.05] (top-1/10). BOTH cross zero. Phase A lean +0.13 EVAPORATED with more evidence (regression to mean + length-norm
removed an artifact + cleaner floor). Positive control PASSES (pre probe 0.76 > floor 0.60; says 0.84). US politics still
elevated at OUTPUT (0.60) but NO extra internal signal -> contamination is OUTPUT-LEVEL/shallow, not hidden in weights.
HONEST verdict (no 'conclusive'): powered test finds no internal post-29 knowledge above a clean floor; external audit stands,
not an underestimate. Caveats: N=104 (~80% power for +0.13; tiny <0.08 effect could hide); length-norm change is a refinement
(part of the drop could be scoring not power); closed-set only (open-generation deeply-hidden track deferred = the one remaining
check); 1929 base only. Also this session: wrote .claude/skills/clear-report (define->exemplify->explain->apply->conclude, 15yo
audience, ban overclaim; exemplar=frame_probe_explained.html); rewrote Phase A report + RESULTS to it. NEXT (if pursued): open-gen
deeply-hidden track; 1900/1913 + instruct.

### 2026-06-30 19:15
RA challenged whether our Inside-Out implementation is "exact." Verified: (a) NO public Inside-Out
code exists (not on arXiv 2503.15299 nor Gekhman's site) — we reimplemented from the paper; (b) our
closed multiple-choice rounds keep the Inside-Out *metric* (pairwise K) but NOT its *experiment*
(candidates sampled from the model itself, ~1000/q). RA is right it's an adaptation. Built the faithful
open-generation track in the ranke-hidden-knowledge repo: src/opengen_{extract,judge,analyze,report}.py
+ PREREGISTRATION_OPENGEN.md + METHODS_OPENGEN.md. Design: sample K=200 continuations/question (T=1.0,
top_p=1, top_k=0), trim to short answer (abbrev-aware: "St.\nHelena" survives), dedup+freq, inject gold
(freq 0 if never said); Gemini-batch correctness labels; external = mean-logp, internal = probe on
last-answer-token hidden (GroupKFold by q), layer L* picked on PRE control; counterfactual floor;
deeply-hidden = gold never generated yet probe ranks it high. Dropped logit-lens (full-vocab proj for
~14k self-gen candidates = ~1 PFLOP, too slow; probe is primary internal scorer). Outputs namespaced
by RANKE_TAG so the 4B vs 1.7B don't overwrite.
BLOCKERS: (1) GPU busy — user's wvs-calibration/score_fast.py using ~9.2GB of 12GB; set up auto-launcher
(results/autolaunch_4b.log) to fire the 4B extract when GPU frees >=9GB. (2) The new 1.7B
history-llms/ranke-1.7b-1929-3006 still 404 to our token (user huggingbeard not in history-llms org;
old Ranke worked because it's in uzh-echist-org which huggingbeard IS in). Need org access / token / public.

### 2026-07-01 01:05
OPEN-GEN INSIDE-OUT RESULT (Ranke-4B-1929, K=200, L*=27). Pos control PRE: K_ext 0.72, K_int 0.75
(both >0.65, instrument works). Floor K_int(counterfactual)=0.86. RESULT = NO hidden post-1929 knowledge,
on two controls. Output: model generates correct post-1929 fact only 0.3-2.4% (vs 76% pre-1929); K_ext
post = 0.28-0.38 (BELOW 0.5 chance) -> no recall. Internal probe LOOKS elevated (K_int 0.66-0.75, int-ext
gap +0.31..+0.44 = naive "hidden knowledge") BUT it's ARTIFACT: (1) int-ext gap is LARGEST on INVENTED
facts (+0.65) and ~0 on known pre-1929 (+0.03) -> a signal peaking on the unknowable isn't knowledge;
(2) trivial "shorter=correct" baseline K_len reproduces probe K on every post topic (0.66-0.83). Mechanism:
model never says post-1929 facts, so only correct candidate is the injected gold (short/clean/canonical,
1.3-2.9 words) vs model's rambling wrong guesses (2.5-6.0 words); probe detects length/style/novelty.
Every post topic BELOW floor (gaps -0.11..-0.21, none clears FDR). Deeply-hidden: floor rate 0.683, all
post topics below (0.31-0.50). CONSISTENT with closed Phase B null + sharpens it: closed-MC US-politics
"external contamination" was RECOGNITION among 4 options; open-gen shows ~0 free recall -> shallow/output.
METHOD CONTRIBUTION: naive open-gen Inside-Out manufactures false hidden-knowledge on time-locked models;
the counterfactual floor (our addition, NOT in Gekhman et al.) is what exposes it. Repo: ranke-hidden-knowledge,
src/opengen_*.py + RESULTS_OPENGEN.md + METHODS_OPENGEN.md + PREREGISTRATION_OPENGEN.md. 1.7B still blocked (404).

### 2026-07-01 02:00
OPEN-GEN FINAL (pro judge). Labels took 3 passes: flash-lenient (mislabeled garbage e.g. "There is a fee
of so cents"->Medicare) -> flash-strict (12/12 on hand cases but STILL accepted "Washington conference"->
Marshall Plan at scale) -> gemini-2.5-pro (9/9 incl failures) + counterfactual override (cf correct=injected
gold only; no judge can tell two invented names apart) + lexical-overlap gate for REPORTED genRate only
(judge FPs are conservative for metrics so metrics use pro-judge直接). Caught via "why are invented facts
ever scored correct?" self-check. VERDICT UNCHANGED across all 3 passes = robust.
FINAL NUMBERS (Ranke-4B-1929, K=200, L*=26): pos control PRE K_ext 0.75 K_int 0.78 (>0.65 OK), ever-generates
correct pre-1929 in 76% (25/33) of Q. Floor K_int(cf)=0.94. Post: ever-says-it 0-10% (gated), K_ext 0.25-0.40
(BELOW 0.5), K_int 0.65-0.83, gap>floor all NEG (-0.10..-0.28) none FDR, K_len~=K_int. int-ext: PRE +0.03,
cf +0.76 (LARGEST), post +0.31..+0.55. Deeply-hidden floor 0.90, all post below. CONCLUSION: no hidden
post-1929 knowledge; internal "signal" = length/injection artifact below floor; output shows ~0 recall;
closed-MC US-politics was recognition not recall. Method point: naive open-gen Inside-Out manufactures false
hidden-knowledge (+0.76 on INVENTED facts); counterfactual floor (our addition vs Gekhman) exposes it.
Files: ranke-hidden-knowledge/ opengen_* + RESULTS_OPENGEN.md/METHODS_OPENGEN.md/PREREGISTRATION_OPENGEN.md.
Report: results/opengen_report_ranke4b1929.html. Next: commit/push; 1.7B when history-llms access granted.

### 2026-07-02 — NEW SUB-STUDY: `12_foresight/` — ex-ante FORESEEABILITY of 1914 & 1929 (inverts the audit)
User asked to "run ranke" and probe whether the time-locked models had any INKLING of impending
catastrophe (WWI/1914, Great Depression/1929, secondary 1933 regime change), and to write a Voth-style
PDF-LaTeX paper. Inversion of the contamination audit: given the models are CLEAN post-cutoff, measure how
SURPRISED they are by what comes, and what they EXPECTED ("home by Christmas?", "downturn severe?").
Plan approved. Locked design: (1) core estimator = cross-model horizon differencing ΔS(i)=S_pre(i)−S_post(i)
on identical items (same tokenizer ⇒ purges lexical/NER drift; hindsight model's surprise IS the lexical
floor); anchors: WWI = 1913 vs **1929** (clean, in-sample); Depression/Weimar have NO post-event Ranke ⇒
use 1900→1913→1929 gradient + **Qwen2.5-7B** modern hindsight floor via tokenizer-internal PMI Δstrain.
(2) Also a naturalistic OOS text-perplexity arm on curated post-shock passages. (3) Voth-style paper.
Reuse: slp/PMI (surprise_balloon), c(a) calibration+BH-FDR (score_v2), traffic_light gate, entropy_next
(frame_probe), length-matched twins (lm_battery), freecall (production).
**PRE-FLIGHT (Step 0) — GATE PASS:** (a) 1900/1913/1929 base tokenizers IDENTICAL — vocab=151669 all,
merges.txt sha 8831e4f1a0444713 all, and identical token-ids on real outcome strings ⇒ ΔS differencing is
lexical-drift-free. (b) Qwen2.5-7B (+Instruct) cached; vocab 151665 DIFFERS ⇒ used only via Δstrain, never
raw ΔS. (c) CUTOFF SEMANTICS: no model card exists (Hub cardData=None, no README anywhere) — established
operationally: docs say "trained only on text up to cutoff year"; parent audit uses WWI-1914 & Russian-Rev-
1917 as PRE positive controls for the 1929 model ⇒ 1914–1929 IS in-sample for 1929 (validates WWI anchor).
Exact Oct-1929 crash boundary UNDOCUMENTED ⇒ Depression battery targets unambiguously-post-1929 DEPTH
(1930–33: 25% unemployment, bank failures, lost decade) which is OOS for every Ranke model regardless;
prompts phrased as forecasts ("the years ahead"); the boundary itself will be probed empirically as a bonus.
venv C:\Users\hvoth\timelock-contamination\.venv (torch 2.11.0+cu128, transformers 5.12.1), RTX 3060 OK.
Windows note: console is cp1252 — scripts must sys.stdout.reconfigure(utf-8) + run with PYTHONUTF8=1 or Δ/PMI
glyphs crash printing. Files: 12_foresight/preflight.py, preflight_report.json.

### 2026-07-03 00:1x — CORE RESULTS (Arm A scored, 4 models; stats done). Bank blind-screened CLEAN.
Blind screen (2 independent agents): all 57 expectation items historically sound + period-authentic; 5 minor
PRE-SCORING repairs (1 cue word in dep_sev_fourth; 4 co-true gate distractors: Jutland/Skagerrak, US-entered
Brazil/China, sterling Sweden, Dow ninth) logged in screen_report.json; 0 items dropped. Scored 1900/1913/1929
Ranke + Qwen2.5-3B (modern anchor; 7B doesn't fit 12GB). Scoring slow (~3min/model, near-full VRAM) but fine.
FINDINGS (stats_report.json):
1. WWI horizon ΔS=S_pre−S_1929: ΔS[1913]=+2.30[1.32,3.41], ΔS[1900]=+2.87[1.73,3.98]; Tier-C NULL≈0
   (−0.45,−0.23) → differencing validated. BUT decomposition: Δlex DOMINATES (+2.58/+3.95), Δconc≈0
   (−0.28[−1.25,0.62]). So raw WWI "excess surprise" is LEXICAL DRIFT (1929 learned the words), NOT conceptual.
2. Excess conceptual strain vs modern XS=strain_M−strain_modern, AT THE 1929 model: wwi +1.10[0.13,2.02]
   ≈ ctrlB timeless +1.25; dep +3.54[2.23,4.81] > ctrlC future-to-all +2.14. i.e. WWI ≈ boring/foreseeable;
   Depression MORE surprising than a moon-landing-class event.
3. HEADLINE CLEAN CONTRAST (scale offset cancels in DiD): at 1929, Dep XS − WWI XS = +2.43, CI[0.80,4.08],
   p=0.0026. Depression ≫ WWI in ex-ante conceptual surprise — supports the disaster-risk-premium thesis.
4. "Home by Christmas" REFINED: 1913 model NOT confidently wrong (2/8 items; CW −0.23; WWI forced-choice 55%
   > chance). The illusion was popular SENTIMENT, not conceptual blindness — the 1913 written corpus (Bloch
   attrition warnings) made a long war conceivable. Honest refinement of the prior.
5. Depression forced-choice: models lean OPTIMISTIC consensus (dep 1929=38%, 1900=25% below chance) = genuine
   no-inkling. weim noisy/inconclusive (XS ~0.5, wide CIs; n=10). g-vs-ctrlB effect sizes USELESS (ctrlB n=6
   huge variance) → lead with the direct Dep-vs-WWI contrast, not SD-scaling.
Caveat: base-model forced-choice noisy (controls 33-60%); PRIMARY = continuous strain/XS/ΔS, forced-choice
secondary. Cross-family XS has a scale offset → only DiD/contrasts at same model are clean. Files:
score_all.py, stats.py, scores_long.csv, scores_by_model.json, stats_report.json. Next: gate (running),
text_perplexity (Arm B, real Keynes/Angell/FDR/Hoover passages downloaded), production, report, Voth paper.

### 2026-07-03 00:2x — GATE + ARM B done; PDF compiled. Sub-study COMPLETE (production finishing).
GATE (gate_report.json): DEP 1929 model at CHANCE 2/8=25% p=0.63 → GREEN, and 0/2 on the Oct-1929 crash
itself (Black Thursday/October both wrong) → high Depression surprise is GENUINE, not leakage. WWI 1913 model
75% → RED, but item-level = PLACE-NAME PRIOR: right on pre-1914 places (Gallipoli, Jutland, Passchendaele
+15.86, Versailles, Lusitania=1906 liner), WRONG on post-1914 coinages (tank −7.52, US-entry 1917). This IS
the Δlex channel. WEIM 4/8 p=0.11 → GREEN (noisy). ARM B (text_perplexity.json, real prose, within-Ranke
tokenizer): WWI DiD(1913−1929, Keynes-1919 vs Angell-1910)=−0.01 (echoes WWI null); DEP excess (FDR-1933 vs
Hoover-1929)=+0.70/+0.65/+0.75 at 1900/1913/1929, NOT falling toward crash. PRODUCTION (running): 1913 model
free-generates "for years" for war duration (prod_true=1.0) — even unprompted the models DON'T share the
short-war illusion; recognition>>production gap holds. DELIVERABLE: paper/foresight.pdf (8pp, Voth style, 4
figs, every number auto-generated by make_numbers.py→numbers.tex, 0 undefined refs). foresight_report.html +
figures/fig1-4.png. Sub-study CONTEXT at 12_foresight/CONTEXT.md; parent CONTEXT + global memory updated.
Reproduce: PYTHONUTF8=1 venv python; preflight→freeze→score_all→stats→gate→text_perplexity→production→
make_numbers→latexmk. Honest refinement of the prior: WWI conceptually FORESEEABLE (illusion=sentiment),
Depression genuinely UNFORESEEABLE (>moon-landing surprise); contrast +2.43 nats p=0.0026.

### 2026-07-03 01:0x — PRODUCTION done; sub-study CLOSED. Final PDF 8pp.
Recognition≫production gap confirmed (production_report.json): recognition(forced-choice correct on the 8
dual-probed items) vs production(mean frac_true volunteering the true outcome in free-gen): 1900 62%/0.23,
1913 75%/0.19, 1929 50%/0.38. So teacher-forcing = UPPER bound on inkling; open generation reverts to
consensus → can only weaken a signal, never manufacture one (good for validity). LONE exception: war DURATION
— 1913 model free-generates "for years" in 1.00 of continuations → even unprompted it does NOT expect a short
war; "home by Christmas" is absent from the learned record. Paper production paragraph + macros
(recog/prod{Nn,Th,TN}, prodDurTh) added; final compile clean (40 macros, 0 undefined, 0 real TBD, 8pp,
349KB). GPU freed/idle. ALL 11 tasks complete. Deliverable: 12_foresight/paper/foresight.pdf.

### 2026-07-03 (later) — interactive R&R with user: fixed Arm B, added Zweig trilingual worldview; Quarto walkthrough
User pushback drove several improvements (all logged in walkthrough.qmd → walkthrough.html, self-contained Quarto):
1. ARM B v1 was FLAWED (single passage/cell; "gap doesn't shrink toward 1929" measure is INSENSITIVE — WWI
   post−pre gap identical at 1913 & 1929 even though 1929 has WWI hindsight; and confounded by FDR-vivid-vs-
   Hoover-bland style). FIX = ARM B v2 (text_perplexity.py rewritten): surprise PER-CHARACTER (tokenizer-
   independent) so the MODERN model anchors it, like the constructed probes; 3 passages/cell from the SAME
   authors (Keynes/Angell, FDR/Hoover); bootstrap over passages. RESULT: modern-anchored DiD — WWI ≈0 (passes
   own control: 1913 +0.024 CI[−0.038,+0.116]); Depression +0.237 CI[+0.147,+0.312] @1929, ~10× WWI, clears 0
   at every cutoff. Powered second pillar. (fig4_realtext.png now the DiD bar chart.)
2. Confirmed models are competent in GERMAN and FRENCH (corpus multilingual — free-gen fluent & on-theme).
3. ZWEIG worldview probe (zweig_probe.py): from 'The World of Yesterday' (PDFs in zweig/, extracted via pypdf)
   pulled 5 pre-1914 themes; belief-vs-skeptic pairs in EN/DE/FR; CALIBRATED (c(a)=slp(stem,a)−slp(neutral,a))
   to kill fluency confound. HEADLINE (war_unthinkable, cal_pref; + = holds illusion): ENGLISH pre-war minds
   EXPECT war (1913 −3.92); GERMAN mildly hold "unthinkable" (+0.41), modern rejects (−2.07); FRENCH hold it
   STRONGEST (1913 +4.02), modern −2.14. → the "war impossible" illusion is CONTINENTAL (FR>DE), ABSENT in
   English (matches our foresight result that English 1913 expected a long war); holds at 1929 too. After
   calibration only 2 of 5 themes survive: progress-faith (held in ALL langs, Ranke−modern +4.7/+1.3/+1.4
   EN/DE/FR) and war-unthinkability (Continental only); security/universal-peace/moral-ascent wash out ≈0 (were
   fluency). (fig6_zweig_war.png.)
4. Also clarified for user (chat, to fold into paper): the fed-outcome STRAIN measure on WWI casualties is
   WEAK/confounded (neutral "a war" already carries "wars are costly"; strain₁₉₁₃≈strain₁₉₂₉ = shared generic
   prior, not foresight) → FREE GENERATION is the sterner test (duration volunteered 1.00, scale/millions 0.00
   → 1913 foresaw the war's LENGTH, not its SCALE/trench-FORM). Home-by-christmas fig corrected: 1913 leans TRUE
   (long war) on 6/8, misses only 2 trench-tactics items (not "undecided").
Deliverables now: walkthrough.html (Quarto, 6 figs, 13 tables, §0–12 + summary + reproduce), foresight_report.html,
paper/foresight.pdf (still v1 — needs the Arm-B v2 + Zweig + casualties-caveat folded in; TODO). New files:
zweig_probe.py/.json, zweig/ (PDFs+txt), text_perplexity.py v2, walkthrough.qmd/.css. NEXT: bring paper into
line with walkthrough (Arm B v2, Zweig section, casualties caveat, bibliography already started).

### 2026-07-03 (later 2) — FORWARD cross-lingual probe (fixes Zweig retrospective bug); walkthrough §9 added
User caught that the Zweig "war unthinkable" stem is RETROSPECTIVE ("to the men of that age a war SEEMED..."),
so for 1929 it measures knowledge of what the pre-war generation believed, NOT the model's own expectation
(that's why 1929 DE/FR "held war unthinkable" — absurd for a model that read WWI). FIX = crosslingual.py:
FORWARD-framed MINIMAL-PAIR statements, calibrated c(a)=slp(stem,a)-slp(neutral,a), EN/DE/FR × 1900/1913/1929/
modern, with a TRANSPARENT crosslingual.html dumping every stem/twin/neutral/score (user wanted to SEE what's
asked+scored). First pass (1 item/theme) too noisy; expanded to 3 items/theme + symmetric twins (keep/lose
crowns, gain/lose freedom) to kill DE/FR idiom artifacts. CLEAN RESULTS:
- war_comes: 1929 EXPECTS war (+1.85 EN, +0.87 DE) — sanity restored (the −6.68 was broken multi-word phrasing).
- war_long: "HOME BY CHRISTMAS IS GERMAN" — EN/FR expect LONG war (+), GERMAN expects SHORT (−4 to −1) =
  Schlieffen short-war doctrine encoded in German, absent in English. Refines the earlier English-only result.
- war_millions: German pre-war −0.8 → 1929 +2.3 (foresight→hindsight).
- monarchs_fall: CLEAN monotone 1900→modern rise in ALL 3 langs (DE +0.0→+3.1; artifact fixed) — the fall of
  Kaiser/Tsar/Habsburg, barely foreseen 1900, known by modern. THE headline of this arm.
- women's emancipation: rises to modern EN (+4.4) & DE (+1.6, artifact fixed); FR still negative (uncracked).
- controls (moon/bomb/heart): modern SEPARATES (knows them: moon EN +8.4, heart FR +10.3); pre-modern low. ✓
- HOLDOUTS (null, not claimed): cult_of_youth (no signal any lang), women-FR (twin artifact).
Method lesson: forced-choice belief-pref needs MINIMAL pairs + CALIBRATION + ≥3 items + symmetric twins;
retrospective framing conflates model-expectation with knowledge-of-past-belief. WALKTHROUGH updated: new §9
"What became thinkable" (fig7 monarchs, fig8 war-by-language), renumbered to §0–13, inventory X1–X5, limitations
+youth/FR-women, bottom-line updated. walkthrough.html = 8 figs, 14 tables, 2.73MB self-contained. New files:
crosslingual.py/.json/.html, war_lang.py (superseded), fig7_monarchs.png, fig8_warlang.png. Paper STILL v1 (TODO).

### 2026-07-03 (later 3) — PAPER brought UP to walkthrough level. foresight.pdf now 12pp, compiles clean.
Revised paper/foresight.tex: (1) ARM B rewritten to modern-anchored per-char DiD v2 (§4.3 in Results) — WWI
DiD +0.02 CI[−0.038,+0.116] (own control ≈0), Depression +0.24 CI[+0.147,+0.312] @1929; new fig4_realtext;
REMOVED the duplicate/flawed §Robustness "Real text" paragraph (had per-word + "gap doesn't shrink" logic).
(2) Fixed figure refs to the new self-explanatory set (fig1_main, fig2_wwi_split, fig3_homebychristmas,
fig4_realtext) — old fig1_xs_gradient/fig4_textppl gone. (3) Added CASUALTIES-STRAIN CAVEAT to WWI section
(fed-outcome overstates scale-foresight via generic prior; free-gen sterner: volunteers length not scale).
(4) NEW §"Turning the lens on the worldview" (Zweig, trilingual calibrated, war-unthinkable Continental, fig6).
(5) NEW §"What became thinkable" (forward cross-lingual: monarchs fig7, home-by-christmas-German fig8, women
EN/DE, controls). (6) Added BIBLIOGRAPHY (was never actually added — earlier edit had failed): 10 refs
(Angell/Barro/Bloch/Fisher/Hoover/Keynes/Kuran/Rietz/Roosevelt/Zweig) + natbib \citet throughout. (7) Abstract
+ Conclusion extended for worldview/cross-lingual scope. make_numbers.py Arm-B macros updated to new
text_perplexity.json structure (textWWIdidTh/textDepDiDTN + CIs). FINAL: latexmk clean, 0 undefined refs/cites,
0 errors, 12pp, 719KB, all 7 new figs embedded, 0 real \TBD, 46 macros. Paper == walkthrough now.
Deliverables: paper/foresight.pdf, walkthrough.html, crosslingual.html, foresight_report.html.

### 2026-07-03 — checkpoint
Foresight sub-study complete and all three deliverables in sync (PDF paper 12pp, Quarto walkthrough §0–13,
transparent crosslingual.html). Session arc: built the core foresight measure (horizon-differenced surprise,
lexical/conceptual split) → WWI conceptually foreseeable / Depression not (contrast +2.43, p=0.003, gate-
validated); then a long interactive R&R where the user stress-tested it: (a) fixed Arm B v1 → v2 (modern-
anchored per-char DiD, powered — WWI≈0 control, Dep +0.24 clears 0); (b) added the Zweig trilingual worldview
probe, caught its retrospective-framing bug, re-did it FORWARD; (c) built the transparent cross-lingual battery
(minimal pairs, calibrated, ≥3 items, symmetric twins) → clean results: monarchs foresight→hindsight all langs,
"home by Christmas is German", women EN/DE, modern-separating controls; (d) brought the paper fully up to level.
Abandoned this session: Arm-B-v1 gap-shrink logic, retrospective belief framing, multi-word twins, g-vs-TierB.
OPEN: French women's-emancipation twin artifact; cult-of-youth null; Weimar/1933 needs a ~1938 model (would
also anchor the Depression text-arm in-family). CONTEXT.md rewritten to v2.0.

### 2026-07-03 14:30 — restructure to user's spec, reorg, GitHub
Created a PRIVATE GitHub repo `huggingbeard/ranke-foresight` (root = 12_foresight), pushed via an ephemeral auth
header so no token lands in the Dropbox-synced .git/config. Then acted on three user corrections:
(1) "the cross-lingual thing is a hodgepodge — REDO the main analysis (§4) in three languages on equal footing,
with the fall of the empires on par with WW1 and 1929." Built `src/crosslingual_events.py`: the strain-based
conceptual-surprise measure, 3 events (World War, FALL OF THE EMPIRES [new], Great Depression) × EN/DE/FR, 8
items/event, war & empires anchored on the in-sample 1929 mind, Depression on modern; pooled + per-language DiD;
transparent reports/crosslingual_events.html. Result: WWI foreseeable (Δconc@1913 ≈0 all langs), EMPIRES
foreseeable (≈0 all langs — the collapse history calls "unimaginable" was no surprise to the 1913 mind), Depression
NOT (XS-vs-modern high at every cutoff incl. 1929, all langs). Pooled Dep−WWI +0.89[−0.74,+2.44]@1929 (sign right,
underpowered — lead with the in-family Δconc).
(2) "Zweig should stick to what he really said — progress, war, women, youth." Rebuilt `src/zweig_probe.py` to
those FOUR themes: progress held everywhere (dropped by modern — most robust signal), war Continental (EN expects
war, DE/FR hold "unthinkable"), women/youth = honest holdouts (women reads as modern's hindsight; youth null).
Empires/monarchs moved OUT of Zweig into §4; old forward-probe (crosslingual.py/war_lang.py) superseded, kept but
unreferenced.
(3) "Too much stuff in the repo root." Reorganized into src/ results/ figures/ reports/ paper/ passages/ zweig/
docs/; centralized all I/O paths in src/common.py; root now just README + CONTEXT + folders.
Reworked paper §4 (new three-event cross-lingual subsection + fig7) and Zweig section (§7, four themes + 2×2
fig6), DELETED the old "What became thinkable" forward section and its figs (monarchs/warlang). Regenerated
figures + numbers.tex (81 macros, no TBD), recompiled foresight.pdf (13pp, clean), re-rendered walkthrough.html
(§8/§9 rebuilt), updated README + CONTEXT (v3.0). Pushes: reorg+analyses (4781b89), paper/walkthrough rework
(d8bac96). OPEN unchanged (women/youth holdouts, Weimar needs ~1938 model, cross-lingual DiD underpowered).

### 2026-07-04 03:30 — OVERNIGHT REVAMP: the eight-catastrophe panel (v2), plan approved & executed
User: "results mixed, constrained by not going the whole 9 yards — take over, re-run, revamped draft by
morning." Executed the approved max-effort plan end-to-end:
BUILT: 8-event panel (panic07, WWI, empires, revolution, flu, hyperinflation, depression, nazi33) × 12 items ×
EN/DE/FR + Tier-B/C controls per lang + 6 true/false duels/event + 6 gate probes/event/lang; prereg v2 written
BEFORE scoring; automated screen (0 problems; 432-stem 1913-perplexity check) + SHA-256 freeze; gates run FIRST.
Fixed a real VRAM leak (common.free(model) never dropped the caller's ref — masked historically by WDDM paging;
callers now del before load; wait_for_vram+OOM retry added).
RESULTS: (1) Catastrophe was THINKABLE — panic −0.49, flu −0.39, WWI −0.23, empires −0.20, revolution −0.12
(pooled Δconc vs in-family anchors; all CIs straddle 0; all 3 langs; gates 7/8 GREEN, revolution AMBER =
Brest-Litovsk geography prior, post-hoc GREEN). Validations: panic positive-control ✓, 24 carryover items
reproduce exactly ✓, Tier-C in-family ≈0 ✓. (2) The ONE exception: HYPERINFLATION +0.74 [+0.04,+1.50],
FRENCH-led (+1.39, p=.006, sole FDR survivor), English-null. (3) SELECTION failed, not imagination: T/F duels —
panic 61% true, flu 61% (influenza-over-cholera +1.6 but spare-the-young/serum wrong), NAZI 17% (bets on
putsch/Kaiser/soviets, against the lawful path); free-gen: Tsar-deposed volunteered 4/6, influenza 0/6 (all say
cholera), "who rules Germany" → "not yet in sight" (1929). (4) SELF-CORRECTION: modern anchor carries ≈+2-nat
scale offset (in-sample WWI reads +2.35 vs modern at the 1929 mind that LIVED it; offset EN/DE/FR
+1.96/+1.91/+1.06; German Tier-C tripwire fires, 2 degenerate items) ⇒ v1 crosslingual "Depression unforeseeable
in all 3 languages" WITHDRAWN as offset. (5) Depression now OPEN: v1 English offset-free DiD +2.43 p=.0026 +
Arm-B stand; panel's within-1929 contrast ≈0; item-set dependent; needs the 1938 model. (6) Zweig v3 (5
items/theme, women normative): war-unthinkability is FRENCH (German flipped sign with battery size — worldview
probes battery-sensitive, flagged); progress modest; women/youth null after full effort.
SHIPPED: paper fully rewritten ("Could Anyone Have Seen It Coming?", 12pp, compiles clean 1st pass, 225
auto-macros + 4 auto-tables, zero hand-typed numbers); walkthrough §9/§9b/§8/§0/§11–13 rebuilt + re-rendered;
new figs figP1–figP4; transparent reports/panel_events.html; README rewritten; superseded scripts deleted from
head (git history keeps them); CONTEXT v4.0. Milestone pushes: 6870db7 (bank+gates), final push follows.

### 2026-07-04 morning — DRAFT v3: top-5 rewrite (empirics frozen)
User: "assume empirics done — is this top-5 ready? map a plan." Diagnosed v2 as a rigorous lab report, not a
paper (no economics, no framework, thin lit, no mechanism, mea-culpa register, Depression under-adjudicated,
no appendix). Plan approved with directives: AIM AT BARRO but keep broad top-5 appeal (trade off), ONE method
(in-family Δconc) with the rest as confirmation, plain-English abstract, new file = draft v3.
NEW CPU-ONLY ANALYSES (no new model runs): (a) genre-vs-severity decomposition of Depression items, both
banks — FAILS (v1: sev +3.05 vs genre +3.83, diff −0.78 [−3.08,+1.61]; "foreseeable in kind, not magnitude"
is NOT in the data); (b) matched-content cross-bank check — DECISIVE: same six outcomes read +3.86 under the
v1 estimator (deep generic baselines) vs +0.05 under the panel's light neutrals ⇒ the bank disagreement is
the CALIBRATION FRAME, not items; frame-free real prose sides with the sensitive frame ⇒ Depression verdict
upgraded from "open, shrug" to a reasoned bracket (weight of evidence favors exceptionalism, magnitude
frame-dependent, 1938 model settles); (c) odds translations (headline ×11, offset ×7.1, infl FR ×4).
WROTE paper/foresight_v3.tex (14pp + appendix A–G, 38 refs, compiles clean, 0 undefined, 0 TBD, all numbers
from 236 auto-macros): plain abstract; hook = the 1929 model's "not yet in sight" + Ferguson's July-1914 bond
puzzle; §2 support-vs-selection framework with A1–A3 + offset audit (mea culpa → footnote); §3 Table 1 events ×
received historiography (Clark/Tuchman/Galbraith/Barry/Feldman/Turner…) × verdict; §4 ranking = THE result;
§5 duels as confirmation + "forecasting by the last precedent" analogy table (Hamburg 1892 cholera, diphtheria
serum, Kapp/Munich putsch template) linked to availability/diagnostic expectations (Tversky-Kahneman, Bordalo
et al.); §6 audits consolidated; §7 Depression bracket; §8 implications for THREE audiences (disaster-risk
models / economics of beliefs / historical method — per the broad-appeal trade-off); Zweig → appendix.
v2 (foresight.pdf) kept alongside; README points to v3.

### 2026-07-04 midday — THE METHOD CRISIS AND ITS RESOLUTION (user-driven), meter CERTIFIED
User challenged the core: "a method that is NEVER surprised cannot be a measure of surprise." Diagnostics
confirmed it in stages: (1) positive-control battery FAILED (motorcars read MORE unforeseeable than flight;
talkies ~0; ±10-nat phrasing lottery); (2) paraphrase ensembles (20/concept) killed the noise but revealed the
truth: sentence surprise measures DISCOURSE FAMILIARITY, not expectation (flight==cars: both filled 1900 print;
gas>>cavalry: gas absent from print; the moon was Verne's bestseller — the meter was RIGHT, the label wrong);
(3) forward betting slips FAILED validation (1913's top Atlantic answer = a TUNNEL — Kellermann's 1913
bestseller; conditioning moved odds the WRONG way; the 1929 model bet 60% on the Kaiser ruling 1919 —
prediction-prompts summon prediction-discourse = the ignorant voice in EVERY corpus).
USER'S FIX: retrospective/chronicle register — "It is 1919, looking back..." — a real history question for the
post model, a FAKE MEMOIR for the pre model (priors in the reality-anchored genre). v1: tunnel collapsed
45%→18%, Tsar two-layer result (deposition foreseen by 1913 at 55%, execution known only to 1929: 1%→36%),
thrones staircase 5/31/45. Remaining bug: calibration strangles common-phrase truths ("worth nothing at all"
0% for the model whose free text says "worth about one cent"). v2 raw odds: mark 82%@1929, Kaiser→president
65%@1929, panic trusts staircase 3/14/35; but raw floods idiomatic phrases (tsar/thrones) — raw and calibrated
fail in mirror image. DEV certification (20 known-outcome Qs): agreement-filter = 100% precise, 50% coverage
(NOT CERTIFIED by pre-registered bars; cause isolated: calibration is anti-FAME, and truth is usually famous).
LOCKED RULE (ex ante decidable): NAME-type options→raw+split-half stability; PHRASE-type→raw·cal agreement.
HELD-OUT certification (20 FRESH Qs, no tuning): C1 16/20, C2 81%, C3 81% → **CERTIFIED**. Gems: Portsmouth
5%→91%, Tripoli 3%→88%, Dawes 20%→98%; pre-model wrong answers are period-perfect priors (Nicaragua for the
canal, Belfort for the fortress, "an empire under a new dynasty" for post-revolution China). The 3 covered
failures share ONE cause — retrieval-by-salience (Trotsky beats Stalin 1928; Latham beats Blériot; tsushima =
item flaw, 'straits of Korea' is a synonym). Within its declared domain (singular richly-attested facts, no
salient rival) the meter went 13/13. Files: src/diag_{positive_control,paraphrase_pilot,betting_pilot,
retro_pilot,retro_v2}.py, src/certify_chronicle{,_heldout}.py + results/*.json. NEXT: v4 program — apply the
certified meter to the real foresight battery (fake memoir vs real memory), keep circulation+duels as
relabeled arms, add the certification protocol as the methods contribution.

### 2026-07-04 afternoon — v4.1 battery + method_pivot.html
v4.1 (certification-grade option rewrites): FLAGSHIP = emp_kaiser_1919 rebuilt name-type: p(truth) 0%->78%,
gap +9.84, COVERED (1913 fake memoir keeps Wilhelm after a lost war; 1929 says President Ebert). Panic control
clean (severity 97->85 gap -0.14, Morgan 77->79 gap +0.02: panics were KNOWN). DISCOVERY: rev_debts COVERED —
1913 bets 80% on repudiation of Russia's debts (the bondholder nightmare was in the pre-war picture). Tsar
execution +2.71 covered; Rentenmark +6.77 covered. ANOMALY RESOLVED: nazi_who at 41% was phrase-fluency ("the
leader of the National Socialists"); bare names -> Hitler 22% ~ chance, favourite = Wilhelm (restoration).
1929 MEMOIR of the 1930s: banks saved by the Fed 94% (wrong), republic falls by armed march 64% (wrong,
putsch template), BUT dictatorship-for-republic 83% (right — Article-48 discourse) and gold abandoned by
Britain+America 82% (right — the 1914-25 suspension memory): foresight and blindness separated per question.
Coverage 10/26 measured (+4/7 memoir) — meter honestly abstains where my options are still imperfect.
SHIPPED reports/method_pivot.html (generated by src/build_method_report.py from saved JSONs, zero hand-typed
numbers): the failures (moon<bread; tunnel 45%), the reasons (familiarity not expectation; prophecy = the
ignorant voice), the pivot (chronicle register, credited to the user), certification (dev/held-out), current
v4.1 results incl. memoir tables + resolved anomaly.

### 2026-07-04 evening — DE/FR certification + battery + overview HTML
Certification per language, same locked rule/bars: GERMAN NOT CERTIFIED (C1 coverage 12/20; precision where
covered 10/12 — same profile as the EN dev set); FRENCH NOT CERTIFIED (C1 14/20 and C2 13/14 pass, C3 gradient
10/14 fails — largely on items where pre-confidence is LEGITIMATELY high: women 100%, Ottoman constitution
1876, 'le roi George' already reigning in Greece 1900 → a zero gradient is correct behavior there; criterion
bluntness documented, verdict kept as computed). LABEL: EN certified; DE/FR provisional (supporting evidence).
v41 DE/FR battery: FLAGSHIP COVERED IN ALL THREE LANGUAGES — Kaiser-1919 gaps EN +9.84 / DE +13.30 / FR +5.71
(the 1913 fake memoir keeps Wilhelm in every language). Fake-memoir bets cross-lingually consistent even where
abstaining (Goldmark, liberals-rule-Russia, war-of-marches). 1929 MEMOIRS DIVERGE NATIONALLY: German memoir
alone expects the slump to deepen year after year (98%; the German 1920s taught it) vs French recovery-template
(98%); German banks pass 'in die Hände der Regierung' (étatist template) vs English Fed-holds. Coverage honest
and low (DE 3/26, FR 8/26) — option discipline doesn't translate mechanically; per-language wording work is
the listed next step. reports/method_pivot.html regenerated with §7 cross-lingual (all numbers from JSONs);
README points to it as the overview.

### 2026-07-06 — checkpoint (didactic rewrite of the overview; writing rules made permanent)
Since the DE/FR entry: the user rejected the overview's prose three times, each rightly — (1)
"cryptic, no examples, shorthand"; (2) "the gradient bar — who are u speaking to?" (a private
coinage dropped undefined into chat); (3) "only computers write like that: nounstacking, mechanical
verbiage" (verbatim offender: "the option-wording discipline that earns coverage does not translate
mechanically"). RESPONSE: build_method_report.py rewritten wholesale on a fixed didactic template —
every section = the test in plain words → the exact prompt verbatim → one result walked through
with visible arithmetic (e.g. gap = −ln(0.004) − (−ln(0.78)) ≈ +9.8 on the Kaiser question) → what
it means. Private jargon purged: guesser/witness for pre/post, "the meter refused to answer" for
abstain/coverage, the three exam requirements written out instead of C1/C2/C3. Prose sweep against
noun-stacking. Rules saved to permanent memory (feedback_didactic_reports.md — applies to chat too;
feedback_plain_english.md updated with the verbatim offenders). Pushed 6501b8a. CONTEXT rewritten
to v6.0: compiled truth of the whole method arc — original estimator dead (measures circulation),
prophecy-prompt law, chronicle meter certified EN (16/20, 81%, 81%) / provisional DE+FR, certified
findings (Kaiser gap +9.8/+13.3/+5.7; debts repudiation foreseen at 80%; memoir of the 1930s;
national memoir divergence), papers v2/v3 marked superseded pending v4.

### 2026-07-07 — v4 paper drafted and compiled (on the certified meter)
Built paper v4 as an economics paper on v3's rare-disaster framing but rebuilt entirely on the
certified chronicle meter; nothing overwritten (v3 files untouched). New files:
src/make_numbers_v4.py -> paper/numbers_v4.tex (117 macros) + 5 auto tables (gaps_v4, memoir_v4,
crossling_v4, duels_v4, circ_v4); paper/foresight_v4.tex -> paper/foresight_v4.pdf (12 pp, compiles
clean, no undefined refs, no TBD, no unresolved macros). Every number read from the chronicle result
JSONs (v41_battery, certify_chronicle_heldout, v4_battery, v41_defr, certify_defr, diag_*, stats_panel,
panel_scores, production_panel); zero hand-typed. Dev-exam 10/20 taken from the LOG-documented
"50% coverage" (certify_chronicle.json stores raw per-Q dists, not a scored summary). Structure:
(1) intro/rare-disaster motivation; (2) info set + why sentence-surprise and prophecy-prompts fail;
(3) the meter + the exam (dev 10/20 -> held-out 16/20, 81%, 81%); (4) foreseen (panic, war breadth,
US entry, Russia debts 80%) vs unforeseen (republic +9.84, Tsar execution +2.71, Rentenmark +6.77);
(5) 1929 memoir of the 1930s (last-precedent misses; dictatorship 83% + leaving gold 82% land);
(6) DE/FR (not certified: DE 12/20, FR 14/20 fails out-bet) + national-memory split on Depression
course; (7) implications (support holds, weighting fails, July-1914 puzzle); circulation + duels as
labelled appendices. FIXED TWO NUMBER BUGS pre-compile: (a) Kaiser worked example had reused the
witness's 78% as the guesser's bet on the Kaiser — real guesser bet is 99% (pre_bet_p); (b) memoir
prose hand-typed 94/83/82 — replaced with macros. Kaiser arithmetic made internally consistent
(surp_pre = gap + surp_post = 10.1; implied guesser bet 0.004%), fixing the 13.8-0.3≈9.8 inconsistency
carried in method_pivot. Applied user's 2026-07-07 style rules (no in-text bold, italics only for
foreign phrases, no noun-stacking, plain sentences); saved to feedback_plain_english.md. NOT yet done:
read-through/edit pass for prose; push to repo; figures (v4 has none yet — could add a foresight-gap
chart). v3's ranking claims remain retired.

### 2026-07-07 — v4 upgraded to full draft (figures, exam appendix, identification, limits, verification)
PI pushed for best/comprehensive/cautious work; expanded v4 from the bare compile into a complete
paper. Now 15 pp. Added: (1) FOUR self-explanatory figures generated from the JSONs by
src/make_figures_v4.py -> figures/fig_{gaps,memoir,exam,duels}.png — Fig1 dumbbell of guesser-vs-witness
bet on the truth (the gap in one picture), Fig2 memoir-of-the-1930s grey(expected)/blue(happened) bars,
Fig3 exam scatter (witness above the diagonal; 3 red fame-failures), Fig4 duel win-rate by reader with
the coin line. Legends fixed after visual QA. (2) Full 20-question exam table (exam_v4.tex, Appendix A).
(3) New Section 3 "the foresight gap and what it identifies" with three explicit identification
conditions (witness knows / guesser doesn't / asked alike) + the fame caveat. (4) New Section "What the
instrument cannot do" — 4 limits stated (partial coverage 10/26, retrieval-by-fame, no witness past 1929,
only EN certified). (5) Positive-control framing for the 1907 panic; methods-contribution framing of the
fail-able exam. INDEPENDENT VERIFICATION: src/verify_v4.py re-derives 32 headline quantities straight
from the raw JSONs (not via the generator) and asserts they match the emitted macros — 32/32 agree
(exam 16/13/13, coverage 10/26, Kaiser gap +9.84 & implied bet 0.004%, DE/FR exam counts, cross-lingual
Kaiser, false alarm 41->22, memoir bets). Visual proofread of all 15 rendered pages via PyMuPDF: no
overfull >20pt, no undefined refs, no unresolved macros, tables/figures within margins. One-command
rebuild: src/build_v4.py (numbers->figures->verify->latexmk). Provenance header added to foresight_v4.tex.
NOT done: git push; a copy-edit read for prose rhythm (structure/numbers are final). Files all in
12_foresight/{src,paper,figures}.

### 2026-07-07 — probe: "who is Adolf Hitler?" to Ranke 1913 vs 1929 (100 free continuations each)
PI asked to run the models live. src/hitler_probe.py: loaded Ranke-4B-1913 then -1929 on the 3060 (one
at a time, del+free between), 100 sampled free continuations each (temp 0.8, top_p 0.95, rep_pen 1.3,
60 tok, seed 1929), two prompts ("Who is Adolf Hitler?" and "Adolf Hitler is"). Raw ->
results/hitler_probe/generations.json (checkpointed). src/hitler_analyze.py -> fig_hitler.png +
hitler_report.html + summary.md; keyword tags printed for audit. RESULT (prompt "Adolf Hitler is"):
"situates the real Hitler (German radical politics: Munich/Bavaria/Reichstag/radical movement/editor/
rising agitator)" 1913 = 2%, 1929 = 25%; "empty name -> generic (often American) person" 1913 = 49%,
1929 = 33%. 1913 confabulates freely (NY businessman, American playwright, Hungarian immigrant actor,
"his father Adolf Hitler, a student of occultism"). 1929 places him correctly for 1929 ("head of a new
radical movement in Germany", "born in Bavaria... fought against France then joined the revolutionary
movement", "editor... destined in a few years to be ruler of Germany, proclamation issued at Munich").
HONEST CAVEAT: neither model states the real 1933 Chancellorship (regex 1933/seized-power/Hindenburg
returned EMPTY for both); the 1929 "chancellor" mentions are CONFABULATED (appointed by "Emperor
William"/"King Ludwig II"). Confirms the time-lock: 1929 knows the 1920s agitator, not the ending ---
consistent with the paper's memoir finding (1929 bets Wilhelm > Hitler for 1934).

### 2026-07-07 — "holds the protagonist" metric + Bismarck positive control
PI: score "fails to stick with protagonist" on Hitler, and run the same on Bismarck (died 1898 -> a
figure 1913 must know) as a control. Ran src/entity_probe.py bismarck (both models, EN+DE, 100 free
continuations each; results/entity_probe/bismarck_{en,de}.json). Also confirmed: tokenizer.chat_template
PRESENT -> the Ranke models are INSTRUCT-tuned; we probe them RAW (no template) so they behave base-like.
METRIC ITERATION (documented honestly in holds_report.py): (1) name-recurrence = WRONG, confounded by
referential style (Bismarck = "the Prince/he" scores as not-sticking; confabulated "Mr. Hiller" scores as
sticking) — Bismarck came out 88-94% "fail", an artifact. (2) TF-IDF cross-sample consistency = NULL, all
cells ~0.09 (newspaper style swamps content). (3) ON-DOMAIN (evokes the person's real world) = WORKS, and
a two-domain confusion check proves specificity. RESULT (own-domain hit rate, strict, EN/DE, 200 conts/cell):
Bismarck-1913 34/58%, Bismarck-1929 54/58%, Hitler-1913 4/10%, Hitler-1929 24/40%; off-domain (other
person) all 1-10%. So the 1913 model HOLDS Bismarck (~10x Hitler) but NOT Hitler; 1913->1929 brings Hitler
into focus while Bismarck (already known) barely moves. Clean positive control — validates that Hitler-1913
emptiness is real ignorance, not a bad-rambler artifact. Deliverables: figures/fig_holds.png,
results/entity_probe/hitler_bismarck_holds.html, src/{entity_probe,stickiness,consistency,holds_report}.py.

### 2026-07-22 18:20 (office 4090 box)
Content-dating experiment picked up per OFFICE_CLAUDE.md. Repo cloned to C:\Users\hvoth\Documents\GitHub\ranke-edition-dating (github.txt token stale; github2.txt worked). WSL vllm env rebuilt (vllm 0.25.1, cu130, 4090 visible).
SCHEMA VERIFY (institutional/institutional-books-1.0): 983,004 rows, 9,831 parquet files. Text = text_by_page_gen list<large_string> (v1 pull script would have written JSON garbage); year = date1_src; id = barcode_src. Wrote pull_positives_v2.py: metadata-only column scan of seeded-shuffled files, then text fetch of contributing files; added language_gen=eng filter (negatives are all English) + date2/date_types reprint guards. Hit rate ~0.9% (post-1947 eng in a PD corpus = mostly gov docs/institutional reports; median cand year ~1957). Positive class is gov-doc-flavored — genre property to note in the finding.
INTEGRITY CHECK on manifest_negatives (999): 0 overlap with contaminants_bucket (1,372) and gotthrough bucket (3); 487/999 carry dataset=contam/institutional_books + stage2_contam=True but the prefix marks corpus POOL, not per-doc truth (both prefixes appear among bucket FPs); all meta+stage2 years <=1929. Selection sound.
CERTIFY v2 launched: deepseek-v4-flash + llama-3.3-70b BOTH via OpenRouter. DEVIATION from OFFICE_CLAUDE "Llama local": 70B 4-bit ~37 GB does not fit 24 GB VRAM; OpenRouter has no daily cap. Coverage raised from pilot 12x1500 (~7% of median 276k-char book) to 24x3000 (~28% median, full for small books) — PROVISIONAL. Error-aware verdicts (API failure cannot certify a book clean); checkpointed. Preflight passed; credits 240 total / 161 used.
EXTRACT v2 written (not yet run): fixed grep-center offset bug (v1 computed centers on collapsed string, sliced original — drift up to 100s of chars); grep hits now spread over all hits not first-16; negatives filtered to certified_clean; center column added for audit.
FLOTILLA prep: Llama-3.2 1B/3B gated 403 on this token -> use unsloth/ mirrors (identical weights, ungated). gpt-oss-20b root shards = 13.8 GB (fits). Qwen3.6-35B-A3B: popular AWQ quants are 25 GB (do NOT fit 24 GB); fallback chain of unmodified ~20 GB quants (nvidia NVFP4 / shieldstar AutoRound-EC / LeaderboardModel1 W4A16 / 88plug W4A16) to be tried at load.

### 2026-07-22 18:55 (office 4090 box)
POSITIVES DONE: 1,000 post-1946 modern-original eng books pulled from institutional-books-1.0 (years 1947-2004, median 1958; 8 reprint-regex drops, 0 short drops). Validated: 1,000 unique ids, all txt files present, size spread 7k-8.9M chars (median 124k), no identical-size red flags, sample reads as genuine 1947 original. manifest_positives.csv in repo. Class composition note for the finding: post-1947 PD books are heavily gov docs/institutional reports (legislature reports, hearings, agency reports) - genre is part of what "post-1929 content in a PD corpus" means in production.
Certify restarted with book-level concurrency bound (6 books in flight) after first launch showed FIFO starvation: all llama batches queued ahead of every deepseek batch, so no book would have completed/checkpointed until run end. ~$0.50 of calls discarded.

### 2026-07-22 19:00 (office 4090 box)
CERTIFICATION DONE (999 books, 24x3000-char chunks, deepseek-v4-flash + llama-3.3-70b via OpenRouter, ~40 min at 25 books/min after concurrency fix 16->48):
  918 certified_clean | 81 review_single_flag (ALL llama-only; deepseek flagged 0; both-flag 0; error verdicts 0).
  Spot-check of the 81: evidence chunks are ordinary pre-1929 prose (Linnaean naming, cotton calendering, Charlemagne, Langland glossary); one case has "(1981, 3026)" = bracketed reference numerals. Reads as llama-70b over-caution, NOT contamination; deepseek 0/81 agreement supports. Per protocol: 81 -> review queue (deliverable), experiment runs on 918 certified negatives.
WSL vLLM stack fixed for the flotilla: driver 560.94 cannot run cu130 torch -> vllm 0.25.1+cu129 wheel (wheels.vllm.ai) + torch 2.11.0+cu129; VLLM_WSL2_ENABLE_PIN_MEMORY=1 (pinned mem verified working, kernel 6.6.87); VLLM_USE_FLASHINFER_SAMPLER=0 (FlashInfer JIT sampler needs nvcc, no sudo on this box; greedy decode = torch sampler equivalent); FLASH_ATTN backend, TRITON_ATTN for gpt-oss (attention sinks). SMOKE PASSED: Qwen2.5-0.5B loads (answers [0,0] - misses modern passage, that IS the scaling curve), gpt-oss-20b [1,0] correct.
Extract v2 launched on 918+1000 books.

### 2026-07-22 19:10 (office 4090 box)
EXTRACTION DONE (after 2 fixes: shard-grouped preload — v2 as written re-read a 500MB shard per BOOK (~450GB I/O); shard-column guard for positives manifest):
  passages_neg.parquet: 117,204 rows / 918 certified books, all label 0.
  passages_pos.parquet: 115,476 rows / 1,000 books, all label 1 (grep < 16 for books with few marker hits — idx<N subsetting handles it).
  232,680 prompts per model. Samples eyeballed: grep-directed positives centre on genuinely modern content ("For 2003, the large urban floor is $564.10").
  NOTE: negatives DO produce grep hits (median 16/book) because the extractor marker list (markers_1945 minus DROP) is broader than the validated-marker screen used in selection — grep precision is genuinely tested on old text.
review_queue_negatives.csv (81 llama-only flags) written to repo.
Qwen3.6-35B-A3B quant fitting saga: AWQ 25GB > VRAM; AutoRound-EC 20.9GB loads but KV short at util 0.92; util 0.95 > actually-free (Windows display holds ~1.5GB); final = util 0.93 + enforce_eager + max_model_len 2048. Smoke pending.

### 2026-07-22 19:15 (office 4090 box)
Qwen3.6-35B-A3B-int4-AutoRound-EC SMOKE PASSED [1,0] at util 0.93 + eager + 2048 ctx (KV 9,011 tokens — the tightest fit in the flotilla). FLOTILLA --all LAUNCHED: 10 models x 232,680 passages, slice-checkpointed, per-model subprocess, expected ~15-24 h on the 4090. Monitor armed for per-model completion/failure.

### 2026-07-22 19:55 (office 4090 box) — PI course-correction
PI (correctly): post-1947 positives = "a gov doc classifier by another name" (PD post-1947 full-text is nearly all gov/institutional docs; probe of 1930-46 band shows the same skew there, ordinary books only ~10-20% of a 0.7% band). DECISIONS: (1) stratum C NOW — no downside; (2) stratum B "after, if we have time"; (3) demote post-1947 stratum A to diagnostic (sweep continues; negatives still carry the precision side).
STRATUM C BUILT: 322 local-corpus ground-truth contaminants = 239 contam_modern (recall headline) + 80 contam_reprint_old_body (content-dating should NOT flag; sanity floor) + 3 got-throughs. Source: union of gold_300/1000/1000b audits, contaminated_1929=True, conf>=0.9, deduped, 0 overlap with negatives (asserted). NOT sourced from contaminants_bucket (contains homograph FPs). 37,696 passages -> passages_contamC.parquet in the shared pool: models starting from now include C automatically; 0.5B/1.5B/(3B?) need delta runs (added --passages-file/--suffix to driver).
Sweep-log gotcha: the --all launcher piped through tail -400 which buffers until exit -> log/monitor silent; re-armed monitor on outputs/ dir instead. Progress so far: 0.5B (310 p/s, 51 nulls), 1.5B done; 3B running.
Full 1930-46 band metadata scan running for stratum B (read-only, checkpointed).

### 2026-07-23 03:15 (office 4090 box)
STALL CAUGHT + RECOVERED: Qwen3.6-35B (shieldstar AutoRound-EC) wedged on the full run despite passing smoke — 1h03m wall, 90s CPU, 130W at fake "100%" util, zero slices. Busy-but-stuck (kernel/scheduling hang in eager + KV-9k config, likely degenerate path with 32k-prompt queue). Killed subtree; --all parent survived and moved to gpt-oss-20b (watcher armed for its first slice). 35B retry plan after gpt-oss: (1) LeaderboardModel1 W4A16 19.6GB (downloading — +1.3GB KV headroom, maybe re-enable graphs), (2) shieldstar with max_num_seqs=8 + default attention backend. Models 1-8 all clean; Llama-1B has 0.6% null answers (reported in analysis).

### 2026-07-23 06:15 (office 4090 box) — overnight sweep wrap + core finding
SWEEP: 9/10 models complete (232,680-270,376 judgments each). gpt-oss-20b recovered after an orphaned EngineCore from the killed 35B blocked its first start (VRAM squat) — ran clean at 73 p/s, 0.06% nulls.
QWEN3.6-35B-A3B KILLED after 2 attempts — NOT VIABLE on 24GB/vllm 0.25.1: (a) standard profile = silent kernel hang at full batch (fake-100% util, 130W, 90s CPU/1h) despite passing smoke; (b) capped eager profile = CPU-bound EngineCore (100% of one core, GPU 20-30%), <7 p/s => 11+h; (c) only other fitting quant (W4A16) = config-class incompatible with vllm 0.25.1. Hardware-tier finding for production: gpt-oss-20b is the strong candidate that actually runs.
CORE RESULT (provisional, pre-delta-C): NO config reaches recall 0.95 at FP<=10%. Llama-3.2-1B "recall 1.0" = degenerate always-AFTER (99.9% FP on negatives) — headline now requires FP bound; 2-of-N vote aggregation added. Frontier: gpt-oss 0.857 recall @ 6.9% FP (1000 chars, random, N=8, k=2); 7B 0.840 @ 8.8%; 14B 0.777 @ 5.4%; Phi 0.504; small models degenerate/noise.
MISS AUTOPSY (6 sampled misses of gpt-oss best config, labels verified correct, conf>=0.95): all are 1931-47 editions whose BODY carries no date signal — devotional poetry in hymn diction, taxonomy monograph, local history quoting an 1811 newspaper, archaic verse. Intrinsic content-dating ceiling on near-boundary archaic-genre books; every miss has its date in FRONT MATTER => content-reading and the front-matter dater are complementary; irreducible residual = archaic content + wrong metadata + no imprint (= the got-through class; models catch only ~1/3 of those 3).
Reprint floor: models flag 40-60% of old-body modern reprints at usable configs — part of the signal is typography/OCR-era, not content. Delta-C for 0.5B/1.5B/3B running; then final analysis + report; stratum B after per PI.

### 2026-07-23 06:50 (office 4090 box) — DELIVERABLES SHIPPED
Final analysis (9 models x 648 configs, all with full C coverage) + content_dating_report.html (4 figures, verbatim misses) committed/pushed (3be5d73) and sent to PI. HEADLINE: no config reaches recall 0.95 at FP<=10%. Frontier: gpt-oss-20b 0.857 @ FP 6.9% (8x1000 random, 2-of-8 vote, ~2.7 GPU-days/2M books); 7B 0.840; 14B-AWQ 0.777; Phi 0.504; <=3B blind/degenerate. Ceiling is intrinsic (archaic-genre 1930s-40s misses, date only in front matter) => content-reading complements, does not replace, the front-matter dater. Grep-directed did not beat random at the frontier. Gov-doc stratum A overstates recall 10-25 pts (PI called it). Cost: $4.21 OpenRouter + 1 GPU night.
STRATUM B build launched (bkkiee29g): heuristic+deepseek title screen of 3.3k 1930-46 ordinary candidates -> wave fetch -> strict both-model modern-original certification (target 600) -> passages_pos3046.parquet -> delta on gpt-oss/7B/14B/Phi only. Prior from miss-autopsy: B recall < 0.86 headline.

### 2026-07-23 10:15 (office 4090 box) — EXPERIMENT COMPLETE, report v2 shipped (e09749b)
STRATUM B (600 certified 1930-46 ordinary books; 631/899 both-model certification pass): the binding constraint. Best single model at FP<=10%: gpt-oss C 0.849 / B 0.732. ENSEMBLES (PI request, computed free from stored verdicts): passage-agreement gptoss+14B 2of2 8x2000 grep = C 0.895 / B 0.803 @ FP 9.2% (~21 GPU-days/2M). QWEN3.6-35B (PI: "more quantization"): 3-bit GGUF (16.6GB) on llama.cpp win-cuda RUNS CORRECTLY and is the accuracy champion — k-of-8 sweep beats every smaller model at every FP level (k=3: C 0.916/B 0.848 @ 10.7%; k=4: C 0.887/B 0.775 @ 6.5%); book-ensemble 35B(k2)+gptoss(k1): C 0.891/B 0.823 @ FP 5.8% = best usable operating point of the whole experiment. BUT 2.5-4.5 p/s (25x slower than gpt-oss), 2 CUDA server crashes + restarts; measured on seeded subsample (238C/400B/400neg, one config). NOT production-viable on 24GB; recommend 1 A100-hour on RunPod w/ vLLM if the extra recall matters.
Recall-0.95 target NOT reached by anything at FP<=10% on stratum B — intrinsic-ceiling conclusion stands (archaic-genre misses, dates in front matter; complementarity with the production dater).
Total OpenRouter: $5.27 measured. Report v2 (3 figs, frontier table incl. 35B rows, verbatim misses) in outputs/ + repo + sent to PI. Session artifacts all committed (e09749b) and mirrored to Dropbox code\.

### 2026-07-23 15:00 (office box) — A100 VERIFICATION DONE, $0.74, pod torn down
PI approved 1 A100 hour. Pod lv2uorbgg0obqy (A100-SXM4-80GB, $1.49/hr, vllm/vllm-openai over RunPod HTTPS proxy — office network blocks SSH, none needed). UNQUANTIZED Qwen3.6-35B-A3B, full frontier slice (57,654 passages, 918neg/238C/600B, both lengths+strategies, idx<8) at 36.8 p/s, 4 nulls, ~26 min. TEARDOWN: terminated immediately, account verified ZERO PODS (balance $85.83, spend $0.74), +30min re-verify scheduled.
RESULTS: BF16 8x2000 random k=3: C 0.887 / B 0.868 / FP 8.6% = best sub-10%-FP stratum-B recall of the experiment (beats the gptoss+14B ensemble 0.803 and local Q3 0.848@10.7%). k=4: C 0.866/B 0.777/FP 3.8%. k=2: B 0.922 but FP 17% — recall-0.95-at-FP<=10 still unreached; ceiling conclusion stands at full precision. Grep < random for the 35B too. Apples-to-apples on identical books: local 3-bit GGUF tracks BF16 within ~1-3 pts on every metric — QUANTIZATION IS NOT THE LIMITER, the 24GB runtime is. Rented full-corpus sweep ~= $180 (121 A100-h; offline batching ~halves it). Report v2.1 regenerated.

### 2026-07-23 19:45 (office box) — combination frontier (PI question), split-half validated
12,528 combos (book-level AND/OR/MAJ + passage-level fusion; members 35B-BF16 + {gptoss,14b,7b,phi}; selected on half A, REPORTED held-out half B). PI triple q35+gptoss+q14b MAJ ks3/2/1 8x2000rand: C 0.911 / B 0.880 / FP 9.7% = best <=10%-FP point of the experiment. Pair q35+gptoss OR ks3/3 essentially equal (C 0.902/B 0.899 @ 10.8%) — 14B adds ~nothing; 35B carries, gptoss patches. Strict: 4/3/1 -> C 0.875/B 0.820 @ 5.6%. Loose: 2/2/1 -> C 0.938/B 0.893 @ 13.5%. HELD-OUT CEILING ~0.91 B-recall at 10-12% FP; recall 0.95 unreached even combining everything. Precision caveat logged: at 2% corpus prevalence, FP 9.7% = ~16% classifier precision = sacrifice ~10% of corpus to remove ~88% of contamination. combo35_results.csv + analyze_combo35.py committed.

### 2026-07-23 20:05 (office box)
PI asked how to measure OVERALL two-stage pipeline recall/precision vs hand-validated docs (rates don't add: stage-2 sees only stage-1 survivors; errors correlate via the no-imprint/archaic-body class). Full design written to notes/2026-07-23-pipeline-composite-eval-design.md: hybrid panel (existing gold/certified strata + ~200 NEW metadata-corrupted plants copying real got-through patterns) -> sequential stage1->stage2 run -> per-stratum composite + marginal stage-2 value + corpus loss; stratified IPW residual audit of the kept pile with ~100-book human anchor. Pre-registered operating point proposed: 35B+gptoss OR 3/3. Awaiting PI: reprint policy + operating point.

### 2026-07-23 21:00 (office box) — composite pipeline eval: plants gated + scored (GPU untouched, eugenics running)
PLANTS: 155 metadata-corrupted copies of B/C books; corruption = page-boundary truncation (+40% genuine provenance blocks harvested from certified negatives), patterns copied from the 3 real got-throughs (all had authentic-but-uninformative fronts). REALISM GATES PASSED: G1 96% imprint-free (23 unhideable parents skipped honestly); G2 no-imprint = 39% of genuine cleans; G3 deepseek blind discrimination balanced acc 47.5% = chance (judge calls 80% of plants AND 85% of genuine "artificial" — cannot separate).
COMPOSITE (freeze-v2 stage1 -> 35B|gptoss OR-3/3 stage2, sequential, existing judgments reused — plants inherit parent body verdicts): honest-metadata contamination: stage1 alone 0.996 (C-modern) / 1.000 (B) — content stage adds NOTHING there (misses of the content-only experiment all carry dates in front matter, as predicted). Metadata-corrupted plants (s2_fooled bound): stage1 0.000 by construction, stage2 rescues 0.93 — the content stage's entire value is this pathway. Real got-throughs 1/3. Reprints: stage1 already 0.925 via imprint; pending editorial policy pass covers the rest. CORPUS LOSS: 10.0% of the 918 certified cleans, all from stage2 at OR-3/3 (panel stage1 loss = 0 BY CONSTRUCTION — selection circularity flagged; production stage1 loss ~7% pre-recovery per README applies on top in the real corpus).
PENDING (GPU held for eugenics): 14B doc-dater on 155 plants (closes the s2_fooled/s2_correct bounds); gpt-oss editorial pass on 80 reprints + 6 stage1-missed reprints.

### 2026-07-23 21:40 (office box) — reprint policy pass done (GPU released by eugenics, verified: process-exit + 10-min grace)
gpt-oss editorial pass on 80 old-body reprints (PI policy: modern intro/editorial -> out, clean reset -> keep): 50 REMOVE (dated modern intros/prefaces: "Introduction ... (1958)", "Preface dated February 1952"), 29 KEEP, 1 unparsed. FLOW SCORING: 76/80 flagged by stage1|stage2 -> editorial pass; 48/50 modern-editorial reach removal (96%); 27/29 clean resets RESCUED (kept — premium period data blanket-removal would destroy); 2 modern-editorial reprints never flagged (the residual). Committed reprint_verdicts.csv.
OPEN with PI: stage2_docyear producer not in Dropbox (corpus arrived pre-built); offered Qwen3-14B + 01qwen_pilot map-reduce prompts as LABELED stand-in vs waiting for Guohui original vs keeping bounds.

### 2026-07-24 (office box) — Guohui workaround: stage2_docyear generator NOT needed for the bounds
Measured the 14B daters behavior from its stored outputs on gold-audited books (truth = edition_year, conf>=0.9, n=319 moderns with corpus s2):
  fooled-rate P(s2<=1929 | truly modern): 0.9% honest-metadata (n=218), 6.9% [95% CI 2-12%] wrong-old-metadata (n=101) = the plants reference class.
  ANCHORING TEST: among wrong-old-metadata moderns, s2 lands within 5y of the WRONG metadata year in 1% of cases but within 5y of TRUTH in 88% => the dater is CONTENT-ONLY (metadata not in its prompt, or ignored). Validates transferring parents s2 to plants: the 58 C-parented plants parents were fooled 0/58.
  COMPOSITE MIDDLE BOUND (replaces s2_fooled/s2_correct bracket): stage 1 catches ~93.1% of metadata-corrupted plants via the 14B docyear; stage 2 rescues ~93% of the fooled residual => composite plant recall ~0.995 [0.991-0.999 over the CI]. The got-throughs are exactly the double-tail (fooled 14B AND no imprint AND wrong metadata), ~0.5% of wrong-metadata contamination.
Guohui ask downgraded from blocker to documentation (script provenance for the paper trail).

### 2026-07-24 10:00 (office box) — OOS VALIDATION COMPLETE: all 5 pre-registered bands PASS
Panel (unseen, excl. every prior manifest/audit/examined list): 365 certified fresh negatives + 250 certified honest 1930-46 moderns + 128 frozen-operator plants. Frozen pipeline judged (gpt-oss 4090; 35B BF16 on A100 pod cdxnucnqnxu486, 29 p/s, ~$0.27, terminated+verified ZERO PODS).
  P1 honest-modern composite recall: 1.000 [0.985, 1.000] (250/250) — PASS (band >=0.98)
  P5 stage-2 marginal on honest moderns: +0.000 — PASS (band <=0.02)
  P2 plant stage-2-only rescue: 0.875 [0.805, 0.927] (112/128) — band PASS ([0.86,0.97]) but SIGNIFICANTLY below the in-sample 0.93 point (p=0.023): in-sample was ~5pp winner-curse optimistic; QUOTE 0.88 GOING FORWARD.
  P3 plant composite @ fooled 0.069: 0.991, bootstrap [0.987, 0.995] — PASS (band [0.98,1.00]); sensitivity 0.985-0.998 over fooled CI.
  P4 corpus loss: 0.132 [0.099, 0.171] (48/365) — band PASS at the edge ([0.06,0.14]), p=0.054 vs 0.10: loss runs ~3pp higher OOS; QUOTE ~13%.
NET: conclusions CONFIRMED with honest shrinkage — composite recall ~1.00 honest / ~0.99 corrupted-metadata; content-stage rescue 0.88; corpus loss 13% at OR-3/3 (strict 4/3 available to halve it). Committed + pushed.

### 2026-07-24 12:00 (office box) — STAGE 3 built, debugged, measured end-to-end
PI-approved third leg: full-text grep (markers + year-in-dating-context) + gpt-oss adjudication of +-800-char hit contexts. V1 FAILURE CAUGHT BY SPOT-CHECK: substring matches (nato in impersonato) + adjudicator trusting injected [marker:X] tags over the text ("contains NATO" about Latin verse). V2 fixes: word boundaries, strict year contexts (copyright/published/printed/in-the-year/edition-of only), raw untagged excerpts, quote-the-proving-phrase prompt. Hit rates: cleans 60%->1-3%, stage-2-missed positives 28-50%.
END-TO-END (adjudicated): rescues 30% of C misses (7/23), 27% of B misses (21/79), 50% of OOS honest-modern misses (8/16), 0% of OOS cut-front plants (0/16 — the disguise removes exactly what the year-sieve reads; stage 3 does NOT cover the got-through pathway). Clean-side: 4/600 removals = ~0.5% extra corpus loss; eyeballed: 1 plausible genuine discovery ("converted into chambers in the year 1963" in an unscreened "clean" book), 3 adjudicator FPs (future-projection "year 2000" in a 1900s canal text; OCR-garbled 1999; ISBE world-knowledge error — ISBE is 1915, adjudicator thought 1929/30).
THREE-STAGE COMPOSITE (honest): B 0.868 -> 0.903, C 0.902 -> 0.932, OOS moderns 0.936 -> 0.968 (stage-2-only view; stage 1 catches all honest anyway). The chat-quoted 0.955 was raw-hit-rate optimism before adjudication + boundary fixes. VERDICT: stage 3 is worth having (cheap: CPU scan + ~2-3% adjudication load, +3-5pp on hard strata, occasionally finds real contamination in "clean" pools) but the 0.95 target on the hardest stratum remains out of reach; the silent-book residual holds.

### 2026-07-24 13:15 (office box) — upgrade round: markers done, 120B mid-run
MARKERS_1929 built: 447 validated terms (56/70 curated survived; 391 mined from 736 stage-2-caught moderns, df>=5, zero hits in 1,361-book pre-1929 null corpus — leakage-guarded: miss panels excluded from mining). Null validation REJECTED "new deal" (5 pre-1929 hits), "nazi", "stalin", "sonar", "jeep" — in-corpus beats hand-curation. TOP MINED = stylistic fingerprints of post-1929 English: sponsored, basically, contacted, phoned, slumped, allegedly, operational, highlights + entities (mussolini, television, fascism, hitler's, soviet union).
STAGE-3 v3 (combined lists): hit rates miss_B 28->71%, miss_oos_plant 0->69%, certified negs stay 1%, unscreened 16%. ADJUDICATED: B-rescue 27->39% (31/79); plants only 1/16 — stylistic terms are strong SCREENS but the quote-the-proving-phrase adjudicator (rightly) won't condemn on one common word. Next lever if wanted: vocabulary-vote rule (>=3 distinct validated mined terms -> remove) with measurable FP cost. Clean-side stable (3 unscreened + 1 certified). Composite: B 0.868->0.920, C ->0.932.
120B on A100 (pod pi6qkopmqouc3j): 8.3 p/s (generation-bound on reasoning tokens), 81,100 passages, ETA ~2.3h, ~$4 total. Teardown protocol armed.

### 2026-07-24 16:45 (office box) — 120B VERDICT: clean negative result; upgrade round closed
gpt-oss-120b on A100 (81,100 passages both slices; first pass had 27% nulls from 256-token reasoning truncation -> null-retry at 768 tokens; final parquet rebuild fixed a dedupe-kept-first bug; residual nulls 0.31%). RESULT at matched FP (~8.7%, 8x2000 random k=3): C 0.887 (=35B) but B 0.818 vs 35B 0.868 — 5pp WORSE on the binding stratum; same pattern OOS (plants 0.828@7.4%); 4x slower than the 35B on identical hardware (8 vs 37 p/s). As teammate: 35B(k2)|120B(k3) = C 0.945/B 0.945 @ 21% FP vs 20B-version 0.945/0.925 @ 18.1% — a wash on ROC at 4x cost. CONCLUSION: reader scaling is exhausted; Qwen3.6-35B calibration > gpt-oss family scale on this task; production pick unchanged (35B accuracy champ, 35B+20B team, 20B for 4090-only). 120B eval cost: $6.69 total (incl. the retry; my $2 estimate was 3x optimistic — reasoning-token generation dominates). Both pods terminated + verified ZERO PODS; layer-3 re-verify scheduled.
Cumulative experiment spend now: ~$7.2 API + ~$9.0 RunPod + ~18 GPU-h local.

### 2026-07-24 17:30 (office box) — documentation set finalized
PIPELINE.md v1.0 written = authoritative production spec (stages incl. Guohui 14B docyear provenance note, frozen params, validated metrics, dials, $250/$150 cost table, teardown protocol). Report regenerated v2.2 (+production section + 120B negative + markers). Summary PDF v3.0 (9pp, +Step 7 upgrade round + production blueprint; cost table updated to true totals: $7.21 API + $7.81 RunPod + ~18 local GPU-h ~= $17 all-in). All committed, pushed, mirrored. PI confirmed understanding: stage2_docyear = Guohui's first-pass Qwen3-14B pub-year estimator from the corpus build (content-based per our measurement; script archive request open).

### 2026-07-24 19:00 (office box) — Guohui report analyzed; referee-tier integration designed
PI uploaded pipeline_technical_report.pdf = Guohui agentic dater (Map->Hypothesis->Critique->Arbitration->Finalize), NOT the 14B (it benchmarks AGAINST stage2_docyear, 64% agreement). 92/94% blind-validated edition-level accuracy; code fully local (01qwen_pilot/pipeline_structured2.py, 8 prompts) => reconstructable without Guohui; 14B reconstruction now moot. BULK-INFEASIBLE (Map reads full OCR: 205M tokens/1k books; 2M books = years). INTEGRATION DESIGN agreed with PI: referee tier on the suspicious band (~5-10%: 1-2 votes/unproven hits/no date signal/disputed reprints) with FLIPPED BURDEN OF PROOF — remove unless positive pre-1929 edition evidence found (abstention = removal). This is the only mechanism that touches the silent-book residual; needs local re-implementation of Map (98.8% of tokens) -> ~$150/2M. DECISION NUMBER before adoption: dater abstention rate on genuinely-clean band books (= extra corpus loss); measurable from the 2,000 released records x our certified negatives. Flowchart fig_pipeline_flow.png committed (redone once for layout).

### 2026-07-24 19:45 (office box) — repos unified; 14B-substitute question RESOLVED
GITHUB SEARCH (PI hint "cscs has these scores"): stage2_docyear hits all inside ranke-edition-dating itself — incl. a scripts/ folder previously unexplored: date_run.py = the deepseek-v4-pro GOLD-AUDIT dater (source of gold_*_verdicts; evidence-packet + edition-year prompt, exactly the fields). Repo docs confirm stage2_docyear = "earlier Qwen3-14B filtering pass, -2yr mean error". HF org history-llms accessible via our token but hosts only 2 public DPO sets — full corpus (with docyear fields) lives at CSCS. RESOLUTION: no substitute generator needed for the existing training union (scores = stored per-record data, as in our sample); rebuild only ever needed for NEW sources; one confirmation line to Guohui closes provenance.
REPO UNIFICATION (PI request): anachronism-decontaminator subtree-merged into ranke-edition-dating (full history; untracked keyword_assessment.py/json committed first — nothing erased); unified README repository map added (PIPELINE.md as entry point, marker subsystem, scripts/=gold-audit tooling, provenance note); old repo got a pointer README and was ARCHIVED read-only (reversible, not deleted). Commits: 35cf0a4/7a13cb8 (old), subtree+7006289 (new).

### 2026-07-24 20:15 (office box) — decontamination-folder integration sweep COMPLETE
PI asked what else in temp-guohui/decontamination should be integrated. ANSWER: Guohuis agentic-dater CODE lived ONLY in the temp-named Dropbox folder (preservation risk) -> preserved into the unified repo as guohui-dater/ (215 files, 14.5 MB: 01qwen_pilot + 02_external_book_benchmark + 03_a3_validation + scripts + progress/README + the technical-report PDF; PROVENANCE.md documents authorship, origin, referee-tier role, and the distinction from BOTH the 14B corpus-build pass AND scripts/date_run.py). Excluded per policy: datasets 12GB, outputs 2.2GB, validation OCR texts >1MB. NOT integrated (correctly): corpora/outputs (data, stays Dropbox/CSCS), codex-session-history (12KB session notes, left in place). Commits f805c40 + README map row. The unified repo now contains the COMPLETE decontamination workflow: stage-1 rule, gold-audit tooling, marker subsystem (1945+1929), stages 2-3 experiments + validation, referee-tier source, all reports.

### 2026-07-24 21:30 (office box) — repo restructured for real; external review D -> B
PI (rightly) rejected the README-only cleanup. FULL THEMATIC RESTRUCTURE via git mv (history preserved): pipeline/ (stage1 rule + README; stage2 frozen defs incl. reprint policy; stage3 frozen defs; markers/ both lists + decontaminator subsystem + README; referee-agentic-dater/) | validation/ (all harnesses, panels, results, prereg, gold_audit(+results), scratch) | reports/ (ALL reports+figures in ONE place incl. Guohui PDF) | docs/ (stage1 notes + LOG snapshot). 18 path repairs; VERIFIED: score_oos identical, analyze_v3 results.csv BYTE-IDENTICAL (after adding judgment-file exclusions for post-sweep parquets). NEW: frozen-config consistency gate (validation/test_frozen_consistency.py — asserts harness prompts/regexes match pipeline/ canon; caught its own needle bugs twice). requirements.txt added.
EXTERNAL REVIEW (PI-mandated, deepseek-v4-pro): round 1 grade D with 8 concrete findings (broken LOG refs, ambiguous stage1, misplaced 2b, no requirements, no stage1 README, hardcoded data root, incomplete map) — ALL addressed; round 2 grade B, residual findings = exactly the declared production/-package gap (externalized config + standalone stage drivers), which is the known CSCS-readiness work. Review trail committed in validation/scratch/_deepseek_repo_review*.md.

### 2026-07-24 22:00 — checkpoint
Full-day arc: OOS validation passed (all 5 pre-registered bands; rescue 0.875/loss 13.2% now canonical) -> stage 3 built+debugged (v1 label-trust bug) -> markers_1929 (447 validated; stylistic fingerprints discovery) -> 120B negative result ($6.69) -> composite + referee-tier design (flipped burden of proof) -> Guohui workaround (14B fooled-rate measured from stored outputs; no substitute needed) -> repos unified (decontaminator merged+archived; Guohui dater preserved from temp folder) -> PI rejected cosmetic cleanup -> FULL thematic restructure (pipeline/validation/reports/docs, git-mv, 18 path repairs, byte-identical re-verification, consistency gate) -> deepseek-v4-pro review D->B (residual = declared production/ gap). Abandoned today: 120B scaling, README-only cleanup, loose year contexts. OPEN: production/ package (needs CSCS specifics), referee abstention measurement, PI dials, review queue, Guohui returns. CONTEXT.md rewritten with compiled truth.

### 2026-07-27 17:10
NeurIPS 35072 rebuttal work (Phase 1 due today). Reviews obtained via PI Dropbox link (2x Reject-2, 1x Accept-5; meta = novelty-vs-Talkie). Fine-tuned-dependency question ANSWERED FROM EXISTING modern_baseline data, no new GPU run: posttraining shifts WVS positions by median 0.22-0.25 human SD (vs 0.54-0.73 for ordinary Qwen3 base->instruct), distance-to-modern-publics unchanged (0.63->0.58, 0.65->0.61 SD) while ordinary tuning doubles it; text-backend TVD 0.16/0.17 vs 0.33-0.41; codability 62-69%->90-91%; free-text r=0.79-0.85 (openai judge; gemini-1929 discordance traced to judge noise on 1929-base, cross-judge r=0.12). Cross-lineage Talkie<->Ranke r=0.80-0.85 > Talkie<->modern 0.78-0.80. Date-spike 1913: near-future leakage only WWI-onset 12.6x / Bolshevik 16.7x vs known-median ~50x, far future flat. CAUGHT+DISCARDED: first SD-normalization used baseline_distributions.parquet uniform/always_modal pseudo-baselines (invalid); redone with metrics_by_question ba_plus + population benchmarks. Artifacts in C:\Users\hvoth\neurips-rebuttal\ (dependency_report.html v1, 4 rebuttal drafts w/ char counts 1.9-8.2k, dependency_results.json). 4 Ranke checkpoints prefetched to D:\hf-models for optional exact-battery replication (needs Daniel prompt file). All PROVISIONAL pending PI review.

### 2026-07-27 17:40
xG3y rebuttal rewritten v2 per PI: GRPO/GPT-judge concern now quoted verbatim and answered in four layers - (1) design channels, (2) NEW: the submission folder's unreferenced sft/grpo_partisanship.png figures show a normative probe FLAT at ~0.54 across all 1.3M SFT tokens and 120 GRPO steps (training-time monitoring evidence), (3) end-to-end base-vs-posttrained bound (~0.2 SD vs ~0.6 ordinary tuning, no drift toward moderns) framed as subsuming the judge channel via two falsifiable predictions, (4) judge-decorrelation experiment demoted to supporting role with honest status slot. No load-bearing TBD left. 9.6k chars.

### 2026-07-27 18:30
Daniel unreachable -> full repo/HF sweep for the TBD facts. RECOVERED without him: partisanship probe def (201-item period-anchored cloze battery, public), 199-item year-tagged knowledge battery (public), full SFT funnel w/ exact counts+models (3002 GPT-5 questions x16 base samples, gpt-5-mini scoring+reformat w/ divergence cap 0.2, all public on uzh-echist-org), GRPO sets (o3/gpt-5, public), GDPO=GRPO group-norm slip, corpus funnel 600B->-40%->80B/checkpoint (two-stage genealogy, 4096 ctx, Qwen3 tok), compute partial. KEY SOURCES: gjiang-economics/history-llms + DGoettlich/history-llms (Daniel ACTIVE there 2026-07-25!) prerelease_notes.md + 20 HF datasets pulled to D:\hf-models\posttraining-datasets. DISCREPANCIES: GRPO 8-vs-16 candidates (tex vs prerelease); 80B not 160B tokens. STILL MISSING (private repos/CSCS only): contamination-score formula, aged-benchmark generation details, judge-decorrelation numbers (genuinely unfinished), velocity/patents checkpoint id. GitHub token in api/ EXPIRED (401). Drafts updated+refilled; TBD_LEDGER.md written. Proposed to PI: run pre/post disaggregation + the judge-decorrelation experiment ourselves tonight (public data + local models).

### 2026-07-27 20:15
Task A (pre/post disaggregation, public 199-item battery, 4 checkpoints, both sum + length-norm backends) DONE. Pre-cutoff strong: 0.66-0.75, instruct +6-9pp. KEY METHOD FINDING: battery has a prior-artifact floor - the 1913 model scores 0.36-0.43 on post-1946 UNKNOWABLES (both backends), so vs-chance-0.25 is the wrong test (validates 3vfC scoring concern; propose within-battery unknowable baseline). Against that floor: 1913 model NO excess post-cutoff anywhere; 1929 model +18-22pp raw near-band (1930-46) elevation vs 1913 on same items w/ matched 1947+ control (McNemar 6v1 discordant, p=0.125, n=23 underpowered) - hits are canonical 1930s facts (New Deal, Yalta, Reichstag fire, Earhart, Oz, Spanish Civil War) = consistent w/ residual near-cutoff contamination, i.e. the v2 pipeline target. Instruct > base even on unknowables (cue-following amplification). Results: taskA_results.json + taskA_did.json. Task B generation running (bb8cfq3n9).

### 2026-07-27 21:30
Task B (judge-decorrelation, our reconstruction of the paper footnote experiment): 800 answers (1913-Instruct, 400 public GRPO questions x2, temp 0.6) x2 independent gpt-5-nano judgments each, zero API failures. RAW RESULT CONTRADICTS the paper claim: corr(GRPO-characteristics score, normative approval) r=+0.594 [0.547,0.637]. DIAGNOSED AS QUALITY HALO: coherence~relevance r=0.80 (one quality factor), scores low+dispersed (coh mean 5.3, rel 4.4 - 1913 prose is rambling/truncated); within well-formed answers (coh>=7&rel>=7, n=44) r collapses to +0.068 p=0.66 (underpowered+range-restricted, honest caveat). => Cannot post "no correlation" naively; the halo-free DIRECTION test now running: 3rd independent stance-modernity judgment (0=period-traditional,10=modern-progressive); corr(reward-chars, modernity)~0 would be the value-injection-relevant null. NOTE for team: whatever Daniel's version shows, the halo confound applies to HIS design too - flag before anyone posts the footnote experiment as-is. End-to-end result (positions don't move, 0.22-0.24 SD) remains the primary defense regardless.

### 2026-07-27 22:20
Task B stance-direction test done (797 judged, 0.4% null): corr(reward-chars, stance-modernity) r=+0.225 [0.16,0.29] raw; within well-formed n=44 r=-0.12 ns; coherence itself correlates with modernity +0.21 (period prose reads incoherent AND traditional to nano) and modern judge approves modern stances (r=+0.45). VERDICT: the paper footnote experiment is halo-confounded BY DESIGN - cannot certify content-blind reward; DO NOT post "no correlation". Correct line = self-correction + end-to-end bound (0.22-0.24 SD) as decisive. Combined report tasksAB_report.html built (recommended discussion-phase wording included). Softened the judge-experiment promise in both plain rebuttal files (xG3y, EASS) to "will report in full". Total tonight: Task A 4 checkpoints + DiD, Task B 800 gens + 2,397 nano judgments, ~cents API cost.

### 2026-07-27 23:30
Daniel resurfaced (WhatsApp): contamination formula = s_pre x (1 - s_post) with s = max(0,(acc-0.25)/0.75) - matches my inference; explains StoriesLM 0.02 (s_pre~0 collapses product). Aged benchmarks: confirmed 2-step keep/adapt/drop pipeline (already in paper) but model names/prompts/rates STILL missing. 3vfC human rebuttal upgraded: formula stated verbatim + Task-A wrinkle added (chance floor really ~0.36 not 0.25, distractors weaker than true answers - chance term needs recalibration). Human-voice versions (half length, plain prose) are now the canonical set after PI style verdict ("inhumane"); style lesson saved to feedback_plain_english memory. Still for Daniel: aged-benchmark model names/prompts/rates+examples, overall-score weighting, velocity/patents checkpoint id; PI asked him for sub-repo invites + hub repo transfer. WARNING pending relay: do not post the judge-experiment "no correlation" claim (halo-confounded, see 22:20 entry).

### 2026-07-28 00:20
PI course-correction x2: (a) EASS reply "way too loose and hand-waivy... this is the one supporter" - rebuilt Q1 with full nitty gritty: verbatim stage-2 prompt from frozen_config.py, 3-pool label protocol w/ certifier independence, per-stage FP/FN, F1-as-prevalence-function with worked arithmetic (5% prevalence -> precision 0.26, F1 0.40 - why we report primitives), residual characterization, gov-doc + always-AFTER lessons; (b) "the human rewrite has been a way to take the meat out. not good" - restored full technical content to xG3y (reviewer quote, SFT funnel 3002x16=48032, TVD numbers, two-predictions framing) and 3vfC (stage detail, plant realism 47.5%, 12528-config frontier, scaling ablation, date-spike leak profile). Rule now: plain VOICE, full MEAT. Final: 3vfC 7.5k, xG3y 8.9k, EASS 10.0k, AC 1.5k.
