# Ranke-1929 x Talkie — round 4: close the gap to Talkie (DRAFT for approval)

Version: v4.0-draft · Date: 2026-09-30 09:40 Europe/Zurich · Plan ID: `ranke-gap-0930`
Files used: `G:\detritus\ranke1929_talkie_sft_20260928\LOG.md` (entries 2026-09-29 13:57 to 2026-09-30 09:25),
`runs\night_20260929\judging_night\analysis\results_sonnet.json`, `agreement.json`,
`scripts\v3\eval_adapters_v3.py`, `scripts\v3\train_ranke_lora_v3.py` receipts.

## CHECKED (what we know, Sonnet-only preliminary; DeepSeek pending)
- History content 0-100: base 6.9, best Ranke 36.8 (Talkie answers only, 6,370 pairs), Talkie-13B 72.2.
- Helps: Talkie's direct answers (+10.4 over books alone mix, +12.5 alone vs books alone).
- Does NOT help: continued pretraining on period text (-3.5), encyclopaedia pairs (-2.8), 47 more books (+2.1),
  5 vs 3 passes (-0.7). Round 2: no dose effect 300 -> 2,213 Talkie passage records.
- Book-pair training makes answers short (median 24 words); Talkie-answer training gives 71 words; Talkie 97.
- Every Ranke model is trained AND evaluated on a 4-bit (NF4) copy of the base; Talkie ran in bf16. Greedy decoding.
- Main Ranke failure = invented facts (Jacobites in 1832, friars unifying Italy), some post-1913 leaks, loops.
- Never tested: Talkie answers alone at larger size (job 1 + job 2 kept = 13,330 pairs); full precision; best-of-N.

## Step 1 — Diagnose (knowledge missing vs. not drawn out)
1a. Full-precision check: best adapter (Talkie-only ep3) answers the 84 prompts on a bf16 base instead of 4-bit.
1b. Best-of-8: best adapter samples 8 answers (temperature 0.7) for the 36 history prompts; both judges grade all 8.
    Read: if the best of 8 scores >= 55, Ranke holds the knowledge and selection/preference training can close
    much of the gap; if the best of 8 stays near 37, the limit is knowledge in the 4B weights.
## CHANGE 2026-09-30 09:45 (after approval, JV objected to 4-bit): ALL Step-2 training and ALL answering in bf16
## (full precision). 4-bit was never JV-approved (see LOG 09:45). Step 2 also gets JV's long-book arm:
## books_long_3753 vs talkie_3753 (same size, same recipe).
## Step 2 — Talkie-answer dose, only Talkie data
2a. Talkie-only arms on the continued-pretraining model AND on base Ranke: 3,000 / 6,370 / 13,330 pairs, same recipe
    (LoRA r64, lr 1e-4, 3 passes). Pass bar: 13,330 beats 6,370 by >= 5 points (history content), CI above 0.
2b. Same 13,330 in bf16 (no 4-bit) with LoRA r256, and one full fine-tune (all weights), if 1a shows precision matters.
## Step 3 — only if Step 2 shows a dose effect
3a. Talkie writes 10 numbered questions per complete EB11 history/biography article (it does this well, trial 2),
    then answers each directly; same mechanical filters (stand-alone question, >= 20 words, clean stop, no year > 1913,
    agrees with the article). Target 50,000-100,000 pairs. Train Talkie-only at that size.
## Step 4 — only if Step 1b shows best-of-8 >= 55
4a. Preference pairs from Ranke's own 8 samples, chosen by rules + Talkie-13B log-probability (a period model, no
    modern model chooses), then DPO on the best Step-2 model.

## Not doing
More book pairs; more passes; more continued pretraining; encyclopaedia pairs as a separate ingredient; modern models
writing or editing training text.

## Heavy steps this approval covers (cluster via router only)
Cluster generation + training for Steps 1-2 (<= 5 CHF). DeepSeek-V4 Flash grading, new cap $6 (Sonnet on plan).
Step 3 and Step 4 need a separate OK after Step 1-2 results.
