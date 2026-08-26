# Ranke-4B-1929 — contamination audit

**What this is.** A full audit of whether the time-locked model
`uzh-echist-org/Ranke-4B-1929` (and its `…-1202-Instruct` fine-tune) secretly knows
things from *after* its 1929 cutoff — plus every tool built along the way.

**Bottom line (2026-06-25, corrected & length-matched).** The Ranke models are **mostly
clean** time-locked models: ~90%+ correct on pre-cutoff facts, **at chance post-cutoff** —
across UK / France / Germany / Italy / Spain / Russia politics, Western culture, and science.
The **one contaminated area is US 20th-century political history**: post-cutoff US politics
sits at **73–78% correct (significantly above the 50% chance, pooled p≈0.02)**, statistically
as good as — or only marginally below — pre-cutoff US politics. Confirmed across all three
cutoffs (1900/1913/1929) and the instruct fine-tunes ⇒ a **pretraining leak from US-centric
English sources**, surfaced (not created) by instruction-tuning. Earlier sweeps overstated
this (a metric artifact, see below); under the corrected metric contamination is **real but
localized to American politics**.

**Two methodological musts (learned the hard way — see CONTEXT.md / LOG.md):**
1. **Length-matched alternatives.** Score a minimal pair (true vs false completion) only where
   the true and false answers have the **same token count**. Per-token *average* logp inflates
   the score when the true answer is longer; *total* logp inflates it when shorter; *first-token*
   logp compares subword fragments (`" Ken"`, not `"Kennedy"`) and is useless. The three metrics
   agree **only on length-matched pairs**. So every battery item carries several candidate false
   twins, and the scorer picks one whose token count matches the true answer.
2. **Red / Amber / Green per category — two tests, not one.** (A) Is should-not-know (post-cutoff)
   different from 50% chance? (B) Is it different from should-know (pre-cutoff)?
   **GREEN** = post not above chance (doesn't know the forbidden material); **AMBER** = post above
   chance *and* significantly below pre (knows some, clearly less than legit); **RED** = post above
   chance *and* indistinguishable from pre (knows forbidden ≈ as well as allowed). A non-significant
   cliff is evidence of contamination **only if post is also clearly above chance** — otherwise it's
   just noise/small-N, not contamination.

**Complementary view:** the **rank** of the true answer in the next-token distribution (one
forward pass), *not* softmax confidence. See `report.html`.

➡ **Start here: open `report.html`**, then `2_model_probes/traffic_light.py` for the
red/amber/green-by-category result and `compare_metrics.py` for the length-bias proof.

## Folder guide
| folder | what's in it |
|---|---|
| `report.html` | The visual write-up. Read first. |
| `1_advice_and_panel_review/` | The original methodology advice + the codex/fusion review of it. `run_chain.py` produced 02/03. |
| `2_model_probes/` | Probes that test the **model**. **`battery50.py`** = the 55-topic clincher battery (key result); `jfk_probe.py`, `freecall.py`; **`feasibility_demo.py`** proves the rank probe is *one forward pass*, not vocab^tokens; plus earlier `probe*.py`, `battery.py`. |
| `3_fasttext_date_classifier/` | The "when was this written?" classifier for **corpus** cleaning (`build_dataset`→`train`→`eval`→`score_corpus`), its own README, and the Institutional-Books pullers. |
| `4_sft_data_scan/` | Scans of the SFT datasets — `scan_sft.py`, and the disambiguated `scan_sft2.py`. |
| `5_results/` | The evidence: `battery50_results.csv` (all probes × both models), per-model battery JSONLs, a sample Institutional-Books record. |
| `6_corpus_contamination_finder/` | **`find_contamination.py`** — grep the 160B pretraining corpus for post-1929 leakage, with the false-positive lessons baked in. **This is the next step** (run on server 2). |
| `7_chat/` | **`chat_ranke.py`** — a local web chat UI to talk to Ranke-Instruct. |

## How to run (the venv is NOT included)
Needs Python 3.12 with `torch transformers datasets fasttext-wheel numpy==1.26 huggingface_hub`.
Recreate a venv (keep it OUT of Dropbox):
```
uv venv C:\Users\hvoth\ranke\.venv --python 3.12
uv pip install --python C:\Users\hvoth\ranke\.venv\Scripts\python.exe torch transformers datasets fasttext-wheel "numpy==1.26" huggingface_hub
```
- **`numpy==1.26` is required** (fastText 0.9.2 breaks on NumPy 2.x).
- HF access needs the `huggingbeard` token scoped to the `uzh-echist-org` org.
- Some scripts use absolute paths (to the old working dir and to `…\claudecode\api\HF.txt`) — adjust if needed.
- On Windows, `set HF_HUB_DISABLE_SYMLINKS=1` before downloading models.

## The next step (the actual fix)
1. `python 6_corpus_contamination_finder/find_contamination.py --input <160B corpus>.jsonl --out flagged.jsonl`
2. Read the snippets, find the recurring **source**, drop that source wholesale, fix the date filter.
3. Retrain the base; re-run `2_model_probes/battery50.py` as the acceptance test (a clean model should sit at chance on every post-1929 clincher).

## Chat with the model
```
C:\Users\hvoth\timelock-contamination\.venv\Scripts\python.exe "7_chat\chat_ranke.py"
```
then open **http://localhost:7860** (CPU — replies take ~20–60 s).

## Deliberately excluded (regenerable)
The venv (~0.8 GB), fastText training data + trained model (~1.8 GB), and trial corpora
`dated*.jsonl` (~0.1 GB) were left out to keep this folder light. Regenerate via
`3_fasttext_date_classifier/`: `pull_institutional.py` → `build_dataset.py` → `train.py`.
