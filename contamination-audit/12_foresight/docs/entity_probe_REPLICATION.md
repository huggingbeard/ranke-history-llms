# Replication — "Does a frozen model hold the person it is asked about?" (Hitler vs Bismarck)

**Date:** 2026-07-07 · **Version:** 1.0
**Repo root:** `D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\12_foresight`
**Author:** Ranke project

This file is written so the experiment can be reproduced from scratch by someone who has never seen it.
Everything needed — the exact weights, prompts, sampling knobs, seed, code, and the honest reading of the
results — is here.

---

## 1. What the experiment does

Ask two time-locked language models "who is X?", 100 free continuations each, in English and German, for
two people, and measure whether the model keeps describing the *right* person.

- **X = Adolf Hitler** — an unknown 24-year-old in Vienna in 1913; the NSDAP leader / 1923 Munich-putsch
  defendant / *Mein Kampf* author by 1929, but **not** yet Chancellor (that is January 1933).
- **X = Otto von Bismarck** — died 1898, so canonical in *both* corpora. This is the positive control: a
  clean 1913 model must hold Bismarck, or the whole method is suspect.

## 2. Models (pin the revision, not just the name)

| model | HuggingFace id | revision (snapshot commit) |
|:--|:--|:--|
| Ranke 1913 | `uzh-echist-org/Ranke-4B-1913` | `6f0602bfc34eff82c198b6b3f3b5a5991ad81321` |
| Ranke 1929 | `uzh-echist-org/Ranke-4B-1929` | `c54cabc48d129bd727f52255f7227a668d23baf9` |

4B params, Qwen3-4B architecture, trained from scratch on multilingual text truncated at the cutoff year.
The tokenizer carries a **chat template** (the models are instruct-tuned), but we probe them **raw**:
`add_special_tokens=False`, **no** chat template applied — free text completion, matching the foresight
pipeline. HF token read from `D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt`.

## 3. Environment

- venv: `C:\Users\hvoth\timelock-contamination\.venv` (Python 3.12)
- torch 2.11.0+cu128, CUDA 12.8, GPU NVIDIA RTX 3060 (12 GB), bf16.
- Shared scoring primitives: `src/common.py` (loads bf16, waits for VRAM, retries on OOM).

## 4. Exact prompts (verbatim)

| key | English | German |
|:--|:--|:--|
| `who_is` | `Who is Adolf Hitler?` | `Wer ist Adolf Hitler?` |
| `def` | `Adolf Hitler is` | `Adolf Hitler ist` |

For Bismarck, substitute `Otto von Bismarck` for `Adolf Hitler` in all four. The continuation is the model's
completion of the prompt string; the prompt itself is not included in the saved text.

## 5. Sampling (identical for every cell)

```
do_sample=True, temperature=0.8, top_p=0.95, repetition_penalty=1.3,
max_new_tokens=60, num_return_sequences batched (10 at a time),
torch.manual_seed(1929) once per model load, pad_token_id=eos.
N = 100 continuations per (model, prompt, language).
```

Determinism caveat: with a fixed seed and identical weights/torch/driver the *distribution* reproduces;
exact bitwise generations can differ across GPUs/driver versions because some CUDA kernels are
non-deterministic. Pin the model revision (section 2) for a faithful redraw.

## 6. Code and run order

```
python src/entity_probe.py hitler      # (already run as src/hitler_probe.py; kept in results/hitler_probe/)
python src/entity_probe.py bismarck    # -> results/entity_probe/bismarck_{en,de}.json
python src/holds_report.py             # metric + figures/fig_holds.png + results/entity_probe/hitler_bismarck_holds.html
python src/consistency.py              # the TF-IDF null check (documented dead end)
python src/stickiness.py               # the name-recurrence dead end (documented)
```

All generation is resume-safe (checkpointed per batch); re-running skips completed cells.

## 7. Output files

- Raw text: `results/hitler_probe/generations.json` (en), `generations_de.json`;
  `results/entity_probe/bismarck_en.json`, `bismarck_de.json`. Each is `{model: {prompt: [100 strings]}}`.
- Annotated on-domain hits with trigger word + seed: `results/entity_probe/holds_examples_annotated.md`.
- Figures: `figures/fig_holds.png`, `figures/fig_hitler.png`, `figures/fig_hitler_de.png`.
- Reports (figure embedded, self-contained): `results/entity_probe/hitler_bismarck_holds.html`,
  `results/hitler_probe/hitler_report.html`, `hitler_report_de.html`.

## 8. The metric, stated exactly

Three attempts, two dead, one used:

1. **Name-recurrence** ("does the surname recur") — REJECTED, confounded by referential style (Bismarck is
   "the Prince / he"; a confabulated "Mr. Hiller" recurs). It rated Bismarck as *failing* 88–94%.
2. **TF-IDF cross-sample cohesion** — REJECTED, a null: every cell ≈ 0.09 (period-newspaper style dominates).
3. **On-domain** — USED. Share of continuations whose lowercased text matches the person's domain regex.

Domain regexes (verbatim, `re.I`):

- Bismarck (strict; broad terms like "german/berlin/reich" removed to avoid generic-politics inflation):
  `chancellor|kanzler|preuß|preuss|prussia|kaiser|emperor|fürst|\bprince\b|staatsmann|statesman|friedrichsruh|hohenzollern|blood and iron|eisen und blut|1866|1870|1871|1890`
- Hitler:
  `bavaria|bayern|münchen|munich|münchner|nationalsozialist|national ?socialis|nazi|völkisch|volkisch|putsch|agitat|hetzer|nsdap|braunau|hakenkreuz|bewegung|movement|umsturz|aufstand|radical|reichstag`

Specificity check: score every cell against BOTH domains; the off-diagonal (wrong person's domain) must be
near zero.

## 9. Results — keyword metric (own-domain %, pooled over the two prompts, 200 continuations per cell)

| asked about | model | English | German | (other-domain EN/DE) |
|:--|:--|--:|--:|:--|
| Bismarck | 1913 | 34 | 58 | 2 / 8 |
| Bismarck | 1929 | 54 | 58 | 1 / 5 |
| Hitler | 1913 | 4 | 10 | 3 / 7 |
| Hitler | 1929 | 24 | 40 | 10 / 7 |

## 10. THE IMPORTANT CAVEAT — the keyword rate is an upper bound; hand-read before believing it

The on-domain rate is a keyword proxy and it **overstates** the small numbers. Every one of the 8 English +
20 German Hitler-1913 "hits" was read by hand and **all are false positives**: the trigger word appears
inside a confabulation about a different person (e.g. trigger `munich` inside "born at Munich, 1905";
trigger `agitat` inside "agitates Washington"; trigger `reichstag` inside "a member of the Reichstag for ten
years" describing an invented unpopular deputy). **None** name the NSDAP, the putsch, or anything
Hitler-specific. So the true Hitler-1913 hold-rate ≈ **0**, not 4–10%.

By contrast, hand-reading confirms:
- **Bismarck-1913** hits are genuinely about Bismarck (birthplaces Schönhausen and Kniephof; "in 1862 he
  entered political life"; "the first Reich Chancellor"; "called by the Prussian King to lead Germany's
  foreign policy") — with fuzzy dates and the occasional meta-confusion ("Is he a real person?"). Real
  knowledge, blurred at the edges — what a 4B model "knowing" a figure looks like.
- **Hitler-1929** hits include genuine knowledge, strongest in German: "über den **Hitler-Putsch**", the
  **Münchener Post**, "Ist Adolf Hitlers Aufenthalt in **Bayern** rechtmäßig" (his real 1929 speaking-ban /
  residence question), "völkische" party — mixed with confabulation (sometimes a "Bolshevik" or "Communist
  party" leader).

**Corrected reading:** the true dissociation is *sharper* than the bar chart. Bismarck is held by both
models (~half the time, real); Hitler is held ≈0 by 1913 (the keyword rate is noise) and partially, really,
by 1929. The time-lock leaks nothing: no continuation from either model states the post-1929 Chancellorship.

## 11. Known gaps / how to harden further

- Replace the keyword domain with an LLM judge ("does this text describe the real historical X?") to remove
  the false-positive upper-bound; expect Hitler-1913 to fall to ~0 and the others to firm up.
- Pin the HF model revision in code (currently resolved to `main` at run time; hashes recorded in section 2).
- Add a third control: a figure obscure in both eras, to read the floor.
