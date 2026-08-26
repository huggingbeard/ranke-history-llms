# Aho-Corasick decontamination — post-1945 tripwires for LoC-PD-Books

Fast detection of books that **cannot genuinely be pre-1945** because they contain
anachronistic vocabulary — terms that were effectively absent from books before 1945 but
common after (validated empirically against Google Books Ngrams). Built for the time-locked
LLM contamination project: cleaning corpora like HuggingFace `storytracer/LoC-PD-Books`
(140,000 books, ~8B words, ~48 GB OCR text) so a pre-cutoff training set stays clean.

## Why Aho-Corasick

A single left-to-right pass finds **every** occurrence of **all** markers at once, in time
`O(len(text) + #hits)` that is **independent of the number of patterns**. It is the ideal
cheap first-pass filter: it rejects the ~99% of clean text instantly. The slow part of the
naive approach — a 130-way alternation regex re-scanning the text for each alternative — just
disappears.

But **a bare substring match lies** (the lesson baked into `6_corpus_contamination_finder/
find_contamination.py`: "Normandy"=1066, "Israel"=the Bible). So every raw hit is verified:
1. **word boundaries** — flanking chars must be non-alphanumeric ("nato" must not fire inside
   "Donato"; "laser" not inside "eraser");
2. **proximity context** — a few homographs require a disambiguating regex *within ±200 chars*
   of the hit (whole-document context is too loose — a book of sonnets contains "nation"
   *somewhere*). Guards: `lsd` (drug, not £sd money), `iron curtain` (Cold War, not a theatre
   safety curtain), `common market` (EEC, not a marketplace), `third world` (geopolitical, not
   poetic), `nato` (the alliance, not Italian *nato*="born" / Spanish *vireinato*).

Text is also **de-hyphenated** at OCR line breaks ("agri¬\ncultural" → "agricultural") and has
**Internet-Archive digitization boilerplate stripped** ("digitized by the internet archive …
http://archive.org") before scanning — both are large false-positive sources in real OCR.

## Files

| file | what it does |
|---|---|
| `validate_markers_1945.py` | candidate n-grams → Google Ngrams → keep terms absent pre-1945, common after. Emits `markers_1945.json` + `markers_rejected.json`. |
| `aho_scan.py` | the engine: build automaton, normalize (de-hyphenate, strip IA boilerplate), scan with boundary + proximity-context checks. CLI streams the real dataset (resume/checkpoint). |
| `benchmark.py` | the sample exercise: stream N real books + inject synthetic post-1945 docs, run Aho-Corasick vs union-regex, report throughput + flagged books vs their stated `year`. |
| `contamination_trial.py` | the validation: plant length-X sections of NEW books into clean old books, sweep the dose, compute ROC-AUC / precision-recall / FP-FN. Writes `trial_results.json` + `dose_response.png` / `roc.png` / `pr.png`. |
| `markers_1945.json` | 130 Ngram-validated tripwires (128 deployed after dropping `civil rights act` and `motel`), each with pre/post frequency, log-ratio, birth year. |
| `markers_rejected.json` | rejected candidates **with the reason** — the audit trail. |
| `benchmark_results.json` | machine-readable benchmark output. |

Run in venv `C:\Users\hvoth\timelock-contamination\.venv` (has `datasets`; `pyahocorasick` added).

```
python validate_markers_1945.py                 # rebuild the lexicon (~4 min of Ngram calls)
python benchmark.py --limit 500                  # the speed demo
python aho_scan.py --repo storytracer/LoC-PD-Books --limit 140000   # full production sweep
```

## The lexicon (how it was built)

167 candidate n-grams (seeded from the project's existing marker lists + a fresh sweep across
tech / Cold-War / science / culture / orgs / people / events), deliberately including
**negative controls** I expected to fail. Each candidate's yearly frequency 1800–2019 was
pulled from Google Ngrams; a term is kept only if its **pre-1945 mean** frequency is below an
"effectively absent" floor (robust to Google's mis-dated volumes, which inject single-year
spikes — `pre_max` is too brittle) **and** its post-1950 frequency is clearly present **and**
`log10(post/pre) ≥ 1.5` (~30× more common after).

The method visibly filtered out ~22% of my guesses on hard data, including surprising ones:
- **rejected** (genuinely present before 1945): `penicillin` (1928), `television` (word coined
  1900, demoed 1930s), `united nations` (the phrase = allied nations, used in WWI),
  `chemotherapy` (coined ~1907), `new deal`, `pearl harbor`, `world war ii`, `nylon`, `cia`,
  `continental drift` (1912), `pacemaker`.
- **kept** (iconic post-1945): `cold war`, `internet`, `nato`, `nasa`, `laser`, `software`,
  `semiconductor`, `transistor`, `korean war`, `berlin wall`, `apartheid`, `khrushchev`,
  `gorbachev`, `perestroika`, `world bank`, `unesco`, `who`, `space shuttle`, `microprocessor`.

Caveat: `birth_year` is an approximate annotation (Google's mis-dating still fools it for a few
terms, e.g. `website`→1937); the keep/reject decision rests on `pre_mean`+`ratio`, not it.

## Results (3,000-book sample, single core)

| engine | time | MB/s | books/s |
|---|---|---|---|
| **Aho-Corasick** | 11.3 s | **~71** | **~265** |
| union regex (baseline) | 96.5 s | ~8 | ~31 |

**Aho-Corasick is ~8.5× faster** than the dependency-free union-regex baseline the project
currently uses — confirming the `find_contamination.py` docstring's "~10x" hypothesis. The two
engines flag *exactly* the same books. Building the 128-pattern automaton takes ~0.1 ms, and
its cost is independent of the number of patterns (grow the lexicon to thousands for free).
Throughput is stable across runs and sample sizes (~66–75 MB/s on 92 MB and on 800 MB of text).

**Extrapolation:** the full ~48 GB / 140k-book dataset → **~11 min on one core**, **~1.4 min on
8 cores**.

**Detection quality (the real point):** of 3,000 genuine "public-domain" books, **16 (0.53%)
were flagged — and all 16 are true post-1945 contamination, every one confirmed independently
by its own `year` metadata** (stated years 1952, 1954, 1959, 1961, 1964, 1967, 1969×2, 1971,
1978, 1983, 1985, 1986, 1991, 1992×2). **Zero false positives in 3,000 books.** Both injected
synthetic docs were caught; the decoy (£sd money + theatre "iron curtain") was correctly ignored.

The contamination is **post-war US-government area-studies bibliographies and "country study"
volumes** (Library of Congress Federal Research Division handbooks, etc.) — US Govt works are
public-domain at *any* date, so they leaked into a corpus assumed to be pre-1929. This is the
value-add over trusting metadata: the scan cross-checks **content against the stated year**,
catching books whose anachronistic vocabulary betrays a post-1945 origin. **LoC-PD-Books is not
all old.**

## Precision: the false-positive audit (why a bare match is not enough)

Reaching zero FPs on 3,000 real OCR'd books required hunting and killing every false-positive
class on evidence (a 500-book slice was deceptively clean; 3,000 exposed the real mess):

| FP source | example found | fix |
|---|---|---|
| IA digitization boilerplate | "digitized by **tine internet arciiive** in 2011 …" (OCR-garbled) | OCR-robust strip of the whole `digitized by … library of congress` block + any `internet arc*` |
| OCR character confusion | "hotel"→"**motel**" ("Eastman Motel"=the Eastman Hotel); "later"/"lazar"/"leisure"→"**laser**" | drop `motel`; require optics context for `laser` |
| Foreign-language homographs | Italian *nato*="born", Spanish *vireinato* → `nato` | require alliance/treaty/soviet context near `nato` |
| English homonyms / literal phrases | Richard Nixon (1821 settler); Martin Luther King (b.1882, genealogy); "a great leap forward" (idiom); ships that "rock and roll"; "double helix" (descriptive geometry); "KGB" (1922 radio call-sign); "common market value"; "a european community"; "third world-era" (Kabbalah) | proximity topic-context guard per term |
| OCR line-break hyphenation | "agri¬cultural revolution" → `cultural revolution` | de-hyphenate across the OCR hyphen `¬` |
| unbounded-substring bug | "nation" matched inside "ema**nation**"; "gene" inside "**gene**rated" | word-bound (`\b`) every context alternative |

The architecture that makes this cheap: Aho-Corasick rejects ~99.5% of text in one pass; the
expensive per-hit verification (word boundary + ±200-char topic context) runs only on the rare
candidates. 14 of the 128 markers carry a context guard; the rest are unambiguous neologisms.

## Validation: a controlled contamination trial (`contamination_trial.py`)

To prove it works on ground truth, plant a length-X section of a NEW (post-1945) book into a
verified-clean pre-1929 book and see if it's caught. 400 trials per dose; negatives = 600 real
old books, untouched; score = #distinct markers (graded, for ROC/PR). Three injectants bracket
reality: real post-1945 books (the actual threat), Wikipedia (vocabulary-rich), TinyStories
(bland fiction — worst case). Seed fixed (20260629).

**Precision is perfect: 0 / 600 old books ever falsely flagged → precision = 100% at every dose,
FP = 0.** The whole story is recall vs how much modern text is present:

| planted modern text | real post-1945 books | Wikipedia | TinyStories |
|---|---|---|---|
| ~1 sentence (60 ch)   | 0.2% | 0.5% | 0% |
| ~1 paragraph (1,500)  | 2%   | 16%  | 0.2% |
| ~a few pages (8,000)  | 17%  | 35%  | 0.5% |
| ~a chapter (40,000)   | 40%  | 81%  | 5% |
| ~a whole book (150,000) | **74%** | **100%** | 3% |

**Plant a whole modern book → almost always caught (74–100%). Plant one modern sentence →
usually missed** (a single sentence rarely contains a tell-tale word). TinyStories stays near
zero: bland modern prose uses no period-specific vocabulary and is invisible by design.

- **ROC-AUC** rises from ~0.50 (one sentence = coin-flip) to **1.000** (book-length Wikipedia).
  Pooled over doses: real 0.61, Wikipedia 0.70, TinyStories 0.51; **average precision** (the apter
  summary given the class balance) 0.85 / 0.88 / 0.80. Pooled AUC reads only "moderate" because it
  averages in the tiny-dose cells where detection is impossible in principle — the dose-response
  curve, not the pooled AUC, is the truthful picture. Curves: `roc.png`, `pr.png`, `dose_response.png`.
- **FP = 0 everywhere; FN = the misses**, shrinking with dose (e.g. real book-length: 295 caught,
  105 missed, 0 false alarms).
- **Integrity check:** recall ≥ the "detectability ceiling" (does the planted chunk alone contain
  a marker) in **every** cell — the scanner never misses a chunk that contains a giveaway word.
  Every miss is a chunk with zero period vocabulary (a Wikipedia article *about the 1860s*; a
  children's story about a butterfly) — a limit of language, not a code bug.
- **Real-data recall (no injection):** vocabulary alone flags **50% (60/120)** of genuine
  post-1945 books in LoC. Lesson: use this scan **with** the metadata `year` filter — the year
  field catches correctly-dated modern books, this scan catches the *mislabeled* ones; recall
  rises further as the lexicon grows.
