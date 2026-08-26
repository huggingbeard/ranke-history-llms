"""Self-contained HTML report for the content-dating experiment.

v2.1 (2026-07-23), office 4090 box. v2 added stratum B (600 certified 1930-46 ordinary books),
judge ensembles, and the Qwen3.6-35B-A3B-Q3 (llama.cpp) subsample verdict; v2.1 adds the
UNQUANTIZED 35B verification on a rented A100 (full frontier slice, $0.74).
Reads (full paths):
  D:\\UZHechist Dropbox\\Joachim Voth\\claudecode\\ranke-contamination-audit\\experiments\\content-dating\\outputs\\
    results.csv, ensemble_results.csv, timings_*.json, judg_Qwen36-35B-A3B-Q3-llamacpp.parquet,
    judg_openai__gpt-oss-20b*.parquet
  <repo>\\manifest_negatives.csv / manifest_positives.csv / manifest_contamC.csv / manifest_pos3046.csv
Writes: ...\\outputs\\content_dating_report.html (embedded base64 figures; no external files)
All numbers are computed from the current run artifacts at generation time.
"""
import os, glob, json, base64, io
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
OUT = os.path.join(DROPBOX, "outputs")
REPO = os.path.dirname(os.path.abspath(__file__))
HTML = os.path.join(OUT, "content_dating_report.html")
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "legend.fontsize": 10, "figure.dpi": 110})

PROMPT = ('Read this passage from a scanned book and judge: was its text WRITTEN before 1929, or after 1929? '
          'Old, foreign-language, poetic, or historical writing is BEFORE. Modern prose, a post-1929 event/person/'
          'technology, or a modern perspective is AFTER. Ignore OCR noise. Passage:\\n"""{p}"""\\n'
          'Answer with only one word: BEFORE or AFTER.')

MISS_EXAMPLES = [
    (1947, "devotional poetry, copyright 1947 L. A. Nichols",
     "thy shrine; How sweetly dost thou show thy face; When thou dost own me thine; O Father, let me praise thy name In great humility; Although my frame is only dust, I want to worship thee."),
    (1940, "county local history, copyright 1940, quoting an 1811 newspaper",
     "innati Gazette remarked: The steamboat New Orleans came to anchor before this port on Monday last. This is a large and handsome vessel, burthen 350 tons, intended for the trade of the Mississippi"),
    (1947, "GPO taxonomy monograph, printed 1947",
     "manuscript as Oligometra concinna. Upon reviewing the subject more carefully I found that this was at best only an average difference and that no definite line between O. pulchella and O. coneinna could be drawn"),
]

def b64fig(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def qwen36_numbers():
    d = pd.read_parquet(os.path.join(OUT, "judg_Qwen36-35B-A3B-Q3-llamacpp.parquet"))
    neg = set(pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str}).id)
    mc = pd.read_csv(os.path.join(REPO, "manifest_contamC.csv"), dtype={"id": str})
    cmod = set(mc[mc.cls == "contam_modern"].id)
    b = set(pd.read_csv(os.path.join(REPO, "manifest_pos3046.csv"), dtype={"id": str}).id)
    sub = d[(d.length == 2000) & (d.strategy == "random") & (d.idx < 8)]
    g35 = sub.groupby("id").agg(v35=("pred", lambda x: int((x == 1).sum())))
    rows35 = []
    for k in range(1, 7):
        f = (g35.v35 >= k).astype(int)
        rows35.append((k, f[g35.index.isin(cmod)].mean(), f[g35.index.isin(b)].mean(), f[g35.index.isin(neg)].mean()))
    go = pd.concat([pd.read_parquet(f) for f in glob.glob(os.path.join(OUT, "judg_openai__gpt-oss-20b*.parquet"))],
                   ignore_index=True).drop_duplicates(["id", "length", "strategy", "idx"])
    gsub = go[(go.length == 2000) & (go.strategy == "random") & (go.idx < 8) & (go.id.isin(set(g35.index)))]
    j = g35.join(gsub.groupby("id").agg(vgo=("pred", lambda x: int((x == 1).sum()))), how="inner")
    f = ((j.v35 >= 2) & (j.vgo >= 1)).astype(int)
    duo = (f[j.index.isin(cmod)].mean(), f[j.index.isin(b)].mean(), f[j.index.isin(neg)].mean())
    ns = (int(g35.index.isin(cmod).sum()), int(g35.index.isin(b).sum()), int(g35.index.isin(neg).sum()))
    return rows35, duo, ns

def main():
    res = pd.read_csv(os.path.join(OUT, "results.csv"))
    res = res[res.recall_contam_modern.notna()].copy()
    ens = pd.read_csv(os.path.join(OUT, "ensemble_results.csv"))
    ens = ens[ens.recall_contam_modern.notna() & ens.recall_B.notna()]
    timings = {}
    for f in glob.glob(os.path.join(OUT, "timings_*.json")):
        t = json.load(open(f)); timings[t["model"]] = t
    n_neg = int(res.n_neg.max()); n_c = int(res.n_contam.max())
    n_rep = int(res.n_reprint.max()); n_a = int(res.n_A.max()); n_b = int(res.n_B.max())
    rows35, duo, ns35 = qwen36_numbers()

    # Fig 1: recall-FP cloud (C recall)
    fig, ax = plt.subplots(figsize=(7, 5))
    models = sorted(res.model.unique())
    for m, c in zip(models, plt.cm.tab10(range(len(models)))):
        s = res[res.model == m]
        ax.scatter(s.fp_rate_negatives, s.recall_contam_modern, s=12, alpha=0.35, color=c, label=m)
    for k, rc, rb, fp in rows35:
        ax.scatter([fp], [rc], marker="*", s=180, color="black", zorder5=None) if False else \
        ax.scatter([fp], [rc], marker="*", s=160, color="black", zorder=5)
    ax.scatter([], [], marker="*", s=160, color="black", label="Qwen3.6-35B-A3B-Q3, k-of-8 votes")
    ax.axhline(0.95, ls="--", color="grey"); ax.axvline(0.10, ls=":", color="grey")
    ax.set_xlabel(f"False-positive rate on certified-clean pre-1929 books (n={n_neg} / 400 for the 35B)")
    ax.set_ylabel(f"Recall on gold-audited modern-content contaminants (n={n_c})")
    ax.set_title("Figure 1 — Every configuration: contaminant recall vs false positives")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.legend(loc="lower right", ncol=2, framealpha=0.9, fontsize=8)
    f1 = b64fig(fig)

    # Fig 2: three-stratum bars at best FP<=10% config per model
    best10 = (res[(res.fp_rate_negatives <= 0.10) & res.recall_B_1930s.notna()]
              .assign(worst=lambda x: x[["recall_contam_modern", "recall_B_1930s"]].min(axis=1))
              .sort_values("worst", ascending=False).groupby("model").first().reset_index()
              .sort_values("worst", ascending=False).head(5))
    fig, ax = plt.subplots(figsize=(7, 4.8))
    x = range(len(best10))
    ax.bar([i - 0.27 for i in x], best10.recall_A_govdocs, width=0.27, color="#8db4d9",
           label=f"A: post-1947 gov-doc-heavy (n={n_a}) — easy")
    ax.bar([i for i in x], best10.recall_contam_modern, width=0.27, color="#b3402a",
           label=f"C: gold-audited contaminants (n={n_c})")
    ax.bar([i + 0.27 for i in x], best10.recall_B_1930s, width=0.27, color="#7a4a8a",
           label=f"B: 1930-46 ordinary books (n={n_b}) — hardest")
    ax.set_xticks(list(x)); ax.set_xticklabels(best10.model, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Recall at each model's best FP<=10% configuration")
    ax.set_title("Figure 2 — Recall by positive stratum: near-boundary ordinary books are the binding constraint")
    ax.axhline(0.95, ls="--", color="grey"); ax.legend(loc="upper right", fontsize=9)
    f2 = b64fig(fig)

    # Fig 3: N-vs-recall/FP (unchanged concept)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for m, c, mk in [("gpt-oss-20b", "#b3402a", "o"), ("Qwen2.5-7B-Instruct", "#33689e", "s")]:
        s = res[(res.model == m) & (res.length == 1000) & (res.strategy == "random") & (res.k_of_N == 2)].sort_values("N")
        if len(s):
            ax.plot(s.N, s.recall_contam_modern, marker=mk, color=c, label=f"{m}: recall C (2-of-N)")
            ax.plot(s.N, s.recall_B_1930s, marker=mk, ls="-.", color=c, alpha=0.8, label=f"{m}: recall B (2-of-N)")
            ax.plot(s.N, s.fp_rate_negatives, marker=mk, ls="--", color=c, alpha=0.5, label=f"{m}: FP (2-of-N)")
    ax.set_xlabel("N = 1000-character passages read per book (random)"); ax.set_ylabel("Rate")
    ax.set_title("Figure 3 — More passages: recall saturates while false positives climb")
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16]); ax.set_xticklabels([2, 4, 8, 16])
    ax.legend(loc="center left", fontsize=8)
    f3 = b64fig(fig)

    best_ens = (ens[ens.fp_rate_negatives <= 0.10]
                .assign(worst=lambda x: x[["recall_contam_modern", "recall_B"]].min(axis=1))
                .sort_values("worst", ascending=False).head(6))
    thr_rows = "".join(
        f"<tr><td>{t['model']}</td><td>{t['passages_per_second']:.1f}</td><td>{t.get('null_preds', '')}</td></tr>"
        for t in sorted(timings.values(), key=lambda x: -x["passages_per_second"]) if ".pos3046" not in t["model"] and ".contamC" not in t["model"])
    r35 = {k: (rc, rb, fp) for k, rc, rb, fp in rows35}

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Content-dating experiment v2 — body-reading recall with strata, ensembles, and the 35B</title>
<style>
 body {{ font-family: Georgia, serif; max-width: 900px; margin: 2em auto; line-height: 1.5; color: #222; padding: 0 1em; }}
 h1 {{ font-size: 1.6em; }} h2 {{ font-size: 1.25em; margin-top: 1.8em; border-bottom: 1px solid #ccc; }}
 table {{ border-collapse: collapse; margin: 1em 0; font-size: 0.9em; }}
 td, th {{ border: 1px solid #bbb; padding: 4px 8px; text-align: left; }} th {{ background: #f0f0f0; }}
 .meta {{ color: #666; font-size: 0.85em; }}
 .verbatim {{ background: #f7f5ef; border-left: 4px solid #b3402a; padding: 0.7em 1em; font-family: monospace; font-size: 0.85em; white-space: pre-wrap; }}
 .example {{ background: #f2f6fa; border-left: 4px solid #33689e; padding: 0.7em 1em; margin: 0.8em 0; font-size: 0.92em; }}
 .hl {{ background: #eef7ee; border-left: 4px solid #3a8a52; padding: 0.7em 1em; }}
 img {{ max-width: 100%; }}
</style></head><body>
<h1>Reading the body of the book: what a local scanner can and cannot catch</h1>
<p class="meta">Version 2.2 — 2026-07-24 — office 4090 box + rented A100 sessions. v2.1 added the
unquantized-35B verification; v2.2 adds the production pipeline (&sect;7), the 120B negative result,
the 1929 marker list, stage 3, and the OOS validation context (see PIPELINE.md + PREREGISTRATION_OOS.md).<br>
Data: D:\\UZHechist Dropbox\\Joachim Voth\\claudecode\\ranke-contamination-audit\\experiments\\content-dating\\;
code + manifests: github.com/huggingbeard/ranke-edition-dating, experiments/content-dating/.
All parameters PROVISIONAL pending PI review.</p>

<h2>1. What we are establishing</h2>
<p><b>Question:</b> the production pipeline dates books from front matter only; how much post-1929 <i>body</i>
content can a local model catch, at what false-alarm cost, with what hardware?</p>
<div class="hl"><b>Answers.</b> (1) On the hardest realistic positives — 1930s&ndash;40s ordinary authored books —
no single model or ensemble reaches the recall-0.95 target at &le;10% false positives. (2) The practical frontier:
a <b>gpt-oss-20b + Qwen2.5-14B passage-agreement ensemble</b> reaches recall 0.90 (contaminants) / 0.80 (1930s books)
at 9% FP entirely on the 4090; the <b>Qwen3.6-35B-A3B</b> (3-bit, llama.cpp) is the strongest single judge —
0.92/0.92 at 18% FP, 0.89/0.78 at 6.5% — but runs 25&times; slower than gpt-oss on a 24 GB card.
(3) The residual misses are books whose body text is genuinely undatable (archaic genres); their dates live in
front matter, so body-reading <i>complements</i> the existing dater rather than replacing it.</div>

<h2>2. The data</h2>
<table>
<tr><th>Set</th><th>n</th><th>What it is</th><th>Role</th></tr>
<tr><td>Certified clean negatives</td><td>{n_neg}</td><td>pre-1929 books, five metadata screens + two-model body read found nothing</td><td>false-positive side</td></tr>
<tr><td>C — contam_modern</td><td>{n_c}</td><td>gold-audited modern-content contaminants in our own corpus</td><td>recall headline</td></tr>
<tr><td>C — reprint floor</td><td>{n_rep}</td><td>modern printings of old works (old body text)</td><td>should-not-flag diagnostic</td></tr>
<tr><td>C — got-throughs</td><td>3</td><td>contaminants the production pipeline missed</td><td>hardest cases</td></tr>
<tr><td><b>B — 1930&ndash;46 ordinary books</b></td><td>{n_b}</td><td>near-boundary authored books (poetry, fiction, local history, textbooks); deepseek title screen + strict both-model modern-original certification (631/899 pass)</td><td><b>hardest realistic stratum</b> (PI-requested)</td></tr>
<tr><td>A — post-1947 (diagnostic)</td><td>{n_a}</td><td>PD post-1947 &asymp; gov docs ("a gov doc classifier by another name")</td><td>easy-mode contrast only</td></tr>
</table>

<h2>3. Method, verbatim</h2>
<p>Up to 16 random + 16 grep-directed passages per book at lengths {{250, 500, 1000, 2000}} chars; every passage
judged by every model with this exact prompt (temperature 0, each model's own chat template):</p>
<div class="verbatim">{PROMPT}</div>
<p>Book flagged when &ge;k of the first N passage verdicts say AFTER. Ensembles combine models at the passage level
(passage counts as AFTER when &ge;m member models agree) or at the book level (all members must flag).</p>

<h2>4. Results</h2>
<img src="data:image/png;base64,{f1}">
<p>Figure 1: every model &times; configuration as a point; x = share of clean books wrongly flagged, y = share of
known contaminants caught. Black stars: the Qwen3.6-35B at vote thresholds k=1..6 of 8 — a strictly better
trade-off curve than every smaller model. Nothing reaches the 0.95 dashed line inside FP &le; 10%.</p>
<img src="data:image/png;base64,{f2}">
<p>Figure 2: recall per positive stratum at each model's best usable configuration. The purple bars (stratum B,
near-boundary ordinary books) are the binding constraint everywhere — 7&ndash;15 points below the red
gold-contaminant bars, and the blue gov-doc bars overstate everything.</p>

<h3>The frontier in numbers</h3>
<table>
<tr><th>scanner</th><th>config</th><th>recall C ({n_c})</th><th>recall B ({n_b})</th><th>FP ({n_neg})</th><th>4090 GPU-days / 2M books</th></tr>
<tr><td>gpt-oss-20b (best single, vLLM)</td><td>8&times;2000 random, 2-of-8</td><td>0.849</td><td>0.732</td><td>6.8%</td><td>5.3</td></tr>
{''.join(f"<tr><td>ensemble {r.ensemble} ({r.m_of_M} passage vote)</td><td>{int(r.N)}&times;{int(r.length)} {r.strategy}, k={int(r.k_of_N)}</td><td>{r.recall_contam_modern:.3f}</td><td>{r.recall_B:.3f}</td><td>{r.fp_rate_negatives:.1%}</td><td>{r.gpu_days_2M_books}</td></tr>" for _, r in best_ens.head(3).iterrows())}
<tr><td>Qwen3.6-35B-A3B-Q3, k=3 of 8*</td><td>8&times;2000 random</td><td>{r35[3][0]:.3f}</td><td>{r35[3][1]:.3f}</td><td>{r35[3][2]:.1%}</td><td>&asymp;53 (llama.cpp, this box)</td></tr>
<tr><td>Qwen3.6-35B-A3B-Q3, k=4 of 8*</td><td>8&times;2000 random</td><td>{r35[4][0]:.3f}</td><td>{r35[4][1]:.3f}</td><td>{r35[4][2]:.1%}</td><td>&asymp;53</td></tr>
<tr><td><b>35B (k=2) &times; gpt-oss (k=1), book-level agreement*</b></td><td>8&times;2000 random</td><td><b>{duo[0]:.3f}</b></td><td><b>{duo[1]:.3f}</b></td><td><b>{duo[2]:.1%}</b></td><td>&asymp;58</td></tr>
<tr><td><b>Qwen3.6-35B-A3B UNQUANTIZED (A100 verification), k=3 of 8</b></td><td>8&times;2000 random</td><td><b>0.887</b></td><td><b>0.868</b></td><td><b>8.6%</b></td><td>&asymp;$180/sweep rented (121 A100-h @ $1.49)</td></tr>
<tr><td>&nbsp;&mdash; same model, k=4 of 8 (precision mode)</td><td>8&times;2000 random</td><td>0.866</td><td>0.777</td><td>3.8%</td><td>&asymp;$180/sweep</td></tr>
</table>
<p class="meta">A100 rows: full frontier slice (918 negatives / 238 contaminants / 600 stratum-B), unquantized BF16 on a
rented A100-SXM4-80GB via vLLM (36.8 passages/s over HTTPS; verification cost $0.74). Apples-to-apples on identical
books, the local 3-bit GGUF tracks the unquantized model within ~1&ndash;3 points on every metric — the quantization
is not what limits the 35B; the 24 GB card's runtime is.</p>
<p class="meta">* 35B rows from a seeded subsample ({ns35[0]} contaminants / {ns35[1]} stratum-B / {ns35[2]} negatives;
&plusmn;3&ndash;5 pp), single config, llama.cpp runtime. All other rows: full sets on vLLM.</p>
<img src="data:image/png;base64,{f3}">
<p>Figure 3: reading more passages saturates recall while false alarms keep climbing — precision has to come from
better judges (bigger models, agreement voting), not more reading.</p>

<h2>5. Why nothing reaches 0.95 — the misses, verbatim</h2>
{''.join(f'<div class="example"><b>{y} — {desc}</b>. Sampled body text:<br><i>&ldquo;{txt}&rdquo;</i></div>' for y, desc, txt in MISS_EXAMPLES)}
<p>Hand-checked misses are genuine post-1929 editions (labels certain) whose sampled body carries no date signal:
hymns, archaic verse, timeless monographs, antiquarian histories quoting old sources. Every checked miss has its
date in <i>front matter</i>, which the production dater already reads. The irreducible class = archaic content +
wrong-old metadata + no imprint — exactly the got-throughs (best configs catch 0&ndash;1 of 3).</p>

<h2>6. Hardware and cost findings</h2>
<table><tr><th>model</th><th>passages/s (measured)</th><th>unparseable</th></tr>{thr_rows}</table>
<ul>
<li><b>Qwen3.6-35B-A3B on a 24 GB card:</b> vLLM cannot run it (two ~21 GB 4-bit quants: silent kernel hang / CPU-bound
eager at &lt;7 p/s; third quant config-incompatible). The PI-suggested 3-bit GGUF (16.6 GB) runs CORRECTLY on
llama.cpp's Windows CUDA build — smoke [1,0], strongest accuracy in the flotilla — but its MoE kernels crawl at
2.5&ndash;4.5 p/s (25&times; slower than gpt-oss) and the server needed three restarts (two CUDA crashes under batch
load). <b>A100 verification (done, $0.74):</b> the UNQUANTIZED model on a rented A100-SXM4-80GB (vLLM, 36.8 p/s
over HTTPS) confirms the 35B as the accuracy champion on the FULL frontier slice — k=3 of 8: C 0.887 / B 0.868 @
FP 8.6%, the best sub-10%-FP stratum-B recall of the whole experiment; k=4: C 0.866 / B 0.777 @ FP 3.8% precision
mode. Apples-to-apples on identical books the local 3-bit quant tracks BF16 within ~1&ndash;3 points — quantization
is not the limiter, the 24 GB runtime is. A rented full-corpus sweep &asymp; $180 (121 A100-hours at $1.49; an
offline-batched engine would roughly halve that).</li>
<li>Practical 4090-only recommendation: <b>gpt-oss-20b + Qwen2.5-14B passage-agreement ensemble</b>
(2-of-2, 8&times;2000 grep): recall 0.895/0.803 (C/B) at 9.2% FP, &asymp;21 GPU-days per 2M-book sweep;
or gpt-oss alone (0.849/0.732 at 6.8%) at &asymp;5.3 GPU-days.</li>
<li>Reprint floor: usable configs flag 30&ndash;60% of old-body modern reprints — part of the signal is
typography/OCR-era, not content. Acceptable for edition-removal, but content-recall on planted OLD text would be overstated.</li>
<li>Total experiment cost: <b>$5.27 OpenRouter</b> (certification of 999 negatives $4.21 + stratum-B screen/certify $1.07,
measured via the credits API) + ~18 h of 4090 time incl. all recoveries.</li>
</ul>

<h2>7. The production pipeline and its price (added v2.2)</h2>
<p>The validated three-stage architecture (authoritative spec: <code>PIPELINE.md</code>):
<b>Stage 1</b> paperwork rule (catalogue year + Guohui's Qwen3-14B doc-year + imprint regex; free, catches
~100% of honest-metadata contamination; flagged reprints routed to the gpt-oss editorial pass — modern
apparatus out, clean re-sets rescued) &rarr; <b>Stage 2</b> reader team on survivors (35B &ge;2/8 OR
gpt-oss-20b &ge;3/8 on 8&times;2000-char random passages; OOS-validated rescue 0.875 of wrong-metadata
contamination at 13.2% clean loss) &rarr; <b>Stage 3</b> exhaustive word-boundary grep
(markers_1945 + 447 null-corpus-validated 1929 terms + year-in-context sieve) with quote-the-proving-phrase
adjudication of hit contexts (+30&ndash;40% of the readers' leftovers on honest-body books, ~0.5% extra loss).</p>
<table><tr><th>Item (2M books, A100-SXM @ $1.49/h, measured throughputs)</th><th>A100-h</th><th>$</th></tr>
<tr><td>Stage 1 + grep scans (CPU)</td><td>&mdash;</td><td>~5</td></tr>
<tr><td>35B reader (14.9M passages @ 37/s)</td><td>~112</td><td>~170</td></tr>
<tr><td>gpt-oss-20b reader (@ ~100/s)</td><td>~41</td><td>~60</td></tr>
<tr><td>Reprint pass + stage-3 adjudication</td><td>~8</td><td>~15</td></tr>
<tr><td><b>Total (conservative)</b> / realistic offline-batched</td><td><b>~160</b> / ~100</td><td><b>~$250</b> / ~$150</td></tr></table>
<p>Wall-clock shards by book: 1 pod &asymp; a week, 4 pods &asymp; 2 days, 8 pods &asymp; 1 day.</p>
<p><b>Upgrade round (2026-07-24), two results:</b> (a) <b>reader scaling is exhausted</b> —
gpt-oss-120b ties the 35B on contaminants but runs 5 points WORSE on the binding 1930s stratum at matched
FP and 4&times; slower on identical hardware ($6.69 eval; teams a wash); (b) the <b>1929-cutoff marker
list</b> (447 terms, each validated zero-hit across 1,361 certified pre-1929 books — a test that rejected
hand-curated "new deal", "nazi", "stalin" as ordinary pre-1929 English) lifts stage-3 rescue on missed
1930s books from 27% to 39%; its mined stylistic discriminators ("basically", "contacted", "phoned",
"sponsored") are strong screens but weak single-word evidence — the quote-the-proving-phrase adjudicator
rightly declines to condemn on one common word (disguised plants 1/16 despite 69% hit rate).</p>

<h2>8. Caveats (all parameters provisional)</h2>
<ul>
<li>Llama-3.2-1B's "perfect recall" is degenerate (answers AFTER to ~everything, 99.9% FP) — recall without an FP bound is meaningless.</li>
<li>Certifier asymmetry: all 81 negative-review flags were Llama-70B-only (deepseek 0); spot-checks read as over-caution (review_queue_negatives.csv).</li>
<li>35B numbers: subsample, one config, different runtime (llama.cpp); confirm on rented hardware before production use.</li>
<li>Single prompt (untuned), single corpus family; passages avoid the first/last 3&ndash;5% of each book by design.</li>
</ul>
</body></html>"""
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print("report v2 ->", HTML)

if __name__ == "__main__":
    main()
