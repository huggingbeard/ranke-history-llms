# -*- coding: utf-8 -*-
"""'Does the frozen model HOLD the person?' — final metric + figure + self-contained report.

Metric history (kept honest):
  1. name-recurrence  -> FAILED: confounded by referential style (Bismarck is "the Prince/he"; the
     Hitler confabulations keep a fake "Mr. Hiller" going). Measures style, not knowledge.
  2. TF-IDF cross-sample consistency -> NULL: all cells ~0.09, period-newspaper style swamps content.
  3. ON-DOMAIN (does the continuation evoke the person's real world) -> WORKS, and a two-domain
     confusion check shows it is specific: each known person lights up its own domain, not the other's.

Bismarck (died 1898) is the positive control a clean 1913 model must hold. Result: 1913 holds Bismarck
but not Hitler; by 1929 Hitler comes into focus. Builds fig_holds.png + hitler_bismarck_holds.html.
Run:  python src/holds_report.py
"""
import os, re, json, base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import common

B = common.RESULTS
DATA = {("hitler", "en"): os.path.join(B, "hitler_probe", "generations.json"),
        ("hitler", "de"): os.path.join(B, "hitler_probe", "generations_de.json"),
        ("bismarck", "en"): os.path.join(B, "entity_probe", "bismarck_en.json"),
        ("bismarck", "de"): os.path.join(B, "entity_probe", "bismarck_de.json")}
# strict, entity-specific domains (broad terms like "german/berlin" removed from Bismarck's)
DOM = {"bismarck": r"chancellor|kanzler|preuß|preuss|prussia|kaiser|emperor|fürst|\bprince\b|staatsmann|statesman|"
                   r"friedrichsruh|hohenzollern|blood and iron|eisen und blut|1866|1870|1871|1890",
       "hitler": r"bavaria|bayern|münchen|munich|münchner|nationalsozialist|national ?socialis|nazi|völkisch|"
                 r"volkisch|putsch|agitat|hetzer|nsdap|braunau|hakenkreuz|bewegung|movement|umsturz|aufstand|radical|reichstag"}
OTHER = {"bismarck": "hitler", "hitler": "bismarck"}
raw = {k: json.load(open(v, encoding="utf-8")) for k, v in DATA.items()}


def rate(cs, rx):
    return 100 * sum(1 for g in cs if re.search(rx, g.lower())) / len(cs)


own, cross = {}, {}
for (ent, lang), d in raw.items():
    for m in ["1913", "1929"]:
        cs = [g for p in ["who_is", "def"] for g in d[m][p][:100]]
        own[(ent, lang, m)] = rate(cs, DOM[ent])
        cross[(ent, lang, m)] = rate(cs, DOM[OTHER[ent]])

cells = [("bismarck", "1913"), ("bismarck", "1929"), ("hitler", "1913"), ("hitler", "1929")]
labs = ["Bismarck\n1913", "Bismarck\n1929", "Hitler\n1913", "Hitler\n1929"]
print(f"{'cell':16} | own-domain (EN/DE) | other-domain (EN/DE)")
for e, m in cells:
    print(f"{e+' '+m:16} |   {own[(e,'en',m)]:3.0f}% / {own[(e,'de',m)]:3.0f}%     |   "
          f"{cross[(e,'en',m)]:3.0f}% / {cross[(e,'de',m)]:3.0f}%")

# ---- figure ----
x = np.arange(4); w = 0.38
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
for ax, D, ttl in [(a1, own, "Holds the protagonist: evokes the RIGHT person's world"),
                   (a2, cross, "Specificity check: evokes the OTHER person's world")]:
    for off, lang, col, nm in [(-w/2, "en", "#2b6cb0", "English"), (w/2, "de", "#c0392b", "German")]:
        vals = [D[(e, lang, m)] for e, m in cells]
        ax.bar(x + off, vals, w, color=col, edgecolor="#1a1a1a", lw=.7, label=nm)
        for i, v in enumerate(vals):
            ax.annotate(f"{v:.0f}", (i + off, v + 1.3), ha="center", fontsize=8.4)
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=9)
    ax.set_title(ttl, fontsize=10, loc="left")
    ax.set_ylim(0, 70)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#9aa0a6", alpha=.3, lw=.6)
a1.set_ylabel("share of 100 free continuations (%)")
a1.legend(frameon=False, fontsize=9, title="prompt language")
fig.suptitle("Does the frozen model hold the person it is asked about? (200 continuations per bar)\n"
             "Bismarck (died 1898) is held by both models; Hitler is empty for 1913 and comes into focus by 1929.\n"
             "Right panel near zero = the signal is the right person, not generic politics.",
             fontsize=10.6, y=1.08)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_holds.png")
fig.savefig(FIG, bbox_inches="tight", dpi=150)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

# ---- verbatim for the report ----
def ex(ent, lang, m, rx_match, k=5):
    cs = [g.strip() for p in ["who_is", "def"] for g in raw[(ent, lang)][m][p][:100]]
    hit = [g for g in cs if re.search(rx_match, g.lower())]
    return "\n".join(f"<li>{g[:230]}</li>" for g in hit[:k])

html = f"""<!doctype html><meta charset=utf-8><title>Does the model hold the protagonist? Hitler vs Bismarck</title>
<style>body{{font:16px/1.55 Georgia,serif;max-width:920px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:1.7rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0}} td,th{{border:1px solid #ccc;padding:5px 10px}}
.n{{text-align:right}} li{{margin:.35rem 0}} .cap{{color:#555;font-size:.9rem}} img{{max-width:100%}}</style>
<h1>Does a frozen model &ldquo;hold&rdquo; the person it is asked about?</h1>
<p>We asked Ranke-1913 and Ranke-1929 &ldquo;who is X?&rdquo;, 100 free continuations each, for two people:
<b>Otto von Bismarck</b> (died 1898, so known to <i>both</i> models &mdash; the control) and <b>Adolf Hitler</b>
(an unknown in 1913, a Munich agitator by 1929). A model that holds a person evokes that person&rsquo;s real
world; an empty name scatters. We score the share of continuations that evoke each person&rsquo;s domain, and
check that the signal is specific by also scoring against the <i>other</i> person&rsquo;s domain.</p>
<img src="data:image/png;base64,{b64}" alt="holds the protagonist">
<h2>The numbers (own domain, pooled over both prompts)</h2>
<table><tr><th>model asked about</th><th>English</th><th>German</th></tr>
<tr><td>Bismarck &mdash; Ranke 1913</td><td class=n>{own[('bismarck','en','1913')]:.0f}%</td><td class=n>{own[('bismarck','de','1913')]:.0f}%</td></tr>
<tr><td>Bismarck &mdash; Ranke 1929</td><td class=n>{own[('bismarck','en','1929')]:.0f}%</td><td class=n>{own[('bismarck','de','1929')]:.0f}%</td></tr>
<tr><td>Hitler &mdash; Ranke 1913</td><td class=n>{own[('hitler','en','1913')]:.0f}%</td><td class=n>{own[('hitler','de','1913')]:.0f}%</td></tr>
<tr><td>Hitler &mdash; Ranke 1929</td><td class=n>{own[('hitler','en','1929')]:.0f}%</td><td class=n>{own[('hitler','de','1929')]:.0f}%</td></tr></table>
<p class=cap>The 1913 model holds Bismarck about ten times as often as Hitler. Acquiring the 1913&ndash;1929
newspapers moves Hitler from near-zero into focus; Bismarck, already known, barely moves.</p>
<h2>Ranke 1913 holds Bismarck &mdash; verbatim (English)</h2><ul>{ex('bismarck','en','1913',DOM['bismarck'])}</ul>
<h2>Ranke 1913 does not hold Hitler &mdash; verbatim (English)</h2><ul>{ex('hitler','en','1913',r'.')}</ul>
<h2>Metric history (kept honest)</h2>
<p>Two earlier attempts failed and are recorded so the choice can be judged. <b>Name-recurrence</b>
(&ldquo;does the surname reappear&rdquo;) was confounded by style: Bismarck is called &ldquo;the Prince&rdquo;
or &ldquo;he&rdquo;, so he scored as <i>not</i> sticking, while a confabulated &ldquo;Mr.&nbsp;Hiller&rdquo;
scored as sticking. <b>TF-IDF cross-sample consistency</b> was a null (every cell &asymp;&nbsp;0.09) because
shared newspaper style swamps the entity content. The on-domain rate above is what separates knowing from
not knowing, and the specificity panel confirms it is the right person, not generic politics.</p>
"""
# ===================== JUDGE-based "is it the same person/concept?" =====================
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

jf = os.path.join(B, "entity_probe", "judge_results.json")
jp = os.path.join(B, "entity_probe", "judge_prompt.txt")
if os.path.exists(jf) and os.path.exists(jp):
    judge = json.load(open(jf, encoding="utf-8"))
    jprompt = open(jp, encoding="utf-8").read()

    def same_rate(ent, lang, m):
        vs = [judge.get(f"{ent}|{lang}|{m}|{p}|{i}") for p in ["who_is", "def"] for i in range(100)]
        vs = [v for v in vs if v]
        return round(100 * vs.count("SAME") / len(vs)) if vs else "--"
    jrows = "\n".join(
        f"<tr><td>{e.title()} &mdash; Ranke {m}</td>"
        f"<td class=n>{own[(e,'en',m)]:.0f}%</td><td class=n>{same_rate(e,'en',m)}%</td>"
        f"<td class=n>{own[(e,'de',m)]:.0f}%</td><td class=n>{same_rate(e,'de',m)}%</td></tr>"
        for e, m in cells)
    html += f"""
<h2>The &ldquo;same person/concept?&rdquo; judgment &mdash; the exact prompt</h2>
<p>The on-domain figure above is a keyword proxy (an upper bound). To decide, per continuation, whether the
text is really about the same person &mdash; without my hand-adjudication &mdash; each of the 1,600 Hitler and
Bismarck continuations was classified by an independent judge (<code>gemini-3.1-flash-lite</code>,
temperature&nbsp;0). This is the verbatim prompt, with <code>{{desc}}</code> filled by a one-line identity of
the person and <code>{{text}}</code> by the continuation:</p>
<pre style="background:#f4f6f8;padding:1rem;border-radius:5px;white-space:pre-wrap;font-size:.86rem">{esc(jprompt)}</pre>
<p>The judge corroborates the keyword proxy and removes the hand-reading from the loop:</p>
<table><tr><th rowspan=2>asked about</th><th colspan=2>English</th><th colspan=2>German</th></tr>
<tr><th>keyword</th><th>judge SAME</th><th>keyword</th><th>judge SAME</th></tr>
{jrows}</table>
<p class=cap>Validation: of the eight English Hitler-1913 keyword &ldquo;hits&rdquo; that I had hand-read as
false positives, the judge independently called seven DIFFERENT; the one it called SAME is a borderline
&ldquo;serious figure in the Socialist movement&hellip;&rdquo; case. The judge&rsquo;s Hitler-1913 rate (5&ndash;6%)
is thus an upper estimate; the true rate is near zero.</p>
"""

# ===================== "Hitler's regime persecuted ___" (with the obscure alternative) =====================
rf = os.path.join(B, "regime_probe", "results.json")
if os.path.exists(rf):
    reg = json.load(open(rf, encoding="utf-8"))
    OBS = {"en": {" the Jehovah's Witnesses", " the Bible Students"},
           "de": {" die Zeugen Jehovas", " die Ernsten Bibelforscher"}}
    FRAME = {"en": "Adolf Hitler's regime persecuted", "de": "Adolf Hitlers Regime verfolgte"}
    blocks = ""
    for lang in ["en", "de"]:
        F = reg["forced"].get(lang, {})
        if "1913" not in F or "1929" not in F:
            continue
        E = F["1913"]["endings"]
        rows = "\n".join(
            f"<tr><td>{esc(e.strip())}{' <b>&larr; obscure, real</b>' if e in OBS[lang] else ''}</td>"
            f"<td class=n>{F['1913']['p'][i]:.0%}</td><td class=n>{F['1929']['p'][i]:.0%}</td></tr>"
            for i, e in enumerate(E))
        blocks += (f"<p><b>{esc(FRAME[lang])} ___</b> &mdash; P(each ending):</p>"
                   f"<table><tr><th>ending</th><th>Ranke 1913</th><th>Ranke 1929</th></tr>{rows}</table>")
    html += f"""
<h2>Postscript: &ldquo;Adolf Hitler&rsquo;s regime persecuted ___&rdquo;</h2>
<p>&ldquo;Hitler&rsquo;s regime&rdquo; is a post-1933 thing &mdash; past the cutoff for <i>both</i> models, so
neither can recall it. The tell is what they fill in. The real headline target (the Jews) is also
period-plausible: a regime persecuting the Jews was ordinary pre-1913 news (Russian pogroms, the Dreyfus
affair), so producing it is not evidence of knowing the Nazi regime. The discriminating target is one the
regime really persecuted but that no 1913/1929 prior reaches &mdash; the <b>Bible Students</b> (Ernste
Bibelforscher, the pre-1931 name for Jehovah&rsquo;s Witnesses).</p>
{blocks}
<p class=cap>Both models &mdash; including 1913, which holds no concept of Hitler at all &mdash; put almost
all their weight on &ldquo;the Jews&rdquo;, the period-plausible answer, and <b>zero</b> on the obscure-but-real
Bible Students. Period-plausible truth is confabulated from priors; genuinely unforeseeable truth is not.
This is the paper&rsquo;s thesis in one sentence.</p>
"""

# ===================== placebo control: is "persecuted the Jews" specific, or a frame default? =====================
cf = os.path.join(B, "regime_probe", "control.json")
if os.path.exists(cf):
    ctrl = json.load(open(cf, encoding="utf-8"))
    LEAD = {"hitler": "Adolf Hitler", "churchill": "Winston Churchill",
            "bethmann": "Bethmann-Hollweg (Chancellor 1909–17; no Jewish persecution)"}
    ptbl = ""
    for m in ["1913", "1929"]:
        rows = "\n".join(
            f"<tr><td>{LEAD[l]}</td>"
            f"<td class=n>{ctrl[m][l]['en']['p'][0]:.0%}</td>"
            f"<td class=n>{ctrl[m][l]['de']['p'][0]:.0%}</td>"
            f"<td class=n>{ctrl[m][l]['fr']['p'][0]:.0%}</td></tr>" for l in LEAD)
        ptbl += (f"<p><b>Ranke {m}</b> &mdash; P(&ldquo;persecuted the Jews&rdquo;):</p>"
                 f"<table><tr><th>leader</th><th>English</th><th>German</th><th>French</th></tr>{rows}</table>")
    html += f"""
<h2>Placebo control: is &ldquo;persecuted the Jews&rdquo; specific, or a frame default?</h2>
<p>Swap Hitler for leaders who persecuted no Jews &mdash; Winston Churchill, and Chancellor Bethmann-Hollweg
(1909&ndash;1917, a real figure in both corpora) &mdash; and read the same P(&ldquo;persecuted the Jews&rdquo;)
in three languages.</p>
{ptbl}
<p class=cap>The completion &ldquo;persecuted the Jews&rdquo; is largely a <b>frame default</b>, not knowledge:
in English and French, Bethmann-Hollweg &mdash; who persecuted no one &mdash; scores as high as Hitler
(&asymp;&nbsp;80&ndash;90%). The default is weaker for the British &ldquo;Churchill&rdquo; and weakest in
German, which spreads its guess across period-salient targets (Kulturkampf Catholics, the anti-socialist
tradition) and, for the 1929 Hitler, toward the Socialists it knows were his real enemies. So German is
<b>lower</b>, not higher &mdash; the opposite of a German antisemitic default. (Cross-language absolute levels
are affected by how each object set tokenizes; the clean test is Hitler vs the placebos <i>within</i> a
language, where Bethmann &asymp; Hitler in EN and FR.) The one invariant across every leader and language:
zero weight on the obscure-but-real Bible Students &mdash; the target that would require actual regime
knowledge.</p>
"""

open(os.path.join(B, "entity_probe", "hitler_bismarck_holds.html"), "w", encoding="utf-8").write(html)
print("\nwrote figures/fig_holds.png and results/entity_probe/hitler_bismarck_holds.html")
