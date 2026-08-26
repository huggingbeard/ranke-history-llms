# -*- coding: utf-8 -*-
"""German counterpart of hitler_analyze.py: build fig_hitler_de.png + a self-contained (figure embedded)
hitler_report_de.html from results/hitler_probe/generations_de.json. No GPU.

Two rates, both honest: LOOSE = any Munich/Reichstag/movement keyword (inflated by German obituary/news
confabulation that just happens to name Munich); STRICT = a genuine 1920s-Hitler marker (the National
Socialists, the Hitler-Putsch, the voelkisch movement, the Muenchner Post, mass-movement leadership).
Run:  python src/hitler_report_de.py
"""
import os, re, json, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import common

OUT = os.path.join(common.RESULTS, "hitler_probe")
d = json.load(open(os.path.join(OUT, "generations_de.json"), encoding="utf-8"))
PROMPTS = {"who_is": "Wer ist Adolf Hitler?", "def": "Adolf Hitler ist"}
MODELS = ["1913", "1929"]
INK, BLUE, GREY = "#1a1a1a", "#2b6cb0", "#9aa0a6"

LOOSE = (r"bayern|bayerisch|münchen|münchner|reichstag|nationalsozialist|nazi|völkisch|putsch|agitator|"
         r"hetzer|vaterland|weimar|hindenburg|ludendorff|nsdap|braunau|hakenkreuz|bewegung|umsturz|aufstand|revolution")
STRICT = (r"nationalsozialist|\bnazi|völkisch|hakenkreuz|nsdap|\bputsch|hetzer|agitator|umsturz|"
          r"(führer|leiter).{0,20}(partei|bewegung)|(starke|neue|große).{0,15}bewegung|masse des volkes|münchner post|braunau")


def bucket(t):
    t = t.lower()
    if re.search(STRICT, t):
        return "identifies the real agitator (National Socialists / Hitler-Putsch / voelkisch)"
    if re.search(LOOSE, t):
        return "names Munich/Reichstag but confabulates (no real identity)"
    return "generic confabulation (name treated as an unknown person)"


ORDER = ["identifies the real agitator (National Socialists / Hitler-Putsch / voelkisch)",
         "names Munich/Reichstag but confabulates (no real identity)",
         "generic confabulation (name treated as an unknown person)"]
dist = {p: {m: {b: 0 for b in ORDER} for m in MODELS} for p in PROMPTS}
for p in PROMPTS:
    for m in MODELS:
        for g in d[m][p][:100]:
            dist[p][m][bucket(g)] += 1

# ---- figure ----
fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.6), sharey=True)
short = ["identifies the real\nagitator (strict)", "names Munich etc.\nbut confabulates", "generic\nconfabulation"]
for ax, p in zip(axes, ["def", "who_is"]):
    x = np.arange(3); w = 0.38
    v13 = [dist[p]["1913"][b] for b in ORDER]; v29 = [dist[p]["1929"][b] for b in ORDER]
    ax.bar(x - w/2, v13, w, color=GREY, edgecolor=INK, lw=.7, label="Ranke 1913")
    ax.bar(x + w/2, v29, w, color=BLUE, edgecolor=INK, lw=.7, label="Ranke 1929")
    for i, (a, b) in enumerate(zip(v13, v29)):
        ax.annotate(str(a), (i - w/2, a + 1), ha="center", fontsize=8)
        ax.annotate(str(b), (i + w/2, b + 1), ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(short, fontsize=8.2)
    ax.set_title(f"prompt: “{PROMPTS[p]}”", fontsize=10)
    ax.set_ylim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GREY, alpha=.3, lw=.6)
axes[0].set_ylabel("share of 100 free continuations (%)")
axes[0].legend(frameon=False, fontsize=9, loc="upper left")
fig.suptitle("German: Ranke 1913 vs 1929, „Wer ist Adolf Hitler?“, 100 free continuations each.\n"
             "1913 never identifies him; the 1929 model names the National Socialists, the Hitler-Putsch and the Muenchner Post.",
             fontsize=11, y=1.06)
fig.tight_layout()
FIG = os.path.join(common.FIGURES, "fig_hitler_de.png")
fig.savefig(FIG, bbox_inches="tight", dpi=150)
b64 = base64.b64encode(open(FIG, "rb").read()).decode()

# ---- HTML (figure embedded) ----
def rows(p):
    return "\n".join(f"<tr><td>{b}</td><td class=n>{dist[p]['1913'][b]}%</td><td class=n>{dist[p]['1929'][b]}%</td></tr>"
                     for b in ORDER)
def ex(m, p, rx, k=7):
    hits = [g.strip() for g in d[m][p][:100] if re.search(rx, g.lower())]
    return "\n".join(f"<li>{g[:260]}</li>" for g in hits[:k])

html = f"""<!doctype html><meta charset=utf-8><title>Ranke 1913 vs 1929 — Wer ist Adolf Hitler?</title>
<style>body{{font:16px/1.55 Georgia,serif;max-width:900px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{font-size:1.15rem;margin-top:1.8rem;border-bottom:1px solid #ddd}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}} td,th{{border:1px solid #ccc;padding:5px 9px}}
.n{{text-align:right}} li{{margin:.35rem 0}} .cap{{color:#555;font-size:.9rem}} img{{max-width:100%}}</style>
<h1>Zwei eingefrorene Leser, eine Frage: „Wer ist Adolf Hitler?“ (deutsch)</h1>
<p>Dieselbe Frage, hundertmal, freie Fortsetzung (Temperatur&nbsp;0.8). <b>Ranke&nbsp;1913</b> hat nichts nach 1913
gelesen; <b>Ranke&nbsp;1929</b> nichts nach 1929. 1913 war Hitler ein unbekannter 24-Jaehriger in Wien; 1929
war er der Fuehrer der NSDAP, Verfasser von <i>Mein Kampf</i> (1925), verurteilt wegen des Muenchner Putsches
von 1923 — aber noch <b>nicht</b> Reichskanzler (Januar 1933).</p>
<h2>Was jedes Modell sagt (100 freie Fortsetzungen, Prompt „Adolf Hitler ist“)</h2>
<table><tr><th>Kategorie</th><th>Ranke 1913</th><th>Ranke 1929</th></tr>{rows('def')}</table>
<img src="data:image/png;base64,{b64}" alt="Verteilung">
<p class=cap>x-Achse: die drei Kategorien; y-Achse: Anteil der 100 Fortsetzungen. Grau = 1913, blau = 1929.
Links der definitorische Prompt, rechts die Frage.</p>
<h2>Ranke 1929 — echtes Wissen (Auszuege)</h2><ul>{ex('1929','def',STRICT)}</ul>
<h2>Ranke 1913 — die „Treffer“ sind Konfabulation (Auszuege)</h2><ul>{ex('1913','def',LOOSE)}</ul>
<h2>Kein Modell kennt 1933</h2>
<p>Wo das 1929er-Modell „Reichskanzler“ sagt, konfabuliert es Weimarer Kabinettspolitik (Marx, Wirth) oder
knuepft an den Putsch von 1923 an; keine Fortsetzung nennt die Machtergreifung 1933. Das 1929er-Modell kennt
den aufsteigenden Agitator der zwanziger Jahre — nicht das Ende der Geschichte.</p>
"""
open(os.path.join(OUT, "hitler_report_de.html"), "w", encoding="utf-8").write(html)
print("def bucket dist:", {m: dist["def"][m] for m in MODELS})
print("wrote figures/fig_hitler_de.png and results/hitler_probe/hitler_report_de.html")
