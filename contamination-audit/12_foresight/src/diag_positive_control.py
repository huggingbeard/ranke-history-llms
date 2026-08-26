# -*- coding: utf-8 -*-
"""DIAGNOSTIC: can the in-family estimator register unforeseeability AT ALL?

The panel validated the null side (panic reads foreseeable; moon reads ~0 in-family BY DESIGN, both minds
ignorant) but never the alternative: an event that MUST read unforeseeable in-family. This battery supplies
it: 1901-1929 novelties that are BANAL to the 1929 mind (in its corpus as daily life) but conceptually novel
to the 1900 mind. Internal gradient prediction if the instrument works:
    motorcars (widely predicted in 1900)  ~ 0
    women voting (contested, movement existed) ~ low +
    flight / Atlantic / radio / talkies / tanks  >> 0
If ALL read ~0, the instrument cannot detect even the arrival of aviation and radio -> broken for its purpose
(candidate mechanics: register cosplay - the 1929 corpus contains 1913-style speculation and forward-framed
prompts summon it; signal dilution over long targets; light neutrals absorbing the load).

Scores each item under two frames (light per-item neutral, deep constant generic neutral) and also reports a
FIRST-CONTENT-TOKEN variant (dilution test). Writes results/diag_positive_control.json.
Run: python src/diag_positive_control.py
"""
import os, json, statistics as st
import torch
import common
from foresight_bank import RANKE, MODERN

# item[lang] = (subject, target, light_neutral); DEEP neutral is constant per language.
DEEP = {"en": "In time, they", "de": "Mit der Zeit werden sie", "fr": "Avec le temps, ils"}
# NOTE DE grammar: subjects end "...werden die/der X", targets are verb-final WITHOUT the auxiliary,
# light neutrals include "werden" so every frame licenses the same target string.
ITEMS = [
 ("flight", dict(
   en=("In the century now beginning, men", " will fly through the air in machines heavier than air", "Men"),
   de=("In dem Jahrhundert, das nun beginnt, werden die Menschen", " in Maschinen, schwerer als die Luft, durch die Lüfte fliegen", "Die Menschen werden"),
   fr=("Dans le siècle qui commence, les hommes", " voleront dans les airs sur des machines plus lourdes que l'air", "Les hommes"))),
 ("atlantic", dict(
   en=("Before this century is old, a flying man", " will cross the whole Atlantic without once alighting", "A man"),
   de=("Ehe dieses Jahrhundert alt ist, wird ein fliegender Mensch", " den ganzen Atlantik überqueren, ohne ein einziges Mal zu landen", "Ein Mensch wird"),
   fr=("Avant que ce siècle soit vieux, un homme volant", " traversera tout l'Atlantique sans se poser une seule fois", "Un homme"))),
 ("radio", dict(
   en=("In the years to come, music and the day's news", " will be sent through the empty air into every parlour", "Music and news"),
   de=("In den kommenden Jahren werden Musik und die Nachrichten des Tages", " durch die leere Luft in jede Wohnstube gesandt werden", "Musik und Nachrichten werden"),
   fr=("Dans les années à venir, la musique et les nouvelles du jour", " seront envoyées à travers l'air vide dans chaque salon", "La musique et les nouvelles"))),
 ("talkies", dict(
   en=("Before long, the moving pictures", " will speak aloud with the voices of the players", "The pictures"),
   de=("Binnen kurzem werden die lebenden Bilder", " mit den Stimmen der Schauspieler laut sprechen", "Die Bilder werden"),
   fr=("Avant longtemps, les images animées", " parleront tout haut avec les voix des acteurs", "Les images"))),
 ("tanks", dict(
   en=("In the wars to come, great armoured machines", " will crawl over trench and wire on belts of iron", "Great machines"),
   de=("In den kommenden Kriegen werden große gepanzerte Maschinen", " auf eisernen Bändern über Graben und Draht kriechen", "Große Maschinen werden"),
   fr=("Dans les guerres à venir, de grandes machines blindées", " ramperont sur des chenilles de fer par-dessus tranchées et barbelés", "De grandes machines"))),
 ("women_vote", dict(
   en=("In the new century, the women", " will cast their votes in Britain, in Germany and in America alike", "The women"),
   de=("Im neuen Jahrhundert werden die Frauen", " in England, in Deutschland und in Amerika gleichermaßen ihre Stimme abgeben", "Die Frauen werden"),
   fr=("Dans le siècle nouveau, les femmes", " voteront en Angleterre, en Allemagne et en Amérique également", "Les femmes"))),
 ("motorcars", dict(   # widely predicted in 1900 -> the LOW end of the gradient
   en=("Before many years have passed, the streets of the great cities", " will be filled with motor carriages and empty of horses", "The streets"),
   de=("Ehe viele Jahre vergangen sind, werden die Straßen der großen Städte", " voller Motorwagen und leer von Pferden sein", "Die Straßen werden"),
   fr=("Avant que bien des années aient passé, les rues des grandes villes", " seront pleines de voitures à moteur et vides de chevaux", "Les rues"))),
 ("moon", dict(        # future-to-ALL: must stay ~0 in-family (both minds ignorant) - the design null
   en=("One day men", " will walk upon the face of the moon", "Men"),
   de=("Eines Tages werden die Menschen", " auf dem Antlitz des Mondes wandeln", "Die Menschen werden"),
   fr=("Un jour, les hommes", " marcheront sur la face de la lune", "Les hommes"))),
]

MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"]), ("modern", MODERN)]
LANGS = ("en", "de", "fr")


def slp_and_first(model, tok, prompt, cont):
    """(sum logp, logp of the first content token) — the first-token variant tests dilution."""
    pid = tok(prompt, add_special_tokens=False).input_ids
    cid = tok(cont, add_special_tokens=False).input_ids
    full = pid + cid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    tot, first = 0.0, None
    for k in range(len(cid)):
        v = lp[len(pid) + k - 1, full[len(pid) + k]].item()
        tot += v
        if first is None and k >= 1:   # skip leading " will"/aux token
            first = v
    return tot, first if first is not None else tot


res = {}
for tag, mid in MODELS:
    print(f"=== {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    for name, it in ITEMS:
        for lg in LANGS:
            subj, tgt, lightn = it[lg]
            s_c, f_c = slp_and_first(model, tok, subj, tgt)
            s_l, f_l = slp_and_first(model, tok, lightn, tgt)
            s_d, f_d = slp_and_first(model, tok, DEEP[lg], tgt)
            res[f"{name}/{lg}/{tag}"] = {
                "strain_light": round(-(s_c - s_l), 3), "strain_deep": round(-(s_c - s_d), 3),
                "first_light": round(-(f_c - f_l), 3), "first_deep": round(-(f_c - f_d), 3)}
    del tok, model
    common.free()

json.dump(res, open(os.path.join(common.RESULTS, "diag_positive_control.json"), "w", encoding="utf-8"), indent=1)

print("\n" + "=" * 96)
print("POSITIVE-CONTROL DIAGNOSTIC — Dconc(1900 vs 1929), the in-family estimator, per frame")
print("  (instrument works => flight/atlantic/radio/talkies/tanks >> 0; motorcars ~0; moon ~0 by design)")
print("=" * 96)
print(f"{'item':12s}{'lang':6s}{'LIGHT frame':>12s}{'DEEP frame':>12s}{'FIRST-TOKEN(deep)':>18s}")
for name, it in ITEMS:
    for lg in LANGS:
        dl = res[f"{name}/{lg}/1900"]["strain_light"] - res[f"{name}/{lg}/1929"]["strain_light"]
        dd = res[f"{name}/{lg}/1900"]["strain_deep"] - res[f"{name}/{lg}/1929"]["strain_deep"]
        df = res[f"{name}/{lg}/1900"]["first_deep"] - res[f"{name}/{lg}/1929"]["first_deep"]
        print(f"{name:12s}{lg:6s}{dl:>+12.2f}{dd:>+12.2f}{df:>+18.2f}")
    ml = st.mean(res[f"{name}/{lg}/1900"]["strain_light"] - res[f"{name}/{lg}/1929"]["strain_light"] for lg in LANGS)
    md = st.mean(res[f"{name}/{lg}/1900"]["strain_deep"] - res[f"{name}/{lg}/1929"]["strain_deep"] for lg in LANGS)
    print(f"{name:12s}{'MEAN':6s}{ml:>+12.2f}{md:>+12.2f}")
print("\nwrote results/diag_positive_control.json")
