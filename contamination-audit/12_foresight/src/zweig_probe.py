# -*- coding: utf-8 -*-
"""ZWEIG WORLDVIEW PROBE (v3: five items/theme; women items purely NORMATIVE; calibrated; EN/DE/FR).

Stefan Zweig, 'The World of Yesterday' (1942), catalogues the pre-1914 mind -- the 'Golden Age of Security'.
We turn the FOUR themes he himself dwells on -- faith in progress, the unthinkability of war, the confinement
of women, and the veneration of age over youth -- into belief-vs-skeptic forced choices, in ENGLISH, GERMAN
and FRENCH, and ask each Ranke model (1900/1913/1929) + a modern model which it prefers.

Metric -- CALIBRATED preference (removes the fluency confound: a belief phrasing may just be more idiomatic):
    c(a)      = slp(stem, a) - slp(neutral, a)          # 'a' scored in the loaded frame vs a de-loaded frame
    cal_pref  = c(belief) - c(skeptic)                  # >0 => the PERIOD framing raises the belief above the
                                                        #        skeptic MORE than a neutral framing does
The fluency of a phrasing is the same in the neutral frame, so it cancels; cal_pref is worldview, not idiom.

Cross-lingual point: the same model, prompted in different languages, encodes different pre-war worldviews
(the English corpus was war-aware; the German 'world of security' was war-complacent; French = ?). We read the
'war_unthinkable' theme in all three languages, for 1900/1913/1929, and against the modern (disillusioned) model.
Writes zweig_probe.json + console. Run: python zweig_probe.py
"""
import os, json, statistics as st
import common
from foresight_bank import RANKE, MODERN

# item[lang] = (stem, belief[pre-1914], skeptic[contrary], neutral[de-loaded frame for calibration])
# The FOUR themes Zweig himself dwells on in 'The World of Yesterday': the faith in irresistible progress,
# the unthinkability of a European war, the confinement of women, and the veneration of age over youth.
THEMES = [
 ("progress", [
   dict(en=("The men of that age believed that progress was", " uninterrupted and irresistible", " fragile and easily reversed", "A thing might be"),
        de=("Die Menschen jener Zeit glaubten, der Fortschritt sei", " ununterbrochen und unaufhaltsam", " zerbrechlich und leicht umkehrbar", "Eine Sache sei"),
        fr=("Les hommes de cette époque croyaient que le progrès était", " ininterrompu et irrésistible", " fragile et facilement renversé", "Une chose était")),
   dict(en=("Each passing decade was seen as", " the prelude to a better one", " the prelude to a darker one", "A period was seen as"),
        de=("Jedes verflossene Jahrzehnt galt als", " Auftakt zu einem besseren", " Auftakt zu einem finstereren", "Ein Zeitabschnitt galt als"),
        fr=("Chaque décennie écoulée était vue comme", " le prélude à une meilleure", " le prélude à une plus sombre", "Une période était vue comme")),
   dict(en=("The last remnants of evil and violence would soon be", " conquered for ever by reason", " loosed again upon the world", "Such things would soon be"),
        de=("Die letzten Reste des Bösen und der Gewalt würden bald", " für immer durch die Vernunft besiegt", " aufs Neue über die Welt entfesselt", "Solche Dinge würden bald"),
        fr=("Les derniers restes du mal et de la violence seraient bientôt", " à jamais vaincus par la raison", " de nouveau déchaînés sur le monde", "De telles choses seraient bientôt")),
 ]),
 ("war", [
   dict(en=("To the men of that age a great European war seemed", " as impossible as witches and ghosts", " as near and certain as the sunrise", "A distant thing seemed"),
        de=("Den Menschen jener Zeit schien ein großer europäischer Krieg", " so unmöglich wie Hexen und Gespenster", " so nah und gewiss wie der Sonnenaufgang", "Eine ferne Sache schien"),
        fr=("Aux hommes de cette époque une grande guerre européenne semblait", " aussi impossible que les sorcières et les fantômes", " aussi proche et certaine que le lever du soleil", "Une chose lointaine semblait")),
   dict(en=("The reasonable men of that day expected", " a long and lasting peace", " a great and coming war", "A man expected"),
        de=("Die vernünftigen Menschen jener Zeit erwarteten", " einen langen und dauerhaften Frieden", " einen großen und kommenden Krieg", "Ein Mensch erwartete"),
        fr=("Les hommes raisonnables de ce temps attendaient", " une paix longue et durable", " une guerre grande et prochaine", "Un homme attendait")),
   dict(en=("That the nations of Europe might fall upon one another seemed", " a barbaric relic of ruder times", " the plain danger of the coming years", "Such an event seemed"),
        de=("Dass die Völker Europas übereinander herfallen könnten, schien", " ein barbarisches Überbleibsel roher Zeiten", " die deutliche Gefahr der kommenden Jahre", "Ein solches Ereignis schien"),
        fr=("Que les nations d'Europe pussent se jeter les unes sur les autres semblait", " une relique barbare de temps plus rudes", " le danger manifeste des années à venir", "Un tel événement semblait")),
 ]),
 ("women", [
   # v2: all five items are NORMATIVE ('the age held/thought/felt') — the v1 items conflated the
   # period's beliefs with the period's facts, which the modern mind knows as history.
   dict(en=("The men of that age held that a respectable woman's place", " was in the home and nowhere else", " was in the world, the same as a man's", "It was held that the proper place"),
        de=("Die Männer jener Zeit hielten dafür, der Platz einer anständigen Frau", " sei im Hause und nirgendwo sonst", " sei in der Welt, ganz wie der des Mannes", "Man hielt dafür, der rechte Platz"),
        fr=("Les hommes de cette époque tenaient que la place d'une femme respectable", " était au foyer et nulle part ailleurs", " était dans le monde, tout comme celle d'un homme", "On tenait que la juste place")),
   dict(en=("To the minds of that age, a woman who spoke in public", " disgraced herself and her family", " did honour to herself and her family", "It was felt that such a person"),
        de=("Nach den Begriffen jener Zeit hat eine Frau, die öffentlich sprach,", " sich und ihre Familie entehrt", " sich und ihrer Familie Ehre gemacht", "Man fand, eine solche Person hat"),
        fr=("Pour les esprits de cette époque, une femme qui parlait en public", " se déshonorait, elle et sa famille", " s'honorait, elle et sa famille", "On sentait qu'une telle personne")),
   dict(en=("In that age it was thought that giving women the vote", " would be against nature and against order", " would be simple justice, too long delayed", "It was thought that the change"),
        de=("In jener Zeit meinte man, das Stimmrecht für die Frauen", " wäre wider die Natur und wider die Ordnung", " wäre einfache, zu lange versagte Gerechtigkeit", "Man meinte, die Änderung"),
        fr=("En ce temps-là, on pensait que donner le vote aux femmes", " serait contre la nature et contre l'ordre", " serait une simple justice, trop longtemps différée", "On pensait que le changement")),
   dict(en=("The fathers of that age held that a daughter", " must obey, first her father, then her husband", " should choose her own path in life", "It was held that a child"),
        de=("Die Väter jener Zeit hielten dafür, eine Tochter", " müsse gehorchen, erst dem Vater, dann dem Gatten", " solle ihren eigenen Weg im Leben wählen", "Man hielt dafür, ein Kind"),
        fr=("Les pères de cette époque tenaient qu'une fille", " devait obéir, d'abord au père, puis au mari", " devait choisir sa propre voie dans la vie", "On tenait qu'un enfant")),
   dict(en=("In that age, a wife who earned her own living", " was thought a reproach to her husband", " was thought a credit to her husband", "Such a person"),
        de=("In jener Zeit galt eine Frau, die ihr eigenes Brot verdiente,", " als ein Vorwurf für ihren Mann", " als eine Ehre für ihren Mann", "Eine solche Person galt"),
        fr=("En ce temps-là, une épouse qui gagnait sa propre vie", " passait pour un reproche à son mari", " passait pour un honneur à son mari", "Une telle personne passait")),
 ]),
 ("youth", [
   # v2: sharper, Zweig-faithful items (young men aged themselves; grey hair ruled; youth = obstacle).
   dict(en=("In that world it was", " age and grey hairs that commanded respect", " youth and daring that commanded respect", "In some worlds it was"),
        de=("In jener Welt waren es", " das Alter und graue Haare, die Achtung geboten", " die Jugend und die Kühnheit, die Achtung geboten", "In mancher Welt waren es"),
        fr=("Dans ce monde, c'étaient", " l'âge et les cheveux gris qui imposaient le respect", " la jeunesse et l'audace qui imposaient le respect", "Dans certains mondes, c'étaient")),
   dict(en=("The young men of that age", " wore beards and spectacles to seem older", " flaunted their youth as an advantage", "The young men"),
        de=("Die jungen Männer jener Zeit", " trugen Bärte und Brillen, um älter zu scheinen", " stellten ihre Jugend als Vorzug zur Schau", "Die jungen Männer"),
        fr=("Les jeunes gens de cette époque", " portaient barbe et lunettes pour paraître plus vieux", " affichaient leur jeunesse comme un avantage", "Les jeunes gens")),
   dict(en=("In the offices of that age, advancement came", " with grey hairs and long years of waiting", " to talent, however young it might be", "Advancement came"),
        de=("In den Ämtern jener Zeit kam die Beförderung", " mit grauen Haaren und langen Jahren des Wartens", " dem Talent zu, wie jung es auch sein mochte", "Die Beförderung kam"),
        fr=("Dans les bureaux de cette époque, l'avancement venait", " avec les cheveux gris et de longues années d'attente", " au talent, si jeune fût-il", "L'avancement venait")),
   dict(en=("To the men of that age, youth was", " an obstacle, to be hurried through and hidden", " a glory, to be prized and prolonged", "Youth was"),
        de=("Für die Menschen jener Zeit war die Jugend", " ein Hindernis, das man rasch durchlief und verbarg", " ein Ruhm, den man schätzte und verlängerte", "Die Jugend war"),
        fr=("Pour les hommes de cette époque, la jeunesse était", " un obstacle, à franchir vite et à cacher", " une gloire, à priser et à prolonger", "La jeunesse était")),
   dict(en=("In that age, a man of thirty was reckoned", " still too green to be trusted with affairs", " in the very prime of life for great affairs", "A man of thirty was reckoned"),
        de=("In jener Zeit galt ein Mann von dreißig", " noch als zu grün für ernste Geschäfte", " als in der besten Blüte für große Geschäfte", "Ein Mann von dreißig galt"),
        fr=("En ce temps-là, un homme de trente ans passait", " pour trop vert encore pour les affaires", " pour être dans la fleur de l'âge pour les grandes affaires", "Un homme de trente ans passait")),
 ]),
]

# v2 additions to the first two themes (progress, war) — appended to reach 5 items each.
_PROGRESS_EXTRA = [
   dict(en=("The men of that age were sure the coming century", " would heap comfort upon comfort for all", " would bring calamity upon calamity", "The coming century"),
        de=("Die Menschen jener Zeit waren gewiss, das kommende Jahrhundert", " werde Wohlstand auf Wohlstand häufen für alle", " werde Unheil auf Unheil bringen", "Das kommende Jahrhundert"),
        fr=("Les hommes de cette époque étaient sûrs que le siècle à venir", " entasserait bien-être sur bien-être pour tous", " apporterait calamité sur calamité", "Le siècle à venir")),
   dict(en=("To that age, the railways, the telegraph and the electric light", " were proofs of a mankind rising ever higher", " were toys that changed nothing essential in man", "These inventions"),
        de=("Jener Zeit galten die Eisenbahnen, der Telegraph und das elektrische Licht", " als Beweise einer immer höher steigenden Menschheit", " als Spielzeuge, die am Menschen nichts Wesentliches änderten", "Diese Erfindungen galten"),
        fr=("Pour cette époque, les chemins de fer, le télégraphe et la lumière électrique", " étaient les preuves d'une humanité montant toujours plus haut", " étaient des jouets qui ne changeaient rien d'essentiel en l'homme", "Ces inventions étaient")),
]
_WAR_EXTRA = [
   dict(en=("The bankers of that age proved to their satisfaction that a great war", " had become impossible: the credit system forbade it", " was certain to come: no credit system could stop it", "It was proved that such a thing"),
        de=("Die Bankiers jener Zeit bewiesen zu ihrer Zufriedenheit, ein großer Krieg", " sei unmöglich geworden: das Kreditwesen verbiete ihn", " werde gewiss kommen: kein Kreditwesen könne ihn aufhalten", "Man bewies, eine solche Sache"),
        fr=("Les banquiers de cette époque prouvaient à leur satisfaction qu'une grande guerre", " était devenue impossible : le crédit l'interdisait", " viendrait sûrement : nul crédit ne pouvait l'arrêter", "On prouvait qu'une telle chose")),
   dict(en=("The men of that age heard talk of a coming war", " as one hears old wives' tales, smiling", " as one hears a storm warning, with dread", "Such talk was heard"),
        de=("Die Menschen jener Zeit hörten das Gerede von einem kommenden Krieg", " wie man Ammenmärchen hört, lächelnd", " wie man eine Sturmwarnung hört, mit Bangen", "Solches Gerede hörte man"),
        fr=("Les hommes de cette époque écoutaient parler d'une guerre à venir", " comme on écoute des contes de bonne femme, en souriant", " comme on écoute un avis de tempête, avec effroi", "De tels propos s'écoutaient")),
]
THEMES[0][1].extend(_PROGRESS_EXTRA)
THEMES[1][1].extend(_WAR_EXTRA)

MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"]), ("modern", MODERN)]
LANGS = ("en", "de", "fr")


def cal_pref(model, tok, stem, belief, skeptic, neutral):
    """Calibrated preference: c(belief)-c(skeptic), c(a)=slp(stem,a)-slp(neutral,a). >0 => holds the belief."""
    cb = common.slp(model, tok, stem, belief) - common.slp(model, tok, neutral, belief)
    cs = common.slp(model, tok, stem, skeptic) - common.slp(model, tok, neutral, skeptic)
    return cb - cs


items = [(theme, it) for theme, lst in THEMES for it in lst]
scores = {tag: {} for tag, _ in MODELS}
for tag, mid in MODELS:
    print(f"\n=== {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    for lang in LANGS:
        for i, (theme, it) in enumerate(items):
            cp = cal_pref(model, tok, *it[lang])
            scores[tag][f"{lang}/{theme}/{i}"] = {"theme": theme, "lang": lang, "cal_pref": round(cp, 3),
                                                  "holds": int(cp > 0)}
    del tok, model
    common.free()

themes = [t for t, _ in THEMES]
report = {"models": {t: m for t, m in MODELS}, "metric": "calibrated belief-preference (nats)", "scores": scores, "summary": {}}
for tag, _ in MODELS:
    report["summary"][tag] = {}
    for lang in LANGS:
        rows = [v for v in scores[tag].values() if v["lang"] == lang]
        d = {"pct_holds": round(100 * st.mean(r["holds"] for r in rows)), "mean": round(st.mean(r["cal_pref"] for r in rows), 2)}
        for th in themes:
            tr = [r for r in rows if r["theme"] == th]
            d[th] = round(st.mean(r["cal_pref"] for r in tr), 2)
        report["summary"][tag][lang] = d
# Ranke(1913) minus modern per theme/lang -- fluency-controlled worldview signal
report["ranke_minus_modern"] = {lang: {th: round(report["summary"]["1913"][lang][th] - report["summary"]["modern"][lang][th], 2)
                                       for th in themes} for lang in LANGS}
json.dump(report, open(os.path.join(common.RESULTS, "zweig_probe_v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- console ----
LN = {"en": "ENGLISH", "de": "GERMAN", "fr": "FRENCH"}
print("\n" + "=" * 96)
print("ZWEIG WORLDVIEW PROBE (calibrated)  —  cal_pref>0 = the model HOLDS the pre-1914 belief")
print("=" * 96)
for lang in LANGS:
    print(f"\n### {LN[lang]} — calibrated belief-preference (nats) by theme and model")
    print(f"{'theme':18s}" + "".join(f"{tag:>9}" for tag, _ in MODELS))
    for th in themes:
        print(f"{th:18s}" + "".join(f"{report['summary'][tag][lang][th]:>+9.2f}" for tag, _ in MODELS))
    print(f"{'% holds belief':18s}" + "".join(f"{report['summary'][tag][lang]['pct_holds']:>8d}%" for tag, _ in MODELS))
print("\n" + "-" * 70)
print("WAR (unthinkability) across languages (the cross-lingual worldview question):")
print(f"{'lang':6s}" + "".join(f"{tag:>9}" for tag, _ in MODELS) + "   (>0 = 'war was unthinkable' held; <0 = war expected)")
for lang in LANGS:
    print(f"{LN[lang]:6s}" + "".join(f"{report['summary'][tag][lang]['war']:>+9.2f}" for tag, _ in MODELS))
print("\nRanke(1913) - modern  [fluency-controlled: large +ve = time-locked mind holds an illusion the modern mind rejects]:")
for lang in LANGS:
    print(f"  {LN[lang]:8s}: " + "  ".join(f"{th}={report['ranke_minus_modern'][lang][th]:+.1f}" for th in themes))
print("\nwrote zweig_probe_v2.json")
