# -*- coding: utf-8 -*-
"""FORESIGHT ITEM BANK (pure data; no torch). The frozen batteries for the ex-ante foreseeability study.

Each EXPECTATION item is a matched-counterfactual forecast:
  subject  : a LOADED pre-shock prompt (period register; presupposes nothing about the outcome)
  target   : the ex-post-CORRECT continuation (leading space)
  twins    : several length-varied FALSE continuations = the period CONSENSUS / plausible alternatives.
             The scorer keeps the one whose token count == target's (length-matched twin), else nearest.
  neutral  : a minimal, de-loaded frame for the SAME target -> the PMI marginal (lexical base rate).
             calibrated score  c(a) = slp(subject,a) - slp(neutral,a) = -strain(a).
  outcome_set : (binary items) the 2 canonical outcomes [true, consensus-false] for the entropy readout.

Bookkeeping fields (frozen before scoring, used in analysis, NOT by the scorer):
  probed_by        : models for which this is a genuine FORWARD forecast (cutoff < event).
  hindsight_anchor : model that has the outcome in-sample ("" if none exists in the Ranke set).
  event_year       : when the forecast resolved.
  true_valence     : catastrophic | neutral | benign  (must NOT be always catastrophic -> drama-bias guard).
  period_consensus : which side an average contemporary held: "false" (consensus=twin), "true", or "split".
  specificity      : "direction" (a gestalt any pessimist could voice) | "figure" (exact number/date; only
                     post-event text can nail it -> separates genuine inkling from leakage).

Design note on the metric: because targets are CLAUSES, the calibration baseline is the de-loaded `neutral`
frame (bare topic), not score_v2's noun-answer neutrals ("The answer is"). c(a) = -strain(a) then unifies the
forced choice, the PMI decomposition, and the entropy readout on one scale. See horizon_diff.py / entropy_probe.py.

The GATE batteries (post-shock FACTS, forced choice w/ 3 distractors, batteries.py schema) test whether the
PRE model is CLEAN on each event: a low-surprise reading is "foresight" only if the PRE model is at chance here.
"""

RANKE = {"1900": "uzh-echist-org/Ranke-4B-1900",
         "1913": "uzh-echist-org/Ranke-4B-1913",
         "1929": "uzh-echist-org/Ranke-4B-1929"}
MODERN = "Qwen/Qwen2.5-3B"          # hindsight floor for events with no post-event Ranke (Δstrain only).
# NB: Qwen2.5-3B (bf16 ~6GB) fits the 12GB RTX 3060; 7B (~14GB) does not. Any modern model has fully "learned"
# the Depression/WWI, so 3B is a valid hindsight-saturated floor. Qwen2.5-7B kept as an optional CPU robustness
# check. All Qwen2.5 share ONE tokenizer (vocab 151665) which DIFFERS from Ranke/Qwen3 -> used via Δstrain only.
MODERN_XL = "Qwen/Qwen2.5-7B"       # optional larger check (CPU)

# ============================ BATTERY 1: WWI-CHARACTER ============================
# Probed by 1900 & 1913 (forecast); hindsight anchor = 1929 (has 1914-1929 in-sample). Register strictly
# pre-1914: "the powers", "the crowned heads", "before the leaves fall". No post-hoc names in subjects.
WWI = [
 dict(id="wwi_dur_years", facet="duration", event_year=1918, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="Should the great powers of Europe come to blows, the war",
      neutral="The war",
      target=" will drag on for years, and bleed the nations white",
      twins=[" will be over by the coming Christmas", " will be short, and won at a stroke",
             " will be done before the leaves fall", " will be brief, and settled in a season"]),
 dict(id="wwi_dur_home", facet="duration", event_year=1918, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The young men who march away to the next great war",
      neutral="The young men who march away",
      target=" will come home only after long and bitter years",
      twins=[" will come home victorious by that Christmas", " will be home before the autumn is out",
             " will return again in a few short weeks", " will be back at their hearths by the new year"]),
 dict(id="wwi_dur_decided", facet="duration", event_year=1918, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="A general war among the Great Powers",
      neutral="A war",
      target=" will be decided only after years of ruinous stalemate",
      twins=[" will be decided in one great and open battle", " will be settled in a single short campaign",
             " will be over in a few weeks of swift manoeuvre"]),
 dict(id="wwi_dur_economy", facet="duration", event_year=1918, true_valence="catastrophic",
      period_consensus="split", specificity="direction",
      subject="The vast cost of a modern war means that the fighting",
      neutral="The fighting",
      target=" will grind on for years despite the ruin",
      twins=[" will collapse within a very few weeks", " will bankrupt the powers in a month",
             " will end quickly for want of money"]),
 dict(id="wwi_nat_spade", facet="nature", event_year=1916, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="In the next great war among the powers, the soldiers",
      neutral="The soldiers",
      target=" will pass their days dug into ditches in the earth",
      twins=[" will march swiftly onward to a bright victory", " will settle it by bold and open charges",
             " will win their glory in the open field"]),
 dict(id="wwi_nat_deadlock", facet="nature", event_year=1916, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The new quick-firing guns will make the coming war",
      neutral="The coming war will be",
      target=" one of siege and deadlock, neither side able to move",
      twins=[" one of swift movement and quick decision", " a war of bold marches and open battle",
             " a short and glorious rush to victory"]),
 dict(id="wwi_nat_machine", facet="nature", event_year=1916, true_valence="neutral",
      period_consensus="false", specificity="direction",
      subject="The weapon that will most shape the coming war",
      neutral="The chief weapon",
      target=" is the gun in a fixed line, and the spade",
      twins=[" is the lance of the cavalry in a charge", " is the bayonet in a bold assault",
             " is the sabre and the massed column"]),
 dict(id="wwi_cas_millions", facet="casualties", event_year=1918, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="A war among the Great Powers",
      neutral="A war",
      target=" would cost the lives of many millions of young men",
      twins=[" would cost the lives of some thousands of men", " would cost a great many thousand brave men",
             " would take tens of thousands of men at the most"]),
 dict(id="wwi_cas_figure", facet="casualties", event_year=1918, true_valence="catastrophic",
      period_consensus="false", specificity="figure",
      subject="The soldiers killed in the coming war will, in the end, number about",
      neutral="The number killed will be about",
      target=" nine or ten millions of men",
      twins=[" two hundred thousand of men", " some half a million men", " a hundred thousand men"]),
 dict(id="wwi_cas_civilians", facet="casualties", event_year=1918, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The coming war will fall",
      neutral="The war will fall",
      target=" upon whole peoples, and not on armies alone",
      twins=[" upon the soldiers alone, as wars have", " but lightly on those left at home",
             " only upon the fields of battle"]),
 dict(id="wwi_con_thrones", facet="consequence", event_year=1918, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="When the great war at last comes to its end, the old crowned heads of Europe",
      neutral="The old crowned heads of Europe",
      target=" will have fallen, one throne after another",
      twins=[" will stand yet firmer than before", " will reign on as they always have",
             " will be restored to a fuller glory"]),
 dict(id="wwi_con_empires", facet="consequence", event_year=1918, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="By the close of the war, the great land empires of the east",
      neutral="The great land empires of the east",
      target=" will have broken into many pieces",
      twins=[" will have gathered new lands and power", " will stand whole and quite unshaken",
             " will divide fresh spoils among them"]),
 dict(id="wwi_con_russia", facet="consequence", event_year=1917, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="Should this war come, the mighty empire of Russia",
      neutral="The empire of Russia",
      target=" will be swept away by revolution",
      twins=[" will emerge stronger than ever before", " will win the prize it has long sought",
             " will hold quite firm through every storm"]),
 dict(id="wwi_con_map", facet="consequence", event_year=1919, true_valence="neutral",
      period_consensus="false", specificity="direction",
      subject="The map of Europe, once the war is over,",
      neutral="The map of Europe",
      target=" will be redrawn, its old empires gone",
      twins=[" will stand much as it does today", " will be little changed by the fighting",
             " will be restored to its former shape"]),
 dict(id="wwi_int_angell", facet="integrity", event_year=1918, true_valence="neutral",
      period_consensus="split", specificity="direction",
      subject="Though trade and credit now bind the nations tightly together, a great war",
      neutral="A great war",
      target=" will come all the same, and grind on for years",
      twins=[" will prove impossible, and be soon avoided", " will ruin all so fast it must be brief",
             " will be called off before it is begun"]),
 dict(id="wwi_int_offensive", facet="integrity", event_year=1916, true_valence="neutral",
      period_consensus="true", specificity="direction",
      subject="The generals who trust that a bold attack will swiftly win the war",
      neutral="The generals who trust in the attack",
      target=" are gravely mistaken; the defence will hold",
      twins=[" are very likely to be proven quite right", " will surely be home before the winter",
             " understand the true ways of modern war"]),
 dict(id="wwi_int_naval", facet="integrity", event_year=1916, true_valence="benign",
      period_consensus="true", specificity="direction",
      subject="The great battle-fleets the powers have built at such cost will, in the war,",
      neutral="The great battle-fleets",
      target=" prove strangely idle, kept back in port",
      twins=[" swiftly decide the whole of the war", " win a great and glorious sea-battle",
             " master all the seas in a quick blockade"]),
 dict(id="wwi_dur_short_illusion", facet="duration", event_year=1914, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The soldiers who set out in the first days, sure of a quick and easy triumph,",
      neutral="The soldiers who set out",
      target=" are marching into four years of slaughter",
      twins=[" will be home in a month or two at most", " shall see the whole thing finished by autumn",
             " will win it before the winter snows come"]),
 dict(id="wwi_con_america", facet="consequence", event_year=1917, true_valence="neutral",
      period_consensus="false", specificity="direction",
      subject="Before the great war is over, the far republic of America",
      neutral="The republic of America",
      target=" will be drawn across the sea into the fighting",
      twins=[" will keep well clear of Europe's quarrel", " will stay at peace behind its oceans",
             " will trade with all and fight with none"]),
 dict(id="wwi_int_wounded", facet="casualties", event_year=1918, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="Besides the dead, the coming war will leave behind",
      neutral="The war will leave behind",
      target=" whole nations of maimed and broken men",
      twins=[" a modest tally of wounded soldiers", " but few beyond the ordinary hurt",
             " the usual butcher's bill of a battle"]),
]

# ============================ BATTERY 2: DEPRESSION-SEVERITY ============================
# Probed by 1929 (primary), 1900 & 1913 (deep horizon). NO post-event Ranke -> hindsight floor = Qwen (Δstrain)
# + the 1900->1913->1929 gradient. Prompts phrased as FORECASTS ("the years ahead") so they hold whether or
# not the Oct-1929 crash sits inside the 1929 corpus. Fisher's "permanently high plateau" is a false twin.
DEP = [
 dict(id="dep_dur_slump", facet="duration", event_year=1932, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="After the recent break in the share market, the trade of the nation",
      neutral="The trade of the nation",
      target=" will sink into a slump that lasts for years",
      twins=[" will recover briskly within the year", " will soon climb back to its old height",
             " will rebound well before the summer is out"]),
 dict(id="dep_dur_decade", facet="duration", event_year=1939, true_valence="catastrophic",
      period_consensus="false", specificity="figure",
      subject="Business will not climb back to the height of this year for",
      neutral="Business will recover in",
      target=" the better part of ten long years",
      twins=[" a mere handful of quiet months", " a single year or two at most",
             " no more than one poor season"]),
 dict(id="dep_dur_prosperity", facet="duration", event_year=1936, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="Prosperity, after this present setback,",
      neutral="Prosperity",
      target=" will return only after long and bitter years",
      twins=[" will return within a very few months", " will come back almost at once, as before",
             " will arrive with the next season's trade"]),
 dict(id="dep_sev_fourth", facet="severity", event_year=1933, true_valence="catastrophic",
      period_consensus="false", specificity="figure",
      subject="In the years now beginning, the men out of work",   # 'hard' removed per blind screen (pre-scoring cue fix)
      neutral="The men out of work",
      target=" will swell until a fourth of all who labour stand idle",
      twins=[" will rise but a little, and then fall again", " will scarcely alter from what they are",
             " will trouble but a single trade or two"]),
 dict(id="dep_sev_millions", facet="severity", event_year=1933, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The coming slump will throw out of their work",
      neutral="The slump will idle",
      target=" many millions of willing men",
      twins=[" a modest number of idle hands", " but few beyond the usual poor",
             " only the unfit and the idle few"]),
 dict(id="dep_sev_family", facet="severity", event_year=1933, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="For the ordinary working family, the years just ahead hold",
      neutral="The years ahead hold",
      target=" want, and hunger, and long idleness",
      twins=[" steady wages and their daily bread", " a modest but unbroken comfort",
             " little change at all from the present"]),
 dict(id="dep_opt_plateau", facet="optimism", event_year=1932, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The learned economists judge that share prices have now reached",
      neutral="Share prices have reached",
      target=" a peak from which they will presently crash",
      twins=[" a high and a permanent plateau", " a sound and a lasting new level",
             " firm ground for a further great rise"]),
 dict(id="dep_opt_boom", facet="optimism", event_year=1932, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The great boom of these past years will, in the end,",
      neutral="The great boom will",
      target=" break in the worst crash the world has known",
      twins=[" carry on for many years to come", " give way to a gentle, steady growth",
             " settle into a broad and lasting wealth"]),
 dict(id="dep_bank_fail", facet="banks", event_year=1933, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The banks of the nation, in the coming years,",
      neutral="The banks of the nation",
      target=" will fail by the thousand",
      twins=[" will stand quite sound and secure", " will weather the storm with ease",
             " will prove their great strength anew"]),
 dict(id="dep_bank_savings", facet="banks", event_year=1933, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="Those who trust their savings to the banks will, in these years,",
      neutral="The savings in the banks will",
      target=" be lost in a great wave of failures",
      twins=[" be perfectly safe throughout it all", " be well rewarded for their trust",
             " grow larger with each passing year"]),
 dict(id="dep_gold_off", facet="institutions", event_year=1933, true_valence="neutral",
      period_consensus="false", specificity="direction",
      subject="The gold standard, that firm anchor of all sound money,",
      neutral="The gold standard",
      target=" will be cast off by one nation after another",
      twins=[" will hold quite firm against every storm", " will remain the world's one sure anchor",
             " will be defended by all at any cost"]),
 dict(id="dep_trade_walls", facet="institutions", event_year=1932, true_valence="neutral",
      period_consensus="false", specificity="direction",
      subject="The free flow of goods between the nations will, in the slump,",
      neutral="The trade between nations will",
      target=" be choked by new walls of tariff",
      twins=[" continue on much as it does today", " grow yet freer as the prices fall",
             " bind the peoples closer still"]),
 dict(id="dep_realwage_up", facet="benign", event_year=1933, true_valence="benign",
      period_consensus="split", specificity="direction",
      subject="For the man who keeps his place, the falling prices will mean his wages",
      neutral="His wages will",
      target=" buy rather more than they did before",
      twins=[" be cut away to almost nothing at all", " buy far less with every passing year",
             " vanish along with all his savings"]),
 dict(id="dep_pol_govts", facet="politics", event_year=1933, true_valence="catastrophic",
      period_consensus="split", specificity="direction",
      subject="Hard times on so great a scale will, before they end,",
      neutral="The hard times will",
      target=" shake the governments of the world",
      twins=[" pass without any great disturbance", " leave the old order quite untouched",
             " be borne by all in patient quiet"]),
 dict(id="dep_pol_radical", facet="politics", event_year=1933, true_valence="catastrophic",
      period_consensus="split", specificity="direction",
      subject="Men left long without work or bread will, in the end,",
      neutral="Idle men will",
      target=" turn to angry and to radical leaders",
      twins=[" wait in patience for the better days", " trust the old parties to set it right",
             " bear their hard lot without complaint"]),
 dict(id="dep_intl_contagion", facet="severity", event_year=1932, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The slump that has begun in America will",
      neutral="The slump will",
      target=" spread to every land and darken the whole world",
      twins=[" stay within America's own broad borders", " spare the older nations across the sea",
             " trouble that one country all alone"]),
]

# ============================ BATTERY 3: WEIMAR / 1933 (secondary, WEAKEST) ============================
# Probed by 1929 (4 years out, compound counterfactual: needs the Depression AND its political effect).
# NO post-event Ranke. Separate the FORESEEABLE (fragility, Article-48 rule) from the NOT-consensus
# (one party by 1933). Headline on the broad, robust item (weim_central_europe). Register avoids Nazi/Weimar.
WEIM = [
 dict(id="weim_oneparty", facet="regime", event_year=1933, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The German republic, in the years now just ahead,",
      neutral="The German republic",
      target=" will fall to a single ruling party",
      twins=[" will grow into a settled democracy", " will endure long as a free republic",
             " will find its feet at the long last"]),
 dict(id="weim_oneman", facet="regime", event_year=1933, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="Within a very few years, Germany will be governed by",
      neutral="Germany will be governed by",
      target=" one man and one party alone",
      twins=[" a broad and a lawful coalition", " its parliament, much as it is now",
             " the moderate men of the centre"]),
 dict(id="weim_fringe", facet="regime", event_year=1933, true_valence="catastrophic",
      period_consensus="false", specificity="figure",
      subject="The small party of extreme nationalists will, in a few years,",
      neutral="The extreme nationalists will",
      target=" seize the whole of the German state",
      twins=[" stay small and shut out of all power", " fade away as prosperity returns",
             " win only a modest and lawful share"]),
 dict(id="weim_press", facet="liberty", event_year=1933, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="Freedom of the press and of assembly in Germany will, in these years,",
      neutral="The freedoms of Germany will",
      target=" be crushed within a very few years",
      twins=[" be widened and made yet more secure", " continue on much as they are now",
             " grow stronger with every passing year"]),
 dict(id="weim_article48", facet="liberty", event_year=1933, true_valence="catastrophic",
      period_consensus="split", specificity="direction",
      subject="The republic's power to rule by emergency decree will, in the end,",
      neutral="The emergency power will",
      target=" become the very tool of one-man rule",
      twins=[" be used but sparingly and lawfully", " safeguard the republic in its crises",
             " fall away again as the calm returns"]),
 dict(id="weim_central_europe", facet="regime", event_year=1938, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="Free government across central Europe, over the next ten years,",
      neutral="Free government in Europe",
      target=" will give way to strongmen and to dictators",
      twins=[" will take a firm and a lasting root", " will spread outward from Germany itself",
             " will hold on right through the hard times"]),
 dict(id="weim_kaiser", facet="benign", event_year=1933, true_valence="benign",
      period_consensus="split", specificity="direction",
      subject="The Kaiser who was driven from his throne in 1918",
      neutral="The old Kaiser",
      target=" will never sit upon it again",
      twins=[" will be called back to rule at the last", " will return amid the coming turmoil",
             " will regain his lost crown in the end"]),
 dict(id="weim_violence", facet="liberty", event_year=1933, true_valence="catastrophic",
      period_consensus="split", specificity="direction",
      subject="The fighting between armed bands in the German streets will, in a few years,",
      neutral="The street fighting will",
      target=" end in the victory of one side alone",
      twins=[" die down again as the tempers cool", " be checked at last by a firm police",
             " give way in time to lawful debate"]),
 dict(id="weim_persecution", facet="liberty", event_year=1935, true_valence="catastrophic",
      period_consensus="false", specificity="direction",
      subject="The Jews of Germany, in the years now just ahead,",
      neutral="The Jews of Germany",
      target=" will be stripped of their rights and driven out",
      twins=[" will live secure as equal citizens", " will prosper much as they long have",
             " will see their place grow ever stronger"]),
 dict(id="weim_elites", facet="regime", event_year=1933, true_valence="catastrophic",
      period_consensus="split", specificity="direction",
      subject="The old army chiefs and the men of property will, in a crisis,",
      neutral="The old elites will",
      target=" hand the power over to a demagogue",
      twins=[" stand firm by the lawful republic", " call the old Kaiser home once again",
             " hold all the parties in a balance"]),
]

# ============================ BATTERY 4: CONTROLS (metric floor & ceiling) ============================
# Tier B: period-normal boring forecasts -> should score EASY/CORRECT, LOW strain for all models (lower anchor).
# b_kings_neutral is the MIRROR of wwi_con_thrones: in the NEUTRAL frame the true answer is "reign on", so a
# collapse-lean that appears only in the war frame is frame-conditional, not a blanket doom-bias.
CTRL_B = [
 dict(id="b_winter", facet="tierB", event_year=0, true_valence="neutral", period_consensus="true",
      specificity="direction", subject="When the winter comes, the days", neutral="The days",
      target=" will grow short and cold", twins=[" will grow long and warm", " will turn bright and hot",
             " will stay mild and green"]),
 dict(id="b_sun", facet="tierB", event_year=0, true_valence="neutral", period_consensus="true",
      specificity="direction", subject="Tomorrow at dawn the sun", neutral="The sun",
      target=" will rise in the east, as ever", twins=[" will rise in the west, for once",
             " will rise in the north this time", " will climb the south by noon"]),
 dict(id="b_harvest", facet="tierB", event_year=0, true_valence="neutral", period_consensus="true",
      specificity="direction", subject="Next year's harvest will depend, as always,", neutral="The harvest depends",
      target=" upon the sun and the rain", twins=[" upon the whim of the machines",
             " upon the vote of the crowd", " upon the price of the gold"]),
 dict(id="b_ocean", facet="tierB", event_year=0, true_valence="neutral", period_consensus="true",
      specificity="direction", subject="A ship crossing the wide ocean", neutral="A ship",
      target=" will take many days upon the sea", twins=[" will take but an hour aloft in the air",
             " will cross in one short single morning", " will need but a moment to pass over"]),
 dict(id="b_motorcar", facet="tierB", event_year=1929, true_valence="neutral", period_consensus="true",
      specificity="direction", subject="In the years ahead the motor-car upon our roads", neutral="The motor-car",
      target=" will grow more common year by year", twins=[" will grow rare and then quite vanish",
             " will be banned from every public road", " will fall behind the horse once more"]),
 dict(id="b_kings_neutral", facet="tierB_mirror", event_year=0, true_valence="neutral", period_consensus="true",
      specificity="direction", subject="In ordinary and settled times, the kings of Europe",
      neutral="The kings of Europe", target=" will reign on over their own lands",
      twins=[" will vanish all in a single night", " will be gone by the next springtime",
             " will flee from their thrones at once"]),
]
# Tier C: future to ALL three models (even 1929) -> should score MAX surprise for all (upper anchor). If the
# 1929 model finds any of these UN-surprising, that is a leak tripwire. Single-token targets -> forced choice.
CTRL_C = [
 dict(id="c_moon", facet="tierC", event_year=1969, true_valence="neutral", period_consensus="false",
      specificity="figure", subject="In time to come, men will walk upon the face of the",
      neutral="Men will reach the", target=" moon", twins=[" sun", " sea", " sky"]),
 dict(id="c_bomb", facet="tierC", event_year=1945, true_valence="catastrophic", period_consensus="false",
      specificity="figure", subject="In time to come, a single bomb will destroy a whole",
      neutral="A bomb will destroy a", target=" city", twins=[" ship", " bridge", " tower"]),
 dict(id="c_heart", facet="tierC", event_year=1967, true_valence="neutral", period_consensus="false",
      specificity="figure", subject="In time to come, the doctors will place in a man the living heart of",
      neutral="The doctors will use the heart of", target=" another man", twins=[" a great ox", " a fine horse", " a strong hound"]),
 dict(id="c_ether", facet="tierC", event_year=1936, true_valence="neutral", period_consensus="false",
      specificity="figure", subject="In time to come, voices and pictures will fly through the air into every",
      neutral="Voices will reach every", target=" home", twins=[" church", " palace", " harbour"]),
 dict(id="c_engine", facet="tierC", event_year=1997, true_valence="neutral", period_consensus="false",
      specificity="figure", subject="In time to come, a machine will beat the finest masters at the game of",
      neutral="A machine will master the game of", target=" chess", twins=[" cards", " dice", " billiards"]),
]

# battery -> which cutoffs treat it as a FORWARD forecast (probed_by) and the hindsight anchor
BATTERY_META = {
 "wwi":  dict(items=WWI,  probed_by=["1900", "1913"], hindsight_anchor="1929"),
 "dep":  dict(items=DEP,  probed_by=["1900", "1913", "1929"], hindsight_anchor=""),   # Qwen via Δstrain
 "weim": dict(items=WEIM, probed_by=["1929"], hindsight_anchor=""),                    # Qwen via Δstrain
 "ctrlB": dict(items=CTRL_B, probed_by=["1900", "1913", "1929"], hindsight_anchor=""),
 "ctrlC": dict(items=CTRL_C, probed_by=["1900", "1913", "1929"], hindsight_anchor="1929"),  # anchor still surprised
}


def all_items():
    """Yield every expectation item in the unified schema (with battery + probed_by + anchor attached)."""
    out = []
    for bat, meta in BATTERY_META.items():
        for it in meta["items"]:
            r = dict(it)
            r["battery"] = bat
            r["probed_by"] = meta["probed_by"]
            r["hindsight_anchor"] = meta["hindsight_anchor"]
            r.setdefault("outcome_set", [it["target"], it["twins"][0]])   # [true, consensus-false]
            out.append(r)
    return out


# ============================ GATE BATTERIES (post-shock FACTS; batteries.py schema) ============================
# Forced choice, 1 true + 3 matched distractors (chance 25%). Tests whether the PRE model is CLEAN on the event:
# a low-surprise foresight reading is trustworthy only if the PRE model sits at chance here. Every fact requires
# knowledge dated strictly AFTER the PRE model's cutoff. (For WWI, PRE=1913 -> all 1914+; for DEP/WEIM, PRE=1929
# -> all 1930+.) A few crash-boundary items (dated Oct-1929) empirically probe what the 1929 corpus contains.
GATE = {
 "wwi": {   # PRE model under test = Ranke-1913 (cutoff < 1914). All items are 1914-1919.
  "pre_model": "uzh-echist-org/Ranke-4B-1913",
  "facts": [
   (1916, "Verdun", "The longest and bloodiest battle of the war, at a French fortress in 1916, was", " Verdun", [" Sedan", " Cambrai", " Metz"]),
   (1916, "Somme", "The great British offensive of 1916, with sixty thousand casualties on its first day, was the", " Somme", [" Marne", " Aisne", " Yser"]),
   (1915, "Gallipoli", "The failed 1915 Allied landing against the Turks was at", " Gallipoli", [" Salonika", " Smyrna", " Beirut"]),
   (1916, "Jutland", "The one great fleet battle of the war, fought in 1916 in the North Sea, was", " Jutland", [" Heligoland", " Dogger Bank", " Coronel"]),
   (1917, "US entered", "The great American republic entered the war against Germany in the year", " 1917", [" 1915", " 1919", " 1914"]),
   (1918, "Armistice year", "The great war finally ended with an armistice in the year", " 1918", [" 1916", " 1921", " 1924"]),
   (1919, "Versailles", "The 1919 treaty that formally ended the war was signed at the palace of", " Versailles", [" Vienna", " Potsdam", " Trianon"]),
   (1916, "tank", "The new armoured machine first sent into battle by the British in 1916 was the", " tank", [" turret", " lorry", " ram"]),
   (1915, "gas", "The new weapon first loosed as a green cloud at Ypres in 1915 was poison", " gas", [" fire", " oil", " smoke"]),
   (1917, "Passchendaele", "The 1917 offensive that drowned in the mud of Flanders was", " Passchendaele", [" Loos", " Arras", " Festubert"]),
   (1915, "Lusitania", "The great liner sunk by a German submarine in 1915 was the", " Lusitania", [" Aquitania", " Britannic", " Caledonia"]),
   (1918, "Kaiser fled", "At the war's end in 1918 the German Kaiser fled into exile in", " Holland", [" Sweden", " Denmark", " Spain"]),
  ]},
 "dep": {   # PRE model under test = Ranke-1929. All items are 1930+ (+ two Oct-1929 boundary probes).
  "pre_model": "uzh-echist-org/Ranke-4B-1929",
  "facts": [
   (1933, "unemployment peak", "At the very worst of the slump, in 1933, the share of American workers without work reached about", " a quarter", [" a fiftieth", " a twentieth", " a half"]),
   (1931, "sterling off gold", "The nation that abandoned the gold standard in 1931, shaking world finance, was", " Britain", [" France", " America", " Italy"]),
   (1930, "tariff", "The steep 1930 American tariff that deepened the trade collapse bore the names Smoot and", " Hawley", [" Hoover", " Mellon", " Glass"]),
   (1933, "bank holiday", "Soon after taking office in 1933, the new American president shut every bank in a national bank", " holiday", [" merger", " census", " audit"]),
   (1932, "Dow bottom", "The American share index fell, by 1932, to barely a", " tenth of its peak", [" half of its peak", " third of its peak", " quarter of its peak"]),
   (1933, "New Deal", "The programme of recovery begun by the American president in 1933 was called the New", " Deal", [" Order", " Dawn", " Way"]),
   (1931, "Credit-Anstalt", "The great Austrian bank whose 1931 collapse spread panic across Europe was the Credit-", " Anstalt", [" Verein", " Gesell", " Kasse"]),
   (1936, "recovery slow", "American output did not climb back to its 1929 height until nearly the year", " 1937", [" 1931", " 1933", " 1935"]),
   # --- Oct-1929 boundary probes (empirically test what the 1929 corpus contains) ---
   (1929, "Black Thursday", "The Wall Street panic of late October 1929 began on a day remembered as Black", " Thursday", [" Friday", " Monday", " Sunday"]),
   (1929, "crash month", "The great crash of the American stock market came in the month of", " October", [" March", " July", " January"]),
  ]},
 "weim": {  # PRE model under test = Ranke-1929. All items are 1930+.
  "pre_model": "uzh-echist-org/Ranke-4B-1929",
  "facts": [
   (1933, "chancellor 1933", "The leader of the extreme nationalists who became German chancellor in 1933 was Adolf", " Hitler", [" Hugenberg", " Papen", " Strasser"]),
   (1933, "Reichstag fire", "The 1933 fire used as a pretext to seize emergency powers destroyed the German", " Reichstag", [" Rathaus", " Schloss", " Dom"]),
   (1932, "largest party", "By 1932 the largest single party in the German parliament was the National", " Socialists", [" Liberals", " Catholics", " Monarchists"]),
   (1933, "Enabling Act", "The 1933 law that let the chancellor rule without parliament was the Enabling", " Act", [" Charter", " Decree", " Pact"]),
   (1934, "one leader", "After the old president died in 1934, the chancellor merged both offices and took the title of", " Leader", [" Regent", " Consul", " Tribune"]),
   (1935, "citizenship laws", "The 1935 laws that stripped German Jews of their citizenship were named for the city of", " Nuremberg", [" Potsdam", " Weimar", " Munich"]),
   (1933, "camps", "The first camps for political prisoners opened by the new regime in 1933 were the concentration", " camps", [" prisons", " barracks", " colonies"]),
   (1936, "one party", "By the mid-1930s the number of political parties permitted in Germany had fallen to", " one", [" three", " five", " none"]),
  ]},
}


def gate_items():
    """Yield gate facts in the batteries.py forced-choice schema, tagged with event + pre_model + year."""
    out = []
    for event, g in GATE.items():
        for (year, lab, prompt, true, distractors) in g["facts"]:
            out.append({"event": event, "pre_model": g["pre_model"], "year": year,
                        "label": lab, "prompt": prompt, "true": true, "distractors": distractors})
    return out


# ============================ PRODUCTION (recognition vs generation) subset ============================
# Open-ended free_prompt (no cue). Greedy + sampled output coded by markers. Production is the HIGHER bar:
# a model may recognise the true twin under teacher-forcing yet never volunteer it. Only a production hit
# supports a strong "inkling" claim.
PRODUCTION = [
 dict(id="wwi_dur_years", free_prompt="Should a great war break out among the powers of Europe, it will most likely last",
      true_markers=["year", "years", "long", "attrition", "deadlock", "exhaust", "bitter"],
      false_markers=["christmas", "month", "week", "season", "swift", "short", "decisive", "soon"]),
 dict(id="wwi_nat_spade", free_prompt="When the next great European war is fought, the soldiers will spend their days",
      true_markers=["dig", "dug", "ditch", "trench", "entrench", "hold", "wait", "earth"],
      false_markers=["march", "charge", "horse", "advance", "manoeuvre", "gallop", "open field"]),
 dict(id="wwi_cas_millions", free_prompt="A great war among the powers of Europe would cost the lives of",
      true_markers=["million", "millions"],
      false_markers=["thousand", "hundred", "few", "score"]),
 dict(id="wwi_con_thrones", free_prompt="When the great war among the powers is over, the old thrones of Europe will",
      true_markers=["fall", "fallen", "gone", "sweep", "swept", "revolution", "crumble", "topple"],
      false_markers=["stand", "firm", "secure", "remain", "restore", "endure"]),
 dict(id="dep_dur_slump", free_prompt="After the recent break in share prices, the years ahead for business will be",
      true_markers=["depress", "slump", "long", "hard", "years", "lean", "bad", "worse"],
      false_markers=["recover", "brief", "soon", "rebound", "prosper", "plateau", "bright"]),
 dict(id="dep_sev_fourth", free_prompt="In the hard years ahead, the number of men out of work will",
      true_markers=["million", "quarter", "fourth", "vast", "great", "mass", "swell", "rise sharply"],
      false_markers=["little", "few", "modest", "scarcely", "small", "fall"]),
 dict(id="dep_bank_fail", free_prompt="The nation's banks, in the coming years, will",
      true_markers=["fail", "fall", "close", "collapse", "run", "break"],
      false_markers=["sound", "safe", "secure", "strong", "weather", "stand"]),
 dict(id="weim_central_europe", free_prompt="Free government across central Europe, over the next ten years, will",
      true_markers=["dictator", "strongman", "strong man", "fall", "crush", "one party", "tyrann", "seize"],
      false_markers=["root", "endure", "spread", "flourish", "hold", "secure", "thrive"]),
]


if __name__ == "__main__":
    items = all_items()
    gates = gate_items()
    from collections import Counter
    bats = Counter(r["battery"] for r in items)
    val = Counter(r["true_valence"] for r in items)
    print("EXPECTATION ITEMS:", len(items), dict(bats))
    print("  valence balance :", dict(val))
    print("  figure items    :", sum(1 for r in items if r["specificity"] == "figure"))
    print("  consensus=true (consensus on FALSE side):", sum(1 for r in items if r["period_consensus"] == "true"))
    print("GATE FACTS      :", len(gates), dict(Counter(g["event"] for g in gates)))
    print("PRODUCTION items:", len(PRODUCTION))
