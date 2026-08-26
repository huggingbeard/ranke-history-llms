# -*- coding: utf-8 -*-
"""PARAPHRASE-ENSEMBLE PILOT. Three pairs, 20 paraphrases each, English.
Each pair = (familiar to a 1900 reader) vs (novel to a 1900 reader):
    motorcars vs airplanes | cavalry charges vs poison gas | state default vs hyperinflation
Paraphrases within a pair share sentence templates where possible, so wording noise partly cancels.
Readouts per concept, averaged over 20 paraphrases:
    (1) plain surprise of the 1900 model (per token)
    (2) the old estimator: surprise(1900) - surprise(1929), per token, same string
Pair verdicts = novel minus familiar, bootstrap CI over paraphrases.
If the novel side does not clearly beat the familiar side, the surprise measure is dead.
Run: python src/diag_paraphrase_pilot.py
"""
import os, json, random, statistics as st
import common
from foresight_bank import RANKE

random.seed(20260704)

BANK = {
"airplanes": [
 "One day men will fly through the air in machines of their own making.",
 "The day is coming when a man will rise from the ground and travel the sky.",
 "Men will soon journey through the air as the birds do.",
 "Before long, flying machines will carry passengers above the clouds.",
 "In years to come, men will cross whole countries through the air.",
 "The heavens will one day carry travellers in winged machines.",
 "A man will board a machine in London and alight from the sky in Paris.",
 "Soon the mails will be carried through the air by flying machines.",
 "Men will build wings that bear them over mountains and seas.",
 "The next generation will think nothing of a voyage through the sky.",
 "Machines heavier than the air will one day rise and fly.",
 "In the coming age, the sky will have its ships as the sea has hers.",
 "Men will one day look down on the clouds from a flying carriage.",
 "It will soon be common for a man to fly a hundred miles in a morning.",
 "The air will become a highway, and men will travel it in machines.",
 "One day soldiers will watch their enemies from machines high in the air.",
 "Flying machines will one day race one another above the housetops.",
 "A day will come when great cities are joined by lines of aerial travel.",
 "Men will learn to steer through the winds as sailors steer through waves.",
 "In a few years, men will mount into the air and go where they please.",
],
"motorcars": [
 "One day the streets will be full of carriages that move without horses.",
 "The day is coming when the horse will vanish from the city streets.",
 "Motor carriages will soon carry men about the town.",
 "Before long, every man of means will keep a motor car.",
 "In years to come, the roads will belong to the motor car.",
 "The horseless carriage will one day be as common as the bicycle.",
 "A man will drive his own carriage by an engine and not a horse.",
 "Soon goods will be hauled through the streets by motor wagons.",
 "Men will travel the country roads in swift motor carriages.",
 "The next generation will think nothing of a drive of fifty miles.",
 "Carriages driven by engines will one day fill our avenues.",
 "In the coming age, the road will hum with machines instead of hooves.",
 "Men will one day cross whole counties in an afternoon by motor.",
 "It will soon be common to ride behind an engine instead of a horse.",
 "The highway will become the home of the motor carriage.",
 "One day doctors will hasten to the sick in swift motor cars.",
 "Motor carriages will one day race one another on the open road.",
 "A day will come when great cities are joined by roads full of motors.",
 "Men will learn to drive their engines as coachmen once drove their teams.",
 "In a few years, men will step into a motor and go where they please.",
],
"poison_gas": [
 "In the wars to come, soldiers will be slain by clouds of poison.",
 "One day armies will loose choking vapours upon their enemies.",
 "The day is coming when battles are decided by poisoned air.",
 "Soldiers will one day wear masks against the deadly vapours of battle.",
 "Before long, the wind itself will be made a weapon of war.",
 "In future wars, men will die gasping in clouds of poisonous gas.",
 "Armies will send drifting clouds of poison across the field.",
 "A day will come when the air of the battlefield kills more surely than the bullet.",
 "In the next great war, poison vapour will creep into the trenches.",
 "Men will fight beneath masks, breathing through machines, amid poisoned air.",
 "Chemists will one day brew the weapons that win battles.",
 "The next war will be fought with poisons loosed upon the wind.",
 "Soldiers will one day fear the changing of the wind more than the guns.",
 "Whole regiments will one day be choked by gases their enemies release.",
 "In the coming wars, the laboratory will arm the soldier with poison.",
 "One day a green cloud will roll over the trenches and kill all it touches.",
 "Battles will one day be won by those who poison the very air.",
 "In future campaigns, gas will blind and choke the armies of both sides.",
 "The day is near when war is waged with vapours instead of steel.",
 "Men will one day bottle death itself and loose it on the battlefield.",
],
"cavalry": [
 "In the wars to come, the cavalry will charge with sabre and lance.",
 "One day soon our horsemen will again decide a great battle.",
 "The day is coming when massed squadrons of horse sweep the field once more.",
 "Soldiers will one day ride down their enemies as their fathers did.",
 "Before long, the cavalry will prove again that the horse rules the battlefield.",
 "In future wars, the charge of horsemen will break the enemy's line.",
 "Armies will send their squadrons of horse thundering across the field.",
 "A day will come when the lance and the sabre settle a great war.",
 "In the next great war, the cavalry will ride through the enemy's guns.",
 "Men will fight on horseback, knee to knee, as soldiers always have.",
 "Horsemen will one day carry the day where the infantry cannot.",
 "The next war will be won by the boldness of the mounted arm.",
 "Soldiers will one day hear again the trumpets of a great charge.",
 "Whole brigades of horse will one day sweep the enemy from the field.",
 "In the coming wars, the horse will bear the soldier to victory as ever.",
 "One day a line of lancers will break an army as in the old wars.",
 "Battles will one day be won by those who ride hardest.",
 "In future campaigns, the cavalry will screen, scout and charge as always.",
 "The day is near when the sabre once more decides the fate of nations.",
 "Men will one day mount and charge, and the enemy will scatter before them.",
],
"state_default": [
 "One day soon some great state will fail to pay its debts.",
 "The day is coming when a proud government repudiates its bonds.",
 "Before long, another treasury will suspend payment on its loans.",
 "In years to come, some kingdom will confess itself bankrupt.",
 "A great power will one day leave its creditors unpaid.",
 "The bondholders will one day whistle for their money.",
 "Some government will soon fail its lenders as governments have before.",
 "A day will come when a nation's bonds sell for a song.",
 "In the next crisis, some state will stop the interest on its debt.",
 "Men will one day read that a great nation has gone bankrupt.",
 "Another government will one day beg its creditors for delay.",
 "The next decade will see some exchequer fail in its obligations.",
 "Some proud state will one day settle with its lenders at a few shillings in the pound.",
 "Before long a finance minister will announce that the treasury is empty.",
 "In the coming years, a great borrower among the nations will default.",
 "One day the rentiers will learn their government cannot pay.",
 "A kingdom's credit will one day collapse in a week.",
 "In future crises, states will fail like merchants, owing more than they hold.",
 "The day is near when some nation's debt is written off as lost.",
 "Men will one day see a mighty empire unable to meet its coupons.",
],
"hyperinflation": [
 "One day a great nation's money will lose its worth entirely.",
 "The day is coming when a loaf of bread costs a basketful of banknotes.",
 "Before long, men will paper their walls with worthless banknotes.",
 "In years to come, a workman's wages will buy less each day than the day before.",
 "A great currency will one day die in the hands of those who hold it.",
 "The savings of a lifetime will one day not buy a week's food.",
 "Some nation will soon print money until it is worth nothing at all.",
 "A day will come when prices double between morning and evening.",
 "In the next age, men will carry their wages home in wheelbarrows.",
 "Men will one day burn banknotes for warmth because coal costs more.",
 "A million of a great nation's money will one day buy a single egg.",
 "The next generation will see a currency worth less than the paper it is printed on.",
 "Some proud nation's money will one day be refused by its own shopkeepers.",
 "Before long a pension that kept a family will not buy a postage stamp.",
 "In the coming years, prices in some land will rise a thousandfold in a season.",
 "One day the thrifty will be beggared and the debtors set free by dying money.",
 "A wage will one day be spent within the hour, before it rots away.",
 "In future crises, whole peoples will barter goods because money has failed.",
 "The day is near when a nation reckons its prices in thousands of millions.",
 "Men will one day sweep banknotes from the gutters, too worthless to steal.",
],
}
PAIRS = [("airplanes", "motorcars"), ("poison_gas", "cavalry"), ("hyperinflation", "state_default")]
MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"])]

res = {}
for tag, mid in MODELS:
    print(f"=== {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    for concept, sents in BANK.items():
        for i, s in enumerate(sents):
            n = common.toklen(tok, s)
            res[f"{concept}/{i}/{tag}"] = {"nll_pt": round(-common.slp(model, tok, "\n", s) / n, 4), "ntok": n}
    del tok, model
    common.free()

json.dump(res, open(os.path.join(common.RESULTS, "diag_paraphrase_pilot.json"), "w", encoding="utf-8"), indent=1)


def vals(concept, readout):
    out = []
    for i in range(20):
        if readout == "S1900":
            out.append(res[f"{concept}/{i}/1900"]["nll_pt"])
        else:  # delta = S1900 - S1929 per token, same string
            out.append(res[f"{concept}/{i}/1900"]["nll_pt"] - res[f"{concept}/{i}/1929"]["nll_pt"])
    return out


def boot_diff(a, b, B=10000):
    na, nb = len(a), len(b)
    ds = sorted(st.mean([a[random.randrange(na)] for _ in range(na)])
                - st.mean([b[random.randrange(nb)] for _ in range(nb)]) for _ in range(B))
    return ds[int(.025 * B)], ds[int(.975 * B)]


print("\n" + "=" * 92)
print("PILOT VERDICT — 20 paraphrases per concept, per-token nats, English, novel minus familiar")
print("=" * 92)
for novel, familiar in PAIRS:
    for readout, label in [("S1900", "plain 1900 surprise"), ("delta", "1900 minus 1929 (old estimator)")]:
        a, b = vals(novel, readout), vals(familiar, readout)
        lo, hi = boot_diff(a, b)
        verdict = "NOVEL WINS" if lo > 0 else ("familiar wins (!)" if hi < 0 else "no separation")
        print(f"{novel:15s} vs {familiar:14s} | {label:31s} | {st.mean(a):+.3f} vs {st.mean(b):+.3f} "
              f"| diff {st.mean(a)-st.mean(b):+.3f} [{lo:+.3f},{hi:+.3f}]  {verdict}")
    print()
print("per-concept means (per-token nats):")
for c in BANK:
    print(f"  {c:15s} 1900 {st.mean(vals(c,'S1900')):+.3f}   Δ(1900-1929) {st.mean(vals(c,'delta')):+.3f}"
          f"   Δ(1913-1929) {st.mean(res[f'{c}/{i}/1913']['nll_pt'] - res[f'{c}/{i}/1929']['nll_pt'] for i in range(20)):+.3f}")
print("\nwrote results/diag_paraphrase_pilot.json")
