# -*- coding: utf-8 -*-
"""FRAME PROBE v2 — the full trichotomy: NORMAL vs BROKE vs UNTHINKABLE, across Ranke 1900/1913/1929 (base).

Adds over v1:
  (1) a GENUINELY-UNTHINKABLE class (post-1929 facts alien to all three eras) — the flat baseline.
  (2) a PRODUCTION-REVERSION metric from the free generations (does the planted entity survive, or does the
      old default creep back), giving a RECOGNITION (forced-choice) vs PRODUCTION (generation) axis.
  (3) 4 phrasings x 13 dimensions = 52 items, bootstrap CIs over phrasings, richer classifier.

Signals per (model, item):
  pref0   = logP(A+|neutral+probe) - logP(A-|neutral+probe)        default preference, NO fact (per dim)
  pref_far= logP(A+|fact+filler+probe) - logP(A-|...)              RECOGNITION: holds the fact across distance?
  gen     = logP(consequence|fact) - logP(consequence|neutral)     USE-IT: can it run one step forward?
  ent_far = entropy at the probe slot                              confidence
  reversion = (neg_markers - pos_markers)/(neg+pos) in free gens   PRODUCTION: >0 old default creeps back
Fingerprint (calibrated to controls within each model, using BOTH recognition and production):
  PRIMED      held + generative + production does not revert            (behaves like a normal fact)
  BROKE       represents/confident BUT production reverts to default    (fluent distortion: field-marshal move)
  UNTHINKABLE not held + weak generativity                             (nothing to run forward from)
  MIXED       partial

Base models only (Instruct = modern contamination). Honest scope: measures the era's WRITTEN-CORPUS frame, not minds.
Output: frame_probe_v2.json / .csv
"""
import os, re, json, csv, math, random, statistics as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = {"1900": "uzh-echist-org/Ranke-4B-1900", "1913": "uzh-echist-org/Ranke-4B-1913", "1929": "uzh-echist-org/Ranke-4B-1929"}
NEUTRAL = "The season turned slowly that year."
FILLER = " The market kept its usual hours. Rain came and went. Men walked to their work as always. "
NGEN = 2          # phrasings per dim to free-generate (for reversion)
BOOT = 2000       # bootstrap resamples

# dim -> class, probe (ends before the A+/A- slot), a_plus, a_minus, consequence, pos/neg gen markers, plants[4]
DIMS = {
 # ---------------- period-normal CONTROLS ----------------
 "inheritance": dict(cls="control", probe=" Ever after, the firm was run by the", ap=" son", am=" father",
   cons=" He carried on the trade just as his father had.",
   pos=["son","he","him","his","heir"], neg=["father","stranger","widow","outsider"],
   plants=["When the old merchant died, his son took over the family firm.",
           "The firm passed, as was right, from father to son.",
           "On the father's death the son inherited the whole business.",
           "The old man left his trading house to his only son."]),
 "railway": dict(cls="control", probe=" A journey that once took weeks now took only a few", ap=" hours", am=" weeks",
   cons=" Goods reached the far city in a single morning.",
   pos=["railway","train","rail","hours","quick","swift","fast"], neg=["cart","horse","weeks","slow","days","wagon"],
   plants=["A new railway now joined the two distant cities.",
           "They built a railway between the city and the sea.",
           "An iron railway was laid down across the whole county.",
           "The two towns were joined at last by a line of rail."]),
 "succession": dict(cls="control", probe=" The throne now passed to the late king's", ap=" son", am=" foe",
   cons=" He was crowned before the assembled lords.",
   pos=["son","prince","king","crowned","heir","he"], neg=["foe","enemy","usurper","stranger","rebel"],
   plants=["When the old king died, his son was crowned in his place.",
           "The crown passed from the dead king to his eldest son.",
           "On the king's death his heir ascended the throne.",
           "The young prince was made king after his father died."]),
 "treaty": dict(cls="control", probe=" With the treaty signed at last, the long war finally", ap=" ended", am=" worsened",
   cons=" The soldiers laid down their arms and went home.",
   pos=["peace","ended","signed","treaty","home","arms"], neg=["war","fighting","battle","worsened","continued"],
   plants=["The two great powers signed a treaty, and the war was over.",
           "A treaty of peace was signed between the warring nations.",
           "The powers met and made a lasting peace between them.",
           "At last the long war was ended by a solemn treaty."]),
 # ---------------- BOUNDARY (became thinkable 1900->1929) ----------------
 "woman_authority": dict(cls="boundary", probe=" The one who now led the bank was a", ap=" woman", am=" man",
   cons=" Her judgment now guided the bank's lending.",
   pos=["she","her","woman","women","herself","lady","madam"], neg=["he","him","his","man","men","himself","gentleman"],
   plants=["A woman was appointed to lead the national bank.",
           "The new director of the great bank was a woman.",
           "The bank came at last to be headed by a woman.",
           "A woman now sat at the head of the board of directors."]),
 "human_flight": dict(cls="boundary", probe=" To cross the ocean now took not weeks but a few", ap=" hours", am=" weeks",
   cons=" Letters now crossed the wide sea in a single day.",
   pos=["flew","flight","air","wings","aeroplane","plane","pilot","aviator","sky"],
   neg=["ship","sail","sailed","steamer","boat","deck","voyage","harbour"],
   plants=["Two men flew across the great ocean in a single day.",
           "A flying machine carried men over the wide ocean.",
           "Men crossed the ocean through the air, from shore to shore.",
           "An airship bore two travellers across the sea in one day."]),
 "empire_fall": dict(cls="boundary", probe=" The vast country was now ruled by its", ap=" people", am=" emperor",
   cons=" The old nobles had lost all of their power.",
   pos=["republic","people","revolution","council","citizens","elected"],
   neg=["tsar","czar","emperor","king","crown","throne","imperial","majesty"],
   plants=["The empire of Russia was overthrown, and the Tsar was no more.",
           "The people rose up and cast down the Russian crown.",
           "Russia's ancient throne was destroyed in a great revolt.",
           "The mighty Russian empire fell, and a republic took its place."]),
 "wireless_voice": dict(cls="boundary", probe=" The distant message reached the town not by any wire but through the open", ap=" air", am=" post",
   cons=" Music played in one city was heard at once in many homes.",
   pos=["air","wireless","waves","broadcast","heard","signal","ether"],
   neg=["wire","cable","telegraph","post","letter","messenger"],
   plants=["Voices were carried through the air with no wire at all.",
           "Men spoke into a machine and were heard a thousand miles away.",
           "Music played in one city was heard at once in distant homes.",
           "Words now travelled through the empty air, without any wire."]),
 # ---------------- UNTHINKABLE (post-1929, beyond all three) ----------------
 "thinking_machine": dict(cls="unthinkable", probe=" When the hardest problem had to be solved, it was solved by the", ap=" machine", am=" clerk",
   cons=" It could reason and decide far faster than any man.",
   pos=["machine","engine","it","iron","metal","wheels"], neg=["man","clerk","scholar","men","person","hand"],
   plants=["A machine was built that could think and reason like a man.",
           "They made an engine of metal that could think for itself.",
           "A great calculating machine now did the work of a thousand clerks.",
           "Men built a thinking mind out of wire and metal."]),
 "atom_city": dict(cls="unthinkable", probe=" The whole city was laid waste, not by a long siege, but in a single", ap=" instant", am=" year",
   cons=" One weapon alone could now destroy a great city.",
   pos=["instant","flash","moment","single","seconds","blast","light"], neg=["siege","months","army","soldiers","years","slowly"],
   plants=["A single bomb destroyed an entire city in a flash of light.",
           "One weapon wiped a whole city from the earth in an instant.",
           "In a single blast, a great city was utterly destroyed.",
           "A lone bomb laid a city waste in the blink of an eye."]),
 "television": dict(cls="unthinkable", probe=" Sound and living pictures were sent into the family's home through the open", ap=" air", am=" door",
   cons=" A family saw, in their own parlour, events from across the world.",
   pos=["air","waves","screen","picture","pictures","faces","broadcast","sent"], neg=["letter","post","messenger","book","newspaper","door"],
   plants=["Living faces and voices were sent through the air into every home.",
           "Men saw, in their own parlour, things happening far away.",
           "Moving pictures of distant events appeared inside the home.",
           "The faces of far-off people were cast through the air onto a screen."]),
 "dna_code": dict(cls="unthinkable", probe=" The deepest pattern of life, once thought fixed forever, could now be", ap=" changed", am=" admired",
   cons=" Men could now alter the very seed of life itself.",
   pos=["code","read","written","rewritten","changed","altered","life","seed","blood"], neg=["god","nature","fate","soul","fixed","divine","unchanging"],
   plants=["The very code of life was read, and then rewritten by hand.",
           "Men learned to read and change the secret pattern of life.",
           "The hidden plan within every living seed was rewritten at will.",
           "Scholars unrolled the code of life and altered it as they pleased."]),
 "moon_landing": dict(cls="unthinkable", probe=" At last men stood and walked upon the face of the", ap=" moon", am=" earth",
   cons=" Men left the earth and set foot upon another world.",
   pos=["moon","space","rocket","heavens","stars","world"], neg=["earth","ground","dream","story","fancy"],
   plants=["Men left the earth and walked upon the face of the moon.",
           "Two travellers set their feet upon the distant moon.",
           "A ship of men flew to the moon and landed there.",
           "Mankind crossed the void and stood upon the moon."]),
}

ITEMS = []
for dim, d in DIMS.items():
    for i, plant in enumerate(d["plants"]):
        ITEMS.append(dict(id=f"{dim}_{i+1}", dim=dim, cls=d["cls"], plant=plant))


def logp_cont(model, tok, ctx, cont):
    pid = tok(ctx, add_special_tokens=False).input_ids
    cid = tok(cont, add_special_tokens=False).input_ids
    full = pid + cid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(cid)))


def entropy_next(model, tok, ctx):
    ids = tok(ctx, add_special_tokens=False).input_ids
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=model.device)).logits[0, -1].float()
    p = torch.softmax(logits, dim=-1)
    return float(-(p * torch.log(p + 1e-12)).sum())


def generate(model, tok, prompt, n=45):
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=n, do_sample=False, repetition_penalty=1.3, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True).strip()


def rev_score(text, pos, neg):
    t = text.lower()
    pc = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", t)) for w in pos)
    nc = sum(len(re.findall(r"\b" + re.escape(w) + r"\b", t)) for w in neg)
    if pc + nc == 0:
        return None, pc, nc
    return (nc - pc) / (pc + nc), pc, nc


results = {tag: {} for tag in MODELS}
gens = {tag: {} for tag in MODELS}
for tag, mid in MODELS.items():
    print(f"\n=== {mid} ({tag}) ===", flush=True)
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    pref0_cache = {}
    for it in ITEMS:
        d = DIMS[it["dim"]]; probe, ap, am = d["probe"], d["ap"], d["am"]
        if it["dim"] not in pref0_cache:
            base = NEUTRAL + probe
            pref0_cache[it["dim"]] = logp_cont(model, tok, base, ap) - logp_cont(model, tok, base, am)
        pref0 = pref0_cache[it["dim"]]
        far = it["plant"] + FILLER + probe
        near = it["plant"] + probe
        pref_near = logp_cont(model, tok, near, ap) - logp_cont(model, tok, near, am)
        pref_far = logp_cont(model, tok, far, ap) - logp_cont(model, tok, far, am)
        gen = logp_cont(model, tok, it["plant"], d["cons"]) - logp_cont(model, tok, NEUTRAL, d["cons"])
        ent_far = entropy_next(model, tok, far)
        results[tag][it["id"]] = dict(pref0=round(pref0,3), pref_near=round(pref_near,3), pref_far=round(pref_far,3),
                                      gen=round(gen,3), ent_far=round(ent_far,3))
    # free generations + reversion for first NGEN phrasings of each dim
    for dim, d in DIMS.items():
        gens[tag][dim] = []
        for i in range(NGEN):
            plant = d["plants"][i]
            g_use = generate(model, tok, plant + " What followed from this was that")
            g_hold = generate(model, tok, plant + FILLER + " The one who truly held the highest place was")
            rv, pc, nc = rev_score(g_use + " " + g_hold, d["pos"], d["neg"])
            gens[tag][dim].append(dict(plant=plant, use_it=g_use, hold_it=g_hold, reversion=rv, pos=pc, neg=nc))
    print(f"  done {tag}", flush=True)
    del model
    import gc; gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()


def boot_ci(vals, fn=st.mean):
    if len(vals) < 2:
        return (vals[0], vals[0]) if vals else (0, 0)
    rng = random.Random(12345)
    bs = sorted(fn([vals[rng.randrange(len(vals))] for _ in vals]) for _ in range(BOOT))
    return (round(bs[int(0.025*BOOT)],2), round(bs[int(0.975*BOOT)],2))


# ---- aggregate per (model,dim) + classify ----
agg = {tag: {} for tag in MODELS}
rows = []
for tag in MODELS:
    # control calibration floors (20th pct of control ITEMS)
    cprf = sorted(results[tag][it["id"]]["pref_far"] for it in ITEMS if it["cls"] == "control")
    cgen = sorted(results[tag][it["id"]]["gen"] for it in ITEMS if it["cls"] == "control")
    cent = [results[tag][it["id"]]["ent_far"] for it in ITEMS if it["cls"] == "control"]
    floor_pref = cprf[max(0, int(0.2*len(cprf))-1)]
    floor_gen = cgen[max(0, int(0.2*len(cgen))-1)]
    ent_med = st.median(cent)
    for dim, d in DIMS.items():
        ids = [it["id"] for it in ITEMS if it["dim"] == dim]
        pf = [results[tag][i]["pref_far"] for i in ids]
        p0 = [results[tag][i]["pref0"] for i in ids]
        gn = [results[tag][i]["gen"] for i in ids]
        en = [results[tag][i]["ent_far"] for i in ids]
        revs = [g["reversion"] for g in gens[tag][dim] if g["reversion"] is not None]
        rev_mean = st.mean(revs) if revs else None
        m_pf, m_gn = st.mean(pf), st.mean(gn)
        held = m_pf >= floor_pref
        generative = m_gn >= floor_gen
        reverts = rev_mean is not None and rev_mean > 0.2
        confident = st.mean(en) <= ent_med
        if held and generative and not reverts:
            fp = "PRIMED"
        elif reverts and (held or confident):
            fp = "BROKE"
        elif (not held) and (not generative):
            fp = "UNTHINKABLE"
        else:
            fp = "MIXED"
        agg[tag][dim] = dict(cls=d["cls"], pref0=round(st.mean(p0),2), pref_far=round(m_pf,2),
                             pref_far_ci=boot_ci(pf), gen=round(m_gn,2), gen_ci=boot_ci(gn),
                             ent=round(st.mean(en),2), reversion=(round(rev_mean,2) if rev_mean is not None else None),
                             fingerprint=fp, floor_pref=round(floor_pref,2), floor_gen=round(floor_gen,2))
        rows.append(dict(model=tag, dim=dim, **agg[tag][dim]))

json.dump({"models": MODELS, "neutral": NEUTRAL, "filler": FILLER, "dims_meta": {k: {"cls": v["cls"], "probe": v["probe"], "ap": v["ap"], "am": v["am"], "cons": v["cons"]} for k, v in DIMS.items()},
           "results": results, "agg": agg, "generations": gens, "items": ITEMS},
          open(os.path.join(HERE, "frame_probe_v2.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with open(os.path.join(HERE, "frame_probe_v2.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["model","dim","cls","pref0","pref_far","gen","ent","reversion","fingerprint"])
    w.writeheader()
    for r in rows:
        w.writerow({k: r[k] for k in w.fieldnames})

# ---- console matrix ----
print("\n" + "=" * 120)
print("FINGERPRINT MATRIX  (pf=pref_far hold-it | g=generativity | rev=production reversion >0 means default creeps back)")
print(f"{'dimension':18s}{'class':12s}| " + " | ".join(f"{t:^30s}" for t in MODELS))
print("-" * 120)
order = ["control", "boundary", "unthinkable"]
for cls in order:
    for dim, d in DIMS.items():
        if d["cls"] != cls: continue
        cells = []
        for tag in MODELS:
            a = agg[tag][dim]
            rv = f"{a['reversion']:+.2f}" if a["reversion"] is not None else "  na "
            cells.append(f"{a['fingerprint']:11s} pf{a['pref_far']:+5.1f} g{a['gen']:+5.1f} r{rv}")
        print(f"{dim:18s}{cls:12s}| " + " | ".join(cells))
    print("-" * 120)
print("PRIMED=normal-fact | BROKE=recognises but production reverts (distortion) | UNTHINKABLE=can't hold or run forward")
print("wrote frame_probe_v2.json / .csv")
