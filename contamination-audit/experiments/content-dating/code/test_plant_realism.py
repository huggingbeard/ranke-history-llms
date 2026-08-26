"""G3 realism gate: can a blind judge tell corrupted plants from genuine no-imprint books?

v1 (2026-07-23). deepseek-v4-flash sees the first 1,500 chars of shuffled fronts — 80 plants vs
80 genuine certified-clean no-imprint books — and answers "does this look like a NATURAL book
scan opening, or artificially assembled/tampered?". If accuracy ~50% (AUC ~0.5), plants are
indistinguishable and the PI's realism condition is met. Output: _plant_realism.csv + verdict.
"""
import os, sys, json, random, asyncio
import pandas as pd
import pyarrow.parquet as pq
import httpx

sys.path.insert(0, r"C:\Users\hvoth\Documents\GitHub\ranke-edition-dating")
from freeze_pipeline import imprint_year

REPO = os.path.dirname(os.path.abspath(__file__))
DROPBOX = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\experiments\content-dating"
CORPUS = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\temp-guohui\decontamination\datasets\history-llms\sample-10gb\data"
OKEY = open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\openrouter.txt").read().strip()
OURL = "https://openrouter.ai/api/v1/chat/completions"
SEED = 7
N = 80

PROMPT = ('Below is the opening of a digitized book scan (OCR text). Real scans often start with '
          'library bookplates, provenance notes, shelf marks, OCR noise, or even mid-content when '
          'front pages are missing. Question: does this opening look like a NATURAL book scan, or '
          'does it look ARTIFICIALLY ASSEMBLED (e.g., text spliced together, tampered, or edited '
          'to remove information)? Opening:\n"""{p}"""\n'
          'Return JSON: {{"artificial": true/false, "why": "<=10 words"}}')

async def ask(c, sem, p):
    b = {"model": "deepseek/deepseek-v4-flash", "reasoning": {"enabled": False},
         "messages": [{"role": "user", "content": PROMPT.format(p=p)}],
         "temperature": 0, "max_tokens": 120, "provider": {"sort": "price"}}
    async with sem:
        for a in range(4):
            try:
                r = await c.post(OURL, headers={"Authorization": f"Bearer {OKEY}"}, json=b, timeout=90)
                if r.status_code >= 500 or r.status_code == 429:
                    await asyncio.sleep(2 * (a + 1)); continue
                r.raise_for_status()
                ct = r.json()["choices"][0]["message"]["content"]
                return bool(json.loads(ct[ct.find("{"):ct.rfind("}") + 1]).get("artificial"))
            except Exception:
                if a == 3: return None
                await asyncio.sleep(2 * (a + 1))

async def main():
    rng = random.Random(SEED)
    plants = pd.read_csv(os.path.join(REPO, "manifest_plants.csv"), dtype=str)
    ps = plants.sample(N, random_state=SEED)
    plant_fronts = [(r.plant_id, "plant",
                     open(os.path.join(DROPBOX, "plants_text", r.plant_id + ".txt"), encoding="utf-8").read()[:1500])
                    for r in ps.itertuples()]
    neg = pd.read_csv(os.path.join(REPO, "manifest_negatives.csv"), dtype={"id": str}).sample(260, random_state=SEED)
    fronts = []
    import pyarrow.compute as pc
    for sh, g in neg.groupby("shard"):
        t = pq.read_table(os.path.join(CORPUS, sh), columns=["id", "text"])
        t = t.filter(pc.is_in(t.column("id"), value_set=__import__("pyarrow").array(list(g.id))))
        for i, tx in zip(t.column("id").to_pylist(), t.column("text").to_pylist()):
            if tx and len(tx) > 5000 and imprint_year(tx) is None:
                fronts.append((i, "genuine", tx[:1500]))
    rng.shuffle(fronts)
    items = plant_fronts + fronts[:N]
    rng.shuffle(items)
    print(f"G3 panel: {len(plant_fronts)} plants vs {len(fronts[:N])} genuine no-imprint", flush=True)
    sem = asyncio.Semaphore(24)
    async with httpx.AsyncClient() as c:
        verdicts = await asyncio.gather(*[ask(c, sem, p) for _, _, p in items])
    df = pd.DataFrame({"id": [i for i, _, _ in items], "truth": [t for _, t, _ in items],
                       "called_artificial": verdicts})
    df.to_csv(os.path.join(REPO, "_plant_realism.csv"), index=False)
    d = df[df.called_artificial.notna()]
    tp = ((d.truth == "plant") & d.called_artificial).sum()
    fp = ((d.truth == "genuine") & d.called_artificial).sum()
    n_p = (d.truth == "plant").sum(); n_g = (d.truth == "genuine").sum()
    acc = (tp + (n_g - fp)) / (n_p + n_g)
    print(f"judge called artificial: plants {tp}/{n_p} ({tp/n_p:.0%}), genuine {fp}/{n_g} ({fp/n_g:.0%})")
    print(f"balanced accuracy: {(tp/n_p + (n_g-fp)/n_g)/2:.1%}  (50% = indistinguishable = GATE PASSED)")

if __name__ == "__main__":
    asyncio.run(main())
