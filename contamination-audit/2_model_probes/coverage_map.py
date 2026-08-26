# -*- coding: utf-8 -*-
"""ERA x REGION coverage map. Forced-choice recognition (length-matched twin, correct = gap>0)
of should-know facts, binned by era and region -> heatmap of WHERE the model's pre-cutoff
knowledge is solid vs thin (explains Nelson >> Hindenburg). Saves 5_results/coverage_results.csv.
All four models, one process."""
import os, re, csv, gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEV = "cuda" if torch.cuda.is_available() else "cpu"; DT = torch.bfloat16 if DEV == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\coverage_results.csv"

# (era, region, prompt, true, [twins])
F = [
 # ===== BRITISH =====
 ("E1<=1789", "BR", "The English king beheaded in 1649 was Charles", " I", [" V", " X"]),
 ("E1<=1789", "BR", "The fleet Spain sent against England in 1588 was the", " Armada", [" Flota", " Mesta"]),
 ("E1<=1789", "BR", "The Norman who conquered England in 1066 was William the", " Conqueror", [" Bastard", " Norman"]),
 ("E2 1789-1815", "BR", "The admiral who died at Trafalgar in 1805 was", " Nelson", [" Collingwood", " Jervis"]),
 ("E2 1789-1815", "BR", "Napoleon was beaten at Waterloo by the Duke of", " Wellington", [" Marlborough", " Cumberland"]),
 ("E2 1789-1815", "BR", "The 1801 Act of Union joined Britain with", " Ireland", [" Scotland", " Hanover"]),
 ("E3 1815-1865", "BR", "Victoria's Liberal prime minister, rival of Disraeli, was", " Gladstone", [" Palmerston", " Russell"]),
 ("E3 1815-1865", "BR", "The 1832 act that widened the vote was the Great", " Reform", [" Charter", " Union"]),
 ("E3 1815-1865", "BR", "The queen crowned in 1837 was", " Victoria", [" Adelaide", " Caroline"]),
 ("E4 1865-1900", "BR", "Victoria's Conservative prime minister, an author too, was", " Disraeli", [" Salisbury", " Derby"]),
 ("E4 1865-1900", "BR", "The 1899-1902 war in South Africa was the", " Boer", [" Zulu", " Ashanti"]),
 ("E4 1865-1900", "BR", "The 1888 serial killer in London was Jack the", " Ripper", [" Slasher", " Strangler"]),
 ("E5 1900-1914", "BR", "The liner that sank on its maiden voyage in 1912 was the", " Titanic", [" Lusitania", " Mauretania"]),
 ("E5 1900-1914", "BR", "The militant suffragette leader was Emmeline", " Pankhurst", [" Fawcett", " Davison"]),
 ("E5 1900-1914", "BR", "The Liberal PM who introduced old-age pensions around 1909 was", " Asquith", [" Balfour", " Campbell"]),
 ("E6 1914-1929", "BR", "The British prime minister for most of the First World War was David Lloyd", " George", [" Asquith", " Bonar"]),
 ("E6 1914-1929", "BR", "The 1916 Dublin rising against British rule was the Easter", " Rising", [" Revolt", " Rebellion"]),
 ("E6 1914-1929", "BR", "The major British industrial action of 1926 was the General", " Strike", [" Lockout", " Walkout"]),
 # ===== AMERICAN =====
 ("E1<=1789", "US", "The first president of the United States was George", " Washington", [" Adams", " Hancock"]),
 ("E1<=1789", "US", "The principal author of the 1776 Declaration of Independence was Thomas", " Jefferson", [" Adams", " Paine"]),
 ("E1<=1789", "US", "The 1787 convention in Philadelphia drafted the U.S.", " Constitution", [" Confederation", " Charter"]),
 ("E2 1789-1815", "US", "The 1803 purchase from France that doubled U.S. territory was the", " Louisiana", [" Florida", " Missouri"]),
 ("E2 1789-1815", "US", "The 1812 war was fought mainly against", " Britain", [" Spain", " France"]),
 ("E3 1815-1865", "US", "The U.S. president during the Civil War was Abraham", " Lincoln", [" Douglas", " Seward"]),
 ("E3 1815-1865", "US", "The Union's commanding general, later president, was Ulysses S.", " Grant", [" Sherman", " Meade"]),
 ("E3 1815-1865", "US", "The commander of the Confederate army was Robert E.", " Lee", [" Bragg", " Hood"]),
 ("E4 1865-1900", "US", "The inventor of the telephone in 1876 was Alexander Graham", " Bell", [" Morse", " Edison"]),
 ("E4 1865-1900", "US", "In 1898 the United States went to war against", " Spain", [" Mexico", " Britain"]),
 ("E4 1865-1900", "US", "The era of rebuilding the South after the Civil War was called", " Reconstruction", [" Restoration", " Redemption"]),
 ("E5 1900-1914", "US", "The U.S. president during the First World War was Woodrow", " Wilson", [" Taft", " Hughes"]),
 ("E5 1900-1914", "US", "The brothers who made the first powered flight in 1903 were the", " Wright", [" Curtiss", " Wright"]),
 ("E5 1900-1914", "US", "The 'Rough Rider' who became president in 1901 was Theodore", " Roosevelt", [" Bryan", " Taft"]),
 ("E6 1914-1929", "US", "The Republican president elected in 1920 on 'normalcy' was Warren", " Harding", [" Coolidge", " Hughes"]),
 ("E6 1914-1929", "US", "American women won the right to vote nationwide in 19", " 20", [" 16", " 28"]),
 ("E6 1914-1929", "US", "The 1919-33 nationwide ban on alcohol was called", " Prohibition", [" Abstinence", " Temperance"]),
 # ===== FRENCH =====
 ("E1<=1789", "FR", "The French 'Sun King' was Louis", " XIV", [" XII", " XVI"]),
 ("E1<=1789", "FR", "The Paris prison stormed on 14 July 1789 was the", " Bastille", [" Temple", " Louvre"]),
 ("E1<=1789", "FR", "The radical leader of the Reign of Terror was Maximilien", " Robespierre", [" Danton", " Mirabeau"]),
 ("E2 1789-1815", "FR", "The general who crowned himself Emperor in 1804 was", " Napoleon", [" Bernadotte", " Murat"]),
 ("E2 1789-1815", "FR", "Napoleon's great 1805 victory over Austria and Russia was", " Austerlitz", [" Borodino", " Wagram"]),
 ("E3 1815-1865", "FR", "The 1848 revolution toppled the French king Louis", " Philippe", [" Charles", " Bourbon"]),
 ("E3 1815-1865", "FR", "The author of 'Les Misérables' was Victor", " Hugo", [" Balzac", " Zola"]),
 ("E4 1865-1900", "FR", "In 1870-71 France lost a war to", " Prussia", [" Austria", " Britain"]),
 ("E4 1865-1900", "FR", "The army officer wrongly convicted of treason in 1894 was Alfred", " Dreyfus", [" Esterhazy", " Picquart"]),
 ("E4 1865-1900", "FR", "The Impressionist painter of water lilies was Claude", " Monet", [" Manet", " Degas"]),
 ("E5 1900-1914", "FR", "The first man to fly across the English Channel, in 1909, was Louis", " Bleriot", [" Farman", " Voisin"]),
 ("E5 1900-1914", "FR", "The author of 'In Search of Lost Time', begun 1913, was Marcel", " Proust", [" Gide", " Mauriac"]),
 ("E6 1914-1929", "FR", "The longest battle of the First World War, at a French fortress, was", " Verdun", [" Sedan", " Cambrai"]),
 ("E6 1914-1929", "FR", "The 1919 peace treaty was signed at", " Versailles", [" Trianon", " Sevres"]),
 ("E6 1914-1929", "FR", "The supreme Allied commander at the end of the World War was Marshal", " Foch", [" Joffre", " Petain"]),
 # ===== GERMAN (the key region for the Hindenburg test) =====
 ("E1<=1789", "DE", "The Prussian king known as 'the Great' was Frederick", " II", [" III", " IV"]),
 ("E1<=1789", "DE", "The composer of the Brandenburg Concertos was Johann Sebastian", " Bach", [" Handel", " Telemann"]),
 ("E1<=1789", "DE", "The monk who launched the Reformation in 1517 was Martin", " Luther", [" Calvin", " Zwingli"]),
 ("E2 1789-1815", "DE", "Prussia was crushed by Napoleon in 1806 at the Battle of", " Jena", [" Leipzig", " Bautzen"]),
 ("E2 1789-1815", "DE", "The philosopher of Königsberg, author of the Critiques, was Immanuel", " Kant", [" Hegel", " Fichte"]),
 ("E3 1815-1865", "DE", "Napoleon was defeated in 1813 at the Battle of the Nations at", " Leipzig", [" Jena", " Dresden"]),
 ("E3 1815-1865", "DE", "The 1848 liberal German national parliament met at", " Frankfurt", [" Vienna", " Munich"]),
 ("E3 1815-1865", "DE", "The composer of the 'Ring' cycle of operas was Richard", " Wagner", [" Strauss", " Brahms"]),
 ("E4 1865-1900", "DE", "The 'Iron Chancellor' who unified Germany in 1871 was Otto von", " Bismarck", [" Moltke", " Roon"]),
 ("E4 1865-1900", "DE", "The German Kaiser who came to the throne in 1888 was Wilhelm", " II", [" I", " III"]),
 ("E4 1865-1900", "DE", "The composer of four symphonies who died in 1897 was Johannes", " Brahms", [" Bruckner", " Mahler"]),
 ("E5 1900-1914", "DE", "The physicist who published relativity in 1905 was Albert", " Einstein", [" Planck", " Hertz"]),
 ("E5 1900-1914", "DE", "The physicist who founded quantum theory in 1900 was Max", " Planck", [" Sommerfeld", " Born"]),
 ("E6 1914-1929", "DE", "The German victor at the 1914 Battle of Tannenberg was Paul von", " Hindenburg", [" Ludendorff", " Falkenhayn"]),
 ("E6 1914-1929", "DE", "Hindenburg's quartermaster-general and partner in command was Erich", " Ludendorff", [" Falkenhayn", " Mackensen"]),
 ("E6 1914-1929", "DE", "The German republic founded after the First World War was the", " Weimar", [" Bonn", " Rhenish"]),
 # ===== RUSSIAN =====
 ("E1<=1789", "RU", "The tsar who founded St Petersburg was Peter the", " Great", [" Terrible", " Wise"]),
 ("E1<=1789", "RU", "The German-born empress who ruled Russia from 1762 was Catherine the", " Great", [" Wise", " Bold"]),
 ("E2 1789-1815", "RU", "The 1812 invader of Russia, defeated by the winter, was", " Napoleon", [" Charles", " Gustavus"]),
 ("E3 1815-1865", "RU", "The tsar who emancipated the serfs in 1861 was Alexander", " II", [" I", " III"]),
 ("E3 1815-1865", "RU", "The author of 'War and Peace' was Leo", " Tolstoy", [" Turgenev", " Gogol"]),
 ("E3 1815-1865", "RU", "The composer of the '1812 Overture' was Pyotr", " Tchaikovsky", [" Borodin", " Glinka"]),
 ("E4 1865-1900", "RU", "The author of 'Crime and Punishment' was Fyodor", " Dostoevsky", [" Turgenev", " Goncharov"]),
 ("E4 1865-1900", "RU", "The Russian chemist who devised the periodic table was Dmitri", " Mendeleev", [" Butlerov", " Markovnikov"]),
 ("E5 1900-1914", "RU", "In 1904-05 Russia lost a war to", " Japan", [" China", " Britain"]),
 ("E5 1900-1914", "RU", "The composer of the 1913 ballet 'The Rite of Spring' was Igor", " Stravinsky", [" Prokofiev", " Scriabin"]),
 ("E6 1914-1929", "RU", "The Bolshevik leader of the 1917 October Revolution was Vladimir", " Lenin", [" Kerensky", " Martov"]),
 ("E6 1914-1929", "RU", "The last Russian tsar, who abdicated in 1917, was Nicholas", " II", [" I", " III"]),
 ("E6 1914-1929", "RU", "The union of soviet republics formed in 1922 was abbreviated the", " USSR", [" RSFSR", " CCCP"]),
]


def toklen(tok, s):
    return len(tok(s, add_special_tokens=False).input_ids)


def slp(model, tok, prompt, ans):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(ans, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


with open(OUT, "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow(["model", "cutoff", "era", "region", "true", "correct", "gap"])
for M in MODELS:
    cut = int(re.search(r"(\d{4})", M).group(1))
    tok = AutoTokenizer.from_pretrained(M)
    model = AutoModelForCausalLM.from_pretrained(M, dtype=DT, low_cpu_mem_usage=True).to(DEV).eval()
    rows = []
    for (era, reg, prompt, true, tws) in F:
        nt = toklen(tok, true); cand = [x for x in tws if toklen(tok, x) == nt]
        tw = cand[0] if cand else min(tws, key=lambda x: abs(toklen(tok, x) - nt))
        g = slp(model, tok, prompt, true) - slp(model, tok, prompt, tw)
        rows.append([M.split("/")[-1], cut, era, reg, true.strip(), int(g > 0), round(g, 3)])
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(M.split("/")[-1], "done", flush=True)
    del model; gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()
print("DONE ->", OUT)
