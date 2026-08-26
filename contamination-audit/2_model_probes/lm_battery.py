"""
Length-matched-by-design battery for red/amber/green BY CATEGORY.
Each item gives several candidate false twins; the scorer picks one with the SAME token
count as the true answer, then teacher-forces the whole answer (sum logp). Bias-free.
Output: 5_results/lm_battery.csv  (model,cutoff,country,domain,era,item,true,false,matched,gap,correct)
US-politics items are imported from us_politics_power.py (already powered, verdict RED).
"""
import os, re, csv, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from us_politics_power import PRE as US_PRE, POST as US_POST
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929",
          "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\lm_battery.csv"

# (country, domain, year, prompt, true, [false candidates of varied length])
B = []
def add(country, domain, items):
    for (y, p, t, fs) in items:
        B.append((country, domain, y, p, t, fs))

# US politics: reuse the powered set
for (y, p, t, fs) in US_PRE + US_POST:
    B.append(("USA", "politics", y, p, t, fs))

add("UK", "politics", [
 (1837,"The queen who reigned over Britain from 1837 for 63 years was Queen"," Victoria",[" Adelaide"," Caroline"," Charlotte"," Anne"]),
 (1868,"Victoria's Liberal prime minister, rival of Disraeli, was William"," Gladstone",[" Palmerston"," Russell"," Peel"," Asquith"]),
 (1874,"Victoria's favourite Conservative prime minister was Benjamin"," Disraeli",[" Salisbury"," Derby"," Balfour"," Peel"]),
 (1805,"The admiral who died at the Battle of Trafalgar in 1805 was Horatio"," Nelson",[" Collingwood"," Howe"," Jervis"," Rodney"]),
 (1815,"The British general who beat Napoleon at Waterloo was the Duke of"," Wellington",[" Marlborough"," Cumberland"," York"," Argyll"]),
 (1707,"The 1707 Act of Union joined England with"," Scotland",[" Ireland"," Wales"," Hanover"]),
 (1854,"Britain's 1854-56 war against Russia in the Crimea allied it with"," France",[" Prussia"," Austria"," Spain"]),
 (1916,"The Liberal PM who led Britain for most of WWI, from 1916, was David Lloyd"," George",[" Asquith"," Grey"," Balfour"]),
 (1901,"The monarch who succeeded Queen Victoria in 1901 was Edward"," VII",[" VIII"," VI"," IV"]),
 (1832,"The 1832 act that widened the British vote was the Great"," Reform",[" Charter"," Settlement"," Petition"]),
 (1940,"The prime minister who led Britain through most of WWII was Winston"," Churchill",[" Halifax"," Chamberlain"," Eden"," Attlee"]),
 (1945,"The Labour PM who won the 1945 landslide was Clement"," Attlee",[" Morrison"," Bevin"," Gaitskell"," Bevan"]),
 (1979,"Britain's first woman prime minister, elected 1979, was Margaret"," Thatcher",[" Williams"," Castle"," Jenkins"," Boothroyd"]),
 (1997,"The Labour prime minister elected in 1997 was Tony"," Blair",[" Smith"," Kinnock"," Brown"," Major"]),
 (1936,"The British king who abdicated in 1936 to marry was Edward"," VIII",[" VII"," VI"," IX"]),
 (1956,"The 1956 crisis when Britain seized an Egyptian canal was the"," Suez",[" Aden"," Nile"," Sinai"]),
 (1982,"In 1982 Britain went to war with Argentina over the"," Falklands",[" Hebrides"," Orkneys"," Shetlands"]),
 (2016,"Britain's 2016 vote to leave the European Union was nicknamed"," Brexit",[" Bremain"," Frexit"," Grexit"]),
 (1990,"The Conservative who succeeded Thatcher as PM in 1990 was John"," Major",[" Hurd"," Heseltine"," Lamont"," Howe"]),
 (2010,"The Conservative who became PM in a 2010 coalition was David"," Cameron",[" Clegg"," Osborne"," Hague"," Davis"]),
 (1964,"The Labour PM elected in 1964, 'white heat of technology', was Harold"," Wilson",[" Gaitskell"," Callaghan"," Brown"," Jenkins"]),
])

add("France", "politics", [
 (1804,"The general who crowned himself Emperor of the French in 1804 was Napoleon"," Bonaparte",[" Bernadotte"," Murat"," Masséna"," Ney"]),
 (1789,"The Paris prison stormed on 14 July 1789 was the"," Bastille",[" Conciergerie"," Temple"," Châtelet"]),
 (1870,"The emperor who fell after the 1870 defeat by Prussia was Napoleon"," III",[" II"," IV"," I"]),
 (1894,"The Jewish officer wrongly convicted of treason in 1894 was Alfred"," Dreyfus",[" Esterhazy"," Picquart"," Henry"]),
 (1799,"Napoleon seized power in the 1799 coup of 18"," Brumaire",[" Fructidor"," Thermidor"," Vendémiaire"]),
 (1815,"Napoleon was finally defeated in 1815 at the Battle of"," Waterloo",[" Austerlitz"," Borodino"," Jena"," Leipzig"]),
 (1871,"The radical 1871 Paris uprising crushed by the government was the"," Commune",[" Fronde"," Terror"," Directory"]),
 (1848,"The 1848 revolution toppled the French king Louis"," Philippe",[" Napoleon"," Charles"," Bourbon"]),
 (1940,"The general who led the Free French in WWII was Charles de"," Gaulle",[" Lattre"," Leclerc"," Juin"]),
 (1940,"The WWI marshal who led the collaborationist Vichy regime was Philippe"," Pétain",[" Weygand"," Laval"," Darlan"," Gamelin"]),
 (1981,"The first Socialist president of the Fifth Republic, from 1981, was François"," Mitterrand",[" Rocard"," Fabius"," Mauroy"," Defferre"]),
 (1995,"The Gaullist elected French president in 1995 was Jacques"," Chirac",[" Jospin"," Balladur"," Juppé"," Barre"]),
 (2017,"The youngest French president, elected 2017, was Emmanuel"," Macron",[" Hamon"," Fillon"," Valls"," Mélenchon"]),
 (2007,"The French president elected in 2007, nicknamed 'Sarko', was Nicolas"," Sarkozy",[" Royal"," Bayrou"," Villepin"]),
 (1916,"The longest battle of WWI, at a French fortress in 1916, was"," Verdun",[" Sedan"," Cambrai"," Somme"]),
 (1944,"The 1944 Allied landings in France took place in"," Normandy",[" Brittany"," Provence"," Picardy"," Gascony"]),
 (1962,"The North African colony that won independence from France in 1962 was"," Algeria",[" Morocco"," Tunisia"," Senegal"," Mali"]),
 (1958,"The republic de Gaulle founded in 1958 was the"," Fifth",[" Fourth"," Third"," Second"]),
 (2012,"The Socialist French president elected in 2012 was François"," Hollande",[" Fabius"," Ayrault"," Valls"," Aubry"]),
])

add("Germany", "politics", [
 (1871,"The 'Iron Chancellor' who unified Germany in 1871 was Otto von"," Bismarck",[" Moltke"," Roon"," Caprivi"," Bülow"]),
 (1888,"The Kaiser who reigned over Germany through WWI was Wilhelm"," II",[" I"," III"," IV"]),
 (1848,"The 1848 liberal German national parliament met in"," Frankfurt",[" Vienna"," Berlin"," Munich"," Dresden"]),
 (1919,"The republic founded in Germany after WWI was the"," Weimar",[" Bonn"," Rhenish"," Saxon"]),
 (1813,"Napoleon was defeated in 1813 at the Battle of the Nations at"," Leipzig",[" Jena"," Dresden"," Bautzen"]),
 (1933,"The Nazi leader who became German chancellor in 1933 was Adolf"," Hitler",[" Hugenberg"," Papen"," Schleicher"," Strasser"]),
 (1949,"The first chancellor of West Germany, from 1949, was Konrad"," Adenauer",[" Schumacher"," Erhard"," Heuss"," Ollenhauer"]),
 (1969,"The West German chancellor of Ostpolitik, Nobel 1971, was Willy"," Brandt",[" Kiesinger"," Wehner"," Schiller"]),
 (1990,"The chancellor who oversaw German reunification was Helmut"," Kohl",[" Genscher"," Schäuble"," Vogel"," Lafontaine"]),
 (2005,"The first woman chancellor of Germany, from 2005, was Angela"," Merkel",[" Künast"," Nahles"," Roth"]),
 (1933,"The 1933 fire Hitler used to seize emergency powers was the"," Reichstag",[" Bundestag"," Kremlin"," Bastille"]),
 (1938,"The 1938 Nazi pogrom against Germany's Jews was called"," Kristallnacht",[" Anschluss"," Blitzkrieg"," Sonderaktion"]),
 (1961,"The wall dividing a German city, built in 1961, was the Berlin"," Wall",[" Gate"," Line"," Fence"]),
 (1974,"The West German chancellor 1974-82, a crisis manager, was Helmut"," Schmidt",[" Genscher"," Wehner"," Apel"]),
 (1998,"The Social Democrat chancellor elected in 1998 was Gerhard"," Schröder",[" Stoiber"," Scharping"," Lafontaine"]),
 (1946,"The 1945-46 trials of Nazi leaders were held in"," Nuremberg",[" Potsdam"," Frankfurt"," Leipzig"]),
])

add("Italy", "politics", [
 (1860,"The general whose Redshirts helped unify Italy was Giuseppe"," Garibaldi",[" Mazzini"," Crispi"," Rattazzi"]),
 (1861,"The statesman who became Italy's first prime minister in 1861 was Camillo di"," Cavour",[" Crispi"," Rattazzi"," Minghetti"]),
 (1870,"Italian unification was completed in 1870 with the capture of"," Rome",[" Venice"," Naples"," Milan"," Turin"]),
 (1849,"The nationalist who proclaimed the 1849 Roman Republic was Giuseppe"," Mazzini",[" Garibaldi"," Manin"," Gioberti"]),
 (1922,"The Fascist leader who marched on Rome in 1922 was Benito"," Mussolini",[" Balbo"," Grandi"," Bottai"," Ciano"]),
 (1946,"Italy abolished the monarchy and became a republic in 19"," 46",[" 24"," 18"," 70"]),
 (1978,"The former PM kidnapped and murdered by the Red Brigades in 1978 was Aldo"," Moro",[" Andreotti"," Fanfani"," Cossiga"]),
 (1994,"The media tycoon who became Italian PM in the 1990s was Silvio"," Berlusconi",[" Prodi"," Dini"," Amato"]),
 (1945,"The Christian Democrat who led postwar Italy from 1945 was Alcide De"," Gasperi",[" Nenni"," Togliatti"," Saragat"]),
 (2014,"The young centre-left Italian prime minister from 2014 was Matteo"," Renzi",[" Letta"," Gentiloni"," Alfano"]),
])

add("Spain", "politics", [
 (1492,"The monarchs who funded Columbus and unified Spain were Ferdinand and"," Isabella",[" Joanna"," Catherine"," Maria"]),
 (1588,"The fleet Spain sent against England in 1588 was the"," Armada",[" Flota"," Hermandad"," Mesta"]),
 (1898,"In 1898 Spain lost Cuba in a war against the"," Americans",[" British"," French"," Dutch"]),
 (1936,"The general who won the Spanish Civil War and ruled to 1975 was Francisco"," Franco",[" Mola"," Sanjurjo"," Queipo"]),
 (1937,"The Basque town bombed in 1937 and painted by Picasso was"," Guernica",[" Bilbao"," Santander"," Oviedo"]),
 (1975,"The king who restored democracy to Spain after Franco was Juan"," Carlos",[" Felipe"," Alfonso"]),
 (1982,"Spain's first Socialist prime minister after Franco, from 1982, was Felipe"," González",[" Suárez"," Guerra"," Calvo"]),
 (1996,"The conservative Spanish prime minister elected in 1996 was José María"," Aznar",[" Rajoy"," Arenas"," Mayor"]),
 (1959,"The Basque separatist group founded in 1959 was"," ETA",[" FARC"," IRA"," PKK"]),
])

add("Russia", "politics", [
 (1917,"The last Tsar of Russia, who abdicated in 1917, was Nicholas"," II",[" I"," III"," IV"]),
 (1861,"The Tsar who emancipated the serfs in 1861 was Alexander"," II",[" I"," III"," IV"]),
 (1812,"The 1812 invader of Russia, defeated by the winter, was"," Napoleon",[" Charles"," Frederick"," Gustavus"]),
 (1905,"The failed Russian revolution against the Tsar came in 19"," 05",[" 25"," 48"," 61"]),
 (1917,"The Bolshevik leader of the 1917 October Revolution was Vladimir"," Lenin",[" Kerensky"," Martov"," Plekhanov"]),
 (1924,"The dictator who took over the USSR after Lenin was Joseph"," Stalin",[" Bukharin"," Zinoviev"," Kamenev"," Rykov"]),
 (1956,"The Soviet leader who denounced Stalin in 1956 was Nikita"," Khrushchev",[" Malenkov"," Bulganin"," Molotov"]),
 (1985,"The Soviet leader of perestroika and glasnost was Mikhail"," Gorbachev",[" Ligachev"," Ryzhkov"," Shevardnadze"]),
 (1991,"The first president of post-Soviet Russia was Boris"," Yeltsin",[" Gaidar"," Rutskoi"," Chernomyrdin"]),
 (2000,"The Russian president from 2000, a former KGB officer, was Vladimir"," Putin",[" Medvedev"," Primakov"," Stepashin"]),
 (1964,"The Soviet leader of the 'stagnation' era from 1964 was Leonid"," Brezhnev",[" Kosygin"," Podgorny"," Suslov"]),
 (1943,"The Soviet city of the decisive 1942-43 battle was"," Stalingrad",[" Smolensk"," Kharkov"," Voronezh"," Rostov"]),
])

add("West", "culture", [
 (1600,"The Elizabethan playwright who wrote 'Hamlet' was William"," Shakespeare",[" Marlowe"," Jonson"," Webster"," Dryden"]),
 (1808,"The German composer of the Fifth and Ninth Symphonies was Ludwig van"," Beethoven",[" Brahms"," Schubert"," Handel"]),
 (1787,"The Austrian composer of 'The Magic Flute' was Wolfgang Amadeus"," Mozart",[" Haydn"," Schubert"," Gluck"," Salieri"]),
 (1860,"The Victorian novelist who wrote 'Oliver Twist' was Charles"," Dickens",[" Thackeray"," Trollope"," Hardy"," Collins"]),
 (1862,"The French author of 'Les Misérables' was Victor"," Hugo",[" Balzac"," Zola"," Dumas"," Flaubert"]),
 (1872,"The French Impressionist painter of water lilies was Claude"," Monet",[" Manet"," Degas"," Renoir"," Pissarro"]),
 (1869,"The Russian author of 'War and Peace' was Leo"," Tolstoy",[" Turgenev"," Gogol"," Chekhov"," Gorky"]),
 (1866,"The Russian author of 'Crime and Punishment' was Fyodor"," Dostoevsky",[" Turgenev"," Goncharov"," Leskov"]),
 (1851,"The Italian composer of 'Aida' and 'Rigoletto' was Giuseppe"," Verdi",[" Rossini"," Bellini"," Donizetti"]),
 (1808,"The German author of 'Faust' was Johann Wolfgang von"," Goethe",[" Schiller"," Heine"," Lessing"," Herder"]),
 (1512,"The artist who painted the Sistine Chapel ceiling was"," Michelangelo",[" Raphael"," Botticelli"," Titian"," Caravaggio"]),
 (1605,"The Spanish author of 'Don Quixote' was Miguel de"," Cervantes",[" Quevedo"," Góngora"," Calderón"]),
 (1810,"The Spanish painter of 'The Third of May 1808' was Francisco"," Goya",[" Velázquez"," Murillo"," Ribera"]),
 (1876,"The German composer of the 'Ring' cycle of operas was Richard"," Wagner",[" Strauss"," Mahler"," Bruckner"," Brahms"]),
 (1956,"The American singer crowned 'King of Rock and Roll' was Elvis"," Presley",[" Holly"," Domino"," Cochran"]),
 (1962,"The Liverpool band of Lennon and McCartney was The"," Beatles",[" Searchers"," Hollies"," Animals"," Kinks"]),
 (1982,"The American singer called the 'King of Pop' was Michael"," Jackson",[" Prince"," Wonder"," Richie"]),
 (1972,"Coppola's 1972 Mafia film epic was The"," Godfather",[" Untouchables"," Goodfellas"," Departed"]),
 (1937,"The Cubist painter of 'Guernica' (1937) was Pablo"," Picasso",[" Braque"," Gris"," Miró"," Léger"]),
 (1931,"The Surrealist painter of melting clocks was Salvador"," Dalí",[" Miró"," Ernst"," Magritte"," Tanguy"]),
 (1943,"The French existentialist of 'Being and Nothingness' was Jean-Paul"," Sartre",[" Aron"," Camus"," Merleau"]),
 (1942,"The French author of 'The Stranger' was Albert"," Camus",[" Sartre"," Malraux"," Gide"," Mauriac"]),
 (1960,"The Italian director of 'La Dolce Vita' was Federico"," Fellini",[" Visconti"," Antonioni"," Rossellini"]),
 (1937,"The Soviet composer of the 1937 Fifth Symphony was Dmitri"," Shostakovich",[" Khachaturian"," Kabalevsky"," Myaskovsky"]),
 (1997,"J.K. Rowling's boy wizard, introduced in 1997, was Harry"," Potter",[" Wotter"," Porter"," Trotter"]),
 (1949,"The British author of the dystopian novel '1984' was George"," Orwell",[" Huxley"," Koestler"," Greene"]),
 (1990,"The Italian operatic tenor of the 1990 'Three Tenors' was Luciano"," Pavarotti",[" Domingo"," Carreras"," Gigli"]),
 (1988,"The Spanish film director of 'Women on the Verge' was Pedro"," Almodóvar",[" Saura"," Buñuel"," Amenábar"]),
])

add("West", "science", [
 (1859,"The naturalist who wrote 'On the Origin of Species' was Charles"," Darwin",[" Owen"," Huxley"," Wallace"," Lyell"]),
 (1687,"The English physicist of universal gravitation was Isaac"," Newton",[" Hooke"," Halley"," Boyle"," Cavendish"]),
 (1865,"The French chemist behind pasteurization and vaccines was Louis"," Pasteur",[" Lavoisier"," Bernard"," Berthelot"]),
 (1898,"The Polish-French discoverer of radium was Marie"," Curie",[" Becquerel"," Meitner"," Joliot"]),
 (1869,"The Russian who devised the periodic table was Dmitri"," Mendeleev",[" Butlerov"," Lomonosov"," Beilstein"]),
 (1895,"The German who discovered X-rays in 1895 was Wilhelm"," Röntgen",[" Hertz"," Lenard"," Crookes"," Geiger"]),
 (1800,"The Italian who invented the electric battery in 1800 was Alessandro"," Volta",[" Galvani"," Avogadro"," Fermi"]),
 (1900,"The German founder of quantum theory in 1900 was Max"," Planck",[" Sommerfeld"," Born"," Hertz"]),
 (1831,"The English pioneer of electromagnetic induction was Michael"," Faraday",[" Maxwell"," Joule"," Ohm"]),
 (1900,"The Viennese founder of psychoanalysis was Sigmund"," Freud",[" Breuer"," Adler"," Janet"]),
 (1879,"The American inventor of the practical light bulb was Thomas"," Edison",[" Tesla"," Westinghouse"," Swan"]),
 (1919,"The physicist whose general relativity was confirmed in 1919 was Albert"," Einstein",[" Planck"," Lorentz"," Eddington"," Bohr"]),
 (1928,"The Scot who discovered penicillin in 1928 was Alexander"," Fleming",[" Florey"," Chain"," Lister"," Domagk"]),
 (1953,"The structure of DNA was discovered in 1953 by Watson and"," Crick",[" Wilkins"," Franklin"," Pauling"," Chargaff"]),
 (1947,"The device invented at Bell Labs in 1947 that replaced the vacuum tube was the"," transistor",[" capacitor"," magnetron"," thyristor"," klystron"]),
 (1942,"The physicist who built the first nuclear reactor in 1942 was Enrico"," Fermi",[" Rubbia"," Segrè"," Pontecorvo"]),
 (1957,"The first artificial satellite, launched in 1957, was"," Sputnik",[" Explorer"," Vanguard"," Soyuz"]),
 (1961,"The first human in space, in 1961, was Yuri"," Gagarin",[" Titov"," Leonov"," Komarov"]),
 (1969,"The first man to walk on the Moon, in 1969, was Neil"," Armstrong",[" Aldrin"," Glenn"," Collins"," Shepard"]),
 (1936,"The English mathematician who formalized computation in the 1930s was Alan"," Turing",[" Babbage"," Shannon"," Church"," Newman"]),
 (1988,"The British physicist of 'A Brief History of Time' was Stephen"," Hawking",[" Penrose"," Dyson"," Bohm"]),
 (1955,"The American who developed the 1955 polio vaccine was Jonas"," Salk",[" Sabin"," Enders"," Pasteur"]),
 (1938,"The 1938 splitting of the uranium atom was achieved by Otto"," Hahn",[" Strassmann"," Bohr"," Frisch"]),
 (1932,"The German physicist awarded the 1932 Nobel for quantum mechanics was Werner"," Heisenberg",[" Schrödinger"," Born"," Jordan"," Pauli"]),
 (1986,"The Soviet nuclear reactor that exploded in 1986 was"," Chernobyl",[" Mayak"," Kyshtym"," Beloyarsk"]),
])


def toklen(tok, s):
    return len(tok(s, add_special_tokens=False).input_ids)


def sumlogp(model, tok, prompt, answer):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        lp = torch.log_softmax(model(torch.tensor([full], device=model.device)).logits[0].float(), dim=-1)
    return sum(lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))


def main():
    print(f"battery size: {len(B)} items", flush=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["model", "cutoff", "country", "domain", "era", "item", "true", "false", "matched", "gap", "correct"])
    for M in MODELS:
        cut = int(re.search(r"Ranke-4B-(\d{4})", M).group(1))
        print("#### " + M, flush=True)
        tok = AutoTokenizer.from_pretrained(M)
        model = AutoModelForCausalLM.from_pretrained(M, dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE).eval()
        with open(OUT, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for i, (country, domain, year, prompt, t, fs) in enumerate(B):
                nt = toklen(tok, t)
                cand = [fa for fa in fs if toklen(tok, fa) == nt]
                matched = bool(cand)
                fa = cand[0] if cand else min(fs, key=lambda x: abs(toklen(tok, x) - nt))
                g = sumlogp(model, tok, prompt, t) - sumlogp(model, tok, prompt, fa)
                era = "pre" if year <= cut else "post"
                w.writerow([M.split("/")[-1], cut, country, domain, era, i, t.strip(), fa.strip(),
                            int(matched), round(g, 4), int(g > 0)])
        del model
        print("  done", flush=True)
    print("ALL DONE ->", OUT)


if __name__ == "__main__":
    main()
