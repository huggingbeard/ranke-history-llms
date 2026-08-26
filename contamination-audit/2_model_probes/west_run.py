"""
Western-focused contamination battery (bigger, for statistical power).
Countries: USA, UK, France, Germany, Italy, Spain, Russia/USSR.
Domains: politics, war, culture, science (light). Dates span ~1789-2012 so each
cutoff model gets pre- and post-cutoff items. Minimal pairs (true vs matched false).

Scores each item on all five models, 2 forward passes/item (true & false), and
writes west_results.csv with the same columns as the main pipeline + 'language'.
"""
import os, re, csv, math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
torch.set_num_threads(os.cpu_count() or 8)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
print(f"[device] {DEVICE}  dtype={DTYPE}", flush=True)

MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929",
          "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\west_results.csv"

# (language, country, domain, year, prompt, TRUE, FALSE)
W = [
 # ================= USA (English) =================
 ("English","USA","politics",1861,"The U.S. President during the Civil War was Abraham"," Lincoln"," Douglas"),
 ("English","USA","politics",1865,"Lincoln was assassinated in 1865 by John Wilkes"," Booth"," Surratt"),
 ("English","USA","politics",1869,"The Union general who became President in 1869 was Ulysses S."," Grant"," Sherman"),
 ("English","USA","politics",1901,"The U.S. President assassinated in 1901 was William"," McKinley"," Garfield"),
 ("English","USA","politics",1904,"The 'Square Deal' President, in office 1901-09, was Theodore"," Roosevelt"," Bryan"),
 ("English","USA","politics",1917,"The U.S. President for most of the First World War was Woodrow"," Wilson"," Hughes"),
 ("English","USA","politics",1920,"The Republican elected President in 1920, 'return to normalcy', was Warren"," Harding"," Cox"),
 ("English","USA","politics",1933,"The President who launched the New Deal in the 1930s was Franklin"," Roosevelt"," Landon"),
 ("English","USA","politics",1945,"The President who ordered the atomic bombings of Japan was Harry"," Truman"," Dewey"),
 ("English","USA","politics",1953,"The D-Day general who became President in 1953 was Dwight"," Eisenhower"," Stevenson"),
 ("English","USA","politics",1961,"The President inaugurated in 1961, youngest elected, was John F."," Kennedy"," Humphrey"),
 ("English","USA","politics",1963,"After Kennedy's assassination the Presidency passed to Lyndon"," Johnson"," Goldwater"),
 ("English","USA","politics",1974,"The President who resigned over Watergate in 1974 was Richard"," Nixon"," Agnew"),
 ("English","USA","politics",1977,"The Democrat elected President in 1976, a peanut farmer, was Jimmy"," Carter"," Mondale"),
 ("English","USA","politics",1981,"The former actor elected President in 1980 was Ronald"," Reagan"," Anderson"),
 ("English","USA","politics",1993,"The President elected in 1992 from Arkansas was Bill"," Clinton"," Dukakis"),
 ("English","USA","politics",2009,"The first African-American President, elected 2008, was Barack"," Obama"," Edwards"),
 ("English","USA","politics",2017,"The businessman elected U.S. President in 2016 was Donald"," Trump"," Cruz"),
 ("English","USA","war",1941,"The 1941 Japanese attack bringing the U.S. into WWII struck Pearl"," Harbor"," Cove"),
 ("English","USA","war",1962,"The 1962 nuclear standoff with the USSR over missiles was the Cuban Missile"," Crisis"," Standoff"),
 ("English","USA","war",2001,"The towers destroyed on September 11, 2001 were the World Trade"," Center"," Exchange"),
 ("English","USA","politics",1963,"The civil-rights leader of the 'I Have a Dream' speech was Martin Luther"," King"," Abernathy"),
 ("English","USA","culture",1956,"The 1950s singer crowned 'King of Rock and Roll' was Elvis"," Presley"," Holly"),
 ("English","USA","culture",1977,"George Lucas's 1977 space saga was called Star"," Wars"," Voyager"),
 ("English","USA","culture",1982,"The American singer called the 'King of Pop' was Michael"," Jackson"," Prince"),
 ("English","USA","culture",1928,"The Disney cartoon mouse introduced in 1928 was named"," Mickey"," Oswald"),
 ("English","USA","culture",1939,"The 1939 film of a Kansas girl and a tornado was The Wizard of"," Oz"," Id"),
 ("English","USA","culture",1925,"Fitzgerald's 1925 novel of the Jazz Age was The Great"," Gatsby"," Babbitt"),
 ("English","USA","science",1879,"The American who made the first practical light bulb was Thomas"," Edison"," Tesla"),
 ("English","USA","science",1903,"The brothers who made the first powered flight in 1903 were the"," Wright"," Curtiss"),
 ("English","USA","science",1953,"The American co-discoverer of DNA's structure (1953) was James"," Watson"," Pauling"),
 ("English","USA","science",1955,"The American who developed the 1955 polio vaccine was Jonas"," Salk"," Sabin"),
 ("English","USA","space",1969,"The first man on the Moon, in 1969, was Neil"," Armstrong"," Glenn"),
 ("English","USA","sport",1974,"The boxer, formerly Cassius Clay, who beat Foreman in 1974 was Muhammad"," Ali"," Frazier"),
 ("English","USA","tech",1976,"The company Steve Jobs co-founded in 1976 was"," Apple"," Atari"),
 ("English","USA","tech",2004,"The social network Mark Zuckerberg launched in 2004 was"," Facebook"," Friendster"),
 # ================= UK (English) =================
 ("English","UK","politics",1837,"The queen who reigned over Britain for most of the 19th century was Queen"," Victoria"," Adelaide"),
 ("English","UK","politics",1868,"The Liberal rival of Disraeli, four-time PM, was William"," Gladstone"," Palmerston"),
 ("English","UK","politics",1874,"Victoria's favourite Conservative prime minister was Benjamin"," Disraeli"," Salisbury"),
 ("English","UK","politics",1940,"The PM for most of the Second World War was Winston"," Churchill"," Halifax"),
 ("English","UK","politics",1945,"The Labour PM who won the 1945 election was Clement"," Attlee"," Morrison"),
 ("English","UK","politics",1979,"Britain's 'Iron Lady' PM, elected 1979, was Margaret"," Thatcher"," Williams"),
 ("English","UK","politics",1997,"The Labour PM elected in 1997 was Tony"," Blair"," Smith"),
 ("English","UK","politics",1936,"The British king who abdicated in 1936 was Edward"," VIII"," VII"),
 ("English","UK","politics",1953,"The British queen crowned in 1953 was Elizabeth"," II"," I"),
 ("English","UK","war",1916,"The 1916 Dublin rising against British rule was the Easter"," Rising"," Revolt"),
 ("English","UK","war",1915,"The Cunard liner sunk by a U-boat in 1915 was the"," Lusitania"," Aquitania"),
 ("English","UK","war",1982,"In 1982 Britain fought Argentina over the"," Falklands"," Hebrides"),
 ("English","UK","science",1859,"The naturalist who wrote On the Origin of Species was Charles"," Darwin"," Owen"),
 ("English","UK","science",1928,"The Scot who discovered penicillin in 1928 was Alexander"," Fleming"," Florey"),
 ("English","UK","science",1940,"The mathematician who broke Enigma at Bletchley was Alan"," Turing"," Welchman"),
 ("English","UK","science",1989,"The Englishman who invented the World Wide Web in 1989 was Tim"," Berners-Lee"," Cerf"),
 ("English","UK","culture",1600,"The Elizabethan playwright of Hamlet was William"," Shakespeare"," Marlowe"),
 ("English","UK","culture",1860,"The Victorian novelist of Oliver Twist was Charles"," Dickens"," Thackeray"),
 ("English","UK","culture",1962,"The Liverpool band of Lennon and McCartney was The"," Beatles"," Searchers"),
 ("English","UK","culture",1965,"The London rock band of Jagger and Richards was The Rolling"," Stones"," Kinks"),
 ("English","UK","culture",1962,"Ian Fleming's secret agent, code 007, was James"," Bond"," Drummond"),
 ("English","UK","culture",1997,"J.K. Rowling's boy wizard, introduced in 1997, was Harry"," Potter"," Wotter"),
 ("English","UK","culture",1969,"The absurdist BBC comedy troupe of 1969 was Monty"," Python"," Cobra"),
 # ================= France (French) =================
 ("French","France","politics",1804,"The general who crowned himself Emperor of France in 1804 was Napoleon"," Bonaparte"," Bernadotte"),
 ("French","France","politics",1870,"The emperor who fell after the 1870 defeat by Prussia was Napoleon"," III"," II"),
 ("French","France","politics",1894,"The army officer wrongly convicted of treason in 1894 was Alfred"," Dreyfus"," Esterhazy"),
 ("French","France","politics",1940,"The general who led the Free French and founded the Fifth Republic was Charles de"," Gaulle"," Lattre"),
 ("French","France","politics",1940,"The WWI marshal who led the Vichy regime was Philippe"," Pétain"," Weygand"),
 ("French","France","politics",1981,"The first Socialist president of the Fifth Republic was François"," Mitterrand"," Rocard"),
 ("French","France","politics",1995,"The Gaullist elected French president in 1995 was Jacques"," Chirac"," Jospin"),
 ("French","France","politics",2017,"The youngest French president, elected 2017, was Emmanuel"," Macron"," Hamon"),
 ("French","France","war",1916,"The longest battle of WWI, at a French fortress in 1916, was the Battle of"," Verdun"," Sedan"),
 ("French","France","war",1919,"The 1919 treaty ending WWI was signed at"," Versailles"," Trianon"),
 ("French","France","war",1944,"The 1944 Allied D-Day landings were in"," Normandy"," Brittany"),
 ("French","France","politics",1968,"France's wave of student-worker protests came in May 19"," 68"," 48"),
 ("French","France","science",1898,"The Polish-French discoverer of radium was Marie"," Curie"," Joliot"),
 ("French","France","science",1865,"The chemist behind pasteurization and vaccines was Louis"," Pasteur"," Lamarck"),
 ("French","France","culture",1862,"The author of Les Misérables was Victor"," Hugo"," Balzac"),
 ("French","France","culture",1872,"The Impressionist painter of water lilies was Claude"," Monet"," Degas"),
 ("French","France","culture",1913,"The author of 'In Search of Lost Time' was Marcel"," Proust"," Gide"),
 ("French","France","culture",1943,"The existentialist of 'Being and Nothingness' was Jean-Paul"," Sartre"," Aron"),
 ("French","France","culture",1942,"The author of 'The Stranger', Nobel 1957, was Albert"," Camus"," Malraux"),
 ("French","France","culture",1950,"The French chanteuse of 'La Vie en rose' was Édith"," Piaf"," Greco"),
 ("French","France","tech",1969,"The supersonic airliner built by Britain and France was the"," Concorde"," Caravelle"),
 # ================= Germany / Austria (German) =================
 ("German","Germany","politics",1871,"The 'Iron Chancellor' who unified Germany in 1871 was Otto von"," Bismarck"," Moltke"),
 ("German","Germany","politics",1888,"The Kaiser who reigned through WWI was Wilhelm"," II"," I"),
 ("German","Germany","politics",1933,"The Nazi who became German chancellor in 1933 was Adolf"," Hitler"," Hugenberg"),
 ("German","Germany","politics",1949,"The first chancellor of West Germany, from 1949, was Konrad"," Adenauer"," Schumacher"),
 ("German","Germany","politics",1969,"The West German chancellor of Ostpolitik, Nobel 1971, was Willy"," Brandt"," Kiesinger"),
 ("German","Germany","politics",1990,"The chancellor who oversaw German reunification was Helmut"," Kohl"," Lafontaine"),
 ("German","Germany","politics",2005,"The first woman German chancellor, from 2005, was Angela"," Merkel"," von der Leyen"),
 ("German","Germany","war",1941,"The 1941 German invasion of the USSR was Operation"," Barbarossa"," Sealion"),
 ("German","Germany","war",1943,"The turning-point 1942-43 battle on the Volga was at"," Stalingrad"," Kursk"),
 ("German","Germany","war",1945,"The largest Nazi extermination camp, in Poland, was"," Auschwitz"," Ravensbrück"),
 ("German","Germany","politics",1961,"The wall dividing a German city, built 1961, was the Berlin"," Wall"," Gate"),
 ("German","Germany","politics",1989,"The Berlin Wall came down in 19"," 89"," 79"),
 ("German","Germany","science",1900,"The physicist who founded quantum theory in 1900 was Max"," Planck"," Sommerfeld"),
 ("German","Germany","science",1895,"The physicist who discovered X-rays in 1895 was Wilhelm"," Röntgen"," Hertz"),
 ("German","Germany","science",1944,"The rocket engineer behind the V-2 and later NASA was Wernher von"," Braun"," Dornberger"),
 ("German","Austria","science",1900,"The Viennese founder of psychoanalysis was Sigmund"," Freud"," Breuer"),
 ("German","Austria","politics",1914,"The archduke whose 1914 assassination sparked WWI was Franz"," Ferdinand"," Karl"),
 ("German","Germany","culture",1808,"The composer of the Fifth and Ninth Symphonies was Ludwig van"," Beethoven"," Brahms"),
 ("German","Germany","culture",1808,"The author of 'Faust' was Johann Wolfgang von"," Goethe"," Schiller"),
 ("German","Germany","culture",1876,"The composer of the 'Ring' cycle was Richard"," Wagner"," Strauss"),
 ("German","Germany","culture",1901,"The author of 'Buddenbrooks', Nobel 1929, was Thomas"," Mann"," Hesse"),
 ("German","Germany","culture",1919,"The influential German design school founded in 1919 was the"," Bauhaus"," Werkbund"),
 # ================= Italy (Italian) =================
 ("Italian","Italy","politics",1860,"The general whose Redshirts helped unify Italy was Giuseppe"," Garibaldi"," Mazzini"),
 ("Italian","Italy","politics",1861,"The statesman who became Italy's first PM in 1861 was Camillo di"," Cavour"," Crispi"),
 ("Italian","Italy","politics",1922,"The Fascist who marched on Rome in 1922 was Benito"," Mussolini"," Balbo"),
 ("Italian","Italy","politics",1978,"The ex-PM kidnapped and killed by the Red Brigades in 1978 was Aldo"," Moro"," Andreotti"),
 ("Italian","Italy","politics",1994,"The media-tycoon PM of the 1990s-2000s was Silvio"," Berlusconi"," Prodi"),
 ("Italian","Italy","politics",1946,"Italy abolished the monarchy and became a republic in 19"," 46"," 24"),
 ("Italian","Italy","science",1901,"The Italian who sent the first transatlantic radio signal (1901) was Guglielmo"," Marconi"," Meucci"),
 ("Italian","Italy","science",1942,"The physicist of the first nuclear reactor (1942) was Enrico"," Fermi"," Rubbia"),
 ("Italian","Italy","science",1610,"The astronomer tried by the Inquisition for heliocentrism was Galileo"," Galilei"," Bruno"),
 ("Italian","Italy","culture",1308,"The medieval poet of the Divine Comedy was Dante"," Alighieri"," Petrarch"),
 ("Italian","Italy","culture",1503,"The Renaissance painter of the Mona Lisa was Leonardo da"," Vinci"," Botticelli"),
 ("Italian","Italy","culture",1851,"The composer of 'Rigoletto' and 'Aida' was Giuseppe"," Verdi"," Rossini"),
 ("Italian","Italy","culture",1900,"The composer of 'La Bohème' and 'Tosca' was Giacomo"," Puccini"," Mascagni"),
 ("Italian","Italy","culture",1960,"The director of 'La Dolce Vita' (1960) was Federico"," Fellini"," Visconti"),
 ("Italian","Italy","culture",1990,"The operatic tenor of the 1990 'Three Tenors', heavyset, was Luciano"," Pavarotti"," Domingo"),
 ("Italian","Italy","sport",1950,"The Italian sports-car marque founded by Enzo is"," Ferrari"," Maserati"),
 ("Italian","Italy","religion",1962,"The Church council that modernized Catholicism (1962-65) was the Second Vatican"," Council"," Synod"),
 ("Italian","Italy","religion",1978,"The Polish-born pope elected in 1978 took the name John Paul"," II"," I"),
 # ================= Spain (Spanish) =================
 ("Spanish","Spain","politics",1936,"The general who won the Spanish Civil War and ruled to 1975 was Francisco"," Franco"," Mola"),
 ("Spanish","Spain","war",1937,"The Basque town bombed in 1937, painted by Picasso, was"," Guernica"," Bilbao"),
 ("Spanish","Spain","politics",1975,"The king who restored democracy after Franco's death in 1975 was Juan"," Carlos"," Felipe"),
 ("Spanish","Spain","politics",1492,"The monarchs who funded Columbus and unified Spain were Ferdinand and"," Isabella"," Joanna"),
 ("Spanish","Spain","culture",1605,"The author of 'Don Quixote' was Miguel de"," Cervantes"," Lope"),
 ("Spanish","Spain","culture",1810,"The court painter of 'The Third of May 1808' was Francisco"," Goya"," Velázquez"),
 ("Spanish","Spain","culture",1937,"The Cubist painter of 'Guernica' was Pablo"," Picasso"," Miró"),
 ("Spanish","Spain","culture",1930,"The Surrealist painter of melting clocks was Salvador"," Dalí"," Gris"),
 ("Spanish","Spain","culture",1883,"The architect of Barcelona's Sagrada Família was Antoni"," Gaudí"," Domènech"),
 ("Spanish","Spain","sport",2010,"Spain won its first football World Cup in 20"," 10"," 06"),
 ("Spanish","Spain","politics",1898,"In 1898 Spain lost Cuba and the Philippines in a war with the"," United States"," British"),
 # ================= Russia / USSR (Russian) =================
 ("Russian","Russia","politics",1917,"The last Tsar, who abdicated in 1917, was Nicholas"," II"," I"),
 ("Russian","Russia","politics",1917,"The Bolshevik leader of the 1917 October Revolution was Vladimir"," Lenin"," Kerensky"),
 ("Russian","Russia","politics",1924,"The dictator who took over the USSR after Lenin was Joseph"," Stalin"," Bukharin"),
 ("Russian","Russia","politics",1956,"The Soviet leader who denounced Stalin in 1956 was Nikita"," Khrushchev"," Malenkov"),
 ("Russian","Russia","politics",1964,"The Soviet leader of the 'stagnation' era from 1964 was Leonid"," Brezhnev"," Kosygin"),
 ("Russian","Russia","politics",1985,"The Soviet leader of perestroika, last of the USSR, was Mikhail"," Gorbachev"," Ligachev"),
 ("Russian","Russia","politics",1991,"The first president of post-Soviet Russia was Boris"," Yeltsin"," Gaidar"),
 ("Russian","Russia","politics",2000,"The Russian president from 2000, ex-KGB, was Vladimir"," Putin"," Medvedev"),
 ("Russian","Russia","war",1943,"The Soviet city of the decisive 1942-43 battle was"," Stalingrad"," Smolensk"),
 ("Russian","Russia","space",1957,"The first artificial satellite, launched by the USSR in 1957, was"," Sputnik"," Soyuz"),
 ("Russian","Russia","space",1961,"The first human in space, a Soviet cosmonaut (1961), was Yuri"," Gagarin"," Titov"),
 ("Russian","Russia","science",1986,"The Soviet reactor that exploded in 1986 in Ukraine was"," Chernobyl"," Mayak"),
 ("Russian","Russia","science",1869,"The Russian who devised the periodic table was Dmitri"," Mendeleev"," Butlerov"),
 ("Russian","Russia","culture",1869,"The author of 'War and Peace' was Leo"," Tolstoy"," Turgenev"),
 ("Russian","Russia","culture",1866,"The author of 'Crime and Punishment' was Fyodor"," Dostoevsky"," Gogol"),
 ("Russian","Russia","culture",1880,"The composer of the '1812 Overture' was Pyotr"," Tchaikovsky"," Borodin"),
 ("Russian","Russia","culture",1913,"The composer of 'The Rite of Spring' (1913) was Igor"," Stravinsky"," Prokofiev"),
 ("Russian","Russia","culture",1957,"The author of 'Doctor Zhivago', 1958 Nobel, was Boris"," Pasternak"," Sholokhov"),
]

# ---- Batch 2: more per-country politics + culture + science (for power) ----
W += [
 # USA
 ("English","USA","politics",1776,"The principal author of the 1776 Declaration of Independence was Thomas"," Jefferson"," Adams"),
 ("English","USA","politics",1789,"The first President of the United States was George"," Washington"," Adams"),
 ("English","USA","politics",1823,"The 1823 doctrine warning Europe off the Americas is named after President"," Monroe"," Madison"),
 ("English","USA","politics",1863,"Lincoln's famous 1863 wartime address was delivered at"," Gettysburg"," Vicksburg"),
 ("English","USA","tech",1908,"Henry Ford's affordable car of 1908 was the Model"," T"," A"),
 ("English","USA","politics",1954,"The 1954 Supreme Court case ending school segregation was Brown v. Board of"," Education"," Trustees"),
 ("English","USA","politics",1955,"The woman whose 1955 bus protest sparked the Montgomery boycott was Rosa"," Parks"," Baker"),
 ("English","USA","politics",1950,"The senator behind the early-1950s anti-communist hearings was Joseph"," McCarthy"," Mundt"),
 ("English","USA","war",1965,"The Southeast Asian country of America's long 1960s war was"," Vietnam"," Korea"),
 ("English","USA","politics",1963,"Kennedy was shot in 1963 in the Texas city of"," Dallas"," Houston"),
 ("English","USA","politics",1968,"The civil-rights leader assassinated in Memphis in 1968 was Martin Luther"," King"," Evers"),
 ("English","USA","culture",1939,"Steinbeck's 1939 novel of Dust Bowl migrants was The Grapes of"," Wrath"," Eden"),
 ("English","USA","culture",1962,"The pop artist of the Campbell's Soup cans was Andy"," Warhol"," Lichtenstein"),
 ("English","USA","culture",1972,"Coppola's 1972 Mafia epic was The"," Godfather"," Untouchables"),
 ("English","USA","culture",1975,"Spielberg's 1975 shark blockbuster was"," Jaws"," Orca"),
 ("English","USA","culture",1953,"The American screen icon Marilyn was surnamed"," Monroe"," Mansfield"),
 ("English","USA","culture",1965,"The American folk-rock songwriter who 'went electric' in 1965 was Bob"," Dylan"," Baez"),
 ("English","USA","science",1945,"The physicist who led the Manhattan Project laboratory was Robert"," Oppenheimer"," Lawrence"),
 ("English","USA","science",1947,"The amplifying device invented at Bell Labs in 1947 was the"," transistor"," capacitor"),
 ("English","USA","tech",1969,"The 1969 U.S. military computer network, ancestor of the internet, was"," ARPANET"," CYCLADES"),
 ("English","USA","science",1960,"The first working laser was built in 1960 by Theodore"," Maiman"," Townes"),
 ("English","USA","politics",2021,"The U.S. president inaugurated in January 2021 was Joe"," Biden"," Sanders"),
 # UK
 ("English","UK","politics",1707,"The 1707 Act of Union joined England with"," Scotland"," Ireland"),
 ("English","UK","politics",1815,"The British commander who beat Napoleon at Waterloo was the Duke of"," Wellington"," Marlborough"),
 ("English","UK","war",1854,"The 1854-56 war against Russia in the Black Sea was the"," Crimean"," Boer"),
 ("English","UK","war",1956,"The 1956 crisis when Britain and France seized the canal was the"," Suez"," Aden"),
 ("English","UK","politics",1948,"Britain's tax-funded health service, founded 1948, is the"," NHS"," BUPA"),
 ("English","UK","politics",2016,"Britain's 2016 referendum to leave the EU was nicknamed"," Brexit"," Bremain"),
 ("English","UK","politics",1990,"The Conservative who succeeded Thatcher as PM in 1990 was John"," Major"," Hurd"),
 ("English","UK","politics",2010,"The Conservative who became PM in a 2010 coalition was David"," Cameron"," Clegg"),
 ("English","UK","politics",1964,"The Labour PM elected in 1964, 'white heat of technology', was Harold"," Wilson"," Gaitskell"),
 ("English","UK","culture",1843,"Dickens's 1843 Christmas miser was Ebenezer"," Scrooge"," Marley"),
 ("English","UK","culture",1865,"Lewis Carroll's 1865 heroine down the rabbit hole was"," Alice"," Wendy"),
 ("English","UK","culture",1994,"The Britpop band of the Gallagher brothers was"," Oasis"," Blur"),
 ("English","UK","science",1687,"The English physicist of universal gravitation was Isaac"," Newton"," Hooke"),
 ("English","UK","science",1953,"The British co-discoverer of DNA's structure, with Watson, was Francis"," Crick"," Wilkins"),
 ("English","UK","science",1988,"The British physicist of 'A Brief History of Time' was Stephen"," Hawking"," Penrose"),
 # France
 ("French","France","politics",1789,"The Paris prison stormed on 14 July 1789 was the"," Bastille"," Conciergerie"),
 ("French","France","politics",1799,"Napoleon seized power in the 1799 coup of 18"," Brumaire"," Fructidor"),
 ("French","France","culture",1830,"The author of 'The Red and the Black' (1830) was"," Stendhal"," Flaubert"),
 ("French","France","culture",1857,"Flaubert's 1857 novel of an adulterous doctor's wife was Madame"," Bovary"," Arnoux"),
 ("French","France","war",1954,"France's 1954 defeat in Indochina came at Dien Bien"," Phu"," Khe"),
 ("French","France","war",1962,"The North African colony that won independence from France in 1962 was"," Algeria"," Morocco"),
 ("French","France","politics",1958,"The republic de Gaulle founded in 1958 was the"," Fifth"," Fourth"),
 ("French","France","politics",2007,"The French president elected in 2007, 'Sarko', was Nicolas"," Sarkozy"," Royal"),
 ("French","France","politics",2012,"The Socialist French president elected in 2012 was François"," Hollande"," Fabius"),
 ("French","France","culture",1959,"Truffaut's 1959 New Wave debut was 'The 400'"," Blows"," Nights"),
 ("French","France","culture",1949,"The author of 'The Second Sex' (1949) was Simone de"," Beauvoir"," Staël"),
 ("French","France","science",1935,"Marie Curie's daughter, Nobel laureate 1935, was Irène Joliot-"," Curie"," Langevin"),
 ("French","France","tech",1981,"France's high-speed train, launched 1981, is the"," TGV"," ICE"),
 # Germany / Austria
 ("German","Germany","politics",1848,"The 1848 liberal German parliament met in"," Frankfurt"," Vienna"),
 ("German","Germany","science",1919,"The physicist whose general relativity was confirmed in 1919 was Albert"," Einstein"," Planck"),
 ("German","Germany","politics",1919,"The republic founded in Germany after the First World War was the"," Weimar"," Bonn"),
 ("German","Germany","politics",1933,"The 1933 fire Hitler used to seize emergency powers was the"," Reichstag"," Bundestag"),
 ("German","Germany","war",1938,"The 1938 Nazi pogrom against Jews was called"," Kristallnacht"," Anschluss"),
 ("German","Germany","sport",1936,"The Nazi-hosted Olympic Games of 1936 were held in"," Berlin"," Munich"),
 ("German","Germany","politics",1946,"The 1945-46 trials of Nazi leaders were held in"," Nuremberg"," Potsdam"),
 ("German","Germany","politics",1974,"The West German chancellor 1974-82, the crisis manager, was Helmut"," Schmidt"," Genscher"),
 ("German","Germany","politics",1998,"The SPD chancellor elected in 1998 was Gerhard"," Schröder"," Stoiber"),
 ("German","Germany","politics",2021,"The SPD chancellor who succeeded Merkel in 2021 was Olaf"," Scholz"," Laschet"),
 ("German","Germany","science",1938,"The 1938 splitting of the uranium atom by Otto Hahn was nuclear"," fission"," fusion"),
 # Italy
 ("Italian","Italy","culture",1508,"The artist who painted the Sistine Chapel ceiling was"," Michelangelo"," Raphael"),
 ("Italian","Italy","science",1800,"The Italian who invented the electric battery in 1800 was Alessandro"," Volta"," Galvani"),
 ("Italian","Italy","politics",1870,"Italian unification was completed in 1870 with the capture of"," Rome"," Venice"),
 ("Italian","Italy","politics",1945,"The Christian Democrat who led postwar Italy from 1945 was Alcide De"," Gasperi"," Nenni"),
 ("Italian","Italy","politics",1983,"Italy's first Socialist prime minister, from 1983, was Bettino"," Craxi"," Fanfani"),
 ("Italian","Italy","politics",2014,"The young centre-left Italian PM from 2014 was Matteo"," Renzi"," Letta"),
 ("Italian","Italy","culture",1966,"The Italian composer of spaghetti-western scores was Ennio"," Morricone"," Rota"),
 ("Italian","Italy","culture",1960,"The Italian screen icon of the 1960s, 'La Loren', was Sophia"," Loren"," Lollobrigida"),
 ("Italian","Italy","culture",1980,"The Italian author of 'The Name of the Rose' (1980) was Umberto"," Eco"," Calvino"),
 ("Italian","Italy","sport",2006,"Italy most recently won the football World Cup in 20"," 06"," 02"),
 # Spain
 ("Spanish","Spain","war",1588,"The fleet Spain sent against England in 1588 was the"," Armada"," Flota"),
 ("Spanish","Spain","culture",1656,"Velázquez's masterpiece of the royal household is Las"," Meninas"," Hilanderas"),
 ("Spanish","Spain","politics",1982,"Spain's first Socialist PM after Franco, from 1982, was Felipe"," González"," Suárez"),
 ("Spanish","Spain","politics",1996,"The conservative Spanish PM elected in 1996 was José María"," Aznar"," Rajoy"),
 ("Spanish","Spain","politics",2014,"The Spanish king who succeeded Juan Carlos in 2014 was Felipe"," VI"," V"),
 ("Spanish","Spain","politics",1959,"The Basque separatist group founded in 1959 was"," ETA"," FARC"),
 ("Spanish","Spain","culture",1936,"The Spanish poet of 'Gypsy Ballads', killed in 1936, was Federico García"," Lorca"," Machado"),
 ("Spanish","Spain","culture",1988,"The Spanish director of 'Women on the Verge of a Nervous Breakdown' was Pedro"," Almodóvar"," Saura"),
 ("Spanish","Spain","culture",1950,"The exiled Spanish Surrealist filmmaker was Luis"," Buñuel"," Berlanga"),
 # Russia / USSR
 ("Russian","Russia","politics",1812,"The 1812 invader of Russia, defeated by winter, was"," Napoleon"," Charles"),
 ("Russian","Russia","politics",1861,"The Tsar who emancipated the serfs in 1861 was Alexander"," II"," III"),
 ("Russian","Russia","culture",1833,"Russia's national poet, author of 'Eugene Onegin', was Alexander"," Pushkin"," Lermontov"),
 ("Russian","Russia","politics",1922,"The union of soviet republics formed in 1922 was abbreviated the"," USSR"," RSFSR"),
 ("Russian","Russia","politics",1937,"Stalin's 1936-38 purge of party and army was the Great"," Terror"," Leap"),
 ("Russian","Russia","war",1979,"The country the USSR invaded in 1979 was"," Afghanistan"," Iran"),
 ("Russian","Russia","politics",1955,"The Soviet-led military alliance formed in 1955 was the"," Warsaw"," Comecon"),
 ("Russian","Russia","politics",1986,"Gorbachev's twin reforms were perestroika and"," glasnost"," kolkhoz"),
 ("Russian","Russia","politics",2008,"The Russian president 2008-2012, Putin's stand-in, was Dmitry"," Medvedev"," Zhirinovsky"),
 ("Russian","Russia","culture",1962,"The author of 'One Day in the Life of Ivan Denisovich' was Aleksandr"," Solzhenitsyn"," Sholokhov"),
 ("Russian","Russia","culture",1925,"The Soviet director of 'Battleship Potemkin' (1925) was Sergei"," Eisenstein"," Vertov"),
 ("Russian","Russia","culture",1937,"The Soviet composer of the 1937 Fifth Symphony was Dmitri"," Shostakovich"," Khachaturian"),
]


def score(model, tok, prompt, answer):
    """Return (sum_logp, avg_logp, n_tokens, first_token_logp) for the answer."""
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        ids = torch.tensor([full], device=model.device)
        lp = torch.log_softmax(model(ids).logits[0].float(), dim=-1)
    tl = [lp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid))]
    s = sum(tl)
    return s, s / len(tl), len(tl), tl[0]


def main():
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["model", "cutoff", "country", "language", "domain", "year", "era", "true", "false",
                                "gap", "gap_sum", "gap_first", "ntok_t", "ntok_f", "p_true"])
    for M in MODELS:
        cut = int(re.search(r"Ranke-4B-(\d{4})", M).group(1))
        print("\n#### " + M + f" (cutoff {cut})", flush=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(M)
            tok = AutoTokenizer.from_pretrained(M)
            model = AutoModelForCausalLM.from_pretrained(M, dtype=DTYPE, low_cpu_mem_usage=True).to(DEVICE).eval()
        except Exception as e:
            print("  SKIP", repr(e)[:140]); continue
        n = 0
        with open(OUT, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for (lang, country, dom, year, prompt, t, fa) in W:
                try:
                    st, at, nt, ft = score(model, tok, prompt, t)
                    sf, af, nf, ff = score(model, tok, prompt, fa)
                except Exception:
                    continue
                era = "pre" if year <= cut else "post"
                w.writerow([M.split("/")[-1], cut, country, lang, dom, year, era, t.strip(), fa.strip(),
                            round(at - af, 4), round(st - sf, 4), round(ft - ff, 4), nt, nf, round(math.exp(at), 5)])
                n += 1
                if n % 50 == 0:
                    f.flush(); print(f"  {n}/{len(W)} ...", flush=True)
        del model
        print(f"  done {n} items")
    print("\nALL DONE ->", OUT)


if __name__ == "__main__":
    main()
