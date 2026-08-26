"""
CONTAMINATION MAP — large clustered teacher-forced minimal-pair sweep.

~230 true/false minimal pairs tagged by (country, language-family, domain, year),
scored on every Ranke cutoff model. Each item is judged against THAT model's cutoff:
  era = 'pre'  if year <= cutoff  (model SHOULD know it  -> positive gap = validation)
  era = 'post' if year >  cutoff  (model should NOT      -> positive gap = CONTAMINATION)

Metric = teacher-forced per-token log-prob gap, logp(true) - logp(false).
Single items are noisy (that's expected); the signal is the CLUSTER AGGREGATE.

Outputs a per-item CSV and prints, per model: pre-vs-post mean gap, and the
post-cutoff ('shouldn't know') gap broken down by language family, by country,
and by domain — i.e. WHERE the leakage is — plus a cross-model panel to see if
1900/1913/1929 share the same leaked clusters.

NOTE: items are probe stimuli generated for breadth, not a vetted ground-truth
dataset; a few may contain errors. The aggregate is robust to that by design.
Runs unattended; writes the CSV incrementally per model.
"""
import os, re, csv, math, statistics
from collections import defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"; os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_TOKEN", open(r"D:\UZHechist Dropbox\Joachim Voth\claudecode\api\HF.txt").read().strip())
torch.set_num_threads(os.cpu_count() or 8)

MODELS = ["uzh-echist-org/Ranke-4B-1900", "uzh-echist-org/Ranke-4B-1913", "uzh-echist-org/Ranke-4B-1929",
          "uzh-echist-org/Ranke-4B-1913-1202-Instruct", "uzh-echist-org/Ranke-4B-1929-1202-Instruct"]
OUT = r"D:\UZHechist Dropbox\Joachim Voth\claudecode\ranke-contamination-audit\5_results\contamination_map_results.csv"

# (country, langfam, domain, year, prompt, TRUE, FALSE)
ITEMS = [
 # ================= USA (Germanic / English) =================
 ("USA","Germanic","politics",1861,"The U.S. President during the Civil War was Abraham"," Lincoln"," Jackson"),
 ("USA","Germanic","politics",1865,"Abraham Lincoln was assassinated in 1865 by John Wilkes"," Booth"," Oswald"),
 ("USA","Germanic","politics",1901,"The U.S. President assassinated in 1901 was William"," McKinley"," Garfield"),
 ("USA","Germanic","politics",1904,"The U.S. President of the 'Square Deal', in office 1901-1909, was Theodore"," Roosevelt"," Cleveland"),
 ("USA","Germanic","politics",1917,"The U.S. President through most of the First World War was Woodrow"," Wilson"," Taft"),
 ("USA","Germanic","politics",1933,"The U.S. President who launched the New Deal in the 1930s was Franklin"," Roosevelt"," Landon"),
 ("USA","Germanic","politics",1945,"The U.S. President who ordered the atomic bombings of Japan was Harry"," Truman"," Dewey"),
 ("USA","Germanic","politics",1953,"The U.S. general of D-Day who became President in 1953 was Dwight"," Eisenhower"," Bradley"),
 ("USA","Germanic","politics",1963,"The U.S. President shot dead in Dallas in 1963 was John F."," Kennedy"," Humphrey"),
 ("USA","Germanic","politics",1974,"The U.S. President who resigned over Watergate in 1974 was Richard"," Nixon"," Rockefeller"),
 ("USA","Germanic","politics",1980,"The U.S. President elected in 1980, a former actor, was Ronald"," Reagan"," Mondale"),
 ("USA","Germanic","politics",2008,"The first African-American U.S. President, elected 2008, was Barack"," Obama"," Powell"),
 ("USA","Germanic","war",1941,"The 1941 Japanese attack that brought the U.S. into WWII struck Pearl"," Harbor"," Cove"),
 ("USA","Germanic","politics",1963,"The U.S. civil-rights leader of the 'I Have a Dream' speech was Martin Luther"," King"," Powell"),
 ("USA","Germanic","war",2001,"The skyscrapers destroyed in the September 11, 2001 attacks were the World Trade"," Center"," Exchange"),
 ("USA","Germanic","science",1879,"The American who built the first practical light bulb was Thomas"," Edison"," Westinghouse"),
 ("USA","Germanic","science",1953,"The structure of DNA was co-discovered in 1953 by the American James"," Watson"," Pauling"),
 ("USA","Germanic","tech",2007,"Apple's touchscreen phone, launched in 2007, is called the"," iPhone"," Newton"),
 ("USA","Germanic","culture",1956,"The American singer crowned 'King of Rock and Roll' was Elvis"," Presley"," Holly"),
 ("USA","Germanic","culture",1977,"George Lucas's 1977 science-fiction film was called Star"," Wars"," Trek"),
 ("USA","Germanic","culture",1982,"The American singer called the 'King of Pop' was Michael"," Jackson"," Richie"),
 ("USA","Germanic","space",1969,"The first man to walk on the Moon, in 1969, was Neil"," Armstrong"," Shepard"),
 ("USA","Germanic","politics",1962,"The U.S. President during the 1962 Cuban Missile Crisis was John F."," Kennedy"," Johnson"),
 ("USA","Germanic","tech",1947,"The electronic switch invented at Bell Labs in 1947 was the"," transistor"," capacitor"),
 ("USA","Germanic","politics",1789,"The first President of the United States was George"," Washington"," Adams"),
 # ================= UK & Ireland (Germanic / English) =================
 ("UK","Germanic","politics",1837,"The British queen who reigned for most of the 19th century was Queen"," Victoria"," Anne"),
 ("UK","Germanic","politics",1940,"The British prime minister through most of the Second World War was Winston"," Churchill"," Eden"),
 ("UK","Germanic","politics",1979,"Britain's 'Iron Lady' prime minister, elected 1979, was Margaret"," Thatcher"," Castle"),
 ("UK","Germanic","politics",1945,"The Labour prime minister who won the 1945 election was Clement"," Attlee"," Bevin"),
 ("UK","Germanic","politics",1936,"The British king who abdicated in 1936 to marry Wallis Simpson was Edward"," VIII"," VII"),
 ("UK","Germanic","politics",1938,"The British PM who returned from Munich in 1938 promising 'peace for our time' was Neville"," Chamberlain"," Baldwin"),
 ("UK","Germanic","science",1859,"The British naturalist who wrote On the Origin of Species was Charles"," Darwin"," Lyell"),
 ("UK","Germanic","science",1876,"The Scotsman who patented the telephone in 1876 was Alexander Graham"," Bell"," Watt"),
 ("UK","Germanic","science",1928,"The Scottish bacteriologist who discovered penicillin in 1928 was Alexander"," Fleming"," Lister"),
 ("UK","Germanic","science",1988,"The British physicist of black holes and 'A Brief History of Time' was Stephen"," Hawking"," Penrose"),
 ("UK","Germanic","science",1940,"The British mathematician who broke the Enigma code at Bletchley Park was Alan"," Turing"," Babbage"),
 ("UK","Germanic","culture",1962,"The Liverpool band of Lennon and McCartney was The"," Beatles"," Hollies"),
 ("UK","Germanic","war",1916,"The Irish nationalist rising against British rule in 1916 was the Easter"," Rising"," Revolt"),
 ("UK","Germanic","politics",1922,"The treaty creating the Irish Free State was signed with Britain in 19"," 21"," 25"),
 ("UK","Germanic","war",1915,"The Cunard liner sunk by a German U-boat in 1915 was the"," Lusitania"," Mauretania"),
 ("UK","Germanic","politics",1953,"The British monarch crowned in 1953 was Queen Elizabeth"," II"," I"),
 ("UK","Germanic","sport",2012,"The 2012 Summer Olympic Games were held in"," London"," Manchester"),
 ("UK","Germanic","science",1908,"The New Zealand-British physicist who modelled the atom's nucleus was Ernest"," Rutherford"," Thomson"),
 ("UK","Germanic","war",1982,"In 1982 Britain fought Argentina over the"," Falklands"," Azores"),
 ("UK","Germanic","politics",1997,"The Labour prime minister elected in 1997 was Tony"," Blair"," Brown"),
 # ================= France (Romance) =================
 ("France","Romance","politics",1870,"The French emperor who fell from power after defeat by Prussia in 1870 was Napoleon"," III"," II"),
 ("France","Romance","politics",1940,"The general who led the Free French and founded the Fifth Republic was Charles de"," Gaulle"," Lattre"),
 ("France","Romance","politics",1940,"The WWI marshal who headed the collaborationist Vichy regime was Philippe"," Pétain"," Foch"),
 ("France","Romance","war",1916,"The longest battle of the First World War, at a French fortress in 1916, was the Battle of"," Verdun"," Sedan"),
 ("France","Romance","politics",1919,"The treaty ending the First World War, signed near Paris in 1919, was the Treaty of"," Versailles"," Amiens"),
 ("France","Romance","science",1898,"The Polish-French scientist who discovered radium was Marie"," Curie"," Lavoisier"),
 ("France","Romance","science",1865,"The French chemist behind pasteurization and vaccines was Louis"," Pasteur"," Lamarck"),
 ("France","Romance","politics",1894,"The French army officer wrongly convicted of treason in 1894 was Alfred"," Dreyfus"," Esterhazy"),
 ("France","Romance","culture",1898,"The novelist who defended Dreyfus with the article 'J'accuse' was Émile"," Zola"," Hugo"),
 ("France","Romance","war",1935,"France's belt of border forts against Germany, built in the 1930s, was the"," Maginot"," Verdun"),
 ("France","Romance","politics",1968,"France's wave of student and worker protests came in May 19"," 68"," 48"),
 ("France","Romance","tech",1969,"The supersonic airliner built jointly by Britain and France was the"," Concorde"," Caravelle"),
 ("France","Romance","culture",1943,"The French existentialist who wrote 'Being and Nothingness' was Jean-Paul"," Sartre"," Camus"),
 ("France","Romance","politics",1981,"The first Socialist president of the Fifth Republic, elected 1981, was François"," Mitterrand"," Rocard"),
 ("France","Romance","politics",2017,"The French president elected in 2017, the youngest ever, was Emmanuel"," Macron"," Fillon"),
 ("France","Romance","culture",1900,"The French Impressionist painter of water lilies was Claude"," Monet"," Manet"),
 ("France","Romance","politics",1958,"The French Fifth Republic was founded in 19"," 58"," 46"),
 ("France","Romance","science",1889,"The Paris tower built for the 1889 exposition was designed by Gustave"," Eiffel"," Haussmann"),
 ("France","Romance","politics",1789,"The French Revolution began with the storming, in 1789, of the"," Bastille"," Tuileries"),
 ("France","Romance","war",1944,"The Allied D-Day landings of 1944 took place on the beaches of"," Normandy"," Brittany"),
 # ================= Germany & Austria (Germanic) =================
 ("Germany","Germanic","politics",1871,"The 'Iron Chancellor' who unified Germany in 1871 was Otto von"," Bismarck"," Moltke"),
 ("Germany","Germanic","politics",1914,"The German Kaiser who reigned during the First World War was Wilhelm"," II"," I"),
 ("Germany","Germanic","politics",1933,"The Nazi leader who became German chancellor in 1933 was Adolf"," Hitler"," Hugenberg"),
 ("Germany","Germanic","war",1941,"The 1941 German invasion of the Soviet Union was code-named Operation"," Barbarossa"," Sealion"),
 ("Germany","Germanic","politics",1949,"The first chancellor of West Germany, from 1949, was Konrad"," Adenauer"," Schumacher"),
 ("Germany","Germanic","politics",1969,"The West German chancellor of 'Ostpolitik', Nobel 1971, was Willy"," Brandt"," Erhard"),
 ("Germany","Germanic","politics",1990,"The German chancellor who oversaw reunification in 1990 was Helmut"," Kohl"," Genscher"),
 ("Germany","Germanic","politics",2005,"The German chancellor elected in 2005, a woman, was Angela"," Merkel"," Steinmeier"),
 ("Germany","Germanic","politics",1961,"The wall dividing a German city, built in 1961, was the Berlin"," Wall"," Gate"),
 ("Germany","Germanic","politics",1989,"The Berlin Wall was torn down in 19"," 89"," 79"),
 ("Germany","Germanic","science",1915,"The German-born physicist of general relativity was Albert"," Einstein"," Heisenberg"),
 ("Germany","Germanic","science",1900,"The German physicist who founded quantum theory in 1900 was Max"," Planck"," Roentgen"),
 ("Austria","Germanic","science",1900,"The Viennese founder of psychoanalysis was Sigmund"," Freud"," Adler"),
 ("Austria","Germanic","politics",1914,"The Austrian heir whose 1914 assassination triggered WWI was Archduke Franz"," Ferdinand"," Joseph"),
 ("Germany","Germanic","politics",1933,"Hitler's chief of propaganda was Joseph"," Goebbels"," Bormann"),
 ("Germany","Germanic","war",1945,"The largest Nazi extermination camp, in occupied Poland, was"," Auschwitz"," Buchenwald"),
 ("Germany","Germanic","science",1944,"The German engineer behind the V-2 rocket and later NASA was Wernher von"," Braun"," Oberth"),
 ("Germany","Germanic","sport",1936,"The 1936 Summer Olympics, used for Nazi propaganda, were held in"," Berlin"," Munich"),
 ("Germany","Germanic","war",1943,"The turning-point 1942-43 battle on the Volga was the Battle of"," Stalingrad"," Kursk"),
 ("Germany","Germanic","politics",1919,"The democratic German republic founded after WWI was the"," Weimar"," Rhenish"),
 # ================= Italy (Romance) =================
 ("Italy","Romance","politics",1860,"The general whose Redshirts helped unify Italy in 1860 was Giuseppe"," Garibaldi"," Mazzini"),
 ("Italy","Romance","politics",1922,"The Fascist who seized power by marching on Rome in 1922 was Benito"," Mussolini"," Badoglio"),
 ("Italy","Romance","war",1940,"The Italian dictator allied with Hitler in WWII was Benito"," Mussolini"," Ciano"),
 ("Italy","Romance","science",1901,"The Italian who sent the first transatlantic radio signal (1901) was Guglielmo"," Marconi"," Meucci"),
 ("Italy","Romance","science",1942,"The Italian physicist who built the first nuclear reactor (1942) was Enrico"," Fermi"," Rubbia"),
 ("Italy","Romance","politics",1926,"The Italian Communist theorist imprisoned by Mussolini was Antonio"," Gramsci"," Togliatti"),
 ("Italy","Romance","politics",1994,"The Italian media-tycoon prime minister of the 1990s-2000s was Silvio"," Berlusconi"," Prodi"),
 ("Italy","Romance","religion",1962,"The Church council that modernized Catholicism (1962-65) was the Second Vatican"," Council"," Synod"),
 ("Italy","Romance","sport",1982,"Italy's goalkeeper-captain when they won the 1982 World Cup was Dino"," Zoff"," Rossi"),
 ("Italy","Romance","culture",1960,"The Italian director of 'La Dolce Vita' (1960) was Federico"," Fellini"," Visconti"),
 ("Italy","Romance","politics",1978,"The Italian ex-PM kidnapped and killed by the Red Brigades in 1978 was Aldo"," Moro"," Fanfani"),
 ("Italy","Romance","politics",1946,"Italy abolished its monarchy and became a republic in 19"," 46"," 24"),
 ("Italy","Romance","science",1610,"The Italian astronomer tried by the Inquisition for heliocentrism was Galileo"," Galilei"," Bruno"),
 ("Italy","Romance","culture",1308,"The medieval Italian poet of the Divine Comedy was Dante"," Alighieri"," Petrarch"),
 ("Italy","Romance","politics",1861,"The statesman who became Italy's first prime minister in 1861 was Camillo di"," Cavour"," Crispi"),
 ("Italy","Romance","culture",1503,"The Renaissance Italian who painted the Mona Lisa was Leonardo da"," Vinci"," Raphael"),
 ("Italy","Romance","tech",2002,"The euro replaced Italy's former currency, the"," lira"," franc"),
 ("Italy","Romance","politics",1929,"The 1929 treaty creating Vatican City was the"," Lateran"," Tridentine"),
 # ================= Russia / USSR (Slavic) =================
 ("Russia","Slavic","politics",1917,"The last Tsar of Russia, who abdicated in 1917, was Nicholas"," II"," I"),
 ("Russia","Slavic","politics",1917,"The Bolshevik leader of the 1917 October Revolution was Vladimir"," Lenin"," Kerensky"),
 ("Russia","Slavic","politics",1924,"The dictator who took control of the USSR after Lenin's death was Joseph"," Stalin"," Bukharin"),
 ("Russia","Slavic","politics",1940,"Stalin's exiled rival, assassinated in Mexico in 1940, was Leon"," Trotsky"," Zinoviev"),
 ("Russia","Slavic","politics",1956,"The Soviet leader who denounced Stalin in a secret 1956 speech was Nikita"," Khrushchev"," Malenkov"),
 ("Russia","Slavic","politics",1964,"The Soviet leader of the long 'stagnation' era, from 1964, was Leonid"," Brezhnev"," Kosygin"),
 ("Russia","Slavic","politics",1985,"The Soviet leader of 'perestroika' who oversaw the USSR's collapse was Mikhail"," Gorbachev"," Andropov"),
 ("Russia","Slavic","politics",1991,"The first president of post-Soviet Russia was Boris"," Yeltsin"," Chernomyrdin"),
 ("Russia","Slavic","politics",2000,"The Russian president from 2000, a former KGB officer, was Vladimir"," Putin"," Medvedev"),
 ("Russia","Slavic","space",1957,"The first artificial satellite, launched by the USSR in 1957, was called"," Sputnik"," Soyuz"),
 ("Russia","Slavic","space",1961,"The first human to orbit the Earth, a Soviet cosmonaut in 1961, was Yuri"," Gagarin"," Titov"),
 ("Russia","Slavic","war",1943,"The Soviet city of the decisive 1942-43 battle against the Nazis was"," Stalingrad"," Smolensk"),
 ("Russia","Slavic","science",1986,"The Soviet nuclear reactor that exploded in 1986 in Ukraine was"," Chernobyl"," Mayak"),
 ("Russia","Slavic","culture",1869,"The Russian author of 'War and Peace' was Leo"," Tolstoy"," Turgenev"),
 ("Russia","Slavic","culture",1880,"The Russian composer of the '1812 Overture' was Pyotr"," Tchaikovsky"," Borodin"),
 ("Russia","Slavic","science",1869,"The Russian chemist who devised the periodic table was Dmitri"," Mendeleev"," Butlerov"),
 ("Russia","Slavic","science",1904,"The Russian physiologist famous for conditioned reflexes in dogs was Ivan"," Pavlov"," Sechenov"),
 ("Russia","Slavic","war",1979,"The Soviet Union invaded its southern neighbour in 1979, beginning the war in"," Afghanistan"," Persia"),
 ("Russia","Slavic","politics",1922,"The communist union of republics formed out of Russia in 1922 was the Soviet"," Union"," Federation"),
 ("Russia","Slavic","culture",1958,"The Russian author of 'Doctor Zhivago', forced to decline the 1958 Nobel, was Boris"," Pasternak"," Sholokhov"),
 # ================= China (Sino-Tibetan) =================
 ("China","Sino-Tibetan","politics",1912,"The last emperor of China, who abdicated as a child in 1912, was"," Puyi"," Guangxu"),
 ("China","Sino-Tibetan","politics",1911,"The leader hailed as father of the 1911 Chinese republic was Sun"," Yat-sen"," Yuan"),
 ("China","Sino-Tibetan","politics",1949,"The Nationalist leader who fled to Taiwan in 1949 was Chiang"," Kai-shek"," Soong"),
 ("China","Sino-Tibetan","politics",1949,"The Communist who founded the People's Republic of China in 1949 was Mao"," Zedong"," Lin"),
 ("China","Sino-Tibetan","politics",1966,"Mao's violent 1966-76 ideological upheaval was the Cultural"," Revolution"," Awakening"),
 ("China","Sino-Tibetan","politics",1958,"Mao's disastrous 1958 industrial-agricultural drive was the Great Leap"," Forward"," Onward"),
 ("China","Sino-Tibetan","politics",1978,"The leader who opened China's economy to markets from 1978 was Deng"," Xiaoping"," Hua"),
 ("China","Sino-Tibetan","politics",1989,"The 1989 pro-democracy protests were crushed in Beijing's Tiananmen"," Square"," Palace"),
 ("China","Sino-Tibetan","politics",2013,"The Chinese leader who consolidated power from 2013 is Xi"," Jinping"," Wen"),
 ("China","Sino-Tibetan","politics",1997,"The British colony handed back to China in 1997 was Hong"," Kong"," Macau"),
 ("China","Sino-Tibetan","war",1934,"The epic 1934 retreat of the Chinese Communists was the Long"," March"," Road"),
 ("China","Sino-Tibetan","war",1931,"Japan invaded the Chinese region of Manchuria in 19"," 31"," 11"),
 ("China","Sino-Tibetan","tech",2006,"The vast Yangtze dam completed around 2006 is the Three"," Gorges"," Rivers"),
 ("China","Sino-Tibetan","sport",2008,"The 2008 Summer Olympic Games were held in"," Beijing"," Shanghai"),
 ("China","Sino-Tibetan","culture",-500,"The ancient Chinese philosopher of the Analects was"," Confucius"," Mencius"),
 ("China","Sino-Tibetan","politics",1900,"The 1900 anti-foreign uprising in China was the"," Boxer"," Taiping"),
 ("China","Sino-Tibetan","politics",1644,"The Chinese dynasty that fell in 1911 was the"," Qing"," Tang"),
 ("China","Sino-Tibetan","war",1937,"Japan's brutal 1937 capture of China's then-capital is called the Rape of"," Nanking"," Canton"),
 ("China","Sino-Tibetan","politics",1976,"The radical faction blamed for the Cultural Revolution's excesses was the Gang of"," Four"," Five"),
 ("China","Sino-Tibetan","tech",2003,"China's first astronaut, launched in 2003, was Yang"," Liwei"," Fei"),
 # ================= Japan (Japonic) =================
 ("Japan","Japonic","war",1941,"The 1941 Japanese surprise attack on the U.S. fleet was at Pearl"," Harbor"," Bay"),
 ("Japan","Japonic","war",1945,"The Japanese city destroyed by the first atomic bomb in 1945 was"," Hiroshima"," Yokohama"),
 ("Japan","Japonic","war",1945,"The second Japanese city hit by an atomic bomb in 1945 was"," Nagasaki"," Sapporo"),
 ("Japan","Japonic","politics",1941,"Japan's militarist wartime prime minister, hanged in 1948, was Hideki"," Tojo"," Konoe"),
 ("Japan","Japonic","politics",1945,"The U.S. general who governed occupied Japan after 1945 was Douglas"," MacArthur"," Patton"),
 ("Japan","Japonic","tech",1964,"Japan's high-speed 'bullet train', launched in 1964, is the"," Shinkansen"," Maglev"),
 ("Japan","Japonic","sport",1964,"The first Asian city to host the Summer Olympics, in 1964, was"," Tokyo"," Osaka"),
 ("Japan","Japonic","science",2011,"Japan's 2011 earthquake and tsunami caused a nuclear meltdown at"," Fukushima"," Sendai"),
 ("Japan","Japonic","politics",1868,"The 1868 restoration that modernized Japan was named for Emperor"," Meiji"," Taisho"),
 ("Japan","Japonic","war",1905,"Japan stunned the world by defeating which empire in the war of 1904-05?"," Russia"," Britain"),
 ("Japan","Japonic","war",1905,"Japan's crushing 1905 naval victory over Russia was at the Strait of"," Tsushima"," Malacca"),
 ("Japan","Japonic","tech",1979,"The Japanese firm that launched the Walkman and PlayStation is"," Sony"," Sanyo"),
 ("Japan","Japonic","culture",1954,"The Japanese director of 'Seven Samurai' (1954) was Akira"," Kurosawa"," Ozu"),
 ("Japan","Japonic","politics",2012,"Japan's long-serving 2010s PM, assassinated in 2022, was Shinzo"," Abe"," Aso"),
 ("Japan","Japonic","economy",1980,"The Japanese carmaker, maker of the Corolla, is"," Toyota"," Hyundai"),
 ("Japan","Japonic","war",1942,"Japan's naval defeat that turned the Pacific War in 1942 was the Battle of"," Midway"," Jutland"),
 ("Japan","Japonic","politics",1853,"The American commodore whose 1853 'black ships' forced Japan to open was"," Perry"," Dewey"),
 ("Japan","Japonic","culture",1980,"The Japanese company behind Mario and the Game Boy is"," Nintendo"," Konami"),
 # ================= India & South Asia (Indo-Aryan) =================
 ("India","Indo-Aryan","politics",1948,"The leader of Indian independence by nonviolence, killed in 1948, was Mahatma"," Gandhi"," Bose"),
 ("India","Indo-Aryan","politics",1948,"Gandhi was assassinated in 1948 by Nathuram"," Godse"," Savarkar"),
 ("India","Indo-Aryan","politics",1947,"India's first prime minister, from 1947, was Jawaharlal"," Nehru"," Patel"),
 ("India","Indo-Aryan","politics",1966,"India's woman prime minister, assassinated in 1984, was Indira"," Gandhi"," Desai"),
 ("India","Indo-Aryan","politics",1947,"The Muslim state created by the 1947 partition of British India was"," Pakistan"," Bangladesh"),
 ("India","Indo-Aryan","politics",1947,"Pakistan's founding leader, 'the Quaid', was Muhammad Ali"," Jinnah"," Iqbal"),
 ("India","Indo-Aryan","politics",1971,"The country that won independence from Pakistan in 1971 was"," Bangladesh"," Nepal"),
 ("India","Indo-Aryan","politics",1947,"The last British viceroy, who oversaw India's 1947 partition, was Lord"," Mountbatten"," Curzon"),
 ("India","Indo-Aryan","war",1974,"India conducted its first nuclear weapon test in 19"," 74"," 64"),
 ("India","Indo-Aryan","war",1919,"British troops massacred Indian civilians in 1919 at"," Amritsar"," Lucknow"),
 ("India","Indo-Aryan","politics",2014,"India's Hindu-nationalist prime minister, in office from 2014, is Narendra"," Modi"," Advani"),
 ("India","Indo-Aryan","culture",1632,"The marble mausoleum built by Mughal emperor Shah Jahan is the Taj"," Mahal"," Mahal"),
 ("India","Indo-Aryan","culture",-200,"The revered ancient Hindu scripture, part of the Mahabharata, is the Bhagavad"," Gita"," Veda"),
 ("India","Indo-Aryan","politics",1885,"The party that led India's independence movement was the Indian National"," Congress"," League"),
 ("India","Indo-Aryan","politics",1947,"Until 1972 Sri Lanka was known by the colonial name"," Ceylon"," Burma"),
 ("India","Indo-Aryan","culture",1913,"The first non-European Nobel laureate in Literature (1913), an Indian poet, was Rabindranath"," Tagore"," Naidu"),
 ("India","Indo-Aryan","economy",2000,"The southern Indian city known as the country's IT hub is"," Bangalore"," Karachi"),
 ("India","Indo-Aryan","politics",1857,"The 1857 uprising against British rule in India is called the Sepoy"," Mutiny"," Revolt"),
 # ================= Middle East / Turkey / Iran (Semitic / Turkic / Iranic) =================
 ("Turkey","Turkic","politics",1923,"The founder of the modern secular Turkish republic in 1923 was Mustafa Kemal"," Atatürk"," İnönü"),
 ("MiddleEast","Semitic","politics",1948,"The Jewish state founded in the Middle East in 1948 was"," Israel"," Judah"),
 ("MiddleEast","Semitic","politics",1948,"Israel's first prime minister, in 1948, was David"," Ben-Gurion"," Weizmann"),
 ("MiddleEast","Semitic","politics",1956,"The Egyptian leader who nationalized the Suez Canal in 1956 was Gamal Abdel"," Nasser"," Naguib"),
 ("MiddleEast","Semitic","politics",1977,"The Egyptian president who made peace with Israel, killed in 1981, was Anwar"," Sadat"," Mubarak"),
 ("Iran","Iranic","politics",1979,"The cleric who led Iran's 1979 Islamic Revolution was Ayatollah"," Khomeini"," Khamenei"),
 ("Iran","Iranic","politics",1953,"The Iranian PM who nationalized oil and was ousted in a 1953 coup was Mohammad"," Mossadegh"," Zahedi"),
 ("Iran","Iranic","politics",1979,"The Shah of Iran overthrown in 1979 was Mohammad Reza"," Pahlavi"," Qajar"),
 ("Iraq","Semitic","politics",1990,"The Iraqi dictator who invaded Kuwait in 1990 was Saddam"," Hussein"," Aziz"),
 ("MiddleEast","Semitic","politics",1969,"The long-time chairman of the Palestine Liberation Organization was Yasser"," Arafat"," Habash"),
 ("MiddleEast","Semitic","war",1967,"The 1967 war in which Israel defeated its Arab neighbours in six days was the"," Six-Day"," Yom-Kippur"),
 ("Turkey","Turkic","politics",1922,"The empire that collapsed after the First World War, ruled from Istanbul, was the"," Ottoman"," Byzantine"),
 ("MiddleEast","Semitic","war",1917,"The British officer who aided the WWI Arab revolt, 'of Arabia', was T. E."," Lawrence"," Allenby"),
 ("MiddleEast","Semitic","economy",1960,"The oil-exporters' cartel founded in 1960 is abbreviated"," OPEC"," NATO"),
 ("MiddleEast","Semitic","geography",1869,"The canal opened in 1869 linking the Mediterranean and Red Sea is the"," Suez"," Panama"),
 ("Iran","Iranic","politics",1979,"In 1979 Iranian students seized the U.S. embassy, holding diplomats as"," hostages"," prisoners"),
 ("Turkey","Turkic","war",1915,"The failed WWI Allied landing in Turkey in 1915 was at"," Gallipoli"," Salonika"),
 # ================= Latin America (Romance) =================
 ("LatinAmerica","Romance","politics",1820,"The 'Liberator' who freed much of South America from Spain was Simón"," Bolívar"," Sucre"),
 ("LatinAmerica","Romance","politics",1946,"Argentina's populist president of the 1940s, husband of Eva, was Juan"," Perón"," Yrigoyen"),
 ("LatinAmerica","Romance","politics",1959,"The revolutionary who seized power in Cuba in 1959 was Fidel"," Castro"," Batista"),
 ("LatinAmerica","Romance","politics",1959,"Castro's Argentine-born guerrilla comrade was Che"," Guevara"," Cienfuegos"),
 ("LatinAmerica","Romance","politics",1973,"Chile's Marxist president overthrown in a 1973 coup was Salvador"," Allende"," Frei"),
 ("LatinAmerica","Romance","politics",1973,"The general who seized power in Chile's 1973 coup was Augusto"," Pinochet"," Videla"),
 ("LatinAmerica","Romance","politics",1911,"The peasant leader of the Mexican Revolution in the south was Emiliano"," Zapata"," Madero"),
 ("LatinAmerica","Romance","politics",1916,"The Mexican revolutionary nicknamed 'Pancho' was"," Villa"," Carranza"),
 ("LatinAmerica","Romance","tech",1960,"Brazil's purpose-built modernist capital, inaugurated in 1960, is"," Brasília"," Recife"),
 ("LatinAmerica","Romance","sport",1958,"The Brazilian footballer who won three World Cups was"," Pelé"," Garrincha"),
 ("LatinAmerica","Romance","sport",1986,"The Argentine footballer of the 1986 'Hand of God' goal was Diego"," Maradona"," Kempes"),
 ("LatinAmerica","Romance","culture",1967,"The Colombian Nobel novelist of magical realism was Gabriel García"," Márquez"," Borges"),
 ("LatinAmerica","Romance","tech",1914,"The interoceanic canal that opened in Central America in 1914 was the"," Panama"," Nicaragua"),
 ("LatinAmerica","Romance","politics",1999,"Venezuela's socialist 'Bolivarian' president from 1999 was Hugo"," Chávez"," Maduro"),
 ("LatinAmerica","Romance","religion",2013,"The first Latin American pope, elected in 2013, took the name"," Francis"," Benedict"),
 ("LatinAmerica","Romance","politics",1521,"The Spaniard who conquered the Aztec empire was Hernán"," Cortés"," Pizarro"),
 ("LatinAmerica","Romance","politics",1533,"The Spaniard who conquered the Inca empire was Francisco"," Pizarro"," Balboa"),
 ("LatinAmerica","Romance","politics",1822,"The South American country whose colonial ruler was Portugal, not Spain, is"," Brazil"," Peru"),
 # ================= Africa (sub-Saharan) =================
 ("Africa","Bantu","politics",1994,"South Africa's first black president, jailed 27 years, elected 1994, was Nelson"," Mandela"," Sisulu"),
 ("Africa","Bantu","politics",1948,"South Africa's system of legalized racial segregation was called"," apartheid"," partition"),
 ("Africa","Akan","politics",1957,"The leader of Ghana, first black African colony to win independence (1957), was Kwame"," Nkrumah"," Busia"),
 ("Africa","Bantu","politics",1963,"Kenya's independence leader and first president was Jomo"," Kenyatta"," Mboya"),
 ("Africa","Bantu","politics",1961,"The Congolese independence leader assassinated in 1961 was Patrice"," Lumumba"," Tshombe"),
 ("Africa","Bantu","war",1994,"The 1994 genocide of Tutsis by Hutus took place in"," Rwanda"," Uganda"),
 ("Africa","Bantu","politics",1980,"Zimbabwe's autocratic ruler from independence in 1980 was Robert"," Mugabe"," Nkomo"),
 ("Africa","Semitic","politics",1930,"Ethiopia's emperor crowned in 1930, deposed in 1974, was Haile"," Selassie"," Menelik"),
 ("Africa","Bantu","politics",1960,"Africa's most populous nation, independent from Britain in 1960, is"," Nigeria"," Ghana"),
 ("Africa","Bantu","politics",1984,"The South African anti-apartheid archbishop, Nobel 1984, was Desmond"," Tutu"," Boesak"),
 ("Africa","colonial","politics",1884,"European powers carved up Africa at the 1884 conference in"," Berlin"," Vienna"),
 ("Africa","colonial","war",1902,"Britain fought Dutch-descended settlers in South Africa in the war that ended in 19"," 02"," 12"),
 ("Africa","colonial","exploration",1871,"The reporter who found the missing missionary Dr. Livingstone in Africa was Henry"," Stanley"," Burton"),
 ("Africa","Semitic","culture",-2500,"The ancient pyramids at Giza were built by the people of"," Egypt"," Nubia"),
 ("Africa","Bantu","economy",1960,"Most of France's African colonies gained independence in the year 19"," 60"," 45"),
 ("Africa","Afrikaans","politics",1989,"The South African president who freed Mandela and ended apartheid was F. W. de"," Klerk"," Villiers"),
]


def avg_logp(model, tok, prompt, answer):
    pid = tok(prompt, add_special_tokens=False).input_ids
    aid = tok(answer, add_special_tokens=False).input_ids
    full = pid + aid
    with torch.no_grad():
        logp = torch.log_softmax(model(torch.tensor([full])).logits[0], dim=-1)
    tot = sum(logp[len(pid) + k - 1, full[len(pid) + k]].item() for k in range(len(aid)))
    return tot / len(aid)


def main():
    # fresh CSV
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["model", "cutoff", "country", "langfam", "domain", "year", "era",
                                "true", "false", "gap", "p_true"])
    for M in MODELS:
        cutoff = int(re.search(r"Ranke-4B-(\d{4})", M).group(1))
        print("\n" + "#" * 90 + f"\n# {M}  (cutoff {cutoff})\n" + "#" * 90, flush=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(M)
            tok = AutoTokenizer.from_pretrained(M)
            model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32, low_cpu_mem_usage=True).eval()
        except Exception as e:
            print("  SKIP (load failed):", repr(e)[:160]); continue

        rows = []
        with open(OUT, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for (country, lf, dom, year, prompt, t, fa) in ITEMS:
                try:
                    gap = avg_logp(model, tok, prompt, t) - avg_logp(model, tok, prompt, fa)
                    pt = math.exp(avg_logp(model, tok, prompt, t))
                except Exception:
                    continue
                era = "pre" if year <= cutoff else "post"
                w.writerow([M.split("/")[-1], cutoff, country, lf, dom, year, era, t.strip(), fa.strip(),
                            round(gap, 4), round(pt, 5)])
                rows.append((country, lf, dom, year, era, gap))
            f.flush()
        del model

        # ---- per-model summaries ----
        def mean(xs): return statistics.mean(xs) if xs else float("nan")
        pre = [g for *_, e, g in rows if e == "pre"]
        post = [g for *_, e, g in rows if e == "post"]
        print(f"\n  PRE  (should know,  n={len(pre)}):  mean gap {mean(pre):+.2f},  prefers-truth {sum(g>0 for g in pre)}/{len(pre)}")
        print(f"  POST (should NOT,   n={len(post)}):  mean gap {mean(post):+.2f},  prefers-truth {sum(g>0 for g in post)}/{len(post)}   <-- >0 = contamination")
        print("\n  POST-cutoff leakage by LANGUAGE FAMILY:")
        bylf = defaultdict(list)
        for c, lf, dom, y, e, g in rows:
            if e == "post": bylf[lf].append(g)
        for lf in sorted(bylf, key=lambda k: -mean(bylf[k])):
            print(f"     {lf:<14} n={len(bylf[lf]):<3} mean gap {mean(bylf[lf]):+.2f}  prefers-truth {sum(g>0 for g in bylf[lf])}/{len(bylf[lf])}")
        print("\n  POST-cutoff leakage by COUNTRY:")
        byc = defaultdict(list)
        for c, lf, dom, y, e, g in rows:
            if e == "post": byc[c].append(g)
        for c in sorted(byc, key=lambda k: -mean(byc[k])):
            print(f"     {c:<14} n={len(byc[c]):<3} mean gap {mean(byc[c]):+.2f}")
        print("\n  POST-cutoff leakage by DOMAIN:")
        byd = defaultdict(list)
        for c, lf, dom, y, e, g in rows:
            if e == "post": byd[dom].append(g)
        for d in sorted(byd, key=lambda k: -mean(byd[k])):
            print(f"     {d:<12} n={len(byd[d]):<3} mean gap {mean(byd[d]):+.2f}")
    print("\n\nALL MODELS DONE ->", OUT)


if __name__ == "__main__":
    main()
