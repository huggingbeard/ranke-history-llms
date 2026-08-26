"""
COMPREHENSIVE clean diagnostic battery. ONE rule for every POST facet:
  the ANSWER must require post-cutoff text (a coined term, a date, a post-cutoff-specific
  detail, or a post-1929-EMERGENT surname) AND the prompt must NOT cue it (no first name,
  no shared prefix). The TOPIC may be anything -- pre-1929-famous people/places are kept,
  but appear only as CONTEXT in the prompt, with the answer being a post-cutoff fact.
  (Shostakovich -> his 1937 'Fifth'; Churchill -> became PM '40 / 'Iron Curtain'; Sartre ->
  refused the Nobel in '64.) PRE facets are pre-cutoff should-know baselines (the ceiling).
Format (validate_facets-compatible): (category, "POST"/"PRE", repr_year, [(prompt, true, [falses])]).
"""

CANDIDATES = [
 # ===================== US POLITICS =====================
 ("US politics", "POST", 1960, [
   ("Roosevelt's 1930s economic recovery programme was called the New", " Deal", [" Pact", " Order", " Way", " Plan"]),
   ("Franklin Roosevelt first won the presidency in the year 19", " 32", [" 24", " 28", " 40", " 44"]),
   ("The atomic bomb dropped on Hiroshima was nicknamed", " Little Boy", [" Fat Man", " Thin Man", " Big Boy"]),
   ("The atomic bomb dropped on Nagasaki was nicknamed", " Fat Man", [" Little Boy", " Thin Man", " Big Boy"]),
   ("The aircraft that dropped the Hiroshima bomb was the", " Enola Gay", [" Bockscar", " Big Stink", " Full House"]),
   ("Japan was struck by atomic bombs in the year 19", " 45", [" 41", " 43", " 47", " 49"]),
   ("The scandal that forced a US president to resign in 1974 was called", " Watergate", [" Teapot", " Whitewater", " Irangate"]),
   ("A US president resigned from office for the first time in the year 19", " 74", [" 68", " 72", " 79", " 81"]),
   ("The man who assassinated President Kennedy in Dallas in 1963 was", " Oswald", [" Sirhan", " Ruby", " Chapman", " Hinckley"]),
   ("President Kennedy was assassinated in the year 19", " 63", [" 61", " 65", " 68", " 60"]),
   ("Nixon's policy of handing the ground war to South Vietnamese forces was called", " Vietnamization", [" Detente", " Containment", " Pacification"]),
   ("Saigon fell, ending the Vietnam War, in the year 19", " 75", [" 65", " 68", " 73", " 79"]),
   ("The 1980s US policy of sweeping tax cuts and deregulation was nicknamed", " Reaganomics", [" Nixonomics", " Bushonomics", " Clintonomics"]),
   ("The first African-American US president was elected in the year 20", " 08", [" 00", " 04", " 12", " 16"]),
   ("The 2010 US health-care reform was popularly nicknamed", " Obamacare", [" Trumpcare", " Hillarycare", " Romneycare"]),
   ("The Western military alliance the US led in founding in 1949 was", " NATO", [" SEATO", " CENTO", " ANZUS", " NORAD"]),
 ]),
 ("US politics", "PRE", 1850, [
   ("The Declaration of Independence was adopted in the year 17", " 76", [" 12", " 49", " 83"]),
   ("The American Civil War ended in the year 18", " 65", [" 12", " 48", " 77"]),
   ("The 1803 purchase from France that doubled US territory was the", " Louisiana", [" Florida", " Oregon", " Missouri"]),
   ("The Confederate capital during the Civil War was", " Richmond", [" Atlanta", " Charleston", " Savannah"]),
   ("The 1846-48 war that gained California was fought against", " Mexico", [" Spain", " Britain", " France"]),
 ]),
 # ===================== UK POLITICS (Churchill kept, probed via post-cutoff facts) =====================
 ("UK politics", "POST", 1960, [
   ("Winston Churchill first became prime minister in the year 19", " 40", [" 22", " 35", " 51", " 45"]),
   ("In a 1946 speech Churchill warned of an 'Iron' descending across Europe: the Iron", " Curtain", [" Wall", " Gate", " Fence", " Veil"]),
   ("Churchill's wartime coalition lost the 1945 election; the new Labour prime minister was", " Attlee", [" Morrison", " Bevin", " Gaitskell"]),
   ("Britain's tax-funded national health service was founded in the year 19", " 48", [" 32", " 38", " 58", " 28"]),
   ("Britain and France seized the Egyptian canal in a crisis in the year 19", " 56", [" 36", " 46", " 66", " 72"]),
   ("Britain went to war with Argentina in 1982 over a group of South Atlantic", " islands", [" colonies", " harbours", " stations"]),
   ("The Falklands War took place in the year 19", " 82", [" 56", " 74", " 88"]),
   ("Britain's 2016 referendum on leaving the European Union was nicknamed", " Brexit", [" Bremain", " Grexit", " Frexit"]),
   ("Britain's EU referendum was held in the year 20", " 16", [" 04", " 11", " 19"]),
   ("Britain's first woman prime minister took office in the year 19", " 79", [" 64", " 70", " 90", " 97"]),
   ("The 1984-85 industrial dispute that defined the Thatcher era was the strike by the", " miners", [" dockers", " printers", " railmen"]),
 ]),
 ("UK politics", "PRE", 1850, [
   ("The British admiral who died at Trafalgar in 1805 was Horatio", " Nelson", [" Collingwood", " Howe", " Jervis"]),
   ("Napoleon was finally defeated in 1815 at the Battle of", " Waterloo", [" Austerlitz", " Borodino", " Leipzig"]),
   ("The 1707 Act of Union joined England with", " Scotland", [" Ireland", " Wales", " Hanover"]),
   ("The queen who gave her name to a long 19th-century era was", " Victoria", [" Adelaide", " Caroline", " Anne"]),
 ]),
 # ===================== FRANCE POLITICS =====================
 ("France politics", "POST", 1960, [
   ("The republic that de Gaulle founded in France in 1958 was the", " Fifth", [" Fourth", " Third", " Sixth", " Second"]),
   ("De Gaulle founded that republic in the year 19", " 58", [" 44", " 46", " 62", " 68"]),
   ("France's wave of student and worker protests erupted in May of 19", " 68", [" 48", " 58", " 72", " 81"]),
   ("Algeria won its independence from France in the year 19", " 62", [" 46", " 54", " 58", " 68"]),
   ("France fell to the German invasion in the year 19", " 40", [" 14", " 18", " 44", " 36"]),
   ("The 1944 Allied landings that began the liberation of France took place in the year 19", " 44", [" 40", " 18", " 45", " 39"]),
   ("France's high-speed passenger train, launched in 1981, is the", " TGV", [" ICE", " AVE", " HST"]),
   ("France tested its first atomic bomb in the year 19", " 60", [" 45", " 52", " 68", " 74"]),
 ]),
 ("France politics", "PRE", 1850, [
   ("The Paris prison stormed by a crowd in 1789 was the", " Bastille", [" Conciergerie", " Temple", " Louvre"]),
   ("The French Revolution began in the year 17", " 89", [" 76", " 92", " 99"]),
   ("Napoleon crowned himself Emperor of the French in the year 18", " 04", [" 12", " 15", " 21"]),
   ("The radical leader of the Reign of Terror was Maximilien", " Robespierre", [" Danton", " Marat", " Mirabeau"]),
 ]),
 # ===================== GERMANY POLITICS =====================
 ("Germany politics", "POST", 1960, [
   ("Germany's 1941 invasion of the Soviet Union was code-named Operation", " Barbarossa", [" Sealion", " Citadel", " Typhoon", " Valkyrie"]),
   ("Germany's rapid armoured-warfare tactic was called", " Blitzkrieg", [" Sitzkrieg", " Stellungskrieg", " Kesselschlacht"]),
   ("Germany invaded Poland, starting the Second World War, in the year 19", " 39", [" 33", " 41", " 45", " 36"]),
   ("The 1944 plot by army officers to assassinate Hitler was code-named Operation", " Valkyrie", [" Barbarossa", " Citadel", " Anvil"]),
   ("The German field marshal nicknamed the 'Desert Fox' was", " Rommel", [" Guderian", " Manstein", " Kesselring"]),
   ("The 1938 Nazi pogrom against Germany's Jews was called", " Kristallnacht", [" Anschluss", " Blitzkrieg", " Sonderaktion"]),
   ("The wall dividing Berlin was built in the year 19", " 61", [" 45", " 53", " 71", " 89"]),
   ("The Berlin Wall fell in the year 19", " 89", [" 79", " 85", " 91", " 75"]),
   ("The famous Allied crossing point in divided Berlin was Checkpoint", " Charlie", [" Bravo", " Delta", " Echo", " Alpha"]),
   ("Germany was reunified in the year 19", " 90", [" 45", " 61", " 95", " 72"]),
   ("Germany's first woman chancellor took office in the year 20", " 05", [" 98", " 02", " 09", " 13"]),
 ]),
 ("Germany politics", "PRE", 1850, [
   ("The statesman who unified Germany in 1871 was Otto von", " Bismarck", [" Moltke", " Roon", " Caprivi"]),
   ("Napoleon was defeated in 1813 at the Battle of the Nations at", " Leipzig", [" Jena", " Dresden", " Bautzen"]),
   ("The 1848 liberal German national parliament met in the city of", " Frankfurt", [" Vienna", " Berlin", " Munich"]),
   ("The Prussian royal seat near Berlin, with Frederick's palace, was", " Potsdam", [" Dresden", " Weimar", " Kassel"]),
 ]),
 # ===================== RUSSIA / SOVIET =====================
 ("Russia politics", "POST", 1960, [
   ("The Soviet policy of political 'openness' under Gorbachev was called", " glasnost", [" perestroika", " kolkhoz", " samizdat"]),
   ("The Soviet policy of economic 'restructuring' under Gorbachev was called", " perestroika", [" glasnost", " uskorenie", " khozraschet"]),
   ("The Soviet Union was formally dissolved in the year 19", " 91", [" 85", " 89", " 95", " 79"]),
   ("The first artificial satellite, launched by the USSR in 1957, was named", " Sputnik", [" Vanguard", " Explorer", " Telstar"]),
   ("The first satellite was launched in the year 19", " 57", [" 49", " 53", " 61", " 47"]),
   ("Stalin's great purge of party and army peaked in the year 19", " 37", [" 17", " 24", " 45", " 53"]),
   ("The 1955 Soviet-led military alliance was the Warsaw", " Pact", [" Treaty", " Accord", " League"]),
   ("A Soviet nuclear reactor in Ukraine suffered a catastrophic disaster in the year 19", " 86", [" 76", " 79", " 91", " 71"]),
   ("The USSR launched the first human into space in the year 19", " 61", [" 53", " 57", " 65", " 69"]),
   ("The USSR invaded Afghanistan, beginning a decade-long war, in the year 19", " 79", [" 56", " 68", " 85", " 89"]),
 ]),
 ("Russia politics", "PRE", 1850, [
   ("The 1812 invader of Russia, defeated by the winter, was", " Napoleon", [" Charles", " Frederick", " Gustavus"]),
   ("The Russian imperial capital founded by Peter the Great was St", " Petersburg", [" Moscow", " Kiev", " Novgorod"]),
   ("The Russian tsar who modernized the country around 1700 was Peter the", " Great", [" Terrible", " Wise", " Bold"]),
   ("The dynasty that ruled Russia from 1613 was the", " Romanov", [" Rurik", " Godunov", " Shuysky"]),
 ]),
 # ===================== ITALY POLITICS =====================
 ("Italy politics", "POST", 1960, [
   ("Mussolini's Fascist regime fell and Italy left the Axis in the year 19", " 43", [" 22", " 36", " 45", " 39"]),
   ("Italy abolished its monarchy and became a republic in the year 19", " 46", [" 22", " 18", " 60", " 36"]),
   ("The former Italian PM kidnapped and murdered by the Red Brigades in 1978 was", " Moro", [" Andreotti", " Fanfani", " Cossiga"]),
   ("The far-left group behind kidnappings in 1970s Italy was the Red", " Brigades", [" Guard", " Army", " Front"]),
   ("The media tycoon who repeatedly served as Italian prime minister from the 1990s was", " Berlusconi", [" Prodi", " Dini", " Amato"]),
   ("Italy adopted the euro, replacing the lira, in the year 20", " 02", [" 92", " 96", " 08"]),
   ("Italy most recently won the football World Cup in the year 20", " 06", [" 82", " 94", " 10"]),
 ]),
 ("Italy politics", "PRE", 1850, [
   ("The general whose Redshirts helped unify Italy was Giuseppe", " Garibaldi", [" Mazzini", " Cavour", " Crispi"]),
   ("Italian unification was largely completed in the year 18", " 70", [" 15", " 48", " 90"]),
   ("The Renaissance city ruled by the Medici was", " Florence", [" Venice", " Milan", " Naples"]),
 ]),
 # ===================== SPAIN POLITICS =====================
 ("Spain politics", "POST", 1960, [
   ("The general who won the Spanish Civil War and ruled until 1975 was", " Franco", [" Mola", " Sanjurjo", " Queipo"]),
   ("The Spanish Civil War began in the year 19", " 36", [" 14", " 23", " 45", " 31"]),
   ("Spain's monarchy was restored after the dictator's death in the year 19", " 75", [" 45", " 56", " 81", " 92"]),
   ("The Basque separatist group founded in 1959 was", " ETA", [" FARC", " IRA", " PKK"]),
   ("Spain joined the European Community in the year 19", " 86", [" 57", " 73", " 92", " 04"]),
   ("Spain won its first football World Cup in the year 20", " 10", [" 82", " 94", " 06"]),
   ("The 2004 Madrid train bombings were carried out by extremists linked to", " al-Qaeda", [" ETA", " FARC", " the IRA"]),
 ]),
 ("Spain politics", "PRE", 1850, [
   ("The monarchs who funded Columbus in 1492 were Ferdinand and", " Isabella", [" Joanna", " Catherine", " Maria"]),
   ("The fleet Spain sent against England in 1588 was the", " Armada", [" Flota", " Hermandad"]),
   ("The Spanish Habsburg king at the height of empire was Philip", " II", [" III", " IV", " V"]),
 ]),
 # ===================== WESTERN CULTURE (Shostakovich/Sartre/Orwell kept, via post-cutoff facts) =====================
 ("Culture", "POST", 1960, [
   ("Shostakovich's 1937 symphony, billed as a 'reply to just criticism', was his", " Fifth", [" Third", " Sixth", " Eighth", " Tenth"]),
   ("Shostakovich's 1962 symphony commemorating the Babi Yar massacre was his", " Thirteenth", [" Eleventh", " Ninth", " Fifteenth"]),
   ("George Orwell's dystopian novel published in 1949 was titled", " 1984", [" 2001", " 1948", " 1999"]),
   ("Orwell's 1945 satire of the Russian Revolution was titled Animal", " Farm", [" House", " Kingdom", " Republic"]),
   ("The French existentialist Sartre refused the Nobel Prize in the year 19", " 64", [" 38", " 49", " 57", " 72"]),
   ("Sartre's major 1943 philosophical work was titled Being and", " Nothingness", [" Time", " Becoming", " Essence"]),
   ("The record-breaking 1982 album by the 'King of Pop' was titled", " Thriller", [" Bad", " Off the Wall", " Dangerous"]),
   ("George Lucas's space-opera 'Star Wars' was first released in the year 19", " 77", [" 68", " 72", " 82", " 89"]),
   ("Coppola's 1972 film about a Mafia family was titled The", " Godfather", [" Untouchables", " Goodfellas", " Departed"]),
   ("J.K. Rowling's first Harry Potter novel was published in the year 19", " 97", [" 87", " 92", " 01"]),
   ("Picasso painted his anti-war mural of a bombed Basque town in the year 19", " 37", [" 14", " 25", " 45", " 50"]),
 ]),
 ("Culture", "PRE", 1850, [
   ("The Elizabethan playwright who wrote Hamlet was William", " Shakespeare", [" Marlowe", " Jonson", " Webster"]),
   ("The German composer of the Fifth and Ninth Symphonies was Ludwig van", " Beethoven", [" Brahms", " Schubert", " Handel"]),
   ("The Italian poet of the medieval Divine Comedy was Dante", " Alighieri", [" Petrarch", " Boccaccio", " Tasso"]),
   ("The Renaissance painter of the Mona Lisa was Leonardo da", " Vinci", [" Raffaello", " Bramante", " Donatello"]),
 ]),
 # ===================== WESTERN SCIENCE =====================
 ("Science", "POST", 1960, [
   ("Watson and Crick described the DNA molecule in 1953 as a double", " helix", [" spiral", " strand", " coil", " ribbon"]),
   ("The structure of DNA was worked out in the year 19", " 53", [" 43", " 47", " 61", " 39"]),
   ("The device invented at Bell Labs in 1947 that replaced the vacuum tube was the", " transistor", [" capacitor", " magnetron", " thyristor"]),
   ("Nuclear fission of uranium was discovered in the year 19", " 38", [" 28", " 19", " 45", " 33"]),
   ("The United States tested the first hydrogen bomb in the year 19", " 52", [" 42", " 45", " 62", " 38"]),
   ("The first person to walk on the Moon, in 1969, was", " Armstrong", [" Aldrin", " Glenn", " Shepard"]),
   ("The first Moon landing took place in the year 19", " 69", [" 61", " 65", " 72", " 59"]),
   ("The mission that first landed humans on the Moon was Apollo", " 11", [" 13", " 10", " 17", " 8"]),
   ("The first polio vaccine, announced in 1955, was developed by", " Salk", [" Sabin", " Enders", " Francis"]),
   ("The author of the 1988 bestseller 'A Brief History of Time' was", " Hawking", [" Penrose", " Dyson", " Sagan"]),
   ("The World Wide Web was proposed in 1989 by", " Berners-Lee", [" Cerf", " Kahn", " Postel"]),
 ]),
 ("Science", "PRE", 1850, [
   ("The naturalist who wrote On the Origin of Species in 1859 was Charles", " Darwin", [" Owen", " Huxley", " Wallace"]),
   ("The English physicist of universal gravitation was Isaac", " Newton", [" Hooke", " Halley", " Boyle"]),
   ("The Italian astronomer tried by the Inquisition for heliocentrism was", " Galileo", [" Bruno", " Cassini", " Kepler"]),
   ("The astronomer who placed the Sun at the centre of the cosmos was", " Copernicus", [" Brahe", " Kepler", " Ptolemy"]),
 ]),
]
