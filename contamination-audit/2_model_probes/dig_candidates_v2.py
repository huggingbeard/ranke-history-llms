"""Power-up clusters for the two underpowered positive leads: Soviet/Russia politics and
German WWII. ~13 facets each, post-cutoff-ONLY AND with NO first-name (or any) cue to the
answer in the prompt -- the model must produce the answer from the descriptive fact alone,
not by completing a famous name. Coined terms / years preferred. DeepSeek then screens."""

CANDIDATES = [
 ("Russia politics", "Soviet era (Cold War)", 1964, [
   ("The Soviet leader who denounced Stalin in a 1956 secret speech was", " Khrushchev", [" Malenkov", " Bulganin", " Molotov", " Mikoyan"]),
   ("The Soviet leader of the 'era of stagnation' from 1964 was", " Brezhnev", [" Kosygin", " Podgorny", " Suslov", " Gromyko"]),
   ("The Soviet leader who introduced perestroika and glasnost in the 1980s was", " Gorbachev", [" Ligachev", " Ryzhkov", " Shevardnadze", " Yakovlev"]),
   ("The former secret-police chief who briefly led the USSR after Brezhnev was", " Andropov", [" Chernenko", " Ustinov", " Tikhonov", " Romanov"]),
   ("The frail Soviet leader who succeeded Andropov in 1984 was", " Chernenko", [" Grishin", " Romanov", " Kunaev", " Solomentsev"]),
   ("The Soviet policy of political openness in the 1980s was called", " glasnost", [" perestroika", " kolkhoz", " samizdat", " druzhba"]),
   ("The Soviet policy of economic restructuring in the 1980s was called", " perestroika", [" glasnost", " khozraschet", " uskorenie", " nomenklatura"]),
   ("The first president of post-Soviet Russia was", " Yeltsin", [" Gaidar", " Rutskoi", " Chernomyrdin", " Primakov"]),
   ("The Soviet Union was dissolved in the year 19", " 91", [" 85", " 89", " 95", " 79"]),
   ("The Soviet-led Cold War military alliance founded in 1955 was the Warsaw", " Pact", [" Treaty", " Accord", " League", " Union"]),
   ("The Soviet premier ousted by Khrushchev in the mid-1950s was", " Malenkov", [" Bulganin", " Mikoyan", " Pervukhin", " Saburov"]),
   ("Russia's president from the year 2000, a former intelligence officer, was", " Putin", [" Medvedev", " Primakov", " Stepashin", " Kiriyenko"]),
   ("The Soviet leader during the 1962 Cuban missile crisis was", " Khrushchev", [" Brezhnev", " Kosygin", " Mikoyan", " Gromyko"]),
 ]),
 ("Germany politics", "Germany in WWII", 1941, [
   ("Germany's 1941 invasion of the Soviet Union was code-named Operation", " Barbarossa", [" Sealion", " Citadel", " Typhoon", " Valkyrie"]),
   ("Germany's planned 1940 invasion of Britain was code-named Operation Sea", " Lion", [" Eagle", " Wolf", " Bear", " Hawk"]),
   ("The 1944 plot by army officers to assassinate Hitler was code-named Operation", " Valkyrie", [" Barbarossa", " Citadel", " Anvil", " Cobra"]),
   ("Germany's rapid armoured-warfare tactic was called", " Blitzkrieg", [" Sitzkrieg", " Stellungskrieg", " Kesselschlacht"]),
   ("Germany invaded Poland, starting the Second World War, in the year 19", " 39", [" 33", " 41", " 45", " 36"]),
   ("Germany surrendered, ending the war in Europe, in the year 19", " 45", [" 39", " 41", " 43", " 47"]),
   ("The German field marshal nicknamed the 'Desert Fox' was", " Rommel", [" Guderian", " Manstein", " Kesselring", " Model"]),
   ("The German general who surrendered the Sixth Army at Stalingrad was", " Paulus", [" Manstein", " Hoth", " Kleist", " Weichs"]),
   ("Germany's 1938 annexation of Austria was called the", " Anschluss", [" Blitzkrieg", " Heimkehr", " Einmarsch", " Drang"]),
   ("The Nazi genocide of roughly six million Jews became known as the", " Holocaust", [" Pogrom", " Inquisition", " Diaspora", " Exodus"]),
   ("Hitler's deputy who flew to Scotland in 1941 was", " Hess", [" Bormann", " Speer", " Frick", " Ley"]),
   ("The 1942 conference that organised the genocide of Europe's Jews was the", " Wannsee Conference", [" Munich Conference", " Potsdam Conference", " Yalta Conference"]),
   ("The German long-range ballistic rocket fired at London was the V-", " 2", [" 1", " 3", " 4"]),
   ("The Nazi air force during the Second World War was the", " Luftwaffe", [" Wehrmacht", " Kriegsmarine", " Reichswehr", " Schutzstaffel"]),
 ]),
]
