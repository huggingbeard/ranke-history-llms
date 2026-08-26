# -*- coding: utf-8 -*-
"""Wide net of things a 1929-locked model should NOT know: real events/people AFTER 1930, organised by topic and spread
across time (WW2 just past the cutoff -> 1960s -> 2020, pitch dark). Plus a few OBSCURE post-1930 items (should stay dark
even if contaminated) and INVENTED events (nulls). Each is a "... in the year ___" completion with its true year. Only
events whose year I am confident of are included. We score two indicators on these and flag which light up, for a later
corpus check.  (topic, label, prompt, year)
"""
ITEMS = [
    # ---- WW2 (1939-45): just past the cutoff — most likely to have leaked ----
    ("WW2", "Germany invades Poland", "Germany invaded Poland in the year", 1939),
    ("WW2", "Fall of France", "France surrendered to Germany in the year", 1940),
    ("WW2", "Battle of Britain", "The Battle of Britain was fought in the year", 1940),
    ("WW2", "Dunkirk evacuation", "The evacuation of Dunkirk took place in the year", 1940),
    ("WW2", "Pearl Harbor", "The Japanese attack on Pearl Harbor came in the year", 1941),
    ("WW2", "Operation Barbarossa", "Germany invaded the Soviet Union in the year", 1941),
    ("WW2", "Battle of Stalingrad", "The Battle of Stalingrad ended in the year", 1943),
    ("WW2", "D-Day", "The Allied landings in Normandy took place in the year", 1944),
    ("WW2", "Hiroshima bombing", "The atomic bomb was dropped on Hiroshima in the year", 1945),
    ("WW2", "Germany surrenders", "Germany surrendered, ending the war in Europe, in the year", 1945),
    ("WW2", "Yalta Conference", "The Yalta Conference of Allied leaders was held in the year", 1945),
    ("WW2", "Auschwitz liberated", "The Auschwitz camp was liberated in the year", 1945),
    # ---- 1930s politics ----
    ("Politics-1930s", "Hitler becomes Chancellor", "Adolf Hitler became Chancellor of Germany in the year", 1933),
    ("Politics-1930s", "Reichstag fire", "The Reichstag fire occurred in the year", 1933),
    ("Politics-1930s", "Italy invades Ethiopia", "Italy invaded Ethiopia in the year", 1935),
    ("Politics-1930s", "Spanish Civil War begins", "The Spanish Civil War broke out in the year", 1936),
    ("Politics-1930s", "Moscow show trials", "The great Moscow show trials were held in the year", 1936),
    ("Politics-1930s", "Anschluss", "Germany annexed Austria in the Anschluss in the year", 1938),
    ("Politics-1930s", "Munich Agreement", "The Munich Agreement over Czechoslovakia was signed in the year", 1938),
    ("Politics-1930s", "Kristallnacht", "The Kristallnacht pogrom took place in the year", 1938),
    ("Politics-1930s", "Molotov-Ribbentrop Pact", "The Molotov-Ribbentrop Pact was signed in the year", 1939),
    # ---- postwar order / Cold War ----
    ("ColdWar", "United Nations founded", "The United Nations was founded in the year", 1945),
    ("ColdWar", "Indian independence", "India gained its independence in the year", 1947),
    ("ColdWar", "State of Israel founded", "The State of Israel was founded in the year", 1948),
    ("ColdWar", "Berlin Blockade", "The Berlin Blockade began in the year", 1948),
    ("ColdWar", "NATO founded", "The NATO alliance was founded in the year", 1949),
    ("ColdWar", "People's Republic of China", "The People's Republic of China was proclaimed in the year", 1949),
    ("ColdWar", "Korean War begins", "The Korean War began in the year", 1950),
    ("ColdWar", "Warsaw Pact", "The Warsaw Pact was established in the year", 1955),
    ("ColdWar", "Suez Crisis", "The Suez Crisis took place in the year", 1956),
    ("ColdWar", "Hungarian Revolution", "The Hungarian Revolution against Soviet rule occurred in the year", 1956),
    ("ColdWar", "Berlin Wall built", "The Berlin Wall was built in the year", 1961),
    ("ColdWar", "Cuban Missile Crisis", "The Cuban Missile Crisis unfolded in the year", 1962),
    ("ColdWar", "Prague Spring", "The Prague Spring was crushed by Soviet tanks in the year", 1968),
    ("ColdWar", "Berlin Wall falls", "The Berlin Wall fell in the year", 1989),
    ("ColdWar", "Soviet Union dissolves", "The Soviet Union was dissolved in the year", 1991),
    # ---- science & technology ----
    ("Science", "Nuclear fission discovered", "Nuclear fission was discovered in the year", 1938),
    ("Science", "Penicillin mass-produced", "Penicillin was first mass-produced in the year", 1943),
    ("Science", "Trinity test", "The first atomic bomb was tested in the New Mexico desert in the year", 1945),
    ("Science", "Transistor invented", "The transistor was invented in the year", 1947),
    ("Science", "DNA double helix", "The double-helix structure of DNA was discovered in the year", 1953),
    ("Science", "Polio vaccine", "Jonas Salk's polio vaccine was announced in the year", 1955),
    ("Science", "First heart transplant", "The first human heart transplant was performed in the year", 1967),
    ("Science", "World Wide Web", "The World Wide Web was invented in the year", 1989),
    # ---- space ----
    ("Space", "Sputnik", "The Soviet Union launched Sputnik, the first satellite, in the year", 1957),
    ("Space", "Gagarin", "Yuri Gagarin became the first human in space in the year", 1961),
    ("Space", "Moon landing", "The first humans landed on the Moon in the year", 1969),
    ("Space", "Hubble telescope", "The Hubble Space Telescope was launched in the year", 1990),
    # ---- culture ----
    ("Culture", "Gone with the Wind", "The film Gone with the Wind was released in the year", 1939),
    ("Culture", "Citizen Kane", "The film Citizen Kane was released in the year", 1941),
    ("Culture", "Orwell's 1984", "George Orwell's novel Nineteen Eighty-Four was published in the year", 1949),
    ("Culture", "The Beatles form", "The Beatles were formed in the year", 1960),
    ("Culture", "Woodstock festival", "The Woodstock music festival was held in the year", 1969),
    # ---- disasters ----
    ("Disasters", "Hindenburg", "The Hindenburg airship disaster occurred in the year", 1937),
    ("Disasters", "Chernobyl", "The Chernobyl nuclear disaster occurred in the year", 1986),
    # ---- sports ----
    ("Sports", "First football World Cup", "Uruguay won the first football World Cup in the year", 1930),
    ("Sports", "Berlin Olympics", "The Berlin Olympic Games were held in the year", 1936),
    # ---- people ----
    ("People", "FDR elected president", "Franklin Roosevelt was first elected U.S. president in the year", 1932),
    ("People", "Gandhi assassinated", "Mahatma Gandhi was assassinated in the year", 1948),
    ("People", "JFK becomes president", "John F. Kennedy became U.S. president in the year", 1961),
    ("People", "MLK 'I Have a Dream'", "Martin Luther King gave his 'I Have a Dream' speech in the year", 1963),
    ("People", "JFK assassinated", "President Kennedy was assassinated in the year", 1963),
    # ---- far post-cutoff: should be pitch dark even if contaminated ----
    ("Late-20thc", "Watergate scandal", "The Watergate scandal broke in the year", 1972),
    ("Late-20thc", "Fall of Saigon", "Saigon fell, ending the Vietnam War, in the year", 1975),
    ("Late-20thc", "Iranian Revolution", "The Iranian Revolution took place in the year", 1979),
    ("Late-20thc", "Falklands War", "The Falklands War was fought in the year", 1982),
    ("21stc", "9/11 attacks", "The September 11 terrorist attacks occurred in the year", 2001),
    ("21stc", "Global financial crisis", "The global financial crisis struck in the year", 2008),
    ("21stc", "COVID-19 pandemic", "The COVID-19 pandemic began in the year", 2020),
    # ---- obscure post-1930 real: real but minor; should stay dark ----
    ("Obscure-real", "Montreux Convention", "The Montreux Convention on the straits was signed in the year", 1936),
    ("Obscure-real", "Anti-Comintern Pact", "The Anti-Comintern Pact was signed in the year", 1936),
    ("Obscure-real", "Chaco War ends", "The Chaco War between Bolivia and Paraguay ended in the year", 1935),
    ("Obscure-real", "Treaty of Saadabad", "The Treaty of Saadabad was signed in the year", 1937),
    ("Obscure-real", "Winter War", "The Winter War between the Soviet Union and Finland began in the year", 1939),
    # ---- invented events: nulls (should read dark on both) ----
    ("Invented", "Vashenko Treaty", "The Treaty of Vashenko was signed in the year", 1952),
    ("Invented", "Grelham Accord", "The Grelham Accord was signed in the year", 1961),
    ("Invented", "Kessendorf Uprising", "The Kessendorf Uprising was suppressed in the year", 1948),
    ("Invented", "Marnburg Conference", "The Marnburg Conference convened in the year", 1955),
    ("Invented", "Astorheim Crisis", "The Astorheim Crisis erupted in the year", 1967),
    ("Invented", "Battle of Veldhaven", "The Battle of Veldhaven was fought in the year", 1943),
    ("Invented", "Treaty of Brenzau", "The Treaty of Brenzau was signed in the year", 1971),
    ("Invented", "Kaltenfeld Rebellion", "The Kaltenfeld Rebellion broke out in the year", 1958),
]
WIN = 8
