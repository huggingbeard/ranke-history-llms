# -*- coding: utf-8 -*-
"""Shared, model-agnostic contamination batteries (pure data; no torch). Used by suite.py (Ranke, via
transformers) and exported to question_bank.json for Guohui to score on Talkie with the same items.

Every item is a forced choice: (label, prompt, true_continuation, [3 matched distractors]). The scored
continuation strings are designed so that ONLY post-cutoff knowledge (for POST) or genuine pre-cutoff
knowledge (for PRE) can pick the true one over equally-plausible alternatives. Matched distractors net out
the generic-name/answer floor -- which scales with model size -- so a 4B and a 13B are compared on the SAME
chance line (1/4 = 25%). That is the whole point: it makes Ranke-4B vs Talkie-13B a FAIR comparison, unlike a
raw rank-ratio that inflates with model size.

POST_*: facts that exist ONLY in text after the ~1930 wall. A clean time-locked model lands at chance.
PRE_CONTROL: facts well within the pre-1930 window. A working model should ACE these -- they prove the probe
fires before we trust any post-cutoff null."""

# ---- POST-CUTOFF (should be at chance for a clean ~1930 model) ----
POST = {
 "Vietnam-US war": [
  ("17th parallel", "After the 1954 Geneva conference, Vietnam was divided into a communist north and a non-communist south at the", " 17th parallel", [" 16th parallel", " 38th parallel", " 49th parallel"]),
  ("Tet Offensive", "The great surprise offensive the communists launched across the south in early 1968 became known as the", " Tet Offensive", [" Spring Offensive", " Winter Offensive", " Monsoon Offensive"]),
  ("Gulf of Tonkin", "The 1964 naval incident that led the United States to escalate the war took place in the Gulf of", " Tonkin", [" Siam", " Bengal", " Oman"]),
  ("Agent Orange", "To strip the jungle of cover, American forces sprayed a toxic defoliant code-named Agent", " Orange", [" Blue", " Purple", " Silver"]),
  ("Saigon fell 1975", "After years of war, the southern capital of Saigon finally fell to the communist forces in the year", " 1975", [" 1945", " 1919", " 1898"]),
  ("Viet Cong", "The communist guerrillas who fought the Americans in the southern countryside were called the Viet", " Cong", [" Minh", " Lao", " Nam"]),
  ("Ho Chi Minh Trail", "The hidden network of jungle paths used to move supplies south was called the Ho Chi Minh", " Trail", [" Road", " Canal", " Wall"]),
  ("My Lai", "The 1968 massacre of unarmed villagers by American soldiers took place at the hamlet of My", " Lai", [" Son", " Tho", " Hoa"]),
  ("Paris accords", "The peace accords that ended direct American involvement in the war were signed in the city of", " Paris", [" Geneva", " Vienna", " Potsdam"]),
  ("Khe Sanh", "A long and famous 1968 siege of an American Marine base was fought at Khe", " Sanh", [" Trang", " Binh", " Dak"]),
  ("Rolling Thunder", "The sustained American bombing campaign against the north was code-named Operation Rolling", " Thunder", [" Storm", " Hammer", " Lightning"]),
  ("Reunified 1976", "The north and the south of the country were formally reunified into one state in the year", " 1976", [" 1945", " 1902", " 1918"]),
  ("Saigon=S capital", "Throughout the war the capital of South Vietnam was the city of", " Saigon", [" Hanoi", " Hue", " Vinh"]),
  ("domino theory", "American strategists feared that if one country turned communist its neighbours would topple like", " dominoes", [" cards", " bricks", " leaves"]),
  ("napalm", "The jellied-gasoline firebomb that American aircraft dropped on the jungle was called", " napalm", [" phosgene", " mustard gas", " chlorine"]),
 ],
 "WWII": [
  ("Pearl Harbor 1941", "The surprise Japanese carrier attack on the American Pacific fleet at Pearl Harbor came in the year", " 1941", [" 1905", " 1898", " 1919"]),
  ("Overlord", "The great Allied seaborne invasion of northern France in June 1944 was code-named Operation", " Overlord", [" Sledgehammer", " Avalanche", " Torch"]),
  ("Barbarossa", "Germany's massive 1941 invasion of the Soviet Union was code-named Operation", " Barbarossa", [" Valkyrie", " Citadel", " Sealion"]),
  ("the Blitz", "The sustained German bombing of British cities through the winter of 1940 became known as the", " Blitz", [" Terror", " Onslaught", " Siege"]),
  ("Stalingrad", "The battle widely seen as the turning point of the war on the eastern front was fought at", " Stalingrad", [" Smolensk", " Voronezh", " Rostov"]),
  ("Hiroshima A-bomb", "The first atomic bomb ever used in war was dropped on the Japanese city of", " Hiroshima", [" Yokohama", " Sendai", " Niigata"]),
  ("Auschwitz", "The largest of the Nazi extermination camps was located at the Polish town of", " Auschwitz", [" Dachau", " Buchenwald", " Belsen"]),
  ("Yalta 1945", "In 1945 the three Allied leaders met to plan the postwar order at the Crimean resort of", " Yalta", [" Sochi", " Odessa", " Batumi"]),
  ("Midway", "The decisive 1942 carrier battle that turned the Pacific war was fought near the island of", " Midway", [" Samoa", " Guam", " Fiji"]),
  ("V-E 1945", "Germany surrendered unconditionally, ending the war in Europe, in the year", " 1945", [" 1918", " 1923", " 1929"]),
  ("Final Solution", "The Nazi euphemism for the systematic murder of Europe's Jews was the Final", " Solution", [" Reckoning", " Judgment", " Order"]),
  ("Battle of the Bulge", "Germany's last great western offensive, through the Ardennes in 1944, is called the Battle of the", " Bulge", [" Ridge", " Gap", " Pass"]),
  ("El Alamein", "The 1942 battle that turned the North African desert war was fought at El", " Alamein", [" Agheila", " Adem", " Hamma"]),
  ("Manhattan Project", "The secret American programme that built the atomic bomb was code-named the", " Manhattan Project", [" Brooklyn Project", " Hudson Project", " Lincoln Project"]),
  ("Dunkirk 1940", "In 1940 the trapped British army was evacuated by sea from the French port of", " Dunkirk", [" Calais", " Brest", " Cherbourg"]),
 ],
 "Cold War": [
  ("Berlin Wall 1961", "The concrete wall sealing off West Berlin from the communist east was built in the year", " 1961", [" 1919", " 1923", " 1929"]),
  ("Cuban crisis island", "The 1962 nuclear standoff between Washington and Moscow was over missiles placed on the island of", " Cuba", [" Haiti", " Jamaica", " Cyprus"]),
  ("Sputnik", "The first artificial satellite to orbit the Earth, launched by the Soviets in 1957, was called", " Sputnik", [" Vostok", " Luna", " Soyuz"]),
  ("cosmonaut", "The first human being to travel into space, in 1961, was a Soviet", " cosmonaut", [" astronaut", " aviator", " pilot"]),
  ("Korea 38th", "After the war Korea was divided between a communist north and a capitalist south at the", " 38th parallel", [" 17th parallel", " 49th parallel", " 16th parallel"]),
  ("NATO founded", "The western military alliance known as NATO was first founded in the year", " 1949", [" 1919", " 1929", " 1815"]),
  ("iron curtain", "Churchill warned in 1946 that across Europe there had descended an iron", " curtain", [" wall", " gate", " fence"]),
  ("Marshall Plan", "The huge American programme to rebuild war-torn Europe after 1948 was called the", " Marshall Plan", [" Truman Plan", " Morgenthau Plan", " Hoover Plan"]),
  ("Moon landing 1969", "Human beings first walked on the surface of the Moon in the year", " 1969", [" 1929", " 1945", " 1957"]),
  ("Apollo", "The American programme that first landed men on the Moon was named", " Apollo", [" Mercury", " Gemini", " Vanguard"]),
  ("Berlin Airlift", "When the Soviets blockaded West Berlin in 1948, the Western powers supplied the city entirely by", " airlift", [" convoy", " railway", " pipeline"]),
  ("Warsaw Pact", "The Soviet-led military alliance set up in 1955 to counter the West was called the", " Warsaw Pact", [" Moscow Pact", " Prague Pact", " Budapest Pact"]),
  ("Wall fell 1989", "The Berlin Wall was finally torn down, opening the east, in the year", " 1989", [" 1929", " 1945", " 1918"]),
  ("Mutual Assured Destr.", "The nuclear doctrine holding that any attack would bring annihilation to both sides was Mutual Assured", " Destruction", [" Disarmament", " Deterrence", " Defence"]),
  ("Korean War country", "Between 1950 and 1953 United Nations forces fought communist armies in the divided country of", " Korea", [" Burma", " Malaya", " Tibet"]),
 ],
}

# ---- PRE-CUTOFF POSITIVE CONTROLS (a working ~1930 model should ACE these; proves the probe fires) ----
PRE_CONTROL = {
 "pre-cutoff controls": [
  ("Waterloo 1815", "Napoleon was finally and decisively defeated in 1815 at the battle of", " Waterloo", [" Austerlitz", " Borodino", " Jena"]),
  ("US Independence 1776", "The American Declaration of Independence was adopted in the year", " 1776", [" 1789", " 1798", " 1812"]),
  ("Bastille 1789", "The French Revolution began in 1789 with the storming of the", " Bastille", [" Tuileries", " Louvre", " Sorbonne"]),
  ("Trafalgar Nelson", "The British admiral who fell at the moment of victory at Trafalgar was Lord", " Nelson", [" Collingwood", " Hawke", " Rodney"]),
  ("Waterloo village", "Napoleon's final defeat was fought near the small Belgian village of", " Waterloo", [" Wavre", " Ligny", " Gembloux"]),
  ("Civil War ended 1865", "The American Civil War came to an end in the year", " 1865", [" 1848", " 1871", " 1812"]),
  ("Origin of Species", "Charles Darwin set out his theory of evolution in a book called On the Origin of", " Species", [" Life", " Man", " Nature"]),
  ("Suez 1869", "The Suez Canal, joining the Mediterranean to the Red Sea, was opened in the year", " 1869", [" 1825", " 1901", " 1789"]),
  ("German unif. 1871", "The unification of Germany under Prussian leadership was completed in the year", " 1871", [" 1848", " 1815", " 1806"]),
  ("Victoria 1837", "Queen Victoria came to the throne of Great Britain in the year", " 1837", [" 1801", " 1860", " 1714"]),
  ("Wittenberg theses", "Martin Luther began the Reformation by posting his theses on a church door at", " Wittenberg", [" Worms", " Erfurt", " Mainz"]),
  ("Wellington", "The allied army at Waterloo was commanded by the British Duke of", " Wellington", [" Marlborough", " Cumberland", " York"]),
  ("Titanic 1912", "The great ocean liner Titanic struck an iceberg and sank in the year", " 1912", [" 1898", " 1905", " 1888"]),
  ("WWI began 1914", "The First World War broke out across Europe in the year", " 1914", [" 1905", " 1898", " 1888"]),
  ("Russian Rev 1917", "The Tsar of Russia was swept away by revolution in the year", " 1917", [" 1905", " 1881", " 1825"]),
  ("Versailles 1919", "The 1919 treaty that formally ended the First World War was signed at the palace of", " Versailles", [" Vienna", " Potsdam", " Trianon"]),
 ],
}


def all_items():
    """Yield dicts in the shared schema."""
    out = []
    for era, groups in (("post", POST), ("pre", PRE_CONTROL)):
        for topic, facts in groups.items():
            for (lab, prompt, true, twins) in facts:
                out.append({"era": era, "topic": topic, "label": lab,
                            "prompt": prompt, "true": true, "distractors": twins})
    return out
