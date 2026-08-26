"""DECISIVE re-test of US politics with NO first-name (or any) cue to the answer -- to see if the
66% was real knowledge or just name-completion (as Russia/Germany turned out to be). Answers are
produced from the descriptive fact alone; coined terms / years preferred. DeepSeek then screens."""

CANDIDATES = [
 ("US politics", "Atomic bombing of Japan", 1945, [
   ("The U.S. president who ordered the atomic bombing of Japan was", " Truman", [" Dewey", " Wallace", " Hoover", " Stimson"]),
   ("The atomic bomb dropped on Hiroshima was nicknamed", " Little Boy", [" Fat Man", " Thin Man", " Big Boy"]),
   ("The atomic bomb dropped on Nagasaki was nicknamed", " Fat Man", [" Little Boy", " Thin Man", " Big Boy"]),
   ("The B-29 aircraft that dropped the Hiroshima bomb was the", " Enola Gay", [" Bockscar", " Big Stink", " Straight Flush"]),
   ("The B-29 that dropped the Nagasaki bomb was named", " Bockscar", [" Enola Gay", " Big Stink", " Full House"]),
   ("The atomic bombings of Japan took place in the year 19", " 45", [" 41", " 43", " 47"]),
   ("The U.S. project that secretly built the atomic bomb was the Manhattan", " Project", [" Program", " Mission", " Venture"]),
   ("The physicist who scientifically directed the atomic-bomb laboratory was", " Oppenheimer", [" Lawrence", " Compton", " Bethe", " Teller"]),
 ]),
 ("US politics", "New Deal / FDR", 1933, [
   ("The 1930s U.S. economic recovery program was called the New", " Deal", [" Pact", " Order", " Way"]),
   ("The vice president who became president on Roosevelt's death in 1945 was", " Truman", [" Wallace", " Garner", " Barkley"]),
   ("The U.S. president was elected to an unprecedented number of terms in the 1930s-40s:", " four", [" three", " seven", " nine"]),
   ("The 1935 New Deal program providing old-age pensions was called Social", " Security", [" Welfare", " Insurance", " Provision"]),
   ("Roosevelt's radio addresses to the nation were known as fireside", " chats", [" talks", " hours", " notes"]),
 ]),
 ("US politics", "Kennedy", 1961, [
   ("The U.S. president inaugurated in January 1961 was", " Kennedy", [" Humphrey", " Symington", " Stevenson"]),
   ("President Kennedy was assassinated in the year 19", " 63", [" 61", " 65", " 68"]),
   ("The man who assassinated President Kennedy in 1963 was", " Oswald", [" Sirhan", " Chapman", " Bremer"]),
   ("The 1962 nuclear confrontation between the U.S. and the USSR was the Cuban Missile", " Crisis", [" Standoff", " Affair", " Incident"]),
   ("The vice president who became president after Kennedy's assassination was", " Johnson", [" Humphrey", " Goldwater", " Symington"]),
 ]),
 ("US politics", "Nixon / Watergate", 1974, [
   ("The U.S. president who resigned from office in 1974 was", " Nixon", [" Humphrey", " Agnew", " Rockefeller"]),
   ("The U.S. president resigned over the scandal in the year 19", " 74", [" 68", " 72", " 76"]),
   ("The political scandal that forced a U.S. president to resign in 1974 was", " Watergate", [" Teapot", " Whitewater", " Irangate"]),
   ("The vice president who became president when Nixon resigned was", " Ford", [" Agnew", " Rockefeller", " Connally"]),
 ]),
 ("US politics", "Reagan", 1981, [
   ("The former Hollywood actor elected U.S. president in 1980 was", " Reagan", [" Anderson", " Mondale", " Connally"]),
   ("This actor-turned-president took office in the year 19", " 81", [" 77", " 85", " 89"]),
   ("Reagan's supply-side economic policy was nicknamed", " Reaganomics", [" Nixonomics", " Bushonomics", " Clintonomics"]),
   ("Reagan's proposed space-based missile defence was nicknamed Star", " Wars", [" Trek", " Shield", " Guard"]),
 ]),
 ("US politics", "MLK / Civil Rights", 1963, [
   ("The civil-rights leader who gave the 1963 'I Have a Dream' speech was", " King", [" Abernathy", " Wilkins", " Randolph"]),
   ("This civil-rights leader was assassinated in the year 19", " 68", [" 63", " 65", " 72"]),
   ("The woman whose 1955 refusal to give up a bus seat sparked the Montgomery boycott was", " Parks", [" Baker", " Colvin", " Nash"]),
 ]),
 ("US politics", "Vietnam War", 1965, [
   ("The U.S. president who escalated the Vietnam War in 1965 was", " Johnson", [" Humphrey", " Goldwater", " Mansfield"]),
   ("The 1968 surprise offensive by the North in Vietnam was called the", " Tet", [" Tonkin", " Hue", " Linebacker"]),
   ("Nixon's policy of handing combat to South Vietnamese forces was called", " Vietnamization", [" Detente", " Containment", " Pacification"]),
   ("Saigon fell, ending the Vietnam War, in the year 19", " 75", [" 65", " 68", " 73"]),
 ]),
 ("US politics", "Obama / recent", 2009, [
   ("The first African-American U.S. president, elected in 2008, was", " Obama", [" Edwards", " Jackson", " Powell"]),
   ("This president's signature 2010 health-care law was nicknamed", " Obamacare", [" Trumpcare", " Hillarycare", " Romneycare"]),
   ("The vice president under Obama who later became president was", " Biden", [" Kaine", " Kerry", " Dodd"]),
   ("Obama took office as president in the year 20", " 09", [" 05", " 13", " 01"]),
 ]),
]
