# Ranke 1913 vs 1929 — "who is Adolf Hitler?" (100 free continuations each)

prompts: who_is = 'Who is Adolf Hitler?';  def = 'Adolf Hitler is'
sampling: temperature 0.8, top_p 0.95, repetition_penalty 1.3, 60 new tokens, seed 1929


## Prompt: 'Who is Adolf Hitler?'

### Themes present (share of 100 continuations that mention it)

| theme | 1913 | 1929 |
|:--|--:|--:|
| Nazi / National Socialist | 0% | 0% |
| Munich putsch / 1923 / treason | 4% | 11% |
| agitator / party leader / orator | 3% | 7% |
| Mein Kampf / author | 2% | 5% |
| Austria / Vienna / early life | 17% | 25% |
| antisemitism / race | 2% | 6% |
| professes ignorance / no record | 4% | 3% |
| POST-1929: Chancellor/Reich/dictator | 1% | 2% |
| POST-1929: WWII / Holocaust | 0% | 0% |

### What each sample is, in one bucket (mutually exclusive; sums to 100)

| bucket | 1913 | 1929 |
|:--|--:|--:|
| Nazi / putsch-era figure | 4 | 11 |
| German political figure (generic) | 3 | 6 |
| biographical (Austrian / early life) | 15 | 21 |
| professes ignorance | 3 | 2 |
| generic / confabulation | 74 | 58 |
| post-1929 (anachronistic) | 1 | 2 |

## Prompt: 'Adolf Hitler is'

### Themes present (share of 100 continuations that mention it)

| theme | 1913 | 1929 |
|:--|--:|--:|
| Nazi / National Socialist | 0% | 0% |
| Munich putsch / 1923 / treason | 0% | 17% |
| agitator / party leader / orator | 2% | 12% |
| Mein Kampf / author | 4% | 2% |
| Austria / Vienna / early life | 9% | 14% |
| antisemitism / race | 2% | 1% |
| professes ignorance / no record | 1% | 0% |
| POST-1929: Chancellor/Reich/dictator | 0% | 5% |
| POST-1929: WWII / Holocaust | 0% | 2% |

### What each sample is, in one bucket (mutually exclusive; sums to 100)

| bucket | 1913 | 1929 |
|:--|--:|--:|
| Nazi / putsch-era figure | 0 | 14 |
| German political figure (generic) | 2 | 11 |
| biographical (Austrian / early life) | 9 | 8 |
| professes ignorance | 1 | 0 |
| generic / confabulation | 88 | 60 |
| post-1929 (anachronistic) | 0 | 7 |

## Verbatim examples (first 6 of each, prompt = "Who is Adolf Hitler?")


**Ranke 1913:**
- “Who Is the man who has just been convicted of murder in New York and sentenced to death for killing two men with a hammer? He was an anarchist. 1t will be remembered that he, together with his brother. Gustave HINER, wer”
- “The word "Hillcr" is the name of a German engineer and scientist, who in recent years has made himself famous for his discovery Of process whereby diamonds can be produced from natural gas.  The first great achievement w”
- “And what was the object of his mission to this country at this time?  What are these men who have come here in large numbers and established themselves as socialists? They profess that they believe socialism will eventua”
- “He has been making some very interesting discovcries in the old world. and his latest story of discoveries made by him IN Russia Is one Of thrilling interest.  The great Russian novelist, Gogol. whose death was recently ”
- “What is he? What does the name mean to American readers? In a recent number of The Worlds Work Mr.. Hiller's biography, by himself and his biographer. has been published in full, with many illus trations.  He was born at”
- “He has been indicted by the United States grand jury at Indianapolis. Ind, on a charge of violating federal laws.  Hillers case will be heard in March and it was announced that District At torney Miller had sent for coun”

**Ranke 1929:**
- “Who Is the German General Ludendorff, who has been mentioned in connection with the presidency Of Germany and possibly that of Austria? These are questions which Americans will be asked before long. They cannot be answer”
- “The answer to that question, as well as the other one Who Is Frankenstein?" will be given by Charles M. Schwab in a series of lectures on international affairs at the City Club next week. Mr.. Schwab, who was born In Aus”
- “And what was the real purpose of his campaign in Germany?  What is he doing now. and who are his lieutenantsf He Is a lieutenant Of General Ludendorff, who has recently been retired from active service as head OF Ger man”
- “He who has made possible the German revolution and who seems to have been chosen by fate as its leader. What his real political position, what are his ambitions or ideals?  Hiller's record in Germany has been one Of bloo”
- “What kind of a man Is he and what do his views on the war mean for the United States?  He was born in 1896, at Oberam. mergau In Bavaria. His father was an artist Of some distinction and had been professor OF drawing in ”
- “He has been mentioned as the new leader of Ger many. The last two years have seen a remarkable change in German pol itics, especially since 1920, when he became head Of the Communist party. He was elected to the Reichsta”