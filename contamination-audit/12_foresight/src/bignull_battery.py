# -*- coding: utf-8 -*-
"""
BIG multi-category battery for the realness-leak stress test (data only, no model).
Real entities verified against Wikipedia (research agents, 2026-07-10); invented entities verified
NON-existent (research agent). Each item is a dated prompt; MEMORY is measured behaviourally later
(studied vs blank), REALNESS is known here (real entity vs invented).

Classes:
  famous   : real, well-known  -> the model likely KNOWS the date (positives)
  obscure  : real, minor       -> the model likely BLANK (the enriched real-but-unknown "Stack C")
  invented : fake, surface-matched -> blank + not real (negatives)
  false    : real famous NAME in a FALSE frame (non-presidents as president) -> real name, no such memory
All dates < 1929 (the model's era), so within a category the only thing separating famous from obscure
is fame/exposure, not era. Templates end in "in the year" so the date-spike + activation extraction work.
"""

TEMPLATES = {
    "war":      "The {} broke out in the year",
    "nobel":    "{} was awarded the Nobel Prize in the year",
    "painter":  "The painter {} was born in the year",
    "novelist": "The novelist {} was born in the year",
    "president": "{} became President of the United States in the year",
}

# =================== WARS (start year) ===================
WAR_FAMOUS = [("Franco-Prussian War", 1870), ("Crimean War", 1853), ("Napoleonic Wars", 1803),
    ("American Civil War", 1861), ("Second Boer War", 1899), ("Thirty Years' War", 1618),
    ("American Revolutionary War", 1775), ("Spanish–American War", 1898), ("Russo-Japanese War", 1904),
    ("World War I", 1914), ("Seven Years' War", 1756), ("Mexican–American War", 1846),
    ("English Civil War", 1642), ("Wars of the Roses", 1455), ("Russian Civil War", 1917),
    ("First Sino-Japanese War", 1894), ("Hundred Years' War", 1337), ("Balkan Wars", 1912)]
WAR_OBSCURE = [("War of the Bavarian Succession", 1778), ("Anglo-Zanzibar War", 1896),
    ("War of Jenkins' Ear", 1739), ("Klang War", 1866), ("Toledo War", 1835), ("Pig War", 1859),
    ("Cape Frontier Wars", 1779), ("Pastry War", 1838), ("War of the Oranges", 1801), ("Aroostook War", 1838),
    ("War of the Golden Stool", 1900), ("Anglo-Aro War", 1901), ("Utah War", 1857), ("War of the Stray Dog", 1925),
    ("War of the Confederation", 1836), ("Sonderbund War", 1847), ("Padri War", 1803), ("Java War", 1825),
    ("War of Devolution", 1667), ("War of the Quadruple Alliance", 1718), ("War of the Polish Succession", 1733),
    ("Kalmar War", 1611)]
WAR_INVENTED = [("War of the Rhenish Succession", 1662), ("Anglo-Moravian War", 1688),
    ("War of the Cornish Succession", 1701), ("War of the Lusatian Succession", 1713), ("Anglo-Lombard War", 1725),
    ("Anglo-Wallachian War", 1738), ("War of the Frisian Succession", 1749), ("Anglo-Savoyard War", 1756),
    ("War of the Bergamasque Succession", 1763), ("War of the Valdostan Succession", 1774),
    ("War of the Roussillon Succession", 1785), ("War of the Carniolan Succession", 1792), ("Anglo-Bavarian War", 1803),
    ("Anglo-Estonian War", 1812), ("War of the Dalmatian Succession", 1821), ("Anglo-Genoese War", 1834),
    ("War of the Vorarlberg Succession", 1848), ("Anglo-Ligurian War", 1859), ("War of the Auvergnat Succession", 1866),
    ("War of the Picard Succession", 1877)]

# =================== NOBEL laureates (award year) ===================
NOBEL_FAMOUS = [("Wilhelm Röntgen", 1901), ("Henri Becquerel", 1903), ("Pierre Curie", 1903), ("Marie Curie", 1903),
    ("Ivan Pavlov", 1904), ("Robert Koch", 1905), ("Theodore Roosevelt", 1906), ("Rudyard Kipling", 1907),
    ("Ernest Rutherford", 1908), ("Guglielmo Marconi", 1909), ("Selma Lagerlöf", 1909), ("Rabindranath Tagore", 1913),
    ("Romain Rolland", 1915), ("Max Planck", 1918), ("Fritz Haber", 1918), ("Woodrow Wilson", 1919),
    ("Knut Hamsun", 1920), ("Albert Einstein", 1921), ("Anatole France", 1921), ("Niels Bohr", 1922),
    ("William Butler Yeats", 1923), ("George Bernard Shaw", 1925), ("Henry Dunant", 1901)]
NOBEL_OBSCURE = [("Sully Prudhomme", 1901), ("Élie Ducommun", 1902), ("Charles Albert Gobat", 1902),
    ("Bjørnstjerne Bjørnson", 1903), ("Frédéric Mistral", 1904), ("José Echegaray", 1904), ("Giosuè Carducci", 1906),
    ("Eduard Buchner", 1907), ("Rudolf Christoph Eucken", 1908), ("Gabriel Lippmann", 1908), ("Klas Pontus Arnoldson", 1908),
    ("Paul Heyse", 1910), ("Johannes Diderik van der Waals", 1910), ("Allvar Gullstrand", 1911), ("Nils Gustaf Dalén", 1912),
    ("Charles Richet", 1913), ("Robert Bárány", 1914), ("Richard Willstätter", 1915), ("Carl Spitteler", 1919),
    ("Charles Édouard Guillaume", 1920), ("Jacinto Benavente", 1922), ("Manne Siegbahn", 1924), ("Grazia Deledda", 1926),
    ("Johannes Fibiger", 1926)]
NOBEL_INVENTED = [("Aldous Brennt", 1902), ("Emeric Tolliver", 1904), ("Casimir Vantorre", 1905), ("Thorvald Mesk", 1907),
    ("Reinald Osterhues", 1909), ("Matthias Vellinga", 1911), ("Hendrik Salvoort", 1913), ("Aurelio Fenn", 1915),
    ("Lucien Vaandrager", 1917), ("Jasper Wendtland", 1919), ("Emil Norrbo", 1921), ("Nils Ekvall", 1923),
    ("Anselm Doryk", 1925), ("Viktor Salmensuu", 1927)]

# =================== PAINTERS (birth year) ===================
PAINTER_FAMOUS = [("Leonardo da Vinci", 1452), ("Albrecht Dürer", 1471), ("Michelangelo", 1475), ("Raphael", 1483),
    ("El Greco", 1541), ("Caravaggio", 1571), ("Peter Paul Rubens", 1577), ("Diego Velázquez", 1599),
    ("Rembrandt", 1606), ("Johannes Vermeer", 1632), ("Francisco Goya", 1746), ("Katsushika Hokusai", 1760),
    ("J.M.W. Turner", 1775), ("Édouard Manet", 1832), ("Edgar Degas", 1834), ("Paul Cézanne", 1839),
    ("Claude Monet", 1840), ("Pierre-Auguste Renoir", 1841), ("Paul Gauguin", 1848), ("Vincent van Gogh", 1853),
    ("Gustav Klimt", 1862), ("Wassily Kandinsky", 1866), ("Henri Matisse", 1869), ("Pablo Picasso", 1881),
    ("Georgia O'Keeffe", 1887), ("Salvador Dalí", 1904)]
PAINTER_OBSCURE = [("Cornelis van Poelenburgh", 1594), ("Adriaen van Ostade", 1610), ("Bartholomeus van der Helst", 1613),
    ("Emanuel de Witte", 1617), ("Willem Kalf", 1619), ("Philips Wouwerman", 1619), ("Nicolaes Berchem", 1620),
    ("Carel Fabritius", 1622), ("Frans van Mieris the Elder", 1635), ("Caspar Netscher", 1639), ("Godfried Schalcken", 1643),
    ("Ernest Meissonier", 1815), ("Thomas Couture", 1815), ("Augustus Egg", 1816), ("William Powell Frith", 1819),
    ("Alexandre Cabanel", 1823), ("Edwin Long", 1829), ("Vasily Perov", 1834), ("Marcus Stone", 1840),
    ("Albert Joseph Moore", 1841), ("Solomon J. Solomon", 1860), ("John William Godward", 1861)]
PAINTER_INVENTED = [("Cornelis Vandeghast", 1602), ("Bartolomeo Ferren", 1618), ("Adriaen Wouterloot", 1631),
    ("Matthias Grunholt", 1644), ("Piet Roelvink", 1657), ("Lorenzo Cavagni", 1670), ("Willem de Grave", 1688),
    ("Sebastiano Falchetti", 1701), ("Frans Oosterlinck", 1722), ("Jacobus Reindert", 1745), ("Aert van Sloten", 1768),
    ("Niccolo Sarpetti", 1790), ("Karel Munsterhuis", 1815), ("Rutger Blomhoff", 1840)]

# =================== NOVELISTS (birth year) ===================
NOVELIST_FAMOUS = [("Charles Dickens", 1812), ("Jane Austen", 1775), ("Leo Tolstoy", 1828), ("Mark Twain", 1835),
    ("Victor Hugo", 1802), ("Fyodor Dostoevsky", 1821), ("Charlotte Brontë", 1816), ("Emily Brontë", 1818),
    ("Herman Melville", 1819), ("Nathaniel Hawthorne", 1804), ("George Eliot", 1819), ("Gustave Flaubert", 1821),
    ("Thomas Hardy", 1840), ("Oscar Wilde", 1854), ("Jules Verne", 1828), ("Alexandre Dumas", 1802),
    ("Henry James", 1843), ("Joseph Conrad", 1857), ("Virginia Woolf", 1882), ("James Joyce", 1882),
    ("Ernest Hemingway", 1899), ("F. Scott Fitzgerald", 1896), ("Franz Kafka", 1883)]
NOVELIST_OBSCURE = [("Mary Elizabeth Braddon", 1835), ("Charlotte Mary Yonge", 1823), ("Fanny Burney", 1752),
    ("Ann Radcliffe", 1764), ("R. D. Blackmore", 1825), ("Margaret Oliphant", 1828), ("Dinah Craik", 1826),
    ("Ellen Wood", 1814), ("Ouida", 1839), ("Charles Reade", 1814), ("George Gissing", 1857),
    ("Sabine Baring-Gould", 1834), ("Rhoda Broughton", 1840), ("Elizabeth Inchbald", 1753), ("Susan Ferrier", 1782),
    ("Catherine Gore", 1798), ("Grant Allen", 1848), ("William Harrison Ainsworth", 1805), ("Frances Milton Trollope", 1779),
    ("Amelia B. Edwards", 1831), ("Mary Augusta Ward", 1851), ("Flora Annie Steel", 1847)]
NOVELIST_INVENTED = [("Marcus Ashwell", 1742), ("Julian Hartsook", 1758), ("Miriam Caldstone", 1771),
    ("Edmund Farrsley", 1784), ("Beatrix Vennor", 1799), ("Henry Alderwick", 1810), ("Clara Mossbeck", 1821),
    ("Rupert Halbane", 1830), ("Isolde Marrable", 1839), ("Gideon Thornby", 1848), ("Cecily Harnwell", 1855),
    ("Vivian Aldrington", 1863), ("Nathaniel Crulls", 1872), ("Lucian Farthing", 1885)]

# =================== PRESIDENTS (real, inauguration year) + NON-PRESIDENTS (false frame) ===================
PRESIDENT_REAL = [("George Washington", 1789), ("Thomas Jefferson", 1801), ("James Madison", 1809), ("James Monroe", 1817),
    ("Andrew Jackson", 1829), ("Martin Van Buren", 1837), ("James K. Polk", 1845), ("Millard Fillmore", 1850),
    ("Franklin Pierce", 1853), ("James Buchanan", 1857), ("Abraham Lincoln", 1861), ("Ulysses S. Grant", 1869),
    ("Rutherford B. Hayes", 1877), ("Chester A. Arthur", 1881), ("Grover Cleveland", 1885), ("Benjamin Harrison", 1889),
    ("William McKinley", 1897), ("Theodore Roosevelt", 1901), ("William Howard Taft", 1909), ("Woodrow Wilson", 1913),
    ("Warren G. Harding", 1921), ("Calvin Coolidge", 1923)]
# FALSE-FACT: real famous Americans who were NEVER president, framed as president -> real NAME, no such memory.
PRESIDENT_FALSE = [("Benjamin Franklin", 1785), ("Alexander Hamilton", 1793), ("Frederick Douglass", 1870),
    ("Mark Twain", 1885), ("Thomas Edison", 1900), ("Henry Ford", 1915), ("Andrew Carnegie", 1895),
    ("Susan B. Anthony", 1875), ("Daniel Webster", 1840), ("Henry Clay", 1830), ("John C. Calhoun", 1825),
    ("William Jennings Bryan", 1900), ("Aaron Burr", 1805), ("P. T. Barnum", 1870)]


def build():
    """Return a flat list of items: dict(cat, name, year, real_entity, cls)."""
    items = []
    def add(cat, lst, real_entity, cls):
        for name, yr in lst:
            items.append({"cat": cat, "name": name, "year": yr, "real_entity": real_entity, "cls": cls})
    add("war", WAR_FAMOUS, True, "famous"); add("war", WAR_OBSCURE, True, "obscure"); add("war", WAR_INVENTED, False, "invented")
    add("nobel", NOBEL_FAMOUS, True, "famous"); add("nobel", NOBEL_OBSCURE, True, "obscure"); add("nobel", NOBEL_INVENTED, False, "invented")
    add("painter", PAINTER_FAMOUS, True, "famous"); add("painter", PAINTER_OBSCURE, True, "obscure"); add("painter", PAINTER_INVENTED, False, "invented")
    add("novelist", NOVELIST_FAMOUS, True, "famous"); add("novelist", NOVELIST_OBSCURE, True, "obscure"); add("novelist", NOVELIST_INVENTED, False, "invented")
    add("president", PRESIDENT_REAL, True, "famous"); add("president", PRESIDENT_FALSE, True, "false")
    return items


def prompt(item):
    return TEMPLATES[item["cat"]].format(item["name"])


if __name__ == "__main__":
    it = build()
    from collections import Counter
    print(f"total items: {len(it)}")
    print("by class:", dict(Counter(i["cls"] for i in it)))
    print("real entities:", sum(i["real_entity"] for i in it), " invented:", sum(not i["real_entity"] for i in it))
    print("by category:", dict(Counter(i["cat"] for i in it)))
    print("\nexamples:")
    for i in (it[0], it[18], it[40], it[100], it[-1]):
        print(f"  [{i['cls']:8s} {i['cat']:9s} real={i['real_entity']}]  {prompt(i)} ___  (true {i['year']})")
