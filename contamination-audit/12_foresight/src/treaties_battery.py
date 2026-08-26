# -*- coding: utf-8 -*-
"""Battery for the v2 memory detector. Treaties are the substrate because every one has a uniform SECOND attribute —
who signed it — so we can label "does the model actually remember this" from TWO independent checks (date + parties),
not one. REAL = genuine historical treaties spanning famous -> deeply obscure, all regions (each with a verified year;
bilateral ones also carry their signatories for the parties check; multilateral famous ones carry parties=None and are
labelled by the date alone). INVENTED = treaties that never existed (real event-free cities + plainly invented names).
Only treaties whose year (and, where used, signatories) I am confident of are included — no fabrication.
"""
# (name, year, "PartyA and PartyB" | None)   — None = famous multilateral, labelled by date alone
REAL = [
    ("Paris", 1783, "Britain and the United States"), ("Ghent", 1814, "Britain and the United States"),
    ("Amiens", 1802, "Britain and France"), ("Frankfurt", 1871, "France and Germany"),
    ("Tilsit", 1807, "France and Russia"), ("Campo Formio", 1797, "France and Austria"),
    ("Pressburg", 1805, "France and Austria"), ("Lunéville", 1801, "France and Austria"),
    ("Nanking", 1842, "Britain and China"), ("Portsmouth", 1905, "Russia and Japan"),
    ("Brest-Litovsk", 1918, "Germany and Russia"), ("Shimonoseki", 1895, "Japan and China"),
    ("San Stefano", 1878, "Russia and the Ottoman Empire"), ("Adrianople", 1829, "Russia and the Ottoman Empire"),
    ("Kiel", 1814, "Sweden and Denmark"), ("Nystad", 1721, "Russia and Sweden"),
    ("Guadalupe Hidalgo", 1848, "the United States and Mexico"),
    ("Utrecht", 1713, None), ("Versailles", 1919, None), ("Vienna", 1815, None), ("Aix-la-Chapelle", 1748, None),
    ("Ryswick", 1697, None), ("Berlin", 1878, None), ("Westphalia", 1648, None),
    # obscure, bilateral (the hard cases — parties do the work)
    ("Andrusovo", 1667, "Russia and Poland"), ("Roskilde", 1658, "Sweden and Denmark"),
    ("Stolbovo", 1617, "Russia and Sweden"), ("Passarowitz", 1718, "the Ottoman Empire and Austria"),
    ("Belgrade", 1739, "the Ottoman Empire and Austria"), ("Küçük Kaynarca", 1774, "Russia and the Ottoman Empire"),
    ("Jassy", 1792, "Russia and the Ottoman Empire"), ("Gulistan", 1813, "Russia and Persia"),
    ("Turkmenchay", 1828, "Russia and Persia"), ("Hünkâr İskelesi", 1833, "Russia and the Ottoman Empire"),
    ("Whampoa", 1844, "France and China"), ("Wanghia", 1844, "the United States and China"),
    ("Aigun", 1858, "Russia and China"), ("Nerchinsk", 1689, "Russia and China"),
    ("Björkö", 1905, "Germany and Russia"), ("Nöteborg", 1323, "Sweden and Novgorod"),
    ("Zuhab", 1639, "the Ottoman Empire and Persia"), ("Hubertusburg", 1763, "Prussia and Austria"),
    ("Bucharest", 1812, "Russia and the Ottoman Empire"), ("Karlowitz", 1699, "the Ottoman Empire and Austria"),
    ("Tientsin", 1858, "Britain and China"),
    # --- expanded obscure early-modern set (standard diplomatic history; signatories are the anchor for the label) ---
    ("Teusina", 1595, "Russia and Sweden"), ("Deulino", 1618, "Russia and Poland"),
    ("Polyanovka", 1634, "Russia and Poland"), ("Kardis", 1661, "Russia and Sweden"),
    ("Copenhagen", 1660, "Denmark and Sweden"), ("Oliva", 1660, "Sweden and Poland"),
    ("Travendal", 1700, "Sweden and Denmark"), ("Frederiksborg", 1720, "Denmark and Sweden"),
    ("Åbo", 1743, "Russia and Sweden"), ("Zsitvatorok", 1606, "the Ottoman Empire and Austria"),
    ("Vasvár", 1664, "the Ottoman Empire and Austria"), ("Sistova", 1791, "the Ottoman Empire and Austria"),
    ("Khotyn", 1621, "the Ottoman Empire and Poland"), ("Buchach", 1672, "the Ottoman Empire and Poland"),
    ("Żurawno", 1676, "the Ottoman Empire and Poland"), ("Amasya", 1555, "the Ottoman Empire and Persia"),
    ("Bakhchisarai", 1681, "Russia and the Ottoman Empire"), ("Pruth", 1711, "Russia and the Ottoman Empire"),
    ("Niš", 1739, "Russia and the Ottoman Empire"), ("Breslau", 1742, "Prussia and Austria"),
    ("Dresden", 1745, "Prussia and Austria"), ("Teschen", 1779, "Prussia and Austria"),
    ("Reichenbach", 1790, "Prussia and Austria"), ("Rastatt", 1714, "France and Austria"),
    ("Dover", 1670, "England and France"), ("Pyrenees", 1659, "France and Spain"),
    ("Seville", 1729, "Britain and Spain"), ("Badajoz", 1801, "Spain and Portugal"),
    ("San Ildefonso", 1777, "Spain and Portugal"), ("Breda", 1667, "England and the Dutch Republic"),
    ("Basel", 1795, "France and Prussia"),
]

# INVENTED treaties — none ever existed. Real event-free cities + plainly invented place-names. Year only sets the
# scoring window; these are NO-MEMORY by construction.
_FAKE_CITY = ["Bordeaux", "Rotterdam", "Glasgow", "Aberdeen", "Plymouth", "Cardiff", "Leeds", "Dundee", "Swansea",
              "Cork", "Rennes", "Bristol", "Grenoble", "Marseille", "Toulouse", "Le Havre", "Strasbourg", "Rouen",
              "Nancy", "Dijon", "Bremen", "Antwerp", "Lyon", "Nice", "Bilbao", "Porto", "Bergen", "Malmö",
              "Gdansk", "Riga", "Naples", "Turin", "Genoa", "Valencia", "Coimbra", "Cologne", "Hamburg", "Stuttgart"]
_FAKE_NAME = ["Kesselheim", "Marnburg", "Astorheim", "Brenzau", "Kaltenfeld", "Ravensmoor", "Vornau", "Kessendorf",
              "Halbrecht", "Eldreth", "Wolmirstadt", "Traunfels", "Reichenmoor", "Saldenburg", "Mornhausen",
              "Veldenstein", "Kranzberg", "Osterwald"]
_yrs = list(range(1600, 1920, 6))
INVENTED = [(c, _yrs[i % len(_yrs)]) for i, c in enumerate(_FAKE_CITY + _FAKE_NAME)]

# forced-choice option set for the "who signed it" check: every real pair used, plus a few distractors
_used = sorted({p for _, _, p in REAL if p})
PARTY_OPTIONS = _used + ["Denmark and Norway", "Portugal and the Netherlands", "Prussia and Denmark", "Austria and Venice"]
