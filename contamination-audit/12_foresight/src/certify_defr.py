# -*- coding: utf-8 -*-
"""GERMAN + FRENCH certification of the chronicle meter. Same 20 held-out questions, same locked
rule, same pre-registered bars (C1 coverage >=14/20, C2 knowledge >=80%, C3 gradient >=80%),
scored per language. ONE declared adaptation: the NAME/PHRASE type of each item is inherited from
the English design (German capitalizes all nouns, breaking the mechanical classifier).
Naming divergence handled openly: the 1916 sea battle is 'Jutland' in French usage but the
'Skagerrak' in German usage — each language's truth follows its own corpus convention.
Run: python src/certify_defr.py
"""
import os, json, math, statistics as st
import common
from foresight_bank import RANKE

# item: name, win, qtype, per-lang dict(truth, neutral, stems[5], options[4])
Q = [
 dict(name="tsushima", win="A", qtype="name",
  de=dict(truth=0, neutral="Sie wurde geschlagen in der Straße von",
   stems=["Es ist das Jahr 1906. Die große Seeschlacht des ostasiatischen Krieges wurde im vorigen Jahr geschlagen in der Straße von",
          "Man schreibt das Jahr 1906. Die russische Flotte wurde im vorigen Mai vernichtet in der Straße von",
          "Es ist 1906. Die entscheidende Seeschlacht zwischen Russland und Japan fiel in der Straße von",
          "Man schreibt 1906. Die baltische Flotte fand, nach halber Weltumsegelung, ihr Ende in der Straße von",
          "Es ist 1906. Der große Seesieg des letzten Krieges wurde errungen in der Straße von"],
   options=[" Tsushima", " Formosa", " Malakka", " Korea"]),
  fr=dict(truth=0, neutral="Elle fut livrée dans le détroit de",
   stems=["Nous sommes en 1906. La grande bataille navale de la guerre d'Orient fut livrée l'an dernier dans le détroit de",
          "L'an est 1906. La flotte russe fut anéantie en mai dernier dans le détroit de",
          "Nous sommes en 1906. La bataille décisive entre la Russie et le Japon se livra dans le détroit de",
          "L'an est 1906. La flotte de la Baltique, après un demi tour du monde, périt dans le détroit de",
          "Nous sommes en 1906. La grande victoire navale de la dernière guerre fut remportée dans le détroit de"],
   options=[" Tsushima", " Formose", " Malacca", " Corée"])),
 dict(name="portsmouth", win="A", qtype="name",
  de=dict(truth=0, neutral="Er wurde unterzeichnet zu",
   stems=["Es ist das Jahr 1906. Der Friede zwischen Russland und Japan wurde im vorigen Jahr unterzeichnet zu",
          "Man schreibt das Jahr 1906. Der Vertrag, der den ostasiatischen Krieg beendete, wurde geschlossen zu",
          "Es ist 1906. Russland und Japan schlossen ihren Frieden im vorigen September zu",
          "Man schreibt 1906. Der Krieg im Fernen Osten endete mit dem Frieden von",
          "Es ist 1906. Die Friedenskonferenz nach dem russisch-japanischen Krieg tagte zu"],
   options=[" Portsmouth", " Washington", " Genf", " Den Haag"]),
  fr=dict(truth=0, neutral="Elle fut signée à",
   stems=["Nous sommes en 1906. La paix entre la Russie et le Japon fut signée l'an dernier à",
          "L'an est 1906. Le traité qui mit fin à la guerre d'Orient fut conclu à",
          "Nous sommes en 1906. La Russie et le Japon firent leur paix en septembre dernier à",
          "L'an est 1906. La guerre d'Extrême-Orient s'acheva par le traité de",
          "Nous sommes en 1906. La conférence de paix russo-japonaise siégea à"],
   options=[" Portsmouth", " Washington", " Genève", " La Haye"])),
 dict(name="king_george", win="A", qtype="name",
  de=dict(truth=0, neutral="Er ist",
   stems=["Es ist das Jahr 1912. Der König, der im vorigen Sommer in England gekrönt wurde, ist",
          "Man schreibt das Jahr 1912. Englands neuer König, im vorigen Jahr in Westminster gekrönt, ist",
          "Es ist 1912. Der Herrscher, der nach dem Tode seines Vaters den englischen Thron bestieg, ist",
          "Man schreibt 1912. Der König von England heißt heute",
          "Es ist 1912. Auf dem englischen Throne sitzt jetzt"],
   options=[" König Georg", " König Eduard", " König Albert", " König Heinrich"]),
  fr=dict(truth=0, neutral="C'est",
   stems=["Nous sommes en 1912. Le roi couronné en Angleterre l'été dernier est",
          "L'an est 1912. Le nouveau roi d'Angleterre, couronné à Westminster l'an dernier, est",
          "Nous sommes en 1912. Le souverain monté sur le trône d'Angleterre à la mort de son père est",
          "L'an est 1912. Le roi d'Angleterre est aujourd'hui",
          "Nous sommes en 1912. Sur le trône d'Angleterre siège maintenant"],
   options=[" le roi George", " le roi Édouard", " le roi Albert", " le roi Henri"])),
 dict(name="dreadnought", win="A", qtype="name",
  de=dict(truth=0, neutral="Sie hieß",
   stems=["Es ist das Jahr 1908. Das umwälzende britische Großkampfschiff, vor zwei Jahren vom Stapel gelaufen, hieß",
          "Man schreibt das Jahr 1908. Das Schlachtschiff, das alle älteren Schiffe veralten ließ, hieß",
          "Es ist 1908. Das große neue britische Schlachtschiff, nur mit schweren Geschützen bestückt, hieß",
          "Man schreibt 1908. Das Schiff, nach dem die neue Klasse der Riesenschlachtschiffe benannt ist, heißt",
          "Es ist 1908. Das berühmte neue Schlachtschiff von 1906 hieß"],
   options=[" Dreadnought", " Invincible", " Majestic", " Formidable"]),
  fr=dict(truth=0, neutral="Il s'appelait le",
   stems=["Nous sommes en 1908. Le cuirassé révolutionnaire lancé par l'Angleterre il y a deux ans s'appelait le",
          "L'an est 1908. Le navire de guerre qui rendit vieux tous les cuirassés anciens s'appelait le",
          "Nous sommes en 1908. Le grand cuirassé nouveau, armé de gros canons seulement, se nommait le",
          "L'an est 1908. Le navire qui donne son nom à la nouvelle classe des cuirassés géants est le",
          "Nous sommes en 1908. Le fameux cuirassé lancé à Portsmouth en 1906 était le"],
   options=[" Dreadnought", " Invincible", " Majestic", " Formidable"])),
 dict(name="bleriot", win="A", qtype="name", flag="Latham = the rival who failed",
  de=dict(truth=0, neutral="Es war",
   stems=["Es ist das Jahr 1910. Der erste Mensch, der den Ärmelkanal überflog, war",
          "Man schreibt das Jahr 1910. Der Kanal wurde im vorigen Sommer zuerst durchflogen von",
          "Es ist 1910. Der Flieger, der im vorigen Juli von Frankreich nach England flog, war",
          "Man schreibt 1910. Den Preis für den ersten Kanalflug gewann",
          "Es ist 1910. Die erste Überquerung des Kanals im Aeroplan gelang"],
   options=[" Herrn Blériot", " Herrn Latham", " Herrn Farman", " Herrn Santos-Dumont"]),
  fr=dict(truth=0, neutral="Ce fut",
   stems=["Nous sommes en 1910. Le premier homme à traverser la Manche en aéroplane fut",
          "L'an est 1910. La Manche fut franchie par les airs, l'été dernier, d'abord par",
          "Nous sommes en 1910. L'aviateur qui vola de France en Angleterre en juillet dernier fut",
          "L'an est 1910. Le prix de la première traversée de la Manche fut gagné par",
          "Nous sommes en 1910. La première traversée aérienne de la Manche fut l'œuvre de"],
   options=[" Monsieur Blériot", " Monsieur Latham", " Monsieur Farman", " Monsieur Santos-Dumont"])),
 dict(name="young_turk", win="A", qtype="name",
  de=dict(truth=0, neutral="Das Land war",
   stems=["Es ist das Jahr 1909. Junge Offiziere zwangen im vorigen Jahr den Sultan, die Verfassung wiederherzustellen. Das Land war",
          "Man schreibt das Jahr 1909. Die Revolution, die einem alten Reich sein Parlament zurückgab, geschah in einem Land, und dieses Land war",
          "Es ist 1909. Der Sultan musste im vorigen Jahr eine Verfassung gewähren. Sein Reich war",
          "Man schreibt 1909. Die jungen Offiziere, die sich gegen ihren Sultan erhoben, dienten einem Reich, und das war",
          "Es ist 1909. Die Verfassungsrevolution des vorigen Jahres ereignete sich in einem Land, nämlich"],
   options=[" der Türkei", " Persien", " Ägypten", " Marokko"]),
  fr=dict(truth=0, neutral="Ce pays était",
   stems=["Nous sommes en 1909. De jeunes officiers ont forcé l'an dernier le Sultan à rétablir la constitution. Ce pays était",
          "L'an est 1909. La révolution qui rendit un parlement à un vieil empire eut lieu dans un pays, et ce pays était",
          "Nous sommes en 1909. Le Sultan dut accorder l'an dernier une constitution. Son empire était",
          "L'an est 1909. Les jeunes officiers levés contre leur Sultan servaient un empire, à savoir",
          "Nous sommes en 1909. La révolution constitutionnelle de l'an dernier se produisit dans un pays, savoir"],
   options=[" la Turquie", " la Perse", " l'Égypte", " le Maroc"])),
 dict(name="tripoli_war", win="A", qtype="name",
  de=dict(truth=0, neutral="Er wurde gewonnen von",
   stems=["Es ist das Jahr 1913. Der Krieg um Tripolis wurde gewonnen von",
          "Man schreibt das Jahr 1913. Tripolis wurde im letzten Krieg erobert von",
          "Es ist 1913. Der afrikanische Krieg von 1911 und 1912 gab Tripolis an",
          "Man schreibt 1913. Die Macht, die den Türken Tripolis entriss, war",
          "Es ist 1913. Der Krieg um die libysche Küste endete mit dem Sieg von"],
   options=[" Italien", " Frankreich", " Spanien", " Griechenland"]),
  fr=dict(truth=0, neutral="Elle fut gagnée par",
   stems=["Nous sommes en 1913. La guerre pour Tripoli fut gagnée par",
          "L'an est 1913. Tripoli fut conquise dans la dernière guerre par",
          "Nous sommes en 1913. La guerre d'Afrique de 1911 et 1912 donna Tripoli à",
          "L'an est 1913. La puissance qui arracha Tripoli aux Turcs fut",
          "Nous sommes en 1913. La guerre pour la côte de Libye finit par la victoire de"],
   options=[" l'Italie", " la France", " l'Espagne", " la Grèce"])),
 dict(name="panama_route", win="A", qtype="name", flag="Nicaragua = the leading route of the 1900 debate",
  de=dict(truth=0, neutral="Er durchquert",
   stems=["Es ist das Jahr 1906. Der große Kanal, den die Amerikaner zwischen den Meeren graben, durchquert",
          "Man schreibt das Jahr 1906. Der Isthmuskanal der Amerikaner läuft durch",
          "Es ist 1906. Die von den Amerikanern gewählte Kanalroute liegt in",
          "Man schreibt 1906. Der Kanal zwischen Atlantik und Pazifik wird gegraben durch",
          "Es ist 1906. Der amerikanische Kanal zwischen den Ozeanen durchschneidet"],
   options=[" Panama", " Nicaragua", " Tehuantepec", " Darién"]),
  fr=dict(truth=0, neutral="Il traverse",
   stems=["Nous sommes en 1906. Le grand canal que creusent les Américains entre les deux mers traverse",
          "L'an est 1906. Le canal isthmique des Américains passe par",
          "Nous sommes en 1906. La route choisie par les Américains pour leur canal traverse",
          "L'an est 1906. Le canal joignant l'Atlantique au Pacifique se creuse à travers",
          "Nous sommes en 1906. Le canal américain entre les océans coupe"],
   options=[" Panama", " le Nicaragua", " Tehuantepec", " le Darién"])),
 dict(name="china_now", win="A", qtype="phrase", flag="1900-salient expectation = partition",
  de=dict(truth=0, neutral="Es ist jetzt",
   stems=["Es ist das Jahr 1913. China ist nach seiner Revolution jetzt",
          "Man schreibt das Jahr 1913. Das chinesische Reich ist seit seiner Revolution",
          "Es ist 1913. Nach dem Sturz der Dynastie steht China heute da als",
          "Man schreibt 1913. Seit der Revolution ist China",
          "Es ist 1913. Das alte chinesische Reich ist geworden zu"],
   options=[" eine Republik unter einem Präsidenten", " ein Kaiserreich unter neuer Dynastie",
            " ein unter den Mächten geteiltes Land", " ein Schutzgebiet Japans"]),
  fr=dict(truth=0, neutral="C'est désormais",
   stems=["Nous sommes en 1913. La Chine, après sa révolution, est désormais",
          "L'an est 1913. L'empire chinois est devenu, depuis sa révolution,",
          "Nous sommes en 1913. Après la chute de la dynastie, la Chine est aujourd'hui",
          "L'an est 1913. Depuis la révolution, la Chine est",
          "Nous sommes en 1913. Le vieil empire chinois s'est changé en"],
   options=[" une république sous un président", " un empire sous une dynastie nouvelle",
            " un pays partagé entre les puissances", " une dépendance du Japon"])),
 dict(name="verdun", win="B", qtype="name", flag="Belfort = the famous fortress of 1870",
  de=dict(truth=0, neutral="Sie war",
   stems=["Es ist das Jahr 1917. Die Festungsstadt, die die Deutschen im vorigen Jahr zehn Monate lang berannten, war",
          "Man schreibt das Jahr 1917. Die große Schlacht des vorigen Jahres tobte vor",
          "Es ist 1917. Die französische Festung, die dem deutschen Ansturm durch 1916 standhielt, war",
          "Man schreibt 1917. Die längste Schlacht des Krieges wurde im vorigen Jahr geschlagen vor",
          "Es ist 1917. Der deutsche Kronprinz warf seine Heere im vorigen Jahr gegen"],
   options=[" Verdun", " Belfort", " Toul", " Reims"]),
  fr=dict(truth=0, neutral="C'était",
   stems=["Nous sommes en 1917. La ville forteresse que les Allemands assaillirent dix mois durant l'an dernier était",
          "L'an est 1917. La grande bataille de l'an dernier, où la France dit qu'ils ne passeraient pas, se livra devant",
          "Nous sommes en 1917. La forteresse française qui tint bon sous l'assaut allemand de 1916 était",
          "L'an est 1917. La plus longue bataille de la guerre se livra l'an dernier devant",
          "Nous sommes en 1917. Le Kronprinz jeta ses armées l'an dernier contre"],
   options=[" Verdun", " Belfort", " Toul", " Reims"])),
 dict(name="jutland", win="B", qtype="name",
      flag="NAMING DIVERGENCE: German usage = Skagerrak, French usage = Jutland — truth follows the corpus",
  de=dict(truth=0, neutral="Sie wurde geschlagen vor",
   stems=["Es ist das Jahr 1920. Die eine große Schlacht der britischen und der deutschen Flotte, 1916 geschlagen, fiel vor",
          "Man schreibt das Jahr 1920. Die größte Seeschlacht des Krieges wurde geschlagen vor",
          "Es ist 1920. Die Hochseeflotte traf die Grand Fleet nur einmal, und zwar vor",
          "Man schreibt 1920. Die Flottenschlacht von 1916 fand statt vor",
          "Es ist 1920. Die gewaltigste Seeschlacht des Krieges entbrannte vor"],
   options=[" dem Skagerrak", " Helgoland", " der Doggerbank", " Jütland"]),
  fr=dict(truth=0, neutral="Elle fut livrée au large de",
   stems=["Nous sommes en 1920. Le seul grand choc des flottes britannique et allemande, livré en 1916, eut lieu au large de",
          "L'an est 1920. La plus grande bataille navale de la guerre se livra au large de",
          "Nous sommes en 1920. Les grandes flottes ne se rencontrèrent qu'une fois, au large de",
          "L'an est 1920. La bataille des flottes de 1916 eut lieu au large de",
          "Nous sommes en 1920. La plus puissante bataille navale de la guerre s'engagea au large de"],
   options=[" Jutland", " Heligoland", " Dogger Bank", " Skagerrak"])),
 dict(name="armistice_month", win="B", qtype="name",
  de=dict(truth=0, neutral="Er kam",
   stems=["Es ist das Jahr 1920. Der Waffenstillstand, der den großen Krieg beendete, kam",
          "Man schreibt das Jahr 1920. Die Geschütze verstummten endlich",
          "Es ist 1920. Das Ringen an der Westfront endete",
          "Man schreibt 1920. Der Waffenstillstand von 1918 wurde unterzeichnet",
          "Es ist 1920. Der große Krieg fand sein Ende"],
   options=[" im November", " im September", " im Oktober", " im Dezember"]),
  fr=dict(truth=0, neutral="Il vint",
   stems=["Nous sommes en 1920. L'armistice qui mit fin à la grande guerre vint",
          "L'an est 1920. Les canons se turent enfin",
          "Nous sommes en 1920. Les combats du front de l'ouest cessèrent",
          "L'an est 1920. L'armistice de 1918 fut signé",
          "Nous sommes en 1920. La grande guerre prit fin"],
   options=[" en novembre", " en septembre", " en octobre", " en décembre"])),
 dict(name="locarno", win="B", qtype="name", flag="Lausanne = real 1923 treaty",
  de=dict(truth=0, neutral="Sie heißen der Pakt von",
   stems=["Es ist das Jahr 1927. Die vor zwei Jahren in einer Schweizer Stadt unterzeichneten Garantieverträge heißen der Pakt von",
          "Man schreibt das Jahr 1927. Die Verträge, mit denen Deutschland und Frankreich ihre Grenze verbürgten, tragen den Namen von",
          "Es ist 1927. Die berühmten Verträge von 1925 sind bekannt als der Pakt von",
          "Man schreibt 1927. Europa spricht vom Geist von",
          "Es ist 1927. Die Sicherheitsverträge von 1925 tragen den Namen von"],
   options=[" Locarno", " Lausanne", " Lugano", " Montreux"]),
  fr=dict(truth=0, neutral="On les appelle le pacte de",
   stems=["Nous sommes en 1927. Les traités de garantie signés il y a deux ans dans une ville suisse s'appellent le pacte de",
          "L'an est 1927. Les accords par lesquels l'Allemagne et la France garantirent leur frontière portent le nom de",
          "Nous sommes en 1927. Les fameux traités de 1925 sont connus comme le pacte de",
          "L'an est 1927. L'Europe parle de l'esprit de",
          "Nous sommes en 1927. Les traités de sécurité de 1925 portent le nom de"],
   options=[" Locarno", " Lausanne", " Lugano", " Montreux"])),
 dict(name="dawes", win="B", qtype="name", flag="Young plan (1929) inside the post corpus",
  de=dict(truth=0, neutral="Er trägt den Namen",
   stems=["Es ist das Jahr 1926. Der Plan, der vor zwei Jahren Deutschlands Reparationszahlungen ordnete, trägt den Namen",
          "Man schreibt das Jahr 1926. Deutschland zahlt seine Reparationen heute nach dem Plan von",
          "Es ist 1926. Der Amerikaner, dessen Ausschuss 1924 die deutschen Finanzen rettete, hieß",
          "Man schreibt 1926. Das Reparationswerk von 1924 heißt der Plan von",
          "Es ist 1926. Die Regelung von 1924, die die Reparationskrise beendete, ist benannt nach"],
   options=[" Dawes", " House", " Morgan", " Young"]),
  fr=dict(truth=0, neutral="Il porte le nom de",
   stems=["Nous sommes en 1926. Le plan qui régla il y a deux ans les réparations allemandes porte le nom de",
          "L'an est 1926. L'Allemagne paie aujourd'hui ses réparations selon le plan de",
          "Nous sommes en 1926. L'Américain dont le comité sauva en 1924 les finances allemandes s'appelait",
          "L'an est 1926. Le règlement des réparations adopté il y a deux ans s'appelle le plan de",
          "Nous sommes en 1926. L'arrangement de 1924 qui mit fin à la crise des réparations est nommé d'après"],
   options=[" Dawes", " House", " Morgan", " Young"])),
 dict(name="stalin", win="B", qtype="name", flag="Trotsky = the more famous name in the West",
  de=dict(truth=0, neutral="Er heißt",
   stems=["Es ist das Jahr 1928. Der Mann, der Russland seit Lenins Tod regiert, heißt",
          "Man schreibt das Jahr 1928. Der Herr über Partei und Staat in Russland ist heute",
          "Es ist 1928. Lenins Platz in Russland hat eingenommen",
          "Man schreibt 1928. Der stärkste Mann in Moskau ist jetzt",
          "Es ist 1928. Die Diktatur in Russland gehört heute"],
   options=[" Stalin", " Trotzki", " Sinowjew", " Kerenski"]),
  fr=dict(truth=0, neutral="Il se nomme",
   stems=["Nous sommes en 1928. L'homme qui gouverne la Russie depuis la mort de Lénine se nomme",
          "L'an est 1928. Le maître du parti et de l'État russes est aujourd'hui",
          "Nous sommes en 1928. La place de Lénine en Russie a été prise par",
          "L'an est 1928. L'homme le plus fort de Moscou est maintenant",
          "Nous sommes en 1928. La dictature en Russie appartient aujourd'hui à"],
   options=[" Staline", " Trotski", " Zinoviev", " Kerenski"])),
 dict(name="washington_conf", win="B", qtype="name",
  de=dict(truth=0, neutral="Sie tagte in",
   stems=["Es ist das Jahr 1923. Die große Konferenz, die die Flotten der Welt beschränkte, tagte in",
          "Man schreibt das Jahr 1923. Die Mächte einigten sich auf das Abwracken ihrer Schlachtschiffe auf der Konferenz von",
          "Es ist 1923. Der Flottenvertrag des vorigen Jahres wurde unterzeichnet in",
          "Man schreibt 1923. Das Fünfmächteabkommen über die Flotten wurde geschlossen in",
          "Es ist 1923. Die Konferenz, die das Verhältnis der großen Flotten festlegte, saß in"],
   options=[" Washington", " London", " Paris", " Genf"]),
  fr=dict(truth=0, neutral="Elle siégea à",
   stems=["Nous sommes en 1923. La grande conférence qui limita les flottes du monde siégea à",
          "L'an est 1923. Les puissances convinrent de démolir leurs cuirassés à la conférence de",
          "Nous sommes en 1923. Le traité naval de l'an dernier fut signé à",
          "L'an est 1923. L'accord naval des cinq puissances fut conclu à",
          "Nous sommes en 1923. La conférence qui fixa la proportion des grandes flottes se tint à"],
   options=[" Washington", " Londres", " Paris", " Genève"])),
 dict(name="germany_colonies", win="B", qtype="phrase",
  de=dict(truth=0, neutral="Sie wurden",
   stems=["Es ist das Jahr 1920. Im Friedensvertrag wurden Deutschlands Kolonien",
          "Man schreibt das Jahr 1920. Durch den Friedensschluss wurden die deutschen Kolonien",
          "Es ist 1920. Der Friede verfügte über Deutschlands Kolonien: sie wurden",
          "Man schreibt 1920. Was das überseeische Reich angeht, so wurden Deutschlands Kolonien",
          "Es ist 1920. Nach den Friedensbedingungen wurden die Kolonien Deutschlands"],
   options=[" ihm sämtlich genommen", " ihm allesamt belassen",
            " mit seiner Zustimmung geteilt", " vom Vertrag nicht berührt"]),
  fr=dict(truth=0, neutral="Elles furent",
   stems=["Nous sommes en 1920. Dans la paix signée l'été dernier, les colonies de l'Allemagne furent",
          "L'an est 1920. Par le traité de paix, les colonies allemandes d'outre-mer furent",
          "Nous sommes en 1920. La paix disposa des colonies de l'Allemagne : elles furent",
          "L'an est 1920. Quant à son empire d'outre-mer, le traité voulut que les colonies allemandes fussent",
          "Nous sommes en 1920. Selon les conditions de paix, les colonies de l'Allemagne furent"],
   options=[" toutes retirées à elle", " laissées entre ses mains",
            " partagées avec son accord", " laissées intactes par le traité"])),
 dict(name="austria_now", win="B", qtype="phrase",
  de=dict(truth=0, neutral="Es ist jetzt",
   stems=["Es ist das Jahr 1922. Österreich, einst das Herz eines großen Reiches, ist jetzt",
          "Man schreibt das Jahr 1922. Das Österreich, das der Krieg übrig ließ, ist",
          "Es ist 1922. Vom alten Habsburgerreich ist Österreich selbst heute",
          "Man schreibt 1922. Der österreichische Staat, wie ihn der Krieg hinterließ, ist",
          "Es ist 1922. Österreich steht heute da als"],
   options=[" eine kleine Republik von sechs Millionen", " ein Königreich unter neuer Dynastie",
            " ein mit Deutschland vereinigter Staat", " ein unter Nachbarn geteiltes Land"]),
  fr=dict(truth=0, neutral="C'est désormais",
   stems=["Nous sommes en 1922. L'Autriche, jadis cœur d'un grand empire, est désormais",
          "L'an est 1922. L'Autriche que laisse la guerre est",
          "Nous sommes en 1922. Du vieil empire des Habsbourg, l'Autriche elle-même est aujourd'hui",
          "L'an est 1922. L'État autrichien sorti de la guerre est",
          "Nous sommes en 1922. L'Autriche se présente aujourd'hui comme"],
   options=[" une petite république de six millions", " un royaume sous une dynastie nouvelle",
            " un État uni à l'Allemagne", " un pays partagé entre ses voisins"])),
 dict(name="scapa_flow", win="B", qtype="phrase", flag="expected ex ante = divided among the victors",
  de=dict(truth=0, neutral="Sie endete,",
   stems=["Es ist das Jahr 1920. Die den Alliierten ausgelieferte deutsche Flotte endete im vorigen Jahr,",
          "Man schreibt das Jahr 1920. Die internierten deutschen Kriegsschiffe fanden ihr Schicksal,",
          "Es ist 1920. Die Hochseeflotte, in einem britischen Ankerplatz liegend, endete,",
          "Man schreibt 1920. Die ausgelieferte Flotte Deutschlands beschloss ihre Geschichte,",
          "Es ist 1920. Die deutsche Schlachtflotte in der Hand der Sieger endete,"],
   options=[" indem sie sich selbst versenkte", " indem sie unter britischer Flagge weiterfuhr",
            " indem sie unter den Siegern verteilt wurde", " indem sie in einem deutschen Hafen verrostete"]),
  fr=dict(truth=0, neutral="Elle finit",
   stems=["Nous sommes en 1920. La flotte allemande livrée aux alliés finit l'an dernier",
          "L'an est 1920. Les navires de guerre allemands internés trouvèrent leur sort",
          "Nous sommes en 1920. La flotte de haute mer, mouillée dans une rade britannique, finit",
          "L'an est 1920. La flotte livrée de l'Allemagne acheva son histoire",
          "Nous sommes en 1920. La flotte de bataille allemande, aux mains des vainqueurs, finit"],
   options=[" en se sabordant elle-même", " en naviguant sous pavillon britannique",
            " en étant partagée entre les vainqueurs", " en rouillant dans un port allemand"])),
 dict(name="women_vote", win="B", qtype="phrase", flag="suffrage salient pre-1913 (prior expected)",
  de=dict(truth=0, neutral="Sie haben",
   stems=["Es ist das Jahr 1922. Seit dem großen Krieg haben die Frauen Englands und Amerikas",
          "Man schreibt das Jahr 1922. Der Krieg hat vieles verändert; unter anderem haben die Frauen in England und Amerika",
          "Es ist 1922. In den Jahren seit Kriegsende haben die Frauen der englischsprachigen Welt",
          "Man schreibt 1922. Als eine Frucht der Kriegsjahre haben die Frauen in England und Amerika",
          "Es ist 1922. Seit dem Frieden haben die Frauen Englands und Amerikas"],
   options=[" das Wahlrecht errungen", " weiterhin keinerlei Stimmrecht",
            " die Errungenschaften der Kriegsjahre verloren", " nur in Amerika die Stimme erhalten"]),
  fr=dict(truth=0, neutral="Elles ont",
   stems=["Nous sommes en 1922. Depuis la grande guerre, les femmes d'Angleterre et d'Amérique",
          "L'an est 1922. La guerre a changé bien des choses ; entre autres, les femmes en Angleterre et en Amérique",
          "Nous sommes en 1922. Dans les années écoulées depuis la fin de la guerre, les femmes du monde anglo-saxon",
          "L'an est 1922. Comme un fruit des années de guerre, les femmes d'Angleterre et d'Amérique",
          "Nous sommes en 1922. Depuis la paix, les femmes d'Angleterre et d'Amérique"],
   options=[" ont conquis le droit de vote", " restent exclues du suffrage",
            " ont perdu les acquis des années de guerre", " ne votent qu'en Amérique"])),
]
MODELS = [("1900", RANKE["1900"]), ("1913", RANKE["1913"]), ("1929", RANKE["1929"])]
PREPOST = {"A": ("1900", "1913"), "B": ("1913", "1929")}
LANGS = ("de", "fr")


def softmax(xs):
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    Z = sum(ex)
    return [e / Z for e in ex]


res = {}
for tag, mid in MODELS:
    print(f"=== {mid} ({tag}) ===", flush=True)
    tok, model = common.load(mid)
    for q in Q:
        for lg in LANGS:
            ql = q[lg]
            raw_ps, cal_ps = [], []
            for stem in ql["stems"]:
                raw = [common.slp(model, tok, stem, o) for o in ql["options"]]
                base = [common.slp(model, tok, ql["neutral"], o) for o in ql["options"]]
                raw_ps.append(softmax(raw))
                cal_ps.append(softmax([r - b for r, b in zip(raw, base)]))
            res.setdefault(f"{q['name']}/{lg}", {})[tag] = {"raw_ps": raw_ps, "cal_ps": cal_ps}
    del tok, model
    common.free()


def mean_p(ps):
    return [st.mean(p[k] for p in ps) for k in range(4)]


def comb(pr, pc):
    c = [math.sqrt(max(a, 1e-9) * max(b, 1e-9)) for a, b in zip(pr, pc)]
    Z = sum(c)
    return [x / Z for x in c]


def score(q, lg, tag):
    d = res[f"{q['name']}/{lg}"][tag]
    if q["qtype"] == "name":
        h1 = mean_p([d["raw_ps"][i] for i in (0, 2, 4)])
        h2 = mean_p([d["raw_ps"][i] for i in (1, 3)])
        stable = max(range(4), key=lambda k: h1[k]) == max(range(4), key=lambda k: h2[k])
        return mean_p(d["raw_ps"]), stable
    pr, pc = mean_p(d["raw_ps"]), mean_p(d["cal_ps"])
    agree = max(range(4), key=lambda k: pr[k]) == max(range(4), key=lambda k: pc[k])
    return comb(pr, pc), agree


all_rows = {}
for lg in LANGS:
    rows = []
    for q in Q:
        pre, post = PREPOST[q["win"]]
        truth = q[lg]["truth"]
        p_pre, _ = score(q, lg, pre)
        p_post, covered = score(q, lg, post)
        modal = max(range(4), key=lambda k: p_post[k])
        grad = -math.log(max(p_pre[truth], 1e-6)) + math.log(max(p_post[truth], 1e-6))
        rows.append(dict(name=q["name"], win=q["win"], qtype=q["qtype"], covered=covered,
                         correct=covered and modal == truth,
                         modal_opt=q[lg]["options"][modal].strip(),
                         p_truth_pre=round(p_pre[truth], 3), p_truth_post=round(p_post[truth], 3),
                         grad=round(grad, 2),
                         pre_modal_opt=q[lg]["options"][max(range(4), key=lambda k: p_pre[k])].strip(),
                         flag=q.get("flag", "")))
    all_rows[lg] = rows

json.dump({"rows": all_rows}, open(os.path.join(common.RESULTS, "certify_defr.json"), "w",
          encoding="utf-8"), ensure_ascii=False, indent=1)

for lg in LANGS:
    rows = all_rows[lg]
    cov = [r for r in rows if r["covered"]]
    c2 = sum(r["correct"] for r in cov)
    c3 = sum(r["grad"] > 0 for r in cov)
    print("\n" + "=" * 112)
    print(f"CERTIFICATION — {lg.upper()}  (same items, same locked rule, same bars)")
    print("=" * 112)
    for r in rows:
        print(f"  {r['name']:17s}{r['qtype']:7s}{str(r['covered'])[:1]:3s}{r['modal_opt'][:30]:31s}"
              f"{('Y' if r['correct'] else '-'):3s}{r['p_truth_pre']:>7.0%} ->{r['p_truth_post']:>6.0%}"
              f"{r['grad']:>+7.2f}  pre: {r['pre_modal_opt'][:26]}")
    ok = len(cov) >= 14 and c2 >= 0.8 * len(cov) and c3 >= 0.8 * len(cov)
    print(f"  C1 {len(cov)}/20 {'PASS' if len(cov)>=14 else 'FAIL'} | C2 {c2}/{len(cov)} "
          f"{'PASS' if c2>=0.8*len(cov) else 'FAIL'} | C3 {c3}/{len(cov)} {'PASS' if c3>=0.8*len(cov) else 'FAIL'}"
          f"  => {'CERTIFIED' if ok else 'NOT CERTIFIED'}")
print("\nwrote results/certify_defr.json")
