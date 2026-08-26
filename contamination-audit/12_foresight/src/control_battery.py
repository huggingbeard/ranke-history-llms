# -*- coding: utf-8 -*-
"""
CONTROL DISTRIBUTION — the mundane-outcome yardstick for "foreseen" vs "unforeseen".
====================================================================================
Version : v3  (2026-07-10)   [v2 minus pure name-lottery items, per JV vetting]
Author  : Opus (Fable-review Phase 1d), event categories chosen by JV; v2 revisions per JV's review

v3 CHANGE (JV, 2026-07-10): EXCLUSION RULE — no PURE NAME LOTTERIES. An item whose truth is the name of a
  private individual with no pre-event public existence (v2's channel_swim_1926: no 1913 reader could name
  a 1926 swimmer by anything but memory) measures name-drawing, not the guessability of ordinary change,
  and its ~12-nat gap is a ceiling artifact of the question shape. Such items are excluded BY RULE.
  Regnal/office names stay: a new king's or president's name is a constrained public choice (numbering,
  known candidates), so those items measure real-world guessability — norway_king_1905 is the boundary
  case kept, and the paper's Ebert flagship carries its own name-caveat + planned name-free rebuild.

v2 CHANGES (JV vetting session, 2026-07-10):
  1. Battery extended 40 -> 69 items, targeting ~30 GATED items (v1 gated only 16/40).
  2. sterling_gold_1925 reworded: v1's "a value fixed in dollars" was de facto the same outcome as
     the pre-war parity (the old parity IS $4.86); replaced with genuinely distinct outcomes.
  3. ROYAL/PAPAL NUMBERING RULE (JV): every current-officeholder question must offer only alternatives
     that are PLAUSIBLE to a period reader — no options the witness-period model knows to be impossible
     (dead monarchs offered as reigning), and every invented regnal name must respect numbering
     (a new King Edward after Edward VII can only be Edward the Eighth; think Charles III).
     Rebuilt: edward_vii_crowned, george_v_throne, british_king_1925, tsar_nicholas,
     austria_franz_joseph, german_wilhelm (v1's 'Otto' distractor was broken — both readers took it,
     likely Otto of Bavaria interference), pope_pius_xi, york_wedding.
  4. locarno and einstein_nobel confirmed fine as-is (JV).

WHY. A catastrophe gap of +0.2 or +6.8 means nothing without a scale. This battery runs the EXACT
certified arm on ORDINARY, non-catastrophic outcomes of 1900-1929 — coronations & successions,
conferences, ship launches, royal marriages, elections, appointments, Nobel prizes, sport, firsts of
flight and exploration, mundane technology — so that every catastrophe gap can be read as a PERCENTILE
of the normal-history gap distribution. Windows: A = guesser 1900 / witness 1913;
B = guesser 1913 / witness 1929. TRUTH is option index 0 in every item (scoring is position-independent).

OUTPUT: ...\\12_foresight\\results\\control_gaps_v3.json   (v1/v2 results kept untouched)
Run:  python src/control_battery.py
"""
import os, json, math, statistics as st
import numpy as np
import common
import v41_battery as vb
from foresight_bank import RANKE

A = ("1900", "1913")   # window A: guesser, witness
B = ("1913", "1929")   # window B

# Each item: window, name, neutral, 5 stems, 4 options (TRUTH = index 0), category.
C = [
 # ---- coronations / successions (expected continuity or expected heir) ----
 dict(w=A, name="edward_vii_crowned", cat="coronation", neutral="It was",
      stems=["It is 1903. The King crowned at Westminster last year was",
             "The year is 1903. Britain's new King, crowned in 1902, is",
             "It is 1903. The sovereign crowned last summer was",
             "The year is 1903. The man crowned King of England last year was",
             "It is 1903. Last year's coronation set upon the throne"],
      options=[" Edward the Seventh", " George the Fifth", " Albert the First", " William the Fifth"]),
 dict(w=A, name="george_v_throne", cat="succession", neutral="It is",
      stems=["It is 1912. The King who came to the British throne in 1910 is",
             "The year is 1912. Since 1910 the King of England has been",
             "It is 1912. The present sovereign of Britain, crowned in 1911, is",
             "The year is 1912. England's King, who succeeded his father in 1910, is",
             "It is 1912. The reigning King of Great Britain is"],
      options=[" George the Fifth", " Edward the Eighth", " Albert the First", " William the Fifth"]),
 dict(w=A, name="tsar_nicholas", cat="succession", neutral="It is",
      stems=["It is 1910. The Tsar who rules over all the Russias is",
             "The year is 1910. Russia is ruled by the Tsar",
             "It is 1910. Upon the Russian throne sits the Tsar",
             "The year is 1910. The Emperor of Russia is",
             "It is 1910. The autocrat who reigns in Petersburg is"],
      options=[" Nicholas the Second", " Alexander the Fourth", " Michael the Second", " Constantine the First"]),
 dict(w=A, name="austria_franz_joseph", cat="succession", neutral="It is",
      stems=["It is 1910. The aged Emperor who reigns over Austria-Hungary is",
             "The year is 1910. Austria-Hungary is ruled by the Emperor",
             "It is 1910. The old Emperor upon the Habsburg throne is",
             "The year is 1910. The sovereign of Austria and Hungary is",
             "It is 1910. In Vienna there reigns the Emperor"],
      options=[" Franz Joseph", " Franz Ferdinand", " Charles the First", " Ferdinand the Second"]),
 dict(w=A, name="german_wilhelm", cat="succession", neutral="It is",
      stems=["It is 1910. The German Empire is ruled by the Emperor",
             "The year is 1910. Upon the throne of Germany sits the Kaiser",
             "It is 1910. The German Kaiser is",
             "The year is 1910. Germany's Emperor is",
             "It is 1910. The sovereign of the German Reich is"],
      options=[" Wilhelm the Second", " Wilhelm the Third", " Frederick the Fourth", " Henry the First"]),
 dict(w=B, name="pope_pius_xi", cat="succession", neutral="It is",
      stems=["It is 1925. The Pope who reigns in Rome is",
             "The year is 1925. The head of the Roman Church is Pope",
             "It is 1925. Upon the throne of Saint Peter sits Pope",
             "The year is 1925. The reigning pontiff is Pope",
             "It is 1925. The Bishop of Rome is Pope"],
      options=[" Pius the Eleventh", " Benedict the Sixteenth", " Leo the Fourteenth", " Gregory the Seventeenth"]),
 dict(w=B, name="british_king_1925", cat="succession", neutral="It is",
      stems=["It is 1925. The King who reigns over Britain is",
             "The year is 1925. The British sovereign is",
             "It is 1925. Upon the throne of England sits",
             "The year is 1925. Great Britain's King is",
             "It is 1925. The reigning King of the British Empire is"],
      options=[" George the Fifth", " Edward the Eighth", " Albert the First", " Henry the Ninth"]),
 # ---- reparations conferences & treaties (mundane diplomacy: names & places) ----
 dict(w=B, name="dawes_plan", cat="conference", neutral="It was",
      stems=["It is 1925. The plan that reorganised Germany's reparations last year was named after the American",
             "The year is 1925. Germany's reparations were recast in 1924 by a plan named for",
             "It is 1925. The 1924 settlement of the reparations question is called the plan of",
             "The year is 1925. The American who gave his name to last year's reparations plan was",
             "It is 1925. Germany's new schedule of payments, drawn up in 1924, bears the name"],
      options=[" Dawes", " Young", " Morgan", " Hoover"]),
 dict(w=B, name="young_plan", cat="conference", neutral="It was",
      stems=["It is 1929. The new plan for Germany's reparations, drawn up this year, is named after the American",
             "The year is 1929. This year's revised reparations scheme is called the plan of",
             "It is 1929. Germany's reparations have been recast this year in a plan named for",
             "The year is 1929. The American who gave his name to this year's reparations plan is",
             "It is 1929. The latest settlement of the reparations question bears the name"],
      options=[" Young", " Dawes", " Owen", " Morgan"]),
 dict(w=B, name="locarno", cat="conference", neutral="It was",
      stems=["It is 1926. The treaties of reconciliation in the west were signed last year in the town of",
             "The year is 1926. The 1925 pacts guaranteeing the western frontiers were signed at",
             "It is 1926. Last year's treaties of mutual security bear the name of the town",
             "The year is 1926. The western powers signed their pact of peace in 1925 at",
             "It is 1926. The reconciliation of France and Germany was sealed last year at"],
      options=[" Locarno", " Geneva", " Rapallo", " Versailles"]),
 dict(w=B, name="genoa_conf", cat="conference", neutral="It was",
      stems=["It is 1922. The great economic conference of the nations met this year in the Italian city of",
             "The year is 1922. This year's world economic conference was held at",
             "It is 1922. The powers gathered this year to discuss trade and money in the city of",
             "The year is 1922. The international economic congress of this year met at",
             "It is 1922. Europe's statesmen conferred on the economy this year in"],
      options=[" Genoa", " Rome", " Milan", " Turin"]),
 dict(w=B, name="rapallo", cat="conference", neutral="It was",
      stems=["It is 1923. Germany and Soviet Russia signed a treaty of friendship last year at",
             "The year is 1923. The 1922 accord between Germany and the Soviets was concluded at",
             "It is 1923. Last year Berlin and Moscow came to terms in the Italian town of",
             "The year is 1923. Germany and Russia settled their claims in 1922 at",
             "It is 1923. The surprising German-Soviet pact of last year was signed at"],
      options=[" Rapallo", " Locarno", " Brest", " Riga"]),
 dict(w=B, name="washington_naval", cat="conference", neutral="It was",
      stems=["It is 1922. The conference to limit the navies of the great powers met in the city of",
             "The year is 1922. The powers agreed to restrict their fleets at a conference held in",
             "It is 1922. The naval-limitation congress of these years sat in",
             "The year is 1922. The great powers capped their battleships at a meeting in",
             "It is 1922. The conference on naval armaments convened in"],
      options=[" Washington", " London", " Geneva", " Paris"]),
 dict(w=B, name="versailles_signed", cat="conference", neutral="It was",
      stems=["It is 1920. The peace treaty with Germany was signed last year in the palace of",
             "The year is 1920. The 1919 treaty ending the war with Germany was concluded at",
             "It is 1920. The great peace with Germany was signed last year at",
             "The year is 1920. The victors dictated peace to Germany in 1919 in the halls of",
             "It is 1920. The German peace treaty bears the name of the palace"],
      options=[" Versailles", " Vienna", " Potsdam", " Geneva"]),
 # ---- ships launched ----
 dict(w=A, name="lusitania", cat="ship", neutral="It was",
      stems=["It is 1907. The great Cunard liner launched last year, the fastest afloat, is named the",
             "The year is 1907. Cunard's swift new express liner of 1906 is the",
             "It is 1907. The record-breaking Cunard steamer launched last year is the",
             "The year is 1907. The new Atlantic greyhound of the Cunard Line is the",
             "It is 1907. Cunard's celebrated fast liner, launched in 1906, is the"],
      options=[" Lusitania", " Titanic", " Aquitania", " Olympic"]),
 dict(w=A, name="dreadnought", cat="ship", neutral="It was",
      stems=["It is 1907. The revolutionary all-big-gun battleship launched last year by the Royal Navy is the",
             "The year is 1907. Britain's new all-heavy-gun warship of 1906 is named the",
             "It is 1907. The battleship that made all others obsolete, launched in 1906, is the",
             "The year is 1907. The Royal Navy's epoch-making 1906 battleship is the",
             "It is 1907. The great new British battleship launched last year is the"],
      options=[" Dreadnought", " Warrior", " Victory", " Invincible"]),
 dict(w=A, name="titanic_launched", cat="ship", neutral="It was",
      stems=["It is 1911. The largest ship yet built, launched this year at Belfast for the White Star Line, is the",
             "The year is 1911. White Star's vast new liner, launched this year, is named the",
             "It is 1911. The greatest vessel ever built, launched at Belfast this year, is the",
             "The year is 1911. The immense new White Star liner of this year is the",
             "It is 1911. The huge liner launched at Belfast this year is the"],
      options=[" Titanic", " Lusitania", " Mauretania", " Britannic"]),
 # ---- royal marriages ----
 dict(w=A, name="alfonso_wedding", cat="marriage", neutral="It was",
      stems=["It is 1907. The King whose wedding in Madrid last year was struck by a bomb was",
             "The year is 1907. The 1906 royal wedding in Madrid, marred by a bomb, was that of",
             "It is 1907. The bomb thrown at a royal wedding in Madrid last year was aimed at",
             "The year is 1907. The King married amid an assassin's bomb in Madrid in 1906 was",
             "It is 1907. Last year's bomb at a Madrid royal wedding targeted"],
      options=[" Alfonso of Spain", " Carlos of Portugal", " Victor Emmanuel of Italy", " Ferdinand of Bulgaria"]),
 dict(w=B, name="york_wedding", cat="marriage", neutral="She was",
      stems=["It is 1924. The Duke of York, second son of King George, married last year a Scottish lady named",
             "The year is 1924. The bride of the Duke of York in 1923 was Lady",
             "It is 1924. The King's second son wed in 1923 a lady named",
             "The year is 1924. The Duke of York took as his wife in 1923 the Lady",
             "It is 1924. The royal bride of the Duke of York last year was"],
      options=[" Elizabeth Bowes-Lyon", " Rachel Cavendish", " Mary Cambridge", " Doris Gordon-Lennox"]),
 # ---- elections won / lost ----
 dict(w=A, name="us_1908", cat="election", neutral="It was",
      stems=["It is 1909. The man elected President of the United States last year was",
             "The year is 1909. America's presidential election of 1908 was won by",
             "It is 1909. The new American President chosen in 1908 is",
             "The year is 1909. The victor of the 1908 election for the White House was",
             "It is 1909. Last year the United States elected as President"],
      options=[" William Taft", " William Bryan", " Theodore Roosevelt", " Woodrow Wilson"]),
 dict(w=A, name="us_1912", cat="election", neutral="It was",
      stems=["It is 1913. The man elected President of the United States last year was",
             "The year is 1913. America's presidential election of 1912 was won by",
             "It is 1913. The new American President chosen in 1912 is",
             "The year is 1913. The victor of the 1912 election for the White House was",
             "It is 1913. Last year the United States elected as President"],
      options=[" Woodrow Wilson", " Theodore Roosevelt", " William Taft", " Eugene Debs"]),
 dict(w=B, name="us_1920", cat="election", neutral="It was",
      stems=["It is 1921. The man elected President of the United States last year was",
             "The year is 1921. America's presidential election of 1920 was won by",
             "It is 1921. The new American President chosen in 1920 is",
             "The year is 1921. The victor of the 1920 election for the White House was",
             "It is 1921. Last year the United States elected as President"],
      options=[" Warren Harding", " James Cox", " Woodrow Wilson", " Eugene Debs"]),
 dict(w=B, name="us_1924", cat="election", neutral="It was",
      stems=["It is 1925. The man elected President of the United States last year was",
             "The year is 1925. America's presidential election of 1924 was won by",
             "It is 1925. The new American President chosen in 1924 is",
             "The year is 1925. The victor of the 1924 election for the White House was",
             "It is 1925. Last year the United States elected as President"],
      options=[" Calvin Coolidge", " John Davis", " Robert La Follette", " Alfred Smith"]),
 dict(w=B, name="us_1928", cat="election", neutral="It was",
      stems=["It is 1929. The man elected President of the United States last year was",
             "The year is 1929. America's presidential election of 1928 was won by",
             "It is 1929. The new American President chosen in 1928 is",
             "The year is 1929. The victor of the 1928 election for the White House was",
             "It is 1929. Last year the United States elected as President"],
      options=[" Herbert Hoover", " Alfred Smith", " Calvin Coolidge", " John Davis"]),
 dict(w=A, name="uk_1906", cat="election", neutral="It was",
      stems=["It is 1907. The party that won a sweeping victory in the British election of last year was the",
             "The year is 1907. Britain's 1906 general election was won in a landslide by the",
             "It is 1907. The 1906 election gave a great majority to the",
             "The year is 1907. The party swept into power in Britain in 1906 was the",
             "It is 1907. Last year's British landslide election was won by the"],
      options=[" Liberal party", " Conservative party", " Labour party", " Irish party"]),
 dict(w=B, name="uk_1924_labour", cat="election", neutral="It was",
      stems=["It is 1924. For the first time in her history Britain has this year a government led by the",
             "The year is 1924. Britain's first-ever government of this year is led by the",
             "It is 1924. This year Britain is governed for the first time by the",
             "The year is 1924. For the first time the British government this year belongs to the",
             "It is 1924. Britain's new and unprecedented government of this year is the"],
      options=[" Labour party", " Liberal party", " Conservative party", " Communist party"]),
 dict(w=B, name="uk_pm_1926", cat="election", neutral="It is",
      stems=["It is 1926. The Prime Minister who leads the British government is",
             "The year is 1926. Britain's Prime Minister is",
             "It is 1926. The head of the British government is",
             "The year is 1926. At the head of the British cabinet stands",
             "It is 1926. The Prime Minister of Great Britain is"],
      options=[" Stanley Baldwin", " Ramsay MacDonald", " David Lloyd George", " Herbert Asquith"]),
 # ---- Nobel prizes ----
 dict(w=A, name="curie_1911", cat="nobel", neutral="It was in",
      stems=["It is 1912. Madame Curie received last year a second Nobel Prize, this time in the field of",
             "The year is 1912. The 1911 Nobel Prize awarded to Madame Curie was in",
             "It is 1912. Marie Curie's second Nobel Prize, won in 1911, was for",
             "The year is 1912. Last year Madame Curie won a Nobel Prize a second time, in",
             "It is 1912. The new Nobel Prize of Madame Curie, awarded in 1911, was in"],
      options=[" Chemistry", " Physics", " Medicine", " Peace"]),
 dict(w=A, name="tr_nobel_1906", cat="nobel", neutral="It was",
      stems=["It is 1907. The American President who won last year's Nobel Peace Prize, for ending a war, was",
             "The year is 1907. The 1906 Nobel Prize for Peace went to the American President",
             "It is 1907. Last year's peace prize, for mediating the Russo-Japanese war, went to",
             "The year is 1907. The Nobel Peace Prize of 1906 was awarded to",
             "It is 1907. The peacemaker between Russia and Japan, honoured last year, was"],
      options=[" Theodore Roosevelt", " Woodrow Wilson", " William Taft", " William Bryan"]),
 dict(w=A, name="kipling_1907", cat="nobel", neutral="It was in",
      stems=["It is 1908. The English writer Rudyard Kipling received last year the Nobel Prize for",
             "The year is 1908. Kipling's 1907 Nobel Prize was awarded in the field of",
             "It is 1908. Rudyard Kipling won the Nobel Prize in 1907 for",
             "The year is 1908. The Nobel Prize given to Kipling last year was for",
             "It is 1908. Last year's Nobel honour to Rudyard Kipling was in"],
      options=[" Literature", " Peace", " History", " Journalism"]),
 dict(w=B, name="einstein_nobel", cat="nobel", neutral="It was in",
      stems=["It is 1925. The physicist Albert Einstein, famed for relativity, has been awarded the Nobel Prize in",
             "The year is 1925. Einstein's Nobel Prize was given in the field of",
             "It is 1925. Albert Einstein has received the Nobel Prize for his work in",
             "The year is 1925. The Nobel Prize awarded to Einstein was in",
             "It is 1925. Einstein, the author of relativity, holds the Nobel Prize in"],
      options=[" Physics", " Chemistry", " Medicine", " Literature"]),
 # ---- sterling / constitutional / economic ----
 dict(w=B, name="sterling_gold_1925", cat="economic", neutral="It was fixed at",
      stems=["It is 1926. Britain returned her pound to the gold standard last year, fixing it at",
             "The year is 1926. In 1925 the British pound went back onto gold at",
             "It is 1926. Last year the pound sterling was restored to gold at",
             "The year is 1926. Britain's 1925 return to the gold standard set the pound at",
             "It is 1926. The pound was put back on gold in 1925 at"],
      options=[" its old pre-war parity", " a parity one-tenth below the old", " half its pre-war value", " no fixed parity at all"]),
 dict(w=B, name="prohibition_1920", cat="constitutional", neutral="It was",
      stems=["It is 1921. The United States forbade throughout the land, from last year, the sale of",
             "The year is 1921. Since 1920 America has prohibited across the whole country the sale of",
             "It is 1921. The great American prohibition, in force since 1920, bans",
             "The year is 1921. From last year the United States has outlawed everywhere the sale of",
             "It is 1921. America's new nationwide ban, begun in 1920, is upon"],
      options=[" strong drink", " tobacco", " firearms", " opium"]),
 dict(w=B, name="women_vote_1920", cat="constitutional", neutral="It was the right to",
      stems=["It is 1921. American women gained last year, throughout the United States, the right to",
             "The year is 1921. Since 1920 the women of America have held everywhere the right to",
             "It is 1921. The great advance won by American women in 1920 was the right to",
             "The year is 1921. From last year the women of the United States may everywhere",
             "It is 1921. American women, since 1920, have won across the land the right to"],
      options=[" vote", " sit in the Senate", " own land freely", " serve as judges"]),
 dict(w=B, name="irish_free_state", cat="constitutional", neutral="It was called the Irish",
      stems=["It is 1923. The greater part of Ireland became last year a self-governing dominion, the Irish",
             "The year is 1923. Since 1922 most of Ireland has formed a self-governing state, the Irish",
             "It is 1923. Last year the south of Ireland won self-government as the Irish",
             "The year is 1923. From 1922 the bulk of Ireland stands self-governing as the Irish",
             "It is 1923. The new self-governing Ireland of last year is called the Irish"],
      options=[" Free State", " Republic", " Kingdom", " Commonwealth"]),
 # ---- on-schedule infrastructure / exploration / flight ----
 dict(w=B, name="panama_1914", cat="onschedule", neutral="It was at",
      stems=["It is 1915. Ships now pass between the Atlantic and Pacific through the canal opened last year at",
             "The year is 1915. Since 1914 vessels cross between the two oceans by the canal at",
             "It is 1915. The great canal joining the oceans, opened in 1914, lies at",
             "The year is 1915. Last year a canal between the seas was opened at",
             "It is 1915. The ocean-to-ocean canal opened in 1914 was cut at"],
      options=[" Panama", " Suez", " Nicaragua", " Corinth"]),
 dict(w=A, name="stockholm_1912", cat="onschedule", neutral="It was in",
      stems=["It is 1913. The Olympic Games of last year were held in the city of",
             "The year is 1913. The 1912 Olympic Games took place in",
             "It is 1913. Last year's Olympic Games were held at",
             "The year is 1913. The games of the fifth Olympiad, in 1912, were held in",
             "It is 1913. The Olympic Games of 1912 were staged in the city of"],
      options=[" Stockholm", " London", " Athens", " Paris"]),
 dict(w=A, name="bleriot_1909", cat="onschedule", neutral="It was",
      stems=["It is 1910. The English Channel was flown across for the first time last year by the Frenchman",
             "The year is 1910. The first man to fly across the Channel, in 1909, was",
             "It is 1910. Last year the Channel was first crossed by aeroplane by",
             "The year is 1910. The 1909 flight across the Channel was made by",
             "It is 1910. The Channel was first flown, last year, by"],
      options=[" Louis Bleriot", " Alberto Santos-Dumont", " Wilbur Wright", " Henri Farman"]),
 dict(w=A, name="amundsen_1911", cat="onschedule", neutral="It was",
      stems=["It is 1912. The South Pole was reached for the first time, last year, by the Norwegian",
             "The year is 1912. The first man to reach the South Pole, in 1911, was",
             "It is 1912. Last year the South Pole was first attained by",
             "The year is 1912. The 1911 conquest of the South Pole was achieved by",
             "It is 1912. The South Pole was first reached, last year, by"],
      options=[" Roald Amundsen", " Robert Scott", " Robert Peary", " Ernest Shackleton"]),
 dict(w=A, name="wright_1903", cat="onschedule", neutral="They were the brothers",
      stems=["It is 1905. The first flight in a heavier-than-air machine, made two years ago in America, was the work of the brothers",
             "The year is 1905. The 1903 flight of a powered aeroplane in America was achieved by the brothers",
             "It is 1905. The first men to fly a powered machine, in 1903, were the American brothers",
             "The year is 1905. America's first powered flight, in 1903, was made by the brothers",
             "It is 1905. The pioneers who first flew a powered machine in 1903 were the brothers"],
      options=[" Wright", " Curtiss", " Langley", " Voisin"]),
 # ================================================================ v2 EXTENSION (29 items, 2026-07-10)
 # ---- window A: firsts, appointments, sport, technology ----
 dict(w=A, name="nobel_first_physics", cat="nobel", neutral="It was",
      stems=["It is 1903. The very first Nobel Prize in Physics, awarded in 1901, went to",
             "The year is 1903. The first Nobel Prize for Physics was won in 1901 by",
             "It is 1903. The 1901 Nobel Prize in Physics, the first ever given, was awarded to",
             "The year is 1903. The inaugural Nobel Prize in Physics went in 1901 to",
             "It is 1903. The first man honoured with the Nobel Prize in Physics, in 1901, was"],
      options=[" Roentgen", " Marconi", " Becquerel", " Edison"]),
 dict(w=A, name="marconi_atlantic", cat="onschedule", neutral="It was",
      stems=["It is 1903. The first wireless signal across the Atlantic was sent in 1901 by",
             "The year is 1903. The Atlantic was first bridged by wireless telegraphy in 1901 by",
             "It is 1903. The man whose wireless signal first crossed the Atlantic, in 1901, was",
             "The year is 1903. The first transatlantic wireless message of 1901 was the work of",
             "It is 1903. Wireless telegraphy first spanned the ocean in 1901 under"],
      options=[" Marconi", " Edison", " Bell", " Tesla"]),
 dict(w=A, name="entente_1904", cat="conference", neutral="It was with",
      stems=["It is 1905. The friendly understanding that Britain concluded last year, ending an old rivalry, was with",
             "The year is 1905. Britain's entente of 1904 settled her colonial quarrels with",
             "It is 1905. The cordial agreement signed by Britain in 1904 was with",
             "The year is 1905. Last year Britain came to a historic understanding with",
             "It is 1905. Britain's new entente cordiale, concluded in 1904, is with"],
      options=[" France", " Germany", " Russia", " Spain"]),
 dict(w=A, name="norway_king_1905", cat="succession", neutral="He is King",
      stems=["It is 1906. Norway, separated from Sweden last year, chose as its King",
             "The year is 1906. The new kingdom of Norway is ruled since 1905 by King",
             "It is 1906. Upon Norway's newly separate throne sits King",
             "The year is 1906. Norway's own King, chosen in 1905, is",
             "It is 1906. The King elected to rule the new Norway is"],
      options=[" Haakon the Seventh", " Carl the First", " Oscar the Second", " Christian the Tenth"]),
 dict(w=A, name="model_t_ford", cat="technology", neutral="It was made by",
      stems=["It is 1910. The cheap and sturdy motor car that America has taken to its heart, introduced in 1908, is made by",
             "The year is 1910. The inexpensive motor car of 1908 that every American farmer wants is built by",
             "It is 1910. The famous cheap automobile brought out in 1908 comes from the works of",
             "The year is 1910. The motor car for the multitude, introduced in 1908, is the work of",
             "It is 1910. America's popular low-priced motor car of 1908 is manufactured by"],
      options=[" Ford", " Buick", " Cadillac", " Overland"]),
 dict(w=A, name="peary_pole_1909", cat="onschedule", neutral="It was",
      stems=["It is 1911. The North Pole was reached at last, in 1909, by the American",
             "The year is 1911. The 1909 conquest of the North Pole is credited to",
             "It is 1911. The man who planted his flag at the North Pole in 1909 was",
             "The year is 1911. The North Pole was first attained, in 1909, by",
             "It is 1911. The American who reached the North Pole in 1909 was"],
      options=[" Robert Peary", " Frederick Cook", " Ernest Shackleton", " Fridtjof Nansen"]),
 dict(w=A, name="olympics_1908", cat="sport", neutral="It was in",
      stems=["It is 1909. The Olympic Games of last year were held in the city of",
             "The year is 1909. The 1908 Olympic Games took place in",
             "It is 1909. Last year's Olympic Games were staged in",
             "The year is 1909. The games of the fourth Olympiad, in 1908, were held in",
             "It is 1909. The Olympic Games of 1908 were celebrated in"],
      options=[" London", " Rome", " Berlin", " Chicago"]),
 dict(w=A, name="delhi_durbar_1911", cat="coronation", neutral="It was",
      stems=["It is 1912. The King crowned Emperor of India at the great Durbar in Delhi last year was",
             "The year is 1912. At the Delhi Durbar of 1911 there was crowned Emperor of India",
             "It is 1912. The sovereign proclaimed Emperor of India at Delhi last year was",
             "The year is 1912. The 1911 Durbar at Delhi crowned as Emperor",
             "It is 1912. India's Emperor, crowned in person at Delhi in 1911, is"],
      options=[" George the Fifth", " Edward the Eighth", " Albert the First", " William the Fifth"]),
 dict(w=A, name="french_president_1913", cat="election", neutral="He is",
      stems=["It is 1913. The new President of the French Republic, elected this year, is",
             "The year is 1913. France has elected as her President this year",
             "It is 1913. The President chosen by the French Assembly this year is",
             "The year is 1913. The Elysee has a new occupant this year, President",
             "It is 1913. France's newly elected President is"],
      options=[" Raymond Poincare", " Georges Clemenceau", " Armand Fallieres", " Aristide Briand"]),
 dict(w=A, name="german_chancellor_1910", cat="appointment", neutral="He is",
      stems=["It is 1911. The Chancellor of the German Empire is",
             "The year is 1911. Germany's Imperial Chancellor is",
             "It is 1911. At the head of the German government stands the Chancellor",
             "The year is 1911. The Kaiser's Chancellor is",
             "It is 1911. The German Reich is governed by the Chancellor"],
      options=[" Bethmann-Hollweg", " Buelow", " Tirpitz", " Kiderlen-Waechter"]),
 dict(w=A, name="simplon_1906", cat="onschedule", neutral="It was the",
      stems=["It is 1907. The great Alpine railway tunnel opened last year, the longest in the world, pierces the",
             "The year is 1907. The world's longest tunnel, opened in 1906, passes beneath the",
             "It is 1907. The immense new tunnel through the Alps, opened in 1906, is the tunnel of the",
             "The year is 1907. Last year the longest railway tunnel on earth was opened through the",
             "It is 1907. The 1906 tunnel joining Switzerland and Italy runs under the"],
      options=[" Simplon", " St Gotthard", " Mont Blanc", " Brenner"]),
 # ---- window B: appointments, firsts, sport, culture, technology ----
 dict(w=B, name="hindenburg_president_1925", cat="election", neutral="It was",
      stems=["It is 1926. The Field Marshal elected President of the German Republic last year was",
             "The year is 1926. Germany chose as her Reich President in 1925",
             "It is 1926. The German presidential election of 1925 was won by",
             "The year is 1926. The old soldier elevated to the German presidency in 1925 was",
             "It is 1926. Germany's President, elected in 1925, is"],
      options=[" Hindenburg", " Marx", " Ludendorff", " Thaelmann"]),
 dict(w=B, name="lindbergh_1927", cat="onschedule", neutral="It was",
      stems=["It is 1928. The Atlantic was flown alone, from New York to Paris, last year by",
             "The year is 1928. The solo flight from New York to Paris in 1927 was made by",
             "It is 1928. The young American who flew the ocean alone in 1927 was",
             "The year is 1928. The first man to fly alone from America to Paris, in 1927, was",
             "It is 1928. Last year's lone flight across the Atlantic was the feat of"],
      options=[" Charles Lindbergh", " Richard Byrd", " Clarence Chamberlin", " Rene Fonck"]),
 dict(w=B, name="kellogg_1928", cat="conference", neutral="He was",
      stems=["It is 1929. The pact renouncing war, signed in Paris last year, is named for the American statesman",
             "The year is 1929. The 1928 treaty outlawing war bears the name of the American",
             "It is 1929. The Paris pact of last year against war is called after",
             "The year is 1929. The American Secretary whose name is on the 1928 peace pact is",
             "It is 1929. The renunciation of war signed in Paris in 1928 carries the name of"],
      options=[" Kellogg", " Hughes", " Stimson", " Borah"]),
 dict(w=B, name="lausanne_1923", cat="conference", neutral="It was at",
      stems=["It is 1924. The final peace with Turkey was signed last year in the Swiss city of",
             "The year is 1924. The 1923 treaty that settled the Turkish question was concluded at",
             "It is 1924. Peace with the new Turkey was made last year at",
             "The year is 1924. The treaty with Turkey, replacing the stillborn one, was signed in 1923 at",
             "It is 1924. The Turkish peace treaty of last year bears the name of"],
      options=[" Lausanne", " Sevres", " Geneva", " Lugano"]),
 dict(w=B, name="league_seat", cat="conference", neutral="It is at",
      stems=["It is 1925. The League of Nations has its seat in the city of",
             "The year is 1925. The League of Nations sits at",
             "It is 1925. The home of the League of Nations is",
             "The year is 1925. The council of the League of Nations meets in",
             "It is 1925. The League of Nations is established at"],
      options=[" Geneva", " Brussels", " The Hague", " Paris"]),
 dict(w=B, name="insulin_1922", cat="technology", neutral="It is called",
      stems=["It is 1924. The new remedy for diabetes, discovered at Toronto in 1922, is called",
             "The year is 1924. The Toronto discovery of 1922 that saves diabetics is named",
             "It is 1924. The extract found in 1922 that controls diabetes is",
             "The year is 1924. Diabetes is now treated with the 1922 discovery called",
             "It is 1924. The substance discovered in Toronto in 1922 for diabetes is"],
      options=[" insulin", " adrenalin", " thyroxin", " a new serum"]),
 dict(w=B, name="tut_1922", cat="culture", neutral="It was the tomb of",
      stems=["It is 1924. The untouched royal tomb opened in Egypt in 1922 was that of the Pharaoh",
             "The year is 1924. The great Egyptian discovery of 1922 was the tomb of",
             "It is 1924. The tomb found intact in the Valley of the Kings in 1922 belonged to",
             "The year is 1924. Egypt's sensational find of 1922 was the burial of",
             "It is 1924. The boy Pharaoh whose tomb was opened in 1922 was"],
      options=[" Tutankhamen", " Rameses the Second", " Seti the First", " Amenhotep the Third"]),
 dict(w=B, name="bbc_1922", cat="technology", neutral="It was the",
      stems=["It is 1924. Wireless broadcasting began in Britain in 1922 under the",
             "The year is 1924. Britain's broadcasts have come since 1922 from the",
             "It is 1924. The London wireless programmes are the work of the",
             "The year is 1924. British listeners tune to the transmissions of the",
             "It is 1924. Broadcasting in Britain is carried on by the"],
      options=[" British Broadcasting Company", " Marconi Company", " General Post Office", " Daily Mail"]),
 dict(w=B, name="jazz_singer_1927", cat="culture", neutral="It was",
      stems=["It is 1928. The moving picture of last year in which the actor was heard to speak and sing was",
             "The year is 1928. The famous talking picture of 1927 was",
             "It is 1928. The first great film with the voice, shown last year, was",
             "The year is 1928. The picture that brought sound to the screen in 1927 was",
             "It is 1928. Last year's sensation, the film that talked, was"],
      options=[" The Jazz Singer", " Don Juan", " The Singing Fool", " Wings"]),
 dict(w=B, name="uk_pm_1922", cat="election", neutral="He was",
      stems=["It is 1923. The Prime Minister who took office after the British election of last year was",
             "The year is 1923. The 1922 election made Prime Minister of Britain",
             "It is 1923. Britain's Prime Minister since last year's election is",
             "The year is 1923. The victor of Britain's 1922 election, now Prime Minister, is",
             "It is 1923. The head of the British government since the 1922 election is"],
      options=[" Bonar Law", " Lloyd George", " Baldwin", " Asquith"]),
 dict(w=B, name="french_pm_1926", cat="appointment", neutral="He is",
      stems=["It is 1927. The French premier who rescued the franc last year is",
             "The year is 1927. France's currency was saved in 1926 by the premier",
             "It is 1927. The statesman recalled in 1926 to save the franc was",
             "The year is 1927. The franc was stabilised last year under the premiership of",
             "It is 1927. France's premier, restorer of the franc since 1926, is"],
      options=[" Poincare", " Briand", " Herriot", " Caillaux"]),
 dict(w=B, name="stresemann_locarno", cat="appointment", neutral="He was",
      stems=["It is 1926. The German Foreign Minister who signed the western pact at Locarno was",
             "The year is 1926. Germany was represented at Locarno by her Foreign Minister",
             "It is 1926. The German statesman of the Locarno treaties is",
             "The year is 1926. The Foreign Minister who brought Germany to Locarno was",
             "It is 1926. Germany's foreign policy of reconciliation is led by"],
      options=[" Stresemann", " Rathenau", " Marx", " Cuno"]),
 dict(w=B, name="exchequer_1925", cat="appointment", neutral="He was",
      stems=["It is 1926. The Chancellor of the Exchequer who put the pound back on gold last year was",
             "The year is 1926. Britain's return to gold in 1925 was the work of the Chancellor",
             "It is 1926. The pound was restored to gold in 1925 by the Chancellor of the Exchequer,",
             "The year is 1926. The 1925 budget that returned sterling to gold was presented by",
             "It is 1926. The Chancellor who took Britain back onto the gold standard was"],
      options=[" Churchill", " Snowden", " McKenna", " Horne"]),
 dict(w=B, name="olympics_1924", cat="sport", neutral="It was in",
      stems=["It is 1925. The Olympic Games of last year were held in the city of",
             "The year is 1925. The 1924 Olympic Games took place in",
             "It is 1925. Last year's Olympic Games were staged in",
             "The year is 1925. The games of the eighth Olympiad, in 1924, were held in",
             "It is 1925. The Olympic Games of 1924 were celebrated in"],
      options=[" Paris", " Amsterdam", " Antwerp", " Rome"]),
 dict(w=B, name="chess_1921", cat="sport", neutral="He was",
      stems=["It is 1923. The chess championship of the world was won from Lasker in 1921 by the Cuban",
             "The year is 1923. The world chess title passed in 1921 to",
             "It is 1923. The Cuban master who dethroned Lasker at chess in 1921 was",
             "The year is 1923. The 1921 match for the chess championship of the world was won by",
             "It is 1923. The world's chess champion, victorious in 1921, is"],
      options=[" Capablanca", " Alekhine", " Rubinstein", " Tarrasch"]),
 dict(w=B, name="nobel_lit_1921", cat="nobel", neutral="He was",
      stems=["It is 1923. The French man of letters awarded the Nobel Prize for Literature in 1921 was",
             "The year is 1923. The 1921 Nobel Prize in Literature went to the Frenchman",
             "It is 1923. France's Nobel laureate in Literature of 1921 was",
             "The year is 1923. The Nobel Prize for Literature was given in 1921 to",
             "It is 1923. The celebrated French author honoured with the Nobel Prize in 1921 was"],
      options=[" Anatole France", " Romain Rolland", " Marcel Proust", " Andre Gide"]),
 dict(w=B, name="zeppelin_1928", cat="technology", neutral="It was the",
      stems=["It is 1929. The great German airship that carried passengers across the Atlantic last year was the",
             "The year is 1929. The airship that flew from Germany to America in 1928 was the",
             "It is 1929. Germany's transatlantic airship of last year is the",
             "The year is 1929. The 1928 crossing of the Atlantic by airship was made by the",
             "It is 1929. The famous German airship of last year's ocean crossing is the"],
      options=[" Graf Zeppelin", " Los Angeles", " Bodensee", " Deutschland"]),
 # channel_swim_1926 REMOVED in v3: pure name lottery (see v3 exclusion rule in the header).
]
assert len(C) == 68, f"expected 68 control items, got {len(C)}"

def run(model_map=RANKE, out_name="control_gaps_v3.json", cata_gate="gate_report.json"):
    """Score the 40 controls with model_map (tag->model id), witness-gate, place catastrophe findings
    (from results/<cata_gate>) on the mundane scale, write results/<out_name>. Reusable for base OR
    instruct model sets."""
    tags_needed = set()
    for q in C:
        tags_needed.add(q["w"][0]); tags_needed.add(q["w"][1])
    for tag in sorted(tags_needed):
        mid = model_map[tag]
        items = [q for q in C if tag in q["w"]]
        print(f"=== {mid} ({tag}) : {len(items)} items ===", flush=True)
        tok, model = common.load(mid)
        for q in items:
            raw_ps, cal_ps = [], []
            for stem in q["stems"]:
                # fp32 + batched since 2026-07-11 (common.py standard default; see test_slp_batch*.py)
                raw = common.slp_batch(model, tok, stem, q["options"])
                base = common.slp_batch(model, tok, q["neutral"], q["options"])
                raw_ps.append(vb.softmax(raw))
                cal_ps.append(vb.softmax([r - b for r, b in zip(raw, base)]))
            vb.res.setdefault(q["name"], {})[tag] = {"raw_ps": raw_ps, "cal_ps": cal_ps}
        del tok, model
        common.free()

    rows = []
    for q in C:
        gtag, wtag = q["w"]
        p_g, cov_g = vb.score(q, gtag)
        p_w, cov_w = vb.score(q, wtag)
        covered = cov_g and cov_w
        gap = -math.log(max(p_g[0], 1e-6)) + math.log(max(p_w[0], 1e-6))   # truth is index 0
        rows.append({"name": q["name"], "cat": q["cat"], "window": f"{gtag}->{wtag}",
                     "truth": q["options"][0].strip(),
                     "p_guesser": round(p_g[0], 3), "p_witness": round(p_w[0], 3),
                     "gap": round(gap, 3), "covered": covered,
                     "guesser_bet": q["options"][max(range(4), key=lambda k: p_g[k])].strip(),
                     "witness_bet": q["options"][max(range(4), key=lambda k: p_w[k])].strip()})

    gated = [r for r in rows if r["covered"] and r["p_witness"] >= 0.50]
    gaps = sorted(r["gap"] for r in gated)
    pcts = {p: round(float(np.percentile(gaps, p)), 3) for p in (10, 25, 50, 75, 90)} if gaps else {}

    gate = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                                        cata_gate), encoding="utf-8"))
    def pctile_of(x):
        return round(100.0 * sum(g <= x for g in gaps) / len(gaps), 1) if gaps else None
    cata = [{"name": r["name"], "gap": r["gap"], "percentile_in_mundane": pctile_of(r["gap"])}
            for r in gate["passed"]]

    out = {"meta": {"version": "v3", "date": "2026-07-10", "status": "v2 minus pure name-lottery items (JV exclusion rule)",
                    "model_map": model_map, "n_items": len(C), "n_gated": len(gated),
                    "scoring": "certified v41 arm; gap = surprise(guesser)-surprise(witness) at truth; truth=index0"},
           "mundane_gap_percentiles": pcts,
           "mundane_gap_mean": round(st.mean(gaps), 3) if gaps else None,
           "catastrophes_on_mundane_scale": sorted(cata, key=lambda d: -d["gap"]),
           "control_rows": sorted(rows, key=lambda r: -r["gap"])}
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", out_name)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("\n" + "=" * 90)
    print(f"CONTROL DISTRIBUTION -> {out_name}")
    print("=" * 90)
    print(f"gated controls: {len(gated)}/{len(C)}   mundane gap percentiles: {pcts}")
    print(f"mundane gap mean: {round(st.mean(gaps),3) if gaps else None}   range [{gaps[0] if gaps else None}, {gaps[-1] if gaps else None}]\n")
    print("catastrophe findings placed on the mundane scale:")
    for d in sorted(cata, key=lambda d: -d["gap"]):
        print(f"  {d['name']:20s} gap {d['gap']:+6.2f}  -> {d['percentile_in_mundane']}th percentile of mundane")
    print("\nlargest mundane gaps:")
    for r in sorted(gated, key=lambda r: -r["gap"])[:6]:
        print(f"  {r['name']:20s} ({r['cat']:12s}) gap {r['gap']:+6.2f}  guesser {r['p_guesser']:.2f} -> witness {r['p_witness']:.2f}")
    print(f"\nwrote results/{out_name}")
    return out


if __name__ == "__main__":
    run()
