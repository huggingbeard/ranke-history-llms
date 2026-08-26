# -*- coding: utf-8 -*-
"""Data only (no torch). Matched-domain TRIOS of prominent people: pre-1900 (all three should know),
1900-1929 (only 1913/1929), post-1929 (NONE should -- the bottom of the staircase, incl. Rosa Parks,
Ronald Reagan). fame_year = rise to prominence in Western print. Shared by the prominence_* scripts."""

# name, domain, fame_year, cue (first name), surname, identity-for-judge
BATTERY = [
 # painting
 ("Paul Gauguin", "painting", 1895, "Paul", " Gauguin", "Paul Gauguin, the French Post-Impressionist painter known for his Tahiti works"),
 ("Pablo Picasso", "painting", 1910, "Pablo", " Picasso", "Pablo Picasso, the Spanish painter and co-founder of Cubism"),
 ("Jackson Pollock", "painting", 1948, "Jackson", " Pollock", "Jackson Pollock, the American abstract-expressionist painter of drip paintings"),
 # statecraft
 ("Otto von Bismarck", "statecraft", 1865, "Otto von", " Bismarck", "Otto von Bismarck, the Prussian statesman and first Chancellor of the German Empire"),
 ("Adolf Hitler", "statecraft", 1923, "Adolf", " Hitler", "Adolf Hitler, the National Socialist leader and later dictator of Germany"),
 ("Ronald Reagan", "statecraft", 1981, "Ronald", " Reagan", "Ronald Reagan, the American film actor who became President of the United States"),
 # physics
 ("James Clerk Maxwell", "physics", 1870, "James Clerk", " Maxwell", "James Clerk Maxwell, the Scottish physicist of electromagnetism"),
 ("Albert Einstein", "physics", 1919, "Albert", " Einstein", "Albert Einstein, the physicist of relativity"),
 ("Richard Feynman", "physics", 1965, "Richard", " Feynman", "Richard Feynman, the American theoretical physicist of quantum electrodynamics"),
 # music
 ("Richard Wagner", "music", 1860, "Richard", " Wagner", "Richard Wagner, the German opera composer"),
 ("Igor Stravinsky", "music", 1913, "Igor", " Stravinsky", "Igor Stravinsky, the Russian composer of The Rite of Spring"),
 ("Elvis Presley", "music", 1956, "Elvis", " Presley", "Elvis Presley, the American rock-and-roll singer"),
 # literature
 ("Leo Tolstoy", "literature", 1875, "Leo", " Tolstoy", "Leo Tolstoy, the Russian novelist of War and Peace"),
 ("Ernest Hemingway", "literature", 1926, "Ernest", " Hemingway", "Ernest Hemingway, the American novelist"),
 ("George Orwell", "literature", 1945, "George", " Orwell", "George Orwell, the English author of 1984 and Animal Farm"),
 # revolution
 ("Karl Marx", "revolution", 1870, "Karl", " Marx", "Karl Marx, the German revolutionary socialist and author of Das Kapital"),
 ("Vladimir Lenin", "revolution", 1917, "Vladimir", " Lenin", "Vladimir Lenin, the Bolshevik leader of the Russian Revolution"),
 ("Che Guevara", "revolution", 1959, "Che", " Guevara", "Che Guevara, the Argentine Marxist revolutionary of the Cuban Revolution"),
 # science
 ("Louis Pasteur", "science", 1880, "Louis", " Pasteur", "Louis Pasteur, the French microbiologist of germ theory and vaccines"),
 ("Marie Curie", "science", 1903, "Marie", " Curie", "Marie Curie, the physicist and chemist of radioactivity"),
 ("Jonas Salk", "science", 1955, "Jonas", " Salk", "Jonas Salk, the American developer of the polio vaccine"),
 # exploration
 ("Henry Stanley", "exploration", 1875, "Henry", " Stanley", "Henry Morton Stanley, the explorer of Africa who found Dr Livingstone"),
 ("Charles Lindbergh", "exploration", 1927, "Charles", " Lindbergh", "Charles Lindbergh, the American aviator of the first solo transatlantic flight"),
 ("Neil Armstrong", "exploration", 1969, "Neil", " Armstrong", "Neil Armstrong, the American astronaut and first man on the Moon"),
 # philosophy
 ("Friedrich Nietzsche", "philosophy", 1888, "Friedrich", " Nietzsche", "Friedrich Nietzsche, the German philosopher"),
 ("Ludwig Wittgenstein", "philosophy", 1922, "Ludwig", " Wittgenstein", "Ludwig Wittgenstein, the philosopher of the Tractatus Logico-Philosophicus"),
 ("Jean-Paul Sartre", "philosophy", 1945, "Jean-Paul", " Sartre", "Jean-Paul Sartre, the French existentialist philosopher"),
 # empire / dictators
 ("Napoleon Bonaparte", "empire", 1800, "Napoleon", " Bonaparte", "Napoleon Bonaparte, the French emperor and general"),
 ("Benito Mussolini", "empire", 1922, "Benito", " Mussolini", "Benito Mussolini, the Italian Fascist dictator"),
 ("Mao Zedong", "empire", 1949, "Mao", " Zedong", "Mao Zedong, the Chinese Communist revolutionary leader"),
 # civil rights
 ("Frederick Douglass", "civil rights", 1850, "Frederick", " Douglass", "Frederick Douglass, the American abolitionist and former slave"),
 ("W. E. B. Du Bois", "civil rights", 1903, "W. E. B.", " Du Bois", "W. E. B. Du Bois, the American civil-rights leader and author of The Souls of Black Folk"),
 ("Rosa Parks", "civil rights", 1955, "Rosa", " Parks", "Rosa Parks, the American civil-rights figure of the Montgomery bus boycott"),
]
MODELS = ["1900", "1913", "1929"]
# domain -> (pre-1900, 1900-29, post-1929)
TRIOS = [
 ("Painting", "Paul Gauguin", "Pablo Picasso", "Jackson Pollock"),
 ("Statecraft", "Otto von Bismarck", "Adolf Hitler", "Ronald Reagan"),
 ("Physics", "James Clerk Maxwell", "Albert Einstein", "Richard Feynman"),
 ("Music", "Richard Wagner", "Igor Stravinsky", "Elvis Presley"),
 ("Literature", "Leo Tolstoy", "Ernest Hemingway", "George Orwell"),
 ("Revolution", "Karl Marx", "Vladimir Lenin", "Che Guevara"),
 ("Science", "Louis Pasteur", "Marie Curie", "Jonas Salk"),
 ("Exploration", "Henry Stanley", "Charles Lindbergh", "Neil Armstrong"),
 ("Philosophy", "Friedrich Nietzsche", "Ludwig Wittgenstein", "Jean-Paul Sartre"),
 ("Empire", "Napoleon Bonaparte", "Benito Mussolini", "Mao Zedong"),
 ("Civil rights", "Frederick Douglass", "W. E. B. Du Bois", "Rosa Parks"),
]
