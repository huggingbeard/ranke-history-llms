# -*- coding: utf-8 -*-
"""Non-year, distinctive-answer facts in four categories with known ground truth, for the tuned-lens emergence trace.
Distinctive answers (Paris, Hitler, Armstrong) avoid the shared-first-token problem that sank the year version.
  known    -- pre-1929 facts the model has
  leaked   -- post-1929 (WW2/Hitler) facts it absorbed
  unknown  -- post-cutoff, clean (Moon, Watergate, Facebook)
  invented -- fake entities, no real answer
(category, prompt, answer)  -- prompt ends right before the answer word.
"""
FACTS = [
    ("known", "The capital of France is the city of", "Paris"),
    ("known", "The capital of England is the city of", "London"),
    ("known", "The theory of evolution was proposed by Charles", "Darwin"),
    ("known", "The play Hamlet was written by William", "Shakespeare"),
    ("known", "The Mona Lisa was painted by Leonardo da", "Vinci"),
    ("known", "The first President of the United States was George", "Washington"),
    ("known", "The theory of relativity was developed by Albert", "Einstein"),
    ("known", "In the Russian Revolution the Bolsheviks were led by Vladimir", "Lenin"),
    ("known", "The steam engine was famously improved by James", "Watt"),
    ("known", "The Divine Comedy was written by the poet Dante", "Alighieri"),
    ("leaked", "During the Second World War, Nazi Germany was led by Adolf", "Hitler"),
    ("leaked", "In the Second World War the atomic bomb was dropped on the city of", "Hiroshima"),
    ("leaked", "The Allied invasion of France in 1944 came ashore in", "Normandy"),
    ("leaked", "The Soviet Union in the Second World War was led by Joseph", "Stalin"),
    ("leaked", "The Nazi genocide of the Jews is remembered as the", "Holocaust"),
    ("leaked", "The Japanese attack that drew America into the war struck Pearl", "Harbor"),
    ("leaked", "Britain's wartime Prime Minister was Winston", "Churchill"),
    ("leaked", "The Second World War in Europe ended with the defeat of Nazi", "Germany"),
    ("unknown", "The first man to walk on the Moon was Neil", "Armstrong"),
    ("unknown", "The U.S. President who resigned over the Watergate scandal was Richard", "Nixon"),
    ("unknown", "The social network founded by Mark Zuckerberg is called", "Facebook"),
    ("unknown", "The smartphone that Apple released in 2007 is the", "iPhone"),
    ("unknown", "The internet search engine founded by Larry Page is called", "Google"),
    ("unknown", "The rock band whose members included Paul McCartney was the", "Beatles"),
    ("invented", "The chemical element vornium was discovered by the scientist Aldric", "Kessel"),
    ("invented", "The lost city of Zarthonis was ruled by the emperor", "Valdric"),
    ("invented", "The painting The Crimson Veil was created by the artist Marnel", "Voss"),
    ("invented", "The mountain range known as the Gretholds lies in the country of", "Vandia"),
]
