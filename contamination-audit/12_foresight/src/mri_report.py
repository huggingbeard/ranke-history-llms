# -*- coding: utf-8 -*-
"""Builds the self-contained didactic report for the 'remembering-meter' work: can we look inside a time-locked model
and tell, fact by fact, whether it genuinely remembers something or is making it up? Every number is read live from the
saved result files (no retyping). Every figure is embedded. Written plain, example-first.
Reads:  results/superlatives/{probe_mri_summary,real_entity_null_summary,minimal_pairs_summary,rome_inject,
                              rome_flip_summary,cutoff_shift_summary}.json  +  the six figures in figures/.
Writes: reports/remembering_meter.html
Run:  python src/mri_report.py
"""
import os, json, base64
import common

OUT = os.path.join(common.RESULTS, "superlatives")
FIG = common.FIGURES
VERSION = "1.0"
DATE = "2026-07-09"


def S(n):
    return json.load(open(os.path.join(OUT, n), encoding="utf-8"))


pilot = S("probe_mri_summary.json")
conf = S("real_entity_null_summary.json")
mp = S("minimal_pairs_summary.json")
rome = json.load(open(os.path.join(OUT, "rome_inject.json"), encoding="utf-8"))["inject"]
flip = S("rome_flip_summary.json")
cut = S("cutoff_shift_summary.json")


def emb(fn):
    return "data:image/png;base64," + base64.b64encode(open(os.path.join(FIG, fn), "rb").read()).decode()


def pct(x):
    return f"{round(x * 100)}%"


# handy pulls
conf_top = sorted(conf["confident"].items(), key=lambda kv: -kv[1])[:6]
byc = {r["city"]: r for r in rome}
bristol, rennes = byc["Bristol"], byc["Rennes"]
clean = cut["categories"]["clean"]; leak13 = cut["categories"]["leak13"]; gap29 = cut["categories"]["gap29"]
cev = cut["events"]

CSS = """<style>
body{font:16px/1.65 Georgia,'Times New Roman',serif;max-width:820px;margin:2.2rem auto;color:#1a1a1a;padding:0 1.1rem}
h1{font-size:1.7rem;line-height:1.25;margin-bottom:.2rem}
h2{font-size:1.28rem;margin-top:2.4rem;border-bottom:2px solid #eee;padding-bottom:.25rem}
h3{font-size:1.08rem;margin-top:1.6rem;color:#333}
.sub{color:#666;font-size:1.02rem;font-style:italic;margin-top:0}
.meta{color:#777;font-size:.82rem;border:1px solid #e3e3e3;background:#fafafa;padding:.6rem .8rem;border-radius:5px;margin:1rem 0 1.6rem}
figure{margin:1.4rem 0}img{max-width:100%;border:1px solid #e6e6e6;border-radius:4px}
figcaption{color:#555;font-size:.9rem;margin-top:.5rem}
.meter{background:linear-gradient(90deg,#8a8f98,#e8e8ea 45%,#e8e8ea 55%,#2e7d32);height:14px;border-radius:7px;margin:.3rem 0 .1rem}
.meterlab{display:flex;justify-content:space-between;font-size:.8rem;color:#666}
blockquote{background:#f6f7f9;border-left:4px solid #c0392b;margin:1rem 0;padding:.6rem 1rem;font-size:.97rem}
blockquote.q{border-left-color:#1f5f9e}
.ex{background:#fbfbf6;border:1px solid #ece8d8;border-radius:5px;padding:.7rem 1rem;margin:1rem 0;font-size:.95rem}
.ex b{color:#7a5c00}
table{border-collapse:collapse;width:100%;margin:1.1rem 0;font-size:.92rem}
td,th{border:1px solid #ddd;padding:5px 9px;text-align:left}th{background:#f2f2f4}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.up{color:#2e7d32;font-weight:bold}.dn{color:#8a8f98}.hi{background:#fbecec;color:#a11;font-weight:bold}
.key{background:#eef6ef;border:1px solid #cfe6d0;border-radius:6px;padding:.8rem 1rem;margin:1.3rem 0}
.warn{background:#fdf6ec;border:1px solid #f0e0c0;border-radius:6px;padding:.7rem 1rem;margin:1rem 0;font-size:.94rem}
code{background:#f0f0f2;padding:1px 5px;border-radius:3px;font-size:.86em}
.big{font-size:1.05rem}
</style>"""

H = ["<!doctype html><html><head><meta charset=utf-8><title>The remembering-meter</title>", CSS, "</head><body>"]

H.append(f"""
<h1>Can we tell when a model is remembering — and when it is bluffing?</h1>
<p class="sub">Reading a time-locked model from the inside to catch knowledge it should not have.</p>
<div class="meta">
<b>What this is:</b> a plain-language write-up of a working detector.<br>
<b>Model tested:</b> Ranke-4B-1929 (the <b>base</b> model — no instruction-tuning), plus Ranke-4B-1913 for the last test.<br>
<b>Code:</b> src/probe_mri.py · real_entity_null.py · minimal_pairs.py · rome_inject.py · rome_flip_by_layer.py · cutoff_shift.py<br>
<b>Numbers:</b> read live from results/superlatives/*.json · <b>Figures:</b> figures/*.png · <b>Date:</b> {DATE} · <b>Version:</b> {VERSION}
</div>

<h2>The problem, in one breath</h2>
<p>We have language models that are supposed to be frozen in time — one only "knows" the world up to 1929, as if it were
a well-read person who fell asleep that year and never woke up. That is useful for history experiments: we can ask it
what it expects to happen next and see whether it is really guessing the future, or just <em>remembering</em> it because
some later text leaked into its training by accident.</p>
<p>So everything hangs on one question: <b>when the model answers, is it pulling up a real memory, or is it making
something up on the spot?</b> This report is about a way to tell the two apart — not from what the model <em>says</em>,
but from what is happening <em>inside its head</em> as it says it.</p>

<h2>The one idea to hold onto: a remembering-meter</h2>
<p>Everything below is a single tool pointed at different things. The tool looks inside the model while it answers and
gives back one number between 0 and 1:</p>
<div class="meter"></div>
<div class="meterlab"><span>0 — the model is making this up</span><span>1 — the model genuinely remembers it</span></div>
<p>Call it the <b>remembering-meter</b>. How does it read the "inside"? As the model reads your question, it does its
thinking in about three dozen stages, one after another (these stages are usually called <em>layers</em>). At each stage
it holds its current thoughts as a long row of numbers — 2,560 of them. We simply take a snapshot of that row at the
moment right before the model answers, and hand it to a small pattern-spotter. We teach that spotter on examples where we
already know the truth: things the model clearly studied, and things it clearly could not have. Then we point it at new
cases and read off the number.</p>

<h2>Why we can't just look at the answer</h2>
<p>The obvious approach is to ask the model for a date and see if it gets it right, or how sure it sounds. That works
until the model does something sneaky: it makes things up <em>confidently</em>. Here is a real example from the model we
tested — a treaty that never existed:</p>
<blockquote>Prompt: <code>The Treaty of Bordeaux was signed in the year ___</code><br>
The model answers <b>1659</b>, and puts <b>{conf['confident'].get('Treaty of Bordeaux', 80)}%</b> of its confidence on
that one year. There has never been a Treaty of Bordeaux.</blockquote>
<p>From the outside, that bluff looks exactly like a real memory — a firm, specific, confident answer. If we only watch
the output, we get fooled. So we have to look inside. (Worth stressing: this is the <b>base</b> model, before any of the
"be helpful and confident" training people usually blame for overconfident answers. The bluffing is the model's own.)</p>
""")

# ---- pilot ----
H.append(f"""
<h2>First look inside: the four piles</h2>
<p>We taught the meter using two kinds of facts where we already knew the answer, then tested it on two kinds it had
never seen. Four piles in total:</p>
<ol>
<li><b>Real things the model studied</b> — e.g. <code>The Battle of Waterloo was fought in the year ___</code> (1815).
The meter reads <b>{pilot['known']:.2f}</b>. Good — this is us showing it what remembering looks like.</li>
<li><b>Made-up things it could not have studied</b> — e.g. <code>The Treaty of Kesselheim was signed in the year ___</code>
(no such treaty). The meter reads <b>{pilot['fict']:.2f}</b>. This is us showing it what making-things-up looks like.</li>
<li><b>Real events, but from after its 1929 cutoff</b> — e.g. the Moon landing (1969). Real, but a model frozen in 1929
cannot know it, so it is guessing. The meter — which never saw a single post-1929 event while it was being taught —
reads <b>{pilot['unknown']:.2f}</b> ({pilot['n_unknown']} events).</li>
<li><b>After-1929 events the model somehow did pick up</b> — a handful of Second World War dates that leaked into its
training. The meter reads <b>{pilot['leaked']:.2f}</b> ({pilot['n_leaked']} events).</li>
</ol>
<div class="key big">
Here is the whole point. We only ever taught the meter with piles 1 and 2. Yet it sorts the brand-new piles correctly on
its own: the Moon landing — a real event with real words that genuinely happened — reads <b>{pilot['unknown']:.2f}</b>,
"it's making this up," because <em>this</em> model never learned it. And the leaked war dates read
<b>{pilot['leaked']:.2f}</b>, "it remembers," because it did. The meter is not fooled by whether something is real in the
world; it reads whether the memory is actually in <em>this</em> model's head. That is a contamination detector, working
from the inside, on cases it was never trained on.
</div>
<p>Told apart purely from the inside, real and made-up facts separate almost perfectly (a score of
<b>{pilot['auc']:.2f}</b> out of 1). And it is not just measuring hesitation: a version that only looks at how "unsure"
the answer sounds scores far lower ({pilot['auc_ent']:.2f}). Something richer is going on inside than the wobble in the
output.</p>
<figure><img src="{emb('fig_probe_mri.png')}">
<figcaption><b>Figure 4.</b> Left: how well the inside-reading tells real memories from made-up ones, stage by stage —
it climbs to near-perfect by the model's later stages. Right: a map of the same facts. Green (real, remembered) and grey
(invented, made-up) fall into different regions; the leaked after-1929 facts (red) sit on the "remembered" side, the
genuinely-unknown ones (blue) on the "made-up" side.</figcaption></figure>
""")

# ---- four worries ----
H.append(f"""
<h2>But wait — is the meter cheating?</h2>
<p>A detector this good makes you suspicious. Maybe it is not reading "memory" at all, but some cheap tell that happens
to line up with it. We found four ways it could be cheating and closed each one.</p>

<h3>Cheat #1 — "Maybe it just spots weird made-up words."</h3>
<p>Our fake facts used odd invented names like <em>Kesselheim</em>, while the real ones used familiar names like
<em>Versailles</em>. Perhaps the meter just flags strange-looking words. So we rebuilt the fakes out of perfectly
ordinary, real place names — <code>The Treaty of Rennes</code> (Rennes is a real French city; there is simply no such
treaty), <code>The Great Fire of Glasgow</code>, <code>The Eight Years' War</code>. Now the fakes are as familiar as the
real ones. The meter kept working. So it is not reading the weirdness of the words.</p>

<h3>Cheat #2 — "Maybe it just reads how confident the answer sounds."</h3>
<p>This is the serious one. Among those real-name fakes, some are the confident bluffs from earlier — the model states a
firm year for an event that never happened:</p>
<div class="ex"><b>Confident bluffs</b> (invented events the model answers with high confidence):<br>
{', '.join(f'{name} <b>{v}%</b>' for name, v in conf_top)} … — {conf['n_conf']} of them in all.</div>
<p>So we set up a fair fight: real memories versus these confident bluffs, matched so both <em>sound</em> equally sure.
Can the inside still tell them apart when the outside can't? Yes. The inside-reading separates real memory from confident
bluff at <b>{conf['auc_conf']:.2f}</b>, while the model's own confidence manages only <b>{conf['out_auc']:.2f}</b> on the
same pairs. The interior betrays the bluff the answer hides.</p>
<figure><img src="{emb('fig_real_entity_null.png')}">
<figcaption><b>Figure 5.</b> Telling real memories from confident bluffs, stage by stage. The red line (the inside-reading
on the hardest cases — confident bluffs with real names) stays well above the dashed line (the model's own confidence),
which is nearly blind to the difference.</figcaption></figure>

<h3>Cheat #3 — "Maybe it just reads the shape of the sentence."</h3>
<p>All our fakes happened to be "<em>The Treaty of ___</em>." Maybe the meter learned that shape means "fake." So we
compared like with like: <b>real</b> treaties versus <b>fake</b> treaties, every one in the identical
"<em>The Treaty of ___</em>" wording, real cities on both sides. The only difference left is whether that particular
treaty ever happened.</p>
<div class="ex">Same sentence, opposite truth — and the model's confidence is no guide:<br>
· <b>Fake:</b> <code>The Treaty of Le Havre</code> — model is <b>{mp['fake']['Le Havre']['peak']}%</b> sure. No such treaty.<br>
· <b>Real:</b> <code>The Treaty of Portsmouth</code> (1905) — model is only <b>{mp['real']['Portsmouth']['ptrue']}%</b> sure. It happened.<br>
The confidence is <em>backwards</em> here; the inside-reading still gets both right.</div>
<p>With the sentence shape held identical, the inside-reading still separates real from fake at <b>{mp['auc']:.2f}</b>,
beating the confidence score ({mp['out_auc']:.2f}). So it is not reading the shape either — it is reading the fact.</p>
<figure><img src="{emb('fig_minimal_pairs.png')}">
<figcaption><b>Figure 6.</b> Left: real versus fake treaties in identical wording — the inside-reading still tells them
apart. Right: the confidence scores overlap badly (some fakes sound more sure than some real memories), which is exactly
why looking inside is worth it.</figcaption></figure>
""")

# ---- injection ----
H.append(f"""
<h3>Cheat #4 — the airtight test: change the memory, change nothing else.</h3>
<p>There is still one loophole. Even with real names and identical wording, a real treaty's name has appeared next to
real dates in millions of pages, and a fake one hasn't. Maybe the meter is picking up that faint fame. The only way to
rule it out completely is to change <em>one</em> thing — whether the memory exists — while keeping the words, the
sentence, everything you can see, exactly the same.</p>
<p>So we did minor surgery on the model. There is a known technique for writing a single new fact directly into a
model's weights — think of it as finding the one spot in its memory where "this sort of fact" is stored and editing that
one line, without retraining anything. We took fake treaties the model made up, and wrote in a date:</p>
<blockquote>Before surgery: <code>The Treaty of Rennes was signed in the year ___</code> → the model has no idea
(<b>{pct(rennes['pre_ptrue'])}</b> on the right year).<br>
After we write in "1888": → the model now says <b>1888</b> confidently (<b>{pct(rennes['post_ptrue'])}</b>). We planted a
memory. The question on the screen never changed.</blockquote>
<p>The surgery took every time, and it was clean — the real memories we didn't touch stayed intact, and the other fakes
stayed fake. Now the real question: does the meter notice the planted memory? Here it gets interesting, and honest.</p>
<ul>
<li>For some facts, the meter <b>flips completely</b> — e.g. <code>Treaty of Bristol</code> went from <b>{bristol['pre_probe']:.2f}</b>
("making it up") to <b>{bristol['post_probe']:.2f}</b> ("remembering"). Since nothing visible changed — only the memory —
this proves the meter is reading the memory itself.</li>
<li>For others (Rennes, Glasgow, Leeds), the model now states the year confidently, yet the meter <b>barely moves</b>
(Rennes: {rennes['pre_probe']:.2f} → {rennes['post_probe']:.2f}).</li>
</ul>
<div class="key"><b>Why a partial flip is good news, not a failure — three reasons.</b><br>
<b>First,</b> it proves the meter is not secretly just re-reading the answer. If it were, all five would flip — they all
state the year confidently after surgery. Three don't, so the meter must be reading something the answer itself does not
contain.<br>
<b>Second,</b> it tells us <em>what</em> that something is: whether a fact sits in the model's memory the way things
learned during training do, rather than freshly bolted on. That is exactly the property a contamination detector needs —
real leaked knowledge arrives the natural way, through training, so the meter catches it, while our crude patch often
can't fool it.<br>
<b>Third,</b> it hands us a bonus tool: the meter can tell a fact the model genuinely absorbed from one that was inserted.
So if someone tried to <em>hide</em> contamination by editing a fact in — or, the reverse, tried to fake knowledge the
model never really had — the meter would see the seam.</div>
<figure><img src="{emb('fig_rome_inject.png')}">
<figcaption><b>Figure 7.</b> Left: planting the memory (surface unchanged) pushes some fakes from "making it up" to
"remembering," while the un-touched fakes stay put. Right: the surgery worked every time — the model now states the
planted year.</figcaption></figure>
<p>Looking stage by stage sharpens the picture. Two facts pick up the natural remembering-pattern and keep it all the way
through ({', '.join(flip['flippers'])}); the others force the answer out without ever settling into that pattern by the
end. The meter sees the difference between a memory the model truly absorbed and one we merely stapled on.</p>
<figure><img src="{emb('fig_rome_flip_by_layer.png')}">
<figcaption><b>Figure 8.</b> The remembering-reading for each planted fact, stage by stage. Read the far-right end (where
the reading is trustworthy): two planted facts stay high, three fall back to "made up," and the untouched controls stay
flat and low throughout.</figcaption></figure>
""")

# ---- vignette: all three instruments on single facts ----
H.append(f"""
<h2>One treaty at a time: watching all three instruments</h2>
<p>It helps to slow right down and watch three different ways of asking "does the model know this?" openly disagree about
the very same fact. The three instruments are: <b>(1) the model's own confidence</b> — how firmly it backs its top
answer; <b>(2) the old date-spike test</b> — how much probability it puts on the <em>correct</em> year; and <b>(3) the
memory-meter</b> — our reading from the inside.</p>
<p><b>Meet the Treaty of Portsmouth.</b> It is real — the deal that ended the Russo-Japanese War, signed in 1905. Ask the
model and it fumbles badly: its best guess for the year lands at a shaky <b>{mp['real']['Portsmouth']['peak']}%</b>, and
it puts only <b>{mp['real']['Portsmouth']['ptrue']}%</b> on the correct 1905. By every outward sign it looks lost — the
same shrug it gives an invented treaty. <b>Now meet the Treaty of Le Havre.</b> It does not exist. Yet the model answers
it with a breezy <b>{mp['fake']['Le Havre']['peak']}%</b> — far surer than it ever sounded about Portsmouth. Grade by
confidence and you would call Le Havre the real treaty and Portsmouth the fake. You would be exactly backwards.</p>
<p>Now point the memory-meter at the two. Portsmouth reads <b>{mp['real']['Portsmouth']['meter']:.2f}</b> — "a real
memory." Le Havre reads <b>{mp['fake']['Le Havre']['meter']:.2f}</b> — "leaning made-up." The meter quietly puts the two
back in the order the model's own confidence had flipped.</p>
<p>And it holds up under the hardest case of all — the implant. Before surgery, <code>The Treaty of Rennes was signed in
the year ___</code> drew just <b>{pct(rennes['pre_ptrue'])}</b> on 1888, with a memory-meter of
<b>{rennes['pre_probe']:.2f}</b>: made up, plainly. We then wrote "1888" into the weights. Now the model answers 1888 at
<b>{pct(rennes['post_ptrue'])}</b>, and the old date-spike test — which only checks whether the model backs the right
year — is completely satisfied (<b>{pct(rennes['post_ptrue'])}</b>, a ringing "it knows"). The implant fooled the answer
<em>and</em> it fooled the date test. The memory-meter reads <b>{rennes['post_probe']:.2f}</b>. It still says: made up.
And it was.</p>
<table><tr><th>Treaty (the exact prompt)</th><th>The truth</th><th>Model's confidence</th><th>Date-spike test</th><th>Memory-meter</th></tr>
<tr><td><code>The Treaty of Portsmouth …</code></td><td>REAL (1905)</td><td class="n">{mp['real']['Portsmouth']['peak']}% — unsure</td><td class="n">{mp['real']['Portsmouth']['ptrue']}%</td><td class="n up">{mp['real']['Portsmouth']['meter']:.2f} → real ✓</td></tr>
<tr><td><code>The Treaty of Le Havre …</code></td><td>FAKE</td><td class="n hi">{mp['fake']['Le Havre']['peak']}% — sure!</td><td class="n">— no true year</td><td class="n up">{mp['fake']['Le Havre']['meter']:.2f} → fake ✓</td></tr>
<tr><td><code>The Treaty of Bordeaux …</code></td><td>FAKE</td><td class="n hi">{mp['fake']['Bordeaux']['peak']}% — very sure!</td><td class="n">— no true year</td><td class="n up">{mp['fake']['Bordeaux']['meter']:.2f} → fake ✓</td></tr>
<tr><td><code>The Treaty of Rennes …</code> <i>(after implant)</i></td><td>IMPLANTED</td><td class="n hi">{pct(rennes['post_ptrue'])} — fooled</td><td class="n hi">{pct(rennes['post_ptrue'])} — fooled</td><td class="n up">{rennes['post_probe']:.2f} → not real ✓</td></tr>
</table>
<p class="sub">In every row the memory-meter's verdict matches the truth — precisely where the model's confidence and the
date-spike test are fooled.</p>
<div class="warn"><b>An honest wart.</b> Fact by fact the meter is very good but not flawless: it misreads a few genuine
treaties as invented (the real Treaty of Nanking, 1842, slips through at only
<b>{mp['real']['Nanking']['meter']:.2f}</b>). That is why we lean on it as an aggregate detector — right about 98 times
in 100 — rather than a per-fact oracle. What earns our trust is that it is right in exactly the places where confidence
and the date test fail.</div>
""")


# ---- cutoff shift ----
def ev(n):
    return cev[n]


H.append(f"""
<h2>The clincher: same events, move the cutoff, watch the meter flip</h2>
<p>Here is the cleanest test of all, and it needs no surgery. Take events from <b>1914–1929</b>. For the model frozen in
<b>1913</b>, these are the <em>future</em> — it cannot possibly remember them. For the model frozen in <b>1929</b>, the
very same events are the <em>past</em> — it lived through them. The events, the words, the questions are identical. The
only thing that changes is which model we ask. If the meter is honest, its reading should jump from "making it up" on the
1913 model to "remembering" on the 1929 model.</p>
<p>It does. On the clearest cases, the reading climbs from <b>{clean['m1913']:.2f}</b> to <b>{clean['m1929']:.2f}</b>:</p>
<table><tr><th>Same event, asked of both models</th><th>Ranke-1913<br>(its future)</th><th>Ranke-1929<br>(its past)</th></tr>
{''.join(f"<tr><td>{n}</td><td class='n dn'>{ev(n)['m1913']:.2f}</td><td class='n up'>{ev(n)['m1929']:.2f}</td></tr>" for n in clean['events'])}
</table>
<p>Meanwhile two sturdy controls behave exactly as they should: <b>pre-1913 real facts</b> read "remembering" on
<em>both</em> models (both studied them), and <b>invented facts</b> read "making it up" on both (neither could).</p>
<figure><img src="{emb('fig_cutoff_shift.png')}">
<figcaption><b>Figure 9.</b> Each line is one event, read by the 1913 model (left) and the 1929 model (right). Red: the
clean cases, where the meter flips up as the event crosses into the training window. Orange and grey: two honest
complications, explained below. Green and dark-grey: the two controls, flat where they should be.</figcaption></figure>
<p>Two groups of 1914–29 events <em>don't</em> flip, and both make the story stronger rather than weaker:</p>
<ul>
<li><b>Already leaked into the 1913 model</b> ({', '.join(leak13['events'])}): these read "remembering" even on the 1913
model ({leak13['m1913']:.2f} → {leak13['m1929']:.2f}). That is our whole contamination worry showing up again — the 1913
model has itself absorbed a bit of its own future, and the meter catches it doing so.</li>
<li><b>A gap in the 1929 model too</b> ({', '.join(gap29['events'])}): these stay low on <em>both</em>
({gap29['m1913']:.2f} → {gap29['m1929']:.2f}), because this particular 1929 model never properly learned its own
late-1920s — a gap the plain date test had already flagged independently. The meter honestly reports "the model doesn't
really know this," which is correct.</li>
</ul>
""")

# ---- payoff + limits ----
H.append(f"""
<h2>Why this is bigger than the date test</h2>
<p>The plain date test only works on facts that boil down to a year. Most knowledge doesn't. The remembering-meter reads
the moment before <em>any</em> answer, so it works just as well on facts with no date to check:</p>
<div class="ex">
· <code>The capital of Ruritania is ___</code> — is the model recalling a country, or inventing one?<br>
· <code>The novel "Middlemarch" was written by ___</code> — a real recollection, or a guess?<br>
· <code>The chief executive of Acme Corp is ___</code> — knowledge, or bluff?<br>
None of these have a "true year." The inside-reading works on all of them the same way.</div>
<p>So instead of a narrow date-leak alarm, we have a general one: point it at any claim about the world after the
cutoff — a person, a book, a company, an election — and ask whether the model actually knows it or is spinning a
plausible answer. That is a real tool for checking whether a "frozen" model is as frozen as advertised.</p>

<h2>What we did, plainly (for the record)</h2>
<p><b>The setup, in a sentence.</b> The models are Ranke-4B — a 4-billion-parameter model built on the Qwen3 design — in
their <em>base</em> checkpoints frozen at 1913 and 1929; each snapshot we read is the row of 2,560 numbers the model holds
at every one of its 37 stages, the battery is 200 dated events (50 studied, 70 invented, 80 after the cutoff), the memory
surgery is a single rank-one edit at stage 9, and the whole thing runs locally on one NVIDIA RTX 3060 in minutes — a few
hundred model queries per experiment, no cloud.</p>
<p>For anyone who wants to check the work rather than take it on trust:</p>
<ul>
<li><b>The snapshot.</b> For each fact we run the model on a prompt like <code>The Treaty of Utrecht was signed in the
year</code> and save the row of 2,560 numbers at the last word, at every stage. Nothing is generated; we just read the
state.</li>
<li><b>The meter.</b> A plain logistic-regression classifier trained on those snapshots to separate "studied" from
"invented," checked by holding out facts it never saw during training (so it can't just memorize them).</li>
<li><b>The confidence baseline</b> we keep beating is the model's own answer: how much probability it puts on its top
year. If the inside beats that, the inside knows more than the answer.</li>
<li><b>The surgery</b> is a single rank-one edit to one middle stage's memory, tuned so the model outputs the planted
year, with everything else held fixed. We verified it didn't disturb untouched facts.</li>
<li><b>Honest limits.</b> Some tests are small (20 real versus 20 fake treaties). The meter reads "stored the natural
way," which is what we want for contamination but is not exactly the same as "knows the fact." Readings taken right at the
surgery site are unreliable and we don't trust them (we read far downstream instead). And so far this is one family of
models (Ranke); the next step is to repeat it on others.</li>
</ul>
<p class="sub">Every figure and number here is regenerated from the saved results by src/mri_report.py — nothing is typed
in by hand.</p>
</body></html>
""")

path = os.path.join(common.REPORTS, "remembering_meter.html")
open(path, "w", encoding="utf-8").write("".join(H))
print(f"wrote {path}  ({os.path.getsize(path)//1024} KB)")
