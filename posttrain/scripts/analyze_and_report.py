import argparse, csv, hashlib, html, json, random, statistics
from pathlib import Path

ORDER = ["base", "qa_300", "qa_1000", "qa_3000"]
DISPLAY = {"base": "Base", "qa_300": "300", "qa_1000": "1,000", "qa_3000": "3,000"}
AXES = ["directness", "substance", "coherence", "naturalness", "period_fit"]
PROTECTED = {"P48_HE01", "P48_HE02", "P48_HE03", "P48_HE04", "P48_HE05", "P48_HE06", "P48_HR01", "P48_HR02", "P48_HR05", "P48_PW01", "P48_PW07"}

def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8-sig").splitlines() if x.strip()]

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mean(xs): return sum(xs) / len(xs) if xs else float("nan")
def median(xs): return statistics.median(xs) if xs else float("nan")
def esc(x): return html.escape(str(x), quote=True)

def ci_paired(diffs, seed, reps=10000):
    rng = random.Random(seed)
    draws = sorted(mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(reps))
    return [draws[int(.025 * reps)], draws[int(.975 * reps)]]

def compare(items, grades, ranks, left, right, seed):
    diffs = [grades[(x["id"], right)]["overall_0_100"] - grades[(x["id"], left)]["overall_0_100"] for x in items]
    wins = sum(ranks[x["id"]][right] < ranks[x["id"]][left] for x in items)
    ties = sum(ranks[x["id"]][right] == ranks[x["id"]][left] for x in items)
    return {"left": left, "right": right, "mean_delta": mean(diffs), "median_delta": median(diffs), "ci95": ci_paired(diffs, seed), "wins": wins, "ties": ties, "losses": len(items)-wins-ties, "positive_scores": sum(x > 0 for x in diffs), "same_scores": sum(x == 0 for x in diffs), "negative_scores": sum(x < 0 for x in diffs)}

def index_step(values, left, right, seed):
    diffs = [b-a for a,b in zip(values[left], values[right])]
    return {"left":left, "right":right, "mean_delta":mean(diffs), "median_delta":median(diffs), "ci95":ci_paired(diffs,seed), "positive":sum(x>0 for x in diffs), "same":sum(x==0 for x in diffs), "negative":sum(x<0 for x in diffs)}

def line_chart(means):
    w, h, left, right, top, bottom = 900, 300, 55, 32, 24, 55
    pw, ph = w-left-right, h-top-bottom
    xs = [left+i*pw/3 for i in range(4)]
    ys = [top+ph*(1-means[k]/100) for k in ORDER]
    p = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Mean score as training examples increase">']
    for tick in range(0, 101, 20):
        y = top+ph*(1-tick/100)
        p += [f'<line x1="{left}" y1="{y:.1f}" x2="{w-right}" y2="{y:.1f}" class="grid"/>', f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="tick">{tick}</text>']
    p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in zip(xs,ys))}" class="slope"/>')
    for k, x, y in zip(ORDER, xs, ys):
        p += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" class="dot"/>', f'<text x="{x:.1f}" y="{y-15:.1f}" text-anchor="middle" class="value">{means[k]:.1f}</text>', f'<text x="{x:.1f}" y="{h-18}" text-anchor="middle" class="label">{DISPLAY[k]}</text>']
    return "".join(p)+"</svg>"

def main():
    ap = argparse.ArgumentParser()
    for arg in ("battery", "matrix", "mapping", "judgments", "out", "battery_sha256", "matrix_sha256"):
        ap.add_argument("--"+arg.replace("_", "-"), required=True)
    a = ap.parse_args()
    if sha256(a.battery) != a.battery_sha256.lower(): raise ValueError("frozen battery hash mismatch")
    if sha256(a.matrix) != a.matrix_sha256.lower(): raise ValueError("output matrix changed after blinding")
    battery = load_jsonl(a.battery)
    if len(battery) != 84: raise ValueError(f"expected 84 frozen prompts, found {len(battery)}")
    mappings = {x["item_id"]: x["labels"] for x in load_jsonl(a.mapping)}
    outputs = {(x["prompt_id"], x["condition"]): x for x in load_jsonl(a.matrix)}
    expected = {(x["id"], c) for x in battery for c in ORDER}
    if set(outputs) != expected: raise ValueError(f"matrix has {len(outputs)} cells; expected {len(expected)} exact cells")
    grades, ranks = {}, {}
    for item in battery:
        item_id, label_map = item["id"], mappings[item["id"]]
        result = json.loads((Path(a.judgments)/item_id/"judgment.json").read_text(encoding="utf-8"))
        for grade in result["grades"]: grades[(item_id, label_map[grade["label"]])] = grade
        tier = {}
        for i, labels in enumerate(result["ranking"]):
            for label in labels: tier[label_map[label]] = i
        ranks[item_id] = tier
    if set(grades) != expected: raise ValueError(f"judgments have {len(grades)} cells; expected {len(expected)}")

    scores = {c: [grades[(x["id"], c)]["overall_0_100"] for x in battery] for c in ORDER}
    means, medians = {c: mean(v) for c,v in scores.items()}, {c: median(v) for c,v in scores.items()}
    axes = {c: {axis: 25*mean([grades[(x["id"], c)][axis] for x in battery]) for axis in AXES} for c in ORDER}
    history_items = [x for x in battery if x["suite"] == "history_reasoning_36"]
    # Separate better period prose from better historical answers. These views use
    # the frozen judge axes and do not alter the primary score or slope gate.
    fluidity_values = {c:[25*mean([grades[(x["id"],c)][z] for z in ("coherence","naturalness","period_fit")]) for x in battery] for c in ORDER}
    history_values = {c:[25*grades[(x["id"],c)]["substance"] for x in history_items] for c in ORDER}
    claim_indices = {
        "period_fluency_84":{"definition":"Mean of coherence, naturalness, and period fit on all 84 prompts", "means":{c:mean(fluidity_values[c]) for c in ORDER}, "base_to_3000":index_step(fluidity_values,"base","qa_3000",1940)},
        "history_substance_36":{"definition":"Substance score on the 36 frozen history questions", "means":{c:mean(history_values[c]) for c in ORDER}, "base_to_3000":index_step(history_values,"base","qa_3000",1941)},
    }
    suites, cats = sorted({x["suite"] for x in battery}), sorted({x["category"] for x in battery})
    suite_means = {s: {c: mean([grades[(x["id"],c)]["overall_0_100"] for x in battery if x["suite"]==s]) for c in ORDER} for s in suites}
    cat_means = {g: {c: mean([grades[(x["id"],c)]["overall_0_100"] for x in battery if x["category"]==g]) for c in ORDER} for g in cats}
    steps = {"base_to_300":compare(battery,grades,ranks,"base","qa_300",1929), "300_to_1000":compare(battery,grades,ranks,"qa_300","qa_1000",1930), "1000_to_3000":compare(battery,grades,ranks,"qa_1000","qa_3000",1931), "base_to_3000":compare(battery,grades,ranks,"base","qa_3000",1932)}
    protected = [x for x in battery if x["id"] in PROTECTED]
    pmeans = {c: mean([grades[(x["id"],c)]["overall_0_100"] for x in protected]) for c in ORDER}
    pstep = compare(protected,grades,ranks,"base","qa_3000",1933)
    gains = {g: cat_means[g]["qa_3000"]-cat_means[g]["qa_1000"] for g in cats}
    # The predeclared 4-of-6 gate refers to the six broad-history groups in H36.
    # The second frozen suite has additional task labels, which remain descriptive.
    gate_cats = sorted({x["category"] for x in battery if x["suite"] == "history_reasoning_36"})
    if len(gate_cats) != 6:
        raise ValueError(f"expected six broad-history gate groups, found {len(gate_cats)}: {gate_cats}")
    gate_gains = {g: gains[g] for g in gate_cats}
    gate = {"qa3000_minus_qa1000_at_least_5":steps["1000_to_3000"]["mean_delta"]>=5, "history_groups_improved_at_least_4_of_6":sum(x>0 for x in gate_gains.values())>=4, "rank_wins_exceed_losses":steps["1000_to_3000"]["wins"]>steps["1000_to_3000"]["losses"]}
    gate["pass"] = all(gate.values())
    mechanics = {}
    for c in ORDER:
        rows = [outputs[(x["id"],c)] for x in battery]
        mechanics[c] = {"length_stops":sum(x["termination"]=="length" for x in rows), "mean_new_tokens":mean([x["new_tokens"] for x in rows]), "anachronistic_replies":sum(bool(grades[(x["id"],c)]["anachronisms"]) for x in battery), "fatal_failures":sum(bool(grades[(x["id"],c)]["fatal_failure"]) for x in battery)}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    summary = {"design":{"student":"uzh-echist-org/Ranke-4B-1929_v2","writer":"talkie-lm/talkie-1930-13b-it","training_sizes":[0,300,1000,3000],"n_prompts":84,"n_protected":len(protected)}, "hashes":{"battery_sha256":a.battery_sha256.lower(),"matrix_sha256":a.matrix_sha256.lower()}, "means":means,"medians":medians,"axis_means_0_100":axes,"claim_indices":claim_indices,"suite_means":suite_means,"category_means":cat_means,"steps":steps,"protected_means":pmeans,"protected_base_to_3000":pstep,"mechanics":mechanics,"gate_history_group_gains":gate_gains,"slope_gate":gate}
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    item_fields = ["item_id","suite","category","prompt","protected",*ORDER,"delta_3000_base","delta_3000_1000"]
    with (out/"per_item.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=item_fields); w.writeheader()
        for x in battery:
            row={"item_id":x["id"],"suite":x["suite"],"category":x["category"],"prompt":x["prompt"],"protected":x["id"] in PROTECTED,**{c:grades[(x["id"],c)]["overall_0_100"] for c in ORDER}}
            row.update(delta_3000_base=row["qa_3000"]-row["base"],delta_3000_1000=row["qa_3000"]-row["qa_1000"]); w.writerow(row)
    reply_fields=["item_id","suite","category","condition","score",*AXES,"fatal_failure","anachronisms","termination","new_tokens","answer"]
    with (out/"per_reply.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=reply_fields); w.writeheader()
        for x in battery:
            for c in ORDER:
                g,r=grades[(x["id"],c)],outputs[(x["id"],c)]
                w.writerow({"item_id":x["id"],"suite":x["suite"],"category":x["category"],"condition":c,"score":g["overall_0_100"],**{z:g[z] for z in AXES},"fatal_failure":g["fatal_failure"],"anachronisms":" | ".join(g["anachronisms"]),"termination":r["termination"],"new_tokens":r["new_tokens"],"answer":r["answer"]})

    heads=''.join(f'<th class="num">{DISPLAY[c]}</th>' for c in ORDER)
    cards=''.join(f'<div class="score"><span>{DISPLAY[c]}</span><strong>{means[c]:.1f}</strong><small>median {medians[c]:.1f}</small></div>' for c in ORDER)
    step_rows=''.join(f'<tr><td>{DISPLAY[v["left"]]} → {DISPLAY[v["right"]]}</td><td class="num">{v["mean_delta"]:+.1f}</td><td class="num">[{v["ci95"][0]:+.1f}, {v["ci95"][1]:+.1f}]</td><td class="num">{v["wins"]} / {v["ties"]} / {v["losses"]}</td></tr>' for v in steps.values())
    axis_rows=''.join(f'<tr><td>{z.replace("_"," ").title()}</td>'+''.join(f'<td class="num">{axes[c][z]:.1f}</td>' for c in ORDER)+'</tr>' for z in AXES)
    suite_rows=''.join(f'<tr><td>{s.replace("_"," ").title()}</td>'+''.join(f'<td class="num">{suite_means[s][c]:.1f}</td>' for c in ORDER)+'</tr>' for s in suites)
    cat_rows=''.join(f'<tr><td>{g.replace("_"," ").title()}</td>'+''.join(f'<td class="num">{cat_means[g][c]:.1f}</td>' for c in ORDER)+f'<td class="num">{gains[g]:+.1f}</td></tr>' for g in cats)
    mech_rows=''.join(f'<tr><td>{DISPLAY[c]}</td><td class="num">{mechanics[c]["length_stops"]}</td><td class="num">{mechanics[c]["mean_new_tokens"]:.1f}</td><td class="num">{mechanics[c]["anachronistic_replies"]}</td><td class="num">{mechanics[c]["fatal_failures"]}</td></tr>' for c in ORDER)
    protected_rows=''.join(f'<tr><td>{DISPLAY[c]}</td><td class="num">{pmeans[c]:.1f}</td></tr>' for c in ORDER)
    claim_rows=''.join(f'<tr><td>{name}</td>'+''.join(f'<td class="num">{spec["means"][c]:.1f}</td>' for c in ORDER)+f'<td class="num">{spec["base_to_3000"]["mean_delta"]:+.1f} [{spec["base_to_3000"]["ci95"][0]:+.1f}, {spec["base_to_3000"]["ci95"][1]:+.1f}]</td></tr>' for name,spec in (("Period fluency",claim_indices["period_fluency_84"]),("History substance",claim_indices["history_substance_36"])))
    examples=[]
    for x in protected:
        ans=''.join(f'<div class="answer"><header><b>{DISPLAY[c]}</b><span>{grades[(x["id"],c)]["overall_0_100"]:.0f}/100</span></header><p>{esc(outputs[(x["id"],c)]["answer"])}</p></div>' for c in ORDER)
        examples.append(f'<details><summary><span>{esc(x["id"])}</span>{esc(x["prompt"])}</summary><div class="answers">{ans}</div></details>')
    gate_word="PASS" if gate["pass"] else "FAIL"
    gate_items=''.join(f'<li class="{str(v).lower()}">{esc(k.replace("_"," "))}: <b>{"yes" if v else "no"}</b></li>' for k,v in gate.items() if k!="pass")
    doc=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Ranke-1929 × Talkie slope</title><style>
:root{{--paper:#f3efe4;--ink:#18201d;--muted:#6c6a60;--red:#a53a2d;--green:#235c49;--rule:#c7c0ae;--card:#faf7ee}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:Georgia,'Times New Roman',serif}}main{{max-width:1180px;margin:auto;padding:55px 34px 90px}}.kicker,.score span,th{{font:700 12px/1.2 'Courier New',monospace;letter-spacing:.12em;text-transform:uppercase}}h1{{font-size:clamp(42px,7vw,82px);line-height:.9;margin:16px 0 22px;max-width:1000px;letter-spacing:-.045em}}.dek{{font-size:21px;line-height:1.5;max-width:850px;color:#343b37}}.rule{{height:5px;border-top:1px solid var(--ink);border-bottom:1px solid var(--ink);margin:38px 0}}.scores{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--ink);border:1px solid var(--ink)}}.score{{background:var(--card);padding:22px}}.score strong{{display:block;font-size:48px;line-height:1;margin:10px 0 6px;font-variant-numeric:tabular-nums}}.score small,.note{{color:var(--muted)}}section{{margin-top:56px}}h2{{font-size:32px;margin:0 0 10px}}.note{{max-width:850px;line-height:1.55}}.chart{{background:var(--card);border:1px solid var(--rule);padding:18px;margin-top:22px}}svg{{display:block;width:100%}}.grid{{stroke:#d9d2c1}}.slope{{stroke:var(--red);stroke-width:5;fill:none}}.dot{{fill:var(--card);stroke:var(--red);stroke-width:4}}svg text{{fill:var(--ink);font-family:'Courier New',monospace}}.tick{{font-size:12px}}.label{{font-size:14px;font-weight:bold}}.value{{font-size:17px;font-weight:bold}}table{{border-collapse:collapse;width:100%;margin-top:22px;background:#faf7eeb3}}th,td{{padding:11px 12px;border-bottom:1px solid var(--rule);text-align:left}}th{{border-top:2px solid var(--ink);border-bottom:1px solid var(--ink)}}.num{{text-align:right;font-variant-numeric:tabular-nums}}.gate{{display:grid;grid-template-columns:180px 1fr;gap:28px;padding:24px;border:2px solid var(--ink);background:var(--card)}}.gate strong{{font:700 42px/1 'Courier New',monospace;color:{'var(--green)' if gate['pass'] else 'var(--red)'}}}.gate ul{{margin:0;line-height:1.8}}.gate .false{{color:var(--red)}}details{{border-top:1px solid var(--ink)}}details:last-child{{border-bottom:1px solid var(--ink)}}summary{{padding:18px 8px;cursor:pointer;font-size:18px}}summary span{{font:700 11px 'Courier New',monospace;background:var(--ink);color:var(--paper);padding:4px 6px;margin-right:10px}}.answers{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;padding-bottom:18px}}.answer{{background:var(--card);border:1px solid var(--rule);padding:16px}}.answer header{{display:flex;justify-content:space-between;font:13px 'Courier New',monospace;border-bottom:1px solid var(--rule);padding-bottom:9px}}.answer p{{white-space:pre-wrap;line-height:1.55;margin-bottom:0}}.method{{columns:2;column-gap:44px;line-height:1.6}}.mono{{font:12px 'Courier New',monospace;overflow-wrap:anywhere}}footer{{border-top:2px solid var(--ink);margin-top:60px;padding-top:20px;color:var(--muted)}}@media(max-width:760px){{main{{padding:32px 18px}}.scores{{grid-template-columns:repeat(2,1fr)}}.answers{{grid-template-columns:1fr}}.method{{columns:1}}.gate{{grid-template-columns:1fr}}}}
</style></head><body><main><div class="kicker">Training experiment · 28 September 2026</div><h1>Does period Q+A training improve Ranke?</h1><p class="dek">Talkie-1930 wrote questions and answers from eight books published before 1913. Ranke-1929 v2 received nested sets of 300, 1,000, and 3,000 accepted examples. The same 84 prompts tested every condition.</p><div class="rule"></div><div class="scores">{cards}</div>
<section><h2>The slope</h2><p class="note">Blind mean score, 0–100. Each point adds training examples while the base model, prompt, decoding, and test stay fixed.</p><div class="chart">{line_chart(means)}</div><table><thead><tr><th>Step</th><th class="num">Mean change</th><th class="num">Paired 95% CI</th><th class="num">Rank wins / ties / losses</th></tr></thead><tbody>{step_rows}</tbody></table></section>
<section><h2>Predeclared slope gate: {gate_word}</h2><div class="gate"><strong>{gate_word}</strong><ul>{gate_items}</ul></div></section>
<section><h2>The two claims</h2><p class="note">Period fluency is the mean of coherence, naturalness, and period fit on all 84 prompts. History substance is the factual and analytical substance score on the 36 frozen history questions. These views do not change the primary score or the predeclared gate.</p><table><thead><tr><th>Claim</th>{heads}<th class="num">Base to 3,000 (95% CI)</th></tr></thead><tbody>{claim_rows}</tbody></table></section>
<section><h2>What changed?</h2><table><thead><tr><th>Measure</th>{heads}</tr></thead><tbody>{axis_rows}</tbody></table></section><section><h2>Two frozen test suites</h2><table><thead><tr><th>Suite</th>{heads}</tr></thead><tbody>{suite_rows}</tbody></table></section><section><h2>Results by task</h2><table><thead><tr><th>Task</th>{heads}<th class="num">3,000 − 1,000</th></tr></thead><tbody>{cat_rows}</tbody></table></section>
<section><h2>Protected Russia, Japan, and Angell slice</h2><p class="note">These {len(protected)} prompts were frozen before training. Candidate records containing the protected names and terms were rejected. A gain here tests transfer from response method, not rehearsal.</p><table><thead><tr><th>Condition</th><th class="num">Mean</th></tr></thead><tbody>{protected_rows}</tbody></table><p class="note">Base → 3,000: {pstep['mean_delta']:+.1f} points; paired 95% CI [{pstep['ci95'][0]:+.1f}, {pstep['ci95'][1]:+.1f}].</p></section>
<section><h2>Read the answers</h2><p class="note">Open a row to compare complete, unedited continuations. Scores used anonymous labels.</p>{''.join(examples)}</section><section><h2>Mechanical failures</h2><table><thead><tr><th>Condition</th><th class="num">Hit token cap</th><th class="num">Mean tokens</th><th class="num">Anachronism flags</th><th class="num">Fatal failures</th></tr></thead><tbody>{mech_rows}</tbody></table></section>
<section><h2>Method</h2><div class="method"><p><b>Writer.</b> <span class="mono">talkie-lm/talkie-1930-13b-it</span>, commit <span class="mono">8033675be6360ae0127fa75f941c12d52064f1dc</span>, wrote one question and one 90–220 word answer from each supplied passage.</p><p><b>Selection.</b> Luna accepted or rejected each whole pair in one pass. It scored source support, relevance, self-containment, period style, anachronism, and visible reasoning. It did not rewrite.</p><p><b>Student.</b> <span class="mono">uzh-echist-org/Ranke-4B-1929_v2</span>, commit <span class="mono">fb4cede16ed59e1249ebeba2b4e5b9ec2a560642</span>. QLoRA used answer-only loss. The training sets are nested.</p><p><b>Evaluation.</b> The frozen test has 36 broad history questions and 48 period-chat questions. GPT-6 Astra graded five axes under blind labels. The report recomputed overall scores and capped fatal failures.</p></div></section><footer><p>Battery SHA-256<br><span class="mono">{esc(a.battery_sha256.lower())}</span></p><p>Output matrix SHA-256<br><span class="mono">{esc(a.matrix_sha256.lower())}</span></p><p>Evidence: <span class="mono">summary.json · per_item.csv · per_reply.csv</span></p></footer></main></body></html>'''
    (out/"report.html").write_text(doc,encoding="utf-8")
    print(json.dumps({"report":str(out/"report.html"),"gate":gate,"means":means}))

if __name__ == "__main__": main()
