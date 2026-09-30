import json,hashlib,pathlib
files=[r'G:\detritus\period_qa_eval_frozen_20260927\battery.jsonl',r'G:\detritus\period_qa_eval_frozen_v2_20260927\battery.jsonl',r'G:\detritus\ranke_period_qa_run1_20260927\preflight_per_item_caps\battery.jsonl',r'G:\detritus\ranke_period_qa_scale_20260928\eval\history_reasoning_v1.jsonl']
terms=('angell','russo','japan','russia','tsushima','port arthur','manchuria','portsmouth')
for f in files:
 p=pathlib.Path(f); rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
 print('\nFILE',f); print('ROWS',len(rows),'SHA',hashlib.sha256(p.read_bytes()).hexdigest()); print('CATS', {c:sum(1 for x in rows if x.get('category')==c) for c in sorted({x.get('category') for x in rows})})
 for x in rows:
  q=x.get('prompt','')
  if any(t in q.lower() for t in terms): print(x['id'],x.get('category'),q)
