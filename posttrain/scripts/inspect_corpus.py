import json,collections,pathlib
p=pathlib.Path(r'G:\detritus\ranke_period_qa_scale_20260928\data\history_passages_v1.jsonl')
rows=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
for field in ['source_title','title','book','source_id','gutenberg_id','author']:
 c=collections.Counter(str(x.get(field)) for x in rows)
 if not (len(c)==1 and list(c)[0]=='None'):
  print(field,len(c)); print(*c.most_common(30),sep='\n')
print('KEYS',sorted(rows[0]))
