#!/usr/bin/env python3
import argparse,json,statistics,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from bench import request
from transformers import AutoTokenizer
from workloads import metrics
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--tokenizer',required=True);a=p.parse_args()
t=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
table='\n'.join(f'item_{i:03d}: status=ready; owner=worker_{i%4}; retries=3' for i in range(64))
texts=[('copy','Copy the following records exactly, with no changes or introduction:\n'+table),
 ('code','Write a Python module implementing an LRU cache using a dictionary and a doubly linked list. Include methods get, put, and eviction.'),
 ('prose','Explain how a processor cache works, including locality, cache lines, misses, and why sequential reads are faster than random reads.')]
raw=a.out/f'{a.label}-spec.jsonl'
if raw.exists():raise RuntimeError('Refusing to overwrite '+str(raw))
with raw.open('w') as f:
 for name,text in texts:
  ids=t.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
  request(18080,ids,512,complete_stream_timing=True)
  trials=[];before=metrics(18080)
  for repeat in range(3):
   start=time.time();result=request(18080,ids,512,complete_stream_timing=True)
   row=dict(label=a.label,case=name,prompt_ids=ids,trial=repeat,wall_start=start,wall_end=time.time(),result=result)
   f.write(json.dumps(row)+'\n');f.flush();trials.append(result)
  print(a.label,name,'median decode tok/s',statistics.median(r['decode_tokens_per_s'] for r in trials),flush=True)
  (a.out/f'{a.label}-{name}-metrics.json').write_text(json.dumps(dict(before=before,after=metrics(18080)),indent=2)+'\n')
