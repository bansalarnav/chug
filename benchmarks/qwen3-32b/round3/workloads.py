#!/usr/bin/env python3
"""Fixed-rate arrivals, real prompts, reuse, abort/recovery and context pressure."""
import argparse,concurrent.futures,hashlib,http.client,json,math,random,re,statistics,sys,threading,time
from pathlib import Path
from http.client import HTTPConnection
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
# Deployed beside the original benchmark clients on bly.
try:from bench import batch as raw_batch,request as raw_request
except ImportError:
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
 from bench import batch as raw_batch,request as raw_request
from functools import partial
batch=partial(raw_batch,complete_stream_timing=True)
request=partial(raw_request,complete_stream_timing=True)
SERVER_ENGINE=None
ENABLE_CACHE=False
MODEL_NAME="qwen3-32b"


def http(port,path,body=None,timeout=180):
 native=SERVER_ENGINE=='lmdeploy' and path=='/v1/completions'
 if native:
  path='/generate';body=dict(body);body['input_ids']=body.pop('prompt');body.pop('model',None)
 conn=HTTPConnection('127.0.0.1',port,timeout=timeout)
 try:
  conn.request('GET' if body is None else 'POST',path,None if body is None else json.dumps(body),{'Content-Type':'application/json'})
  response=conn.getresponse();text=response.read().decode()
  try:data=json.loads(text)
  except json.JSONDecodeError:data=text
  if native and response.status==200 and isinstance(data,dict) and 'meta_info' in data:
   data=dict(choices=[dict(text=data['text'])],usage={k:data['meta_info'][k] for k in ['prompt_tokens','completion_tokens']},native_response=data)
  return dict(status=response.status,data=data)
 finally:conn.close()


def quantile(xs,p):
 if not xs:return None
 xs=sorted(xs);i=(len(xs)-1)*p;a=math.floor(i);b=math.ceil(i)
 return xs[a]+(xs[b]-xs[a])*(i-a)


def metrics(port):
 result={}
 for path in ['/metrics','/slots']:
  try:
   r=http(port,path,timeout=5)
   if r['status']==200:
    result[path]=r['data'] if path=='/slots' else '\n'.join(line for line in str(r['data']).splitlines() if not line.startswith('#') and any(w in line.lower() for w in ['cache','running','waiting','request','token_usage','used','spec','draft','accept']))
  except Exception as error:result[path]={'error':str(error)}
 if SERVER_ENGINE=='sglang':
  try:result['/get_server_info']=http(port,'/get_server_info',timeout=5)
  except Exception as error:result['/get_server_info']={'error':str(error)}
 return result


def cold_cache(port,engine):
 if engine=='vllm':return http(port,'/reset_prefix_cache',{})
 if engine=='sglang':return http(port,'/flush_cache',{})
 if engine=='llama':
  slots=http(port,'/slots')
  if slots['status']==200 and isinstance(slots['data'],list):
   return [http(port,f'/slots/{s["id"]}?action=erase',{}) for s in slots['data']]
 return {'note':'No reset API used; each prefix sequence gets a unique leading nonce.'}


def completion(port,prompt,output=128,ignore_eos=False,logprobs=None):
 body=dict(model=MODEL_NAME,prompt=prompt,max_tokens=output,temperature=0,ignore_eos=ignore_eos,stream=False,skip_special_tokens=False)
 if SERVER_ENGINE=='llama':body['cache_prompt']=ENABLE_CACHE
 if logprobs is not None:body['logprobs']=logprobs
 start=time.perf_counter();r=http(port,'/v1/completions',body)
 if r['status']!=200:raise RuntimeError(r)
 usage=r['data'].get('usage',{})
 if usage.get('prompt_tokens')!=len(prompt) or not 0<=usage.get('completion_tokens',-1)<=output:raise RuntimeError('Natural completion token count mismatch: '+str(usage))
 r['elapsed_s']=time.perf_counter()-start
 return r


def abort(port,prompt,phase):
 conn=HTTPConnection('127.0.0.1',port,timeout=180)
 body=dict(model=MODEL_NAME,prompt=prompt,max_tokens=1024,temperature=0,ignore_eos=True,stream=True)
 start=time.perf_counter();chunks=0
 try:
  endpoint='/v1/completions'
  if SERVER_ENGINE=='lmdeploy':
   endpoint='/generate';body['input_ids']=body.pop('prompt');body.pop('model',None)
  conn.request('POST',endpoint,json.dumps(body),{'Content-Type':'application/json'})
  if phase=='prefill':
   # Close before consuming response headers; do not wait for prefill to finish.
   time.sleep(.05)
  else:
   response=conn.getresponse()
   if response.status!=200:raise RuntimeError(response.read().decode())
   while line:=response.readline():
    if line.startswith(b'data: ') and line[6:].strip()!=b'[DONE]':
     event=json.loads(line[6:])
     if event.get('text') or any(c.get('text') for c in event.get('choices',[])):
      chunks+=1
      if chunks>=4:break
 finally:conn.close()
 return dict(phase=phase,closed_after_s=time.perf_counter()-start,visible_chunks=chunks)


def main():
 p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18080);p.add_argument('--label',required=True)
 p.add_argument('--engine',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True)
 p.add_argument('--cache-enabled',action='store_true');p.add_argument('--model-name',default='qwen3-32b')
 p.add_argument('--tokenizer',type=Path,required=True);p.add_argument('--suite',choices=['all','speed','traffic','conversation','pressure','quality','reuse','thermal'],default='all')
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 global batch,request,SERVER_ENGINE,ENABLE_CACHE,MODEL_NAME
 SERVER_ENGINE=a.engine;ENABLE_CACHE=a.cache_enabled;MODEL_NAME=a.model_name
 fields={'cache_prompt':a.cache_enabled} if a.engine=='llama' else {}
 fields['model']=MODEL_NAME
 batch=partial(raw_batch,complete_stream_timing=True,request_fields=fields,protocol='lmdeploy' if a.engine=='lmdeploy' else 'openai')
 request=partial(raw_request,complete_stream_timing=True,request_fields=fields,protocol='lmdeploy' if a.engine=='lmdeploy' else 'openai')
 corpus=json.loads(a.corpus.read_text());raw=a.out/f'{a.label}-{a.suite}.jsonl'
 previous=[json.loads(l) for l in raw.read_text().splitlines()] if raw.exists() else [];done={r['case_id'] for r in previous}
 log=raw.open('a')
 def case(case_id,fn):
  if case_id in done:return
  start=time.time()
  try:result=fn();row=dict(case_id=case_id,label=a.label,wall_start=start,wall_end=time.time(),result=result)
  except Exception as error:row=dict(case_id=case_id,label=a.label,wall_start=start,wall_end=time.time(),error=repr(error))
  log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush();print(a.label,case_id,'ERROR '+row['error'] if 'error' in row else 'OK',flush=True)
 def selected(suite):return a.suite in ['all',suite]
 if selected('speed') or a.suite=='thermal':
  for length,c in [(128,1),(1024,1),(4096,1),(128,4),(128,16),(128,32),(1024,4),(1024,16)]:
   if a.suite=='thermal' and (length,c) not in [(1024,1),(128,4),(128,32)]:continue
   def run(length=length,c=c):
    batch(a.port,corpus[str(length)][:c],128)
    trials=[batch(a.port,corpus[str(length)][32+i*c:32+(i+1)*c],128) for i in range(3)]
    assert all(len(t['requests'])==c for t in trials)
    return dict(input_tokens=length,output_tokens=128,concurrency=c,trials=trials,
     median_output_tokens_per_s=statistics.median(t['aggregate_output_tokens_per_s'] for t in trials))
   case(f'speed-{length}-c{c}',run)
 if selected('traffic'):
  for rate in [1,2,4,8]:
   def run(rate=rate):
    # Independent Poisson arrivals for a fixed 20-second horizon, identical seed per engine.
    rng=random.Random(20261114);arrivals=[];at=0
    while True:
     at+=rng.expovariate(rate)
     if at>=20:break
     arrivals.append(at)
    batch(a.port,corpus['1024'][:4],128)
    start=time.perf_counter();rows=[]
    def submit(i,scheduled):
     began=time.perf_counter()
     try:
      r=request(a.port,corpus['1024'][(32+i)%128],128)
      r.update(client_dispatch_lag_s=began-start-scheduled,scheduled_after_s=scheduled,
       arrival_to_finish_s=time.perf_counter()-start-scheduled)
      return r
     except Exception as error:return dict(error=str(error),scheduled_after_s=scheduled,arrival_to_finish_s=time.perf_counter()-start-scheduled)
    with concurrent.futures.ThreadPoolExecutor(max_workers=256) as pool:
     futures=[]
     for i,scheduled in enumerate(arrivals):
      delay=start+scheduled-time.perf_counter()
      if delay>0:time.sleep(delay)
      futures.append(pool.submit(submit,i,scheduled))
     rows=[f.result() for f in futures]
    elapsed=time.perf_counter()-start
    ok=[r for r in rows if 'error' not in r]
    return dict(target_rate_rps=rate,arrival_horizon_s=20,requests=len(rows),completed=len(ok),elapsed_s=elapsed,
     output_tokens_per_s=len(ok)*128/elapsed,arrival_to_finish_p50_s=quantile([r['arrival_to_finish_s'] for r in ok],.5),
     arrival_to_finish_p95_s=quantile([r['arrival_to_finish_s'] for r in ok],.95),
     arrival_to_finish_p99_s=quantile([r['arrival_to_finish_s'] for r in ok],.99) if len(ok)>=100 else None,
     max_dispatch_lag_s=max((r['client_dispatch_lag_s'] for r in ok),default=None),rows=rows)
   case(f'poisson-{rate}rps',run)
  def burst():
   return dict(concurrency=64,input_tokens=128,output_tokens=128,metrics_before=metrics(a.port),batch=batch(a.port,corpus['128'][32:96],128),metrics_after=metrics(a.port))
  case('burst-64',burst)
 from transformers import AutoTokenizer
 tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True)
 def encode(messages):return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
 tasks=[('multiply','What is 17 times 23? Respond with just the integer.','391'),
 ('add','What is 127 plus 286? Respond with just the integer.','413'),
 ('subtract','What is 1000 minus 657? Respond with just the integer.','343'),
 ('sort','Sort these integers ascending: 8, 3, 12, 1. Respond with just comma-separated integers.','1,3,8,12'),
 ('reverse','Reverse the ASCII string abcdef. Respond with just the reversed string.','fedcba'),
 ('json','Output exactly this JSON object, with no other text: {"ok":true,"count":3}','{"ok":true,"count":3}'),
 ('extract','Record: user=alice; id=7319; role=reader. Return only the id.','7319'),
 ('remember','Facts: cedar=17, birch=29, oak=43. Return only the tree with value 29.','birch'),
 ('translate','Translate only the German word Hund to English. Return just the lowercase word.','dog'),
 ('python','What does Python print for sum([2, 3, 5])? Return only the integer.','10'),
 ('boolean','Is 97 an even integer? Reply with exactly yes or no.','no'),
 ('letter','Return only the third letter of the lowercase English alphabet.','c')]
 if selected('quality'):
  for name,text,expected in tasks:
   def run(text=text,expected=expected):
    ids=encode([{'role':'user','content':text}]);r=completion(a.port,ids,64)
    output=r['data']['choices'][0]['text'];normalized=re.sub(r'\s+','',output).lower()
    return dict(prompt_tokens=ids,expected=expected,response=r,exact_normalized_match=normalized==re.sub(r'\s+','',expected).lower())
   case('quality-'+name,run)
 if selected('conversation'):
  prompts=['Explain why a hash table can have collisions in two sentences.','Write a Python function that merges two sorted lists without using sorted().','Translate this into English: Der Server hat die Anfrage abgebrochen, aber der Speicher wurde wiederverwendet.','A queue receives six jobs per second and completes four. Explain what happens to latency.']
  for i,text in enumerate(prompts):
   def run(text=text):
    ids=encode([{'role':'user','content':text}]);return dict(prompt_tokens=ids,response=completion(a.port,ids,192))
   case(f'real-prompt-{i}',run)
  def multiturn():
   messages=[{'role':'system','content':'Answer concisely. The project codename is cedar. The retry budget is three attempts.'}];turns=[]
   for text in ['What is the project codename?','Write a short retry pseudocode loop using that budget.','Now explain when this loop should stop.','What were the codename and retry budget?']:
    messages.append({'role':'user','content':text});ids=encode(messages);r=completion(a.port,ids,192)
    answer=r['data']['choices'][0]['text'];turns.append(dict(input_tokens=len(ids),response=r));messages.append({'role':'assistant','content':answer})
   return dict(turns=turns,messages=messages)
  case('multi-turn',multiturn)
 if selected('reuse') or (a.suite=='conversation' and not a.cache_enabled):
  for shared in [256,1024,4096]:
   def run(shared=shared):
    reset=cold_cache(a.port,a.engine)
    # Distinct first token makes successive sizes/configs independent without cache APIs.
    prefix=[2000+shared//128]+corpus[str(shared if str(shared) in corpus else 1024)][0][:shared-1]
    before=metrics(a.port);cold=request(a.port,prefix+corpus['128'][60][:32],64)
    warm=[request(a.port,prefix+corpus['128'][61+j][:32],64) for j in range(3)]
    after=metrics(a.port)
    return dict(shared_tokens=len(prefix),suffix_tokens=32,reset=reset,cold=cold,warm=warm,metrics_before=before,metrics_after=after,
     median_warm_ttft_ms=1000*statistics.median(r['ttft_s'] for r in warm),cold_ttft_ms=1000*cold['ttft_s'])
   case(f'prefix-reuse-{shared}',run)
 if selected('pressure'):
  for phase in ['prefill','decode']:
   def run(phase=phase):
    before=metrics(a.port);aborts=[]
    for i in range(8):
     try:aborts.append(abort(a.port,corpus['4096' if phase=='prefill' else '128'][32+i],phase))
     except Exception as error:aborts.append(dict(error=str(error),phase=phase))
    time.sleep(2);after=metrics(a.port)
    try:recovery=batch(a.port,corpus['128'][60:64],64)
    except Exception as error:recovery=dict(error=str(error))
    return dict(before=before,aborts=aborts,after=after,recovery=recovery,after_recovery=metrics(a.port))
   case('abort-'+phase,run)
  def pressure():
   before=metrics(a.port)
   # A request beyond the declared context must reject rather than crash or silently truncate.
   oversized=http(a.port,'/v1/completions',dict(model=MODEL_NAME,prompt=corpus['8192'][0]*2,max_tokens=128,temperature=0))
   barrier=threading.Barrier(8);started=time.perf_counter()
   def stress_request(prompt):
    try:return request(a.port,prompt,128,barrier=barrier)
    except Exception as error:return dict(error=str(error))
   with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:stress_rows=list(pool.map(stress_request,corpus['4096'][32:40]))
   stress=dict(elapsed_s=time.perf_counter()-started,requests=stress_rows,completed=sum('error' not in r for r in stress_rows),failed=sum('error' in r for r in stress_rows))
   try:recovery=request(a.port,corpus['128'][0],64)
   except Exception as error:recovery=dict(error=str(error))
   return dict(before=before,oversized_request=oversized,expected_rejection=400<=oversized['status']<500,
    stress=stress,recovery=recovery,after=metrics(a.port))
  case('context-pressure',pressure)
 log.close()
 print('WORKLOADS COMPLETE',a.label,a.suite,flush=True)

if __name__=='__main__':main()
