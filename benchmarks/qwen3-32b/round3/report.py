#!/usr/bin/env python3
"""Summarize retained measurements without treating attempted cases as successes."""
import ast,csv,datetime,hashlib,json,re,statistics
from zoneinfo import ZoneInfo
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if not (ROOT/'results').is_dir() and (ROOT.parent/'results').is_dir():ROOT=ROOT.parent
OUT=ROOT/'results'

def read(path,default=None):
 return json.loads(path.read_text()) if path.exists() else default

def write_csv(name,rows):
 if not rows:return
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)

def median(xs):return statistics.median(xs) if xs else None

def requests(node):
 if isinstance(node,dict):
  if all(k in node for k in ['input_tokens','output_tokens','usage','e2e_s']):yield node
  else:
   for v in node.values():yield from requests(v)
 elif isinstance(node,list):
  for v in node:yield from requests(v)

def natural_requests(node):
 if isinstance(node,dict):
  if isinstance(node.get('response'),dict) and isinstance(node.get('prompt_tokens'),list):
   yield dict(expected_prompt_tokens=len(node['prompt_tokens']),usage=node['response'].get('data',{}).get('usage',{}))
  elif isinstance(node.get('response'),dict) and isinstance(node.get('input_tokens'),int):
   yield dict(expected_prompt_tokens=node['input_tokens'],usage=node['response'].get('data',{}).get('usage',{}))
  for v in node.values():yield from natural_requests(v)
 elif isinstance(node,list):
  for v in node:yield from natural_requests(v)

def fmt(x):
 if x is None:return ''
 if isinstance(x,float):return f'{x:.4g}' if abs(x)<1 else f'{x:.2f}'
 return str(x)

def table(rows,keys):
 return '\n'.join(['| '+' | '.join(keys)+' |','|'+'|'.join('---' for _ in keys)+'|']+['| '+' | '.join(fmt(r.get(k,'')) for k in keys)+' |' for r in rows])


def main():
 rows=[];extra_stream_rows=[]
 for p in sorted(OUT.glob('*.jsonl')):
  for line in p.read_text().splitlines():
   if not line.strip():continue
   row=json.loads(line)
   if 'case_id' in row:row['source']=p.name;rows.append(row)
   elif p.name.endswith('-spec.jsonl'):extra_stream_rows.append(row)
 errors=[dict(label=r['label'],case=r['case_id'],error=r['error'],source=r['source']) for r in rows if 'error' in r]

 request_failures=[]
 for row in rows:
  r=row.get('result',{})
  for phase,items in [('arrivals',r.get('rows',[])),('stress',r.get('stress',{}).get('requests',[])),('abort',r.get('aborts',[]))]:
   for item in items:
    if 'error' in item:request_failures.append(dict(label=row['label'],case=row['case_id'],phase=phase,error=item['error']))
  if 'error' in r.get('recovery',{}):request_failures.append(dict(label=row['label'],case=row['case_id'],phase='recovery',error=r['recovery']['error']))
 write_csv('request-failures.csv',request_failures)
 streams=[r for row in rows+extra_stream_rows for r in requests(row.get('result',{}))]
 invalid=[r for r in streams if r['usage'].get('prompt_tokens')!=r['input_tokens'] or r['usage'].get('completion_tokens')!=r['output_tokens']]
 naturals=[r for row in rows for r in natural_requests(row.get('result',{}))]
 invalid_naturals=[r for r in naturals if r['usage'].get('prompt_tokens')!=r['expected_prompt_tokens']]
 speed=[];traffic=[];reuse=[];quality=[];pressure=[];spec=[]
 for row in rows:
  if 'result' not in row:continue
  r=row['result'];base=dict(label=row['label'],case=row['case_id'])
  if row['case_id'].startswith('speed-'):
   rr=[x for t in r['trials'] for x in t['requests']]
   speed.append(dict(**base,input_tokens=r['input_tokens'],output_tokens=r['output_tokens'],concurrency=r['concurrency'],output_tok_s=r['median_output_tokens_per_s'],decode_tok_s=median([x['decode_tokens_per_s'] for x in rr]),ttft_ms=1000*median([x['ttft_s'] for x in rr]),e2e_s=median([x['e2e_s'] for x in rr]),trial_min_tok_s=min(t['aggregate_output_tokens_per_s'] for t in r['trials']),trial_max_tok_s=max(t['aggregate_output_tokens_per_s'] for t in r['trials']),requests=len(rr)))
  elif row['case_id'].startswith('poisson-'):
   traffic.append(dict(**base,**{k:r[k] for k in ['target_rate_rps','arrival_horizon_s','requests','completed','elapsed_s','output_tokens_per_s','arrival_to_finish_p50_s','arrival_to_finish_p95_s','arrival_to_finish_p99_s','max_dispatch_lag_s']}))
  elif row['case_id'].startswith('prefix-reuse-'):
   reuse.append(dict(**base,shared_tokens=r['shared_tokens'],cold_ttft_ms=r['cold_ttft_ms'],warm_ttft_ms=r['median_warm_ttft_ms']))
  elif row['case_id'].startswith('quality-'):
   quality.append(dict(**base,passed=r['exact_normalized_match'],expected=r['expected'],response=r['response']['data']['choices'][0]['text']))
  elif row['case_id'].startswith('abort-') or row['case_id']=='context-pressure':
   pressure.append(dict(**base,expected_rejection=r.get('expected_rejection'),aborts=len(r.get('aborts',[])),recovery_ok=bool(r.get('recovery')) and 'error' not in r.get('recovery',{}),stress_completed=r.get('stress',{}).get('completed'),stress_failed=r.get('stress',{}).get('failed'),recovery_e2e_s=r.get('recovery',{}).get('e2e_s'),recovery_tok_s=r.get('recovery',{}).get('aggregate_output_tokens_per_s')))
 for name,data in [('speed',speed),('traffic',traffic),('reuse',reuse),('quality',quality),('pressure',pressure),('errors',errors)]:write_csv(name+'.csv',data)
 ref=read(ROOT/'fixtures/reference.json',{})
 checks=[c for f in ref.get('fixtures',[]) for c in f['checks']]
 fixtures=[]
 for f in ref.get('fixtures',[]):
  path=ROOT/'fixtures'/f['file'];fixtures.append(dict(name=f['name'],npz_sha256_ok=path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==f['sha256']))
 binaries=[]
 for p in (ROOT/'fixtures').glob('*/manifest.json'):
  data=read(p);arrays=data.get('arrays',data)
  if isinstance(arrays,dict):items=arrays.items()
  else:items=((x.get('name',''),x) for x in arrays)
  for name,a in items:
   if not isinstance(a,dict) or 'sha256' not in a:continue
   path=p.parent/a.get('file',name+'.bin');binaries.append(dict(file=str(path.relative_to(ROOT)),ok=path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==a['sha256']))
 kernels=read(OUT/'kernels.json',{}).get('rows',[]);layers=read(OUT/'layer.json',{}).get('rows',[])
 kernel_errors=[r for r in kernels if 'error' in r]
 write_csv('kernels.csv',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in kernels])
 write_csv('layer.csv',[{**{k:v for k,v in r.items() if not isinstance(v,(list,dict))},**{'comparison_'+k:v for k,v in r['comparison_to_unsharded'].items()}} for r in layers])
 numerical=dict(checks=len(checks),greedy_equal=sum(c['greedy_equal'] for c in checks),max_relative_rmse=max((c['relative_rmse'] for c in checks),default=None),max_centered_relative_rmse=max((c.get('centered_relative_rmse',0) for c in checks),default=None),max_kl_nats=max((c.get('kl_full_to_cached_nats',0) for c in checks),default=None),min_top10_overlap=min((c.get('top10_overlap',0) for c in checks),default=None),npz=fixtures,binaries=binaries)

 expected_jobs=[]
 for node in ast.walk(ast.parse(((ROOT/'run.py') if (ROOT/'run.py').exists() else ROOT/'scripts/run.py').read_text())):
  if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='jobs' for t in node.targets):expected_jobs=[j[0] for j in ast.literal_eval(node.value)]
 completed_jobs=[p.name.replace('-complete.json','') for p in OUT.glob('*-complete.json')]
 expected_addons=['repair-turbomind','serve-llama-cache','llama-pressure-recovery','llama-capacity-40k','kv-recheck','quantize-Q4_K_M','quantize-Q8_0','perplexity-BF16','perplexity-Q8_0','perplexity-Q4_K_M','serve-Q8_0','serve-Q4_K_M','serve-Q4_K_M-single','ngram-control','ngram','tensorrt-probe','tensorrt-mpirun-probe']
 failures=[dict(label=p.name.replace('-failure.json',''),**read(p)) for p in OUT.glob('*-failure.json')]
 addons=[dict(stage=p.name.replace('-addon.json',''),**read(p)) for p in OUT.glob('*-addon.json')]
 ppl=[]
 for p in OUT.glob('perplexity-*.log'):
  text=p.read_text(errors='replace');values=re.findall(r'Final estimate: PPL = ([0-9.]+)',text)
  if values:
   uncertainty=re.findall(r'Final estimate: PPL = [0-9.]+ \+/- ([0-9.]+)',text)
   ppl.append(dict(precision=p.stem.replace('perplexity-',''),perplexity=float(values[-1]),reported_plus_minus=float(uncertainty[-1]) if uncertainty else None))
 write_csv('perplexity.csv',ppl)

 spec_rows=[]
 for p in sorted(OUT.glob('*-spec.jsonl')):
  spec_rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
 control={r['case']+str(r['trial']):r['result']['output_sha256'] for r in spec_rows if r['label']=='vllm-ngram-control'}
 for label,case in sorted({(r['label'],r['case']) for r in spec_rows}):
  selected=[r for r in spec_rows if r['label']==label and r['case']==case]
  spec.append(dict(label=label,case=case,trials=len(selected),decode_tok_s=median([r['result']['decode_tokens_per_s'] for r in selected]),e2e_s=median([r['result']['e2e_s'] for r in selected]),matching_control_hashes=sum(control.get(case+str(r['trial']))==r['result']['output_sha256'] for r in selected)))

 for row in spec:
  path=OUT/f"{row['label']}-{row['case']}-metrics.json"
  if not path.exists():continue
  counters=read(path)
  def counter(phase,name):
   return sum(float(line.rsplit(' ',1)[-1]) for line in counters[phase].get('/metrics','').splitlines() if line.startswith(name+'{'))
  draft=counter('after','vllm:spec_decode_num_draft_tokens_total')-counter('before','vllm:spec_decode_num_draft_tokens_total')
  accepted=counter('after','vllm:spec_decode_num_accepted_tokens_total')-counter('before','vllm:spec_decode_num_accepted_tokens_total')
  row.update(draft_tokens=draft,accepted_tokens=accepted,draft_acceptance_fraction=accepted/draft if draft else None)
 write_csv('speculation.csv',spec)

 quality_summary=[dict(label=l,passed=sum(r['passed'] for r in quality if r['label']==l),tasks=sum(r['label']==l for r in quality)) for l in sorted({r['label'] for r in quality})]
 gpu=[];gpu_cases=[]
 timezone=ZoneInfo((OUT/'timezone.txt').read_text().strip()) if (OUT/'timezone.txt').exists() else None
 for p in OUT.glob('*-gpu.csv'):
  samples=list(csv.DictReader(p.open()))
  prior=OUT/'attempts/storage-pause'/p.name
  if prior.exists():samples+=list(csv.DictReader(prior.open()))
  ts=[float(r[' temperature.gpu'].strip()) for r in samples if ' temperature.gpu' in r and r[' temperature.gpu'].strip().isdigit()]
  thermal=[r for r in samples if any(v.strip()=='Active' for k,v in r.items() if k and 'thermal_slowdown' in k and v)]
  gpu.append(dict(label=p.name.replace('-gpu.csv',''),samples=len(samples),max_temperature_c=max(ts,default=None),thermal_slowdown_samples=len(thermal)))

  if timezone:
   normalized=[{k.strip():v.strip() for k,v in row.items() if k and v} for row in samples]
   timed=[]
   for sample in normalized:
    try:ts=datetime.datetime.strptime(sample['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone).timestamp()
    except (KeyError,ValueError):continue
    timed.append((ts,sample))
   label=p.name.replace('-gpu.csv','')
   for row in rows:
    if row['label']!=label:continue
    active=[sample for ts,sample in timed if row['wall_start']<=ts<=row['wall_end']]
    if not active:continue
    temps=[float(s['temperature.gpu']) for s in active if s.get('temperature.gpu','').isdigit()]
    clocks=[float(s['clocks.current.sm [MHz]'].split()[0]) for s in active if s.get('clocks.current.sm [MHz]','').split() and s.get('clocks.current.sm [MHz]','').split()[0].isdigit()]
    throttled=sum(any(v=='Active' for k,v in s.items() if 'thermal_slowdown' in k) for s in active)
    gpu_cases.append(dict(label=label,case=row['case_id'],samples=len(active),max_temperature_c=max(temps,default=None),median_sm_clock_mhz=median(clocks),thermal_slowdown_samples=throttled))
 write_csv('thermal-by-case.csv',gpu_cases)
 write_csv('thermal.csv',gpu)

 offload=[]
 for p in OUT.glob('llama-*-server.log'):
  log=p.read_text(errors='replace');m=re.findall(r'offloaded (\d+)/(\d+) layers to GPU',log)
  command=read(OUT/(p.name.replace('-server.log','')+'-command.json'),[])
  requested=command[command.index('--n-gpu-layers')+1] if '--n-gpu-layers' in command else None
  offload.append(dict(requested_gpu_layers=requested,label=p.name.replace('-server.log',''),offloaded_layers=int(m[-1][0]) if m else None,total_layers=int(m[-1][1]) if m else None,all_layers_on_gpu=(m[-1][0]==m[-1][1]) if m else None,fit_messages=' | '.join(line for line in log.splitlines() if 'fit' in line.lower() and any(w in line.lower() for w in ['gpu','offload','layer']))[:1500]))
 write_csv('offload.csv',offload)

 result=dict(cases=len(rows),speculative_trials=len(extra_stream_rows),cases_with_result=len(rows)-len(errors),case_errors=errors,request_failures=request_failures,validated_streams=len(streams),invalid_streams=len(invalid),validated_natural_prompts=len(naturals),invalid_natural_prompts=invalid_naturals,numerical=numerical,kernel_cases=len(kernels),kernel_errors=kernel_errors,layer_cases=len(layers),max_layer_relative_rmse=max((r['comparison_to_unsharded']['relative_rmse'] for r in layers),default=None),offload=offload,engine_failures=failures,addons=addons,quality=quality_summary,perplexity=ppl,speculation=spec,complete_base_jobs=completed_jobs,pending_base_jobs=[j for j in expected_jobs if j not in completed_jobs and j not in [f['label'] for f in failures]],pending_addons=[j for j in expected_addons if j not in [a['stage'] for a in addons]])
 (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
 text=['# Qwen3-32B additional baselines on bly','', 'Read [findings](FINDINGS.md) for interpretation and [reproduction instructions](REPRODUCE.md) for a new allocated storage path.', '',f'This snapshot contains {len(rows)} workload cases, {len(errors)} case errors, {len(request_failures)} individual request errors and {len(streams)} validated fixed-token streams. Read [methodology](METHODOLOGY.md) before comparing configurations. Engine failures and unfinished stages appear below.','', '![Serving comparison](results/serving-comparison.png)','', '## Serving speed','', 'Median of three warm measured batches; output throughput includes prefill and HTTP time. Decode speed uses first visible text through stream completion. All weights BF16 unless the label names Q8/Q4. llama.cpp tensor/layer/row settings and KV pool sizes differ; consult launch commands and offload logs.','',table(speed,['label','input_tokens','concurrency','output_tok_s','decode_tok_s','ttft_ms']), '', '## GPU placement','',table(offload,['label','requested_gpu_layers','offloaded_layers','total_layers','all_layers_on_gpu']),'', 'Blank actual-layer counts mean default verbosity suppressed backend placement messages. Explicit GPU-layer requests and the pinned fit-code behavior are documented in the methodology; the pressure diagnostic retains verbose placement logs.', '', '## Fixed-rate arrivals','', 'Twenty-second seeded Poisson arrivals, 1K input and 128 output tokens; requests drain after arrivals stop. P99 is omitted below 100 successful requests.','','![Arrival latency and completion fraction](results/arrival-latency.png)','',table(traffic,['label','target_rate_rps','completed','requests','arrival_to_finish_p50_s','arrival_to_finish_p95_s','arrival_to_finish_p99_s']), '', '## Prefix reuse','', 'One cold prefix and three warm requests with distinct suffixes. Compare the cache-off BF16 label with the corresponding cache label.','','![Prefix reuse](results/prefix-reuse.png)','',table(reuse,['label','shared_tokens','cold_ttft_ms','warm_ttft_ms']), '', '## Quality and numerical fixtures','',table(quality_summary,['label','passed','tasks']), '', f"These twelve exact-answer tasks are sanity checks. The reference fixture has {len(checks)} cached/full-prefix steps, {sum(c['greedy_equal'] for c in checks)} identical greedy decisions, maximum centered relative RMS {fmt(numerical['max_centered_relative_rmse'])}, maximum KL {fmt(numerical['max_kl_nats'])} nats. Raw arrays and hashes are included. There is no universal logit tolerance asserted.",'',f'{len(kernels)} component shapes and {len(layers)} actual TP4 transformer-block shapes were measured. The largest block relative RMS error was {fmt(result["max_layer_relative_rmse"])}. These unfused block timings exclude host serving and are separate implementation references.','','![Weight quantization](results/quantization.png)','',table(ppl,['precision','perplexity','reported_plus_minus']), '', '## Prompt-lookup speculation', '', table(spec,['label','case','decode_tok_s','e2e_s','matching_control_hashes','trials']), '', 'Output hashes compare each speculative trial with its BF16 control. Different hashes require inspection and do not establish a distributional correctness failure. Acceptance counters remain in the per-case metrics files.', '', '## Failures and stage status','', 'Pending base jobs: '+(', '.join(result['pending_base_jobs']) or 'none')+'. Pending addon stages: '+(', '.join(result['pending_addons']) or 'none')+'.', '',table(errors,['label','case','error']), '',table(failures,['label','error']), '',table([dict(stage=r['stage'],ok=r['ok'],error=r.get('error','')) for r in addons],['stage','ok','error']), '', '## Component timings','', '![GEMM reference](results/gemm-reference.png)','', 'See [kernel CSV](results/kernels.csv) and [actual block CSV](results/layer.csv) for exact shapes, timings and numerical comparisons.', '', '## Thermal observations','', '![Controlled repeats](results/thermal-repeat.png)','', 'Each suite starts below 50 C. One-Hz logs cover loading and requests; counts below include both. A-B-B-A repetitions use fresh launches. Cooling does not prevent throttling during long workloads.','',table(gpu,['label','max_temperature_c','thermal_slowdown_samples']), '', 'CSV files retain the measured summaries; JSONL files contain raw requests, outputs, cache metrics and recovery observations. Exact commands, startup logs, GPU traces, versions and numerical fixture hashes are retained beside them.']
 cleanup=read(OUT/'cleanup.json')
 if cleanup:
  text += ['', '## Storage cleanup', '', 'The original benchmark tree was deleted after its result archive and numerical hashes were verified. See [cleanup record](results/cleanup.json) for the exact removed path, archive hash, owner checks and allocated bytes. Results remain in this checkout and the user home archive.']
 (ROOT/'REPORT.md').write_text('\n'.join(text)+'\n')
 print(json.dumps({k:result[k] for k in ['cases','cases_with_result','validated_streams','invalid_streams','kernel_cases','layer_cases']}))

if __name__=='__main__':main()
