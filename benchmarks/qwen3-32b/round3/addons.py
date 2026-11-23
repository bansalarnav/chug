#!/usr/bin/env python3
"""Quantized GGUF comparisons, validation perplexity, speculation, TRT probe."""
import fcntl,glob,json,os,shutil,signal,statistics,subprocess,sys,time
from pathlib import Path
from service import service
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from bench import request
root=Path(os.environ['BENCH_ROOT']);r3=root/'round3';out=r3/'results';scripts=r3/'scripts'
def interrupt(signum,frame):raise SystemExit(signum)
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
print('Waiting for the base serving runner lock',flush=True)
lock=(root/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
model=(root/'results/model-path.txt').read_text().strip();base_env=os.environ.copy()
base_env['PATH']=str(root/'envs/reference/bin')+os.pathsep+base_env['PATH']
cuda=root/'envs/sglang/lib/python3.12/site-packages/nvidia/cu13'
base_env['LD_LIBRARY_PATH']=':'.join([str(cuda/'lib'),str(root/'envs/sglang/lib/python3.12/site-packages/nvidia/nccl/lib')])

def stage(name,fn):
 marker=out/f'{name}-addon.json'
 if marker.exists() and json.loads(marker.read_text()).get('ok'):return
 print('ADDON START',name,flush=True);start=time.time()
 try:
  result=fn();record=dict(ok=True,elapsed_s=time.time()-start,result=result)
 except Exception as error:record=dict(ok=False,elapsed_s=time.time()-start,error=repr(error))
 marker.write_text(json.dumps(record,indent=2)+'\n');print('ADDON FINISH',name,record,flush=True)


def quantize(kind):
 path=r3/'models'/f'Qwen3-32B-{kind}.gguf';command=[str(r3/'sources/llama.cpp/build/bin/llama-quantize'),str(r3/'models/Qwen3-32B-BF16.gguf'),str(path),kind,'4']
 (out/f'quantize-{kind}-command.json').write_text(json.dumps(command,indent=2)+'\n')
 with (out/f'quantize-{kind}.log').open('w') as log:subprocess.run(command,env=base_env,stdout=log,stderr=subprocess.STDOUT,check=True)
 return dict(file=str(path),bytes=path.stat().st_size,quantization=kind)
def repair_turbomind():
 label='turbomind-bf16';archive=out/'attempts/turbomind-openai-input-ids';archive.mkdir(parents=True,exist_ok=True)
 for p in list(out.glob(label+'-*')):
  if p.is_file() and p.suffix in ['.jsonl','.log','.csv']:
   shutil.move(str(p),str(archive/p.name))
 command=json.loads((out/(label+'-command.json')).read_text());env=base_env.copy()
 env['PATH']=str(root/'envs/lmdeploy/bin')+os.pathsep+env['PATH']
 env['LD_LIBRARY_PATH']=':'.join(glob.glob(str(root/'envs/lmdeploy/lib/python3.12/site-packages/nvidia/*/lib')))
 suites=['speed','quality','conversation','traffic','pressure']
 with service(label,command,env,out,health='/v1/models'):
  for suite in suites:
   subprocess.run([str(root/'envs/lmdeploy/bin/python'),str(scripts/'workloads.py'),'--label',label,'--engine','lmdeploy','--suite',suite,'--out',str(out),'--corpus',str(root/'extended-results/corpus.json'),'--tokenizer',model],env=env,check=True)
  rows=[json.loads(line) for suite in suites for line in (out/f'{label}-{suite}.jsonl').read_text().splitlines()]
  assert len(rows)==36 and all('error' not in row for row in rows), 'Native TurboMind cases failed; inspect JSONL records'
 (out/f'{label}-complete.json').write_text(json.dumps(dict(finished_at=time.time(),suites=suites,protocol='native /generate'))+'\n')
 return dict(cases=len(rows),protocol='native /generate',same_token_ids=True)
stage('repair-turbomind',repair_turbomind)

# Prompt-lookup speculation requires no additional checkpoint and preserves BF16 target precision.
for speculative in [False,True]:
 def run_spec(speculative=speculative):
  label='vllm-ngram' if speculative else 'vllm-ngram-control';command=json.loads((out/'vllm-bf16-command.json').read_text())
  if speculative:command+=['--speculative-config',json.dumps(dict(method='ngram',prompt_lookup_min=2,prompt_lookup_max=5,num_speculative_tokens=4))]
  env=base_env.copy();env['PATH']=str(root/'envs/vllm/bin')+os.pathsep+env['PATH']
  with service(label,command,env,out):
   subprocess.run([str(root/'envs/reference/bin/python'),str(scripts/'spec_workload.py'),'--label',label,'--out',str(out),'--tokenizer',model],env=env,check=True)
   subprocess.run([str(root/'envs/reference/bin/python'),str(scripts/'workloads.py'),'--label',label,'--engine','vllm','--suite','quality','--out',str(out),'--corpus',str(root/'extended-results/corpus.json'),'--tokenizer',model],env=env,check=True)
  return dict(speculative=speculative,method='prompt-lookup ngram',target_precision='BF16')
 stage('ngram' if speculative else 'ngram-control',run_spec)

# TensorRT is a compatibility probe first; retain its actual failure if the installed backend cannot run.
def trt_probe():
 helptext=(out/'tensorrt-help.txt').read_text()
 if 'Traceback' in helptext or not helptext.strip():raise RuntimeError('TensorRT import/CLI unavailable; see tensorrt-import.txt and setup logs')
 command=[str(root/'envs/tensorrt/bin/trtllm-serve'),'serve',model,'--host','127.0.0.1','--port','18080','--tp_size','4','--max_batch_size','4','--max_num_tokens','2048','--max_seq_len','4096','--backend','pytorch']
 env=os.environ.copy();env['LD_LIBRARY_PATH']=(out/'tensorrt-library-path.txt').read_text().strip();env['CUDA_HOME']=str(cuda);env['PATH']=str(cuda/'bin')+os.pathsep+str(root/'envs/tensorrt/bin')+os.pathsep+env['PATH']
 # A different API alias can be reported by the server; this first probe targets its native model name.
 with service('tensorrt-bf16-tp4-b4',command,env,out,health='/health'):
  import urllib.request
  data=json.loads(urllib.request.urlopen('http://127.0.0.1:18080/v1/models').read());served=data['data'][0]['id']
  request_body=json.dumps(dict(model=served,prompt='What is 2 + 2? Answer only with the number.',max_tokens=32,temperature=0)).encode()
  r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18080/v1/completions',data=request_body,headers={'Content-Type':'application/json'}),timeout=300)
  result=json.load(r);(out/'tensorrt-smoke.json').write_text(json.dumps(result,indent=2)+'\n')
  from bench import batch
  corpus=json.loads((root/'extended-results/corpus.json').read_text());label='tensorrt-bf16-tp4-b4';raw=out/f'{label}-speed.jsonl'
  with raw.open('w') as log:
   for length,c in [(128,1),(1024,1),(128,4),(1024,4)]:
    batch(18080,corpus[str(length)][:c],128,complete_stream_timing=True,request_fields={'model':served})
    started=time.time()
    trials=[batch(18080,corpus[str(length)][32+i*c:32+(i+1)*c],128,complete_stream_timing=True,request_fields={'model':served}) for i in range(3)]
    record=dict(label=label,case_id=f'speed-{length}-c{c}',wall_start=started,wall_end=time.time(),result=dict(input_tokens=length,output_tokens=128,concurrency=c,trials=trials,median_output_tokens_per_s=statistics.median(t['aggregate_output_tokens_per_s'] for t in trials)))
    log.write(json.dumps(record)+'\n');log.flush()
  subprocess.run([str(root/'envs/reference/bin/python'),str(scripts/'workloads.py'),'--label',label,'--engine','tensorrt','--suite','quality','--model-name',served,'--out',str(out),'--corpus',str(root/'extended-results/corpus.json'),'--tokenizer',model],env=env,check=True)
 return dict(response=result,speed_cases=4,max_batch_size=4,max_seq_len=4096,precision='BF16 checkpoint, auto dtype',note='Four-slot compatibility configuration; not a tuned 32-slot result')
stage('tensorrt-probe',trt_probe)


def llama_cache():
 command=json.loads((out/'llama-bf16-tensor-command.json').read_text());command[command.index('--cache-ram')+1]='2048'
 label='llama-cache';env=base_env.copy()
 with service(label,command,env,out):
  for suite in ['reuse','conversation']:
   subprocess.run([str(root/'envs/reference/bin/python'),str(scripts/'workloads.py'),'--label',label,'--engine','llama','--suite',suite,'--cache-enabled','--out',str(out),'--corpus',str(root/'extended-results/corpus.json'),'--tokenizer',model],env=env,check=True)
 return dict(suites=['reuse','conversation'],shared_kv_tokens=24576,cache_enabled=True)
stage('serve-llama-cache',llama_cache)


def llama_pressure_recovery():
 command=json.loads((out/'llama-bf16-tensor-command.json').read_text())+['--log-verbosity','4']
 label='llama-pressure-recovery';env=base_env.copy()
 with service(label,command,env,out):
  subprocess.run([str(root/'envs/reference/bin/python'),str(scripts/'workloads.py'),'--label',label,'--engine','llama','--suite','pressure','--out',str(out),'--corpus',str(root/'extended-results/corpus.json'),'--tokenizer',model],env=env,check=True)
 return dict(note='Records each constrained-KV failure and always attempts recovery; verbose startup retains actual layer placement.')
stage('llama-pressure-recovery',llama_pressure_recovery)

def llama_capacity():
 command=json.loads((out/'llama-bf16-tensor-command.json').read_text())+['--fit','off','--log-verbosity','4']
 command[command.index('--ctx-size')+1]='40960';label='llama-bf16-tensor-40k';env=base_env.copy()
 with service(label,command,env,out):
  for suite in ['traffic','pressure']:
   subprocess.run([str(root/'envs/reference/bin/python'),str(scripts/'workloads.py'),'--label',label,'--engine','llama','--suite',suite,'--out',str(out),'--corpus',str(root/'extended-results/corpus.json'),'--tokenizer',model],env=env,check=True)
 return dict(shared_kv_tokens=40960,request_slots=32,per_slot_limit=8192,precision='BF16 weights and KV',suites=['traffic','pressure'])
stage('llama-capacity-40k',llama_capacity)

stage('kv-recheck',lambda:subprocess.run([str(root/'envs/vllm/bin/python'),str(scripts/'kv_recheck.py')],check=True).returncode)
for kind in ['Q4_K_M','Q8_0']:stage('quantize-'+kind,lambda kind=kind:quantize(kind))

for kind in ['BF16','Q8_0','Q4_K_M']:
 def evaluate(kind=kind):
  command=[str(r3/'sources/llama.cpp/build/bin/llama-perplexity'),'-m',str(r3/'models'/f'Qwen3-32B-{kind}.gguf'),'-f',str(r3/'fixtures/wikitext-validation.txt'),'-c','512','--chunks','4','--n-gpu-layers','99','--split-mode','tensor','--tensor-split','1,1,1,1','--flash-attn','on','--cache-type-k','bf16','--cache-type-v','bf16','--batch-size','512','--ubatch-size','512']
  (out/f'perplexity-{kind}-command.json').write_text(json.dumps(command,indent=2)+'\n')
  with (out/f'perplexity-{kind}.log').open('w') as log:subprocess.run(command,env=base_env,stdout=log,stderr=subprocess.STDOUT,check=True)
  return dict(log=f'perplexity-{kind}.log',context=512,chunks=4)
 stage('perplexity-'+kind,evaluate)

for kind in ['Q8_0','Q4_K_M','Q4_K_M-single']:
 def run_quant(kind=kind):
  label='llama-'+kind.lower();base=json.loads((out/'llama-bf16-tensor-command.json').read_text())
  base+=['--fit','off']
  base[base.index('-m')+1]=str(r3/'models'/f'Qwen3-32B-{kind.replace("-single","")}.gguf')
  env=base_env.copy()
  if kind.endswith('-single'):
   base[base.index('--batch-size')+1]='512';base[base.index('--ubatch-size')+1]='128'
   env['CUDA_VISIBLE_DEVICES']='0';base[base.index('--split-mode')+1]='none';base[base.index('--tensor-split')+1]='1';base[base.index('--ctx-size')+1]='2048';base[base.index('--kv-unified-per-slot')+1]='2048';base[base.index('--parallel')+1]='4'
  with service(label,base,env,out):
   suites=['thermal','quality'] if kind.endswith('-single') else ['speed','quality','conversation']
   for suite in suites:
    subprocess.run([str(root/'envs/reference/bin/python'),str(scripts/'workloads.py'),'--label',label,'--engine','llama','--suite',suite,'--out',str(out),'--corpus',str(root/'extended-results/corpus.json'),'--tokenizer',model],env=env,check=True)
  return dict(suites=suites,precision=kind,visible_devices=env.get('CUDA_VISIBLE_DEVICES'))
 stage('serve-'+kind,run_quant)

print('ADDONS FINISHED',flush=True)
