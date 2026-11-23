#!/usr/bin/env python3
"""Sequential round-three server runner. Raw commands and all failures persist."""
import atexit,datetime,fcntl,glob,json,os,signal,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(os.environ['BENCH_ROOT']);r3=root/'round3';results=r3/'results';scripts=r3/'scripts'
lock=(root/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
model=(root/'results/model-path.txt').read_text().strip()

def status(s):
 text=datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+s
 print(text,flush=True);(results/'status.txt').write_text(text+'\n')

def stop(p):
 if p is None:return
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:p.wait(timeout=40)
 except subprocess.TimeoutExpired:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  p.wait(timeout=10)

def interrupt(signum,frame):raise SystemExit(signum)
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)

def thermal_start(label):
 deadline=time.monotonic()+600;samples=[]
 while True:
  values=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True)
  temps=[int(v) for v in values.split()];samples.append(dict(time=time.time(),temperatures_c=temps))
  if max(temps)<=50:break
  if time.monotonic()>deadline:raise RuntimeError('GPUs did not cool below 50 C in ten minutes')
  time.sleep(5)
 (results/f'{label}-thermal-start.json').write_text(json.dumps(samples,indent=2)+'\n')

# Wait for the independent setup/correctness stage to release the GPUs.
def resume_conversion():
 paused=results/'paused-conversion.json'
 if paused.exists():
  pid=json.loads(paused.read_text())['pid']
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
atexit.register(resume_conversion)
status('WAIT numerical fixtures, component benchmarks and llama build')
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
 components=(results/'component-status.txt').read_text() if (results/'component-status.txt').exists() else ''
 build=r3/'sources/llama.cpp/build/bin/llama-server'
 if 'reference exit' in components and (results/'llama-help.txt').exists():break
 time.sleep(5)
else:raise TimeoutError('Setup/components exceeded two hours')
resume_conversion()
status('WAIT checkpoint conversion to finish')
deadline=time.monotonic()+3600
while time.monotonic()<deadline:
 log=(results/'convert-llama.log').read_text(errors='replace')
 if 'Model successfully exported' in log:break
 if 'Traceback' in log:raise RuntimeError('Checkpoint conversion failed; see convert-llama.log')
 time.sleep(5)
else:raise TimeoutError('Checkpoint conversion did not finish')
status('Setup stages finished; beginning server tests')

jobs=[('vllm-bf16','vllm',False,['speed','quality','conversation','traffic','pressure']),
      ('sglang-bf16','sglang',False,['speed','quality','conversation','traffic','pressure']),
      ('turbomind-bf16','lmdeploy',False,['speed','quality','conversation','traffic','pressure']),
      ('llama-bf16-tensor','llama',False,['speed','quality','conversation','traffic','pressure']),
      ('llama-bf16-layer','llama',False,['speed']),
      ('llama-bf16-row','llama',False,['speed']),
      ('vllm-cache','vllm',True,['reuse','conversation']),
      ('sglang-cache','sglang',True,['reuse','conversation']),
      ('turbomind-cache','lmdeploy',True,['reuse','conversation']),
      ('vllm-thermal-A1','vllm',False,['thermal']),
      ('sglang-thermal-B1','sglang',False,['thermal']),
      ('sglang-thermal-B2','sglang',False,['thermal']),
      ('vllm-thermal-A2','vllm',False,['thermal'])]
for label,engine,cached,suites in jobs:
 marker=results/f'{label}-complete.json'
 if marker.exists():status('SKIP '+label);continue
 python=str(root/'envs'/('reference' if engine=='llama' else engine)/'bin/python')
 env=os.environ.copy();env['PATH']=str(Path(python).parent)+os.pathsep+env['PATH'];env['MAX_JOBS']='4'
 if engine in ['vllm','sglang']:
  command=json.loads((root/'extended-results'/f'{engine}-tp4-pp1-command.json').read_text())
  command[0]=python
  model_flag='--model' if engine=='vllm' else '--model-path'
  command[command.index(model_flag)+1]=model
  if engine=='vllm':
   command[command.index('--safetensors-load-strategy')+1]='lazy'
   command=[('--enable-prefix-caching' if x=='--no-enable-prefix-caching' and cached else x) for x in command]
  else:
   command+=['--enable-metrics']
   if cached:command.remove('--disable-radix-cache')
   cuda=root/'envs/sglang/lib/python3.12/site-packages/nvidia/cu13';env['CUDA_HOME']=str(cuda);env['PATH']=str(cuda/'bin')+os.pathsep+env['PATH'];env['LIBRARY_PATH']=str(cuda/'lib')
 elif engine=='lmdeploy':
  command=[str(root/'envs/lmdeploy/bin/lmdeploy'),'serve','api_server',model,'--backend','turbomind','--server-name','127.0.0.1','--server-port','18080','--model-name','qwen3-32b','--dtype','bfloat16','--tp','4','--session-len','9216','--max-batch-size','32','--cache-max-entry-count','0.65','--max-prefill-token-num','2048']
  if cached:command+=['--enable-prefix-caching']
  libdirs=glob.glob(str(root/'envs/lmdeploy/lib/python3.12/site-packages/nvidia/*/lib'))
  env['LD_LIBRARY_PATH']=':'.join(libdirs)
 else:
  variant='layer' if label.endswith('-layer') else 'row' if label.endswith('-row') else 'tensor'
  command=[str(r3/'sources/llama.cpp/build/bin/llama-server'),'-m',str(r3/'models/Qwen3-32B-BF16.gguf'),'--alias','qwen3-32b','--host','127.0.0.1','--port','18080','--n-gpu-layers','99','--split-mode',variant,'--tensor-split','1,1,1,1','--ctx-size','24576','--parallel','32','--kv-unified','--kv-unified-per-slot','8192','--batch-size','2048','--ubatch-size','512','--flash-attn','on','--cache-type-k','bf16','--cache-type-v','bf16','--metrics','--slots','--cache-ram','2048' if cached else '0','--no-context-shift','--override-kv','tokenizer.ggml.add_bos_token=bool:false']
  if variant=='row':
   command[command.index('--ctx-size')+1]='8192'
  cuda=root/'envs/sglang/lib/python3.12/site-packages/nvidia/cu13'
  env['LD_LIBRARY_PATH']=':'.join([str(cuda/'lib'),str(root/'envs/sglang/lib/python3.12/site-packages/nvidia/nccl/lib')])
 (results/f'{label}-command.json').write_text(json.dumps(command,indent=2)+'\n')
 (results/f'{label}-environment.json').write_text(json.dumps({k:env.get(k) for k in ['CUDA_HOME','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','OMP_NUM_THREADS','LD_LIBRARY_PATH','PATH']},indent=2)+'\n')
 server=monitor=None;started=time.time()
 try:
  active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if active:raise RuntimeError('GPUs have active processes before launch: '+active)
  status('START '+label)
  with (results/f'{label}-server.log').open('w') as logfile,(results/f'{label}-gpu.csv').open('w') as gpu:
   monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks.mem,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown','--format=csv','--loop-ms=1000'],stdout=gpu,stderr=subprocess.STDOUT,start_new_session=True)
   server=subprocess.Popen(command,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
   deadline=time.monotonic()+2700
   while time.monotonic()<deadline:
    if server.poll() is not None:raise RuntimeError('Server exited '+str(server.returncode))
    try:
     endpoint='/health' if engine in ['vllm','sglang','llama'] else '/v1/models'
     with urllib.request.urlopen('http://127.0.0.1:18080'+endpoint,timeout=2) as response:
      if response.status==200:break
    except Exception:time.sleep(2)
   else:raise TimeoutError('Startup exceeded 45 minutes')
   (results/f'{label}-startup.json').write_text(json.dumps({'startup_s':time.time()-started})+'\n')
   thermal_start(label)
   status('READY '+label)
   for suite in suites:
    argv=[python,str(scripts/'workloads.py'),'--label',label,'--engine',engine,'--suite',suite,'--out',str(results),'--corpus',str(root/'extended-results/corpus.json'),'--tokenizer',model]
    if cached:argv+=['--cache-enabled']
    subprocess.run(argv,env=env,stdout=sys.stdout,stderr=sys.stderr,check=True)
   marker.write_text(json.dumps({'finished_at':time.time(),'suites':suites})+'\n');status('COMPLETE '+label)
 except Exception as error:
  (results/f'{label}-failure.json').write_text(json.dumps({'error':str(error),'time':time.time()})+'\n');status('FAILED '+label+' '+str(error))
 finally:
  stop(server);stop(monitor);time.sleep(5)
status('FINISHED base engine and workload jobs')
