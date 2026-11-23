"""Server lifecycle helpers for isolated addon measurements."""
import contextlib,json,os,signal,subprocess,time,urllib.request
from pathlib import Path


def stop(p):
 if p is None:return
 try:os.killpg(p.pid,signal.SIGTERM)
 except ProcessLookupError:return
 try:p.wait(timeout=40)
 except subprocess.TimeoutExpired:
  try:os.killpg(p.pid,signal.SIGKILL)
  except ProcessLookupError:pass
  p.wait(timeout=10)


@contextlib.contextmanager
def service(label,command,env,results,health='/health'):
 active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if active:raise RuntimeError('GPUs have active processes before launch: '+active)
 process=monitor=None;started=time.time()
 (results/f'{label}-environment.json').write_text(json.dumps({k:env.get(k) for k in ['CUDA_HOME','CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','OMP_NUM_THREADS','LD_LIBRARY_PATH','PATH']},indent=2)+'\n')
 (results/f'{label}-command.json').write_text(json.dumps(command,indent=2)+'\n')
 try:
  with (results/f'{label}-server.log').open('w') as logfile,(results/f'{label}-gpu.csv').open('w') as gpulog:
   monitor=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.sm,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown','--format=csv','--loop-ms=1000'],stdout=gpulog,stderr=subprocess.STDOUT,start_new_session=True)
   process=subprocess.Popen(command,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
   deadline=time.monotonic()+2700
   while time.monotonic()<deadline:
    if process.poll() is not None:raise RuntimeError('Server exited '+str(process.returncode))
    try:
     with urllib.request.urlopen('http://127.0.0.1:18080'+health,timeout=2) as response:
      if response.status==200:break
    except Exception:time.sleep(2)
   else:raise TimeoutError('Server did not become healthy')
   deadline=time.monotonic()+600
   while True:
    temps=[int(v) for v in subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).split()]
    if max(temps)<=50:break
    if time.monotonic()>deadline:raise TimeoutError('Cooling exceeded ten minutes')
    time.sleep(5)
   (results/f'{label}-startup.json').write_text(json.dumps({'startup_and_cooldown_s':time.time()-started,'start_temperatures_c':temps})+'\n')
   yield process
 finally:stop(process);stop(monitor);time.sleep(5)
