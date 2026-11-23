#!/usr/bin/env python3
"""Repeat only the retained TensorRT probe under a direct four-rank MPI launch."""
import ast,json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
from service import service as actual_service
root=Path(os.environ['BENCH_ROOT']);r3=root/'round3';out=r3/'results';scripts=r3/'scripts'
sys.path.insert(0,str(root/'scripts'))
def interrupt(signum,frame):raise SystemExit(signum)
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
model=(root/'results/model-path.txt').read_text().strip();cuda=root/'envs/sglang/lib/python3.12/site-packages/nvidia/cu13'

def service(label,command,env,results,health='/health'):
 command=[str(root/'envs/tensorrt/bin/mpirun'),'-np','4']+command
 return actual_service(label,command,env,results,health)

source=ast.parse((scripts/'addons.py').read_text());function=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='trt_probe')
class Rename(ast.NodeTransformer):
 def visit_Constant(self,node):
  if node.value=='tensorrt-bf16-tp4-b4':return ast.copy_location(ast.Constant('tensorrt-bf16-tp4-b4-mpirun'),node)
  return node
function=Rename().visit(function);exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'retained_trt_probe','exec'))
start=time.time()
try:record=dict(ok=True,result=trt_probe(),elapsed_s=time.time()-start)
except Exception as error:record=dict(ok=False,error=repr(error),elapsed_s=time.time()-start)
(out/'tensorrt-mpirun-probe-addon.json').write_text(json.dumps(record,indent=2)+'\n');print(record,flush=True)
