import json,os,statistics
from pathlib import Path
import torch
root=Path(os.environ['BENCH_ROOT'])/'round3/results';p=root/'kernels.json';data=json.loads(p.read_text())
if not (root/'kernels-initial.json').exists():(root/'kernels-initial.json').write_text(json.dumps(data,indent=2)+'\n')
rows=[]
with torch.inference_mode():
 for m in [1,4,16,32,128,512,2048]:
  kv=torch.zeros(8192,2,128,device='cuda',dtype=torch.bfloat16);new=torch.randn(m,2,128,device='cuda',dtype=torch.bfloat16);slots=torch.arange(m,device='cuda')
  for _ in range(5):kv.index_copy_(0,slots,new)
  torch.cuda.synchronize();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):kv.index_copy_(0,slots,new)
  torch.cuda.current_stream().wait_stream(stream);assert torch.equal(kv[:m],new)
  samples=[]
  for _ in range(5):
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
   for _ in range(100):graph.replay()
   end.record();end.synchronize();samples.append(start.elapsed_time(end)*10)
  row=dict(operation='kv_write_torch_index_copy',m=m,kv_heads=2,head_dim=128,bytes=m*2*128*4,
   median_us=statistics.median(samples),min_us=min(samples),max_us=max(samples),repeat_us=samples,
   note='Zero-initialized cache; verifies written rows only. Replaces initial uninitialized-buffer finiteness check.')
  row['effective_GB_s']=row['bytes']/row['median_us']/1000;rows.append(row);del graph
 data['rows']=[r for r in data['rows'] if r['operation']!='kv_write_torch_index_copy']+rows
 p.write_text(json.dumps(data,indent=2)+'\n')
 print('Rechecked seven KV-write shapes; exact written-row equality passed.')
