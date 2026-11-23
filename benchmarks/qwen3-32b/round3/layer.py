#!/usr/bin/env python3
"""Actual Qwen3 layer 0, TP4 PyTorch reference and graph-timed execution."""
import datetime,json,os,statistics,time
from pathlib import Path
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from safetensors import safe_open
root=Path(os.environ['BENCH_ROOT']);rank=int(os.environ['RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(minutes=5))
model=Path((root/'results/model-path.txt').read_text().strip());index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
weights={}
for name,path in index.items():
 if name.startswith('model.layers.0.'):
  with safe_open(model/path,framework='pt',device='cpu') as f:weights[name.removeprefix('model.layers.0.')]=f.get_tensor(name).to(f'cuda:{rank}')
full=weights if rank==0 else None
local={}
for name,w in weights.items():
 if any(s in name for s in ['q_proj','k_proj','v_proj','gate_proj','up_proj']):local[name]=w.chunk(4,dim=0)[rank].contiguous().clone()
 elif any(s in name for s in ['o_proj','down_proj']):local[name]=w.chunk(4,dim=1)[rank].contiguous().clone()
 else:local[name]=w
if rank!=0:del weights

def norm(x,w):return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(x.dtype)*w

def forward(x,w,heads,kvheads,oldk,oldv,positions,reduce):
 residual=x;z=norm(x,w['input_layernorm.weight']);b,l,_=x.shape
 q=F.linear(z,w['self_attn.q_proj.weight']).view(b,l,heads,128)
 k=F.linear(z,w['self_attn.k_proj.weight']).view(b,l,kvheads,128)
 v=F.linear(z,w['self_attn.v_proj.weight']).view(b,l,kvheads,128).transpose(1,2)
 q=norm(q,w['self_attn.q_norm.weight']).transpose(1,2);k=norm(k,w['self_attn.k_norm.weight']).transpose(1,2)
 freq=positions[:,None].float()*(1/(1e6**(torch.arange(0,128,2,device=x.device).float()/128)))[None]
 angles=torch.cat([freq,freq],dim=-1);cos=angles.cos().to(x.dtype)[None,None];sin=angles.sin().to(x.dtype)[None,None]
 def rotate(t):return torch.cat([-t[...,64:],t[...,:64]],dim=-1)
 q=q*cos+rotate(q)*sin;k=k*cos+rotate(k)*sin
 if oldk is not None:k=torch.cat([oldk,k],dim=2);v=torch.cat([oldv,v],dim=2)
 with sdpa_kernel(SDPBackend.FLASH_ATTENTION):a=F.scaled_dot_product_attention(q,k,v,is_causal=oldk is None,enable_gqa=True)
 a=a.transpose(1,2).reshape(b,l,heads*128);z=F.linear(a,w['self_attn.o_proj.weight'])
 if reduce:dist.all_reduce(z)
 z=z+residual;residual=z;z=norm(z,w['post_attention_layernorm.weight'])
 z=F.linear(F.silu(F.linear(z,w['mlp.gate_proj.weight']))*F.linear(z,w['mlp.up_proj.weight']),w['mlp.down_proj.weight'])
 if reduce:dist.all_reduce(z)
 return z+residual

rows=[]
with torch.inference_mode():
 for case_id,(b,l,past) in enumerate([(1,128,0),(1,512,0),(1,2048,0),(1,1,128),(1,1,1024),(1,1,4096),(4,1,1024),(16,1,1024),(32,1,1024)]):
  torch.manual_seed(1400+case_id)
  x=torch.randn(b,l,5120,device=f'cuda:{rank}',dtype=torch.bfloat16)
  oldk=torch.randn(b,8,past,128,device=f'cuda:{rank}',dtype=torch.bfloat16) if past else None
  oldv=torch.randn_like(oldk) if past else None
  lk=oldk[:,rank*2:(rank+1)*2].contiguous() if past else None;lv=oldv[:,rank*2:(rank+1)*2].contiguous() if past else None
  positions=torch.arange(past,past+l,device=f'cuda:{rank}')
  fn=lambda:forward(x,local,16,2,lk,lv,positions,True)
  actual=fn();check=None
  if rank==0:
   expected=forward(x,full,64,8,oldk,oldv,positions,False)
   error=(actual.float()-expected.float());rmse=float(error.square().mean().sqrt());relative=rmse/float(expected.float().square().mean().sqrt())
   check=dict(max_abs_error=float(error.abs().max()),rmse=rmse,relative_rmse=relative)
   assert relative<.03 and torch.isfinite(actual).all(),check
   del expected
  dist.barrier()
  for _ in range(5):fn()
  torch.cuda.synchronize();dist.barrier()
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):output=fn()
  torch.cuda.current_stream().wait_stream(stream)
  iterations=30;samples=[]
  for _ in range(5):
   dist.barrier();torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
   for _ in range(iterations):graph.replay()
   end.record();end.synchronize();values=[None]*4;dist.all_gather_object(values,start.elapsed_time(end)*1000/iterations)
   if rank==0:samples.append(max(values))
  if rank==0:
   row=dict(batch=b,tokens=l,past_tokens=past,tp=4,median_us=statistics.median(samples),repeat_us=samples,comparison_to_unsharded=check)
   rows.append(row);(root/'round3/results/layer.json').write_text(json.dumps({'method':'Actual checkpoint layer 0, BF16, PyTorch cuBLAS/Flash SDPA, NCCL reductions, CUDA graph replay. Includes unfused norm/RoPE and allocation shapes, excludes host serving.','rows':rows},indent=2)+'\n');print(json.dumps(row),flush=True)
  del graph,actual,output,x,oldk,oldv,lk,lv;torch.cuda.empty_cache()
dist.destroy_process_group()
