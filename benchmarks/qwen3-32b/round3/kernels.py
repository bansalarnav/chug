#!/usr/bin/env python3
"""Single-GPU Qwen3-shaped BF16 primitives, timed with CUDA graphs/events."""
import gc,json,math,os,statistics,time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
import vllm._custom_ops as ops
root=Path(os.environ['BENCH_ROOT'])/'round3/results';rows=[]
torch.cuda.set_device(0);torch.manual_seed(20261114)

def measure(label,fn,metadata):
 try:
  for _ in range(5):fn()
  torch.cuda.synchronize()
  start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
  stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph,stream=stream):
   result=fn()
  torch.cuda.current_stream().wait_stream(stream)
  if isinstance(result,torch.Tensor):assert torch.isfinite(result).all()
  iterations=30 if metadata.get('m',1)>=512 or metadata.get('context',1)>=4096 else 100
  samples=[]
  for _ in range(5):
   start.record()
   for _ in range(iterations):graph.replay()
   end.record();end.synchronize();samples.append(start.elapsed_time(end)*1000/iterations)
  row=dict(operation=label,**metadata,median_us=statistics.median(samples),min_us=min(samples),max_us=max(samples),repeat_us=samples)
  if 'flops' in metadata:row['effective_TF_s']=metadata['flops']/row['median_us']/1e6
  if 'bytes' in metadata:row['effective_GB_s']=metadata['bytes']/row['median_us']/1000
  del graph
 except Exception as error:row=dict(operation=label,**metadata,error=str(error))
 rows.append(row);(root/'kernels.json').write_text(json.dumps({'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'method':'CUDA graph replay, CUDA-event timing, five repeats; GPU0 only; allocated random BF16 operands; fused vLLM ops and PyTorch cuBLAS/SDPA. Serving overhead excluded.','rows':rows},indent=2)+'\n')
 print(json.dumps(row),flush=True)

with torch.inference_mode():
 for tp in [1,2,4]:
  for name,n,k in [('qkv',10240//tp,5120),('attention_output',5120,8192//tp),('gate_up',51200//tp,5120),('mlp_down',5120,25600//tp)]:
   weight=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)/math.sqrt(k)
   for m in [1,4,16,32,128,512,2048]:
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16);y=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
    measure('gemm_'+name,lambda:torch.mm(x,weight.t(),out=y),dict(tp_shard=tp,m=m,n=n,k=k,flops=2*m*n*k,bytes=2*(m*k+n*k+m*n)))
   del weight,x,y;torch.cuda.empty_cache()
 for m in [1,4,16,32,128,512,2048]:
  x=torch.randn(m,5120,device='cuda',dtype=torch.bfloat16);w=torch.ones(5120,device='cuda',dtype=torch.bfloat16);y=torch.empty_like(x)
  ops.rms_norm(y,x,w,1e-6)
  reference=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)).to(torch.bfloat16)
  assert torch.allclose(y,reference,atol=.02,rtol=.02)
  measure('rms_norm_vllm',lambda:ops.rms_norm(y,x,w,1e-6),dict(m=m,hidden=5120,bytes=(2*m+1)*5120*2))
  gate=torch.randn(m,12800,device='cuda',dtype=torch.bfloat16);activation=torch.empty(m,6400,device='cuda',dtype=torch.bfloat16)
  measure('silu_and_mul_vllm',lambda:torch.ops._C.silu_and_mul(activation,gate),dict(m=m,intermediate_tp4=6400,bytes=m*6400*6))
  q=torch.randn(m,2048,device='cuda',dtype=torch.bfloat16);k=torch.randn(m,256,device='cuda',dtype=torch.bfloat16)
  positions=torch.arange(m,device='cuda',dtype=torch.int64);cache=torch.cat([torch.ones(m,64,device='cuda'),torch.zeros(m,64,device='cuda')],-1).to(torch.bfloat16)
  measure('rope_vllm',lambda:ops.rotary_embedding(positions,q,k,128,cache,True),dict(m=m,q_heads=16,kv_heads=2,cos_sin='identity, precomputed'))
  kv=torch.zeros(8192,2,128,device='cuda',dtype=torch.bfloat16);new=torch.randn(m,2,128,device='cuda',dtype=torch.bfloat16);slots=torch.arange(m,device='cuda')
  measure('kv_write_torch_index_copy',lambda:kv.index_copy_(0,slots,new),dict(m=m,kv_heads=2,head_dim=128,bytes=m*2*128*4))
 for b in [1,4,16,32]:
  for context in [128,1024,4096,8192]:
   q=torch.randn(b,16,1,128,device='cuda',dtype=torch.bfloat16)
   k=torch.randn(b,2,context,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
   def attention():
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
     return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=False,enable_gqa=True)
   measure('decode_attention_torch_flash',attention,dict(batch=b,context=context,q_heads=16,kv_heads=2,head_dim=128))
   del q,k,v;torch.cuda.empty_cache()
 for context in [128,512,2048]:
  q=torch.randn(1,16,context,128,device='cuda',dtype=torch.bfloat16);k=torch.randn(1,2,context,128,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
  def prefill():
   with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
  measure('prefill_attention_torch_flash',prefill,dict(batch=1,context=context,q_heads=16,kv_heads=2,head_dim=128))
print('KERNELS COMPLETE',flush=True)
