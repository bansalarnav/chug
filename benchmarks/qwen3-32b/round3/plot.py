#!/usr/bin/env python3
"""Static exportable figures from the saved measurements."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
if not (ROOT/'results').is_dir() and (ROOT.parent/'results').is_dir():ROOT=ROOT.parent
OUT=ROOT/'results'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})

def save(fig,name):
 fig.tight_layout();fig.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight');fig.savefig(OUT/f'{name}.svg',bbox_inches='tight');plt.close(fig)

rows=json.loads((OUT/'kernels.json').read_text())['rows']
fig,axes=plt.subplots(2,2,figsize=(11,7))
for ax,op in zip(axes.flat,['gemm_qkv','gemm_attn_out','gemm_gate_up','gemm_down']):
 actual=sorted({r['operation'] for r in rows if r['operation'].startswith('gemm')})
 matches=[r for r in rows if r['operation']==op]
 if not matches:
  # Names used by the retained kernel harness.
  aliases={'gemm_attn_out':'gemm_attention_output','gemm_down':'gemm_mlp_down'};matches=[r for r in rows if r['operation']==aliases.get(op,op)]
 for tp in [1,2,4]:
  selected=sorted([r for r in matches if r.get('tp_shard')==tp],key=lambda r:r['m'])
  if selected:ax.plot([r['m'] for r in selected],[r['median_us'] for r in selected],marker='o',label=f'1/{tp} weight shard')
 ax.set(xscale='log',yscale='log',xlabel='Tokens in GEMM',ylabel='Latency, µs',title=op.replace('gemm_',''));ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Qwen3-32B BF16 GEMMs on one RTX A4500\nCUDA graph replay, medians; no inter-GPU communication',y=1.03)
save(fig,'gemm-reference')

p=OUT/'speed.csv'
if p.exists():
 data=list(csv.DictReader(p.open()));labels=sorted({r['label'] for r in data if 'thermal-' not in r['label'] and '-q4' not in r['label'] and '-q8' not in r['label']})
 if labels:
  fig,axes=plt.subplots(1,2,figsize=(12,5))
  for label in labels:
   selected=sorted([r for r in data if r['label']==label and int(r['input_tokens'])==128],key=lambda r:int(r['concurrency']))
   if selected:axes[0].plot([int(r['concurrency']) for r in selected],[float(r['output_tok_s']) for r in selected],marker='o',label=label)
   selected=sorted([r for r in data if r['label']==label and int(r['concurrency'])==1],key=lambda r:int(r['input_tokens']))
   if selected:axes[1].plot([int(r['input_tokens']) for r in selected],[float(r['ttft_ms']) for r in selected],marker='o',label=label)
  axes[0].set(xlabel='Simultaneous requests, 128 input tokens',ylabel='Aggregate output tokens/s');axes[1].set(xlabel='Input tokens, one request',ylabel='Time to first visible token, ms')
  for ax in axes:ax.grid(alpha=.2)
  axes[0].legend(fontsize=8);fig.suptitle('Qwen3-32B on bly, 128 output tokens\nMedian of three warm batches; configurations and KV budgets differ',y=1.04)
  save(fig,'serving-comparison')

p=OUT/'traffic.csv'
if p.exists():
 data=list(csv.DictReader(p.open()))
 if data:
  fig,axes=plt.subplots(1,3,figsize=(14,4))
  for label in sorted({r['label'] for r in data}):
   selected=sorted([r for r in data if r['label']==label],key=lambda r:float(r['target_rate_rps']))
   for ax,key in zip(axes,['arrival_to_finish_p50_s','arrival_to_finish_p95_s']):
    valid=[r for r in selected if r.get(key)]
    ax.plot([float(r['target_rate_rps']) for r in valid],[float(r[key]) for r in valid],marker='o',label=label)
   axes[2].plot([float(r['target_rate_rps']) for r in selected],[float(r['completed'])/float(r['requests']) for r in selected],marker='o',label=label)
  for ax,title in zip(axes,['Median completion latency','P95 completion latency']):
   ax.set(xlabel='Target arrival rate, requests/s',ylabel='Seconds from scheduled arrival',title=title,ylim=(0,None));ax.grid(alpha=.2)
  axes[2].set(xlabel='Target arrival rate, requests/s',ylabel='Completed / submitted',title='Completion fraction',ylim=(-.03,1.03));axes[2].grid(alpha=.2)
  axes[0].legend(fontsize=8);fig.suptitle('1K input / 128 output tokens, seeded Poisson arrivals\n20-second arrivals then drain; latency includes successful requests only; failures shown separately',y=1.08);save(fig,'arrival-latency')

p=OUT/'speed.csv'
if p.exists():
 data=list(csv.DictReader(p.open()));order=['vllm-thermal-A1','sglang-thermal-B1','sglang-thermal-B2','vllm-thermal-A2']
 if any(r['label'] in order for r in data):
  fig,axes=plt.subplots(1,3,figsize=(12,4))
  for ax,case in zip(axes,['speed-1024-c1','speed-128-c4','speed-128-c32']):
   for i,label in enumerate(order):
    selected=[r for r in data if r['label']==label and r['case']==case]
    if not selected:continue
    r=selected[0];y=float(r['output_tok_s']);lo=float(r['trial_min_tok_s']);hi=float(r['trial_max_tok_s'])
    ax.errorbar(i,y,yerr=[[y-lo],[hi-y]],fmt='o',color='#2878b5' if label.startswith('vllm') else '#d55e00',capsize=4)
   ax.set(xticks=range(4),xticklabels=['vLLM A1','SGL B1','SGL B2','vLLM A2'],ylabel='Aggregate output tokens/s',title=case.replace('speed-',''),ylim=(0,None));ax.tick_params(axis='x',rotation=35);ax.grid(axis='y',alpha=.2)
  fig.suptitle('A-B-B-A repeats, fresh launches cooled to ≤50°C\nDots are medians of three batches; bars show observed min/max; clocks are not fixed',y=1.07);save(fig,'thermal-repeat')

p=OUT/'reuse.csv'
if p.exists():
 data=list(csv.DictReader(p.open()));pairs=[('vllm-bf16','vllm-cache'),('sglang-bf16','sglang-cache'),('turbomind-bf16','turbomind-cache'),('llama-bf16-tensor','llama-cache')]
 if data:
  fig,axes=plt.subplots(2,2,figsize=(11,7))
  for ax,(off,on) in zip(axes.flat,pairs):
   for label,legend in [(off,'Cache off'),(on,'Cache on')]:
    selected=sorted([r for r in data if r['label']==label],key=lambda r:int(r['shared_tokens']))
    if selected:ax.plot([int(r['shared_tokens']) for r in selected],[float(r['warm_ttft_ms']) for r in selected],marker='o',label=legend)
   ax.set(xscale='log',yscale='log',xlabel='Shared prefix tokens',ylabel='Median warm TTFT, ms',title=off);ax.grid(alpha=.2)
   if ax.get_legend_handles_labels()[0]:ax.legend()
  fig.suptitle('Prefix reuse with distinct 32-token suffixes\nOne cold request, then three warm requests per size',y=1.03);save(fig,'prefix-reuse')


p=OUT/'speed.csv'
if p.exists():
 data=list(csv.DictReader(p.open()));labels=[l for l in ['llama-bf16-tensor','llama-q8_0','llama-q4_k_m','llama-q4_k_m-single'] if any(r['label']==l for r in data)]
 if any('-q' in l for l in labels):
  fig,axes=plt.subplots(1,2,figsize=(11,4))
  for label in labels:
   selected=sorted([r for r in data if r['label']==label and int(r['input_tokens'])==128],key=lambda r:int(r['concurrency']))
   if selected:axes[0].plot([int(r['concurrency']) for r in selected],[float(r['output_tok_s']) for r in selected],marker='o',label=label)
   selected=sorted([r for r in data if r['label']==label and int(r['concurrency'])==1],key=lambda r:int(r['input_tokens']))
   if selected:axes[1].plot([int(r['input_tokens']) for r in selected],[float(r['decode_tok_s']) for r in selected],marker='o',label=label)
  axes[0].set(xlabel='Simultaneous requests, 128 input tokens',ylabel='Aggregate output tokens/s',ylim=(0,None));axes[1].set(xlabel='Input tokens, one request',ylabel='Decode tokens/s',ylim=(0,None))
  for ax in axes:ax.grid(alpha=.2)
  axes[0].legend(fontsize=8);fig.suptitle('llama.cpp weight precision, BF16 KV throughout\nPrimary comparisons use four GPUs; the single-GPU case has fewer slots and a smaller cache',y=1.07);save(fig,'quantization')
