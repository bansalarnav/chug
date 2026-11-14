#!/usr/bin/env python3
"""Plot validated serving results and communication measurements."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS=['vllm-tp4-pp1','vllm-tp2-pp2','sglang-tp4-pp1','sglang-tp2-pp2']
NAMES=['vLLM TP4','vLLM TP2 × PP2','SGLang TP4','SGLang TP2 × PP2']
COLORS=['#2459a6','#2459a6','#b85617','#b85617']
STYLES=['-','--','-','--']
MARKERS=['o','o','s','s']


def clean(ax):
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    for side in ['top','right']: ax.spines[side].set_visible(False)


def serving(root):
    rows=json.loads((root/'validated-summary.json').read_text())
    assert len(rows)==116
    fig,axes=plt.subplots(2,2,figsize=(9,8.4),layout='constrained')
    fig.suptitle('Qwen3-32B BF16 · four RTX A4500 GPUs',fontsize=15)
    specs=[('prefill_probe','input_tokens','median_e2e_s',1000,'One-output-token request latency','Input tokens','Milliseconds'),
           ('context_decode','input_tokens','median_decode_tokens_per_s',1,'Decode with one request, 256 output tokens','Input tokens','Output tokens/s'),
           ('batch_short','concurrency','median_output_tokens_per_s',1,'Batch scaling: 128 input / 128 output','Concurrent requests','Total output tokens/s'),
           ('batch_1k','concurrency','median_output_tokens_per_s',1,'Batch scaling: 1024 input / 128 output','Concurrent requests','Total output tokens/s')]
    for ax,spec in zip(axes.flat,specs):
        suite,xkey,ykey,mult,title,xlabel,ylabel=spec
        for label,name,color,style,marker in zip(LABELS,NAMES,COLORS,STYLES,MARKERS):
            series=sorted([r for r in rows if r['label']==label and r['suite']==suite],key=lambda r:r[xkey])
            x=[r[xkey] for r in series]; y=[r[ykey]*mult for r in series]
            ax.plot(x,y,color=color,linestyle=style,marker=marker,markersize=5,label=name,linewidth=1.6,
                markerfacecolor='white' if label.startswith('vllm') else color)
            if suite.startswith('batch'):
                ax.errorbar(x,y,yerr=[[r[ykey]-r['min_output_tokens_per_s'] for r in series],
                    [r['max_output_tokens_per_s']-r[ykey] for r in series]],fmt='none',ecolor=color,capsize=2)
        ax.set_title(title,fontsize=11,loc='left'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        clean(ax)
        if suite.startswith('batch'):
            values=[1,2,4,8,16,32] if suite=='batch_short' else [1,2,4,8,16]
            ax.set_xscale('log',base=2); ax.set_xticks(values,[str(v) for v in values]); ax.set_ylim(bottom=0)
        else:
            ax.set_xticks([128,1024,4096,8192],['128','1K','4K','8K'])
            ax.set_ylim(bottom=0)
    axes[0,0].legend(fontsize=10)
    fig.supxlabel('Warm runs; batch rate includes prefill. Some SGLang TP4 runs thermally throttled; see report.',fontsize=10)
    fig.savefig(root/'comparison.png',dpi=160,facecolor='white')
    fig.savefig(root/'comparison.svg',facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(9,8.4),layout='constrained')
    fig.suptitle('Closed-loop serving and mixed requests',fontsize=15)
    specs=[('median_output_tokens_per_s',1,'Closed-loop output throughput','Output tokens/s'),
           ('p95_ttft_ms',1,'Closed-loop P95 time to first text','Milliseconds'),
           ('output_tokens_per_gpu_joule',1,'Approximate GPU energy efficiency','Output tokens / GPU joule')]
    for ax,(key,mult,title,ylabel) in zip(list(axes.flat)[:3],specs):
        for label,name,color,style,marker in zip(LABELS,NAMES,COLORS,STYLES,MARKERS):
            s=sorted([r for r in rows if r['label']==label and r['suite']=='closed_loop'],key=lambda r:r['concurrency'])
            ax.plot([r['concurrency'] for r in s],[r[key]*mult for r in s],color=color,linestyle=style,marker=marker,label=name)
        ax.set_title(title,fontsize=11,loc='left'); ax.set_xlabel('Concurrent requests'); ax.set_ylabel(ylabel)
        ax.set_xticks([4,16,32]); clean(ax)
    ax=axes[1,1]
    for label,name,color,style,marker in zip(LABELS,NAMES,COLORS,STYLES,MARKERS):
        raw=[json.loads(line) for line in (root/f'{label}.jsonl').read_text().splitlines()]
        times=sorted(r['e2e_s'] for t in raw if t['case']['suite']=='mixed' for r in t['requests'])
        ax.plot(times,[(i+1)/len(times) for i in range(len(times))],color=color,linestyle=style,label=name)
    ax.set_title('Mixed trace: request-latency distribution',fontsize=11,loc='left')
    ax.set_xlabel('End-to-end request seconds'); ax.set_ylabel('Fraction completed'); ax.set_ylim(0,1); clean(ax)
    axes[0,0].legend(fontsize=10)
    fig.supxlabel('Finite traces include fill/drain; power is GPU-only. Thermal behavior confounds small engine gaps.',fontsize=10)
    fig.savefig(root/'load.png',dpi=160,facecolor='white')
    plt.close(fig)


def pipeline(root):
    rows=json.loads((root/'validated-pipeline-summary.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(9,4.2),layout='constrained')
    for c,color in [(4,'#2459a6'),(16,'#b85617')]:
        series=sorted([r for r in rows if r['concurrency']==c],key=lambda r:r['microbatch_cap'])
        x=[r['microbatch_cap'] for r in series]
        for ax,key in [(axes[0],'median_output_tokens_per_s'),(axes[1],'median_ttft_ms')]:
            ax.plot(x,[r[key] for r in series],marker='o',color=color,label=f'{c} concurrent requests')
    axes[0].set_title('SGLang TP2 × PP2: microbatch sweep',fontsize=11,loc='left')
    axes[0].set_ylabel('Total output tokens/s')
    axes[1].set_title('Time to first text',fontsize=11,loc='left');axes[1].set_ylabel('Median milliseconds')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],['1','2','4','8','16'])
        ax.set_xlabel('Maximum requests per pipeline microbatch');clean(ax)
    axes[0].legend(fontsize=9)
    fig.supxlabel('128 input / 128 output tokens, three repeats. Cap 1 limits admission to two active requests.',fontsize=10)
    fig.savefig(root/'pipeline.png',dpi=160,facecolor='white')
    plt.close(fig)


def communication(root):
    rows=json.loads((root/'communication.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(9,4.3),layout='constrained')
    for kind,name,color in [('allreduce_nvlink_pair','Two NVLink pairs','#2459a6'),('allreduce_tp4','All four GPUs','#b85617')]:
        s=[r for r in rows if r['operation']==kind]
        axes[0].plot([r['payload_bytes']/1024 for r in s],[r['median_us'] for r in s],marker='o',label=name,color=color)
    axes[0].set_yscale('log'); axes[0].set_ylabel('Microseconds / all-reduce'); axes[0].set_title('NCCL all-reduce latency',loc='left')
    for kind,name,color in [('p2p_nvlink','Within NVLink pair','#2459a6'),('p2p_pcie','Across pairs, PCIe','#b85617')]:
        s=[r for r in rows if r['operation']==kind]
        axes[1].plot([r['payload_bytes']/1024 for r in s],[r['effective_payload_GB_s'] for r in s],marker='o',label=name,color=color)
    axes[1].set_ylabel('Effective payload GB/s'); axes[1].set_title('NCCL point-to-point throughput',loc='left')
    for ax in axes:
        ax.set_xscale('log'); ax.set_xlabel('Payload KiB'); ax.legend(fontsize=10); clean(ax)
    fig.supxlabel('Five repeats. Includes Python dispatch and synchronization. Payload/time is not ring bus bandwidth.',fontsize=10)
    fig.savefig(root/'communication.png',dpi=160,facecolor='white')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('results',type=Path); p.add_argument('--communication-only',action='store_true'); a=p.parse_args()
    communication(a.results)
    if not a.communication_only:
        serving(a.results)
        if (a.results/'validated-pipeline-summary.json').exists(): pipeline(a.results)
