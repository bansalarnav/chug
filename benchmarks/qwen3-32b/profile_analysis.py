#!/usr/bin/env python3
"""Inspect GPU trace overlap without counting NCCL spin/wait as model compute."""
import argparse
import collections
import gzip
import json
from pathlib import Path


def merge(intervals):
    result=[]
    for start,end in sorted(intervals):
        if result and start<=result[-1][1]: result[-1]=(result[-1][0],max(result[-1][1],end))
        else: result.append((start,end))
    return result


def duration(intervals): return sum(b-a for a,b in merge(intervals))


def intersection(a,b):
    a,b=merge(a),merge(b); i=j=0; result=[]
    while i<len(a) and j<len(b):
        start,end=max(a[i][0],b[j][0]),min(a[i][1],b[j][1])
        if end>start:result.append((start,end))
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('results',type=Path);a=p.parse_args()
    summaries=[]; grouped={}
    for path in sorted((a.results/'profiles').rglob('*')):
        if not path.is_file() or not (path.name.endswith('.json') or path.name.endswith('.json.gz')):continue
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as f:trace=json.load(f)
        events=trace.get('traceEvents',[]) if isinstance(trace,dict) else trace
        group=grouped.setdefault(path.parent.name,{'compute':collections.defaultdict(list),'comm':collections.defaultdict(list),'base':trace.get('baseTimeNanoseconds',0),'files':[]})
        adjustment=(trace.get('baseTimeNanoseconds',0)-group['base'])/1000
        group['files'].append(path.relative_to(a.results).as_posix())
        kernels=[e for e in events if e.get('ph')=='X' and e.get('cat')=='kernel' and e.get('dur',0)>0]
        compute=collections.defaultdict(list);comm=collections.defaultdict(list);names=collections.Counter()
        for e in kernels:
            device=e.get('args',{}).get('device')
            if device is None:continue
            interval=(e['ts'],e['ts']+e['dur']);name=e.get('name',''); names[name]+=e['dur']
            target=comm if any(word in name.lower() for word in ['nccl','allreduce','all_reduce','sendrecv','send_recv']) else compute
            target[str(device)].append(interval)
            group['comm' if target is comm else 'compute'][str(device)].append((interval[0]+adjustment,interval[1]+adjustment))
        row=dict(file=path.relative_to(a.results).as_posix(),kernel_events=len(kernels),
            categories=dict(collections.Counter(e.get('cat','') for e in events)),
            compute_union_us={d:duration(v) for d,v in compute.items()},
            communication_union_us={d:duration(v) for d,v in comm.items()},
            largest_summed_kernel_us=names.most_common(12))
        # Only claim cross-stage overlap when physical GPU IDs from both stages are present.
        if '0' in compute and '2' in compute:
            left,right=compute['0'],compute['2']
            both=duration(intersection(left,right));either=duration(left+right)
            row.update(gpu0_gpu2_compute_overlap_us=both,
                gpu0_gpu2_compute_union_us=either,
                gpu0_gpu2_overlap_fraction_of_compute_union=both/either if either else None,
                overlap_definition='Intersection of non-NCCL kernel unions on physical GPUs 0 and 2 divided by their union; includes prefill, excludes host overhead and communication kernels. Does not measure occupancy or causal pipeline bubble time.')
        summaries.append(row)
    group_summaries=[]
    for name,g in grouped.items():
        row=dict(profile=name,files=g['files'],physical_gpu_ids=sorted(g['compute']),
            compute_union_us={d:duration(v) for d,v in g['compute'].items()},
            communication_union_us={d:duration(v) for d,v in g['comm'].items()})
        if '0' in g['compute'] and '2' in g['compute']:
            left,right=g['compute']['0'],g['compute']['2']
            both=duration(intersection(left,right));either=duration(left+right)
            row.update(gpu0_gpu2_compute_overlap_us=both,gpu0_gpu2_compute_union_us=either,
                gpu0_gpu2_overlap_fraction_of_compute_union=both/either if either else None,
                overlap_definition='Non-communication GPU-kernel time intersected across physical GPUs 0 and 2, divided by its union. Includes prefill and instrumented execution; does not measure SM occupancy or causal bubble time.')
        else:
            row['limitation']='Both pipeline stages were not saved; cross-stage overlap cannot be measured.'
        group_summaries.append(row)
    (a.results/'profile-groups.json').write_text(json.dumps(group_summaries,indent=2)+'\n')
    (a.results/'profile-analysis.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(group_summaries,indent=2))


if __name__=='__main__':main()
