#!/usr/bin/env python3
"""CUDA-graph NCCL all-reduce timings; run only after model servers stop."""
import json
import datetime
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist

rank=int(os.environ['RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl',device_id=torch.device('cuda',rank),timeout=datetime.timedelta(seconds=90))
pairs=[dist.new_group([0,1]),dist.new_group([2,3])]
# Warm device clocks before measuring very small messages.
warm=torch.ones((2048,2048),dtype=torch.bfloat16,device=f'cuda:{rank}')
warm_out=torch.empty_like(warm)
deadline=time.perf_counter()+1.0
while time.perf_counter()<deadline:
    torch.mm(warm,warm,out=warm_out)
    torch.cuda.synchronize()
del warm,warm_out
dist.barrier(); torch.cuda.synchronize()
rows=[]
for kind in ['allreduce_nvlink_pair','allreduce_tp4']:
    group=pairs[rank//2] if kind=='allreduce_nvlink_pair' else None
    for size in [10240,40960,163840,327680,1048576,20971520]:
        tensor=torch.zeros(size//2,dtype=torch.bfloat16,device=f'cuda:{rank}')
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(30): dist.all_reduce(tensor,group=group)
        stream.synchronize(); dist.barrier(); torch.cuda.synchronize()
        operations=32 if size<1048576 else 8
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            for _ in range(operations): dist.all_reduce(tensor,group=group)
        repeats=100 if size<1048576 else 20
        gpu_samples=[]; wall_samples=[]
        for repeat in range(5):
            for _ in range(3): graph.replay()
            dist.barrier(); torch.cuda.synchronize()
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            wall_start=time.perf_counter(); start.record()
            for _ in range(repeats): graph.replay()
            end.record(); end.synchronize()
            elapsed=time.perf_counter()-wall_start
            sample=(start.elapsed_time(end)/1000/(repeats*operations),elapsed/(repeats*operations))
            values=[None]*4; dist.all_gather_object(values,sample)
            if rank==0:
                gpu_samples.append(max(v[0] for v in values)); wall_samples.append(max(v[1] for v in values))
        if rank==0:
            row=dict(operation=kind,payload_bytes=size,operations_per_graph=operations,
                graph_replays=repeats,repetitions=5,
                gpu_median_us=statistics.median(gpu_samples)*1e6,
                gpu_min_us=min(gpu_samples)*1e6,gpu_max_us=max(gpu_samples)*1e6,
                wall_median_us=statistics.median(wall_samples)*1e6,
                gpu_repeat_s=gpu_samples,wall_repeat_s=wall_samples)
            rows.append(row); print(json.dumps(row),flush=True)
        del graph,tensor
if rank==0:
    (Path(os.environ['BENCH_ROOT'])/'extended-results/communication-graph.json').write_text(json.dumps(rows,indent=2)+'\n')
dist.destroy_process_group()
