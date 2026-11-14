#!/usr/bin/env python3
"""NCCL communication baseline, including Python dispatch and synchronization."""
import datetime
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', device_id=torch.device('cuda', rank))
groups = [([0, 1], dist.new_group([0, 1])),
          ([2, 3], dist.new_group([2, 3])),
          ([0, 2], dist.new_group([0, 2]))]
rows = []
for kind in ['allreduce_nvlink_pair', 'allreduce_tp4', 'p2p_nvlink', 'p2p_pcie']:
    selected = [0, 1] if kind in ['allreduce_nvlink_pair', 'p2p_nvlink'] else [0, 2]
    group = groups[0][1] if selected == [0, 1] else groups[2][1]
    for size in [1024, 10240, 40960, 163840, 327680, 1048576, 20971520, 67108864]:
        iterations = 300 if size < 1048576 else 100 if size < 20971520 else 30
        tensor = torch.zeros(size // 2, dtype=torch.bfloat16, device=f'cuda:{rank}')
        def operation():
            if kind == 'allreduce_tp4':
                dist.all_reduce(tensor)
            elif kind == 'allreduce_nvlink_pair':
                dist.all_reduce(tensor, group=groups[0 if rank < 2 else 1][1])
            elif rank == selected[0]:
                dist.send(tensor, dst=selected[1], group=group)
            elif rank == selected[1]:
                dist.recv(tensor, src=selected[0], group=group)
        samples = []
        for repeat in range(5):
            dist.barrier()
            for _ in range(10):
                operation()
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            for _ in range(iterations):
                operation()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            times = [None] * 4
            dist.all_gather_object(times, elapsed / iterations)
            if rank == 0:
                participating = range(4) if kind.startswith('allreduce') else selected
                samples.append(max(times[r] for r in participating))
        if rank == 0:
            median = statistics.median(samples)
            row = dict(operation=kind, payload_bytes=size, iterations=iterations,
                repetitions=5, median_us=median*1e6, min_us=min(samples)*1e6,
                max_us=max(samples)*1e6, effective_payload_GB_s=size/median/1e9,
                repeat_s=samples)
            rows.append(row)
            print(json.dumps(row), flush=True)
        del tensor
if rank == 0:
    out = Path(os.environ['BENCH_ROOT']) / 'extended-results'
    out.mkdir(exist_ok=True)
    (out/'communication.json').write_text(json.dumps(rows, indent=2)+'\n')
dist.destroy_process_group()
