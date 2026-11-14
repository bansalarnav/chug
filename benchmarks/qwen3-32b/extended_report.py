#!/usr/bin/env python3
"""Validate expanded measurements, derive sampled GPU energy, and export tables."""
import argparse
import bisect
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from extended import cases, summarize

LABELS=['vllm-tp4-pp1','vllm-tp2-pp2','sglang-tp4-pp1','sglang-tp2-pp2']


def sensors(path, offset):
    result={str(i):[] for i in range(4)}
    def number(value):
        try: return float(value.split()[0])
        except (ValueError,AttributeError): return None
    with path.open() as f:
        for row in csv.DictReader(f,skipinitialspace=True):
            try:
                timestamp=dt.datetime.strptime(row['timestamp'].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp()-offset
                gpu=row['index'].strip()
                power=number(row.get('power.draw [W]'))
                memory=number(row.get('memory.used [MiB]'))
                if power is not None and memory is not None:
                    result[gpu].append((timestamp,power,memory))
            except (KeyError,ValueError):
                continue
    return {gpu:sorted(values) for gpu,values in result.items()}


def integral(values,start,end):
    times=[r[0] for r in values]
    if not times or start<times[0] or end>times[-1]: return None
    def at(t):
        i=max(0,min(len(times)-2,bisect.bisect_right(times,t)-1))
        a,b=values[i],values[i+1]
        return a[1]+(b[1]-a[1])*(t-a[0])/(b[0]-a[0]) if b[0]!=a[0] else a[1]
    points=[(start,at(start))]+[(r[0],r[1]) for r in values if start<r[0]<end]+[(end,at(end))]
    return sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in zip(points,points[1:]))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('results',type=Path)
    p.add_argument('--allow-partial',action='store_true')
    args=p.parse_args(); root=args.results
    matrix=cases(); all_summary=[]; total_requests=total_trials=0
    digest=hashlib.sha256((root/'corpus.json').read_bytes()).hexdigest()
    offset=json.loads((root/'sensor-timezone.json').read_text())['offset_seconds']
    thermal=[]
    if (root/'thermal.csv').exists():
        with (root/'thermal.csv').open() as f:
            for r in csv.DictReader(f,skipinitialspace=True):
                timestamp=dt.datetime.strptime(r['timestamp'].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp()-offset
                active=any(v.strip()=='Active' for k,v in r.items() if 'thermal_slowdown' in k)
                thermal.append((timestamp,r['index'].strip(),active,int(r['temperature.gpu'])))
    memory={}; capacities={}
    for label in LABELS:
        rawfile=root/f'{label}.jsonl'
        if not rawfile.exists():
            assert args.allow_partial, f'Missing {label}'
            continue
        data=rawfile.read_text()
        assert data.endswith('\n'), 'Partial final record'
        raw=[json.loads(line) for line in data.splitlines()]
        assert len({(r['case_id'],r['trial']) for r in raw})==len(raw)
        log=(root/f'{label}-server.log').read_text()
        values=re.findall(r'GPU KV cache size: ([0-9,]+) tokens',log) if label.startswith('vllm') else re.findall(r'max_total_num_tokens=(\d+)',log)
        capacities[label]=min((int(v.replace(',','')) for v in values),default=None)
        gpu=sensors(root/f'{label}-gpu.csv',offset)
        cutoff=max(t['wall_finished_at'] for t in raw)
        memory[label]={i:max((r[2] for r in values if r[0]<=cutoff),default=None) for i,values in gpu.items()}
        for i,case in enumerate(matrix):
            trials=sorted([r for r in raw if r['case_id']==i],key=lambda r:r['trial'])
            if len(trials)!=case['repetitions']:
                assert args.allow_partial, f'{label} case {i} incomplete'
                continue
            for j,trial in enumerate(trials):
                assert trial['trial']==j and trial['label']==label and trial['case']==case
                assert trial['corpus_sha256']==digest
                assert len(trial['requests'])==case['requests']
                for k,row in enumerate(trial['requests']):
                    expected_input=case['input_tokens'] if case['suite']!='mixed' else [128,1024,4096,128,1024,4096,128,1024][k%8]
                    expected_output=case['output_tokens'] if case['suite']!='mixed' else 256 if k%4==0 else 128
                    assert row['usage']['prompt_tokens']==expected_input==row['input_tokens']
                    assert row['usage']['completion_tokens']==expected_output==row['output_tokens']
                    assert row['e2e_s']>0
                    if expected_output>1:
                        assert 0<row['ttft_s']<=row['e2e_s'] and row['tpot_s']>0
                        assert abs(row['tpot_s']-(row['e2e_s']-row['ttft_s'])/(expected_output-1))<1e-7
                tokens=sum(r['output_tokens'] for r in trial['requests'])
                assert abs(tokens/trial['elapsed_s']-trial['aggregate_output_tokens_per_s'])<1e-8
                total_requests+=len(trial['requests']); total_trials+=1
            summary=summarize(case,trials,label)
            summary['case_id']=i
            matching=[r for r in thermal if any(t['wall_started_at']<=r[0]<=t['wall_finished_at'] for t in trials)]
            covered=bool(thermal) and min(t['wall_started_at'] for t in trials)>=thermal[0][0] and max(t['wall_finished_at'] for t in trials)<=thermal[-1][0]
            summary['thermal_sensor_coverage']='full' if covered else 'partial' if matching else 'unavailable'
            summary['thermal_slowdown_samples']=sum(r[2] for r in matching) if matching else None
            summary['thermal_peak_c']=max((r[3] for r in matching),default=None)

            summary['gpu_joules']=summary['output_tokens_per_gpu_joule']=summary['mean_total_gpu_power_w']=None
            if case['suite'] in ['long_decode','closed_loop','mixed'] and min(t['elapsed_s'] for t in trials)>=10:
                joules=0; duration=0
                for trial in trials:
                    start,end=trial['wall_started_at'],trial['wall_finished_at']
                    energy=[integral(v,start,end) for v in gpu.values()]
                    assert len(energy)==4 and all(e is not None for e in energy), f'Sensor coverage missing: {label}/{i}'
                    joules+=sum(energy); duration+=end-start
                summary.update(gpu_joules=joules,
                    output_tokens_per_gpu_joule=sum(r['output_tokens'] for t in trials for r in t['requests'])/joules,
                    mean_total_gpu_power_w=joules/duration)
            all_summary.append(summary)
    if not args.allow_partial:
        assert len(all_summary)==116 and total_trials==364 and total_requests==2992
    (root/'validated-summary.json').write_text(json.dumps(all_summary,indent=2)+'\n')
    fields=list(all_summary[0]) if all_summary else []
    if fields:
        # Probe-only derived values are absent from other suites.
        fields=list(dict.fromkeys(k for r in all_summary for k in r))
        with (root/'summary.csv').open('w') as file:
            writer=csv.DictWriter(file,fieldnames=fields); writer.writeheader(); writer.writerows(all_summary)
    original=root.parent/'results'
    model=json.loads((original/'model.json').read_text())
    lines=['# Expanded Qwen3-32B BF16 measurements','',
        f'Main matrix: {len(all_summary)} validated cases, {total_trials} measured trials/traces, and {total_requests} requests. The pipeline microbatch section adds seven cases and 156 requests when complete.',
        f'Model revision `{model["revision"]}`. Corpus SHA256 `{digest}`.','',
        'Four RTX A4500 GPUs. Two NVLink pairs, PCIe between pairs. Context cap 9216, running-request cap 32, prefill chunk/budget 2048, memory fraction 0.88. Prefix reuse and speculation disabled. See [methodology](../EXTENDED.md) for the full configuration and timing definitions.','',
        'For initial engine work, compare context_decode and long_decode for one-request execution, prefill_probe for prompt processing, batch_short/batch_1k for batching, and closed_loop/mixed for scheduling. GPU memory is a separate allocation baseline.','',
        'Decode rate uses a single request where concurrency is 1. Output throughput includes prefill and client overhead. Prefill-probe input throughput includes one output token and a nonstreaming HTTP response. P95 values are descriptive and require at least 20 observed requests. Closed-loop traces include filling and draining.','']
    lines+=['Some SGLang TP4 cases reported thermal slowdown during this sustained session. Small differences between engine implementations are confounded by temperature and clock behavior. Thermal-reason logging started after the vLLM throughput runs, so their absence of flags is unverified. The CSV records per-case sensor coverage and thermal samples; all configurations have clock and temperature samples.','']
    if len(all_summary)==116:
        def find(label,suite,**match):
            return next(r for r in all_summary if r['label']==label and r['suite']==suite and all(r[k]==v for k,v in match.items()))
        vtp=find('vllm-tp4-pp1','context_decode',input_tokens=1024)
        vpp=find('vllm-tp2-pp2','context_decode',input_tokens=1024)
        stp=find('sglang-tp4-pp1','context_decode',input_tokens=1024)
        spp=find('sglang-tp2-pp2','context_decode',input_tokens=1024)
        lines+=['## Findings','',
            f'- Single request, 1024 input/256 output: vLLM TP4 {vtp["median_decode_tokens_per_s"]:.1f} decode tok/s, TP2 × PP2 {vpp["median_decode_tokens_per_s"]:.1f}; SGLang {stp["median_decode_tokens_per_s"]:.1f} and {spp["median_decode_tokens_per_s"]:.1f}.',
            f'- Continuous 32-client short requests: vLLM TP4 {find("vllm-tp4-pp1","closed_loop",concurrency=32)["median_output_tokens_per_s"]:.1f} total output tok/s; SGLang TP4 {find("sglang-tp4-pp1","closed_loop",concurrency=32)["median_output_tokens_per_s"]:.1f}. The pipeline layouts deliver about 400 tok/s.',
            f'- 8192-input/1-output prompt probe: vLLM TP2 × PP2 {find("vllm-tp2-pp2","prefill_probe",input_tokens=8192)["median_e2e_s"]:.2f} s, TP4 {find("vllm-tp4-pp1","prefill_probe",input_tokens=8192)["median_e2e_s"]:.2f} s. The pipeline layout wins this prefill-heavy case.',
            '- Both engines schedule overlapping pipeline work. TP4 winning batched decode does not imply that the pipeline implementation simply leaves one pair idle. Smaller microbatches, local matrix shapes and weight reuse, communication, and scheduling all affect the tradeoff.',
            '- The similar overall engine results do not establish a small implementation advantage: sequential test order and recorded thermal slowdown confound percent-level differences.','']
    if (root/'comparison.png').exists(): lines+=['![Expanded comparison](comparison.png)','']
    if (root/'load.png').exists(): lines+=['![Closed-loop and mixed requests](load.png)','']
    for suite in dict.fromkeys(c['suite'] for c in matrix):
        lines += [f'## {suite.replace("_"," ")}', '',
            '| Engine/layout | Input | Output | C | Requests | Output tok/s | TTFT ms | Decode tok/s/request | P95 E2E s |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        def fmt(v): return '-' if v is None else f'{v:.2f}'
        for r in all_summary:
            if r['suite']!=suite: continue
            lines.append(f'| {r["label"]} | {r["input_tokens"] or "mixed"} | {r["output_tokens"] or "mixed"} | {r["concurrency"]} | {r["measured_requests"]} | {fmt(r["median_output_tokens_per_s"])} | {fmt(r["median_ttft_ms"])} | {fmt(r["median_decode_tokens_per_s"])} | {fmt(r["p95_e2e_s"])} |')
        if suite=='prefill_probe':
            lines+=['','| Engine/layout | Input tokens | One-token request latency ms | Effective input tok/s |','|---|---:|---:|---:|']
            for r in all_summary:
                if r['suite']==suite:
                    lines.append(f'| {r["label"]} | {r["input_tokens"]} | {1000*r["median_e2e_s"]:.2f} | {r["effective_prefill_tokens_per_s"]:.2f} |')
        lines+=['']
    flagged=[r for r in all_summary if r['thermal_slowdown_samples']]
    if flagged:
        lines+=['## Thermal observations','','A sample count combines GPU observations at 1 Hz, not distinct throttling episodes. Measurements are retained as observed.','',
            '| Engine/layout | Suite | Input | C | Flagged GPU samples | Peak C |',
            '|---|---|---:|---:|---:|---:|']
        for r in flagged:
            lines.append(f'| {r["label"]} | {r["suite"]} | {r["input_tokens"] or "mixed"} | {r["concurrency"]} | {r["thermal_slowdown_samples"]} | {r["thermal_peak_c"]} |')
        lines+=['']
    lines+=['## Sampled GPU energy and memory','','GPU board power only, sampled at 1 Hz and integrated over long measured intervals. This excludes the rest of the machine.','',
        '| Engine/layout | Suite | C | Mean total GPU W | Output tok/GPU J |','|---|---|---:|---:|---:|']
    for r in all_summary:
        if r['gpu_joules'] is not None:
            lines.append(f'| {r["label"]} | {r["suite"]} | {r["concurrency"]} | {r["mean_total_gpu_power_w"]:.1f} | {r["output_tokens_per_gpu_joule"]:.4f} |')
    lines+=['','Sampled configuration peak memory by GPU, through the main matrix, including startup and preallocated pools, excluding profiling:','']
    for label,values in memory.items(): lines+=[f'- {label}: {values} MiB; logged KV pool capacity {capacities[label]} tokens.']
    (root/'configuration-summary.json').write_text(json.dumps({'sampled_peak_gpu_memory_mib':memory,'logged_kv_token_capacity':capacities},indent=2)+'\n')
    lines+=['','## Communication baseline','','Python/NCCL dispatch and synchronization are included; model graph replay/custom collectives may differ. Effective bandwidth is payload/time.','',
        '| Operation | Payload KiB | Median us | Effective GB/s |','|---|---:|---:|---:|']
    comm=json.loads((root/'communication.json').read_text())
    for r in comm:
        lines.append(f'| {r["operation"]} | {r["payload_bytes"]/1024:g} | {r["median_us"]:.2f} | {r["effective_payload_GB_s"]:.2f} |')
    with (root/'communication.csv').open('w') as file:
        fields=[k for k in comm[0] if k!='repeat_s']; writer=csv.DictWriter(file,fieldnames=fields,extrasaction='ignore'); writer.writeheader(); writer.writerows(comm)
    if (root/'communication.png').exists(): lines+=['','![Communication](communication.png)']
    graphfile=root/'communication-graph.json'
    if graphfile.exists():
        graph=json.loads(graphfile.read_text())
        assert len(graph)==12 and all(r['gpu_median_us']>0 and len(r['gpu_repeat_s'])==5 for r in graph)
        lines+=['','## Graph-replayed NCCL reductions','','CUDA-event timing of repeated all-reduces in a CUDA graph reduces Python dispatch overhead. This measures standalone NCCL, not engine-specific collectives.','',
            '| Operation | Payload KiB | Median GPU us | Range us |',
            '|---|---:|---:|---:|']
        for r in graph:
            lines.append(f'| {r["operation"]} | {r["payload_bytes"]/1024:g} | {r["gpu_median_us"]:.2f} | {r["gpu_min_us"]:.2f} to {r["gpu_max_us"]:.2f} |')
        with (root/'communication-graph.csv').open('w') as file:
            keys=[k for k in graph[0] if not k.endswith('_repeat_s')]
            writer=csv.DictWriter(file,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(graph)

    sweep=root/'pipeline-sweep.jsonl'
    if not args.allow_partial: assert sweep.exists(), 'Pipeline ablation missing'
    if sweep.exists():
        sweep_rows=[json.loads(line) for line in sweep.read_text().splitlines()]
        expected=[(4,m) for m in [1,2,4,8,16]]+[(16,m) for m in [8,16]]
        validated=[]; sweep_requests=0
        for i,(c,cap) in enumerate(expected):
            trials=sorted([t for t in sweep_rows if t['case_id']==i],key=lambda t:t['trial'])
            if len(trials)!=3:
                assert args.allow_partial, f'Pipeline ablation {i} incomplete'
                continue
            for j,t in enumerate(trials):
                assert t['trial']==j and t['corpus_sha256']==digest
                assert t['case']['concurrency']==c and t['case']['microbatch_cap']==cap
                assert len(t['requests'])==c
                for r in t['requests']:
                    assert r['usage']['prompt_tokens']==r['usage']['completion_tokens']==128
                    assert r['tpot_s']>0 and r['e2e_s']>=r['ttft_s']>0
                assert abs(128*c/t['elapsed_s']-t['aggregate_output_tokens_per_s'])<1e-8
                sweep_requests+=c
            validated.append(summarize(trials[0]['case'],trials,'sglang-tp2-pp2'))
        if not args.allow_partial: assert len(sweep_rows)==21 and sweep_requests==156
        if validated:
            (root/'validated-pipeline-summary.json').write_text(json.dumps(validated,indent=2)+'\n')
            with (root/'pipeline-summary.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(validated[0])); writer.writeheader(); writer.writerows(validated)
        lines+=['','## Pipeline microbatch ablation','',
            f'Additional {len(validated)} cases and {sweep_requests} measured requests on SGLang TP2 × PP2. Each uses 128 input/128 output tokens and three repeats. Two pipeline slots are available; cap 1 limits active admission to two requests. Other tested caps permit the requested concurrency.','',
            'The original four-slot run already defaulted to microbatch size 2. Both engines implement pipeline overlap. A batched throughput gap alone does not establish idle stages: smaller pipeline microbatches also share each weight read across fewer requests.','',
            '| Client C | Microbatch cap | Output tok/s | TTFT ms | Decode tok/s/request |',
            '|---:|---:|---:|---:|---:|']
        for r in validated:
            lines.append(f'| {r["concurrency"]} | {r["microbatch_cap"]} | {r["median_output_tokens_per_s"]:.2f} | {r["median_ttft_ms"]:.2f} | {r["median_decode_tokens_per_s"]:.2f} |')
        if (root/'pipeline.png').exists(): lines+=['','![Pipeline microbatch sweep](pipeline.png)','']
        status=root/'pipeline-profile-status.json'
        if status.exists():
            lines+=['','Separate instrumented profiling diagnostics are excluded from all performance statistics. Status:','']
            for r in json.loads(status.read_text()):
                lines.append(f'- Cap {r["microbatch_cap"]}: '+(r['error'] if 'error' in r else 'profiling API completed; saved traces below.'))
            for path in sorted((root/'profiles').rglob('*')):
                if path.is_file():
                    relative=path.relative_to(root).as_posix()
                    lines.append(f'- [{relative}]({relative}), {path.stat().st_size} bytes.')
    groups=root/'profile-groups.json'
    if groups.exists():
        lines+=['','## Observed pipeline overlap','','Kernel intervals are aligned using the profiler time bases and physical GPU IDs. NCCL and named all-reduce kernels are excluded from compute. Instrumentation changes timing; these traces do not contribute to benchmark speed statistics.','']
        for g in json.loads(groups.read_text()):
            if 'gpu0_gpu2_overlap_fraction_of_compute_union' in g:
                lines.append(f'- {g["profile"]}: physical GPUs 0 and 2 had simultaneous non-communication kernels for {100*g["gpu0_gpu2_overlap_fraction_of_compute_union"]:.1f}% of their compute-interval union. This includes prefill, is not SM occupancy, and does not identify all performance bottlenecks. It does directly show overlapping stage computation.')
            else:
                lines.append(f'- {g["profile"]}: incomplete cross-stage capture, only GPU IDs {g["physical_gpu_ids"]} were saved. The API returned before PP1 export finished, and server shutdown interrupted that export. No overlap comparison is made from this trace.')
        lines+=['','SGLang also logged an automatic merge failure. Per-rank files were analyzed directly. The reproduction script now disables automatic merging and waits for all four complete gzip exports before server shutdown. This fix does not change any saved serving measurements.']
    lines+=['','The original [small-concurrency baseline](../results/report.md) retains its original 4-slot/4096-context/0.90-memory configuration. These extended results use different server capacity and graph limits. Exact commands, environments, server logs, raw request traces and GPU samples accompany this report. The package versions remain in `../results/*-freeze.txt`.','']
    (root/'report.md').write_text('\n'.join(lines))
    print(f'Validated {len(all_summary)} cases / {total_requests} requests. Saved report, CSV and derived summary.')


if __name__=='__main__': main()
