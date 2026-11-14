#!/usr/bin/env python3
"""Validate saved measurements and write a portable CSV and Markdown report."""
import argparse
import csv
import json
from pathlib import Path
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path)
args = parser.parse_args()
root = args.results
rows = []
for summary in sorted(root.glob('*-summary.json')):
    cases = json.loads(summary.read_text())
    label = summary.name.removesuffix('-summary.json')
    raw = [json.loads(line) for line in (root / f'{label}.jsonl').read_text().splitlines()]
    assert len(cases) == 6, (label, 'incomplete summary')
    assert len(raw) == 18, (label, 'incomplete trials')
    for case in cases:
        trials = [t for t in raw if t['input_tokens'] == case['input_tokens'] and t['concurrency'] == case['concurrency']]
        assert len(trials) == 3
        for trial in trials:
            assert len(trial['requests']) == case['concurrency']
            for r in trial['requests']:
                assert r['usage']['prompt_tokens'] == case['input_tokens']
                assert r['usage']['completion_tokens'] == 128
                assert r['e2e_s'] >= r['ttft_s'] > 0
                assert r['tpot_s'] > 0
            rate = sum(r['usage']['completion_tokens'] for r in trial['requests']) / trial['elapsed_s']
            assert abs(rate - trial['aggregate_output_tokens_per_s']) < 1e-9
        assert abs(statistics.median(t['aggregate_output_tokens_per_s'] for t in trials) - case['median_aggregate_output_tokens_per_s']) < 1e-9
        rows.append(case)
if not rows:
    raise RuntimeError('No completed results')
corpus_hashes = {t['corpus_sha256'] for rawfile in root.glob('*.jsonl') for t in map(json.loads, rawfile.read_text().splitlines())}
assert len(corpus_hashes) == 1, 'Workloads do not use the same corpus'
with (root / 'summary.csv').open('w') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
model = json.loads((root / 'model.json').read_text())
versions = []
for engine in ['vllm', 'sglang']:
    dependencies = dict(line.split('==', 1) for line in (root / f'{engine}-freeze.txt').read_text().splitlines() if '==' in line)
    versions.append(f'{engine} {dependencies[engine]}, PyTorch {dependencies["torch"]}')
lines = ['# Qwen3-32B BF16 on bly', '', 'Measured on 2026-11-13. ' + '; '.join(versions) + '.', '',
    f'Model commit: `{model["revision"]}`. Corpus SHA256: `{next(iter(corpus_hashes))}`.', '',
    'Four RTX A4500 20 GB GPUs, two NVLink pairs, PCIe between pairs. All requests run on localhost.', '',
    'Each case uses one excluded warm-up batch and three measured batches. Prompts contain synthetic token IDs; each request produces exactly 128 output tokens. Prefix caching is disabled. See the accompanying README for timing definitions and limits.', '',
    '| Engine/layout | Input tokens | Concurrency | TTFT ms | TPOT ms | Decode tok/s per request | Output tok/s, whole batch | Batch tok/s range |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
if (root / 'comparison.png').exists():
    chart_index = next(i for i, line in enumerate(lines) if line.startswith('| Engine/layout'))
    lines[chart_index:chart_index] = ['![Comparison at 1024 input and 128 output tokens](comparison.png)', '']
for r in rows:
    lines.append(f'| {r["label"]} | {r["input_tokens"]} | {r["concurrency"]} | {r["median_ttft_ms"]:.1f} | {r["median_tpot_ms"]:.2f} | {r["median_decode_tokens_per_s"]:.2f} | {r["median_aggregate_output_tokens_per_s"]:.2f} | {r["min_aggregate_output_tokens_per_s"]:.2f} to {r["max_aggregate_output_tokens_per_s"]:.2f} |')
lines += ['', 'Request latency and decode columns show medians across requests. Whole-batch throughput includes prefill and is the median across the three repeats. Decode speed excludes prefill. These quantities answer different questions.', '', '## Runtime observations', '']
for summary in sorted(root.glob('*-summary.json')):
    label = summary.name.removesuffix('-summary.json')
    startup = json.loads((root / f'{label}-startup.json').read_text())['startup_s']
    smoke = json.loads((root / f'{label}-smoke.json').read_text())
    answer = smoke['choices'][0]['text'].strip()
    peaks = {}
    with (root / f'{label}-gpu.csv').open() as file:
        for sample in csv.DictReader(file, skipinitialspace=True):
            gpu = sample['index'].strip()
            try:
                used = float(sample['memory.used [MiB]'].split()[0])
            except (ValueError, KeyError):
                continue
            peaks[gpu] = max(peaks.get(gpu, 0), used)
    server_log = (root / f'{label}-server.log').read_text()
    attention = 'FlashAttention 2' if 'Using FlashAttention version 2' in server_log else 'FlashInfer' if 'Use flashinfer backend by default' in server_log else 'see server log'
    lines += [f'- {label}: attention backend {attention}, startup {startup:.1f} s, sampled peak GPU memory {peaks} MiB, arithmetic smoke answer `{answer}`.']
pipeline_log = root / 'sglang-tp2-pp2-server.log'
if pipeline_log.exists() and 'Disabling breakable prefill CUDA graph by default for pipeline parallelism' in pipeline_log.read_text():
    lines += ['', 'SGLang 0.5.23 automatically disables prefill CUDA graphs for pipeline parallelism in these runs. Decode graphs remain enabled. The optional breakable-prefill graph override was not measured.']
for failure in sorted(root.glob('*-failure.json')):
    label = failure.name.removesuffix('-failure.json')
    if not (root / f'{label}-summary.json').exists():
        lines += [f'- Failed configuration {label}: {json.loads(failure.read_text())["error"]}. See server log.']
lines += ['', 'Package versions are recorded in `vllm-freeze.txt` and `sglang-freeze.txt`. Each configuration has its exact command, server log, startup observation, raw streaming measurements, summary, and GPU samples. Initial startup and compilation are excluded from measured request timings. Startup depends on HDD and OS page-cache state.', '']
(root / 'report.md').write_text('\n'.join(lines))
print(f'Validated {len(rows)} cases; wrote {root / "report.md"} and summary.csv')
