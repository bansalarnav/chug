#!/usr/bin/env python3
"""Expanded fixed-corpus serving benchmarks. Prefix reuse stays disabled."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from functools import partial
from bench import batch as base_batch, request as base_request
batch = partial(base_batch, complete_stream_timing=True)
request = partial(base_request, complete_stream_timing=True)


def one_token(port, prompt):
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=600)
    connection.connect()
    start = time.perf_counter()
    try:
        body = json.dumps(dict(model='qwen3-32b', prompt=prompt, max_tokens=1,
            temperature=0, ignore_eos=True, stream=False))
        connection.request('POST', '/v1/completions', body, {'Content-Type':'application/json'})
        response = connection.getresponse()
        data = json.loads(response.read())
        end = time.perf_counter()
        if response.status != 200:
            raise RuntimeError(data)
        usage = data['usage']
        assert usage['prompt_tokens']==len(prompt) and usage['completion_tokens']==1, usage
        return dict(input_tokens=len(prompt), output_tokens=1, e2e_s=end-start,
            ttft_s=None, tpot_s=None, decode_tokens_per_s=None, usage=usage,
            output_sample=data['choices'][0]['text'])
    finally:
        connection.close()


def percentile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    x = (len(values)-1)*q
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi]-values[lo])*(x-lo)


def closed_loop(port, jobs, concurrency):
    start = time.perf_counter()
    def run(job):
        prompt, output = job
        submitted = time.perf_counter()
        row = request(port, prompt, output)
        row['client_start_after_s'] = submitted-start
        row['client_finish_after_s'] = time.perf_counter()-start
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(run, jobs))
    elapsed = time.perf_counter()-start
    return dict(elapsed_s=elapsed, requests=rows,
        aggregate_output_tokens_per_s=sum(r['output_tokens'] for r in rows)/elapsed,
        aggregate_total_tokens_per_s=sum(r['input_tokens']+r['output_tokens'] for r in rows)/elapsed)


def cases():
    result = []
    def add(suite, lengths, output, concurrencies, repeats=3):
        for length in lengths:
            for c in concurrencies:
                result.append(dict(suite=suite, input_tokens=length, output_tokens=output,
                    concurrency=c, repetitions=repeats, mode='burst', requests=c))
    add('prefill_probe', [128,512,1024,2048,4096,8192], 1, [1], 5)
    add('context_decode', [128,1024,4096,8192], 256, [1])
    add('long_decode', [128], 1024, [1])
    add('batch_short', [128], 128, [1,2,4,8,16,32])
    add('batch_1k', [1024], 128, [1,2,4,8,16])
    add('batch_4k', [4096], 128, [1,2,4])
    for c, n in [(4,48),(16,96),(32,192)]:
        result.append(dict(suite='closed_loop', input_tokens=128, output_tokens=128,
            concurrency=c, requests=n, repetitions=1, mode='closed_loop'))
    result.append(dict(suite='mixed', input_tokens=None, output_tokens=None,
        concurrency=8, requests=64, repetitions=1, mode='closed_loop'))
    return result


def summarize(case, trials, label):
    rows = [r for t in trials for r in t['requests']]
    def p(key,q):
        return percentile([r.get(key) for r in rows],q)
    result = dict(case, label=label, measured_requests=len(rows),
        median_e2e_s=p('e2e_s',.5),
        p95_e2e_s=p('e2e_s',.95) if len(rows)>=20 else None,
        median_ttft_ms=None if p('ttft_s',.5) is None else 1000*p('ttft_s',.5),
        p95_ttft_ms=None if len(rows)<20 or p('ttft_s',.95) is None else 1000*p('ttft_s',.95),
        median_tpot_ms=None if p('tpot_s',.5) is None else 1000*p('tpot_s',.5),
        p95_tpot_ms=None if len(rows)<20 or p('tpot_s',.95) is None else 1000*p('tpot_s',.95),
        median_decode_tokens_per_s=p('decode_tokens_per_s',.5),
        median_output_tokens_per_s=statistics.median(t['aggregate_output_tokens_per_s'] for t in trials),
        min_output_tokens_per_s=min(t['aggregate_output_tokens_per_s'] for t in trials),
        max_output_tokens_per_s=max(t['aggregate_output_tokens_per_s'] for t in trials),
        elapsed_s=sum(t['elapsed_s'] for t in trials))
    if case['suite']=='prefill_probe':
        result['effective_prefill_tokens_per_s'] = case['input_tokens']/result['median_e2e_s']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18080)
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--original-corpus', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    corpus_path = args.out/'corpus.json'
    if not corpus_path.exists():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        vocab = sorted(i for i in set(tokenizer.get_vocab().values())-set(tokenizer.all_special_ids) if 1000<=i<30000)
        rng = random.Random(2026111401)
        original = json.loads(args.original_corpus.read_text())
        corpus = {}
        for n in [128,512,1024,2048,4096,8192]:
            prompts = original.get(str(n), []).copy()
            while len(prompts)<128:
                prompts.append([rng.choice(vocab) for _ in range(n)])
            corpus[str(n)] = prompts
        corpus_path.write_text(json.dumps(corpus))
    corpus = json.loads(corpus_path.read_bytes())
    digest = hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    (args.out/'matrix.json').write_text(json.dumps(cases(),indent=2)+'\n')
    raw_path = args.out/f'{args.label}.jsonl'
    previous = [json.loads(line) for line in raw_path.read_text().splitlines()] if raw_path.exists() else []
    summaries = []
    with raw_path.open('a') as file:
        for number, case in enumerate(cases()):
            trials = [r for r in previous if r['case_id']==number]
            assert len(trials)<=case['repetitions']
            def jobs(trial):
                count=case['requests']
                if case['suite']=='mixed':
                    mix=[128,1024,4096,128,1024,4096,128,1024]
                    return [(corpus[str(mix[i%8])][(4+i)%128], 256 if i%4==0 else 128) for i in range(count)]
                return [(corpus[str(case['input_tokens'])][(4+trial*case['concurrency']+i)%128],case['output_tokens']) for i in range(count)]
            if len(trials)<case['repetitions']:
                warm=jobs(0)[:case['concurrency']]
                if case['suite']=='prefill_probe':
                    one_token(args.port, warm[0][0])
                elif case['mode']=='burst':
                    batch(args.port,[p for p,o in warm],case['output_tokens'])
                else:
                    closed_loop(args.port,warm,case['concurrency'])
            for repeat in range(len(trials),case['repetitions']):
                current=jobs(repeat)
                wall_started_at=time.time()
                if case['suite']=='prefill_probe':
                    row=one_token(args.port,current[0][0])
                    result=dict(elapsed_s=row['e2e_s'],requests=[row],
                        aggregate_output_tokens_per_s=1/row['e2e_s'],
                        aggregate_total_tokens_per_s=(row['input_tokens']+1)/row['e2e_s'])
                elif case['mode']=='burst':
                    result=batch(args.port,[p for p,o in current],case['output_tokens'])
                else:
                    result=closed_loop(args.port,current,case['concurrency'])
                result.update(wall_started_at=wall_started_at,wall_finished_at=time.time(),label=args.label,case_id=number,case=case,trial=repeat,corpus_sha256=digest)
                file.write(json.dumps(result)+'\n'); file.flush()
                trials.append(result)
                print(f'{args.label} {number+1}/{len(cases())} {case["suite"]} in={case["input_tokens"]} out={case["output_tokens"]} c={case["concurrency"]} repeat={repeat} tok/s={result["aggregate_output_tokens_per_s"]:.2f}',flush=True)
            summaries.append(summarize(case,trials,args.label))
            (args.out/f'{args.label}-summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    if args.label=='sglang-tp2-pp2':
        subprocess.run([sys.executable,str(Path(__file__).with_name('pipeline_sweep.py')),
            '--port',str(args.port),'--out',str(args.out)],check=True)
    print(f'COMPLETE {args.label}: {len(summaries)} cases',flush=True)


if __name__=='__main__':
    main()
