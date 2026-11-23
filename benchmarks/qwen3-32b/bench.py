#!/usr/bin/env python3
"""Identical localhost OpenAI-completions workload for either engine.

Synthetic random token IDs measure execution rather than model quality. Each
request must return exactly the requested token counts or the run fails.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import random
import statistics
import threading
import time
from pathlib import Path


def request(port, prompt, output, barrier=None, complete_stream_timing=False, request_fields=None, protocol="openai"):
    payload = dict(model='qwen3-32b', prompt=prompt,
        max_tokens=output, temperature=0, ignore_eos=True, stream=True,
        skip_special_tokens=not complete_stream_timing,
        stream_options={'include_usage': True})
    payload.update(request_fields or {})
    endpoint = '/v1/completions'
    if protocol == 'lmdeploy':
        endpoint = '/generate'
        payload['input_ids'] = payload.pop('prompt')
        payload.pop('model', None)
        payload.pop('stream_options', None)
    body = json.dumps(payload)
    connection = http.client.HTTPConnection('127.0.0.1', port, timeout=600)
    connection.connect()
    if barrier:
        barrier.wait(timeout=60)
    start = time.perf_counter()
    first = last = None
    usage = None
    chunks = []
    text_parts = []
    try:
        connection.request('POST', endpoint, body, {'Content-Type':'application/json'})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f'HTTP {response.status}: {response.read().decode()}')
        done = False
        while line := response.readline():
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                done = True
                break
            event = json.loads(data)
            if event.get('error'):
                raise RuntimeError(event['error'])
            if protocol == 'lmdeploy' and 'meta_info' in event:
                meta = event['meta_info']
                event = dict(choices=[dict(text=event.get('text',''))], usage={k:meta[k] for k in ['prompt_tokens','completion_tokens']})
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                text = choice.get('text', '')
                if text:
                    now = time.perf_counter()
                    first = now if first is None else first
                    last = now
                    chunks.append(now - start)
                    text_parts.append(text)
        end = time.perf_counter()
        if not done or first is None or usage is None:
            raise RuntimeError(f'Incomplete stream: done={done}, first={first}, usage={usage}')
        if usage['prompt_tokens'] != len(prompt) or usage['completion_tokens'] != output:
            raise RuntimeError(f'Token count mismatch: {usage}, expected {len(prompt)}/{output}')
        tpot = ((end if complete_stream_timing else last) - first) / (output - 1)
        return dict(input_tokens=len(prompt), output_tokens=output, transport_protocol=protocol,
            ttft_s=first-start, e2e_s=end-start, tpot_s=tpot,
            decode_tokens_per_s=1/tpot,
            stream_chunk_times_s=chunks, content_chunks=len(chunks), usage=usage,
            output_sha256=hashlib.sha256(''.join(text_parts).encode()).hexdigest(),
            output_sample=''.join(text_parts)[:300], output_text=''.join(text_parts))
    finally:
        connection.close()


def batch(port, prompts, output, complete_stream_timing=False, request_fields=None, protocol="openai"):
    barrier = threading.Barrier(len(prompts))
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(request, port, p, output, barrier, complete_stream_timing, request_fields, protocol) for p in prompts]
        rows = [f.result() for f in futures]
    elapsed = time.perf_counter() - start
    return dict(elapsed_s=elapsed,
        aggregate_output_tokens_per_s=sum(r['output_tokens'] for r in rows)/elapsed,
        aggregate_total_tokens_per_s=sum(r['input_tokens']+r['output_tokens'] for r in rows)/elapsed,
        requests=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=18080)
    parser.add_argument('--label', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    if not args.corpus.exists():
        # Choose ordinary vocabulary entries, excluding added/special tokens.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        vocab = sorted(set(tokenizer.get_vocab().values()) - set(tokenizer.all_special_ids))
        vocab = [i for i in vocab if 1000 <= i < 30000]
        rng = random.Random(20261113)
        corpus = {str(n): [[rng.choice(vocab) for _ in range(n)] for _ in range(16)]
                  for n in [128, 1024, 2048]}
        args.corpus.write_text(json.dumps(corpus))
    corpus_bytes = args.corpus.read_bytes()
    corpus = json.loads(corpus_bytes)
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / f'{args.label}.jsonl'
    if raw.exists():
        raise RuntimeError(f'Refusing to overwrite {raw}')
    summary = []
    with raw.open('w') as logfile:
        for length in [128, 1024, 2048]:
            for concurrency in [1, 4]:
                # Warm every shape and batch size before collecting measurements.
                batch(args.port, corpus[str(length)][:concurrency], 128)
                trials = []
                for trial in range(args.repetitions):
                    offset = 4 + trial * concurrency
                    prompts = corpus[str(length)][offset:offset+concurrency]
                    result = batch(args.port, prompts, 128)
                    result.update(label=args.label, input_tokens=length,
                        output_tokens=128, concurrency=concurrency, trial=trial,
                        corpus_sha256=hashlib.sha256(corpus_bytes).hexdigest())
                    logfile.write(json.dumps(result) + '\n')
                    logfile.flush()
                    trials.append(result)
                    print(f'{args.label}: input={length} c={concurrency} trial={trial} '
                          f'output_tok/s={result["aggregate_output_tokens_per_s"]:.2f}', flush=True)
                requests = [r for t in trials for r in t['requests']]
                summary.append(dict(label=args.label, input_tokens=length,
                    output_tokens=128, concurrency=concurrency,
                    repetitions=args.repetitions, requests=len(requests),
                    median_ttft_ms=1000*statistics.median(r['ttft_s'] for r in requests),
                    median_tpot_ms=1000*statistics.median(r['tpot_s'] for r in requests),
                    median_decode_tokens_per_s=statistics.median(r['decode_tokens_per_s'] for r in requests),
                    median_e2e_s=statistics.median(r['e2e_s'] for r in requests),
                    median_aggregate_output_tokens_per_s=statistics.median(t['aggregate_output_tokens_per_s'] for t in trials),
                    min_aggregate_output_tokens_per_s=min(t['aggregate_output_tokens_per_s'] for t in trials),
                    max_aggregate_output_tokens_per_s=max(t['aggregate_output_tokens_per_s'] for t in trials)))
                (args.out / f'{args.label}-summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
