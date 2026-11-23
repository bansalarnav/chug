# Qwen3-32B baseline on bly

The [expanded benchmark](EXTENDED.md) adds 8K prompts, long generation, concurrency
up to 32, continuous-load traces, GPU energy estimates, communication measurements
and a pipeline microbatch sweep. Its results are in `extended-results/report.md`.

The [additional findings](round3/FINDINGS.md) and [baseline report](round3/REPORT.md) adds numerical fixtures,
component timings, TurboMind and llama.cpp, arrival-rate traces, cancellation,
prefix reuse, quantization, speculative decoding and controlled repeats.
Its [methodology](round3/METHODOLOGY.md) records compatibility limits and fixes.

This benchmark compares BF16 Qwen/Qwen3-32B on bly's four RTX A4500 GPUs.
Measurements were collected on 2026-11-13. Each engine uses its resolved dependency stack, recorded in full in the results.
It runs vLLM and SGLang sequentially with TP4 and TP2 × PP2. GPUs 0/1 and
2/3 are NVLink pairs; the two pairs communicate over PCIe.

The initial HDD location was inside another user's tree and should not have
been used. At the user's request, the campaign finished there, then
the entire benchmark tree was deleted after its results were archived and verified. Future
runs require an explicit `BENCH_ROOT` in storage allocated to your account.
World-writable permissions do not establish authorization.

## Workload

- Original checkpoint, BF16 weights and BF16 KV cache, no quantization or CPU offload.
- Maximum context 4096, at most four running requests, prefill chunks of 2048 tokens.
- vLLM uses eager per-file weight loading for the HDD. SGLang uses its single-threaded loader. These settings affect startup only.
- Prefix caching disabled. No speculative model. CUDA graphs and engine optimizations enabled.
- Synthetic prompts of exactly 128, 1024, or 2048 token IDs.
- Exactly 128 generated tokens, greedy sampling, EOS ignored.
- Concurrency 1 or 4. All requests in each batch start together.
- One excluded warm-up batch per shape, followed by three measured batches.
- The corpus uses a fixed random seed. All engines/layouts read the same corpus file.
- Requests use the localhost streaming `/v1/completions` endpoint. No SSH transport is timed.
- Every request must report the expected prompt and completion token counts.
- A separate simple arithmetic prompt records a text-generation smoke check.

These are synthetic execution baselines. They do not measure answer quality,
chat-template overhead, prefix reuse, long conversations, or maximum throughput
at high concurrency. Random token prompts may produce unnatural continuations.

TTFT is time from submitting the HTTP request to receiving the first nonempty
text chunk. TPOT is time between the first and last nonempty text chunks divided
by 127. Decode tokens/s is its reciprocal. UTF-8 buffering and empty token text
can affect those streaming estimates. Exact token counts come from API usage.

Aggregate output tokens/s is total completed output tokens divided by batch
wall time, including prefill, decode, and local client overhead. Each summary
reports the median across three batch repeats and the observed min/max.
Request latency summaries pool requests across those repeats. Concurrency 4
is a finite batch, not a steady-state arrival-rate benchmark.

SGLang 0.5.23 automatically disables prefill CUDA graphs for pipeline parallelism
with these settings; decode graphs remain enabled. This baseline retains that
default. The optional breakable-prefill override is not measured.

Startup times include model loading and compilation and are reported separately.
The HDD and OS page cache can affect startup, so startup figures are observations,
not a controlled disk-performance comparison. The machine is shared; GPU samples
and hardware metadata accompany the results. No clock or power limits are changed.

## Reproduce

Copy the scripts into `$BENCH_ROOT/scripts`, then run on bly:

```bash
export BENCH_ROOT=/path/allocated/to/arnavb9/qwen3-baseline
bash "$BENCH_ROOT/scripts/setup.sh"
source "$BENCH_ROOT/scripts/env.sh"
python3 "$BENCH_ROOT/scripts/run.py"
```

`env.sh` requires an explicit BENCH_ROOT. Export it before the first command or
invoke the setup script by absolute path. `setup.sh` creates Python 3.12 environments
with uv. Initial installation pins vLLM 0.33.0 and SGLang 0.5.23 and resolves their published dependencies; preserve and use the
resulting `results/*-freeze.txt` files to recreate the exact measured dependencies.
`results/model.json` pins the checkpoint commit on first download. The same commit
is reused on subsequent downloads.

SGLang uses CUDA 13.0 compiler components to match its runtime headers. The runner
sets CUDA_HOME and adds the environment executables to PATH. It also adds the
standard lib64 and libcudart linker symlinks inside that environment. Compilation
is limited to four jobs. `flashinfer_smoke.py` checks a small BF16 attention call
without loading the model.

The runner writes launch commands, complete server logs, per-request streaming
measurements, JSON summaries, GPU samples, startup times, and failures under
`$BENCH_ROOT/results`. It takes a file lock and runs one engine/layout at a time.
It skips complete summaries and refuses to overwrite partial measurement files.
Servers bind to localhost. Only process groups started by this runner are stopped.

For a new engine, implement the same completions request fields and usage counts,
then call `bench.py` against its localhost port and reuse `results/corpus.json`.
Compare single-request decode speed and TTFT first, then concurrent throughput.
Use the recorded prompts, lengths, precision, layout, and memory limits in each
comparison. Kernel-only timings should be reported separately from these HTTP
measurements.

## Sources

Installation follows the official [vLLM GPU instructions](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
and [SGLang installation instructions](https://github.com/sgl-project/sglang/blob/main/docs/docs/get-started/install.mdx).
The installed package help and actual launch logs determine the flags and backend
used for the measurements.
