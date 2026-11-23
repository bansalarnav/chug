# Expanded Qwen3-32B baseline

The original report covers 128/1024/2048 input tokens, 128 output tokens,
concurrency 1/4, and four running-request slots. This second dataset keeps the
checkpoint and BF16 precision and expands the workload and server capacity.
Results are saved separately in `extended-results` on both machines. Engines and
layouts run sequentially in the listed order. This is a reproducible workload
sweep, not a randomized benchmark campaign or an exhaustive tuning search.

## Matrix

Each of vLLM TP4, vLLM TP2 × PP2, SGLang TP4, and SGLang TP2 × PP2 runs:

| Suite | Input tokens | Output tokens | Concurrent requests | Measured repeats |
|---|---|---|---|---|
| Prefill probe | 128, 512, 1024, 2048, 4096, 8192 | 1 | 1 | 5 |
| Context/decode | 128, 1024, 4096, 8192 | 256 | 1 | 3 |
| Long decode | 128 | 1024 | 1 | 3 |
| Short-prompt batch scaling | 128 | 128 | 1, 2, 4, 8, 16, 32 | 3 |
| 1K-prompt batch scaling | 1024 | 128 | 1, 2, 4, 8, 16 | 3 |
| 4K-prompt batch scaling | 4096 | 128 | 1, 2, 4 | 3 |
| Closed-loop load | 128 | 128 | 4, 16, 32 | 1 trace, 48/96/192 requests |
| Mixed closed-loop load | 128, 1024, 4096 | 128, 256 | 8 | 1 trace, 64 requests |

There are 29 cases and 748 measured requests per engine/layout, excluding warm-up.
All cases receive an excluded warm-up. The mixed trace repeats input lengths
128/1024/4096/128/1024/4096/128/1024; every fourth request generates 256 tokens,
and the others generate 128. Its job order is deterministic. In closed-loop
runs a client worker submits a new request as soon as its previous one completes.
The client keeps at most the configured concurrency in flight, then drains the
remaining requests after the fixed job count is submitted.

Concurrency is the number of in-flight HTTP requests. The engine chooses its
actual model batches and pipeline microbatches; those choices appear in its
server logs. These are serving measurements. They do not force one static GPU
batch size. Capacity is capped at 32, so this sweep does not establish the
hardware's absolute maximum throughput beyond that capacity.

## Configuration and timing

- Same pinned Qwen/Qwen3-32B commit as the original dataset.
- Same installed engine/runtime stacks, including the CUDA 13.0 compiler fix.
- BF16 weights and KV, no quantization, offload, prefix reuse, or speculation.
- Context cap 9216; running-request cap 32; prefill token budget/chunk 2048.
- Memory fraction 0.88 for both engines, leaving room for larger graph captures.
- SGLang decode capture cap 32 and prefill capture cap 2048 tokens.
- SGLang retains its automatic prefill-graph disable for pipeline parallelism.
- The corpus preserves the original first 16 prompts at the shared lengths and
  adds deterministic prompts for the larger workloads. Corpus SHA256 is recorded.
- Greedy sampling and EOS ignored. Every response must report exact token counts.

Prefill probes use a nonstreaming completion with one output token. The reported
latency includes prompt processing, one output token, and local HTTP overhead.
Input tokens divided by this latency is *effective prompt throughput*, rather
than a kernel-only prefill measurement.

Streaming TTFT ends at the first nonempty text chunk. In this expanded dataset,
mean token interval runs from that first chunk through completion of the stream,
divided by output_tokens minus one. Special-token text is retained to avoid
invisible forced-EOS tails in long generations. This differs slightly from the
original last-visible-chunk endpoint; full response throughput is unaffected by
that metric definition. Unicode buffering and HTTP finalization remain part of
the streaming estimate. Request usage supplies exact token counts.

Batch output throughput includes prefill and all requests in the batch. Closed-loop
throughput includes initial filling and final draining. These finite traces
provide tail-latency observations, not confidence intervals or a guaranteed
steady-state capacity. P95 is reported only for cases with at least 20 observed
requests. Low-sample cases retain medians and all raw repeats.

GPU board power is sampled once per second. Approximate GPU joules and output
tokens per GPU joule are calculated only for the long-decode and closed-loop/mixed
traces lasting at least 10 seconds, with interpolated interval boundaries. All request work, including prefill, is charged to the output tokens. These
exclude CPU, system RAM, host chassis fans and HDD power. Sampled NVML memory includes reserved
pools, contexts and graphs; it is not the number of live KV bytes. No clocks,
power caps or fan settings are changed. Some SGLang TP4 cases reported thermal
slowdown during sustained load. Thermal-reason sampling began after the vLLM
throughput runs, so no-throttling claims cannot be made for those earlier cases.
Per-case thermal coverage and flags are included in the CSV. Treat small engine
speed differences as confounded by temperature and clock behavior.

## Communication measurements

`comm.py` uses PyTorch/NCCL on all four GPUs, with five timing repeats per payload:

- Two independent all-reduce groups within the NVLink pairs.
- One four-GPU all-reduce group.
- Point-to-point send/receive within a pair and across the two pairs.

Payloads span 1 KiB to 64 MiB, including 10 KiB, 40 KiB, 160 KiB, 320 KiB and
20 MiB activation-sized messages. Results include Python dispatch and CUDA
synchronization. They are not isolated link or device-kernel latencies. Effective
payload GB/s is payload bytes divided by elapsed time, not NCCL ring bus bandwidth.
The models' CUDA graphs and custom reduction implementations can behave differently.

TP4 keeps weight shards in local VRAM and processes a layer on all four GPUs.
TP2 × PP2 assigns successive layer groups to the two pairs. A single decode request
must traverse both stages in sequence. Small activation reductions across PCIe
can cost less than that loss of simultaneous GPU work. More independent requests
or prefill chunks may allow the pipeline to overlap work; the batching and long
prompt measurements test those cases.

## Run or resume

On bly, after the original setup:

```bash
export BENCH_ROOT=/path/allocated/to/arnavb9/qwen3-baseline
source "$BENCH_ROOT/scripts/env.sh"
"$BENCH_ROOT/envs/vllm/bin/python" "$BENCH_ROOT/scripts/run_extended.py"
```

The runner uses the original exclusive run lock and runs one engine/layout at a
time. Completed cases/trials can be resumed with the same configuration. Keep the
original run separate if changing settings. The exact launch, environment, logs,
per-request metrics, timestamps, corpus, GPU samples and case matrix are saved.

## Pipeline scheduling ablation

The installed SGLang 0.5.23 scheduler defaults its maximum microbatch size to
`max_running_requests // pp_size`. The original four-slot PP2 run therefore
allowed microbatches of two; the expanded 32-slot run defaults to 16. The PP loop
uses `pp_size + pp_async_batch_depth` slots, with async depth zero here. Its source
already implements work overlap. Original decode logs show two-request microbatches.
The installed vLLM config likewise supports PP-size concurrent batches, or PP-size
plus one with its V2 asynchronous scheduler. These are scheduling capabilities,
not proof of a particular amount of overlap in a measured request.

The earlier batched TP4/PP2 gap must not be attributed entirely to idle stages.
At fixed total concurrency, splitting requests into smaller pipeline microbatches
also reduces how many requests share each read of a weight shard. Memory-bound
matrix operations can stay busy while completing fewer tokens per weight read.
Communication, stage balance, graph choices and scheduling overhead also matter.

After the main matrix, SGLang PP2 receives seven additional tests without a model
reload. Its supported internal-state API changes the microbatch cap to 1/2/4/8/16
at client concurrency 4, and 8/16 at concurrency 16. All use 128 input/128 output
tokens, warm-up, and three repeats. This adds 156 measured requests. Cap 1 with two
pipeline slots limits active capacity to two and is an explicit queueing control;
the other cases have enough aggregate slots for the requested client concurrency.
The default cap is restored afterward.

Two separate instrumented traces are attempted at concurrency 4 with caps 2 and
16. CPU/GPU profiling overhead excludes those diagnostic requests from every
performance statistic. The profiler's status and saved trace files record whether
capture succeeded. A stage-overlap claim requires examining those traces rather
than relying on NVML utilization, which can count communication waits as busy.

## Compare a future engine

Use a new result directory and copy the saved corpus into it before running the
client. This keeps the token inputs byte-for-byte identical and lets the client
run with Python's standard library, without loading PyTorch or Transformers.
For example, on bly with the future server listening on localhost port 19090:

```bash
export BENCH_ROOT=/path/allocated/to/arnavb9/qwen3-baseline
source "$BENCH_ROOT/scripts/env.sh"
mkdir -p "$BENCH_ROOT/rust-engine-results"
cp "$BENCH_ROOT/extended-results/corpus.json" "$BENCH_ROOT/rust-engine-results/corpus.json"
read -r checkpoint_path < "$BENCH_ROOT/results/model-path.txt"
python3 "$BENCH_ROOT/scripts/extended.py" \
  --label rust-engine --port 19090 \
  --out "$BENCH_ROOT/rust-engine-results" \
  --model-path "$checkpoint_path" \
  --original-corpus "$BENCH_ROOT/results/corpus.json"
```

The server must accept token-ID completions, greedy sampling, ignored EOS, retained
special-token text, streaming usage, and the lengths/concurrency in the matrix.
The resulting per-case summary uses the same metric names as these baselines.
For an initial engine limited to short contexts and four requests, use `bench.py`
with a copy of the original `results/corpus.json` and compare to the original
small-concurrency report. Correctness/logit comparisons remain a separate task.

A second communication test, `comm_graph.py`, captures repeated NCCL reductions
in CUDA graphs and times graph replay with CUDA events. It warms device clocks,
measures six activation-sized payloads in the two NVLink groups and the TP4 group,
and preserves five repeats. This reduces the Python dispatch floor in the first
communication test. It measures standalone NCCL collectives, not the engine's
custom all-reduce or the end-to-end cost of all communication in a model layer.
The first communication test's similar small-message latencies therefore do not
prove that the physical links have similar device latency.

The captured cap-2 profile contains all four GPU ranks and directly shows
simultaneous model-kernel execution across the two stages. `profile-groups.json`
records the interval-union calculation. The cap-16 capture saved only PP0 before
shutdown interrupted PP1 export; it cannot support a cross-stage comparison.
SGLang's automatic merge also failed. Per-rank cap-2 traces remain complete.
The reproduction script now disables automatic merging and waits for four
complete gzip exports before shutdown. Neither diagnostic trace contributes to
any reported serving-speed number.
