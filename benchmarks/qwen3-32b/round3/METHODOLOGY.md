# Round three: implementation and serving references

During measurement, remote artifacts were under
`/srv/local/bj2/faro/results2/arnavb9-qwen3-baseline/round3` on bly.
That entire benchmark tree has now been deleted. The verified result archive is
`/home/arnavb9/qwen3-32b-benchmark-records/final.tar.gz` on bly and
`bly-records.tar.gz` beside this file in the checkout.
The original BF16 checkpoint revision remains pinned in `../results/model.json`.
The existing benchmark reports remain separate.

## Numerical reference

`reference.py` loads Qwen3-32B with Transformers 4.57.3, BF16 weights, SDPA
attention, and an automatic balanced placement across four GPUs. This is a
correctness fixture generator, not a serving-speed benchmark. It saves five
prompts covering arithmetic, code, Unicode, factual retrieval, and longer context.
Thinking is disabled through the official tokenizer chat template.

Each fixture includes token IDs, final logits, eight teacher-forced continuation
steps, and the last eight positions from hidden-state indices 0, 1, 16, 32, 48,
and 64. Indices follow Transformers conventions: 0 is the embedding output;
64 is the final normalized hidden state. Arrays are saved both in compressed
NumPy files and as little-endian float32/int64 binaries with shape/dtype/hash
manifests for Rust readers. Values capture BF16 execution, exported in float32.

Cached incremental logits are compared with full-prefix recomputation at every
continuation step. The full-prefix argmax chooses the common continuation so
that both paths always see the same token history. Cosine similarity, relative
RMS error, maximum error, and greedy-token agreement are retained. Finite outputs
are required. There is no claim of bit equality or a universal tolerance.
The initial run stopped at a provisional 2% relative-RMS threshold after observing
2.85% error with unchanged argmax. That diagnostic log is preserved; the complete
run records both arrays and all observed differences rather than weakening a
hidden pass criterion. No broad model-quality claim follows from these checks.

## Components

`kernels.py` measures 131 shapes on GPU 0 with allocated BF16 operands, CUDA graph
replay, CUDA-event timing, and five repeats. GEMMs use Qwen3's actual dimensions
for QKV, attention output, gate/up, and down projections, at TP shard factors
1/2/4 and token counts 1/4/16/32/128/512/2048. Other cases cover vLLM RMSNorm,
SiLU/multiply, RoPE, reference KV writes, and PyTorch Flash SDPA prefill/decode.
Random weights are scaled by inverse square root of the input dimension.
RoPE uses a precomputed identity cosine/sine cache to avoid accumulating drift
when its in-place operation is replayed. This measures kernel work, not the
construction of that cache. KV writes use PyTorch index_copy as a reference,
not vLLM's paged-cache update kernel. Approximate effective bandwidth includes
operand/output traffic estimates; GEMMs also report effective FLOP/s.

The first KV-write check inspected an uninitialized unwritten region and recorded
one assertion failure. `kv_recheck.py` repeats all seven KV-write shapes with a
zero-initialized buffer and exact equality checks on written rows. The original
measurements are retained in `kernels-initial.json` and corrected rows are marked.

`layer.py` loads the actual first transformer block and shards its weights across
four workers. It uses BF16 cuBLAS/Flash SDPA, unfused FP32-accumulating norms and
RoPE, and NCCL reductions. It compares each sharded output with an unsharded
block, then times graph replay across nine prompt/decode/batch shapes. Decode
history KV is synthetic and identical before head sharding; this is not a
full-model correctness check. Its provisional 3% relative-RMS diagnostic gate
is reported with the actual errors. Library/kernel fusion differs from serving
engines, so multiplying its latency by 64 is only a rough implementation target.

## Extra engines and reproducibility

LMDeploy 0.18.0 uses TurboMind. llama.cpp is pinned by its recorded source commit,
compiled for SM86 with CUDA and NCCL, and receives a BF16 GGUF converted from the
same checkpoint. CUDA libraries and tools are installed in user-owned HDD
virtual environments. Conversion/build time is excluded. Existing vLLM 0.33.0
and SGLang 0.5.23 environments are retained. Separate freeze files and launch
commands document each stack. TensorRT-LLM 1.2.1 is a compatibility probe first;
its runtime failures are preserved, rather than counted as performance results.

The serving runner tests original BF16 TP4 configurations, TurboMind, and
llama.cpp tensor/layer/legacy-row split modes. Row split is deprecated upstream
and is included only as a requested comparison. llama.cpp receives BF16 K/V
caches explicitly. Its primary unified cache has 24576 token slots shared by
32 request slots, with an 8192-token per-slot limit. Row split uses an 8192-token
pool because its KV resides on the main GPU. These actual memory budgets differ
from the other engines and are recorded. All weights are requested on GPU;
startup logs must be checked for any automatic CPU fallback before comparison.

For vLLM this round uses lazy safetensor loading to reduce HDD staging pressure.
It changes loading behavior only. Warm request timing excludes loading.

## Workloads

The fixed-token speed subset uses 128/1024/4096 input tokens and 128 output tokens
at selected concurrency levels through 32. It reads the previous extended corpus,
uses an excluded warm-up and three repeats, and validates exact token usage.
Greedy generation ignores EOS. Streaming special-token text is retained.
Mean token interval runs from first visible text through stream completion,
divided by output_tokens minus one. Whole-request throughput includes prefill
and HTTP/client work on localhost.

Open-loop tests use identical seeded Poisson schedules with target rates
1/2/4/8 requests/s over a 20-second arrival horizon, 1024 input/128 output tokens,
and up to 256 client threads. Arrivals continue independently of completion.
Arrival-to-finish includes dispatch lag and all server/client queueing. All
requests are drained; failed requests, dispatch lag, and raw per-request times
are retained. Finite-trace throughput includes fill/drain. P99 is descriptive
and omitted below 100 completed observations. A 64-request simultaneous burst is
an additional queueing stress case. This is not a certified production SLO.

Real-prompt tests use self-authored explanation, code, translation and queueing
prompts plus a four-turn conversation. They generate naturally up to 192 tokens.
Prefix-reuse tests use shared prefixes of 256/1024/4096 tokens and distinct
32-token suffixes, with one cold request and three warm requests. Cache-off
controls run in the base conversation suite; cache-on controls run separately.
Engine metrics are captured. llama.cpp receives cache_prompt explicitly on each
request so that slot reuse does not silently enter the cache-off baseline.
Where reset APIs are unavailable, unique leading nonces distinguish prefixes.

Cancellation closes eight streams during prefill, then eight during decoding,
with metrics before/after and a four-request recovery test. GPU pools may remain
reserved; active-cache and running-request metrics are the relevant cleanup
observations. Oversized inputs must reject without crashing. Eight concurrent
4K prompts exercise constrained KV capacity, followed by a recovery request.
Both expected rejection and actual recovery are recorded, not inferred from
resident GPU memory alone.

Each engine is launched from idle GPUs and cooled below 50 C before its suite.
GPU power, clocks, temperature, memory and thermal-slowdown flags are sampled at
1 Hz throughout. A small vLLM/SGLang subset is repeated in A-B-B-A order with the
same cooling criterion. This is temperature-controlled starting state, not fixed
GPU clocks or identical thermals throughout long overload traces. No power caps,
fan settings, or clocks are changed. The shared machine and finite repeat counts
limit small engine-to-engine conclusions.

## Quantization and speculation

llama.cpp quantizes the same BF16 GGUF to Q8_0 and Q4_K_M. Primary comparisons keep
four GPUs and nonquantized BF16 KV; a separate Q4 single-GPU configuration probes
practical memory fit with a smaller cache and four request slots. Its throughput
must not be attributed solely to precision because its deployment layout differs.

Perplexity uses a pinned WikiText-2-raw validation slice, context 512 and four
chunks. Dataset revision, input hash, commands and output logs are retained.
This is a small relative degradation check, supplemented by twelve exact-answer
sanity tasks, not a comprehensive quality evaluation or a guarantee on coding
and reasoning ability. Output text is retained so grading can be inspected.

Prompt-lookup speculation uses vLLM's ngram proposer, lookup sizes 2..5 and four
speculative tokens, against a BF16 control with the same configuration. Copying,
code and prose prompts each generate 512 tokens with three repeats. Full output
hashes, per-request timing, acceptance counters and the exact-answer checks are
saved. This does not measure a learned draft model; speedup is workload-dependent.

## Client compatibility and interrupted launch

The first vLLM text-task attempt encountered a Transformers API difference:
its installed tokenizer returned BatchEncoding by default. The clients now
request `return_dict=False` explicitly. Those initial client failures are saved
separately; the same live server reruns the affected cases after the arrival
trace drains. Timing workloads never share a server with this repair.

I paused the first vLLM launch for the requested storage audit after five
speed cases. They remain valid measured batches. Earlier GPU/startup logs are
retained in `results/attempts/storage-pause`; remaining cases resume after a
fresh launch and cooldown. The later A-B-B-A repetitions use uninterrupted
launches. The user explicitly asked to finish the current tests in place and
then delete the entire benchmark tree from the other user's HDD directory.

SGLang's initial BF16 suite used its default disabled Prometheus endpoint.
Cancellation recovery is validated there, but no cache-release conclusion is
made from missing counters. Remaining SGLang launches enable metrics explicitly
and save native server status too. The setup restart happened during an untimed
TurboMind load; that attempt's logs are retained separately.

LMDeploy 0.18.0's OpenAI completion route interprets a flat integer prompt as a
batch of inputs and rejects each integer during preprocessing. The initial
36-case attempt is invalid and archived. TurboMind measurements instead use its
native `/generate` route with `input_ids`. The benchmark client normalizes its
delta text and cumulative token counts into the same timing/count schema.
Natural completions, cancellation and oversized requests use the native route
as well. Exact prompt IDs, generated counts and the GPU configuration stay the
same. HTTP frontend/serialization overhead can differ; these are localhost
serving comparisons, not isolated engine-device latency measurements.

The original llama.cpp context-pressure batch hit HTTP 500 when eight 4K
contexts exceeded its 24576-token shared pool. The first batch helper raised
before the recovery request. That failure remains visible. A separate
`llama-pressure-recovery` diagnostic records each of the eight outcomes and
attempts recovery even if stress requests fail. Abort tests also retain individual
errors and always attempt recovery. The diagnostic enables verbose startup
logging for actual placement; its timing is not included in serving-speed rows.

The pinned llama.cpp fit code refuses to change a user-specified GPU-layer count
in `common/fit.cpp`, and tensor split does not implement automatic fitting.
All three BF16 launches explicitly request 99 GPU layers. Default verbosity 3
suppresses backend INFO placement messages, so missing counters in those logs
are not interpreted as CPU fallback. Launch arguments and per-GPU allocation
remain available. Quantized serving comparisons additionally disable fitting;
the single-GPU Q4 case uses batch/ubatch limits 512/128 to leave memory for its
2048-token BF16 cache.


Legacy row split fails at the pinned llama.cpp commit because its CUDA registry
no longer exports `ggml_backend_split_buffer_type`, while the row loader requires
that hook. The source excerpt and actual launch failure are retained. Tensor
and layer split succeed. This is a compatibility result for this commit, not
an assertion that every historical llama.cpp CUDA build lacks row split.

The 24576-token llama.cpp pool can hold fewer than 32 simultaneous 1024+128-token
contexts. Arrival cases retain per-request errors, and latency summaries are
conditional on success. Completion fractions are plotted separately. A separate
`llama-bf16-tensor-40k` deployment uses a 40960-token pool, 32 slots and the same
8192-token per-slot limit, BF16 weights/KV and four GPUs. Its arrival/pressure
cases test extra capacity without replacing the original constrained results.
It requests more GPU memory, so it is not a matched-budget speed comparison.

If the TensorRT PyTorch backend runs successfully, its compatibility deployment
also receives four fixed-token speed cases at concurrency 1/4 and 128/1024 input,
plus the twelve answer checks. It has four request slots and context 4096;
these measurements are explicitly labeled as a four-slot configuration rather
than a tuned counterpart to the 32-slot serving engines.
