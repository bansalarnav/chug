# Additional Qwen3-32B baseline findings

The campaign recorded 302 workload cases, 18 speculative-decoding trials,
131 component shapes and nine actual transformer-block shapes. The retained
fixed-token streams and 188 natural-prompt input counts passed validation.
The raw logs include deliberate capacity failures; a collected case does not
imply every request succeeded. See [full tables and figures](REPORT.md),
[methodology](METHODOLOGY.md), and [reproduction](REPRODUCE.md).

## BF16 serving

These are output tokens/s including prefill and localhost HTTP overhead, with
128 input and 128 output tokens. Each cell is the median of three warm batches.
The first column uses one request; the second is aggregate throughput across 32.
All configurations below use four RTX A4500 GPUs.

| Engine / split | One request | 32 requests |
|---|---:|---:|
| vLLM TP4 | 28.6 | 529.7 |
| SGLang TP4 | 28.4 | 510.1 |
| TurboMind TP4 | 28.6 | 513.9 |
| llama.cpp tensor | 27.9 | 349.6 |
| llama.cpp layer | 8.9 | 180.5 |

The A-B-B-A repetitions measured 527.8 and 527.3 tok/s for vLLM, versus 513.1
and 514.4 for SGLang at concurrency 32. Those case windows had no sampled thermal
slowdown flags, with maximum GPU temperatures of 82 C. The roughly 2.7% gap was
repeatable here; three batches per cell and one shared machine do not establish
a universal ranking. Longer overload runs did trigger thermal flags.

At eight scheduled requests/s with 1K inputs, TurboMind completed about 204 output
tok/s over the full trace, versus about 169 for vLLM/SGLang. This is a deployment
observation. KV budgets differ, TurboMind uses its native token-ID endpoint, and
these are finite arrival traces with drain time. It is not a matched-memory
kernel-throughput result. vLLM's pool held 21568 tokens; SGLang reported 23041;
llama.cpp's primary pool held 24576. TurboMind used 0.65 of free memory for its
cache allocator, rather than the other engines' total-memory fraction setting.

## Prefix reuse and memory pressure

For a 4K shared prefix with changed suffixes, median warm TTFT was:

| Engine | Cache off, ms | Cache on, ms |
|---|---:|---:|
| vLLM | 1774 | 66.6 |
| SGLang | 1841 | 60.1 |
| TurboMind | 1744 | 92.1 |
| llama.cpp tensor | 2324 | 77.4 |

The 24K llama.cpp pool failed 241 requests in the higher-rate arrival traces.
The separate pressure diagnostic failed all eight concurrent 4K requests, then
successfully served a short recovery request. Increasing the pool to 40960 tokens
completed all 271 Poisson arrivals and all eight pressure requests. The larger
pool requests more memory and is labeled separately. All engines' recorded
abort/recovery tests recovered; oversized requests received client-error statuses.

## Quantization

These comparisons use llama.cpp tensor split on the same four GPUs, with BF16
KV throughout. Speed is for one 128-input/128-output request. Perplexity uses only
four 512-token chunks from the saved WikiText validation slice.

| Weights | Output tok/s | Perplexity | Tool-reported +/- |
|---|---:|---:|---:|
| BF16 | 27.9 | 5.4115 | 0.44784 |
| Q8_0 | 42.0 | 5.4262 | 0.44911 |
| Q4_K_M | 55.6 | 5.5084 | 0.45866 |

The point estimates change by about +0.27% and +1.79%. This small slice and its
uncertainty cannot establish broad quality preservation. Every tested stack
scored 10/12 on the strict answer-format checks; the two misses were JSON code
fences and `No.` instead of bare `no`, rather than incorrect factual answers.

Q4 also fit on one A4500 with a 2048-token cache and four slots, with automatic
fitting disabled and all GPU layers requested. It measured 20.1 output tok/s for
one 1K-input request, versus 45.0 for four-GPU Q4 and 25.1 for four-GPU BF16.
Its different cache, batching limits and GPU count make it a separate deployment
baseline, not a precision-only comparison.

## Speculation and numerical comparison

Prompt lookup reduced copy-task completion time from 18.38 to 4.60 seconds,
about 4x, with identical complete output hashes in all three trials. About 92.9%
of drafted tokens were accepted. Code and prose accepted about 6.0% and 2.2%.
Their output hashes differed from the control, despite stable hashes within each
configuration. Their speed figures are observations on different generated text,
not proof of lossless output equivalence. Full text and counter deltas are saved.

The five numerical fixtures provide 125 raw arrays with verified hashes, shapes
and finite values. Cached and full-prefix reference decoding agreed on all 40
greedy decisions. Maximum centered relative RMS logit error was 3.45%, with
maximum KL 0.00718 nats; uncentered error reached 10.54% because common offsets
also contribute. The actual TP4 block differed from its unsharded reference by
less than 0.3% relative RMS. These are calibration observations, not universal
floating-point tolerances. Use `compare_fixture.py` for your engine's arrays.

## Compatibility limits

Legacy llama.cpp row split failed because the pinned CUDA backend no longer
exports the required split-buffer hook. Tensor and layer modes succeeded.
TensorRT-LLM 1.2.1 imported successfully but its serving launch stalled at MPI
worker creation for 45 minutes. Basic two-rank MPI worked; a direct four-rank
CLI retry instead collided on the HTTP port and did not serve. No TensorRT
speed or accuracy number is reported, and this does not establish hardware
incompatibility for every TensorRT version or launch method.
