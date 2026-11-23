# Chug project plan

This document records the project's intent and the reasoning behind its initial
direction. It is a reference for the author and future LLM conversations, not a
fixed specification or an instruction to start implementing the whole plan.

## Ownership and working style

The author plans to write the implementation themselves. LLMs can help explain
concepts, examine designs, interpret measurements, review code, and maintain
documentation when asked. Do not assume this plan authorizes writing engine
code, installing software, changing the remote machine, or running benchmarks.
Follow the author's current request.

Keep the implementation understandable and the decisions backed by measurements.
Reusable components matter, but interfaces should evolve with working models.
The exact crate layout, public API, worker process model, and kernel toolchain
remain implementation decisions.

## What we want to build

Build an LLM inference engine in Rust with CUDA GPU execution. It should load
weights and perform inference itself. Existing engines are references and
benchmark baselines, rather than the backend behind a Rust wrapper.

The original motivation was efficient inference for DeepSeek V4.1 Flash or
GLM-5.3-Flash. Those remain possible later targets. First build a useful engine
for a substantial model on hardware already available, with reusable support
for tensor parallelism, pipeline parallelism, and additional architectures.

## Initial target

- Model: `Qwen/Qwen3-32B`.
- Weights and KV state: BF16 initially, with appropriate higher-precision
  accumulation in numerical operations.
- Hardware: all four RTX A4500 GPUs on `bly`.
- Execution: multi-GPU from the first full-model milestone.
- Initial workload: one request, short context, ordinary autoregressive decoding.
  Start with a 2K–4K total context budget and validate actual memory use.
- First interface: a CLI. HTTP serving follows once inference works.

Qwen3-32B provides a conventional dense transformer on which to develop loading,
attention, caching, distributed execution, and scheduling. Small models or small
synthetic layers are useful test fixtures; they are not the main product target.
Quantization is a later optimization, rather than a prerequisite for the first
correct implementation.

The existing benchmark campaign pins checkpoint revision
`9216db5781bf21249d130ec9da846c4624c16137`. Reuse that revision for matched
comparisons unless deliberately establishing a new baseline.

## Available machine

Read-only inspection on 2026-11-13 established:

- Host: SSH alias `bly`, `bly.cs.illinois.edu`.
- CPU: Intel Core i9-10900X, 10 cores / 20 threads.
- RAM: nominal 128 GB. Available RAM varies because the machine is shared.
- GPUs: four RTX A4500s, each reporting 20,470 MiB VRAM, compute capability 8.6,
  and a 200 W power limit.
- Topology: GPUs 0/1 are an NVLink pair; GPUs 2/3 are another NVLink pair.
  Cross-pair communication uses PCIe. Every pair supports peer reads and writes.
  All-to-all P2P capability does not mean an all-to-all NVLink fabric.
- Platform: ASUS WS X299 SAGE, one NUMA node, RHEL 9.9.
- Storage: approximately 2 TB NVMe and 4 TB SATA HDD. At inspection, NVMe-backed
  `/srv` had only about 12 GB free; HDD-backed `/srv/local` had about 2.6 TB free.

The initial inspection found driver 595.71.05 and CUDA compatibility 13.2, but no
toolkit or Rust compiler on the noninteractive PATH. Subsequent benchmark work
created tool environments; consult the retained setup logs and recheck current
availability rather than treating either observation as permanent. Driver CUDA
compatibility alone does not establish toolkit availability.

Use storage explicitly allocated to the author's account. The benchmark README
records cleanup of an earlier location inside another user's directory. Do not
reuse that location or infer authorization from writable permissions. Recheck
free space before downloading weights or producing converted copies.

## Architecture direction

### Parallelism is reusable infrastructure

Implement generic TP and PP building blocks. A topology configuration assigns
devices to groups and layers to stages. The model supplies the rules that make
its tensors and operations partitionable.

Support these initial layouts through the same machinery:

- TP4 / PP1: one tensor-parallel group spanning all GPUs.
- TP2 / PP2: stage 0 on GPUs 0/1, stage 1 on GPUs 2/3, with TP inside each pair.

For TP2 / PP2, frequent TP reductions stay within the NVLink pairs. Pipeline
transfers cross PCIe. This layout is an explicit deployment choice, not a
special-case model implementation. PP also needs scheduling to overlap useful
work; it does not automatically accelerate one autoregressive request.

Use reusable operations such as column-parallel linear layers, row-parallel
linear layers, head-partitioned attention, and vocabulary-parallel embeddings
and output projections. Keep transport and collectives behind a communication
interface, initially backed by NCCL where appropriate.

Make stage boundaries explicit: hidden states, residual state where required,
and necessary metadata. Define whether transferred tensors are replicated or
sharded. Preserve collective ordering and stream dependencies across workers.

### Model adapters compose those operations

A model implementation owns configuration interpretation, checkpoint mapping,
layer construction, forward execution, and its partitioning constraints. It
should reuse parallel layers and communication rather than implementing its own
distributed runtime.

Separate these responsibilities:

- Model execution and model-specific weight loading.
- Device buffers, GPU operations, streams, and kernel dispatch.
- Parallel groups, rank placement, collectives, and pipeline transfers.
- Session state and its allocation/lifecycle.
- Request scheduling and memory admission.
- Tokenization, conversation formatting, sampling, and output parsing.

Session state must allow more than a conventional KV cache. Future architectures
may need recurrent state, sliding windows, or other persistent data. A new model
may require new operations or kernels; an adapter cannot eliminate that work.

Begin with compiled-in Rust model implementations. Dynamic plugins and a general
graph compiler are not initial requirements. Validate the boundaries with a
second architecture before declaring the adapter API stable.

### Rust and CUDA implementation

Use Rust for the runtime and resource ownership. Use established CUDA libraries
such as cuBLAS for the initial matrix multiplications. Evaluate cuda-oxide for
custom kernels, starting with a small operation before relying on it broadly.
CUDA C++ kernels and existing attention implementations are also acceptable.

Keep kernel selection separate from model semantics. Replacing a normalization
or attention implementation should not require redesigning a model adapter.
Pin and validate toolchains on the actual A4500 hardware. Avoid making progress
depend on experimental compiler support or newer-GPU-only instructions.

## Existing evidence and references

The Rust executable is currently a skeleton. Significant benchmark and reference
work already exists; read it before repeating experiments:

- [Baseline setup and methodology](benchmarks/qwen3-32b/README.md).
- [Expanded measurements](benchmarks/qwen3-32b/extended-results/report.md), including
  TP4 versus TP2 / PP2 and communication/pipeline experiments.
- [Additional findings](benchmarks/qwen3-32b/round3/FINDINGS.md),
  [methodology](benchmarks/qwen3-32b/round3/METHODOLOGY.md), and
  [reproduction instructions](benchmarks/qwen3-32b/round3/REPRODUCE.md).
- [Numerical fixtures](benchmarks/qwen3-32b/round3/fixtures/README.md) and the
  associated comparison script.

Recorded results favor TP4 for the measured decode workloads, while TP2 / PP2
helps some long-prefill cases. Do not select a layout solely from link topology.
Retain both options and use the workload to choose a default.

The reports also document thermal effects, differing memory budgets, capacity
failures, and compatibility limits. Their timings are observations under stated
conditions, not performance guarantees. Numerical differences in the fixtures
are calibration evidence, not universal error tolerances.

## Milestones

### 1. Establish the execution foundation

Confirm the current tool environment, allocated storage, model revision, and
GPU rank mapping. Reuse existing communication measurements where applicable.
Implement device ownership, streams/events, GPU buffers, and parallel groups.
Validate collectives and one small custom kernel on the target hardware.

Done when allocations, transfers, and collectives work reliably with explicit
ordering, error reporting, and cleanup.

### 2. Validate one distributed transformer block

Load the required checkpoint slices and implement normalization, Q/K
normalization, rotary embeddings, attention, feed-forward execution, and residual
handling. Exercise TP sharding and a stage-transfer boundary using real tensor
shapes. Avoid materializing unnecessarily large attention intermediates.

Done when intermediate results agree with reference fixtures within justified
numerical tolerances, including checks that catch shard-ordering mistakes.

### 3. Generate with the full Qwen3-32B model

Load weights incrementally into their destination shards without retaining
multiple full host copies. Implement prefill, KV allocation, cached decode,
sharded embeddings/output projection, greedy sampling, and EOS handling.
Preallocate reusable buffers and use bounded prefill chunks.

Done when the CLI generates useful text across all four GPUs, memory fits the
configured context budget, and cached decoding passes numerical comparisons.
Support TP4 and TP2 / PP2 before calling distributed execution complete.

### 4. Make single-request inference reliable and measurable

Add conversation formatting, sampling controls, streaming, cancellation, and
clear capacity errors. Measure startup separately from warm execution. Profile
kernel time, communication, CPU scheduling, and synchronization before tuning.

Done when repeated multi-turn sessions work and matched benchmark runs report
TTFT, decode speed, prompt processing, and peak per-GPU memory use.

### 5. Add scheduling and serving

Add a thin HTTP interface, multiple sessions, memory admission, and continuous
batching. Develop pipeline microbatch scheduling and chunked prefill as explicit
policies. Introduce a cache allocator that can support batching and later prefix
reuse without exposing allocation details to callers.

Done when mixed-length workloads, cancellation, overload, and recovery behave
predictably. Compare latency and throughput at matched precision and memory
budgets, not just aggregate tokens per second.

### 6. Test extensibility and grow toward the original models

Add a second architecture to validate the adapter interfaces. Then choose a
smaller MoE target to develop routing, grouped expert execution, and expert
placement before attempting GLM-5.3-Flash or DeepSeek V4.1 Flash.

Pursue quantization, prefix reuse, session persistence, CUDA graphs, kernel
fusion, speculative decoding, and offloading as measurements justify them.
Select actual weight formats and later models when reaching those milestones;
their tooling and availability may change.

## Correctness and performance principles

- Plausible generated text is not a correctness test. Compare token IDs,
  intermediate states, logits, and cached versus full-prefix execution.
- Account for legitimate floating-point differences between execution layouts.
  Investigate deviations rather than requiring universal bitwise equality.
- Give distributed operations deterministic ordering; a request abort must not
  leave another rank waiting forever in a collective.
- Keep model weights, state, scratch space, and communication buffers in explicit
  per-GPU budgets. Aggregate VRAM is not one automatic allocation pool.
- Compare equivalent prompts, output lengths, precision, cache settings, and
  concurrency. Separate kernel timings from end-to-end serving measurements.
- Record temperature and clock behavior on the shared machine. Repeat tests only
  when a change or unresolved uncertainty calls for them.

## Lessons from other projects

- [vLLM](https://github.com/vllm-project/vllm) and
  [SGLang](https://github.com/sgl-project/sglang): reusable parallel layers,
  model-aware checkpoint loading, explicit stage boundaries, and scheduling
  outside the model implementation.
- [DS4 / DwarfStar](https://github.com/antirez/ds4): narrow model support,
  co-designed weight formats and kernels, selective quantization, and bounded
  expert caching for local hardware.
- [DeepInfer](https://github.com/deepseek-ai/DeepInfer): model-specific execution
  optimizations, scheduling overlap, reproducibility, and later speculation.
  Its deployment assumptions should not dictate this project's hardware needs.
- [Mooncake](https://github.com/kvcache-ai/Mooncake): explicit state placement,
  lifecycle, and transfer costs. Start with local state management; distributed
  cache infrastructure is a later need.
- [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe): a candidate
  Rust component for DeepSeek prompt and output protocol handling later.

Study and reuse appropriate components with their license obligations. The goal
is a comprehensible Rust/CUDA engine that improves through evidence, with useful
multi-GPU 32B inference as its first substantial result.
