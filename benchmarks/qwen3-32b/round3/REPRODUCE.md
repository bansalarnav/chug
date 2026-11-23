# Reproduce on allocated storage

Use a directory allocated to your account with roughly 250 GB free. The former
benchmark tree under another user's HDD directory has been deleted.
The scripts require an explicit `BENCH_ROOT` and do not search for writable
locations. Model weights and environments are not part of the retained archive.

From a checkout on bly, set an authorized path and copy the small inputs/scripts:

```bash
export BENCH_ROOT=/path/allocated/to/arnavb9/qwen3-baseline
export BENCH_SOURCE="$PWD/benchmarks/qwen3-32b"
mkdir -p "$BENCH_ROOT"/{scripts,results,extended-results,round3/scripts,round3/results,round3/fixtures,round3/models,round3/sources}
cp "$BENCH_SOURCE"/*.py "$BENCH_SOURCE"/*.sh "$BENCH_ROOT/scripts/"
cp "$BENCH_SOURCE"/round3/*.py "$BENCH_SOURCE"/round3/*.sh "$BENCH_ROOT/round3/scripts/"
cp "$BENCH_SOURCE/results/model.json" "$BENCH_ROOT/results/"
cp "$BENCH_SOURCE/extended-results/corpus.json" "$BENCH_ROOT/extended-results/"
cp "$BENCH_SOURCE/extended-results/"*-tp4-pp1-command.json "$BENCH_ROOT/extended-results/"
cp "$BENCH_SOURCE/round3/fixtures/quality-data.json" "$BENCH_SOURCE/round3/fixtures/wikitext-validation.txt" "$BENCH_ROOT/round3/fixtures/"
bash "$BENCH_ROOT/scripts/setup.sh"
source "$BENCH_ROOT/scripts/env.sh"
```

Keep the retained freeze files outside the new output directory before setup;
setup writes new freeze files. `setup.sh` pins top-level versions, while the
retained `results/*-freeze.txt` and `round3/results/*-freeze.txt` specify the exact
resolved environments used here. Recreating those exact environments can require
the CUDA 13 PyTorch wheel index and NVIDIA package index shown in `prepare.sh`.
The copied `model.json` pins the original checkpoint instead of resolving HEAD.
The server runner replaces historical executable/checkpoint paths in copied
launch commands with the new root and downloaded checkpoint path.

Prepare the additional environments and pinned llama.cpp source, then build:

```bash
for component in reference lmdeploy trt llama-source; do
  bash "$BENCH_ROOT/round3/scripts/prepare.sh" "$component"
done
bash "$BENCH_ROOT/round3/scripts/build_llama.sh"
bash "$BENCH_ROOT/round3/scripts/convert_llama.sh"
bash "$BENCH_ROOT/round3/scripts/components.sh"
```

Run the serving suites and addons sequentially on idle GPUs:

```bash
"$BENCH_ROOT/envs/reference/bin/python" "$BENCH_ROOT/round3/scripts/run.py"
"$BENCH_ROOT/envs/reference/bin/python" "$BENCH_ROOT/round3/scripts/addons.py"
"$BENCH_ROOT/envs/reference/bin/python" "$BENCH_ROOT/round3/scripts/report.py"
```

The addons run TurboMind through its native token-ID endpoint, then speculation
and TensorRT compatibility, followed by GGUF cache/capacity diagnostics,
quantization, perplexity and quantized serving. This groups weight formats while
retaining fresh server launches and the A-B-B-A order. Completed stages can be
resumed; inspect JSONL request errors and stage failures before interpreting the
results. A completed workload file means observations were collected, including
intentional capacity failures. It does not mean every request succeeded.

For figures, install matplotlib in a separate client environment or run `plot.py`
with `uv run --no-project --with matplotlib`. Retain the original timezone in
`results/timezone.txt` when analyzing old GPU CSVs; write the new server timezone
there for a fresh run. Never mix measurements from different configurations under
one label.

`storage_cleanup.py` is specific to this campaign's original tree, inode and
session marker. It is not a reproduction step. It verifies the home archive and
all numerical fixture hashes before it can delete that exact original tree.
