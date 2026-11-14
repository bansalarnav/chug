#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
umask 077
mkdir -p "$BENCH_ROOT"/{cache,tmp,envs,results,models}
if [[ ! -x "$BENCH_ROOT/envs/vllm/bin/python" ]]; then
  uv venv --python 3.12 "$BENCH_ROOT/envs/vllm"
fi
uv pip install --python "$BENCH_ROOT/envs/vllm/bin/python" vllm==0.33.0
"$BENCH_ROOT/envs/vllm/bin/vllm" download-kernels
uv pip freeze --python "$BENCH_ROOT/envs/vllm/bin/python" > "$BENCH_ROOT/results/vllm-freeze.txt"
"$BENCH_ROOT/envs/vllm/bin/python" "$(dirname "$0")/download.py"
if [[ ! -x "$BENCH_ROOT/envs/sglang/bin/python" ]]; then
  uv venv --python 3.12 "$BENCH_ROOT/envs/sglang"
fi
uv pip install --python "$BENCH_ROOT/envs/sglang/bin/python" --prerelease=allow sglang==0.5.23 \
  "nvidia-cuda-nvcc>=13.0,<13.1" "nvidia-cuda-crt>=13.0,<13.1" \
  "nvidia-nvvm>=13.0,<13.1" "nvidia-nvjitlink>=13.0,<13.1" \
  "nvidia-cuda-cccl>=13.0,<13.1"
uv pip freeze --python "$BENCH_ROOT/envs/sglang/bin/python" > "$BENCH_ROOT/results/sglang-freeze.txt"
