#!/usr/bin/env bash
set -euo pipefail
source "$BENCH_ROOT/scripts/env.sh"
export R3="$BENCH_ROOT/round3"
export CUDA_HOME="$BENCH_ROOT/envs/sglang/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$BENCH_ROOT/envs/reference/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$BENCH_ROOT/envs/sglang/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
if [[ ! -e "$CUDA_HOME/lib64" ]]; then ln -s lib "$CUDA_HOME/lib64"; fi
if [[ ! -e "$CUDA_HOME/lib/libcudart.so" ]]; then ln -s libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"; fi
for library in cublas cublasLt curand; do
 if [[ ! -e "$CUDA_HOME/lib/lib$library.so" ]]; then
  versioned=$(find "$CUDA_HOME/lib" -maxdepth 1 -name "lib$library.so.*" | head -1)
  if [[ -n "$versioned" ]]; then ln -s "$(basename "$versioned")" "$CUDA_HOME/lib/lib$library.so"; fi
 fi
done
cmake -S "$R3/sources/llama.cpp" -B "$R3/sources/llama.cpp/build" -G Ninja \
 -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 \
 -DCUDAToolkit_ROOT="$CUDA_HOME" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
 -DGGML_CUDA_NCCL=ON -DNCCL_INCLUDE_DIR="$BENCH_ROOT/envs/sglang/lib/python3.12/site-packages/nvidia/nccl/include" \
 -DNCCL_LIBRARY="$BENCH_ROOT/envs/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2" \
 -DLLAMA_CURL=OFF -DGGML_CCACHE=OFF
cmake --build "$R3/sources/llama.cpp/build" -j 4 --target llama-server llama-bench llama-quantize llama-perplexity
"$R3/sources/llama.cpp/build/bin/llama-server" --help > "$R3/results/llama-help.txt"
