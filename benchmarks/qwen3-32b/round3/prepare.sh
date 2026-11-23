#!/usr/bin/env bash
set -euo pipefail
source "$BENCH_ROOT/scripts/env.sh"
export R3="$BENCH_ROOT/round3"
mkdir -p "$R3"/{results,scripts,sources,fixtures,models}
case "${1:?component}" in
lmdeploy)
 uv venv --python 3.12 "$BENCH_ROOT/envs/lmdeploy"
 uv pip install --python "$BENCH_ROOT/envs/lmdeploy/bin/python" lmdeploy==0.18.0
 uv pip freeze --python "$BENCH_ROOT/envs/lmdeploy/bin/python" > "$R3/results/lmdeploy-freeze.txt"
 "$BENCH_ROOT/envs/lmdeploy/bin/lmdeploy" serve api_server --help > "$R3/results/lmdeploy-help.txt"
 ;;
reference)
 uv venv --python 3.12 "$BENCH_ROOT/envs/reference"
 uv pip install --python "$BENCH_ROOT/envs/reference/bin/python" torch==2.14.0 transformers==4.57.3 accelerate safetensors sentencepiece gguf cmake ninja
 uv pip freeze --python "$BENCH_ROOT/envs/reference/bin/python" > "$R3/results/reference-freeze.txt"
 ;;
trt)
 uv venv --python 3.12 "$BENCH_ROOT/envs/tensorrt"
 uv pip install --python "$BENCH_ROOT/envs/tensorrt/bin/python" tensorrt-llm==1.2.1 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
 uv pip install --python "$BENCH_ROOT/envs/tensorrt/bin/python" openmpi==5.0.11 pyarrow==20.0.0
 uv pip install --python "$BENCH_ROOT/envs/tensorrt/bin/python" torch==2.9.1+cu130 torchvision==0.24.1+cu130 --index-url https://download.pytorch.org/whl/cu130
 export CUDA_HOME="$BENCH_ROOT/envs/sglang/lib/python3.12/site-packages/nvidia/cu13"
 export PATH="$CUDA_HOME/bin:$BENCH_ROOT/envs/tensorrt/bin:$PATH"
 export LD_LIBRARY_PATH="$BENCH_ROOT/envs/tensorrt/lib"
 for trt_libdir in "$BENCH_ROOT/envs/tensorrt/lib/python3.12/site-packages/nvidia/"*/lib; do
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$trt_libdir"
 done
 printf '%s\n' "$LD_LIBRARY_PATH" > "$R3/results/tensorrt-library-path.txt"
 uv pip freeze --python "$BENCH_ROOT/envs/tensorrt/bin/python" > "$R3/results/tensorrt-freeze.txt"
 "$BENCH_ROOT/envs/tensorrt/bin/python" -c 'import tensorrt_llm; print(tensorrt_llm.__version__)' > "$R3/results/tensorrt-import.txt" 2>&1
 ;;
llama-source)
 git clone --depth 1 https://github.com/ggml-org/llama.cpp "$R3/sources/llama.cpp"
 git -C "$R3/sources/llama.cpp" fetch --depth 1 origin 61dbf49228c8a96b72e40c89d93cb7c003411f8c
 git -C "$R3/sources/llama.cpp" checkout --detach 61dbf49228c8a96b72e40c89d93cb7c003411f8c
 git -C "$R3/sources/llama.cpp" rev-parse HEAD > "$R3/results/llama-commit.txt"
 ;;
esac
