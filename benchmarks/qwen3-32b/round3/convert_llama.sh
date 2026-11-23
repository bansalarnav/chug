#!/usr/bin/env bash
set -euo pipefail
source "$BENCH_ROOT/scripts/env.sh"
R3="$BENCH_ROOT/round3"
read -r checkpoint_path < "$BENCH_ROOT/results/model-path.txt"
"$BENCH_ROOT/envs/reference/bin/python" "$R3/sources/llama.cpp/convert_hf_to_gguf.py" "$checkpoint_path" \
 --outfile "$R3/models/Qwen3-32B-BF16.gguf" --outtype bf16
