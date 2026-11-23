#!/usr/bin/env bash
set -uo pipefail
source "$BENCH_ROOT/scripts/env.sh"
R3="$BENCH_ROOT/round3"
export PATH="$BENCH_ROOT/envs/reference/bin:$PATH"
"$BENCH_ROOT/envs/vllm/bin/python" "$R3/scripts/kernels.py" > "$R3/results/kernels.log" 2>&1
printf 'kernels exit %s\n' "$?" >> "$R3/results/component-status.txt"
"$BENCH_ROOT/envs/reference/bin/torchrun" --standalone --nproc-per-node=4 "$R3/scripts/layer.py" > "$R3/results/layer.log" 2>&1
printf 'layer exit %s\n' "$?" >> "$R3/results/component-status.txt"
"$BENCH_ROOT/envs/reference/bin/python" "$R3/scripts/reference.py" > "$R3/results/reference.log" 2>&1
printf 'reference exit %s\n' "$?" >> "$R3/results/component-status.txt"

"$BENCH_ROOT/envs/reference/bin/python" "$R3/scripts/enrich_reference.py" > "$R3/results/enrich-reference.log" 2>&1
printf 'enrich-reference exit %s\n' "$?" >> "$R3/results/component-status.txt"
"$BENCH_ROOT/envs/vllm/bin/python" "$R3/scripts/kv_recheck.py" > "$R3/results/kv-recheck.log" 2>&1
printf 'kv-recheck exit %s\n' "$?" >> "$R3/results/component-status.txt"
