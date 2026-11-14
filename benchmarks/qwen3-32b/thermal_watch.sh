#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
watched_pid=${1:?benchmark runner PID required}
nvidia-smi --query-gpu=timestamp,index,temperature.gpu,fan.speed,clocks.sm,power.draw,clocks_event_reasons.active,clocks_event_reasons.sw_power_cap,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.hw_thermal_slowdown --format=csv --loop-ms=1000 > "$BENCH_ROOT/extended-results/thermal.csv" &
sensor_pid=$!
trap 'kill "$sensor_pid" 2>/dev/null || true' EXIT INT TERM
while kill -0 "$watched_pid" 2>/dev/null; do sleep 2; done
