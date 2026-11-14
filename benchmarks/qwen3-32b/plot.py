#!/usr/bin/env python3
"""Plot the 1024-input/128-output comparison from saved summaries.

Run locally with: uv run --with matplotlib python plot.py RESULTS_DIRECTORY
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path)
args = parser.parse_args()
labels = ['vllm-tp4-pp1', 'vllm-tp2-pp2', 'sglang-tp4-pp1', 'sglang-tp2-pp2']
names = ['vLLM · TP4', 'vLLM · TP2 × PP2', 'SGLang · TP4', 'SGLang · TP2 × PP2']
colors = ['#2459a6', '#2459a6', '#b85617', '#b85617']
data = {}
for label in labels:
    path = args.results / f'{label}-summary.json'
    if path.exists():
        data[label] = json.loads(path.read_text())
if set(data) != set(labels):
    raise RuntimeError('All four engine/layout results are required')
fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.3), layout='constrained')
fig.suptitle('Qwen3-32B BF16 · four RTX A4500s\n1,024 input tokens, 128 output tokens', fontsize=14)
for ax, concurrency, metric, title in [
    (axes[0], 1, 'median_decode_tokens_per_s', 'Single request: decode tokens/s, excluding prefill'),
    (axes[1], 4, 'median_aggregate_output_tokens_per_s', 'Four concurrent requests: total output tokens/s, including prefill')]:
    for i, (label, name, color) in enumerate(zip(labels, names, colors)):
        if label not in data:
            continue
        row = next(r for r in data[label] if r['input_tokens'] == 1024 and r['concurrency'] == concurrency)
        value = row[metric]
        errors = None
        if concurrency == 4:
            errors = [[value-row['min_aggregate_output_tokens_per_s']], [row['max_aggregate_output_tokens_per_s']-value]]
        ax.barh(i, value, color=color, height=.58, hatch='//' if 'tp2' in label else None,
            xerr=errors, capsize=3, error_kw={'elinewidth': 1})
        label_x = row['max_aggregate_output_tokens_per_s'] + .8 if concurrency == 4 else value + .5
        ax.text(label_x, i, f'{value:.1f}', va='center', fontsize=11)
    ax.set_yticks(range(4), names)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlim(right=ax.get_xlim()[1]*1.10)
    ax.set_title(title, fontsize=11, loc='left', pad=12)
    ax.set_xlabel('Tokens per second')
    ax.grid(axis='x', alpha=.18)
    ax.set_axisbelow(True)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0, labelsize=10)
fig.supxlabel('Medians from three runs. Whiskers show the observed batch-throughput range.', fontsize=9)
fig.savefig(args.results / 'comparison.png', dpi=160, facecolor='white')
fig.savefig(args.results / 'comparison.svg', facecolor='white')
print(args.results / 'comparison.png')
