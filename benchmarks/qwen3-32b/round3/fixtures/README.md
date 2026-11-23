# Numerical fixtures

Each prompt has a directory with `manifest.json` and 25 raw binary arrays.
Integer values are little-endian int64; floating-point values are little-endian
float32 exported after BF16 execution. Shapes and SHA256 hashes are in the
manifest. The corresponding NPZ contains the same values.

`reference.json` records the plain and formatted prompt, exact input token IDs,
teacher-forced continuation IDs, checkpoint, versions and comparison metrics.
Hidden states use Transformers indexing: 0 is embeddings; 64 is final norm.
Each hidden-state array stores the final eight available prompt positions.
No padding is added. Full-prefix and cached logits share the same teacher-forced
history. Centering logits removes common offsets that softmax ignores.

To compare your engine, write arrays with matching names and shapes plus a
manifest with `file`, `dtype`, and `shape` entries. Then run from round3:

```bash
uv run --no-project --with numpy python compare_fixture.py \
  fixtures/code /path/to/your/code --out /path/to/comparison.json
```

Missing arrays, shape mismatches, nonfinite floats or unequal integer arrays
cause a nonzero exit. Floating-point differences are reported without an
arbitrary universal cutoff. Use the measured cached/full-prefix differences
and block comparisons as calibration, then investigate larger deviations.
These five prompts and eight continuation steps per prompt are a limited
correctness reference, not proof of numerical equivalence on every input.
