# MLX conversion and quantization smoke (2026-10-24)

**Scope:** local Apple Silicon serving only. No model training was run locally or as part of
this smoke. The purpose was to verify that the new MLX backend and quantized loader work before
a Colab-trained ANT head is available for the real accuracy/ECE gate.

**Machine:** Apple M3, 16 GB unified memory. **Runtime:** Python 3.13, `mlx 0.32.4`,
`mlx-lm 0.32.2`, torch 2.14.0.

## Commands

```bash
python scripts/mlx_convert.py --hf-path models/Qwen3.5-0.8B \
  --mlx-path models/Qwen3.5-0.8B-mlx --no-quantize

python scripts/mlx_convert.py --hf-path models/Qwen3.5-0.8B \
  --mlx-path models/Qwen3.5-0.8B-mlx-8bit --bits 8 --group-size 64

python scripts/mlx_convert.py --hf-path models/Qwen3.5-0.8B \
  --mlx-path models/Qwen3.5-0.8B-mlx-4bit --bits 4 --group-size 64
```

## Results

| Model folder | Disk size | Loader | Hidden states |
|---|---:|---|---|
| HF source `models/Qwen3.5-0.8B` | 1.6 GB | `MlxBackbone`, `quantization=None` | finite |
| `models/Qwen3.5-0.8B-mlx` | 1.4 GB | `MlxBackbone`, `quantization=None` | reference |
| `models/Qwen3.5-0.8B-mlx-8bit` | 782 MB | `MlxBackbone`, `quantization="mlx"` | finite |
| `models/Qwen3.5-0.8B-mlx-4bit` | 424 MB | `MlxBackbone`, `quantization="mlx"` | finite |

A two-text hidden-state comparison against the unquantized MLX folder:

| Variant | mean \\|Δh\\| | max \\|Δh\\| | pooled cosine |
|---|---:|---:|---:|
| 8-bit | 0.371 | 5.57 | 0.9953 |
| 4-bit | 2.226 | 20.84 | 0.7436 |

## Interpretation

- Conversion, MLX hidden-state extraction and the quantized load path all work.
- 8-bit is numerically plausible and roughly halves the model folder size.
- 4-bit is much lighter but shows large representation drift on this tiny smoke; treat it as
  experimental until the real quality gate passes.
- These are **hidden-state diagnostics, not decision-quality results**. A quantized variant may
  replace the unquantized one only after `scripts/compare_quantization.py` passes against the
  same Colab-trained ANT checkpoint and the same multitask rows.

## Follow-up

1. Train the 0.8B ANT head on Colab.
2. Run the MLX unquantized arm and the 8-bit arm through `scripts/compare_quantization.py`.
3. Report accuracy, ECE, argmax flips and per-row probability deltas before enabling 8-bit.
4. Keep 4-bit disabled unless it passes the same gate.
