# ANT-08B — lightweight Adaptive Neural Triage

ANT-08B is the small, fast variant of the ANT candidate-scoring decision engine, built on
[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B). It keeps the same
`score(state, question, candidate) -> scalar` contract and calibrated execute/escalate policy as
the 4B project, but is intended for local Apple Silicon serving and lower-latency decisions.

The backbone is frozen and used as an encoder: no LM head and no `model.generate(...)` in the
inference path. The trainable artifact is a small ANT head (roughly 2.6M parameters for the
default 512-dimensional scorer), which is why checkpoints are tiny and the backbone stays on
Hugging Face or in an MLX folder.

## Repository split

- `ant-08b` (this repository): `Qwen3.5-0.8B`, optimized for lightweight local serving.
- `ant-4b`: `Qwen3.5-4B`, the higher-capacity reference variant.

Training heads is a **cloud workflow**. Do not run real-backbone training locally: use the
[Colab guide](docs/training/train-08b-on-colab.md), then bring back only the small head checkpoint.
Local work is inference/profiling through MLX.

## What was optimized for speed and size

1. **MLX inference backend** (`backbone_backend="mlx"`): `mlx-lm` runs the frozen Qwen3.5-0.8B
   encoder on Apple Silicon, including the GatedDeltaNet layers that fall back to a slow reference
   implementation under PyTorch MPS. The torch ANT head, calibration and checkpoint format are
   unchanged.
2. **MLX weight quantization** (`quantization="mlx"`): convert a checkpoint once with
   `scripts/mlx_convert.py` at 8-bit (conservative) or 4-bit (lighter), then load the matching
   MLX folder. Quantization never silently requantizes weights at load time.

Both changes are for serving. The head is still trained at full precision on Colab CUDA; the
quality gate must pass before a quantized checkpoint replaces an unquantized one. The local
conversion smoke is recorded in
[`docs/experiments/2026-10-24-mlx-quantization-smoke.md`](docs/experiments/2026-10-24-mlx-quantization-smoke.md).

## Install

```bash
git clone <this-repo> && cd ant-08b
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,mlx]"
```

The `mlx` extra is Apple Silicon only. On CUDA/Colab use `.[train]` for training.

## Convert and serve with MLX

```bash
# Unquantized MLX reference
python scripts/mlx_convert.py --hf-path Qwen/Qwen3.5-0.8B \
  --mlx-path models/Qwen3.5-0.8B-mlx --no-quantize

# Conservative 8-bit MLX variant
python scripts/mlx_convert.py --hf-path Qwen/Qwen3.5-0.8B \
  --mlx-path models/Qwen3.5-0.8B-mlx-8bit --bits 8 --group-size 64

# Lighter 4-bit variant
python scripts/mlx_convert.py --hf-path Qwen/Qwen3.5-0.8B \
  --mlx-path models/Qwen3.5-0.8B-mlx-4bit --bits 4 --group-size 64
```

```python
from ant import ANTConfig
from ant.inference.predictor import ANTPredictor

ant = ANTPredictor.from_config(
    ANTConfig(
        backbone_model="models/Qwen3.5-0.8B-mlx-8bit",
        backbone_backend="mlx",
        quantization="mlx",
        quantization_bits=8,
        quantization_group_size=64,
    )
)

result = ant.predict(
    state="Customer has been charged twice for the same invoice. Refund requested.",
    question="Which department should handle this case?",
    candidates=["Billing", "Technical Support", "Sales", "Legal"],
)
print(result.to_dict())
```

`quantization="mlx"` accepts only an already-converted MLX checkpoint whose embedded bits and
group size match the request. To compare an unquantized arm and a quantized arm on identical
rows, use `scripts/compare_quantization.py`:

```bash
python scripts/compare_quantization.py \
  --checkpoint runs/ant-08b-2026-10-21 \
  --backend mlx --quantization mlx \
  --backbone models/Qwen3.5-0.8B-mlx \
  --quantized-backbone models/Qwen3.5-0.8B-mlx-8bit --bits 8
```

The gate checks `|Δaccuracy| <= 0.005`, `ΔECE <= 0.02`, and reports argmax flips and per-row
probability deltas. Ship a quantized model only after that gate passes.

## Train on Colab (not locally)

Follow [`docs/training/train-08b-on-colab.md`](docs/training/train-08b-on-colab.md). The guide
packages the working tree (including `data/ant-multitask`), trains `head_only` on a Colab GPU,
fits calibration and escalation on validation, and downloads only the small run directory.

The default training command is:

```bash
python -u scripts/train.py \
  --backbone Qwen/Qwen3.5-0.8B \
  --mode head_only \
  --device cuda \
  --dtype auto \
  --data data/ant-multitask \
  --name "ANT 0.8B" \
  --out runs/ant-08b-2026-10-21
```

`--dtype auto` keeps FP16 on a T4; do not force BF16 there. The backbone is frozen, so only the
compact encoders/scorer are trained. The incomplete local run from the 0.8B exploration is not a
valid checkpoint: require `metrics.json`, `calibrator.pt`, `training_report.json`, `ant_head.pt`
and `config.json`. `escalation.json` is expected for the default target risk but is intentionally
absent when no threshold can meet it; the training log records that case.

## Compare with the 4B reference

Compare on identical rows and splits. The complete 4B multitask reference is:

| Split | Rows | Accuracy | ECE |
|---|---:|---:|---:|
| test | 392 | 0.7679 | 0.0697 |
| ood | 216 | 0.5602 | 0.2464 |

The old `0.725 / 0.1553` figures are a 40-row quantization gate sample, not the complete test
baseline. Use the complete rows for the 0.8B decision, and report local warm latency at the same
input lengths, cold/first-predict latency, dtype, and resident memory alongside accuracy/ECE.

## Development checks

```bash
python -m pytest tests/ -q
```

MLX-specific tests cover config validation, padded token batches and quantization matching; the
MLX package itself is optional so the test suite runs on Linux/CUDA too. A real MLX smoke run
requires an Apple Silicon machine and a converted model folder.

## License

Apache-2.0. The Qwen weights are governed by their own license on Hugging Face.
