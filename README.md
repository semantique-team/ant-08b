# ANT-08B

[![Status: WIP](https://img.shields.io/badge/status-WIP-yellow)](#status)
[![Open Source](https://img.shields.io/badge/open--source-yes-brightgreen)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qwen](https://img.shields.io/badge/Backbone-Qwen3.5--0.8B-6E46AE)](https://huggingface.co/Qwen/Qwen3.5-0.8B)
[![License](https://img.shields.io/badge/License-Apache--2.0-3DA639)](LICENSE)

ANT-08B is an open-source **lightweight System 1 decision model** from the **Semantique Team**.
It is built on [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) and scores
candidate answers for a state and question. The output is a calibrated choice over the options
you provide, together with confidence metrics and an execute/escalate decision.

> **Status: work in progress.** The API and checkpoint format are usable, but benchmarks,
> calibration quality and production guidance are still being finalized.

The backbone is used as an encoder. There is no LM head and no text generation in the inference
path. The trainable artifact is a small ANT head; the released checkpoint contains that head and
its calibration metadata.

## Properties

- **Small footprint**: designed for local Apple Silicon serving and lower-latency decisions.
- **Variable candidate count**: N = 2 to 100+; candidates are semantic text objects.
- **Permutation invariant**: candidate order does not change the probabilities.
- **Cacheable**: static candidate sets can be embedded once and reused.
- **Calibrated**: temperature or affine calibration fitted on held-out data.
- **Escalation-aware**: confidence metrics support an execute/escalate policy.

## Install

```bash
git clone <this-repo> && cd ant-08b
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

For Apple Silicon serving with MLX:

```bash
pip install -e ".[mlx]"
```

## Checkpoints and model files

The default released ANT checkpoint is:

```text
runs/ant-08b-2026-10-21/
```

It contains:

```text
ant_head.pt
config.json
calibrator.pt
escalation.json
metrics.json
training_report.json
card.json
examples.json
```

The Qwen backbone is separate from the ANT checkpoint. The default local paths used below are:

| Runtime | Backbone path |
|---|---|
| CUDA | `models/Qwen3.5-0.8B` |
| MLX | `models/Qwen3.5-0.8B-mlx` |
| MLX 8-bit | `models/Qwen3.5-0.8B-mlx-8bit` |

Download the original Qwen checkpoint for CUDA:

```bash
hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B
```

## Serve with CUDA

```python
from ant.backbone.qwen import QwenBackbone
from ant.inference.predictor import ANTPredictor

CHECKPOINT = "runs/ant-08b-2026-10-21"
BACKBONE = "models/Qwen3.5-0.8B"

backbone = QwenBackbone(
    BACKBONE,
    device="cuda",
    torch_dtype="auto",
)
ant = ANTPredictor.load(CHECKPOINT, backbone=backbone)

result = ant.predict(
    state="Customer has been charged twice for the same invoice. Refund requested.",
    question="Which department should handle this case?",
    candidates=["Billing", "Technical Support", "Sales", "Legal"],
)
print(result.to_dict())
```

## Serve with MLX

MLX is the preferred local backend on Apple Silicon. Convert the Qwen checkpoint once:

```bash
python scripts/mlx_convert.py \
  --hf-path models/Qwen3.5-0.8B \
  --mlx-path models/Qwen3.5-0.8B-mlx \
  --no-quantize
```

Then load the same ANT checkpoint with the MLX backbone:

```python
from ant.backbone.mlx import MlxBackbone
from ant.inference.predictor import ANTPredictor

CHECKPOINT = "runs/ant-08b-2026-10-21"
BACKBONE = "models/Qwen3.5-0.8B-mlx"

backbone = MlxBackbone(BACKBONE, device="mps")
ant = ANTPredictor.load(CHECKPOINT, backbone=backbone)
```

### MLX quantized weights

For a smaller footprint, convert an 8-bit variant:

```bash
python scripts/mlx_convert.py \
  --hf-path models/Qwen3.5-0.8B \
  --mlx-path models/Qwen3.5-0.8B-mlx-8bit \
  --bits 8 --group-size 64
```

```python
backbone = MlxBackbone(
    "models/Qwen3.5-0.8B-mlx-8bit",
    device="mps",
    quantization="mlx",
    quantization_bits=8,
    quantization_group_size=64,
)
ant = ANTPredictor.load("runs/ant-08b-2026-10-21", backbone=backbone)
```

Quantized weights are accepted only when the converted checkpoint's bits and group size match
the request. Compare quantized and unquantized decisions before replacing the default:

```bash
python scripts/compare_quantization.py \
  --checkpoint runs/ant-08b-2026-10-21 \
  --backend mlx --quantization mlx \
  --backbone models/Qwen3.5-0.8B-mlx \
  --quantized-backbone models/Qwen3.5-0.8B-mlx-8bit \
  --bits 8
```

The gate checks accuracy, ECE, argmax flips and per-row probability differences.

## Python API

```python
result = ant.predict(state, question, candidates)

result.choice
result.probabilities
result.scores
result.confidence
result.margin
result.entropy
result.action
```

For ranking without the calibrated policy:

```python
scores = ant.score_candidates(state, question, candidates)
```

## Playground

The FastAPI server and Vite interface live in the sibling [`playground`](../playground)
repository. It consumes this package and one or more checkpoint directories.

CUDA serving through the playground:

```bash
cd ../playground
python -m serve \
  --runs ../ant-08b/runs/ant-08b-2026-10-21 \
  --backbone ../ant-08b/models/Qwen3.5-0.8B
```

Then start the interface:

```bash
npm run dev
```

Add the API URL printed by the server to the playground and select the model.

## Tests

```bash
pytest tests/ -q
```

## License

Apache-2.0. The Qwen weights are governed by their own license on Hugging Face.
