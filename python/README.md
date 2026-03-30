# inference-rs-py

Python reference implementation of the OpenVINO inference toolkit, for benchmarking comparison against the Rust version.

Uses the same OpenVINO runtime, same preprocessing pipeline, same benchmark methodology, and same JSON output schema — so results are directly comparable.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- OpenVINO 2026.0+ runtime (or use Docker)
- Models in `../models/`, images in `../images/`, episodes in `../episodes/`

## Install

```bash
cd python
uv sync
```

## Usage

The CLI mirrors the Rust version with three subcommands: `infer`, `check`, `devices`.

### List devices

```bash
uv run inference-rs-py devices
```

### Classification

```bash
uv run inference-rs-py infer \
  --model ../models/card-classification/model.xml \
  --weights ../models/card-classification/model.bin \
  --image ../images/diamond-card.jpg \
  --task classify \
  --top-k 4 \
  --width 224 \
  --height 224
```

### Detection (Geti format)

```bash
uv run inference-rs-py infer \
  --model ../models/fish-detection/model.xml \
  --weights ../models/fish-detection/model.bin \
  --image ../images/fish.png \
  --task detect \
  --detection-format geti \
  --threshold 0.5 \
  --width 992 \
  --height 800
```

### Benchmark

```bash
uv run inference-rs-py infer \
  --model ../models/card-classification/model.xml \
  --weights ../models/card-classification/model.bin \
  --image ../images/diamond-card.jpg \
  --task benchmark \
  --width 224 \
  --height 224 \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --output-json benchmark-result.json
```

### ACT inference

```bash
uv run inference-rs-py infer \
  --model ../models/act-openvino/act.xml \
  --weights ../models/act-openvino/act.bin \
  --task act \
  --episode-dir ../episodes/ep_000_dc7198bd
```

### ACT benchmark

```bash
uv run inference-rs-py infer \
  --model ../models/act-openvino/act.xml \
  --weights ../models/act-openvino/act.bin \
  --task act-benchmark \
  --episode-dir ../episodes/ep_000_dc7198bd \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --output-json act-benchmark-result.json
```

### Check model compatibility

```bash
uv run inference-rs-py check \
  --model ../models/card-classification/model.xml \
  --weights ../models/card-classification/model.bin
```

## Docker

A `Dockerfile` and `docker-compose.python.yml` are provided, using the same `openvino/ubuntu24_runtime:2026.0.0` base image as the Rust version for fair comparison.

```bash
# Build
docker compose -f docker-compose.python.yml build

# Run classification on CPU
docker compose -f docker-compose.python.yml run --rm inference-py

# Run on GPU
docker compose -f docker-compose.python.yml run --rm inference-gpu-py

# Run on NPU
docker compose -f docker-compose.python.yml run --rm inference-npu-py
```

## CLI reference

```
inference-rs-py <COMMAND>

COMMANDS:
  infer    Run inference (classify, detect, benchmark, act, act-benchmark)
  check    Check model compatibility against available devices
  devices  List available OpenVINO devices
```

### `infer` options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to OpenVINO IR model (.xml) |
| `--weights` | required | Path to OpenVINO IR weights (.bin) |
| `--image` | — | Input image (required for classify/detect/benchmark) |
| `--device` | CPU | Inference device (CPU, GPU, NPU, AUTO, ...) |
| `--task` | classify | classify, detect, benchmark, act, act-benchmark |
| `--top-k` | 5 | Top-K classification results |
| `--threshold` | 0.5 | Detection confidence threshold |
| `--width` | 224 | Model input width |
| `--height` | 224 | Model input height |
| `--detection-format` | geti | geti, ssd, yolo |
| `--num-classes` | 80 | Number of classes (YOLO only) |
| `--output-json` | — | Save results as JSON |
| `--benchmark-duration` | 10.0 | Benchmark duration (seconds) |
| `--benchmark-warmup` | 20 | Warmup iterations |
| `--benchmark-iters` | — | Fixed iteration count (overrides duration) |
| `--benchmark-report-every` | 1.0 | Progress report interval |
| `--benchmark-stage-iters` | 30 | Stage timing iterations (0 disables) |
| `--benchmark-stage-read-each-iter` | false | Include I/O in stage timing |
| `--preprocess-backend` | python | python or openvino |
| `--metadata` | — | ACT metadata YAML path |
| `--episode-dir` | — | Episode directory for ACT |

## Architecture

The Python implementation follows the same module structure as the Rust version:

| Python module | Rust equivalent | Purpose |
|---|---|---|
| `preprocessing.py` | `preprocessing.rs` | Image load, resize (Pillow BILINEAR), normalize /255, NHWC/NCHW tensors |
| `labels.py` | `labels.rs` | Parse labels from model XML `<rt_info>` |
| `postprocessing.py` | `postprocessing.rs` | top-k classification, Geti/SSD/YOLO detection decoding |
| `benchmark.py` | `benchmark.rs` | Warmup/measurement loop, microsecond latencies, nearest-rank percentile stats |
| `output.py` | `output.rs` | JSON serialization (same schema as Rust for `jq` compatibility) |
| `engine.py` | `engine.rs` | Vision OpenVINO engine with PPP (NHWC->NCHW, bilinear resize) |
| `act_engine.py` | `act.rs` | ACT engine (3 inputs, no PPP, metadata discovery) |
| `cli.py` | `main.rs` | argparse CLI with subcommands |

## JSON output compatibility

JSON output uses the same schema as the Rust version. The `benchmark-all.sh` script can parse both with the same `jq` expression:

```bash
jq '.report.throughput_fps' result.json
```
