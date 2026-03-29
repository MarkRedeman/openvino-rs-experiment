# inference-rs

Vision model inference in Rust using [OpenVINO](https://docs.openvino.ai/).
Supports image classification, object detection (Geti/OTX, SSD, YOLO output formats), benchmark mode,
and ACT policy inference for Physical AI Studio OpenVINO exports.

The entire build and runtime environment is Dockerized — **only Docker is required** on the host machine.

## Prerequisites

- Docker (or Docker Desktop)
- OpenVINO IR model files (`.xml` + `.bin`)
- Input image(s) (JPEG, PNG, etc.)

## Code structure

The codebase follows a lightweight hexagonal split:

- `src/domain/` — core model abstraction (`InferenceModel`), shared input/output types, domain errors
- `src/models/` — adapters for concrete runtimes (`VisionModel`, `ActModel`) + enum-dispatched `ModelWrapper` + `ModelRegistry` (`load/get/unload`)
- `src/infra/` — shared infra utilities (device parsing, tensor buffer casting)
- `src/output.rs` — JSON output payloads/writers for CLI tasks
- `src/main.rs` — CLI/bootstrap orchestration

## Project structure

```
inference-rs/
├── Cargo.toml
├── Dockerfile                 # Multi-stage: openvino dev → runtime
├── Dockerfile.standalone      # Builds a portable bundle (binary + .so libs)
├── docker-compose.yml         # Convenience runner with volume mounts
├── build-standalone.sh        # Script to build & extract the standalone bundle
├── run-inference.sh           # Launcher script (bundled into standalone/)
├── fonts/
│   └── DejaVuSans.ttf         # Embedded font for detection label rendering
├── src/
│   ├── main.rs                # CLI entry point and app bootstrap
│   ├── lib.rs                 # Module exports
│   ├── domain/                # Inference domain API (trait/types/errors)
│   ├── models/                # Vision/ACT model adapters + registry
│   ├── infra/                 # Shared infra helpers
│   ├── output.rs              # JSON output serialization/writing
│   ├── engine.rs              # OpenVINO vision engine internals
│   ├── act.rs                 # OpenVINO ACT engine internals
│   ├── preprocessing.rs       # Image load, resize, normalize → NHWC f32 Tensor
│   ├── postprocessing.rs      # top-K classification, SSD/YOLO/Geti detection decoding
│   ├── benchmark.rs           # Throughput/latency benchmark runner
│   └── visualization.rs       # Draw detection boxes + labels to output image
├── models/                    # Place your OpenVINO IR models here
└── images/                    # Place your input images here
```

## Build

```bash
docker compose build
```

This runs a multi-stage Docker build:

1. **Stage 1** (`openvino/ubuntu24_dev:2026.0.0`) — installs the Rust toolchain and compiles the binary.
2. **Stage 2** (`openvino/ubuntu24_runtime:2026.0.0`) — copies just the binary into a slim runtime image with OpenVINO shared libraries.

## Most common build/run examples

```bash
# 1) Build image once
docker compose build

# 2) Classification
docker run --rm \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  inference-rs-inference \
  --model /models/card-classification/model.xml \
  --weights /models/card-classification/model.bin \
  --image /images/diamond-card.jpg \
  --task classify \
  --top-k 4 \
  --width 224 \
  --height 224

# 3) Detection (Geti format) + annotated image
docker run --rm --user "$(id -u):$(id -g)" \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  -v ./output:/output \
  inference-rs-inference \
  --model /models/fish-detection/model.xml \
  --weights /models/fish-detection/model.bin \
  --image /images/fish.png \
  --task detect \
  --detection-format geti \
  --threshold 0.5 \
  --width 992 \
  --height 800 \
  --output-image /output/fish-detected.png

# 4) ACT single run + JSON output
docker run --rm --user "$(id -u):$(id -g)" \
  -v ./models:/models:ro \
  -v ./episodes:/episodes:ro \
  -v ./output:/output \
  inference-rs-inference \
  --task act \
  --model /models/act-openvino/act.xml \
  --weights /models/act-openvino/act.bin \
  --episode-dir /episodes/ep_000_dc7198bd \
  --output-json /output/act-output.json

# 5) Vision benchmark
docker run --rm \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  inference-rs-inference \
  --model /models/card-classification/model.xml \
  --weights /models/card-classification/model.bin \
  --image /images/diamond-card.jpg \
  --task benchmark \
  --width 224 \
  --height 224 \
  --benchmark-warmup 20 \
  --benchmark-duration 10
```



## Usage

### Quick start with docker compose

The default `docker-compose.yml` mounts `./models` and `./images` into the container. Edit the `command:` section to match your model and image, then run:

```bash
docker compose up
```

### Running with docker run

Build the image first, then pass arguments directly:

```bash
docker compose build

# Classification
docker run --rm \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  inference-rs-inference \
  --model /models/card-classification/model.xml \
  --weights /models/card-classification/model.bin \
  --image /images/diamond-card.jpg \
  --task classify \
  --top-k 4 \
  --width 224 \
  --height 224

# Classification + save JSON
docker run --rm --user "$(id -u):$(id -g)" \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  -v ./output:/output \
  inference-rs-inference \
  --model /models/card-classification/model.xml \
  --weights /models/card-classification/model.bin \
  --image /images/diamond-card.jpg \
  --task classify \
  --top-k 4 \
  --width 224 \
  --height 224 \
  --output-json /output/card-classification.json

# Object detection (Geti/OTX format)
docker run --rm \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  inference-rs-inference \
  --model /models/fish-detection/model.xml \
  --weights /models/fish-detection/model.bin \
  --image /images/fish.png \
  --task detect \
  --detection-format geti \
  --threshold 0.5 \
  --width 992 \
  --height 800

# Object detection + save annotated image
docker run --rm --user "$(id -u):$(id -g)" \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  -v ./output:/output \
  inference-rs-inference \
  --model /models/fish-detection/model.xml \
  --weights /models/fish-detection/model.bin \
  --image /images/fish.png \
  --task detect \
  --detection-format geti \
  --threshold 0.5 \
  --width 992 \
  --height 800 \
  --output-image /output/fish-detected.png

# Object detection + save machine-readable JSON
docker run --rm --user "$(id -u):$(id -g)" \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  -v ./output:/output \
  inference-rs-inference \
  --model /models/fish-detection/model.xml \
  --weights /models/fish-detection/model.bin \
  --image /images/fish.png \
  --task detect \
  --detection-format geti \
  --threshold 0.5 \
  --width 992 \
  --height 800 \
  --output-json /output/fish-detected.json

# Benchmark mode (load model once, run repeated inference)
docker run --rm \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  inference-rs-inference \
  --model /models/card-classification/model.xml \
  --weights /models/card-classification/model.bin \
  --image /images/diamond-card.jpg \
  --task benchmark \
  --width 224 \
  --height 224 \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --benchmark-stage-iters 30

# Benchmark with OpenVINO resize backend (for A/B comparison)
docker run --rm \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  inference-rs-inference \
  --model /models/fish-detection/model.xml \
  --weights /models/fish-detection/model.bin \
  --image /images/fish.png \
  --task benchmark \
  --detection-format geti \
  --threshold 0.3 \
  --width 992 \
  --height 800 \
  --preprocess-backend openvino \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --benchmark-stage-iters 300 \
  --benchmark-stage-read-each-iter
```

### Device selection (CPU, GPU/XPU, NPU, AUTO)

OpenVINO device selection is controlled with `--device`.

- `CPU` — CPU plugin
- `GPU` — Intel GPU plugin (covers iGPU and Arc/dGPU; OpenVINO does not use `XPU` as a device string)
- `NPU` — Intel NPU plugin
- `AUTO` — automatic plugin selection
- Advanced strings are passed through as-is, for example: `GPU.0`, `AUTO:GPU,CPU`, `MULTI:GPU,CPU`, `HETERO:GPU,CPU`

List available devices at runtime:

```bash
docker run --rm inference-rs-inference --list-devices
```

GPU example (`docker run`):

```bash
docker run --rm \
  --device /dev/dri:/dev/dri \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  inference-rs-inference \
  --model /models/card-classification/model.xml \
  --weights /models/card-classification/model.bin \
  --image /images/diamond-card.jpg \
  --task classify \
  --top-k 4 \
  --device GPU
```

NPU example (`docker run`):

```bash
docker run --rm \
  --device /dev/accel/accel0:/dev/accel/accel0 \
  -v ./models:/models:ro \
  -v ./images:/images:ro \
  inference-rs-inference \
  --model /models/card-classification/model.xml \
  --weights /models/card-classification/model.bin \
  --image /images/diamond-card.jpg \
  --task classify \
  --top-k 4 \
  --device NPU
```

The included `docker-compose.yml` also provides:

- `inference` (CPU)
- `inference-gpu` (passes through `/dev/dri`)
- `inference-npu` (passes through `/dev/accel/accel0`)

Host requirements summary:

| Device | Linux host requirements |
|--------|-------------------------|
| GPU | Intel GPU drivers/runtime (`intel-opencl-icd`, `intel-level-zero-gpu`, `level-zero`) and access to render device nodes (`render` group) |
| NPU | Intel NPU driver stack (`intel-level-zero-npu`, `intel-driver-compiler-npu`), kernel support (typically 6.6+), and render/accel device node access |

Standalone note:

- `Dockerfile.standalone` always bundles CPU/AUTO/HETERO plugins.
- It now also attempts to bundle GPU and NPU plugins when present in the OpenVINO runtime image.
- Hardware-specific host drivers are still required on the target machine.

### Standalone build (run without Docker)

You can compile the binary inside Docker, then extract it alongside the required
OpenVINO shared libraries into a self-contained directory that runs on any
x86_64 Linux host — no Docker or OpenVINO installation needed at runtime.

```bash
./build-standalone.sh
```

This produces a `standalone/` directory:

```
standalone/
├── inference-rs          # the binary (~3.8 MB)
├── lib/                  # OpenVINO + TBB shared libraries (~71 MB)
└── run-inference.sh      # launcher that sets LD_LIBRARY_PATH
```

Run inference directly:

```bash
./standalone/run-inference.sh \
  --model models/card-classification/model.xml \
  --weights models/card-classification/model.bin \
  --image images/diamond-card.jpg \
  --task classify \
  --top-k 4

./standalone/run-inference.sh \
  --model models/fish-detection/model.xml \
  --weights models/fish-detection/model.bin \
  --image images/fish.png \
  --task detect \
  --detection-format geti \
  --threshold 0.3 \
  --width 992 \
  --height 800

# ACT policy inference (OpenVINO export package)
./standalone/run-inference.sh \
  --task act \
  --model models/act-openvino/act.xml \
  --weights models/act-openvino/act.bin \
  --episode-dir episodes/ep_000_dc7198bd \
  --output-json output/act-output-standalone.json

# ACT benchmark mode (reuses one loaded model/request)
./standalone/run-inference.sh \
  --task act-benchmark \
  --model models/act-openvino/act.xml \
  --weights models/act-openvino/act.bin \
  --episode-dir episodes/ep_000_dc7198bd \
  --benchmark-warmup 20 \
  --benchmark-duration 10 \
  --benchmark-report-every 1 \
  --output-json output/act-benchmark.json
```

The `standalone/` directory is portable — you can tar it up and copy it to
another machine. The only requirement is a compatible x86_64 Linux with glibc
(Ubuntu 22.04+, Debian 12+, RHEL 9+, etc.).

### ACT benchmark mode

`--task act-benchmark` benchmarks ACT inference while reusing a single loaded model
and inference request (same strategy as vision benchmark mode). It reports:

- mean/min/max latency (ms)
- p50/p90/p95/p99 latency (ms)
- approximate throughput (runs/sec)
- optional rough stage timing breakdown (I/O+decode, preprocess, inference, postprocess)

Example:

```bash
./standalone/run-inference.sh \
  --task act-benchmark \
  --model ./models/act-openvino/act.xml \
  --weights ./models/act-openvino/act.bin \
  --episode-dir ./episodes/ep_000_dc7198bd \
  --benchmark-warmup 20 \
  --benchmark-iters 100 \
  --benchmark-report-every 1 \
  --benchmark-stage-iters 30 \
  --benchmark-stage-read-each-iter \
  --output-json ./output/act-benchmark.json
```

Notes:

- Set `--benchmark-stage-iters 0` to disable stage timing.
- With `--benchmark-stage-read-each-iter`, stage timing includes re-reading episode inputs each iteration.
- Without it, stage timing reuses prebuilt tensors (similar to throughput loop) and mostly isolates inference cost.

Output files:

- `output/act-benchmark.json` (if `--output-json` is set)

### ACT code walkthrough

For a detailed ACT code walkthrough (model load, episode load, camera mapping, inference, and output shape), see `README-act-inference.md`.

For accelerator troubleshooting (NPU setup, iGPU vs discrete Arc selection, Docker passthrough), see `README-device-troubleshooting.md`.

### CLI reference

```
inference-rs — Run vision-model inference with OpenVINO

Options:
  --list-devices              List available OpenVINO devices and exit
  --model <PATH>               Path to the OpenVINO IR model (.xml)
  --weights <PATH>             Path to the OpenVINO IR weights (.bin)
  --image <PATH>               Path to the input image (required for classify/detect/benchmark)
  --device <DEVICE>            Inference device [default: CPU]
  --task <TASK>                classify | detect | benchmark | act | act-benchmark [default: classify]
  --top-k <N>                  Top-K results for classification [default: 5]
  --threshold <F>              Confidence threshold for detection [default: 0.5]
  --width <PX>                 Model input width [default: 224]
  --height <PX>                Model input height [default: 224]
  --preprocess-backend <BACK>  rust | openvino [default: rust]
  --metadata <PATH>            Path to ACT metadata.yaml (optional; defaults next to --model)
  --episode-dir <PATH>         Episode directory with data.jsonl and stats/images (ACT only)
  --detection-format <FMT>     geti | ssd | yolo [default: geti]
  --num-classes <N>            Number of classes (YOLO only) [default: 80]
  --output-image <PATH>        Save annotated detection image
  --output-json <PATH>         Save results as JSON
  --benchmark-duration <SEC>   Benchmark duration in seconds [default: 10]
  --benchmark-warmup <N>       Warmup iterations [default: 20]
  --benchmark-iters <N>        Measured benchmark iterations (overrides duration)
  --benchmark-report-every <S> Benchmark progress interval seconds [default: 1]
  --benchmark-stage-iters <N>  Rough stage timing iterations [default: 30, 0 disables]
  --benchmark-stage-read-each-iter
                               In stage timing, include read+decode every iteration
  -h, --help                   Print help
  -V, --version                Print version
```

## Detection formats

| Format | Output shape | Description |
|--------|-------------|-------------|
| `geti`  | `boxes` `[1,N,5]` f32 + `labels` `[1,N]` i64 | Intel Geti / OTX exported models. Coordinates in absolute pixels. |
| `ssd`   | `[1,1,N,7]` f32 | Standard SSD: `[image_id, class_id, conf, x1, y1, x2, y2]`. |
| `yolo`  | `[1,N,5+C]` f32 | YOLO: `[cx, cy, w, h, obj_conf, class_scores...]`. |

## File permissions

The runtime container runs as user `openvino` (uid 1001). Model and image files must be readable by this user:

```bash
chmod -R a+r models/ images/
```

When saving an output image, either run the container with `--user "$(id -u):$(id -g)"`
or make `output/` writable by uid 1001.

## Architecture notes

- **Runtime linking**: The `openvino` crate is built with the `runtime-linking` feature. OpenVINO shared libraries are loaded via `dlopen` at runtime rather than linked at compile time. This is why `openvino_sys::library::load()` must be called before any other OpenVINO API call.
- **Preprocessing pipeline**: `--preprocess-backend rust` resizes+normalizes in Rust before inference. `--preprocess-backend openvino` decodes in Rust but delegates resize to the OpenVINO pre-process pipeline.
- **Multi-output support**: Models with multiple outputs (e.g., Geti detection with `boxes` + `labels`) use `infer_multi()` which returns a `HashMap<String, OutputBuffer>`. Single-output models use the simpler `infer()` path.
- **Detection visualization**: When `--output-image` is set for detection tasks, the tool draws class-colored bounding boxes and text labels onto the original image, then saves the annotated image to disk.
- **JSON export**: When `--output-json` is set, the tool writes pretty-printed JSON for all tasks: classification results, detections, or benchmark reports.
- **Benchmark mode**: `--task benchmark` reuses the same loaded model and input image tensor, continuously runs inference, and reports throughput plus latency stats (mean/min/p50/p90/p95/p99/max).
- **Benchmark stage timing**: Benchmark mode reports rough average breakdown for I/O+decode, preprocess (resize+tensor conversion), inference, and postprocess. By default, image decode happens once; add `--benchmark-stage-read-each-iter` to include read+decode cost each iteration.
