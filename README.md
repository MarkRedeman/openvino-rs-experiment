# inference-rs

Vision model inference in Rust using [OpenVINO](https://docs.openvino.ai/).
Supports image classification and object detection (Geti/OTX, SSD, YOLO output formats).

The entire build and runtime environment is Dockerized — **only Docker is required** on the host machine.

## Prerequisites

- Docker (or Docker Desktop)
- OpenVINO IR model files (`.xml` + `.bin`)
- Input image(s) (JPEG, PNG, etc.)

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
│   ├── main.rs                # CLI entry point (clap)
│   ├── lib.rs                 # Module re-exports
│   ├── engine.rs              # OpenVINO Core lifecycle, infer / infer_multi
│   ├── preprocessing.rs       # Image load, resize, normalize → NHWC f32 Tensor
│   ├── postprocessing.rs      # top-K classification, SSD/YOLO/Geti detection decoding
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
```

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
```

The `standalone/` directory is portable — you can tar it up and copy it to
another machine. The only requirement is a compatible x86_64 Linux with glibc
(Ubuntu 22.04+, Debian 12+, RHEL 9+, etc.).

### CLI reference

```
inference-rs — Run vision-model inference with OpenVINO

Options:
  --model <PATH>               Path to the OpenVINO IR model (.xml)
  --weights <PATH>             Path to the OpenVINO IR weights (.bin)
  --image <PATH>               Path to the input image
  --device <DEVICE>            Inference device [default: CPU]
  --task <TASK>                classify | detect [default: classify]
  --top-k <N>                  Top-K results for classification [default: 5]
  --threshold <F>              Confidence threshold for detection [default: 0.5]
  --width <PX>                 Model input width [default: 224]
  --height <PX>                Model input height [default: 224]
  --detection-format <FMT>     geti | ssd | yolo [default: geti]
  --num-classes <N>            Number of classes (YOLO only) [default: 80]
  --output-image <PATH>        Save annotated detection image
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
- **Preprocessing pipeline**: Images are loaded with the `image` crate, resized and normalized to `[0.0, 1.0]` f32 NHWC tensors. The OpenVINO pre-process pipeline then handles NHWC→NCHW layout conversion and any further resizing to match the model's expected input dimensions.
- **Multi-output support**: Models with multiple outputs (e.g., Geti detection with `boxes` + `labels`) use `infer_multi()` which returns a `HashMap<String, OutputBuffer>`. Single-output models use the simpler `infer()` path.
- **Detection visualization**: When `--output-image` is set for detection tasks, the tool draws class-colored bounding boxes and text labels onto the original image, then saves the annotated image to disk.
