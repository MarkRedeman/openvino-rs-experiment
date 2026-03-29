# Model Compatibility Checking

Use the `check` command to verify whether a model can compile and run on each
available OpenVINO device (CPU, GPU, NPU). It parses the model XML to show a
summary (inputs, outputs, op types, element types), then tries `compile_model`
on every device. On failure it prints structured diagnostics with likely causes
and suggestions.

## Available OpenVINO IR models

| Model | Path | Description |
|-------|------|-------------|
| Card classification | `models/card-classification/` | FP32 image classifier (224x224) |
| Card classification FP16 | `models/card-classification-fp16/` | FP16-compressed variant |
| Fish detection | `models/fish-detection/` | FP32 Geti/OTX object detector (992x800) |
| Fish detection FP16 | `models/fish-detection-fp16/` | FP16-compressed variant |
| ACT policy | `models/act-openvino/` | ACT robot policy (state + 2 cameras) |
| ACT policy FP16 | `models/act-openvino-fp16/` | FP16-compressed variant |

## Standalone

Requires a standalone build (see `README.md` for `./build-standalone.sh`).

### Card classification (FP32)

```bash
./standalone/run-inference.sh \
  check \
  --model models/card-classification/model.xml \
  --weights models/card-classification/model.bin
```

### Card classification (FP16)

```bash
./standalone/run-inference.sh \
  check \
  --model models/card-classification-fp16/model.xml \
  --weights models/card-classification-fp16/model.bin
```

### Fish detection (FP32)

```bash
./standalone/run-inference.sh \
  check \
  --model models/fish-detection/model.xml \
  --weights models/fish-detection/model.bin
```

### Fish detection (FP16)

```bash
./standalone/run-inference.sh \
  check \
  --model models/fish-detection-fp16/model.xml \
  --weights models/fish-detection-fp16/model.bin
```

### ACT policy (FP32)

```bash
./standalone/run-inference.sh \
  check \
  --model models/act-openvino/act.xml \
  --weights models/act-openvino/act.bin
```

### ACT policy (FP16)

```bash
./standalone/run-inference.sh \
  check \
  --model models/act-openvino-fp16/act.xml \
  --weights models/act-openvino-fp16/act.bin
```

### Check all models at once

```bash
for xml in \
  models/card-classification/model.xml \
  models/card-classification-fp16/model.xml \
  models/fish-detection/model.xml \
  models/fish-detection-fp16/model.xml \
  models/act-openvino/act.xml \
  models/act-openvino-fp16/act.xml; do
  bin="${xml%.xml}.bin"
  echo ""
  echo "========================================"
  echo "Checking: $xml"
  echo "========================================"
  ./standalone/run-inference.sh check --model "$xml" --weights "$bin"
done
```

## Docker Compose

Build the image first:

```bash
docker compose build
```

### Card classification (FP32)

```bash
docker compose run --rm inference \
  check \
  --model /models/card-classification/model.xml \
  --weights /models/card-classification/model.bin
```

### Card classification (FP16)

```bash
docker compose run --rm inference \
  check \
  --model /models/card-classification-fp16/model.xml \
  --weights /models/card-classification-fp16/model.bin
```

### Fish detection (FP32)

```bash
docker compose run --rm inference \
  check \
  --model /models/fish-detection/model.xml \
  --weights /models/fish-detection/model.bin
```

### Fish detection (FP16)

```bash
docker compose run --rm inference \
  check \
  --model /models/fish-detection-fp16/model.xml \
  --weights /models/fish-detection-fp16/model.bin
```

### ACT policy (FP32)

```bash
docker compose run --rm inference \
  check \
  --model /models/act-openvino/act.xml \
  --weights /models/act-openvino/act.bin
```

### ACT policy (FP16)

```bash
docker compose run --rm inference \
  check \
  --model /models/act-openvino-fp16/act.xml \
  --weights /models/act-openvino-fp16/act.bin
```

### With GPU passthrough

Use the `inference-gpu` service to include GPU in the device check:

```bash
docker compose run --rm inference-gpu \
  check \
  --model /models/fish-detection/model.xml \
  --weights /models/fish-detection/model.bin
```

### With NPU passthrough

Use the `inference-npu` service to include NPU in the device check:

```bash
docker compose run --rm inference-npu \
  check \
  --model /models/act-openvino/act.xml \
  --weights /models/act-openvino/act.bin
```

### With all accelerators (GPU + NPU)

To check against all devices at once, pass through both GPU and NPU device nodes:

```bash
docker run --rm \
  --device /dev/dri:/dev/dri \
  --device /dev/accel/accel0:/dev/accel/accel0 \
  -v ./models:/models:ro \
  inference-rs-inference \
  check \
  --model /models/fish-detection/model.xml \
  --weights /models/fish-detection/model.bin
```

### Check all models at once (CPU only)

```bash
for model in \
  "/models/card-classification/model" \
  "/models/card-classification-fp16/model" \
  "/models/fish-detection/model" \
  "/models/fish-detection-fp16/model" \
  "/models/act-openvino/act" \
  "/models/act-openvino-fp16/act"; do
  echo ""
  echo "========================================"
  echo "Checking: ${model}.xml"
  echo "========================================"
  docker compose run --rm inference \
    check --model "${model}.xml" --weights "${model}.bin"
done
```

### Check all models at once (all devices)

```bash
for model in \
  "/models/card-classification/model" \
  "/models/card-classification-fp16/model" \
  "/models/fish-detection/model" \
  "/models/fish-detection-fp16/model" \
  "/models/act-openvino/act" \
  "/models/act-openvino-fp16/act"; do
  echo ""
  echo "========================================"
  echo "Checking: ${model}.xml"
  echo "========================================"
  docker run --rm \
    --device /dev/dri:/dev/dri \
    --device /dev/accel/accel0:/dev/accel/accel0 \
    -v ./models:/models:ro \
    inference-rs-inference \
    check --model "${model}.xml" --weights "${model}.bin"
done
```

## Checking a single device

Use `--device` on the `check` subcommand to test only a specific device instead
of all available ones:

```bash
# Standalone — check only NPU
./standalone/run-inference.sh \
  check \
  --model models/act-openvino/act.xml \
  --weights models/act-openvino/act.bin \
  --device NPU

# Docker — check only GPU
docker compose run --rm inference-gpu \
  check \
  --model /models/fish-detection/model.xml \
  --weights /models/fish-detection/model.bin \
  --device GPU
```

## What the output looks like

A successful check prints a model summary followed by per-device results:

```
Model: model.xml
  Inputs:  image [1,3,224,224] f32
  Outputs: output [1,1000] f32
  Layers:  94 total, 12 op types
  Element types: f32 (87), i64 (4), i32 (3)

Device compatibility:
  CPU  ✓  compiled in 0.18s
  GPU  ✓  compiled in 1.42s
  NPU  ✗  compiled failed (0.03s)
       Diagnosis: NPU does not support all operations in this model
       Likely causes:
         - NonMaxSuppression is not supported on NPU
         - TopK is not supported on NPU
       Suggestions:
         - Try HETERO:NPU,CPU to offload unsupported ops to CPU
         - Convert model to FP16 for better NPU compatibility
```
