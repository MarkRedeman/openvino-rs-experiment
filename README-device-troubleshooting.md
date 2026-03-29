# Device selection troubleshooting (CPU / GPU / NPU)

This guide helps debug `--device` issues in `inference-rs`, including:

- NPU not detected or failing to compile
- forcing Intel iGPU (for example Lunar Lake) vs discrete Intel Arc GPU
- Docker passthrough and permission issues

## Quick sanity check

From the project root:

```bash
docker compose build
docker run --rm inference-rs-inference --list-devices
```

Expected output is a list such as:

```text
Available OpenVINO devices:
- CPU
- GPU.0
- GPU.1
- NPU
```

If only `CPU` appears, the container cannot see accelerator devices yet.

## Device strings supported by this app

- `CPU`
- `GPU`
- `NPU`
- `AUTO`
- advanced strings pass through directly (`GPU.0`, `GPU.1`, `AUTO:GPU,CPU`, `MULTI:GPU,CPU`, `HETERO:GPU,CPU`)

## Docker passthrough checklist

### GPU (iGPU or Arc)

Run with:

```bash
docker run --rm \
  --device /dev/dri:/dev/dri \
  inference-rs-inference --list-devices
```

Host checks:

```bash
ls -l /dev/dri
groups
```

- You should have render device nodes such as `renderD128`.
- Your user should be in the `render` group.

### NPU

Run with:

```bash
docker run --rm \
  --device /dev/accel/accel0:/dev/accel/accel0 \
  inference-rs-inference --list-devices
```

Host checks:

```bash
ls -l /dev/accel
groups
```

- `/dev/accel/accel0` must exist.
- Your user should be in the `render` group.

## NPU troubleshooting

If `NPU` is missing from `--list-devices`:

1. Verify NPU driver packages are installed on host (Linux):
   - `intel-level-zero-npu`
   - `intel-driver-compiler-npu`
2. Verify kernel/driver support (commonly kernel 6.6+ for recent Intel NPU stacks).
3. Verify device node exists: `ls /dev/accel/accel0`.
4. Re-run container with explicit accel passthrough:
   - `--device /dev/accel/accel0:/dev/accel/accel0`
5. Re-run `--list-devices` inside container.

If `NPU` appears but inference fails:

1. Start with a small known-good model and `--task classify`.
2. Try fallback ordering to confirm pipeline health:
   - `--device AUTO:NPU,CPU`
3. Compare with CPU run using the exact same model and inputs.

## Choosing iGPU vs discrete Arc GPU

When both an Intel iGPU (for example Lunar Lake) and Intel Arc dGPU are present,
OpenVINO typically exposes indexed GPU devices like `GPU.0` and `GPU.1`.

### Step 1: enumerate

```bash
docker run --rm --device /dev/dri:/dev/dri inference-rs-inference --list-devices
```

### Step 2: test explicit indices

```bash
# candidate 1
docker run --rm --device /dev/dri:/dev/dri \
  -v ./models:/models:ro -v ./images:/images:ro \
  inference-rs-inference \
  --model /models/card-classification/model.xml \
  --weights /models/card-classification/model.bin \
  --image /images/diamond-card.jpg \
  --task benchmark \
  --device GPU.0 \
  --benchmark-duration 5

# candidate 2
docker run --rm --device /dev/dri:/dev/dri \
  -v ./models:/models:ro -v ./images:/images:ro \
  inference-rs-inference \
  --model /models/card-classification/model.xml \
  --weights /models/card-classification/model.bin \
  --image /images/diamond-card.jpg \
  --task benchmark \
  --device GPU.1 \
  --benchmark-duration 5
```

Use throughput/latency and expected power profile to identify which index maps to iGPU vs dGPU.

### Step 3 (most deterministic): pass only one render node

Instead of mounting all of `/dev/dri`, mount a specific render node so the container can only see one GPU.

```bash
# Example: expose only renderD128
docker run --rm \
  --device /dev/dri/renderD128:/dev/dri/renderD128 \
  inference-rs-inference --list-devices
```

Repeat with another render node (`renderD129`, etc.) to map each node to iGPU/dGPU.

## Common failure patterns

- `--device NPU` silently behaves like CPU
  - Old binary or old `parse_device` logic; rebuild and verify with `--list-devices`.
- `GPU` or `NPU` missing in container, but present on host
  - Missing `--device` passthrough (`/dev/dri` or `/dev/accel/accel0`).
- Permission denied on device nodes
  - User not in `render` group, or restrictive node permissions.
- `AUTO` picks CPU unexpectedly
  - GPU/NPU not visible or plugin unavailable; verify with `--list-devices` then try explicit `GPU` or `NPU`.

## Recommended debug sequence

1. `--list-devices` without passthrough (baseline)
2. `--list-devices` with GPU passthrough
3. `--list-devices` with NPU passthrough
4. run one small inference on explicit `GPU.0`/`GPU.1`/`NPU`
5. use benchmark mode to confirm you are on intended hardware
