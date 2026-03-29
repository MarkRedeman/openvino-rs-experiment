# ACT (Action Chunking with Transformers) Inference Implementation Plan

> **Note**: This is a historical design document written before implementation.
> The actual CLI now uses subcommands — see `inference-rs infer --task act --help`
> for current usage, and `README.md` for up-to-date examples.

## Goal

Extend `inference-rs` to support **ACT (Action Chunking with Transformers)** policy inference for
models exported by **Intel Physical AI Studio** in OpenVINO IR format.

Scope for this document:

- Inference only (assume `.xml` + `.bin` already exported)
- Full implementation plan (architecture, data shapes, roadmap)
- Compatible with ACT design in the original ALOHA paper

## Executive Summary

ACT is different from the vision models currently in `inference-rs`:

- Current modes (`classify`, `detect`) are single-image, stateless inference.
- ACT is **multi-input and stateful**:
  - Inputs: robot state vector + one or more camera images
  - Output: a chunk of future actions (joint targets)
  - Runtime behavior: queue/dequeue actions between model calls

Physical AI Studio already exports ACT to OpenVINO and provides metadata. That means we can reuse
our OpenVINO runtime stack, but we need new ACT-specific modules for metadata parsing,
preprocessing, and action chunk management.

## ACT Architecture (Inference-Relevant)

References:

- Original ACT paper: <https://arxiv.org/abs/2304.13705>
- Physical AI Studio export/inference docs: <https://github.com/open-edge-platform/physical-ai-studio/blob/main/library/docs/how-to/export/export_inference.md>

### What matters at inference time

- ACT was trained as a CVAE, but **the VAE encoder is not used at inference**.
- Inference sets latent `z` to zeros and uses the policy decoder path.
- The model predicts a full **action chunk** of length `chunk_size` in one forward pass.

### Typical model structure

- Vision backbone (often ResNet18) extracts camera features.
- Transformer encoder fuses:
  - latent token (`z=0`)
  - robot state token
  - image feature tokens
- Transformer decoder (DETR-style queries) predicts `chunk_size` actions in parallel.

## Physical AI Studio Export Artifacts

From `policy.export("./exports", backend="openvino")`, expected output:

```text
exports/
├── model.xml
├── model.bin
├── metadata.yaml
└── metadata.json
```

`metadata.yaml`/`metadata.json` are critical for runtime behavior:

- backend and policy class
- `chunk_size`
- whether action queueing is expected
- input/output shapes
- normalization stats (state, images, action)

## Expected I/O Contract

Exact names can vary by export path, but for Physical AI Studio ACT exports we should expect:

- Inputs
  - state-like input (e.g. `state` or `qpos`), shape `(1, state_dim)`, `f32`
  - image-like input (e.g. `images`), shape `(1, n_cameras, 3, H, W)`, `f32`
- Output
  - action-like output (e.g. `action`), shape `(1, chunk_size, action_dim)`, `f32`

Important:

- We must **discover real input/output names from the IR model** and/or metadata.
- Do not hardcode names without fallback mapping logic.

## Key Differences vs Existing `inference-rs` Paths

1. **Multi-input support**
   - Current `Engine` assumes one input name.
   - ACT requires setting at least two named tensors per request.

2. **Layout assumptions**
   - Existing vision path uses NHWC tensor + OpenVINO pre/post pipeline conversion.
   - ACT exports usually expect already-prepared NCHW camera data (with camera dimension).

3. **Normalization/denormalization**
   - Current path only scales RGB to `[0, 1]`.
   - ACT requires feature-wise mean/std normalization for state/images and inverse normalization for output action.

4. **Stateful action queue**
   - Chunked output should be queued and consumed one action at a time across control ticks.

## Proposed Code Changes

### New module tree

```text
src/
  act/
    mod.rs
    metadata.rs
    engine.rs
    preprocessing.rs
    postprocessing.rs
    action_queue.rs
```

### `src/act/metadata.rs`

Responsibilities:

- Parse `metadata.yaml` (primary) and optionally `metadata.json` fallback.
- Provide typed access to:
  - chunk size
  - input/output shapes
  - normalization stats
  - queue behavior flags

Suggested structs:

```rust
pub struct ActMetadata { ... }
pub struct MeanStd { mean: Vec<f32>, std: Vec<f32> }
```

Dependency to add:

- `serde_yaml = "0.9"`

### `src/act/engine.rs`

Create an ACT-specific engine instead of forcing current `Engine` to become generic.

Responsibilities:

- Load model and compile for device
- Enumerate all input/output names
- Accept `HashMap<String, Tensor>` for inputs
- Return one or more outputs by name (`OutputBuffer` reuse is fine)

Key design choice:

- Skip the existing NHWC resize pre/post pipeline for ACT path.
- Feed preprocessed tensors directly in the shape/layout expected by the model.

### `src/act/preprocessing.rs`

Responsibilities:

- Build state tensor:
  - shape `(1, state_dim)`
  - normalize via metadata mean/std
- Build image tensor:
  - load camera images
  - resize to expected H/W
  - convert to CHW float
  - normalize channel-wise/per-feature
  - pack as `(1, n_cameras, 3, H, W)`

### `src/act/postprocessing.rs`

Responsibilities:

- Parse output buffer as `f32`
- reshape `(1, chunk_size, action_dim)`
- denormalize action with metadata mean/std
- produce Rust-native representation for output JSON and queueing

### `src/act/action_queue.rs`

Simple queue wrapper over `VecDeque<Vec<f32>>`:

- enqueue predicted chunk
- dequeue one action per control step
- clear on episode reset

### `src/main.rs` integration

Add:

- `Task::Act`
- CLI arguments for ACT mode, e.g.:
  - `--metadata`
  - `--state-json` or `--state-file`
  - `--camera-image` (repeatable)
  - `--act-steps` (optional simulation loop length)
  - `--output-json`

Flow in `Task::Act`:

1. Load metadata
2. Build ACT engine
3. Preprocess state + images into tensors
4. If queue empty: infer and enqueue chunk
5. Pop next action and print/save JSON

### `src/lib.rs`

Export module:

```rust
pub mod act;
```

## Validation Strategy

### Unit-level

- metadata parsing with sample `metadata.yaml`
- state normalization shape/value checks
- image tensor shape checks for 1 and multi-camera
- action denormalization checks
- queue behavior (enqueue/dequeue/reset)

### Integration-level

Given real Physical AI Studio export dir:

1. Load model + metadata
2. Provide one state vector and camera set
3. Confirm output shape equals `(1, chunk_size, action_dim)`
4. Confirm first action dequeue works and queue length is `chunk_size - 1`

### Performance-level

- Measure:
  - preprocess time
  - infer time
  - postprocess/queue overhead
- Report effective control step latency with queue amortization.

## Implementation Phases

### Phase 1 (MVP: single ACT forward + JSON)

1. Add `serde_yaml`
2. Implement metadata parser
3. Implement ACT engine (multi-input)
4. Implement preprocessing + postprocessing
5. Add `Task::Act` CLI path

Deliverable:

- Can run one ACT inference call from command line and export action chunk JSON.

### Phase 2 (stateful queue behavior)

1. Add `ActionQueue`
2. Add multi-step loop mode in CLI
3. Return per-step action without re-infer until queue empties

Deliverable:

- Emulates runtime control ticks correctly.

### Phase 3 (benchmarking and polish)

1. ACT benchmark mode (latency/throughput)
2. Better error messaging for mismatched metadata/model
3. README docs for ACT task usage

Deliverable:

- Production-ready ACT path with measurable performance.

## Known Risks and Mitigations

1. **Input/output name variance**
   - Mitigation: discover names dynamically; support alias mapping.

2. **Metadata schema drift**
   - Mitigation: tolerant parser with optional fields and clear diagnostics.

3. **Shape mismatch across camera setups**
   - Mitigation: validate all camera inputs against metadata before inference.

4. **Normalization mismatch causes bad robot behavior**
   - Mitigation: strict checks on mean/std lengths and action scale sanity checks.

5. **Control-loop latency concerns**
   - Mitigation: chunk queueing, warmup run, benchmark instrumentation.

## Recommended Next Steps

1. Implement Phase 1 exactly as above.
2. Validate against at least one real Physical AI Studio ACT export directory.
3. Add queue behavior (Phase 2) before integrating with a real robot control loop.
