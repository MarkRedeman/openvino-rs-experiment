# ACT Inference Walkthrough

This document explains how ACT inference works end-to-end in this repository:

1. load model + metadata
2. load episode state + camera inputs
3. run inference
4. read output

It also highlights how to avoid swapping gripper and overview camera inputs.

## Input/output contract

ACT runs with named inputs (not positional):

- model input `state` <- robot state vector
- model input `images.gripper` <- gripper camera image
- model input `images.overview` <- overview camera image

Output is a single tensor named `action` with shape `[1, chunk_size, action_dim]`.

Because tensor binding is by name, camera ordering is explicit once names are mapped correctly.

## 1) Load model and metadata

```rust
use anyhow::Result;
use std::path::Path;

use inference_rs::act::{load_metadata, ActEngine};

fn load_act() -> Result<(ActEngine, inference_rs::act::ActMetadata)> {
    let model = Path::new("models/act-openvino/act.xml");
    let weights = Path::new("models/act-openvino/act.bin");
    let metadata = Path::new("models/act-openvino/metadata.yaml");

    // Reads YAML and inspects model XML to discover dimensions.
    let meta = load_metadata(metadata, model)?;
    let engine = ActEngine::new(model, weights, "CPU")?;
    Ok((engine, meta))
}
```

`load_metadata()` discovers model-side shapes from XML (for example):

- `state [1,6]`
- `images.gripper [1,3,480,640]`
- `images.overview [1,3,480,640]`
- `action [1,100,6]`

## 2) Load episode data and map camera inputs

```rust
use anyhow::{Context, Result};
use std::path::Path;

use inference_rs::act::{
    load_sample_images_from_episode, parse_state_from_episode, prepare_act_input_tensors, ActInputs,
};

fn load_episode_inputs(
    meta: &inference_rs::act::ActMetadata,
) -> Result<inference_rs::act::ActInputTensors> {
    let episode_dir = Path::new("episodes/ep_000_dc7198bd");

    let state = parse_state_from_episode(&episode_dir.join("data.jsonl"))?;
    let images =
        load_sample_images_from_episode(episode_dir, meta.image_width, meta.image_height)?;

    // Explicit name-based mapping:
    // "gripper"  -> ActInputs.gripper_image  -> model input "images.gripper"
    // "overview" -> ActInputs.overview_image -> model input "images.overview"
    let gripper = images
        .get("gripper")
        .context("missing gripper camera sample")?
        .clone();
    let overview = images
        .get("overview")
        .context("missing overview camera sample")?
        .clone();

    let inputs = ActInputs {
        state,
        gripper_image: gripper,
        overview_image: overview,
    };

    prepare_act_input_tensors(meta, &inputs)
}
```

Episode camera source behavior:

- prefers still images: `cam_gripper.jpg/.png` and `cam_overview.jpg/.png`
- if those are missing, falls back to synthetic images built from `stats.json` camera means
- `cam_gripper.mp4`/`cam_overview.mp4` are currently not decoded directly in this inference path

## 3) Run inference and decode output

```rust
use anyhow::Result;

use inference_rs::act::read_act_output;

fn run_once(
    engine: &mut inference_rs::act::ActEngine,
    meta: &inference_rs::act::ActMetadata,
    tensors: &inference_rs::act::ActInputTensors,
) -> Result<inference_rs::act::ActOutput> {
    let mut request = engine.create_request()?;
    engine.run_request(&mut request, tensors)?;
    read_act_output(&request, meta)
}
```

## 4) Output structure example

ACT output is a chunked action plan:

- `chunk_size`: number of future steps (commonly `100`)
- `action_dim`: per-step action vector size (commonly `6`)
- `actions`: `Vec<Vec<f32>>` with shape `[chunk_size][action_dim]`

Example (truncated):

```json
{
  "chunk_size": 100,
  "action_dim": 6,
  "actions": [
    [0.013, -0.227, 0.041, 0.808, -0.021, 0.104],
    [0.017, -0.219, 0.038, 0.801, -0.018, 0.101],
    [0.020, -0.212, 0.035, 0.793, -0.016, 0.097]
  ]
}
```

## Camera mapping safety checklist

To prevent gripper/overview swaps, verify all three points:

1. Episode keys from loader: `gripper`, `overview`
2. `ActInputs` assignment: `gripper_image <- gripper`, `overview_image <- overview`
3. OpenVINO binding names in engine: `set_tensor("images.gripper", ...)` and `set_tensor("images.overview", ...)`

If these three are aligned, camera ordering is correct regardless of variable ordering in surrounding code.
