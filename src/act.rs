use anyhow::{Context, Result, bail};
use image::RgbImage;
use openvino::{CompiledModel, Core, ElementType, InferRequest, Model, Shape, Tensor};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use crate::infra::device::parse_device;
use crate::infra::tensor_utils::cast_bytes_mut_to_f32;

pub struct ActEngine {
    compiled: CompiledModel,
    metadata: ActMetadata,
}

impl ActEngine {
    pub fn new(model_xml: &Path, model_bin: &Path, device: &str) -> Result<Self> {
        let mut core = Core::new().context("failed to initialise OpenVINO core")?;
        let model = core
            .read_model_from_file(&model_xml.to_string_lossy(), &model_bin.to_string_lossy())
            .context("failed to read ACT model from file")?;
        let metadata = discover_act_metadata(&model)?;
        let compiled = core
            .compile_model(&model, parse_device(device))
            .context("failed to compile ACT model")?;
        Ok(Self { compiled, metadata })
    }

    pub fn metadata(&self) -> &ActMetadata {
        &self.metadata
    }

    pub fn create_request(&mut self) -> Result<InferRequest> {
        self.compiled
            .create_infer_request()
            .context("failed to create ACT infer request")
    }

    pub fn run_request(&self, request: &mut InferRequest, tensors: &ActInputTensors) -> Result<()> {
        request
            .set_tensor("state", &tensors.state)
            .context("failed to set ACT input 'state'")?;
        request
            .set_tensor("images.gripper", &tensors.gripper)
            .context("failed to set ACT input 'images.gripper'")?;
        request
            .set_tensor("images.overview", &tensors.overview)
            .context("failed to set ACT input 'images.overview'")?;
        request.infer().context("ACT inference failed")
    }
}

#[derive(Debug, Clone)]
pub struct ActMetadata {
    pub state_dim: usize,
    pub image_height: u32,
    pub image_width: u32,
    pub camera_names: Vec<String>,
    pub action_dim: usize,
    pub chunk_size: usize,
}

#[derive(Debug, Deserialize)]
struct RawActMetadata {
    #[allow(dead_code)]
    backend: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ActInputs {
    pub state: Vec<f32>,
    pub gripper_image: RgbImage,
    pub overview_image: RgbImage,
}

pub struct ActInputTensors {
    pub state: Tensor,
    pub gripper: Tensor,
    pub overview: Tensor,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ActOutput {
    pub chunk_size: usize,
    pub action_dim: usize,
    pub actions: Vec<Vec<f32>>,
}

pub fn load_metadata(metadata_path: &Path) -> Result<()> {
    let yaml = fs::read_to_string(metadata_path)
        .with_context(|| format!("failed to read metadata YAML: {}", metadata_path.display()))?;

    let _parsed: RawActMetadata =
        serde_yaml::from_str(&yaml).context("failed to parse metadata YAML")?;

    Ok(())
}

fn shape_dims(shape: &Shape) -> &[i64] {
    shape.get_dimensions()
}

fn discover_act_metadata(model: &Model) -> Result<ActMetadata> {
    let mut state_dim: Option<usize> = None;
    let mut gripper_hw: Option<(u32, u32)> = None;
    let mut overview_hw: Option<(u32, u32)> = None;
    let mut action_shape: Option<(usize, usize)> = None;

    let input_count = model
        .get_inputs_len()
        .context("failed to read ACT model input count")?;
    for i in 0..input_count {
        let node = model
            .get_input_by_index(i)
            .with_context(|| format!("failed to read ACT input at index {i}"))?;
        let name = node
            .get_name()
            .with_context(|| format!("failed to read ACT input name at index {i}"))?;
        let shape = node
            .get_shape()
            .with_context(|| format!("failed to read ACT input shape for '{name}'"))?;
        let dims = shape_dims(&shape);

        match name.as_str() {
            "state" => {
                if dims.len() != 2 || dims[0] != 1 || dims[1] <= 0 {
                    bail!(
                        "unexpected ACT input shape for 'state': {:?} (expected [1, state_dim])",
                        dims
                    );
                }
                state_dim = Some(dims[1] as usize);
            }
            "images.gripper" => {
                if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 || dims[2] <= 0 || dims[3] <= 0 {
                    bail!(
                        "unexpected ACT input shape for 'images.gripper': {:?} (expected [1, 3, H, W])",
                        dims
                    );
                }
                gripper_hw = Some((dims[2] as u32, dims[3] as u32));
            }
            "images.overview" => {
                if dims.len() != 4 || dims[0] != 1 || dims[1] != 3 || dims[2] <= 0 || dims[3] <= 0 {
                    bail!(
                        "unexpected ACT input shape for 'images.overview': {:?} (expected [1, 3, H, W])",
                        dims
                    );
                }
                overview_hw = Some((dims[2] as u32, dims[3] as u32));
            }
            _ => {}
        }
    }

    let output_count = model
        .get_outputs_len()
        .context("failed to read ACT model output count")?;
    for i in 0..output_count {
        let node = model
            .get_output_by_index(i)
            .with_context(|| format!("failed to read ACT output at index {i}"))?;
        let name = node
            .get_name()
            .with_context(|| format!("failed to read ACT output name at index {i}"))?;
        if name != "action" {
            continue;
        }

        let shape = node
            .get_shape()
            .context("failed to read ACT output shape for 'action'")?;
        let dims = shape_dims(&shape);
        if dims.len() != 3 || dims[0] != 1 || dims[1] <= 0 || dims[2] <= 0 {
            bail!(
                "unexpected ACT output shape for 'action': {:?} (expected [1, chunk_size, action_dim])",
                dims
            );
        }
        action_shape = Some((dims[1] as usize, dims[2] as usize));
    }

    let state_dim =
        state_dim.ok_or_else(|| anyhow::anyhow!("ACT model is missing input 'state'"))?;
    let gripper_hw =
        gripper_hw.ok_or_else(|| anyhow::anyhow!("ACT model is missing input 'images.gripper'"))?;
    let overview_hw = overview_hw
        .ok_or_else(|| anyhow::anyhow!("ACT model is missing input 'images.overview'"))?;

    if gripper_hw != overview_hw {
        bail!(
            "camera shapes differ (gripper={}x{}, overview={}x{}), unsupported",
            gripper_hw.1,
            gripper_hw.0,
            overview_hw.1,
            overview_hw.0
        );
    }

    let (chunk_size, action_dim) =
        action_shape.ok_or_else(|| anyhow::anyhow!("ACT model is missing output 'action'"))?;

    Ok(ActMetadata {
        state_dim,
        image_height: gripper_hw.0,
        image_width: gripper_hw.1,
        camera_names: vec!["gripper".to_string(), "overview".to_string()],
        action_dim,
        chunk_size,
    })
}

fn state_to_tensor(state: &[f32], expected_dim: usize) -> Result<Tensor> {
    if state.len() != expected_dim {
        bail!(
            "state length {} does not match model state dim {}",
            state.len(),
            expected_dim
        );
    }

    let shape = Shape::new(&[1, expected_dim as i64]).context("failed to create state shape")?;
    let mut tensor =
        Tensor::new(ElementType::F32, &shape).context("failed to allocate state tensor")?;
    let buf = tensor
        .get_raw_data_mut()
        .context("failed to get mutable state tensor buffer")?;
    let f32_buf = cast_bytes_mut_to_f32(buf);
    f32_buf.copy_from_slice(state);
    Ok(tensor)
}

fn image_to_nchw_tensor(img: &RgbImage, width: u32, height: u32) -> Result<Tensor> {
    let resized = if img.width() == width && img.height() == height {
        img.clone()
    } else {
        image::imageops::resize(img, width, height, image::imageops::FilterType::Triangle)
    };

    let shape = Shape::new(&[1, 3, height as i64, width as i64])
        .context("failed to create image tensor shape")?;
    let mut tensor =
        Tensor::new(ElementType::F32, &shape).context("failed to allocate image tensor")?;
    let buf = tensor
        .get_raw_data_mut()
        .context("failed to get mutable image tensor buffer")?;
    let f32_buf = cast_bytes_mut_to_f32(buf);

    let hw = (width * height) as usize;
    for (idx, p) in resized.pixels().enumerate() {
        f32_buf[idx] = p[0] as f32 / 255.0;
        f32_buf[hw + idx] = p[1] as f32 / 255.0;
        f32_buf[2 * hw + idx] = p[2] as f32 / 255.0;
    }

    Ok(tensor)
}

pub fn prepare_act_input_tensors(
    meta: &ActMetadata,
    inputs: &ActInputs,
) -> Result<ActInputTensors> {
    Ok(ActInputTensors {
        state: state_to_tensor(&inputs.state, meta.state_dim)?,
        gripper: image_to_nchw_tensor(&inputs.gripper_image, meta.image_width, meta.image_height)?,
        overview: image_to_nchw_tensor(
            &inputs.overview_image,
            meta.image_width,
            meta.image_height,
        )?,
    })
}

pub fn read_act_output(request: &InferRequest, meta: &ActMetadata) -> Result<ActOutput> {
    let output_tensor = request
        .get_tensor("action")
        .context("failed to retrieve ACT output tensor 'action'")?;
    let vals = output_tensor
        .get_data::<f32>()
        .context("failed to decode ACT output tensor as f32")?;
    let expected = meta.chunk_size * meta.action_dim;
    if vals.len() != expected {
        bail!(
            "unexpected ACT output length: got {}, expected {} ({}x{})",
            vals.len(),
            expected,
            meta.chunk_size,
            meta.action_dim
        );
    }

    let mut actions = Vec::with_capacity(meta.chunk_size);
    for row in vals.chunks_exact(meta.action_dim) {
        actions.push(row.to_vec());
    }

    Ok(ActOutput {
        chunk_size: meta.chunk_size,
        action_dim: meta.action_dim,
        actions,
    })
}

pub fn run_act_once(
    engine: &mut ActEngine,
    meta: &ActMetadata,
    inputs: &ActInputs,
) -> Result<ActOutput> {
    let tensors = prepare_act_input_tensors(meta, inputs)?;
    let mut request: InferRequest = engine.create_request()?;
    engine.run_request(&mut request, &tensors)?;
    read_act_output(&request, meta)
}

pub fn parse_state_from_episode(data_jsonl: &Path) -> Result<Vec<f32>> {
    #[derive(Debug, Deserialize)]
    struct Row {
        state: Vec<f32>,
    }

    let content = fs::read_to_string(data_jsonl)
        .with_context(|| format!("failed to read episode data: {}", data_jsonl.display()))?;
    let first_line = content
        .lines()
        .find(|l| !l.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("episode data file is empty: {}", data_jsonl.display()))?;
    let row: Row = serde_json::from_str(first_line).context("failed to parse first JSONL row")?;
    Ok(row.state)
}

#[derive(Debug, Deserialize)]
struct EpisodeStats {
    images: EpisodeImages,
}

#[derive(Debug, Deserialize)]
struct EpisodeImages {
    gripper: EpisodeCameraStats,
    overview: EpisodeCameraStats,
}

#[derive(Debug, Deserialize)]
struct EpisodeCameraStats {
    mean: Vec<f32>,
}

fn make_uniform_rgb(width: u32, height: u32, mean_rgb_0_1: &[f32]) -> Result<RgbImage> {
    if mean_rgb_0_1.len() != 3 {
        bail!(
            "camera mean length {} is invalid, expected 3",
            mean_rgb_0_1.len()
        );
    }
    let r = (mean_rgb_0_1[0].clamp(0.0, 1.0) * 255.0).round() as u8;
    let g = (mean_rgb_0_1[1].clamp(0.0, 1.0) * 255.0).round() as u8;
    let b = (mean_rgb_0_1[2].clamp(0.0, 1.0) * 255.0).round() as u8;
    Ok(RgbImage::from_pixel(width, height, image::Rgb([r, g, b])))
}

pub fn load_sample_images_from_episode(
    episode_dir: &Path,
    width: u32,
    height: u32,
) -> Result<HashMap<String, RgbImage>> {
    let gripper_try = image::open(&episode_dir.join("cam_gripper.jpg"))
        .or_else(|_| image::open(&episode_dir.join("cam_gripper.png")));
    let overview_try = image::open(&episode_dir.join("cam_overview.jpg"))
        .or_else(|_| image::open(&episode_dir.join("cam_overview.png")));

    let (gripper, overview) = if let (Ok(g), Ok(o)) = (gripper_try, overview_try) {
        (g.to_rgb8(), o.to_rgb8())
    } else {
        let stats_path = episode_dir.join("stats.json");
        let stats_text = fs::read_to_string(&stats_path)
            .with_context(|| format!("failed to read episode stats: {}", stats_path.display()))?;
        let stats: EpisodeStats =
            serde_json::from_str(&stats_text).context("failed to parse episode stats JSON")?;
        (
            make_uniform_rgb(width, height, &stats.images.gripper.mean)?,
            make_uniform_rgb(width, height, &stats.images.overview.mean)?,
        )
    };

    let mut map = HashMap::new();
    map.insert("gripper".to_string(), gripper);
    map.insert("overview".to_string(), overview);
    Ok(map)
}
