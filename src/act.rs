use anyhow::{bail, Context, Result};
use image::RgbImage;
use openvino::{CompiledModel, Core, DeviceType, ElementType, InferRequest, Shape, Tensor};
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct ActEngine {
    compiled: CompiledModel,
}

impl ActEngine {
    pub fn new(model_xml: &Path, model_bin: &Path, device: &str) -> Result<Self> {
        let mut core = Core::new().context("failed to initialise OpenVINO core")?;
        let model = core
            .read_model_from_file(&model_xml.to_string_lossy(), &model_bin.to_string_lossy())
            .context("failed to read ACT model from file")?;
        let compiled = core
            .compile_model(&model, parse_device(device))
            .context("failed to compile ACT model")?;
        Ok(Self { compiled })
    }

    pub fn create_request(&mut self) -> Result<InferRequest> {
        self.compiled
            .create_infer_request()
            .context("failed to create ACT infer request")
    }
}

fn parse_device(device: &str) -> DeviceType<'static> {
    match device.to_uppercase().as_str() {
        "CPU" => DeviceType::CPU,
        "GPU" => DeviceType::GPU,
        _ => DeviceType::CPU,
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

#[derive(Debug, Clone, serde::Serialize)]
pub struct ActOutput {
    pub chunk_size: usize,
    pub action_dim: usize,
    pub actions: Vec<Vec<f32>>,
}

pub fn load_metadata(metadata_path: &Path, model_xml_path: &Path) -> Result<ActMetadata> {
    let yaml = fs::read_to_string(metadata_path)
        .with_context(|| format!("failed to read metadata YAML: {}", metadata_path.display()))?;

    let _parsed: RawActMetadata =
        serde_yaml::from_str(&yaml).context("failed to parse metadata YAML")?;

    let xml = fs::read_to_string(model_xml_path)
        .with_context(|| format!("failed to read model XML: {}", model_xml_path.display()))?;

    let state_re =
        Regex::new(r#"<layer[^>]*name=\"state\"[^>]*>[\s\S]*?<data[^>]*shape=\"1,([0-9]+)\""#)
            .context("failed to compile state regex")?;
    let gripper_re = Regex::new(
        r#"<layer[^>]*name=\"images\.gripper\"[^>]*>[\s\S]*?<data[^>]*shape=\"1,3,([0-9]+),([0-9]+)\""#,
    )
    .context("failed to compile gripper regex")?;
    let overview_re = Regex::new(
        r#"<layer[^>]*name=\"images\.overview\"[^>]*>[\s\S]*?<data[^>]*shape=\"1,3,([0-9]+),([0-9]+)\""#,
    )
    .context("failed to compile overview regex")?;
    let action_re = Regex::new(
        r#"<port[^>]*precision=\"FP32\"[^>]*names=\"action\"[^>]*>[\s\S]*?<dim>1</dim>[\s\S]*?<dim>([0-9]+)</dim>[\s\S]*?<dim>([0-9]+)</dim>"#,
    )
    .context("failed to compile action regex")?;

    let state_dim = state_re
        .captures(&xml)
        .and_then(|c| c.get(1))
        .ok_or_else(|| anyhow::anyhow!("failed to discover state shape from model XML"))?
        .as_str()
        .parse::<usize>()
        .context("failed to parse state dim")?;

    let gripper_caps = gripper_re
        .captures(&xml)
        .ok_or_else(|| anyhow::anyhow!("failed to discover images.gripper shape from model XML"))?;
    let gripper_h = gripper_caps
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("missing gripper height"))?
        .as_str()
        .parse::<u32>()
        .context("failed to parse gripper height")?;
    let gripper_w = gripper_caps
        .get(2)
        .ok_or_else(|| anyhow::anyhow!("missing gripper width"))?
        .as_str()
        .parse::<u32>()
        .context("failed to parse gripper width")?;

    let overview_caps = overview_re.captures(&xml).ok_or_else(|| {
        anyhow::anyhow!("failed to discover images.overview shape from model XML")
    })?;
    let overview_h = overview_caps
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("missing overview height"))?
        .as_str()
        .parse::<u32>()
        .context("failed to parse overview height")?;
    let overview_w = overview_caps
        .get(2)
        .ok_or_else(|| anyhow::anyhow!("missing overview width"))?
        .as_str()
        .parse::<u32>()
        .context("failed to parse overview width")?;

    if gripper_h != overview_h || gripper_w != overview_w {
        bail!(
            "camera shapes differ (gripper={}x{}, overview={}x{}), unsupported",
            gripper_w,
            gripper_h,
            overview_w,
            overview_h
        );
    }

    let action_caps = action_re
        .captures(&xml)
        .ok_or_else(|| anyhow::anyhow!("failed to discover action output shape from model XML"))?;
    let chunk_size = action_caps
        .get(1)
        .ok_or_else(|| anyhow::anyhow!("missing action chunk size"))?
        .as_str()
        .parse::<usize>()
        .context("failed to parse action chunk size")?;
    let action_dim = action_caps
        .get(2)
        .ok_or_else(|| anyhow::anyhow!("missing action dim"))?
        .as_str()
        .parse::<usize>()
        .context("failed to parse action dim")?;

    Ok(ActMetadata {
        state_dim,
        image_height: gripper_h,
        image_width: gripper_w,
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
    let f32_buf = bytemuck_cast_mut(buf);
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
    let f32_buf = bytemuck_cast_mut(buf);

    let hw = (width * height) as usize;
    for (idx, p) in resized.pixels().enumerate() {
        f32_buf[idx] = p[0] as f32 / 255.0;
        f32_buf[hw + idx] = p[1] as f32 / 255.0;
        f32_buf[2 * hw + idx] = p[2] as f32 / 255.0;
    }

    Ok(tensor)
}

pub fn run_act_once(
    engine: &mut ActEngine,
    meta: &ActMetadata,
    inputs: &ActInputs,
) -> Result<ActOutput> {
    let state_tensor = state_to_tensor(&inputs.state, meta.state_dim)?;
    let gripper_tensor =
        image_to_nchw_tensor(&inputs.gripper_image, meta.image_width, meta.image_height)?;
    let overview_tensor =
        image_to_nchw_tensor(&inputs.overview_image, meta.image_width, meta.image_height)?;

    let mut request: InferRequest = engine.create_request()?;
    request
        .set_tensor("state", &state_tensor)
        .context("failed to set ACT input 'state'")?;
    request
        .set_tensor("images.gripper", &gripper_tensor)
        .context("failed to set ACT input 'images.gripper'")?;
    request
        .set_tensor("images.overview", &overview_tensor)
        .context("failed to set ACT input 'images.overview'")?;

    request.infer().context("ACT inference failed")?;

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

pub fn bytemuck_cast_mut(bytes: &mut [u8]) -> &mut [f32] {
    assert!(
        bytes.len() % std::mem::size_of::<f32>() == 0,
        "buffer length is not a multiple of f32 size"
    );
    assert!(
        bytes.as_ptr() as usize % std::mem::align_of::<f32>() == 0,
        "buffer is not aligned for f32"
    );
    unsafe {
        std::slice::from_raw_parts_mut(
            bytes.as_mut_ptr() as *mut f32,
            bytes.len() / std::mem::size_of::<f32>(),
        )
    }
}
