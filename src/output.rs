use anyhow::{Context, Result};
use serde::Serialize;
use std::fs;
use std::path::Path;

use crate::act::ActOutput;
use crate::benchmark::{BenchmarkConfig, BenchmarkReport, StageTimingReport};
use crate::postprocessing::{Classification, Detection};

#[derive(Debug, Serialize)]
struct DetectionJsonRecord {
    class_id: usize,
    label: Option<String>,
    confidence: f32,
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

#[derive(Debug, Serialize)]
struct ClassificationJsonRecord {
    class_id: usize,
    label: Option<String>,
    probability: f32,
}

#[derive(Debug, Serialize)]
struct ClassificationJsonOutput {
    task: &'static str,
    top_k: usize,
    image: String,
    model: String,
    model_input_width: u32,
    model_input_height: u32,
    count: usize,
    results: Vec<ClassificationJsonRecord>,
}

#[derive(Debug, Serialize)]
struct DetectionJsonOutput {
    task: &'static str,
    detection_format: String,
    threshold: f32,
    image: String,
    model: String,
    model_input_width: u32,
    model_input_height: u32,
    count: usize,
    detections: Vec<DetectionJsonRecord>,
}

#[derive(Debug, Serialize)]
struct BenchmarkJsonOutput {
    task: &'static str,
    image: String,
    model: String,
    device: String,
    preprocess_backend: String,
    model_input_width: u32,
    model_input_height: u32,
    config: BenchmarkConfig,
    report: BenchmarkReport,
    stage_timing: Option<StageTimingReport>,
}

#[derive(Debug, Serialize)]
struct ActJsonOutput {
    task: &'static str,
    model: String,
    metadata: String,
    episode_dir: String,
    state_dim: usize,
    chunk_size: usize,
    action_dim: usize,
    actions: Vec<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct ActBenchmarkJsonOutput {
    task: &'static str,
    model: String,
    metadata: String,
    episode_dir: String,
    device: String,
    config: BenchmarkConfig,
    report: BenchmarkReport,
    stage_timing: Option<StageTimingReport>,
    output_chunk_size: usize,
    output_action_dim: usize,
}

fn write_json_file(output_path: &Path, content: &str, kind: &str) -> Result<()> {
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create {kind} JSON directory: {}",
                    parent.display()
                )
            })?;
        }
    }

    fs::write(output_path, content).with_context(|| {
        format!(
            "failed to write {kind} JSON output: {}",
            output_path.display()
        )
    })?;

    Ok(())
}

pub fn write_classification_json(
    output_path: &Path,
    results: &[Classification],
    labels: &[String],
    top_k: usize,
    image_path: &Path,
    model_path: &Path,
    model_width: u32,
    model_height: u32,
) -> Result<()> {
    let records = results
        .iter()
        .map(|c| ClassificationJsonRecord {
            class_id: c.class_id,
            label: labels.get(c.class_id).cloned(),
            probability: c.probability,
        })
        .collect();

    let payload = ClassificationJsonOutput {
        task: "classify",
        top_k,
        image: image_path.display().to_string(),
        model: model_path.display().to_string(),
        model_input_width: model_width,
        model_input_height: model_height,
        count: results.len(),
        results: records,
    };

    let json = serde_json::to_string_pretty(&payload)
        .context("failed to serialize classification JSON")?;
    write_json_file(output_path, &json, "classification")
}

pub fn write_detection_json(
    output_path: &Path,
    detections: &[Detection],
    labels: &[String],
    detection_format: &str,
    threshold: f32,
    image_path: &Path,
    model_path: &Path,
    model_width: u32,
    model_height: u32,
) -> Result<()> {
    let records = detections
        .iter()
        .map(|d| DetectionJsonRecord {
            class_id: d.class_id,
            label: labels.get(d.class_id).cloned(),
            confidence: d.confidence,
            x1: d.x1,
            y1: d.y1,
            x2: d.x2,
            y2: d.y2,
        })
        .collect();

    let payload = DetectionJsonOutput {
        task: "detect",
        detection_format: detection_format.to_string(),
        threshold,
        image: image_path.display().to_string(),
        model: model_path.display().to_string(),
        model_input_width: model_width,
        model_input_height: model_height,
        count: detections.len(),
        detections: records,
    };

    let json =
        serde_json::to_string_pretty(&payload).context("failed to serialize detection JSON")?;
    write_json_file(output_path, &json, "detection")
}

#[allow(clippy::too_many_arguments)]
pub fn write_benchmark_json(
    output_path: &Path,
    report: &BenchmarkReport,
    stage_timing: Option<StageTimingReport>,
    cfg: &BenchmarkConfig,
    image_path: &Path,
    model_path: &Path,
    device: &str,
    preprocess_backend: &str,
    model_width: u32,
    model_height: u32,
) -> Result<()> {
    let payload = BenchmarkJsonOutput {
        task: "benchmark",
        image: image_path.display().to_string(),
        model: model_path.display().to_string(),
        device: device.to_string(),
        preprocess_backend: preprocess_backend.to_string(),
        model_input_width: model_width,
        model_input_height: model_height,
        config: cfg.clone(),
        report: report.clone(),
        stage_timing,
    };

    let json =
        serde_json::to_string_pretty(&payload).context("failed to serialize benchmark JSON")?;
    write_json_file(output_path, &json, "benchmark")
}

pub fn write_act_json(
    output_path: &Path,
    model: &Path,
    metadata: &Path,
    episode_dir: &Path,
    state_dim: usize,
    out: ActOutput,
) -> Result<()> {
    let payload = ActJsonOutput {
        task: "act",
        model: model.display().to_string(),
        metadata: metadata.display().to_string(),
        episode_dir: episode_dir.display().to_string(),
        state_dim,
        chunk_size: out.chunk_size,
        action_dim: out.action_dim,
        actions: out.actions,
    };
    let json = serde_json::to_string_pretty(&payload).context("failed to serialize ACT JSON")?;
    write_json_file(output_path, &json, "act")
}

#[allow(clippy::too_many_arguments)]
pub fn write_act_benchmark_json(
    output_path: &Path,
    model: &Path,
    metadata: &Path,
    episode_dir: &Path,
    device: &str,
    cfg: BenchmarkConfig,
    report: BenchmarkReport,
    stage_timing: Option<StageTimingReport>,
    output_chunk_size: usize,
    output_action_dim: usize,
) -> Result<()> {
    let payload = ActBenchmarkJsonOutput {
        task: "benchmark",
        model: model.display().to_string(),
        metadata: metadata.display().to_string(),
        episode_dir: episode_dir.display().to_string(),
        device: device.to_string(),
        config: cfg,
        report,
        stage_timing,
        output_chunk_size,
        output_action_dim,
    };

    let json =
        serde_json::to_string_pretty(&payload).context("failed to serialize ACT benchmark JSON")?;
    write_json_file(output_path, &json, "act benchmark")
}
