use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use inference_rs::act::{
    load_metadata, load_sample_images_from_episode, parse_state_from_episode, run_act_once,
    ActEngine, ActInputs,
};
use inference_rs::benchmark::{run_benchmark, BenchmarkConfig, BenchmarkReport};
use inference_rs::engine::Engine;
use inference_rs::labels::parse_labels_from_model_xml;
use inference_rs::postprocessing::{
    decode_geti_detections, decode_ssd_detections, decode_yolo_detections, top_k_classifications,
};
use inference_rs::preprocessing::{load_image, load_image_no_resize, rgb8_to_tensor};
use inference_rs::visualization::draw_detections;

/// Run vision-model inference with OpenVINO.
#[derive(Parser, Debug)]
#[command(name = "inference-rs", version, about)]
struct Args {
    /// Path to the OpenVINO IR model definition (.xml).
    #[arg(long)]
    model: PathBuf,

    /// Path to the OpenVINO IR weights (.bin).
    #[arg(long)]
    weights: PathBuf,

    /// Path to the input image (JPEG, PNG, …).
    /// Required for classify/detect/benchmark. Not used for --task act.
    #[arg(long)]
    image: Option<PathBuf>,

    /// Inference device.
    #[arg(long, default_value = "CPU")]
    device: String,

    /// Task to perform.
    #[arg(long, value_enum, default_value_t = Task::Classify)]
    task: Task,

    /// Number of top classification results to display.
    #[arg(long, default_value_t = 5)]
    top_k: usize,

    /// Confidence threshold for detection tasks.
    #[arg(long, default_value_t = 0.5)]
    threshold: f32,

    /// Model input width in pixels.
    #[arg(long, default_value_t = 224)]
    width: u32,

    /// Model input height in pixels.
    #[arg(long, default_value_t = 224)]
    height: u32,

    /// Detection output format (only used when --task detect).
    #[arg(long, value_enum, default_value_t = DetectionFormat::Geti)]
    detection_format: DetectionFormat,

    /// Number of classes (only used when --detection-format yolo).
    #[arg(long, default_value_t = 80)]
    num_classes: usize,

    /// Save an annotated image with bounding boxes drawn on top (only for --task detect).
    /// The output format is determined by the file extension (e.g. .png, .jpg).
    #[arg(long)]
    output_image: Option<PathBuf>,

    /// Save results as JSON (supports both --task classify and --task detect).
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Benchmark duration in seconds (used when --task benchmark and --benchmark-iters is not set).
    #[arg(long, default_value_t = 10.0)]
    benchmark_duration: f64,

    /// Number of warmup iterations before measuring (benchmark mode only).
    #[arg(long, default_value_t = 20)]
    benchmark_warmup: usize,

    /// Number of measured benchmark iterations (overrides --benchmark-duration when set).
    #[arg(long)]
    benchmark_iters: Option<usize>,

    /// Progress report interval in seconds (benchmark mode only, 0 disables periodic logs).
    #[arg(long, default_value_t = 1.0)]
    benchmark_report_every: f64,

    /// Number of iterations used for rough stage timing (pre/infer/post) in benchmark mode.
    /// Set to 0 to disable stage timing.
    #[arg(long, default_value_t = 30)]
    benchmark_stage_iters: usize,

    /// In benchmark stage timing, read+decode the image every iteration.
    /// By default, decode happens once and only resize+tensor conversion are repeated.
    #[arg(long, default_value_t = false)]
    benchmark_stage_read_each_iter: bool,

    /// Preprocessing backend.
    /// - rust: resize + normalize in Rust before inference.
    /// - openvino: decode in Rust, delegate resize to OpenVINO preprocess pipeline.
    #[arg(long, value_enum, default_value_t = PreprocessBackend::Rust)]
    preprocess_backend: PreprocessBackend,

    /// Path to ACT metadata YAML. Defaults to sibling metadata.yaml next to --model.
    #[arg(long)]
    metadata: Option<PathBuf>,

    /// Episode directory used to source ACT sample inputs (state and camera stats/images).
    #[arg(long)]
    episode_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, ValueEnum)]
enum Task {
    Classify,
    Detect,
    Benchmark,
    Act,
}

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
enum DetectionFormat {
    /// Intel Geti (OTX) format: "boxes" [1,N,5] + "labels" [1,N].
    Geti,
    /// SSD-style: output shape [1, 1, N, 7].
    Ssd,
    /// YOLO-style: output shape [1, N, 5 + num_classes].
    Yolo,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize)]
enum PreprocessBackend {
    Rust,
    Openvino,
}

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
    preprocess_backend: PreprocessBackend,
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

#[derive(Debug, Clone, Serialize)]
struct StageTimingReport {
    iterations: usize,
    stage_read_each_iter: bool,
    io_decode_mean_ms: f64,
    preprocess_mean_ms: f64,
    inference_mean_ms: f64,
    postprocess_mean_ms: f64,
    total_mean_ms: f64,
    io_decode_share_pct: f64,
    preprocess_share_pct: f64,
    inference_share_pct: f64,
    postprocess_share_pct: f64,
}

fn write_json_file(output_path: &PathBuf, content: &str, kind: &str) -> Result<()> {
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

fn write_classification_json(
    output_path: &PathBuf,
    results: &[inference_rs::postprocessing::Classification],
    labels: &[String],
    top_k: usize,
    image_path: &PathBuf,
    model_path: &PathBuf,
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

fn write_detection_json(
    output_path: &PathBuf,
    detections: &[inference_rs::postprocessing::Detection],
    labels: &[String],
    detection_format: &DetectionFormat,
    threshold: f32,
    image_path: &PathBuf,
    model_path: &PathBuf,
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
        detection_format: format!("{detection_format:?}").to_lowercase(),
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

fn write_benchmark_json(
    output_path: &PathBuf,
    report: &BenchmarkReport,
    stage_timing: Option<StageTimingReport>,
    cfg: &BenchmarkConfig,
    image_path: &PathBuf,
    model_path: &PathBuf,
    device: &str,
    preprocess_backend: PreprocessBackend,
    model_width: u32,
    model_height: u32,
) -> Result<()> {
    let payload = BenchmarkJsonOutput {
        task: "benchmark",
        image: image_path.display().to_string(),
        model: model_path.display().to_string(),
        device: device.to_string(),
        preprocess_backend,
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

fn run_stage_timing(
    args: &Args,
    image_path: &PathBuf,
    engine: &mut Engine,
    labels: &[String],
) -> Result<Option<StageTimingReport>> {
    if args.benchmark_stage_iters == 0 {
        return Ok(None);
    }

    let mut io_decode_ms_sum = 0.0f64;
    let mut preprocess_ms_sum = 0.0f64;
    let mut inference_ms_sum = 0.0f64;
    let mut postprocess_ms_sum = 0.0f64;
    let mut total_ms_sum = 0.0f64;

    let decoded_once = if args.benchmark_stage_read_each_iter {
        None
    } else {
        let t = Instant::now();
        let img = image::open(image_path)
            .with_context(|| format!("failed to open image: {}", image_path.display()))?;
        let rgb = img.to_rgb8();
        io_decode_ms_sum += t.elapsed().as_secs_f64() * 1000.0;
        Some(rgb)
    };

    let use_openvino_preprocess = matches!(args.preprocess_backend, PreprocessBackend::Openvino);

    for _ in 0..args.benchmark_stage_iters {
        let total_start = Instant::now();

        let tensor = if args.benchmark_stage_read_each_iter {
            let t = Instant::now();
            let img = image::open(image_path)
                .with_context(|| format!("failed to open image: {}", image_path.display()))?;
            let src_rgb = img.to_rgb8();
            io_decode_ms_sum += t.elapsed().as_secs_f64() * 1000.0;

            let t0 = Instant::now();
            let tensor = if use_openvino_preprocess {
                rgb8_to_tensor(&src_rgb)?
            } else {
                let resized = image::imageops::resize(
                    &src_rgb,
                    args.width,
                    args.height,
                    image::imageops::FilterType::Triangle,
                );
                rgb8_to_tensor(&resized)?
            };
            let t1 = Instant::now();
            preprocess_ms_sum += (t1 - t0).as_secs_f64() * 1000.0;
            tensor
        } else {
            let src_rgb = decoded_once
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("benchmark stage timing: missing decoded image"))?;

            let t0 = Instant::now();
            let tensor = if use_openvino_preprocess {
                rgb8_to_tensor(src_rgb)?
            } else {
                let resized = image::imageops::resize(
                    src_rgb,
                    args.width,
                    args.height,
                    image::imageops::FilterType::Triangle,
                );
                rgb8_to_tensor(&resized)?
            };
            let t1 = Instant::now();
            preprocess_ms_sum += (t1 - t0).as_secs_f64() * 1000.0;
            tensor
        };

        let use_detection_path =
            args.detection_format != DetectionFormat::Geti || engine.output_names().len() > 1;

        if use_detection_path {
            let t2 = Instant::now();
            match args.detection_format {
                DetectionFormat::Geti => {
                    let outputs = engine.infer_multi(&tensor)?;
                    let t3 = Instant::now();

                    let boxes_buf = outputs
                        .get("boxes")
                        .or_else(|| outputs.get("bboxes"))
                        .ok_or_else(|| {
                            anyhow::anyhow!("benchmark stage timing: missing boxes/bboxes output")
                        })?;
                    let labels_buf = outputs.get("labels").ok_or_else(|| {
                        anyhow::anyhow!("benchmark stage timing: missing labels output")
                    })?;
                    let _detections = decode_geti_detections(
                        boxes_buf.as_f32(),
                        labels_buf.as_i64(),
                        args.threshold,
                    );

                    inference_ms_sum += (t3 - t2).as_secs_f64() * 1000.0;
                    postprocess_ms_sum += (Instant::now() - t3).as_secs_f64() * 1000.0;
                }
                DetectionFormat::Ssd => {
                    let output = engine.infer(&tensor)?;
                    let t3 = Instant::now();
                    let _detections = decode_ssd_detections(&output, args.threshold);

                    inference_ms_sum += (t3 - t2).as_secs_f64() * 1000.0;
                    postprocess_ms_sum += (Instant::now() - t3).as_secs_f64() * 1000.0;
                }
                DetectionFormat::Yolo => {
                    let output = engine.infer(&tensor)?;
                    let t3 = Instant::now();
                    let _detections =
                        decode_yolo_detections(&output, args.num_classes, args.threshold);

                    inference_ms_sum += (t3 - t2).as_secs_f64() * 1000.0;
                    postprocess_ms_sum += (Instant::now() - t3).as_secs_f64() * 1000.0;
                }
            }
        } else {
            let t2 = Instant::now();
            let output = engine.infer(&tensor)?;
            let t3 = Instant::now();
            let _results = top_k_classifications(&output, args.top_k);
            let _ = labels;

            inference_ms_sum += (t3 - t2).as_secs_f64() * 1000.0;
            postprocess_ms_sum += (Instant::now() - t3).as_secs_f64() * 1000.0;
        }

        total_ms_sum += total_start.elapsed().as_secs_f64() * 1000.0;
    }

    let n = args.benchmark_stage_iters as f64;
    let io_decode_mean_ms = io_decode_ms_sum / n;
    let preprocess_mean_ms = preprocess_ms_sum / n;
    let inference_mean_ms = inference_ms_sum / n;
    let postprocess_mean_ms = postprocess_ms_sum / n;
    let total_mean_ms = total_ms_sum / n;

    let denom = (io_decode_mean_ms + preprocess_mean_ms + inference_mean_ms + postprocess_mean_ms)
        .max(1e-9);

    Ok(Some(StageTimingReport {
        iterations: args.benchmark_stage_iters,
        stage_read_each_iter: args.benchmark_stage_read_each_iter,
        io_decode_mean_ms,
        preprocess_mean_ms,
        inference_mean_ms,
        postprocess_mean_ms,
        total_mean_ms,
        io_decode_share_pct: io_decode_mean_ms * 100.0 / denom,
        preprocess_share_pct: preprocess_mean_ms * 100.0 / denom,
        inference_share_pct: inference_mean_ms * 100.0 / denom,
        postprocess_share_pct: postprocess_mean_ms * 100.0 / denom,
    }))
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load the OpenVINO C shared library (required for runtime-linking mode).
    // Must happen before *any* openvino call, including Tensor::new().
    openvino_sys::library::load()
        .map_err(|e| anyhow::anyhow!("failed to load OpenVINO shared library: {e}"))?;

    // Validate model paths early.
    if !args.model.exists() {
        bail!("model file not found: {}", args.model.display());
    }
    if !args.weights.exists() {
        bail!("weights file not found: {}", args.weights.display());
    }

    if matches!(args.task, Task::Act) {
        let episode_dir = args
            .episode_dir
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--episode-dir is required for --task act"))?;
        if !episode_dir.exists() {
            bail!("episode directory not found: {}", episode_dir.display());
        }

        let metadata_path = args
            .metadata
            .clone()
            .unwrap_or_else(|| args.model.with_file_name("metadata.yaml"));
        if !metadata_path.exists() {
            bail!("ACT metadata file not found: {}", metadata_path.display());
        }

        let act_meta = load_metadata(&metadata_path, &args.model)?;

        eprintln!(
            "Loading ACT model: {} (device: {})",
            args.model.display(),
            args.device
        );
        let mut act_engine = ActEngine::new(&args.model, &args.weights, &args.device)?;

        let state = parse_state_from_episode(&episode_dir.join("data.jsonl"))?;
        let images = load_sample_images_from_episode(
            episode_dir,
            act_meta.image_width,
            act_meta.image_height,
        )?;
        let gripper = images
            .get("gripper")
            .ok_or_else(|| anyhow::anyhow!("missing gripper image sample"))?
            .clone();
        let overview = images
            .get("overview")
            .ok_or_else(|| anyhow::anyhow!("missing overview image sample"))?
            .clone();

        let act_inputs = ActInputs {
            state,
            gripper_image: gripper,
            overview_image: overview,
        };

        let out = run_act_once(&mut act_engine, &act_meta, &act_inputs)?;

        println!(
            "ACT output: chunk_size={}, action_dim={}",
            out.chunk_size, out.action_dim
        );
        for (i, a) in out.actions.iter().take(10).enumerate() {
            println!("step[{i:03}] = {:?}", a);
        }
        if out.actions.len() > 10 {
            println!("... ({} more actions)", out.actions.len() - 10);
        }

        if let Some(ref output_path) = args.output_json {
            let payload = ActJsonOutput {
                task: "act",
                model: args.model.display().to_string(),
                metadata: metadata_path.display().to_string(),
                episode_dir: episode_dir.display().to_string(),
                state_dim: act_meta.state_dim,
                chunk_size: out.chunk_size,
                action_dim: out.action_dim,
                actions: out.actions,
            };
            let json =
                serde_json::to_string_pretty(&payload).context("failed to serialize ACT JSON")?;
            write_json_file(output_path, &json, "act")?;
            eprintln!("ACT JSON saved to: {}", output_path.display());
        }

        return Ok(());
    }

    let image_path = args
        .image
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--image is required for this task"))?;
    if !image_path.exists() {
        bail!("image file not found: {}", image_path.display());
    }

    eprintln!(
        "Loading image: {} (target {}x{}, preprocess backend: {:?})",
        image_path.display(),
        args.width,
        args.height,
        args.preprocess_backend,
    );

    let tensor = match args.preprocess_backend {
        PreprocessBackend::Rust => load_image(image_path, args.width, args.height)?,
        PreprocessBackend::Openvino => load_image_no_resize(image_path)?,
    };

    eprintln!(
        "Loading model: {} (device: {})",
        args.model.display(),
        args.device
    );

    let mut engine = Engine::new(&args.model, &args.weights, &args.device, &tensor)?;

    eprintln!("  input  : {}", engine.input_name());
    eprintln!("  outputs: [{}]", engine.output_names().join(", "));

    // Try to extract class label names from the model XML metadata.
    let labels = parse_labels_from_model_xml(&args.model).unwrap_or_default();
    if !labels.is_empty() {
        eprintln!("  labels : {:?}", labels);
    }

    eprintln!("Running inference …");

    match args.task {
        Task::Classify => {
            let output = engine.infer(&tensor)?;
            let results = top_k_classifications(&output, args.top_k);
            if labels.is_empty() {
                println!("{:<10} {}", "CLASS ID", "PROBABILITY");
                println!("{:-<10} {:-<12}", "", "");
                for c in &results {
                    println!("{:<10} {:.6}", c.class_id, c.probability);
                }
            } else {
                println!("{:<10} {:<20} {}", "CLASS ID", "LABEL", "PROBABILITY");
                println!("{:-<10} {:-<20} {:-<12}", "", "", "");
                for c in &results {
                    let label = labels.get(c.class_id).map(|s| s.as_str()).unwrap_or("?");
                    println!("{:<10} {:<20} {:.6}", c.class_id, label, c.probability);
                }
            }

            if let Some(ref output_path) = args.output_json {
                write_classification_json(
                    output_path,
                    &results,
                    &labels,
                    args.top_k,
                    image_path,
                    &args.model,
                    args.width,
                    args.height,
                )?;
                eprintln!("Classification JSON saved to: {}", output_path.display());
            }
        }
        Task::Detect => {
            let detections = match args.detection_format {
                DetectionFormat::Geti => {
                    let outputs = engine.infer_multi(&tensor)?;

                    // Geti models use "boxes" or "bboxes" for the box output.
                    let boxes_buf = outputs
                        .get("boxes")
                        .or_else(|| outputs.get("bboxes"))
                        .ok_or_else(|| {
                            anyhow::anyhow!(
                                "model has no 'boxes' or 'bboxes' output (found: [{}])",
                                engine.output_names().join(", ")
                            )
                        })?;
                    let labels_buf = outputs
                        .get("labels")
                        .ok_or_else(|| anyhow::anyhow!("model has no 'labels' output"))?;

                    decode_geti_detections(boxes_buf.as_f32(), labels_buf.as_i64(), args.threshold)
                }
                DetectionFormat::Ssd => {
                    let output = engine.infer(&tensor)?;
                    decode_ssd_detections(&output, args.threshold)
                }
                DetectionFormat::Yolo => {
                    let output = engine.infer(&tensor)?;
                    decode_yolo_detections(&output, args.num_classes, args.threshold)
                }
            };

            if labels.is_empty() {
                println!(
                    "{:<8} {:<10} {:<10} {:<10} {:<10} {:<10}",
                    "CLASS", "CONF", "X1", "Y1", "X2", "Y2"
                );
                println!("{:-<60}", "");
            } else {
                println!(
                    "{:<8} {:<16} {:<10} {:<10} {:<10} {:<10} {:<10}",
                    "CLASS", "LABEL", "CONF", "X1", "Y1", "X2", "Y2"
                );
                println!("{:-<76}", "");
            }
            if detections.is_empty() {
                println!("(no detections above threshold {:.2})", args.threshold);
            }
            for d in &detections {
                if labels.is_empty() {
                    println!(
                        "{:<8} {:<10.4} {:<10.2} {:<10.2} {:<10.2} {:<10.2}",
                        d.class_id, d.confidence, d.x1, d.y1, d.x2, d.y2
                    );
                } else {
                    let label = labels.get(d.class_id).map(|s| s.as_str()).unwrap_or("?");
                    println!(
                        "{:<8} {:<16} {:<10.4} {:<10.2} {:<10.2} {:<10.2} {:<10.2}",
                        d.class_id, label, d.confidence, d.x1, d.y1, d.x2, d.y2
                    );
                }
            }
            println!("\nTotal detections: {}", detections.len());

            // If --output-image was specified, draw bounding boxes on the original image.
            if let Some(ref output_path) = args.output_image {
                draw_detections(
                    image_path,
                    output_path,
                    &detections,
                    &labels,
                    args.width,
                    args.height,
                )?;
                eprintln!("Annotated image saved to: {}", output_path.display());
            }

            if let Some(ref output_path) = args.output_json {
                write_detection_json(
                    output_path,
                    &detections,
                    &labels,
                    &args.detection_format,
                    args.threshold,
                    image_path,
                    &args.model,
                    args.width,
                    args.height,
                )?;
                eprintln!("Detection JSON saved to: {}", output_path.display());
            }
        }
        Task::Benchmark => {
            if args.benchmark_duration <= 0.0 && args.benchmark_iters.is_none() {
                bail!("--benchmark-duration must be > 0 when --benchmark-iters is not set");
            }

            let cfg = BenchmarkConfig {
                warmup_iters: args.benchmark_warmup,
                duration_secs: args.benchmark_duration,
                max_iters: args.benchmark_iters,
                report_every_secs: args.benchmark_report_every,
            };

            eprintln!(
                "Benchmark config: warmup={}, duration={:.2}s, max_iters={:?}, report_every={:.2}s",
                cfg.warmup_iters, cfg.duration_secs, cfg.max_iters, cfg.report_every_secs
            );

            let report = run_benchmark(&mut engine, &tensor, &cfg)?;
            let stage_timing = run_stage_timing(&args, image_path, &mut engine, &labels)?;

            println!("Benchmark results");
            println!("-----------------");
            println!("Warmup iterations  : {}", report.warmup_iters);
            println!("Measured iterations: {}", report.measured_iters);
            println!("Measured time (s)  : {:.3}", report.measured_seconds);
            println!("Throughput (img/s) : {:.3}", report.throughput_fps);
            println!(
                "Latency (ms)       : mean={:.3} min={:.3} p50={:.3} p90={:.3} p95={:.3} p99={:.3} max={:.3}",
                report.latency.mean_ms,
                report.latency.min_ms,
                report.latency.p50_ms,
                report.latency.p90_ms,
                report.latency.p95_ms,
                report.latency.p99_ms,
                report.latency.max_ms,
            );

            if let Some(stage) = &stage_timing {
                println!("\nStage timing (rough)");
                println!("--------------------");
                println!("Iterations          : {}", stage.iterations);
                println!("Read each iter      : {}", stage.stage_read_each_iter);
                println!(
                    "IO+Decode (ms)      : {:.3} ({:.1}%)",
                    stage.io_decode_mean_ms, stage.io_decode_share_pct
                );
                println!(
                    "Preprocess (ms)     : {:.3} ({:.1}%)",
                    stage.preprocess_mean_ms, stage.preprocess_share_pct
                );
                println!(
                    "Inference (ms)      : {:.3} ({:.1}%)",
                    stage.inference_mean_ms, stage.inference_share_pct
                );
                println!(
                    "Postprocess (ms)    : {:.3} ({:.1}%)",
                    stage.postprocess_mean_ms, stage.postprocess_share_pct
                );
                println!("Total (ms)          : {:.3}", stage.total_mean_ms);
            }

            if let Some(ref output_path) = args.output_json {
                write_benchmark_json(
                    output_path,
                    &report,
                    stage_timing,
                    &cfg,
                    image_path,
                    &args.model,
                    &args.device,
                    args.preprocess_backend,
                    args.width,
                    args.height,
                )?;
                eprintln!("Benchmark JSON saved to: {}", output_path.display());
            }
        }
        Task::Act => {
            bail!("internal error: ACT task should have returned earlier");
        }
    }

    Ok(())
}
