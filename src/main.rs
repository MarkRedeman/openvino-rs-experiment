use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use openvino::Core;
use std::path::PathBuf;
use std::time::Instant;

use inference_rs::act::{
    load_sample_images_from_episode, parse_state_from_episode, prepare_act_input_tensors,
    read_act_output, ActEngine, ActInputTensors, ActInputs,
};
use inference_rs::benchmark::{
    run_benchmark, run_benchmark_loop, BenchmarkConfig, StageTimingReport,
};
use inference_rs::domain::model::InferenceModel;
use inference_rs::domain::types::{InferenceContext, InferenceInput, InferenceOutput, ModelType};
use inference_rs::engine::Engine;
use inference_rs::infra::device_compat::{check_static_compat, diagnose_compile_error};
use inference_rs::infra::diagnostics::get_last_openvino_error;
use inference_rs::infra::model_info::parse_model_xml;
use inference_rs::labels::parse_labels_from_model_xml;
use inference_rs::models::act::ActModel;
use inference_rs::models::model_wrapper::ModelWrapper;
use inference_rs::models::registry::ModelRegistry;
use inference_rs::models::vision::VisionModel;
use inference_rs::output::{
    write_act_benchmark_json, write_act_json, write_benchmark_json as write_benchmark_json_file,
    write_classification_json as write_classification_json_file,
    write_detection_json as write_detection_json_file,
};
use inference_rs::postprocessing::{
    decode_geti_detections, decode_ssd_detections, decode_yolo_detections, top_k_classifications,
};
use inference_rs::preprocessing::{load_image, load_image_no_resize, rgb8_to_tensor};
use inference_rs::visualization::draw_detections;

// ===========================================================================
// CLI definition
// ===========================================================================

/// OpenVINO inference toolkit for vision and ACT models.
#[derive(Parser, Debug)]
#[command(name = "inference-rs", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run inference (classify, detect, benchmark, act, act-benchmark).
    Infer(InferArgs),

    /// Check model compatibility against available devices.
    Check(CheckArgs),

    /// List available OpenVINO devices and their capabilities.
    Devices,
}

// ---------------------------------------------------------------------------
// Infer subcommand
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
struct InferArgs {
    /// Path to the OpenVINO IR model definition (.xml).
    #[arg(long)]
    model: PathBuf,

    /// Path to the OpenVINO IR weights (.bin).
    #[arg(long)]
    weights: PathBuf,

    /// Path to the input image (JPEG, PNG, ...).
    /// Required for classify/detect/benchmark. Not used for --task act.
    #[arg(long)]
    image: Option<PathBuf>,

    /// Inference device (e.g. CPU, GPU, NPU, HETERO:NPU,CPU).
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
    #[arg(long)]
    output_image: Option<PathBuf>,

    /// Save results as JSON.
    #[arg(long)]
    output_json: Option<PathBuf>,

    /// Benchmark duration in seconds.
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

    /// Number of iterations used for rough stage timing in benchmark mode.
    /// Set to 0 to disable stage timing.
    #[arg(long, default_value_t = 30)]
    benchmark_stage_iters: usize,

    /// In benchmark stage timing, read+decode the image every iteration.
    #[arg(long, default_value_t = false)]
    benchmark_stage_read_each_iter: bool,

    /// Preprocessing backend.
    #[arg(long, value_enum, default_value_t = PreprocessBackend::Rust)]
    preprocess_backend: PreprocessBackend,

    /// Path to ACT metadata YAML. Defaults to sibling metadata.yaml next to --model.
    #[arg(long)]
    metadata: Option<PathBuf>,

    /// Episode directory used to source ACT sample inputs.
    #[arg(long)]
    episode_dir: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Check subcommand
// ---------------------------------------------------------------------------

#[derive(Parser, Debug)]
struct CheckArgs {
    /// Path to the OpenVINO IR model definition (.xml).
    #[arg(long)]
    model: PathBuf,

    /// Path to the OpenVINO IR weights (.bin).
    #[arg(long)]
    weights: PathBuf,

    /// Only check specific device(s). By default all available devices are checked.
    #[arg(long)]
    device: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Shared enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ValueEnum)]
enum Task {
    Classify,
    Detect,
    Benchmark,
    Act,
    ActBenchmark,
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

#[derive(Debug, Clone, Copy, ValueEnum)]
enum PreprocessBackend {
    Rust,
    Openvino,
}

// ===========================================================================
// `devices` command
// ===========================================================================

fn run_devices() -> Result<()> {
    let core = Core::new().context("failed to initialise OpenVINO core")?;
    let devices = core
        .available_devices()
        .context("failed to query available OpenVINO devices")?;

    if devices.is_empty() {
        println!("No OpenVINO devices found.");
        return Ok(());
    }

    println!("Available OpenVINO devices:");
    for device in &devices {
        println!("\n  [{device}]");

        match core.versions(&device.to_string()) {
            Ok(versions) => {
                for (dev, ver) in &versions {
                    println!("    version : {dev} {}", ver.build_number);
                }
            }
            Err(_) => println!("    version : (unavailable)"),
        }

        use openvino::PropertyKey;
        match core.get_property(device, &PropertyKey::DeviceFullName) {
            Ok(name) => println!("    name    : {name}"),
            Err(_) => {}
        }

        match core.get_property(device, &PropertyKey::DeviceCapabilities) {
            Ok(caps) => println!("    caps    : {caps}"),
            Err(_) => {}
        }
    }

    Ok(())
}

// ===========================================================================
// `check` command
// ===========================================================================

fn run_check(args: CheckArgs) -> Result<()> {
    use inference_rs::infra::device::parse_device;

    // Validate paths.
    if !args.model.exists() {
        bail!("model file not found: {}", args.model.display());
    }
    if !args.weights.exists() {
        bail!("weights file not found: {}", args.weights.display());
    }

    // Parse model XML for structural info.
    let model_info = parse_model_xml(&args.model)?;
    print!("{model_info}");

    // Initialise OpenVINO core and read model.
    let mut core = Core::new().context("failed to initialise OpenVINO core")?;
    let model = core
        .read_model_from_file(
            &args.model.to_string_lossy(),
            &args.weights.to_string_lossy(),
        )
        .context("failed to read model")?;

    // Determine which devices to check.
    let all_devices = core
        .available_devices()
        .context("failed to query available devices")?;

    let devices_to_check: Vec<String> = if let Some(ref filter) = args.device {
        filter.iter().map(|d| d.to_uppercase()).collect()
    } else {
        all_devices.iter().map(|d| d.to_string()).collect()
    };

    if devices_to_check.is_empty() {
        println!("\nNo devices to check.");
        return Ok(());
    }

    println!("Device compatibility:");

    let mut any_failed = false;
    for device_str in &devices_to_check {
        let device_type = parse_device(device_str);
        let start = Instant::now();
        match core.compile_model(&model, device_type) {
            Ok(_compiled) => {
                let elapsed = start.elapsed();
                println!("\n  [{device_str}] compile OK ({:.1?})", elapsed);

                // Also show static warnings even on success.
                let warnings = check_static_compat(device_str, &model_info);
                if !warnings.is_empty() {
                    println!("    warnings:");
                    for w in &warnings {
                        println!("      - {w}");
                    }
                }
            }
            Err(e) => {
                any_failed = true;
                let elapsed = start.elapsed();
                let detail = get_last_openvino_error();
                println!("\n  [{device_str}] compile FAILED ({:.1?})", elapsed);
                println!("    error: {e}");

                let diagnosis = diagnose_compile_error(device_str, &detail, Some(&model_info));
                if !diagnosis.likely_causes.is_empty() {
                    println!("    likely causes:");
                    for cause in &diagnosis.likely_causes {
                        println!("      - {cause}");
                    }
                }
                if !diagnosis.suggestions.is_empty() {
                    println!("    suggestions:");
                    for suggestion in &diagnosis.suggestions {
                        println!("      - {suggestion}");
                    }
                }
            }
        }
    }

    // Also suggest HETERO combinations if any device failed.
    if any_failed && devices_to_check.len() > 1 {
        let working: Vec<&str> = devices_to_check
            .iter()
            .filter(|d| {
                let dt = parse_device(d);
                core.compile_model(&model, dt).is_ok()
            })
            .map(|s| s.as_str())
            .collect();

        // Already tried above, but compile_model may mutate core state.
        // Just suggest common HETERO combos.
        let failed: Vec<&str> = devices_to_check
            .iter()
            .filter(|d| {
                // A device failed if it's not in 'working'.
                // Re-check is wasteful, so just use a simpler heuristic:
                // We already printed FAILED above.
                !working.contains(&d.as_str())
            })
            .map(|s| s.as_str())
            .collect();

        if !failed.is_empty() && !working.is_empty() {
            println!(
                "\n  Tip: try heterogeneous execution to run unsupported ops on a fallback device:"
            );
            for f in &failed {
                for w in &working {
                    println!("    --device 'HETERO:{f},{w}'");
                }
            }
        }
    }

    Ok(())
}

// ===========================================================================
// `infer` command
// ===========================================================================

fn run_infer(args: InferArgs) -> Result<()> {
    // Validate model paths early.
    if !args.model.exists() {
        bail!("model file not found: {}", args.model.display());
    }
    if !args.weights.exists() {
        bail!("weights file not found: {}", args.weights.display());
    }

    if matches!(args.task, Task::Act | Task::ActBenchmark) {
        return run_infer_act(args);
    }

    run_infer_vision(args)
}

// ---------------------------------------------------------------------------
// ACT inference
// ---------------------------------------------------------------------------

fn run_infer_act(args: InferArgs) -> Result<()> {
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

    eprintln!(
        "Loading ACT model: {} (device: {})",
        args.model.display(),
        args.device
    );

    let mut registry = ModelRegistry::new();
    registry.load(
        "act",
        ModelWrapper::Act(ActModel::new(&args.model, &args.weights, &args.device)?),
    )?;

    let act_meta = {
        let act_model_wrapper = registry
            .get_mut("act")
            .ok_or_else(|| anyhow::anyhow!("ACT model not found in registry"))?;
        let act_model = act_model_wrapper
            .as_act_mut()
            .ok_or_else(|| anyhow::anyhow!("registry model is not ACT"))?;
        act_model.metadata().clone()
    };

    let state = parse_state_from_episode(&episode_dir.join("data.jsonl"))?;
    let images =
        load_sample_images_from_episode(episode_dir, act_meta.image_width, act_meta.image_height)?;
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

    if matches!(args.task, Task::ActBenchmark) {
        run_infer_act_benchmark(
            &args,
            &mut registry,
            &act_meta,
            episode_dir,
            &act_inputs,
            &metadata_path,
        )
    } else {
        run_infer_act_single(
            &args,
            &mut registry,
            &act_meta,
            episode_dir,
            &act_inputs,
            &metadata_path,
        )
    }
}

fn run_infer_act_benchmark(
    args: &InferArgs,
    registry: &mut ModelRegistry,
    act_meta: &inference_rs::act::ActMetadata,
    episode_dir: &PathBuf,
    act_inputs: &ActInputs,
    metadata_path: &PathBuf,
) -> Result<()> {
    if args.benchmark_duration <= 0.0 && args.benchmark_iters.is_none() {
        bail!("--benchmark-duration must be > 0 when --benchmark-iters is not set");
    }

    let cfg = BenchmarkConfig {
        warmup_iters: args.benchmark_warmup,
        duration_secs: args.benchmark_duration,
        max_iters: args.benchmark_iters,
        report_every_secs: args.benchmark_report_every,
    };

    let act_model_wrapper = registry
        .get_mut("act")
        .ok_or_else(|| anyhow::anyhow!("ACT model not found in registry"))?;
    let act_model = act_model_wrapper
        .as_act_mut()
        .ok_or_else(|| anyhow::anyhow!("registry model is not ACT"))?;
    let tensors = prepare_act_input_tensors(act_meta, act_inputs)?;
    let mut request = act_model.engine_mut().create_request()?;

    let report = run_benchmark_loop(&cfg, || {
        act_model.engine_mut().run_request(&mut request, &tensors)
    })?;
    let stage_timing = run_act_stage_timing(
        args,
        act_model.engine_mut(),
        act_meta,
        episode_dir,
        act_inputs,
    )?;

    println!("ACT benchmark results");
    println!("---------------------");
    println!("Warmup iterations  : {}", report.warmup_iters);
    println!("Measured iterations: {}", report.measured_iters);
    println!("Measured time (s)  : {:.3}", report.measured_seconds);
    println!("Throughput (run/s) : {:.3}", report.throughput_fps);
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
        write_act_benchmark_json(
            output_path,
            &args.model,
            metadata_path,
            episode_dir,
            &args.device,
            cfg,
            report,
            stage_timing,
            act_meta.chunk_size,
            act_meta.action_dim,
        )?;
        eprintln!("ACT benchmark JSON saved to: {}", output_path.display());
    }

    Ok(())
}

fn run_infer_act_single(
    args: &InferArgs,
    registry: &mut ModelRegistry,
    act_meta: &inference_rs::act::ActMetadata,
    episode_dir: &PathBuf,
    act_inputs: &ActInputs,
    metadata_path: &PathBuf,
) -> Result<()> {
    let tensors = prepare_act_input_tensors(act_meta, act_inputs)?;
    let act_model_wrapper = registry
        .get_mut("act")
        .ok_or_else(|| anyhow::anyhow!("ACT model not found in registry"))?;
    let out = match act_model_wrapper
        .infer(InferenceInput::Act(&tensors), &InferenceContext::default())?
    {
        InferenceOutput::Act(out) => out,
        _ => bail!("unexpected non-ACT output from ACT model"),
    };

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
        write_act_json(
            output_path,
            &args.model,
            metadata_path,
            episode_dir,
            act_meta.state_dim,
            out,
        )?;
        eprintln!("ACT JSON saved to: {}", output_path.display());
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Vision inference
// ---------------------------------------------------------------------------

fn run_infer_vision(args: InferArgs) -> Result<()> {
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

    let mut registry = ModelRegistry::new();
    let model_type = match args.task {
        Task::Detect => ModelType::Detection,
        _ => ModelType::Classification,
    };
    registry.load(
        "vision",
        ModelWrapper::Vision(VisionModel::new(
            &args.model,
            &args.weights,
            &args.device,
            &tensor,
            model_type,
        )?),
    )?;

    let vision_model_wrapper = registry
        .get_mut("vision")
        .ok_or_else(|| anyhow::anyhow!("vision model not found in registry"))?;
    let (input_name, output_names_csv) = {
        let vision_model = vision_model_wrapper
            .as_vision_mut()
            .ok_or_else(|| anyhow::anyhow!("registry model is not vision"))?;
        let engine = vision_model.engine_mut();
        (
            engine.input_name().to_string(),
            engine.output_names().join(", "),
        )
    };

    eprintln!("  input  : {input_name}");
    eprintln!("  outputs: [{output_names_csv}]");

    let labels = parse_labels_from_model_xml(&args.model).unwrap_or_default();
    if !labels.is_empty() {
        eprintln!("  labels : {:?}", labels);
    }

    eprintln!("Running inference …");

    match args.task {
        Task::Classify => {
            run_classify(&args, vision_model_wrapper, &tensor, &labels, image_path)?;
        }
        Task::Detect => {
            run_detect(
                &args,
                vision_model_wrapper,
                &tensor,
                &labels,
                image_path,
                &output_names_csv,
            )?;
        }
        Task::Benchmark => {
            run_vision_benchmark(&args, vision_model_wrapper, &tensor, &labels, image_path)?;
        }
        Task::Act | Task::ActBenchmark => {
            bail!("internal error: ACT task should have been handled earlier");
        }
    }

    Ok(())
}

fn run_classify(
    args: &InferArgs,
    model: &mut ModelWrapper,
    tensor: &openvino::Tensor,
    labels: &[String],
    image_path: &PathBuf,
) -> Result<()> {
    let output = match model.infer(
        InferenceInput::Image(tensor),
        &InferenceContext {
            labels: labels.to_vec(),
        },
    )? {
        InferenceOutput::Tensor(v) => v,
        _ => bail!("unexpected non-tensor output from vision model"),
    };
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
        write_classification_json_file(
            output_path,
            &results,
            labels,
            args.top_k,
            image_path,
            &args.model,
            args.width,
            args.height,
        )?;
        eprintln!("Classification JSON saved to: {}", output_path.display());
    }

    Ok(())
}

fn run_detect(
    args: &InferArgs,
    model: &mut ModelWrapper,
    tensor: &openvino::Tensor,
    labels: &[String],
    image_path: &PathBuf,
    output_names_csv: &str,
) -> Result<()> {
    let detections = match args.detection_format {
        DetectionFormat::Geti => {
            let outputs = match model.infer(
                InferenceInput::Image(tensor),
                &InferenceContext {
                    labels: labels.to_vec(),
                },
            )? {
                InferenceOutput::MultiTensor(map) => map,
                _ => bail!("unexpected non-multi output for Geti detection"),
            };

            let boxes_buf = outputs
                .get("boxes")
                .or_else(|| outputs.get("bboxes"))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "model has no 'boxes' or 'bboxes' output (found: [{output_names_csv}])"
                    )
                })?;
            let labels_buf = outputs
                .get("labels")
                .ok_or_else(|| anyhow::anyhow!("model has no 'labels' output"))?;

            decode_geti_detections(boxes_buf.as_f32(), labels_buf.as_i64(), args.threshold)
        }
        DetectionFormat::Ssd => {
            let output = match model.infer(
                InferenceInput::Image(tensor),
                &InferenceContext {
                    labels: labels.to_vec(),
                },
            )? {
                InferenceOutput::Tensor(v) => v,
                _ => bail!("unexpected non-tensor output for SSD detection"),
            };
            decode_ssd_detections(&output, args.threshold)
        }
        DetectionFormat::Yolo => {
            let output = match model.infer(
                InferenceInput::Image(tensor),
                &InferenceContext {
                    labels: labels.to_vec(),
                },
            )? {
                InferenceOutput::Tensor(v) => v,
                _ => bail!("unexpected non-tensor output for YOLO detection"),
            };
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

    if let Some(ref output_path) = args.output_image {
        draw_detections(
            image_path,
            output_path,
            &detections,
            labels,
            args.width,
            args.height,
        )?;
        eprintln!("Annotated image saved to: {}", output_path.display());
    }

    if let Some(ref output_path) = args.output_json {
        write_detection_json_file(
            output_path,
            &detections,
            labels,
            &format!("{:?}", args.detection_format).to_lowercase(),
            args.threshold,
            image_path,
            &args.model,
            args.width,
            args.height,
        )?;
        eprintln!("Detection JSON saved to: {}", output_path.display());
    }

    Ok(())
}

fn run_vision_benchmark(
    args: &InferArgs,
    vision_model_wrapper: &mut ModelWrapper,
    tensor: &openvino::Tensor,
    labels: &[String],
    image_path: &PathBuf,
) -> Result<()> {
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

    let vision_model = vision_model_wrapper
        .as_vision_mut()
        .ok_or_else(|| anyhow::anyhow!("registry model is not vision"))?;
    let engine = vision_model.engine_mut();
    let report = run_benchmark(engine, tensor, &cfg)?;
    let stage_timing = run_stage_timing(args, image_path, engine, labels)?;

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
        write_benchmark_json_file(
            output_path,
            &report,
            stage_timing,
            &cfg,
            image_path,
            &args.model,
            &args.device,
            &format!("{:?}", args.preprocess_backend).to_lowercase(),
            args.width,
            args.height,
        )?;
        eprintln!("Benchmark JSON saved to: {}", output_path.display());
    }

    Ok(())
}

// ===========================================================================
// Stage timing helpers
// ===========================================================================

fn run_stage_timing(
    args: &InferArgs,
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

fn run_act_stage_timing(
    args: &InferArgs,
    engine: &mut ActEngine,
    meta: &inference_rs::act::ActMetadata,
    episode_dir: &PathBuf,
    inputs: &ActInputs,
) -> Result<Option<StageTimingReport>> {
    if args.benchmark_stage_iters == 0 {
        return Ok(None);
    }

    let mut io_decode_ms_sum = 0.0f64;
    let mut preprocess_ms_sum = 0.0f64;
    let mut inference_ms_sum = 0.0f64;
    let mut postprocess_ms_sum = 0.0f64;
    let mut total_ms_sum = 0.0f64;

    let prebuilt: Option<ActInputTensors> = if args.benchmark_stage_read_each_iter {
        None
    } else {
        let t = Instant::now();
        let tensors = prepare_act_input_tensors(meta, inputs)?;
        preprocess_ms_sum += t.elapsed().as_secs_f64() * 1000.0;
        Some(tensors)
    };

    let mut request = engine.create_request()?;

    for _ in 0..args.benchmark_stage_iters {
        let total_start = Instant::now();

        if args.benchmark_stage_read_each_iter {
            let t0 = Instant::now();
            let state = parse_state_from_episode(&episode_dir.join("data.jsonl"))?;
            let images =
                load_sample_images_from_episode(episode_dir, meta.image_width, meta.image_height)?;
            let gripper = images
                .get("gripper")
                .ok_or_else(|| anyhow::anyhow!("stage timing: missing gripper image sample"))?
                .clone();
            let overview = images
                .get("overview")
                .ok_or_else(|| anyhow::anyhow!("stage timing: missing overview image sample"))?
                .clone();
            let local_inputs = ActInputs {
                state,
                gripper_image: gripper,
                overview_image: overview,
            };
            io_decode_ms_sum += t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let tensors = prepare_act_input_tensors(meta, &local_inputs)?;
            preprocess_ms_sum += t1.elapsed().as_secs_f64() * 1000.0;
            let t2 = Instant::now();
            engine.run_request(&mut request, &tensors)?;
            let t3 = Instant::now();
            let _ = read_act_output(&request, meta)?;
            let t4 = Instant::now();

            inference_ms_sum += (t3 - t2).as_secs_f64() * 1000.0;
            postprocess_ms_sum += (t4 - t3).as_secs_f64() * 1000.0;
        } else {
            let t1 = Instant::now();
            let tensors = prebuilt
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("stage timing: missing prebuilt ACT tensors"))?;
            preprocess_ms_sum += t1.elapsed().as_secs_f64() * 1000.0;
            let t2 = Instant::now();
            engine.run_request(&mut request, tensors)?;
            let t3 = Instant::now();
            let _ = read_act_output(&request, meta)?;
            let t4 = Instant::now();

            inference_ms_sum += (t3 - t2).as_secs_f64() * 1000.0;
            postprocess_ms_sum += (t4 - t3).as_secs_f64() * 1000.0;
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

// ===========================================================================
// main
// ===========================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Load the OpenVINO C shared library (required for runtime-linking mode).
    // Must happen before *any* openvino call, including Tensor::new().
    openvino_sys::library::load()
        .map_err(|e| anyhow::anyhow!("failed to load OpenVINO shared library: {e}"))?;

    match cli.command {
        Command::Devices => run_devices(),
        Command::Check(args) => run_check(args),
        Command::Infer(args) => run_infer(args),
    }
}
