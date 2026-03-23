use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

use inference_rs::engine::Engine;
use inference_rs::labels::parse_labels_from_model_xml;
use inference_rs::postprocessing::{
    decode_geti_detections, decode_ssd_detections, decode_yolo_detections, top_k_classifications,
};
use inference_rs::preprocessing::load_image;
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
    #[arg(long)]
    image: PathBuf,

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
}

#[derive(Debug, Clone, ValueEnum)]
enum Task {
    Classify,
    Detect,
}

#[derive(Debug, Clone, ValueEnum)]
enum DetectionFormat {
    /// Intel Geti (OTX) format: "boxes" [1,N,5] + "labels" [1,N].
    Geti,
    /// SSD-style: output shape [1, 1, N, 7].
    Ssd,
    /// YOLO-style: output shape [1, N, 5 + num_classes].
    Yolo,
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

fn main() -> Result<()> {
    let args = Args::parse();

    // Load the OpenVINO C shared library (required for runtime-linking mode).
    // Must happen before *any* openvino call, including Tensor::new().
    openvino_sys::library::load()
        .map_err(|e| anyhow::anyhow!("failed to load OpenVINO shared library: {e}"))?;

    // Validate paths early.
    if !args.model.exists() {
        bail!("model file not found: {}", args.model.display());
    }
    if !args.weights.exists() {
        bail!("weights file not found: {}", args.weights.display());
    }
    if !args.image.exists() {
        bail!("image file not found: {}", args.image.display());
    }

    eprintln!(
        "Loading image: {} (resize to {}x{})",
        args.image.display(),
        args.width,
        args.height
    );

    let tensor = load_image(&args.image, args.width, args.height)?;

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
                    &args.image,
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
                    &args.image,
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
                    &args.image,
                    &args.model,
                    args.width,
                    args.height,
                )?;
                eprintln!("Detection JSON saved to: {}", output_path.display());
            }
        }
    }

    Ok(())
}
