use anyhow::{bail, Result};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

use inference_rs::engine::Engine;
use inference_rs::postprocessing::{
    decode_geti_detections, decode_ssd_detections, decode_yolo_detections, top_k_classifications,
};
use inference_rs::preprocessing::load_image;

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

    eprintln!("Running inference …");

    match args.task {
        Task::Classify => {
            let output = engine.infer(&tensor)?;
            let results = top_k_classifications(&output, args.top_k);
            println!("{:<10} {}", "CLASS ID", "PROBABILITY");
            println!("{:-<10} {:-<12}", "", "");
            for c in &results {
                println!("{:<10} {:.6}", c.class_id, c.probability);
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

            println!(
                "{:<8} {:<10} {:<10} {:<10} {:<10} {:<10}",
                "CLASS", "CONF", "X1", "Y1", "X2", "Y2"
            );
            println!("{:-<60}", "");
            if detections.is_empty() {
                println!("(no detections above threshold {:.2})", args.threshold);
            }
            for d in &detections {
                println!(
                    "{:<8} {:<10.4} {:<10.2} {:<10.2} {:<10.2} {:<10.2}",
                    d.class_id, d.confidence, d.x1, d.y1, d.x2, d.y2
                );
            }
            println!("\nTotal detections: {}", detections.len());
        }
    }

    Ok(())
}
