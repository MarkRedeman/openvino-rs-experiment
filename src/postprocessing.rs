use serde::Serialize;

/// A single classification result: a class id paired with its probability.
#[derive(Debug, Clone, Serialize)]
pub struct Classification {
    pub class_id: usize,
    pub probability: f32,
}

/// Return the top-`k` classification results sorted by descending probability.
pub fn top_k_classifications(output: &[f32], k: usize) -> Vec<Classification> {
    let mut indexed: Vec<Classification> = output
        .iter()
        .enumerate()
        .map(|(id, &prob)| Classification {
            class_id: id,
            probability: prob,
        })
        .collect();

    // Sort descending by probability. NaN values are pushed to the end.
    indexed.sort_by(|a, b| {
        b.probability
            .partial_cmp(&a.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    indexed.truncate(k);
    indexed
}

/// A single object detection result.
#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    /// Bounding box: top-left x (normalized 0..1 or absolute pixels depending
    /// on the model).
    pub x1: f32,
    /// Bounding box: top-left y.
    pub y1: f32,
    /// Bounding box: bottom-right x.
    pub x2: f32,
    /// Bounding box: bottom-right y.
    pub y2: f32,
    /// Detected class id.
    pub class_id: usize,
    /// Detection confidence score.
    pub confidence: f32,
}

/// Decode detections from a standard SSD-style output tensor.
///
/// Many OpenVINO detection models (SSD MobileNet, person-detection, etc.)
/// produce an output shaped `[1, 1, N, 7]` where each of the `N` candidate
/// rows is `[image_id, class_id, confidence, x1, y1, x2, y2]`.
///
/// Only detections with `confidence >= threshold` are returned.
pub fn decode_ssd_detections(output: &[f32], threshold: f32) -> Vec<Detection> {
    // Each detection occupies 7 consecutive floats.
    const STRIDE: usize = 7;

    output
        .chunks_exact(STRIDE)
        .filter_map(|det| {
            let confidence = det[2];
            if confidence < threshold {
                return None;
            }
            Some(Detection {
                x1: det[3],
                y1: det[4],
                x2: det[5],
                y2: det[6],
                class_id: det[1] as usize,
                confidence,
            })
        })
        .collect()
}

/// Decode detections from a YOLO-style output tensor.
///
/// Expects the raw output flattened from shape `[1, num_detections, 5 + num_classes]`
/// where each row is `[cx, cy, w, h, obj_conf, class_0_score, class_1_score, …]`.
///
/// Returns detections whose `obj_conf * max_class_score >= threshold`.
pub fn decode_yolo_detections(
    output: &[f32],
    num_classes: usize,
    threshold: f32,
) -> Vec<Detection> {
    let stride = 5 + num_classes;
    if stride == 0 {
        return Vec::new();
    }

    output
        .chunks_exact(stride)
        .filter_map(|det| {
            let cx = det[0];
            let cy = det[1];
            let w = det[2];
            let h = det[3];
            let obj_conf = det[4];

            // Find the class with the highest score.
            let (class_id, &max_score) = det[5..]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))?;

            let confidence = obj_conf * max_score;
            if confidence < threshold {
                return None;
            }

            Some(Detection {
                x1: cx - w / 2.0,
                y1: cy - h / 2.0,
                x2: cx + w / 2.0,
                y2: cy + h / 2.0,
                class_id,
                confidence,
            })
        })
        .collect()
}

/// Decode detections from Intel Geti (OTX) exported models.
///
/// Geti models produce two outputs:
///
/// - **boxes**: shape `[1, N, 5]` (f32) — each row is `[x1, y1, x2, y2, score]`.
/// - **labels**: shape `[1, N]` (i64) — the class label for each detection.
///
/// Coordinates are typically in absolute pixels relative to the model's input
/// resolution (e.g. 800×992).
///
/// Only detections with `score >= threshold` are returned.
pub fn decode_geti_detections(boxes: &[f32], labels: &[i64], threshold: f32) -> Vec<Detection> {
    const BOX_STRIDE: usize = 5; // x1, y1, x2, y2, score

    boxes
        .chunks_exact(BOX_STRIDE)
        .zip(labels.iter())
        .filter_map(|(bbox, &label)| {
            let score = bbox[4];
            if score < threshold {
                return None;
            }
            Some(Detection {
                x1: bbox[0],
                y1: bbox[1],
                x2: bbox[2],
                y2: bbox[3],
                class_id: label as usize,
                confidence: score,
            })
        })
        .collect()
}
