use ab_glyph::{FontArc, PxScale};
use anyhow::{Context, Result};
use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut, text_size};
use imageproc::rect::Rect;
use std::path::Path;

use crate::postprocessing::Detection;

/// Embedded DejaVu Sans TTF font for label rendering.
static FONT_BYTES: &[u8] = include_bytes!("../fonts/DejaVuSans.ttf");

/// A distinct color palette for up to 20 classes. Each entry is `[R, G, B]`.
/// Colors cycle if there are more classes than palette entries.
const PALETTE: &[[u8; 3]] = &[
    [255, 0, 0],     // red
    [0, 255, 0],     // green
    [0, 0, 255],     // blue
    [255, 255, 0],   // yellow
    [255, 0, 255],   // magenta
    [0, 255, 255],   // cyan
    [255, 128, 0],   // orange
    [128, 0, 255],   // purple
    [0, 128, 255],   // sky blue
    [255, 0, 128],   // rose
    [0, 255, 128],   // spring green
    [128, 255, 0],   // lime
    [64, 224, 208],  // turquoise
    [255, 165, 0],   // dark orange
    [220, 20, 60],   // crimson
    [30, 144, 255],  // dodger blue
    [50, 205, 50],   // lime green
    [255, 105, 180], // hot pink
    [0, 191, 255],   // deep sky blue
    [218, 112, 214], // orchid
];

/// Default font scale (pixels).
const FONT_SCALE: f32 = 16.0;

/// Thickness of bounding box outlines in pixels.
const BOX_THICKNESS: i32 = 2;

/// Draw detections on the original image and save the result to disk.
///
/// Detection coordinates are assumed to be in **model input space** (i.e.,
/// relative to `model_width` x `model_height`). They are scaled to the
/// original image dimensions before drawing.
///
/// # Arguments
///
/// * `input_image`  - Path to the original input image.
/// * `output_path`  - Path to save the annotated output image.
/// * `detections`   - Decoded detection results.
/// * `labels`       - Optional class label names (index = class id).
/// * `model_width`  - Width of the model input (pixels).
/// * `model_height` - Height of the model input (pixels).
pub fn draw_detections(
    input_image: &Path,
    output_path: &Path,
    detections: &[Detection],
    labels: &[String],
    model_width: u32,
    model_height: u32,
) -> Result<()> {
    let font = FontArc::try_from_slice(FONT_BYTES)
        .map_err(|e| anyhow::anyhow!("failed to load embedded font: {e}"))?;

    let img = image::open(input_image)
        .with_context(|| format!("failed to open image: {}", input_image.display()))?;
    let mut canvas: RgbImage = img.to_rgb8();

    let img_w = canvas.width() as f32;
    let img_h = canvas.height() as f32;
    let scale_x = img_w / model_width as f32;
    let scale_y = img_h / model_height as f32;

    let px_scale = PxScale::from(FONT_SCALE);

    for det in detections {
        // Scale coordinates to original image space.
        // - SSD/YOLO often return normalized [0..1] coords.
        // - Geti typically returns absolute coords in model input pixels.
        let (x1f, y1f, x2f, y2f) = if is_normalized_box(det) {
            (
                det.x1 * img_w,
                det.y1 * img_h,
                det.x2 * img_w,
                det.y2 * img_h,
            )
        } else {
            (
                det.x1 * scale_x,
                det.y1 * scale_y,
                det.x2 * scale_x,
                det.y2 * scale_y,
            )
        };

        let mut x1 = x1f.round().max(0.0).min(img_w - 1.0) as i32;
        let mut y1 = y1f.round().max(0.0).min(img_h - 1.0) as i32;
        let mut x2 = x2f.round().max(0.0).min(img_w - 1.0) as i32;
        let mut y2 = y2f.round().max(0.0).min(img_h - 1.0) as i32;

        if x1 > x2 {
            std::mem::swap(&mut x1, &mut x2);
        }
        if y1 > y2 {
            std::mem::swap(&mut y1, &mut y2);
        }

        let color_idx = det.class_id % PALETTE.len();
        let box_color = Rgb(PALETTE[color_idx]);

        // Draw thick bounding box by drawing multiple offset rectangles.
        for offset in 0..BOX_THICKNESS {
            let rx1 = (x1 - offset).max(0);
            let ry1 = (y1 - offset).max(0);
            let rx2 = (x2 + offset).min(canvas.width() as i32 - 1);
            let ry2 = (y2 + offset).min(canvas.height() as i32 - 1);

            let w = (rx2 - rx1) as u32;
            let h = (ry2 - ry1) as u32;
            if w == 0 || h == 0 {
                continue;
            }

            let rect = Rect::at(rx1, ry1).of_size(w, h);
            draw_hollow_rect_mut(&mut canvas, rect, box_color);
        }

        // Build label string: "ClassName 0.95" or "class_id 0.95".
        let class_name = labels.get(det.class_id).map(|s| s.as_str()).unwrap_or("");
        let label_text = if class_name.is_empty() {
            format!("{} {:.2}", det.class_id, det.confidence)
        } else {
            format!("{} {:.2}", class_name, det.confidence)
        };

        // Measure text and draw a filled background rectangle for readability.
        let (text_w, text_h) = text_size(px_scale, &font, &label_text);
        let padding = 2i32;
        let label_y = (y1 - text_h as i32 - padding * 2).max(0);
        let bg_x2 = (x1 + text_w as i32 + padding * 2).min(canvas.width() as i32 - 1);
        let bg_y2 = (label_y + text_h as i32 + padding * 2).min(canvas.height() as i32 - 1);

        let bg_w = (bg_x2 - x1) as u32;
        let bg_h = (bg_y2 - label_y) as u32;
        if bg_w > 0 && bg_h > 0 {
            let bg_rect = Rect::at(x1, label_y).of_size(bg_w, bg_h);
            draw_filled_rect_mut(&mut canvas, bg_rect, box_color);
        }

        // Draw text in white (or black for bright colors) on the background.
        let text_color = contrasting_text_color(PALETTE[color_idx]);
        draw_text_mut(
            &mut canvas,
            text_color,
            x1 + padding,
            label_y + padding,
            px_scale,
            &font,
            &label_text,
        );
    }

    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory: {}", parent.display())
            })?;
        }
    }

    canvas
        .save(output_path)
        .with_context(|| format!("failed to save output image: {}", output_path.display()))?;

    Ok(())
}

fn is_normalized_box(det: &Detection) -> bool {
    // Allow slight out-of-range values to account for model/postprocess jitter.
    det.x1 >= -0.2
        && det.y1 >= -0.2
        && det.x2 <= 1.2
        && det.y2 <= 1.2
        && det.x2 >= -0.2
        && det.y2 >= -0.2
}

/// Pick black or white text based on the perceived brightness of the background.
fn contrasting_text_color(bg: [u8; 3]) -> Rgb<u8> {
    // Relative luminance approximation (ITU-R BT.601).
    let luminance = 0.299 * bg[0] as f32 + 0.587 * bg[1] as f32 + 0.114 * bg[2] as f32;
    if luminance > 160.0 {
        Rgb([0, 0, 0]) // dark text on bright background
    } else {
        Rgb([255, 255, 255]) // white text on dark background
    }
}
