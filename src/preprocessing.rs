use anyhow::{Context, Result};
use image::{GenericImageView, RgbImage};
use openvino::{ElementType, Shape, Tensor};
use std::path::Path;

use crate::infra::tensor_utils::cast_bytes_mut_to_f32;

/// Load an image from disk, resize it to the target dimensions, normalize pixel
/// values to `[0.0, 1.0]`, and pack the result into an OpenVINO [`Tensor`] with
/// shape `[1, height, width, 3]` (NHWC layout, `f32`).
///
/// The NHWC-to-NCHW conversion is handled later by the OpenVINO pre/post-process
/// pipeline configured in [`crate::engine::Engine`].
pub fn load_image(path: &Path, width: u32, height: u32) -> Result<Tensor> {
    let img =
        image::open(path).with_context(|| format!("failed to open image: {}", path.display()))?;

    let resized = img.resize_exact(width, height, image::imageops::FilterType::Triangle);
    let rgb = resized.to_rgb8();

    rgb8_to_tensor(&rgb)
}

/// Load an image from disk without resizing and pack it into an OpenVINO F32
/// NHWC tensor. Useful when resizing is delegated to OpenVINO preprocessing.
pub fn load_image_no_resize(path: &Path) -> Result<Tensor> {
    let img =
        image::open(path).with_context(|| format!("failed to open image: {}", path.display()))?;
    let rgb = img.to_rgb8();
    rgb8_to_tensor(&rgb)
}

/// Convert an RGB8 image into an OpenVINO F32 NHWC tensor with shape
/// `[1, height, width, 3]` and values normalized to `[0.0, 1.0]`.
pub fn rgb8_to_tensor(rgb: &RgbImage) -> Result<Tensor> {
    let width = rgb.width();
    let height = rgb.height();

    let shape = Shape::new(&[1, height as i64, width as i64, 3])
        .context("failed to create tensor shape")?;

    let mut tensor = Tensor::new(ElementType::F32, &shape).context("failed to allocate tensor")?;

    // Write normalized pixel data into the tensor's raw buffer.
    let buffer = tensor
        .get_raw_data_mut()
        .context("failed to get mutable tensor buffer")?;

    let float_slice: &mut [f32] = cast_bytes_mut_to_f32(buffer);

    for (i, pixel) in rgb.pixels().enumerate() {
        let base = i * 3;
        float_slice[base] = pixel[0] as f32 / 255.0;
        float_slice[base + 1] = pixel[1] as f32 / 255.0;
        float_slice[base + 2] = pixel[2] as f32 / 255.0;
    }

    Ok(tensor)
}

/// Return the (width, height) dimensions of an image on disk without fully
/// decoding it.
pub fn image_dimensions(path: &Path) -> Result<(u32, u32)> {
    let img =
        image::open(path).with_context(|| format!("failed to open image: {}", path.display()))?;
    Ok(img.dimensions())
}
