"""Image loading, resizing, normalisation, and tensor creation.

Matches the Rust ``preprocessing.rs`` pipeline:
- Pillow BILINEAR resize (equivalent to Rust ``image`` crate Triangle filter)
- Pixel values normalised to [0.0, 1.0] via /255.0
- NHWC layout ``[1, H, W, 3]`` for vision models (OpenVINO PPP does NHWC→NCHW)
- NCHW layout ``[1, 3, H, W]`` for ACT models (no PPP, raw planar)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def load_image_nhwc(path: Path | str, width: int, height: int) -> np.ndarray:
    """Load an image, resize to *width*×*height*, and return an NHWC f32 tensor.

    Returns:
        ``np.ndarray`` with shape ``[1, height, width, 3]`` and dtype ``float32``.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((width, height), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
    return arr[np.newaxis, ...]  # [1, H, W, 3]


def load_image_nhwc_no_resize(path: Path | str) -> np.ndarray:
    """Load an image without resizing — NHWC f32 tensor at original size."""
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr[np.newaxis, ...]


def image_to_nchw(
    path_or_img: Path | str | Image.Image, width: int, height: int
) -> np.ndarray:
    """Load/resize an image and return an NCHW f32 tensor ``[1, 3, H, W]``.

    Used by the ACT pipeline where tensors are fed directly (no OpenVINO PPP).
    """
    if isinstance(path_or_img, Image.Image):
        img = path_or_img.convert("RGB")
    else:
        img = Image.open(path_or_img).convert("RGB")

    if img.size != (width, height):
        img = img.resize((width, height), Image.BILINEAR)

    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
    # Transpose to [3, H, W] then add batch dim
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    return arr[np.newaxis, ...]  # [1, 3, H, W]


def image_dimensions(path: Path | str) -> tuple[int, int]:
    """Return ``(width, height)`` of an image without fully decoding it."""
    with Image.open(path) as img:
        return img.size
