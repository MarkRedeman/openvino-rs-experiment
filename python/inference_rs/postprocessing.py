"""Post-processing utilities for classification and detection outputs.

Matches the Rust ``postprocessing.rs`` implementations exactly:
- ``top_k_classifications``: top-k by descending probability
- ``decode_geti_detections``: boxes [1,N,5] + labels [1,N] (Intel Geti/OTX)
- ``decode_ssd_detections``: [1,1,N,7] stride-7 SSD format
- ``decode_yolo_detections``: [1,N,5+C] YOLO format
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Classification:
    class_id: int
    probability: float


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float


def top_k_classifications(
    output: list[float] | "numpy.ndarray", k: int
) -> list[Classification]:
    """Return the top-*k* classification results sorted by descending probability."""
    import numpy as np

    probs = np.asarray(output, dtype=np.float32)
    # Sort descending; NaN pushed to end via nan_to_num
    indices = np.argsort(-np.nan_to_num(probs, nan=-np.inf))[:k]
    return [
        Classification(class_id=int(i), probability=float(probs[i])) for i in indices
    ]


def decode_geti_detections(
    boxes: list[float] | "numpy.ndarray",
    labels: list[int] | "numpy.ndarray",
    threshold: float,
) -> list[Detection]:
    """Decode Intel Geti (OTX) detections.

    - *boxes*: flat array from shape ``[1, N, 5]`` — each row ``(x1, y1, x2, y2, score)``
    - *labels*: flat array from shape ``[1, N]`` — class id per detection
    """
    import numpy as np

    boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    labels_arr = np.asarray(labels).flatten()

    detections: list[Detection] = []
    for bbox, label in zip(boxes_arr, labels_arr):
        score = float(bbox[4])
        if score < threshold:
            continue
        detections.append(
            Detection(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                class_id=int(label),
                confidence=score,
            )
        )
    return detections


def decode_ssd_detections(
    output: list[float] | "numpy.ndarray",
    threshold: float,
) -> list[Detection]:
    """Decode SSD-style output ``[1, 1, N, 7]``.

    Each row: ``(image_id, class_id, confidence, x1, y1, x2, y2)``.
    """
    import numpy as np

    data = np.asarray(output, dtype=np.float32).reshape(-1, 7)
    detections: list[Detection] = []
    for det in data:
        confidence = float(det[2])
        if confidence < threshold:
            continue
        detections.append(
            Detection(
                x1=float(det[3]),
                y1=float(det[4]),
                x2=float(det[5]),
                y2=float(det[6]),
                class_id=int(det[1]),
                confidence=confidence,
            )
        )
    return detections


def decode_yolo_detections(
    output: list[float] | "numpy.ndarray",
    num_classes: int,
    threshold: float,
) -> list[Detection]:
    """Decode YOLO-style output ``[1, N, 5 + num_classes]``.

    Each row: ``(cx, cy, w, h, obj_conf, class_0_score, class_1_score, ...)``.
    """
    import numpy as np

    stride = 5 + num_classes
    if stride == 0:
        return []

    data = np.asarray(output, dtype=np.float32).reshape(-1, stride)
    detections: list[Detection] = []
    for det in data:
        cx, cy, w, h, obj_conf = det[:5]
        class_scores = det[5:]
        class_id = int(np.argmax(class_scores))
        max_score = float(class_scores[class_id])
        confidence = float(obj_conf) * max_score
        if confidence < threshold:
            continue
        detections.append(
            Detection(
                x1=float(cx - w / 2),
                y1=float(cy - h / 2),
                x2=float(cx + w / 2),
                y2=float(cy + h / 2),
                class_id=class_id,
                confidence=confidence,
            )
        )
    return detections
