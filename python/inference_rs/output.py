"""JSON output serialisation matching the Rust ``output.rs`` schema exactly.

All JSON payloads use the same field names and structure as the Rust version
so that ``benchmark-all.sh`` can parse both with the same ``jq`` expression
(e.g. ``$.report.throughput_fps``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkConfig, BenchmarkReport, StageTimingReport
from .postprocessing import Classification, Detection


def _write_json_file(output_path: Path | str, content: str, kind: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)


def write_classification_json(
    output_path: Path | str,
    results: list[Classification],
    labels: list[str],
    top_k: int,
    image_path: str,
    model_path: str,
    model_width: int,
    model_height: int,
) -> None:
    records = [
        {
            "class_id": c.class_id,
            "label": labels[c.class_id] if c.class_id < len(labels) else None,
            "probability": c.probability,
        }
        for c in results
    ]

    payload = {
        "task": "classify",
        "top_k": top_k,
        "image": str(image_path),
        "model": str(model_path),
        "model_input_width": model_width,
        "model_input_height": model_height,
        "count": len(results),
        "results": records,
    }
    _write_json_file(output_path, json.dumps(payload, indent=2), "classification")


def write_detection_json(
    output_path: Path | str,
    detections: list[Detection],
    labels: list[str],
    detection_format: str,
    threshold: float,
    image_path: str,
    model_path: str,
    model_width: int,
    model_height: int,
) -> None:
    records = [
        {
            "class_id": d.class_id,
            "label": labels[d.class_id] if d.class_id < len(labels) else None,
            "confidence": d.confidence,
            "x1": d.x1,
            "y1": d.y1,
            "x2": d.x2,
            "y2": d.y2,
        }
        for d in detections
    ]

    payload = {
        "task": "detect",
        "detection_format": detection_format,
        "threshold": threshold,
        "image": str(image_path),
        "model": str(model_path),
        "model_input_width": model_width,
        "model_input_height": model_height,
        "count": len(detections),
        "detections": records,
    }
    _write_json_file(output_path, json.dumps(payload, indent=2), "detection")


def write_benchmark_json(
    output_path: Path | str,
    report: BenchmarkReport,
    stage_timing: StageTimingReport | None,
    cfg: BenchmarkConfig,
    image_path: str,
    model_path: str,
    device: str,
    preprocess_backend: str,
    model_width: int,
    model_height: int,
) -> None:
    payload: dict[str, Any] = {
        "task": "benchmark",
        "image": str(image_path),
        "model": str(model_path),
        "device": device,
        "preprocess_backend": preprocess_backend,
        "model_input_width": model_width,
        "model_input_height": model_height,
        "config": cfg.to_dict(),
        "report": report.to_dict(),
        "stage_timing": stage_timing.to_dict() if stage_timing else None,
    }
    _write_json_file(output_path, json.dumps(payload, indent=2), "benchmark")


def write_act_json(
    output_path: Path | str,
    model_path: str,
    metadata_path: str,
    episode_dir: str,
    state_dim: int,
    chunk_size: int,
    action_dim: int,
    actions: list[list[float]],
) -> None:
    payload = {
        "task": "act",
        "model": str(model_path),
        "metadata": str(metadata_path),
        "episode_dir": str(episode_dir),
        "state_dim": state_dim,
        "chunk_size": chunk_size,
        "action_dim": action_dim,
        "actions": actions,
    }
    _write_json_file(output_path, json.dumps(payload, indent=2), "act")


def write_act_benchmark_json(
    output_path: Path | str,
    model_path: str,
    metadata_path: str,
    episode_dir: str,
    device: str,
    cfg: BenchmarkConfig,
    report: BenchmarkReport,
    stage_timing: StageTimingReport | None,
    output_chunk_size: int,
    output_action_dim: int,
) -> None:
    payload: dict[str, Any] = {
        "task": "benchmark",
        "model": str(model_path),
        "metadata": str(metadata_path),
        "episode_dir": str(episode_dir),
        "device": device,
        "config": cfg.to_dict(),
        "report": report.to_dict(),
        "stage_timing": stage_timing.to_dict() if stage_timing else None,
        "output_chunk_size": output_chunk_size,
        "output_action_dim": output_action_dim,
    }
    _write_json_file(output_path, json.dumps(payload, indent=2), "act benchmark")
