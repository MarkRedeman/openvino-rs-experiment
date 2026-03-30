"""CLI entry point — argparse with subcommands matching the Rust version.

Subcommands:
  infer    — classify, detect, benchmark, act, act-benchmark
  check    — model compatibility check against devices
  devices  — list available OpenVINO devices
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import NoReturn

import numpy as np
import openvino as ov

from .act_engine import (
    ActEngine,
    ActInputs,
    ActOutput,
    load_sample_images_from_episode,
    parse_state_from_episode,
    prepare_act_tensors,
    read_act_output,
)
from .benchmark import BenchmarkConfig, StageTimingReport, run_benchmark_loop
from .engine import Engine
from .labels import parse_labels_from_model_xml
from .output import (
    write_act_benchmark_json,
    write_act_json,
    write_benchmark_json,
    write_classification_json,
    write_detection_json,
)
from .postprocessing import (
    decode_geti_detections,
    decode_ssd_detections,
    decode_yolo_detections,
    top_k_classifications,
)
from .preprocessing import load_image_nhwc, load_image_nhwc_no_resize


# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-rs-py",
        description="OpenVINO inference toolkit — Python reference implementation.",
    )
    sub = parser.add_subparsers(dest="command")

    # ── infer ──
    infer = sub.add_parser(
        "infer", help="Run inference (classify, detect, benchmark, act, act-benchmark)"
    )
    infer.add_argument(
        "--model", required=True, help="Path to OpenVINO IR model (.xml)"
    )
    infer.add_argument(
        "--weights", required=True, help="Path to OpenVINO IR weights (.bin)"
    )
    infer.add_argument("--image", help="Path to input image")
    infer.add_argument(
        "--device", default="CPU", help="Inference device (default: CPU)"
    )
    infer.add_argument(
        "--task",
        choices=["classify", "detect", "benchmark", "act", "act-benchmark"],
        default="classify",
    )
    infer.add_argument("--top-k", type=int, default=5)
    infer.add_argument("--threshold", type=float, default=0.5)
    infer.add_argument("--width", type=int, default=224)
    infer.add_argument("--height", type=int, default=224)
    infer.add_argument(
        "--detection-format",
        choices=["geti", "ssd", "yolo"],
        default="geti",
    )
    infer.add_argument("--num-classes", type=int, default=80)
    infer.add_argument("--output-json", help="Save results as JSON")
    infer.add_argument("--benchmark-duration", type=float, default=10.0)
    infer.add_argument("--benchmark-warmup", type=int, default=20)
    infer.add_argument("--benchmark-iters", type=int, default=None)
    infer.add_argument("--benchmark-report-every", type=float, default=1.0)
    infer.add_argument("--benchmark-stage-iters", type=int, default=30)
    infer.add_argument("--benchmark-stage-read-each-iter", action="store_true")
    infer.add_argument(
        "--preprocess-backend",
        choices=["python", "openvino"],
        default="python",
    )
    infer.add_argument("--metadata", help="Path to ACT metadata YAML")
    infer.add_argument("--episode-dir", help="Episode directory for ACT")

    # ── check ──
    check = sub.add_parser("check", help="Check model compatibility against devices")
    check.add_argument(
        "--model", required=True, help="Path to OpenVINO IR model (.xml)"
    )
    check.add_argument(
        "--weights", required=True, help="Path to OpenVINO IR weights (.bin)"
    )
    check.add_argument("--device", action="append", help="Specific device(s) to check")

    # ── devices ──
    sub.add_parser("devices", help="List available OpenVINO devices")

    return parser


# ──────────────────────────────────────────────────────────────────────────────
# `devices` command
# ──────────────────────────────────────────────────────────────────────────────


def _run_devices() -> None:
    core = ov.Core()
    devices = core.available_devices

    if not devices:
        print("No OpenVINO devices found.")
        return

    print("Available OpenVINO devices:")
    for dev in devices:
        print(f"\n  [{dev}]")
        try:
            full_name = core.get_property(dev, "FULL_DEVICE_NAME")
            print(f"    name    : {full_name}")
        except Exception:
            pass
        try:
            caps = core.get_property(dev, "OPTIMIZATION_CAPABILITIES")
            print(f"    caps    : {caps}")
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# `check` command
# ──────────────────────────────────────────────────────────────────────────────


def _run_check(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    weights_path = Path(args.weights)

    if not model_path.exists():
        _die(f"model file not found: {model_path}")
    if not weights_path.exists():
        _die(f"weights file not found: {weights_path}")

    core = ov.Core()
    model = core.read_model(str(model_path), str(weights_path))

    # Print model summary
    print(f"Model: {model_path}")
    print(f"  Inputs:")
    for inp in model.inputs:
        print(f"    {inp.get_any_name()}: {inp.shape} {inp.element_type}")
    print(f"  Outputs:")
    for out in model.outputs:
        print(f"    {out.get_any_name()}: {out.shape} {out.element_type}")

    # Determine devices to check
    if args.device:
        devices = [d.upper() for d in args.device]
    else:
        devices = list(core.available_devices)

    if not devices:
        print("\nNo devices to check.")
        return

    print("\nDevice compatibility:")
    for dev in devices:
        start = time.perf_counter()
        try:
            core.compile_model(model, dev)
            elapsed = time.perf_counter() - start
            print(f"\n  [{dev}] compile OK ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"\n  [{dev}] compile FAILED ({elapsed:.1f}s)")
            print(f"    error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# `infer` command — vision path
# ──────────────────────────────────────────────────────────────────────────────


def _run_infer(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    weights_path = Path(args.weights)

    if not model_path.exists():
        _die(f"model file not found: {model_path}")
    if not weights_path.exists():
        _die(f"weights file not found: {weights_path}")

    if args.task in ("act", "act-benchmark"):
        _run_infer_act(args)
        return

    _run_infer_vision(args)


def _run_infer_vision(args: argparse.Namespace) -> None:
    if not args.image:
        _die("--image is required for this task")
    image_path = Path(args.image)
    if not image_path.exists():
        _die(f"image file not found: {image_path}")

    print(
        f"Loading image: {image_path} (target {args.width}x{args.height}, "
        f"preprocess backend: {args.preprocess_backend})",
        file=sys.stderr,
    )

    if args.preprocess_backend == "openvino":
        tensor = load_image_nhwc_no_resize(image_path)
    else:
        tensor = load_image_nhwc(image_path, args.width, args.height)

    print(f"Loading model: {args.model} (device: {args.device})", file=sys.stderr)

    engine = Engine(args.model, args.weights, args.device, tensor)
    print(f"  input  : {engine.input_name}", file=sys.stderr)
    print(f"  outputs: [{', '.join(engine.output_names)}]", file=sys.stderr)

    labels = parse_labels_from_model_xml(args.model)
    if labels:
        print(f"  labels : {labels}", file=sys.stderr)

    print("Running inference ...", file=sys.stderr)

    if args.task == "classify":
        _run_classify(args, engine, tensor, labels, image_path)
    elif args.task == "detect":
        _run_detect(args, engine, tensor, labels, image_path)
    elif args.task == "benchmark":
        _run_vision_benchmark(args, engine, tensor, labels, image_path)


def _run_classify(
    args: argparse.Namespace,
    engine: Engine,
    tensor: np.ndarray,
    labels: list[str],
    image_path: Path,
) -> None:
    output = engine.infer(tensor)
    results = top_k_classifications(output, args.top_k)

    if not labels:
        print(f"{'CLASS ID':<10} {'PROBABILITY'}")
        print(f"{'-' * 10} {'-' * 12}")
        for c in results:
            print(f"{c.class_id:<10} {c.probability:.6f}")
    else:
        print(f"{'CLASS ID':<10} {'LABEL':<20} {'PROBABILITY'}")
        print(f"{'-' * 10} {'-' * 20} {'-' * 12}")
        for c in results:
            label = labels[c.class_id] if c.class_id < len(labels) else "?"
            print(f"{c.class_id:<10} {label:<20} {c.probability:.6f}")

    if args.output_json:
        write_classification_json(
            args.output_json,
            results,
            labels,
            args.top_k,
            str(image_path),
            str(args.model),
            args.width,
            args.height,
        )
        print(f"Classification JSON saved to: {args.output_json}", file=sys.stderr)


def _run_detect(
    args: argparse.Namespace,
    engine: Engine,
    tensor: np.ndarray,
    labels: list[str],
    image_path: Path,
) -> None:
    fmt = args.detection_format

    if fmt == "geti":
        outputs = engine.infer_multi(tensor)
        boxes_key = (
            "boxes" if "boxes" in outputs else "bboxes" if "bboxes" in outputs else None
        )
        if boxes_key is None:
            _die(
                f"model has no 'boxes' or 'bboxes' output (found: {list(outputs.keys())})"
            )
        if "labels" not in outputs:
            _die("model has no 'labels' output")
        detections = decode_geti_detections(
            outputs[boxes_key].flatten(),
            outputs["labels"].flatten().astype(np.int64),
            args.threshold,
        )
    elif fmt == "ssd":
        output = engine.infer(tensor)
        detections = decode_ssd_detections(output, args.threshold)
    elif fmt == "yolo":
        output = engine.infer(tensor)
        detections = decode_yolo_detections(output, args.num_classes, args.threshold)
    else:
        _die(f"unknown detection format: {fmt}")

    # Print table
    if not labels:
        print(f"{'CLASS':<8} {'CONF':<10} {'X1':<10} {'Y1':<10} {'X2':<10} {'Y2':<10}")
        print("-" * 60)
    else:
        print(
            f"{'CLASS':<8} {'LABEL':<16} {'CONF':<10} {'X1':<10} {'Y1':<10} {'X2':<10} {'Y2':<10}"
        )
        print("-" * 76)

    if not detections:
        print(f"(no detections above threshold {args.threshold:.2f})")

    for d in detections:
        if not labels:
            print(
                f"{d.class_id:<8} {d.confidence:<10.4f} {d.x1:<10.2f} {d.y1:<10.2f} "
                f"{d.x2:<10.2f} {d.y2:<10.2f}"
            )
        else:
            label = labels[d.class_id] if d.class_id < len(labels) else "?"
            print(
                f"{d.class_id:<8} {label:<16} {d.confidence:<10.4f} {d.x1:<10.2f} "
                f"{d.y1:<10.2f} {d.x2:<10.2f} {d.y2:<10.2f}"
            )

    print(f"\nTotal detections: {len(detections)}")

    if args.output_json:
        write_detection_json(
            args.output_json,
            detections,
            labels,
            fmt,
            args.threshold,
            str(image_path),
            str(args.model),
            args.width,
            args.height,
        )
        print(f"Detection JSON saved to: {args.output_json}", file=sys.stderr)


def _run_vision_benchmark(
    args: argparse.Namespace,
    engine: Engine,
    tensor: np.ndarray,
    labels: list[str],
    image_path: Path,
) -> None:
    if args.benchmark_duration <= 0.0 and args.benchmark_iters is None:
        _die("--benchmark-duration must be > 0 when --benchmark-iters is not set")

    cfg = BenchmarkConfig(
        warmup_iters=args.benchmark_warmup,
        duration_secs=args.benchmark_duration,
        max_iters=args.benchmark_iters,
        report_every_secs=args.benchmark_report_every,
    )

    print(
        f"Benchmark config: warmup={cfg.warmup_iters}, duration={cfg.duration_secs:.2f}s, "
        f"max_iters={cfg.max_iters}, report_every={cfg.report_every_secs:.2f}s",
        file=sys.stderr,
    )

    request = engine.create_request()

    def step() -> None:
        engine.run_request(request, tensor)

    report = run_benchmark_loop(cfg, step)
    stage_timing = _run_vision_stage_timing(args, engine, image_path, labels)

    print("Benchmark results")
    print("-----------------")
    print(f"Warmup iterations  : {report.warmup_iters}")
    print(f"Measured iterations: {report.measured_iters}")
    print(f"Measured time (s)  : {report.measured_seconds:.3f}")
    print(f"Throughput (img/s) : {report.throughput_fps:.3f}")
    print(
        f"Latency (ms)       : mean={report.latency.mean_ms:.3f} "
        f"min={report.latency.min_ms:.3f} p50={report.latency.p50_ms:.3f} "
        f"p90={report.latency.p90_ms:.3f} p95={report.latency.p95_ms:.3f} "
        f"p99={report.latency.p99_ms:.3f} max={report.latency.max_ms:.3f}"
    )

    if stage_timing:
        _print_stage_timing(stage_timing)

    if args.output_json:
        write_benchmark_json(
            args.output_json,
            report,
            stage_timing,
            cfg,
            str(image_path),
            str(args.model),
            args.device,
            args.preprocess_backend,
            args.width,
            args.height,
        )
        print(f"Benchmark JSON saved to: {args.output_json}", file=sys.stderr)


def _run_vision_stage_timing(
    args: argparse.Namespace,
    engine: Engine,
    image_path: Path,
    labels: list[str],
) -> StageTimingReport | None:
    if args.benchmark_stage_iters == 0:
        return None

    n = args.benchmark_stage_iters
    io_sum = 0.0
    pre_sum = 0.0
    inf_sum = 0.0
    post_sum = 0.0
    total_sum = 0.0

    use_openvino_pp = args.preprocess_backend == "openvino"

    # Pre-decode if not reading each iter
    decoded_once = None
    if not args.benchmark_stage_read_each_iter:
        from PIL import Image as PILImage

        t = time.perf_counter()
        img = PILImage.open(image_path).convert("RGB")
        io_sum += (time.perf_counter() - t) * 1000.0
        decoded_once = img

    for _ in range(n):
        total_start = time.perf_counter()

        if args.benchmark_stage_read_each_iter:
            from PIL import Image as PILImage

            t = time.perf_counter()
            img = PILImage.open(image_path).convert("RGB")
            io_sum += (time.perf_counter() - t) * 1000.0
        else:
            img = decoded_once

        t0 = time.perf_counter()
        if use_openvino_pp:
            tensor = load_image_nhwc_no_resize(image_path)
        else:
            tensor = load_image_nhwc(image_path, args.width, args.height)
        pre_sum += (time.perf_counter() - t0) * 1000.0

        t2 = time.perf_counter()
        output = engine.infer(tensor)
        t3 = time.perf_counter()
        inf_sum += (t3 - t2) * 1000.0

        # Postprocess
        _results = top_k_classifications(output, args.top_k)
        post_sum += (time.perf_counter() - t3) * 1000.0

        total_sum += (time.perf_counter() - total_start) * 1000.0

    return _build_stage_report(
        n,
        args.benchmark_stage_read_each_iter,
        io_sum,
        pre_sum,
        inf_sum,
        post_sum,
        total_sum,
    )


# ──────────────────────────────────────────────────────────────────────────────
# `infer` command — ACT path
# ──────────────────────────────────────────────────────────────────────────────


def _run_infer_act(args: argparse.Namespace) -> None:
    if not args.episode_dir:
        _die("--episode-dir is required for --task act")
    episode_dir = Path(args.episode_dir)
    if not episode_dir.exists():
        _die(f"episode directory not found: {episode_dir}")

    metadata_path = (
        Path(args.metadata)
        if args.metadata
        else Path(args.model).with_name("metadata.yaml")
    )
    if not metadata_path.exists():
        _die(f"ACT metadata file not found: {metadata_path}")

    print(f"Loading ACT model: {args.model} (device: {args.device})", file=sys.stderr)

    act_engine = ActEngine(args.model, args.weights, args.device)
    meta = act_engine.metadata

    state = parse_state_from_episode(episode_dir / "data.jsonl")
    images = load_sample_images_from_episode(
        episode_dir, meta.image_width, meta.image_height
    )

    act_inputs = ActInputs(
        state=state,
        gripper_image=images["gripper"],
        overview_image=images["overview"],
    )

    if args.task == "act-benchmark":
        _run_act_benchmark(
            args, act_engine, meta, episode_dir, act_inputs, metadata_path
        )
    else:
        _run_act_single(args, act_engine, meta, episode_dir, act_inputs, metadata_path)


def _run_act_single(
    args: argparse.Namespace,
    act_engine: ActEngine,
    meta,
    episode_dir: Path,
    act_inputs: ActInputs,
    metadata_path: Path,
) -> None:
    state_t, gripper_t, overview_t = prepare_act_tensors(meta, act_inputs)
    request = act_engine.create_request()
    act_engine.run_request(request, state_t, gripper_t, overview_t)
    out = read_act_output(request, meta)

    print(f"ACT output: chunk_size={out.chunk_size}, action_dim={out.action_dim}")
    for i, a in enumerate(out.actions[:10]):
        print(f"step[{i:03d}] = {a}")
    if len(out.actions) > 10:
        print(f"... ({len(out.actions) - 10} more actions)")

    if args.output_json:
        write_act_json(
            args.output_json,
            str(args.model),
            str(metadata_path),
            str(episode_dir),
            meta.state_dim,
            out.chunk_size,
            out.action_dim,
            out.actions,
        )
        print(f"ACT JSON saved to: {args.output_json}", file=sys.stderr)


def _run_act_benchmark(
    args: argparse.Namespace,
    act_engine: ActEngine,
    meta,
    episode_dir: Path,
    act_inputs: ActInputs,
    metadata_path: Path,
) -> None:
    if args.benchmark_duration <= 0.0 and args.benchmark_iters is None:
        _die("--benchmark-duration must be > 0 when --benchmark-iters is not set")

    cfg = BenchmarkConfig(
        warmup_iters=args.benchmark_warmup,
        duration_secs=args.benchmark_duration,
        max_iters=args.benchmark_iters,
        report_every_secs=args.benchmark_report_every,
    )

    state_t, gripper_t, overview_t = prepare_act_tensors(meta, act_inputs)
    request = act_engine.create_request()

    def step() -> None:
        act_engine.run_request(request, state_t, gripper_t, overview_t)

    report = run_benchmark_loop(cfg, step)
    stage_timing = _run_act_stage_timing(
        args, act_engine, meta, episode_dir, act_inputs
    )

    print("ACT benchmark results")
    print("---------------------")
    print(f"Warmup iterations  : {report.warmup_iters}")
    print(f"Measured iterations: {report.measured_iters}")
    print(f"Measured time (s)  : {report.measured_seconds:.3f}")
    print(f"Throughput (run/s) : {report.throughput_fps:.3f}")
    print(
        f"Latency (ms)       : mean={report.latency.mean_ms:.3f} "
        f"min={report.latency.min_ms:.3f} p50={report.latency.p50_ms:.3f} "
        f"p90={report.latency.p90_ms:.3f} p95={report.latency.p95_ms:.3f} "
        f"p99={report.latency.p99_ms:.3f} max={report.latency.max_ms:.3f}"
    )

    if stage_timing:
        _print_stage_timing(stage_timing)

    if args.output_json:
        write_act_benchmark_json(
            args.output_json,
            str(args.model),
            str(metadata_path),
            str(episode_dir),
            args.device,
            cfg,
            report,
            stage_timing,
            meta.chunk_size,
            meta.action_dim,
        )
        print(f"ACT benchmark JSON saved to: {args.output_json}", file=sys.stderr)


def _run_act_stage_timing(
    args: argparse.Namespace,
    act_engine: ActEngine,
    meta,
    episode_dir: Path,
    act_inputs: ActInputs,
) -> StageTimingReport | None:
    if args.benchmark_stage_iters == 0:
        return None

    n = args.benchmark_stage_iters
    io_sum = 0.0
    pre_sum = 0.0
    inf_sum = 0.0
    post_sum = 0.0
    total_sum = 0.0

    # Pre-build tensors if not reading each iter
    prebuilt = None
    if not args.benchmark_stage_read_each_iter:
        t = time.perf_counter()
        prebuilt = prepare_act_tensors(meta, act_inputs)
        pre_sum += (time.perf_counter() - t) * 1000.0

    request = act_engine.create_request()

    for _ in range(n):
        total_start = time.perf_counter()

        if args.benchmark_stage_read_each_iter:
            t0 = time.perf_counter()
            state = parse_state_from_episode(episode_dir / "data.jsonl")
            images = load_sample_images_from_episode(
                episode_dir, meta.image_width, meta.image_height
            )
            local_inputs = ActInputs(
                state=state,
                gripper_image=images["gripper"],
                overview_image=images["overview"],
            )
            io_sum += (time.perf_counter() - t0) * 1000.0

            t1 = time.perf_counter()
            state_t, gripper_t, overview_t = prepare_act_tensors(meta, local_inputs)
            pre_sum += (time.perf_counter() - t1) * 1000.0
        else:
            state_t, gripper_t, overview_t = prebuilt

        t2 = time.perf_counter()
        act_engine.run_request(request, state_t, gripper_t, overview_t)
        t3 = time.perf_counter()
        inf_sum += (t3 - t2) * 1000.0

        _out = read_act_output(request, meta)
        post_sum += (time.perf_counter() - t3) * 1000.0

        total_sum += (time.perf_counter() - total_start) * 1000.0

    return _build_stage_report(
        n,
        args.benchmark_stage_read_each_iter,
        io_sum,
        pre_sum,
        inf_sum,
        post_sum,
        total_sum,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_stage_report(
    n: int,
    read_each_iter: bool,
    io_sum: float,
    pre_sum: float,
    inf_sum: float,
    post_sum: float,
    total_sum: float,
) -> StageTimingReport:
    io_mean = io_sum / n
    pre_mean = pre_sum / n
    inf_mean = inf_sum / n
    post_mean = post_sum / n
    total_mean = total_sum / n

    denom = max(io_mean + pre_mean + inf_mean + post_mean, 1e-9)

    return StageTimingReport(
        iterations=n,
        stage_read_each_iter=read_each_iter,
        io_decode_mean_ms=io_mean,
        preprocess_mean_ms=pre_mean,
        inference_mean_ms=inf_mean,
        postprocess_mean_ms=post_mean,
        total_mean_ms=total_mean,
        io_decode_share_pct=io_mean * 100.0 / denom,
        preprocess_share_pct=pre_mean * 100.0 / denom,
        inference_share_pct=inf_mean * 100.0 / denom,
        postprocess_share_pct=post_mean * 100.0 / denom,
    )


def _print_stage_timing(stage: StageTimingReport) -> None:
    print("\nStage timing (rough)")
    print("--------------------")
    print(f"Iterations          : {stage.iterations}")
    print(f"Read each iter      : {stage.stage_read_each_iter}")
    print(
        f"IO+Decode (ms)      : {stage.io_decode_mean_ms:.3f} ({stage.io_decode_share_pct:.1f}%)"
    )
    print(
        f"Preprocess (ms)     : {stage.preprocess_mean_ms:.3f} ({stage.preprocess_share_pct:.1f}%)"
    )
    print(
        f"Inference (ms)      : {stage.inference_mean_ms:.3f} ({stage.inference_share_pct:.1f}%)"
    )
    print(
        f"Postprocess (ms)    : {stage.postprocess_mean_ms:.3f} ({stage.postprocess_share_pct:.1f}%)"
    )
    print(f"Total (ms)          : {stage.total_mean_ms:.3f}")


def _die(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "devices":
        _run_devices()
    elif args.command == "check":
        _run_check(args)
    elif args.command == "infer":
        _run_infer(args)
    else:
        _die(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
