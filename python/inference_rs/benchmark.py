"""Benchmark loop and latency statistics.

Matches the Rust ``benchmark.rs`` implementation:
- Warmup N iterations (results discarded)
- Measurement loop: time-based or iteration-limited
- Microsecond-resolution latency measurements
- Nearest-rank percentile: ``rank = round((P/100) * (n-1))``
- Throughput: ``measured_iters / max(measured_seconds, 1e-9)``
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchmarkConfig:
    warmup_iters: int = 20
    duration_secs: float = 10.0
    max_iters: int | None = None
    report_every_secs: float = 1.0

    def to_dict(self) -> dict:
        return {
            "warmup_iters": self.warmup_iters,
            "duration_secs": self.duration_secs,
            "max_iters": self.max_iters,
            "report_every_secs": self.report_every_secs,
        }


@dataclass
class LatencyStats:
    min_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "min_ms": self.min_ms,
            "mean_ms": self.mean_ms,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


@dataclass
class BenchmarkReport:
    warmup_iters: int = 0
    measured_iters: int = 0
    measured_seconds: float = 0.0
    throughput_fps: float = 0.0
    latency: LatencyStats = field(default_factory=LatencyStats)

    def to_dict(self) -> dict:
        return {
            "warmup_iters": self.warmup_iters,
            "measured_iters": self.measured_iters,
            "measured_seconds": self.measured_seconds,
            "throughput_fps": self.throughput_fps,
            "latency": self.latency.to_dict(),
        }


@dataclass
class StageTimingReport:
    iterations: int = 0
    stage_read_each_iter: bool = False
    io_decode_mean_ms: float = 0.0
    preprocess_mean_ms: float = 0.0
    inference_mean_ms: float = 0.0
    postprocess_mean_ms: float = 0.0
    total_mean_ms: float = 0.0
    io_decode_share_pct: float = 0.0
    preprocess_share_pct: float = 0.0
    inference_share_pct: float = 0.0
    postprocess_share_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "iterations": self.iterations,
            "stage_read_each_iter": self.stage_read_each_iter,
            "io_decode_mean_ms": self.io_decode_mean_ms,
            "preprocess_mean_ms": self.preprocess_mean_ms,
            "inference_mean_ms": self.inference_mean_ms,
            "postprocess_mean_ms": self.postprocess_mean_ms,
            "total_mean_ms": self.total_mean_ms,
            "io_decode_share_pct": self.io_decode_share_pct,
            "preprocess_share_pct": self.preprocess_share_pct,
            "inference_share_pct": self.inference_share_pct,
            "postprocess_share_pct": self.postprocess_share_pct,
        }


def run_benchmark_loop(
    cfg: BenchmarkConfig, step: Callable[[], None]
) -> BenchmarkReport:
    """Run the benchmark loop: warmup, then timed measurement.

    *step* is called once per iteration and should perform one complete
    inference pass.
    """
    # Warmup
    for _ in range(cfg.warmup_iters):
        step()

    latencies_us: list[int] = []
    start = time.perf_counter()
    last_report = start
    measured_iters = 0

    while True:
        if cfg.max_iters is not None:
            if measured_iters >= cfg.max_iters:
                break
        elif time.perf_counter() - start >= cfg.duration_secs:
            break

        iter_start = time.perf_counter()
        step()
        elapsed = time.perf_counter() - iter_start
        latencies_us.append(int(elapsed * 1_000_000))
        measured_iters += 1

        if cfg.report_every_secs > 0.0:
            now = time.perf_counter()
            if now - last_report >= cfg.report_every_secs:
                secs = now - start
                fps = measured_iters / max(secs, 1e-9)
                print(
                    f"Benchmark progress: {measured_iters} iters, {fps:.2f} img/s",
                    file=sys.stderr,
                )
                last_report = now

    measured_seconds = time.perf_counter() - start
    throughput_fps = measured_iters / max(measured_seconds, 1e-9)
    latency = _compute_latency_stats(latencies_us)

    return BenchmarkReport(
        warmup_iters=cfg.warmup_iters,
        measured_iters=measured_iters,
        measured_seconds=measured_seconds,
        throughput_fps=throughput_fps,
        latency=latency,
    )


def _compute_latency_stats(latencies_us: list[int]) -> LatencyStats:
    if not latencies_us:
        return LatencyStats()

    sorted_us = sorted(latencies_us)
    sum_us = sum(latencies_us)
    mean_us = sum_us / len(latencies_us)

    return LatencyStats(
        min_ms=sorted_us[0] / 1000.0,
        mean_ms=mean_us / 1000.0,
        p50_ms=_percentile_ms(sorted_us, 50.0),
        p90_ms=_percentile_ms(sorted_us, 90.0),
        p95_ms=_percentile_ms(sorted_us, 95.0),
        p99_ms=_percentile_ms(sorted_us, 99.0),
        max_ms=sorted_us[-1] / 1000.0,
    )


def _percentile_ms(sorted_us: list[int], p: float) -> float:
    """Nearest-rank percentile matching Rust: ``rank = round((P/100) * (n-1))``."""
    if not sorted_us:
        return 0.0
    n = len(sorted_us)
    rank = round((p / 100.0) * (n - 1))
    rank = min(rank, n - 1)
    return sorted_us[rank] / 1000.0
