use anyhow::Result;
use openvino::Tensor;
use serde::Serialize;
use std::time::{Duration, Instant};

use crate::engine::Engine;

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkConfig {
    pub warmup_iters: usize,
    pub duration_secs: f64,
    pub max_iters: Option<usize>,
    pub report_every_secs: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LatencyStats {
    pub min_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkReport {
    pub warmup_iters: usize,
    pub measured_iters: usize,
    pub measured_seconds: f64,
    pub throughput_fps: f64,
    pub latency: LatencyStats,
}

#[derive(Debug, Clone, Serialize)]
pub struct StageTimingReport {
    pub iterations: usize,
    pub stage_read_each_iter: bool,
    pub io_decode_mean_ms: f64,
    pub preprocess_mean_ms: f64,
    pub inference_mean_ms: f64,
    pub postprocess_mean_ms: f64,
    pub total_mean_ms: f64,
    pub io_decode_share_pct: f64,
    pub preprocess_share_pct: f64,
    pub inference_share_pct: f64,
    pub postprocess_share_pct: f64,
}

pub fn run_benchmark(
    engine: &mut Engine,
    tensor: &Tensor,
    cfg: &BenchmarkConfig,
) -> Result<BenchmarkReport> {
    let mut request = engine.create_request()?;

    run_benchmark_loop(cfg, || engine.run_request(&mut request, tensor))
}

pub fn run_benchmark_loop<F>(cfg: &BenchmarkConfig, mut step: F) -> Result<BenchmarkReport>
where
    F: FnMut() -> Result<()>,
{
    for _ in 0..cfg.warmup_iters {
        step()?;
    }

    let mut latencies_us: Vec<u64> = Vec::new();
    let start = Instant::now();
    let mut last_report = start;
    let mut measured_iters = 0usize;

    loop {
        if let Some(max_iters) = cfg.max_iters {
            if measured_iters >= max_iters {
                break;
            }
        } else if start.elapsed().as_secs_f64() >= cfg.duration_secs {
            break;
        }

        let iter_start = Instant::now();
        step()?;
        let elapsed = iter_start.elapsed();
        latencies_us.push(duration_to_micros(elapsed));
        measured_iters += 1;

        if cfg.report_every_secs > 0.0
            && last_report.elapsed().as_secs_f64() >= cfg.report_every_secs
        {
            let secs = start.elapsed().as_secs_f64();
            let fps = measured_iters as f64 / secs.max(1e-9);
            eprintln!("Benchmark progress: {measured_iters} iters, {fps:.2} img/s");
            last_report = Instant::now();
        }
    }

    let measured_seconds = start.elapsed().as_secs_f64();
    let throughput_fps = measured_iters as f64 / measured_seconds.max(1e-9);
    let latency = compute_latency_stats(&latencies_us);

    Ok(BenchmarkReport {
        warmup_iters: cfg.warmup_iters,
        measured_iters,
        measured_seconds,
        throughput_fps,
        latency,
    })
}

fn compute_latency_stats(latencies_us: &[u64]) -> LatencyStats {
    if latencies_us.is_empty() {
        return LatencyStats {
            min_ms: 0.0,
            mean_ms: 0.0,
            p50_ms: 0.0,
            p90_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            max_ms: 0.0,
        };
    }

    let mut sorted = latencies_us.to_vec();
    sorted.sort_unstable();

    let sum_us: u128 = latencies_us.iter().map(|&x| x as u128).sum();
    let mean_us = sum_us as f64 / latencies_us.len() as f64;

    LatencyStats {
        min_ms: us_to_ms(*sorted.first().unwrap_or(&0)),
        mean_ms: us_to_ms_f64(mean_us),
        p50_ms: percentile_ms(&sorted, 50.0),
        p90_ms: percentile_ms(&sorted, 90.0),
        p95_ms: percentile_ms(&sorted, 95.0),
        p99_ms: percentile_ms(&sorted, 99.0),
        max_ms: us_to_ms(*sorted.last().unwrap_or(&0)),
    }
}

fn percentile_ms(sorted_us: &[u64], p: f64) -> f64 {
    if sorted_us.is_empty() {
        return 0.0;
    }
    let n = sorted_us.len();
    let rank = ((p / 100.0) * (n as f64 - 1.0)).round() as usize;
    us_to_ms(sorted_us[rank.min(n - 1)])
}

fn duration_to_micros(d: Duration) -> u64 {
    d.as_micros().min(u64::MAX as u128) as u64
}

fn us_to_ms(us: u64) -> f64 {
    us as f64 / 1000.0
}

fn us_to_ms_f64(us: f64) -> f64 {
    us / 1000.0
}
