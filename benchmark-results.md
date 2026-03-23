# Benchmark Results Analysis

Date: 2026-03-23

## Setup Used

- Runtime: `./standalone/run-inference.sh`
- Benchmark mode: `--task benchmark`
- Duration: `10s`
- Warmup: `20 iters`
- Stage timing: `--benchmark-stage-iters 300 --benchmark-stage-read-each-iter`
- Compared preprocess backends:
  - `rust` (resize + normalize in Rust before inference)
  - `openvino` (decode in Rust, resize delegated to OpenVINO pipeline)

## Raw Results Summary

| Model                       |  Backend | Throughput (img/s) | Mean Latency (ms) | p95 (ms) | Stage Total (ms) | IO+Decode (ms) | Preprocess (ms) | Inference (ms) | Postprocess (ms) |
|-----------------------------|---------:|-------------------:|------------------:|---------:|-----------------:|---------------:|----------------:|---------------:|-----------------:|
| Fish detection 992x800      |     rust |             26.620 |            37.565 |   40.471 |           71.765 |          6.008 |          27.612 |         38.142 |            0.001 |
| Fish detection 992x800      | openvino |             26.120 |            38.284 |   41.713 |           48.904 |          7.177 |           2.720 |         39.004 |            0.001 |
| Card classification 224x224 |     rust |            498.339 |             2.006 |    2.248 |           10.659 |          3.065 |           4.577 |          2.835 |            0.001 |
| Card classification 224x224 | openvino |            378.674 |             2.640 |    3.034 |           10.197 |          3.610 |           3.001 |          3.584 |            0.001 |

## Clarifications and Interpretation

### 1) Why OpenVINO shows much lower `Preprocess` but similar/worse throughput

This is expected with the current instrumentation:

- In `openvino` mode, resize is moved from Rust preprocessing into OpenVINO's internal pipeline.
- The stage timer counts that extra work under `Inference` (because it happens during `request.infer()`), not under `Preprocess`.
- So the lower `Preprocess` number in `openvino` mode is not a free win; some of that cost shifts to `Inference`.

### 2) Why classification got significantly slower with `openvino`

For the small 224x224 classification model, the network itself is very fast. Adding OpenVINO-side resize/conversion inside every inference call increases the critical path enough to reduce throughput:

- Rust backend: ~498 img/s
- OpenVINO backend: ~379 img/s (about 24% lower)

On this workload, Rust-side preprocessing done once upfront appears cheaper overall than paying OpenVINO preprocessing every iteration.

### 3) Why detection changed only slightly

For fish detection (992x800), model inference is much heavier (~38-39 ms). Resize strategy has smaller influence on end-to-end throughput:

- Rust backend: ~26.62 img/s
- OpenVINO backend: ~26.12 img/s (about 1.9% lower)

This difference is small and could vary run-to-run based on CPU scheduling and thermal state.

### 4) Stage timing `Total (ms)` does not match `1 / throughput`

This is expected. The two sections measure different things:

- Throughput/latency section uses the main benchmark loop with one prebuilt tensor reused repeatedly.
- Stage timing section intentionally performs per-iteration read/decode and preprocessing (`--benchmark-stage-read-each-iter`) to estimate rough stage costs.

So stage `Total` is a rough per-request pipeline estimate, not the same metric as sustained tensor-reuse throughput.

## Practical Conclusions

1. Keep `--preprocess-backend rust` as the default for best throughput on current tested models.
2. Use `openvino` backend mainly for experimentation or when you want preprocessing delegated/standardized in OpenVINO.
3. For optimization efforts, biggest remaining cost on detection is still inference itself; preprocessing is secondary.
4. For fair backend comparison, run multiple repetitions and compare medians (or average of 3-5 runs).

## Suggested Next Benchmark Commands

Use these to reduce run-to-run noise:

```bash
# Detection, 3 repeats each backend (collect medians)
for i in 1 2 3; do
  ./standalone/run-inference.sh --model models/fish-detection/model.xml --weights models/fish-detection/model.bin --image images/fish.png --task benchmark --detection-format geti --threshold 0.3 --width 992 --height 800 --benchmark-warmup 20 --benchmark-duration 15 --benchmark-report-every 0 --preprocess-backend rust
done

for i in 1 2 3; do
  ./standalone/run-inference.sh --model models/fish-detection/model.xml --weights models/fish-detection/model.bin --image images/fish.png --task benchmark --detection-format geti --threshold 0.3 --width 992 --height 800 --benchmark-warmup 20 --benchmark-duration 15 --benchmark-report-every 0 --preprocess-backend openvino
done
```
