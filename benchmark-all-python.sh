#!/usr/bin/env bash
# benchmark-all-python.sh — Run Python benchmarks inside Docker across all
# model/device/precision combos and print a summary table with throughput
# (img/s or run/s), plus a stage timing breakdown (IO+decode, preprocess,
# inference, postprocess).
#
# Usage:
#   ./benchmark-all-python.sh
#
# Prerequisites:
#   - Docker image built: docker compose -f python/docker-compose.python.yml build
#   - jq installed
#   - Models present in ./models/
#   - ./images/diamond-card.jpg and ./images/fish.png for vision models
#   - ./episodes/ep_000_dc7198bd/ for ACT model
set -euo pipefail

DURATION=10
WARMUP=20
STAGE_ITERS=50

COMPOSE_FILE="python/docker-compose.python.yml"

# ── Model definitions ────────────────────────────────────────────────────────
# Each entry: "label|task|model_fp32_xml|model_fp32_bin|model_fp16_xml|model_fp16_bin|extra_args"
# Paths are container paths (volumes mount host dirs to /models, /images, /episodes).
MODELS=(
  "Fish Detection|benchmark|/models/fish-detection/model.xml|/models/fish-detection/model.bin|/models/fish-detection-fp16/model.xml|/models/fish-detection-fp16/model.bin|--image /images/fish.png --task benchmark --detection-format geti --threshold 0.3 --width 992 --height 800"
  "Card Classification|benchmark|/models/card-classification/model.xml|/models/card-classification/model.bin|/models/card-classification-fp16/model.xml|/models/card-classification-fp16/model.bin|--image /images/diamond-card.jpg --task benchmark --width 224 --height 224"
  "ACT|act-benchmark|/models/act-openvino/act.xml|/models/act-openvino/act.bin|/models/act-openvino-fp16/act.xml|/models/act-openvino-fp16/act.bin|--task act-benchmark --episode-dir /episodes/ep_000_dc7198bd"
)

DEVICES=(CPU GPU NPU)

# Map device → docker compose service
declare -A DEVICE_SERVICE
DEVICE_SERVICE[CPU]=inference-py
DEVICE_SERVICE[GPU]=inference-gpu-py
DEVICE_SERVICE[NPU]=inference-npu-py

# ── Working directory ─────────────────────────────────────────────────────────
TMPDIR_BENCH="$(mktemp -d)"
chmod 777 "$TMPDIR_BENCH"  # container runs as uid 1001 (openvino)
trap 'rm -rf "$TMPDIR_BENCH"' EXIT

# ── Results storage ───────────────────────────────────────────────────────────
# results[label|device|precision] = "123.45" or "DNC"
declare -A RESULTS
# stage timing[label|device|precision] = "12.345" (ms)
declare -A STAGE_IO STAGE_PRE STAGE_INF STAGE_POST STAGE_TOTAL
# notes[label|device|precision] = note index
declare -A NOTES
NOTE_INDEX=0
# note_list[index] = "text"
declare -a NOTE_LIST

# ── Run a single benchmark ───────────────────────────────────────────────────
run_one() {
  local label="$1" device="$2" precision="$3" model_xml="$4" model_bin="$5" extra_args="$6"
  local service="${DEVICE_SERVICE[$device]}"
  local json_file="$TMPDIR_BENCH/result.json"
  local key="${label}|${device}|${precision}"

  rm -f "$json_file"

  echo -n "  ${label} / ${device} / ${precision} ... "

  # Run benchmark inside Docker container.
  local output
  local rc=0
  output=$(docker compose -f "$COMPOSE_FILE" run --rm \
    -v "$TMPDIR_BENCH:/benchout" \
    "$service" \
    infer \
    --model "$model_xml" \
    --weights "$model_bin" \
    --device "$device" \
    $extra_args \
    --benchmark-warmup "$WARMUP" \
    --benchmark-duration "$DURATION" \
    --benchmark-report-every 0 \
    --benchmark-stage-iters "$STAGE_ITERS" \
    --benchmark-stage-read-each-iter \
    --output-json /benchout/result.json \
    2>&1) || rc=$?

  # Check for JSON output first — the benchmark may have succeeded even if
  # the process exited non-zero (e.g. permission error writing JSON).
  if [[ -f "$json_file" ]]; then
    local fps
    fps=$(jq -r '.report.throughput_fps' "$json_file" 2>/dev/null || echo "")
    if [[ -n "$fps" && "$fps" != "null" ]]; then
      fps=$(printf "%.1f" "$fps")
      RESULTS["$key"]="$fps"

      # Extract stage timing if present.
      local st
      st=$(jq -r '.stage_timing // empty' "$json_file" 2>/dev/null || true)
      if [[ -n "$st" ]]; then
        STAGE_IO["$key"]=$(jq -r '.stage_timing.io_decode_mean_ms'  "$json_file" 2>/dev/null || echo "")
        STAGE_PRE["$key"]=$(jq -r '.stage_timing.preprocess_mean_ms' "$json_file" 2>/dev/null || echo "")
        STAGE_INF["$key"]=$(jq -r '.stage_timing.inference_mean_ms'  "$json_file" 2>/dev/null || echo "")
        STAGE_POST["$key"]=$(jq -r '.stage_timing.postprocess_mean_ms' "$json_file" 2>/dev/null || echo "")
        STAGE_TOTAL["$key"]=$(jq -r '.stage_timing.total_mean_ms'    "$json_file" 2>/dev/null || echo "")
      fi

      echo "${fps}/s"
      return
    fi
  fi

  # No JSON output — try to extract throughput from stdout as fallback.
  # The line looks like: "Throughput (img/s) : 243.904" or "Throughput (run/s) : 73.1"
  local stdout_fps
  stdout_fps=$(echo "$output" | grep -oP 'Throughput \([^)]+\)\s*:\s*\K[0-9.]+' | head -1 || true)
  if [[ -n "$stdout_fps" ]]; then
    stdout_fps=$(printf "%.1f" "$stdout_fps")
    RESULTS["$key"]="$stdout_fps"
    echo "${stdout_fps}/s"
    return
  fi

  # Genuine failure — record DNC with error details.
  RESULTS["$key"]="DNC"
  NOTE_INDEX=$((NOTE_INDEX + 1))
  NOTES["$key"]="$NOTE_INDEX"
  local err_line
  err_line=$(echo "$output" | grep -iE '(error|fail|cannot|unable|unsupported|not found)' | tail -1 | sed 's/^[[:space:]]*//')
  if [[ -z "$err_line" ]]; then
    err_line=$(echo "$output" | tail -5 | sed 's/^[[:space:]]*//')
  fi
  NOTE_LIST+=("[$NOTE_INDEX] ${label} / ${device} / ${precision}: ${err_line}")
  echo "DNC [${NOTE_INDEX}]"
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "========================================"
echo " Python Benchmark Suite (Docker)"
echo " Duration: ${DURATION}s per run, ${WARMUP} warmup iters"
echo " Stage timing: ${STAGE_ITERS} iters, read each iter"
echo " Devices: ${DEVICES[*]}"
echo "========================================"
echo ""

for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r label task fp32_xml fp32_bin fp16_xml fp16_bin extra_args <<< "$model_entry"

  for device in "${DEVICES[@]}"; do
    # FP32
    run_one "$label" "$device" "FP32" "$fp32_xml" "$fp32_bin" "$extra_args"
    # FP16
    run_one "$label" "$device" "FP16" "$fp16_xml" "$fp16_bin" "$extra_args"
  done
done

# ── Print throughput table ────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Throughput Results (Python / Docker)"
echo "========================================"
echo ""

# Column widths
MODEL_W=22
CELL_W=12

# Helper: right-pad cell, adding note ref if DNC
cell() {
  local label="$1" device="$2" precision="$3"
  local key="${label}|${device}|${precision}"
  local val="${RESULTS[$key]:-—}"
  if [[ "$val" == "DNC" && -n "${NOTES[$key]:-}" ]]; then
    val="DNC [${NOTES[$key]}]"
  fi
  printf "%-${CELL_W}s" "$val"
}

# Header
printf "| %-${MODEL_W}s" "Model"
for device in "${DEVICES[@]}"; do
  printf "| %-${CELL_W}s" "${device} FP32"
  printf "| %-${CELL_W}s" "${device} FP16"
done
echo "|"

# Separator
printf "| "
printf '%0.s-' $(seq 1 $MODEL_W)
for device in "${DEVICES[@]}"; do
  printf " | "
  printf '%0.s-' $(seq 1 $CELL_W)
  printf " | "
  printf '%0.s-' $(seq 1 $CELL_W)
done
echo " |"

# Data rows
for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r label _ _ _ _ _ _ <<< "$model_entry"
  printf "| %-${MODEL_W}s" "$label"
  for device in "${DEVICES[@]}"; do
    printf "| "
    cell "$label" "$device" "FP32"
    printf "| "
    cell "$label" "$device" "FP16"
  done
  echo "|"
done

# ── Print stage timing table ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Stage Timing (${STAGE_ITERS} iters, disk re-read each iter)"
echo "========================================"
echo ""

SLABEL_W=30
SCELL_W=11

fmt_ms() {
  local val="$1"
  if [[ -z "$val" || "$val" == "null" ]]; then
    printf "%-${SCELL_W}s" "—"
  else
    printf "%-${SCELL_W}s" "$(printf "%.3f" "$val")"
  fi
}

# Header
printf "| %-${SLABEL_W}s| %-${SCELL_W}s| %-${SCELL_W}s| %-${SCELL_W}s| %-${SCELL_W}s| %-${SCELL_W}s|\n" \
  "Model / Device / Prec" "IO+Dec ms" "Preproc ms" "Infer ms" "Postpr ms" "Total ms"

# Separator
printf "| "
printf '%0.s-' $(seq 1 $SLABEL_W)
for _ in $(seq 1 5); do
  printf "| "
  printf '%0.s-' $(seq 1 $SCELL_W)
done
echo "|"

# Data rows
for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r label _ _ _ _ _ _ <<< "$model_entry"
  for device in "${DEVICES[@]}"; do
    for precision in FP32 FP16; do
      key="${label}|${device}|${precision}"
      result="${RESULTS[$key]:-—}"
      if [[ "$result" == "DNC" || "$result" == "—" ]]; then
        printf "| %-${SLABEL_W}s| %-${SCELL_W}s| %-${SCELL_W}s| %-${SCELL_W}s| %-${SCELL_W}s| %-${SCELL_W}s|\n" \
          "${label} ${device} ${precision}" "—" "—" "—" "—" "—"
      else
        printf "| %-${SLABEL_W}s| " "${label} ${device} ${precision}"
        fmt_ms "${STAGE_IO[$key]:-}"
        printf "| "
        fmt_ms "${STAGE_PRE[$key]:-}"
        printf "| "
        fmt_ms "${STAGE_INF[$key]:-}"
        printf "| "
        fmt_ms "${STAGE_POST[$key]:-}"
        printf "| "
        fmt_ms "${STAGE_TOTAL[$key]:-}"
        echo "|"
      fi
    done
  done
done

# ── Notes ─────────────────────────────────────────────────────────────────────
if [[ ${#NOTE_LIST[@]} -gt 0 ]]; then
  echo ""
  echo "Notes:"
  for note in "${NOTE_LIST[@]}"; do
    echo "  $note"
  done
fi

echo ""
echo "Done. ${#NOTE_LIST[@]} failure(s) out of $(( ${#MODELS[@]} * ${#DEVICES[@]} * 2 )) runs."
