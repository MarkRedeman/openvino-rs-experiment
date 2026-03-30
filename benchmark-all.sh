#!/usr/bin/env bash
# benchmark-all.sh — Run benchmarks across all model/device/precision combos
# and print a summary table with throughput (img/s or run/s).
#
# Usage:
#   ./benchmark-all.sh
#
# Prerequisites:
#   - Docker image built: docker compose build
#   - jq installed
#   - Models present in ./models/
#   - ./images/diamond-card.jpg and ./images/fish.png for vision models
#   - ./episodes/ep_000_dc7198bd/ for ACT model
set -euo pipefail

DURATION=10
WARMUP=20

# ── Model definitions ────────────────────────────────────────────────────────
# Each entry: "label|task|model_fp32_xml|model_fp32_bin|model_fp16_xml|model_fp16_bin|extra_args"
MODELS=(
  "Fish Detection|benchmark|/models/fish-detection/model.xml|/models/fish-detection/model.bin|/models/fish-detection-fp16/model.xml|/models/fish-detection-fp16/model.bin|--image /images/fish.png --task benchmark --detection-format geti --threshold 0.3 --width 992 --height 800"
  "Card Classification|benchmark|/models/card-classification/model.xml|/models/card-classification/model.bin|/models/card-classification-fp16/model.xml|/models/card-classification-fp16/model.bin|--image /images/diamond-card.jpg --task benchmark --width 224 --height 224"
  "ACT|act-benchmark|/models/act-openvino/act.xml|/models/act-openvino/act.bin|/models/act-openvino-fp16/act.xml|/models/act-openvino-fp16/act.bin|--task act-benchmark --episode-dir /episodes/ep_000_dc7198bd"
)

DEVICES=(CPU GPU NPU)

# Map device → docker compose service
declare -A DEVICE_SERVICE
DEVICE_SERVICE[CPU]=inference
DEVICE_SERVICE[GPU]=inference-gpu
DEVICE_SERVICE[NPU]=inference-npu

# ── Working directory ─────────────────────────────────────────────────────────
TMPDIR_BENCH="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_BENCH"' EXIT

# ── Results storage ───────────────────────────────────────────────────────────
# results[label|device|precision] = "123.45" or "DNC"
declare -A RESULTS
# notes[label|device|precision] = "error reason"
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

  # Build the command. We mount a temp dir to capture JSON output.
  local output
  if output=$(docker compose run --rm \
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
    --benchmark-stage-iters 0 \
    --output-json /benchout/result.json \
    2>&1); then
    # Success — extract throughput
    if [[ -f "$json_file" ]]; then
      local fps
      fps=$(jq -r '.report.throughput_fps' "$json_file" 2>/dev/null || echo "")
      if [[ -n "$fps" && "$fps" != "null" ]]; then
        # Round to 1 decimal
        fps=$(printf "%.1f" "$fps")
        RESULTS["$key"]="$fps"
        echo "${fps}/s"
        return
      fi
    fi
    # JSON missing or unparseable
    RESULTS["$key"]="DNC"
    NOTE_INDEX=$((NOTE_INDEX + 1))
    NOTES["$key"]="$NOTE_INDEX"
    NOTE_LIST+=("[$NOTE_INDEX] ${label} / ${device} / ${precision}: benchmark completed but no throughput in output")
    echo "DNC [${NOTE_INDEX}]"
  else
    # Failed — capture error
    RESULTS["$key"]="DNC"
    NOTE_INDEX=$((NOTE_INDEX + 1))
    NOTES["$key"]="$NOTE_INDEX"
    # Extract the most useful error line (last non-empty line, trimmed)
    local err_line
    err_line=$(echo "$output" | grep -iE '(error|fail|cannot|unable|unsupported|not found)' | tail -1 | sed 's/^[[:space:]]*//')
    if [[ -z "$err_line" ]]; then
      err_line=$(echo "$output" | tail -5 | sed 's/^[[:space:]]*//')
    fi
    NOTE_LIST+=("[$NOTE_INDEX] ${label} / ${device} / ${precision}: ${err_line}")
    echo "DNC [${NOTE_INDEX}]"
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "========================================"
echo " Benchmark Suite"
echo " Duration: ${DURATION}s per run, ${WARMUP} warmup iters"
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

# ── Print table ───────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Results"
echo "========================================"
echo ""

# Column widths
MODEL_W=22
CELL_W=12

# Helper: pad/truncate string to width
pad() {
  printf "%-${1}s" "$2"
}

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
