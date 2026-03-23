#!/usr/bin/env bash
# Launcher for inference-rs with bundled OpenVINO libraries.
# Sets LD_LIBRARY_PATH so the binary can dlopen the co-located .so files.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "${SCRIPT_DIR}/inference-rs" "$@"
