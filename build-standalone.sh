#!/usr/bin/env bash
# build-standalone.sh
#
# Compiles inference-rs inside Docker, bundles it with the minimal set of
# OpenVINO runtime libraries, and extracts a self-contained directory to
# the host filesystem.
#
# After running this script you will have:
#
#   standalone/
#   ├── inference-rs          # the binary
#   ├── lib/                  # OpenVINO + TBB shared libraries
#   └── run-inference.sh      # launcher (sets LD_LIBRARY_PATH for you)
#
# Usage:
#   ./build-standalone.sh
#   ./standalone/run-inference.sh --help
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="inference-rs-standalone"
OUTPUT_DIR="${SCRIPT_DIR}/standalone"

echo "==> Building Docker image (this compiles Rust + bundles OpenVINO libs)..."
docker build -f "${SCRIPT_DIR}/Dockerfile.standalone" -t "${IMAGE_NAME}" "${SCRIPT_DIR}"

echo "==> Extracting standalone bundle..."
rm -rf "${OUTPUT_DIR}"
docker run --rm "${IMAGE_NAME}" | tar xzf - -C "${SCRIPT_DIR}"

echo ""
echo "==> Done!  Standalone bundle is at: ${OUTPUT_DIR}/"
echo ""
ls -lh "${OUTPUT_DIR}/inference-rs"
echo ""
echo "Libraries bundled:"
ls -lh "${OUTPUT_DIR}/lib/"
echo ""
echo "Run with:"
echo "  ${OUTPUT_DIR}/run-inference.sh --help"
