# ==============================================================================
# Stage 1: Build the Rust binary inside the OpenVINO dev image.
#
# The dev image ships the OpenVINO shared libraries *and* C headers, which
# satisfies the openvino-sys build script even though we use runtime-linking.
# ==============================================================================
FROM openvino/ubuntu24_dev:2026.0.0 AS builder

USER root

# Install the Rust toolchain (stable).
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl build-essential pkg-config && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Cache dependencies by building a dummy project first.
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && \
    echo 'fn main() {}' > src/main.rs && \
    echo '' > src/lib.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src && \
    rm -f  target/release/inference-rs target/release/deps/inference_rs-* && \
    rm -rf target/release/.fingerprint/inference-rs-* && \
    rm -f  target/release/deps/libinference_rs-* && \
    rm -f  target/release/libinference_rs-*

# Copy real source code and build.
COPY fonts/ fonts/
COPY src/ src/
RUN cargo build --release

# ==============================================================================
# Stage 2: Slim runtime image with only the OpenVINO shared libraries.
# ==============================================================================
FROM openvino/ubuntu24_runtime:2026.0.0

COPY --from=builder /app/target/release/inference-rs /usr/local/bin/inference-rs

ENTRYPOINT ["inference-rs"]
