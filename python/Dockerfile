# ==============================================================================
# Python reference implementation — uses the same runtime base image as Rust
# for fair benchmarking comparison.
# ==============================================================================
FROM openvino/ubuntu24_runtime:2026.0.0

USER root

# Install uv (fast Python package manager).
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy project files and install dependencies.
COPY pyproject.toml ./
COPY inference_rs/ inference_rs/

RUN uv sync --no-dev

# Create writable tmp directory for the openvino user.
RUN mkdir -p /tmp/ov && chmod 777 /tmp/ov

USER openvino

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["inference-rs-py"]
