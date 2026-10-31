# syntax=docker/dockerfile:1
# Reproducible production container for SINGULARITY on 8x NVIDIA B200 (SM 10.0 / sm_100a)
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH="/root/.cargo/bin:/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}" \
    CUDA_HOME="/usr/local/cuda" \
    NVCC_CCBIN="g++-13" \
    RUSTUP_HOME="/root/.rustup" \
    CARGO_HOME="/root/.cargo"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    g++-13 \
    gcc-13 \
    git \
    libssl-dev \
    pkg-config \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.88.0 --profile minimal \
    && rustc --version \
    && cargo --version \
    && nvcc --version

WORKDIR /workspace/singularity
COPY Cargo.toml Cargo.lock ./
COPY singularity-core ./singularity-core
COPY singularity-cuda ./singularity-cuda
COPY singularity-cache ./singularity-cache
COPY singularity-grammar ./singularity-grammar
COPY singularity-model ./singularity-model
COPY singularity-scheduler ./singularity-scheduler
COPY singularity-server ./singularity-server
COPY singularity-cli ./singularity-cli
COPY singularity-tests ./singularity-tests

RUN cargo build --release --features cuda --bin singularity \
    && cp /workspace/singularity/target/release/singularity /usr/local/bin/singularity

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/singularity"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--v41-config", "/models/deepseek-v41-flash-1017/config.json", "--model-path", "/models/deepseek-v41-flash-1017/model.safetensors.index.json", "--tokenizer-path", "/models/deepseek-v41-flash-1017/tokenizer.json", "--use-cuda", "--dspark"]
