"""Pinned checkpoint, hardware, and runtime configuration for SINGULARITY Modal deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

# Pinned Hugging Face model artifact constants (singularity2.md Section 2)
MODEL_ID: str = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
PINNED_REVISION: str = "1bae0e1088993b9b82de08f59de5be5a34e42170"
PINNED_INDEX_SHA256: str = (
    "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
)
PINNED_SHARD_COUNT: int = 48
PINNED_TENSOR_COUNT: int = 96_085
PINNED_TOTAL_BYTES: int = 510_286_023_000

# Lossless CSF container checkpoint constants (local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF)
CSF_MODEL_ID: str = "local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF"
CSF_PINNED_REVISION: str = "90d665c19e1a9ff2097cc787515e370d86b616f0"
CSF_SCHEMA: str = "lil-mxfp4-csf-checkpoint/1"
CSF_CODEC: str = "row-base-offset1-u24-exceptions/1"
CSF_COMPRESSED_SCALE_COUNT: int = 46_080
CSF_RAW_HEADER_TENSOR_COUNT: int = PINNED_TENSOR_COUNT + CSF_COMPRESSED_SCALE_COUNT
CSF_TOTAL_BYTES: int = 495_632_175_852

# Modal hardware & container specifications (singularity2.md Sections 6 & 8)
APP_NAME: str = "singularity-v41-flash-1017"
DEEPINFER_APP_NAME: str = "deepinfer-v41-reference"
GPU_SPEC: str = "B200:8"
GPU_COUNT: int = 8
EXPECTED_SM_MAJOR: int = 10
EXPECTED_SM_MINOR: int = 0
HBM_PER_GPU_GIB: int = 180
CPU_CORES: float = 64.0
MEMORY_MB: int = 524_288  # 512 GiB host memory for mmap & shard verification

# Persistent Modal Volumes & Secrets
VOLUME_NAME: str = "singularity-v41-flash-1017-weights"
VOLUME_MOUNT_PATH: str = "/models"
CHECKPOINT_SUBDIR: str = "deepseek-v41-flash-1017"
CHECKPOINT_DIR: str = f"{VOLUME_MOUNT_PATH}/{CHECKPOINT_SUBDIR}"

BENCHMARK_VOLUME_NAME: str = "singularity-v41-benchmarks"
BENCHMARK_MOUNT_PATH: str = "/benchmarks"

HF_SECRET_NAME: str = "huggingface-secret"
API_SECRET_NAME: str = "singularity-api-secret"

# Container scaling and timeouts
MIN_CONTAINERS: int = 1
MAX_CONTAINERS: int = 1
SCALEDOWN_WINDOW_SECS: int = 900
STARTUP_TIMEOUT_SECS: int = 3600
REQUEST_TIMEOUT_SECS: int = 1800
INTERNAL_SERVER_HOST: str = "127.0.0.1"
INTERNAL_SERVER_PORT: int = 8080

# Toolchain & image pins
CUDA_BASE_IMAGE: str = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
RUST_TOOLCHAIN_VERSION: str = "1.88.0"


def expected_shard_filenames(shard_count: int = PINNED_SHARD_COUNT) -> List[str]:
    """Return the exact 48 shard filenames in the pinned checkpoint."""
    return [
        f"model-{idx:05d}-of-{shard_count:05d}.safetensors"
        for idx in range(1, shard_count + 1)
    ]


@dataclass(frozen=True)
class DeploymentConfig:
    app_name: str = APP_NAME
    model_id: str = MODEL_ID
    revision: str = PINNED_REVISION
    csf_model_id: str = CSF_MODEL_ID
    csf_revision: str = CSF_PINNED_REVISION
    csf_schema: str = CSF_SCHEMA
    csf_codec: str = CSF_CODEC
    csf_total_bytes: int = CSF_TOTAL_BYTES
    index_sha256: str = PINNED_INDEX_SHA256
    shard_count: int = PINNED_SHARD_COUNT
    tensor_count: int = PINNED_TENSOR_COUNT
    total_bytes: int = PINNED_TOTAL_BYTES
    gpu_spec: str = GPU_SPEC
    gpu_count: int = GPU_COUNT
    cpu_cores: float = CPU_CORES
    memory_mb: int = MEMORY_MB
    volume_name: str = VOLUME_NAME
    checkpoint_dir: str = CHECKPOINT_DIR
    benchmark_volume_name: str = BENCHMARK_VOLUME_NAME
    benchmark_mount_path: str = BENCHMARK_MOUNT_PATH
    min_containers: int = MIN_CONTAINERS
    max_containers: int = MAX_CONTAINERS
    scaledown_window_secs: int = SCALEDOWN_WINDOW_SECS
    startup_timeout_secs: int = STARTUP_TIMEOUT_SECS
    request_timeout_secs: int = REQUEST_TIMEOUT_SECS
    internal_host: str = INTERNAL_SERVER_HOST
    internal_port: int = INTERNAL_SERVER_PORT
    enable_dspark: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)
