"""Pinned checkpoint, hardware, and runtime configuration for SINGULARITY Modal deployment."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping

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
SMOKE_APP_NAME: str = "singularity-smoke"
KNOWN_APP_NAMES: tuple[str, ...] = (APP_NAME, DEEPINFER_APP_NAME, SMOKE_APP_NAME)
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

# Container scaling and timeouts.
# Cost policy: ZERO demand => ZERO provisioned GPUs. MIN_CONTAINERS must stay 0;
# a pinned warm container is only possible via explicit opt-in (see resolve_scaling_policy).
MIN_CONTAINERS: int = 0
MAX_CONTAINERS: int = 1  # hard cap on concurrently provisioned 8x B200 nodes
SCALEDOWN_WINDOW_SECS: int = 300  # idle seconds before a container is released
SCALEDOWN_WINDOW_MAX_SECS: int = 900  # finite upper bound on warm retention
STARTUP_TIMEOUT_SECS: int = 3600
REQUEST_TIMEOUT_SECS: int = 1800  # per-request execution bound (Modal kills the input)
REQUEST_TIMEOUT_MAX_SECS: int = 3600
MAX_CONCURRENT_REQUESTS: int = 16  # admission limit enforced by the gateway (503 beyond it)
MAX_CONCURRENT_REQUESTS_CAP: int = 256
DRAIN_TIMEOUT_SECS: int = 60  # bounded wait for in-flight work on shutdown
DRAIN_TIMEOUT_MAX_SECS: int = 600
MAX_WORKER_LIFETIME_SECS: int = 6 * 3600  # container stops admitting work after this
MAX_WORKER_LIFETIME_CAP_SECS: int = 24 * 3600
STARTUP_RETRY_MAX_ATTEMPTS: int = 3  # bounded; no infinite retry loops
INTERNAL_SERVER_HOST: str = "127.0.0.1"
INTERNAL_SERVER_PORT: int = 8080

# Toolchain & image pins
CUDA_BASE_IMAGE: str = "nvidia/cuda:13.0.2-devel-ubuntu24.04"
RUST_TOOLCHAIN_VERSION: str = "1.88.0"


@dataclass(frozen=True)
class ScalingPolicy:
    min_containers: int
    max_containers: int
    scaledown_window_secs: int
    request_timeout_secs: int
    startup_timeout_secs: int
    pinned_warm: bool
    max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS
    drain_timeout_secs: int = DRAIN_TIMEOUT_SECS
    max_worker_lifetime_secs: int = MAX_WORKER_LIFETIME_SECS
    gpu_usd_per_gpu_hour: float | None = None  # user-supplied; never assumed
    max_spend_usd: float | None = None  # per-container-lifetime budget guard


POLICY_ENV_VARS: tuple[str, ...] = (
    "SINGULARITY_ALLOW_PINNED_WARM",
    "SINGULARITY_DRAIN_TIMEOUT_SECS",
    "SINGULARITY_GPU_USD_PER_GPU_HOUR",
    "SINGULARITY_MAX_CONCURRENT_REQUESTS",
    "SINGULARITY_MAX_CONTAINERS",
    "SINGULARITY_MAX_SPEND_USD",
    "SINGULARITY_MAX_WORKER_LIFETIME_SECS",
    "SINGULARITY_MIN_CONTAINERS",
    "SINGULARITY_REQUEST_TIMEOUT_SECS",
    "SINGULARITY_SCALEDOWN_WINDOW_SECS",
)


def policy_env_snapshot(env: Mapping[str, str] | None = None) -> Dict[str, str]:
    """Deploy-time values of the scaling-policy env vars (whitelist only; never secrets).

    The container re-imports the app module and re-resolves the policy, so these must be
    baked into the image; otherwise in-container guards (lifetime, spend, concurrency,
    drain) silently fall back to defaults while the decorator values reflect the deploy.
    """
    e = os.environ if env is None else env
    return {k: e[k] for k in POLICY_ENV_VARS if e.get(k, "").strip() != ""}


def resolve_scaling_policy(env: Mapping[str, str] | None = None) -> ScalingPolicy:
    """Resolve and validate cost-control settings from the environment.

    Invariants (violations raise ValueError at import/deploy time, not at runtime):
    - min_containers > 0 is refused unless SINGULARITY_ALLOW_PINNED_WARM=1, because a
      Modal min_containers pin bills 8x B200 continuously with no finite bound.
    - scaledown window is within (0, SCALEDOWN_WINDOW_MAX_SECS].
    - request timeout is within (0, REQUEST_TIMEOUT_MAX_SECS].
    - max_containers is >= 1 and never exceeds the configured hard cap MAX_CONTAINERS
      (can be lowered, not raised, through SINGULARITY_MAX_CONTAINERS).
    """
    e = os.environ if env is None else env

    def _int(name: str, default: int) -> int:
        raw = e.get(name)
        if raw is None or raw.strip() == "":
            return default
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    min_c = _int("SINGULARITY_MIN_CONTAINERS", MIN_CONTAINERS)
    max_c = _int("SINGULARITY_MAX_CONTAINERS", MAX_CONTAINERS)
    window = _int("SINGULARITY_SCALEDOWN_WINDOW_SECS", SCALEDOWN_WINDOW_SECS)
    req_to = _int("SINGULARITY_REQUEST_TIMEOUT_SECS", REQUEST_TIMEOUT_SECS)
    allow_warm = e.get("SINGULARITY_ALLOW_PINNED_WARM", "0") == "1"

    if min_c < 0:
        raise ValueError("SINGULARITY_MIN_CONTAINERS must be >= 0")
    if min_c > 0 and not allow_warm:
        raise ValueError(
            "SINGULARITY_MIN_CONTAINERS>0 pins billable 8x B200 compute indefinitely. "
            "Set SINGULARITY_ALLOW_PINNED_WARM=1 to opt in explicitly, and run "
            "`deploy.py teardown` or redeploy with 0 afterwards."
        )
    if not 1 <= max_c <= MAX_CONTAINERS:
        raise ValueError(f"SINGULARITY_MAX_CONTAINERS must be in [1, {MAX_CONTAINERS}]")
    if min_c > max_c:
        raise ValueError("min_containers cannot exceed max_containers")
    if not 1 <= window <= SCALEDOWN_WINDOW_MAX_SECS:
        raise ValueError(
            f"SINGULARITY_SCALEDOWN_WINDOW_SECS must be in [1, {SCALEDOWN_WINDOW_MAX_SECS}]"
        )
    if not 1 <= req_to <= REQUEST_TIMEOUT_MAX_SECS:
        raise ValueError(
            f"SINGULARITY_REQUEST_TIMEOUT_SECS must be in [1, {REQUEST_TIMEOUT_MAX_SECS}]"
        )
    max_conc = _int("SINGULARITY_MAX_CONCURRENT_REQUESTS", MAX_CONCURRENT_REQUESTS)
    drain_to = _int("SINGULARITY_DRAIN_TIMEOUT_SECS", DRAIN_TIMEOUT_SECS)
    lifetime = _int("SINGULARITY_MAX_WORKER_LIFETIME_SECS", MAX_WORKER_LIFETIME_SECS)
    if not 1 <= max_conc <= MAX_CONCURRENT_REQUESTS_CAP:
        raise ValueError(
            f"SINGULARITY_MAX_CONCURRENT_REQUESTS must be in [1, {MAX_CONCURRENT_REQUESTS_CAP}]"
        )
    if not 1 <= drain_to <= DRAIN_TIMEOUT_MAX_SECS:
        raise ValueError(f"SINGULARITY_DRAIN_TIMEOUT_SECS must be in [1, {DRAIN_TIMEOUT_MAX_SECS}]")
    if not 60 <= lifetime <= MAX_WORKER_LIFETIME_CAP_SECS:
        raise ValueError(
            f"SINGULARITY_MAX_WORKER_LIFETIME_SECS must be in [60, {MAX_WORKER_LIFETIME_CAP_SECS}]"
        )

    def _float_opt(name: str) -> float | None:
        raw = e.get(name)
        if raw is None or raw.strip() == "":
            return None
        try:
            val = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a number, got {raw!r}") from exc
        if val <= 0 or val != val or val == float("inf"):
            raise ValueError(f"{name} must be a positive finite number")
        return val

    rate = _float_opt("SINGULARITY_GPU_USD_PER_GPU_HOUR")
    spend = _float_opt("SINGULARITY_MAX_SPEND_USD")
    if spend is not None and rate is None:
        raise ValueError(
            "SINGULARITY_MAX_SPEND_USD requires SINGULARITY_GPU_USD_PER_GPU_HOUR "
            "(no price is assumed; take it from your Modal pricing page/billing)."
        )
    return ScalingPolicy(
        min_c,
        max_c,
        window,
        req_to,
        STARTUP_TIMEOUT_SECS,
        min_c > 0,
        max_concurrent_requests=max_conc,
        drain_timeout_secs=drain_to,
        max_worker_lifetime_secs=lifetime,
        gpu_usd_per_gpu_hour=rate,
        max_spend_usd=spend,
    )



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
