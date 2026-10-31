"""Bounded B200 validation for SINGULARITY's CUDA primitives, not inference serving.

Run: modal run deploy/modal/gpu_validation.py
Optional real checkpoint expert: modal run deploy/modal/gpu_validation.py --real-weights
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

def _resolve_repo_root(file_path: Path) -> Path:
    resolved = file_path.resolve()
    for parent in resolved.parents:
        if (parent / "deploy" / "modal").is_dir() or (parent / "Cargo.toml").is_file():
            return parent
    candidates = [
        Path("/opt/singularity"),
        Path("/workspace/singularity"),
        Path.cwd(),
        *(Path(p) for p in sys.path if p),
    ]
    for candidate in candidates:
        if (candidate / "Cargo.toml").is_file() or (candidate / "deploy" / "modal").is_dir():
            return candidate
    return resolved.parent


ROOT = _resolve_repo_root(Path(__file__))
CUDA_IMAGE = "nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c"
RUSTUP_SHA256 = "20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c"
CRATES = ("singularity-core", "singularity-cuda", "singularity-cache", "singularity-scheduler", "singularity-model", "singularity-grammar", "singularity-server", "singularity-cli", "singularity-tests")

try:
    import modal

    # Build test binaries in a CPU-only image layer: B200 time is spent on tests,
    # not dependency compilation. No checkpoint or local credentials are mounted.
    image = (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.13")
        .run_commands(
            "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential curl ca-certificates pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*",
            "curl -fsSL https://static.rust-lang.org/rustup/archive/1.28.2/x86_64-unknown-linux-gnu/rustup-init -o /tmp/rustup-init && "
            f"echo '{RUSTUP_SHA256}  /tmp/rustup-init' | sha256sum -c - && "
            "chmod +x /tmp/rustup-init && /tmp/rustup-init -y --no-modify-path --profile minimal --default-toolchain 1.88.0 && rm /tmp/rustup-init",
        )
        .env({"PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/local/nvidia/lib64"})
    )
    if (ROOT / "Cargo.toml").is_file() and (ROOT / "Cargo.lock").is_file():
        image = (
            image.add_local_file(ROOT / "Cargo.toml", "/opt/singularity/Cargo.toml", copy=True)
            .add_local_file(ROOT / "Cargo.lock", "/opt/singularity/Cargo.lock", copy=True)
        )
    for crate in CRATES:
        if (ROOT / crate).is_dir():
            image = image.add_local_dir(ROOT / crate, f"/opt/singularity/{crate}", copy=True, ignore=["target/", "__pycache__/", "*.pyc", ".env", ".env.*"])
    if (ROOT / "scripts/fetch_v41_reference_ranges.py").is_file():
        image = image.add_local_file(ROOT / "scripts/fetch_v41_reference_ranges.py", "/opt/singularity/scripts/fetch_v41_reference_ranges.py", copy=True)
    image = image.run_commands(
        "cd /opt/singularity && cargo test --locked -p singularity-cuda --features cuda --no-run",
        "cd /opt/singularity && cargo test --locked -p singularity-tests --features cuda --test v41_cuda_quant --no-run",
    )
    app = modal.App("singularity-b200-validation")
except ImportError:
    modal = None
    image = None
    app = None


def _identity_decorator(*_args, **_kwargs):
    def _wrap(fn):
        return fn
    return _wrap


_function_decorator = app.function(image=image, gpu="B200:8", cpu=4, memory=16384, timeout=300, startup_timeout=900, min_containers=0, max_containers=1, single_use_containers=True) if app is not None else _identity_decorator()
_entrypoint_decorator = app.local_entrypoint() if app is not None else _identity_decorator()


@_function_decorator
def validate(real_weights: bool = False) -> dict:
    """Run only bounded device checks; no persistent replicas or model server."""
    import ctypes
    import datetime
    import os
    import platform
    import subprocess
    import time

    start = time.monotonic()
    driver = ctypes.CDLL("libcuda.so.1")
    def check(code, operation):
        if code != 0:
            raise RuntimeError(f"{operation} failed with CUDA status {code}")
    check(driver.cuInit(0), "cuInit")
    count = ctypes.c_int()
    check(driver.cuDeviceGetCount(ctypes.byref(count)), "cuDeviceGetCount")
    if count.value != 8:
        raise RuntimeError(f"Expected eight visible GPUs; got {count.value}")
    devices = []
    for ordinal in range(count.value):
        device = ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        check(driver.cuDeviceGet(ctypes.byref(device), ordinal), "cuDeviceGet")
        check(driver.cuDeviceGetName(name, len(name), device), "cuDeviceGetName")
        check(driver.cuDeviceComputeCapability(ctypes.byref(major), ctypes.byref(minor), device), "cuDeviceComputeCapability")
        label=name.value.decode("utf-8")
        if "B200" not in label.upper() or (major.value,minor.value)!=(10,0):
            raise RuntimeError(f"Unexpected device {ordinal}: {label}, SM {major.value}.{minor.value}")
        devices.append({"ordinal": ordinal, "name": label, "compute_capability": f"{major.value}.{minor.value}"})

    root="/opt/singularity"
    env=os.environ.copy()
    commands=[
        ["cargo","test","--locked","-p","singularity-cuda","--features","cuda","physical_cuda_transfers_gemm_and_events","--","--ignored","--nocapture"],
        ["cargo","test","--locked","-p","singularity-cuda","--features","cuda","physical_v41_quantized_matvec","--","--ignored","--nocapture"],
        ["cargo","test","--locked","-p","singularity-tests","--features","cuda","--test","v41_cuda_quant","cuda_v41_matvec_matches_cpu_reference","--","--ignored","--nocapture"],
    ]
    if real_weights:
        subprocess.run(["python3",f"{root}/scripts/fetch_v41_reference_ranges.py","/tmp/singularity-reference"],cwd=root,env=env,check=True,timeout=120)
        env.update({"SINGULARITY_V41_EXPERT_WEIGHT":"/tmp/singularity-reference/expert309-weights.bin", "SINGULARITY_V41_EXPERT_SCALE":"/tmp/singularity-reference/expert309-scales.bin"})
        commands.append(["cargo","test","--locked","-p","singularity-tests","--features","cuda","--test","v41_cuda_quant","cuda_real_expert_w1_matches_cpu_reference","--","--ignored","--nocapture"])
    results=[]
    for command in commands:
        completed=subprocess.run(command,cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        results.append({"command":command,"exit_code":completed.returncode,"output":completed.stdout})
        print(completed.stdout,flush=True)
        if completed.returncode != 0:
            raise RuntimeError(f"Validation failed: {command!r} (exit {completed.returncode})")
    return {"status":"passed", "scope":"device primitives on GPU 0; eight visible B200s; no model inference", "timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(), "devices":devices, "platform":platform.platform(), "real_weights":real_weights, "tests":results, "elapsed_seconds":round(time.monotonic()-start,3)}


@_entrypoint_decorator
def main(real_weights: bool = False):
    import datetime
    import uuid
    result=validate.remote(real_weights)
    directory=ROOT / "artifacts" / "b200-validation"
    directory.mkdir(parents=True,exist_ok=True)
    name=datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8] + ".json"
    destination=directory / name
    destination.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(f"Saved bounded validation results to {destination}")
