"""Modal container image definitions for SINGULARITY and DeepInfer reference on 8x B200."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

def _resolve_repo_root(file_path: Path) -> Path:
    resolved = file_path.resolve()
    for parent in resolved.parents:
        if (parent / "deploy" / "modal").is_dir() or (parent / "Cargo.toml").is_file():
            return parent
    candidates = [
        Path("/workspace/singularity"),
        Path("/opt/singularity"),
        Path.cwd(),
        *(Path(p) for p in sys.path if p),
    ]
    for candidate in candidates:
        if (candidate / "deploy" / "modal").is_dir() or (candidate / "Cargo.toml").is_file():
            return candidate
    return resolved.parent


if __package__ in (None, ""):
    _modal_dir = str(Path(__file__).resolve().parent)
    _repo_root = str(_resolve_repo_root(Path(__file__)))
    if _modal_dir in sys.path and (Path(_modal_dir) / "deploy.py").is_file():
        sys.path.remove(_modal_dir)
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    from deploy.modal.config import CUDA_BASE_IMAGE, RUST_TOOLCHAIN_VERSION
else:
    from .config import CUDA_BASE_IMAGE, RUST_TOOLCHAIN_VERSION

REPO_ROOT = _resolve_repo_root(Path(__file__))

RUST_WORKSPACE_ITEMS: List[str] = [
    "Cargo.toml",
    "Cargo.lock",
    "singularity-core",
    "singularity-cuda",
    "singularity-cache",
    "singularity-grammar",
    "singularity-model",
    "singularity-scheduler",
    "singularity-server",
    "singularity-cli",
    "singularity-tests",
]

WORKSPACE_CRATES: List[str] = RUST_WORKSPACE_ITEMS + ["deploy"]

COPY_IGNORE_PATTERNS: List[str] = [
    "target/",
    "__pycache__/",
    "*.pyc",
    ".DS_Store",
    ".env",
    ".env.*",
]


def get_image_build_spec() -> Dict[str, object]:
    """Return the deterministic build specification for inspection and validation."""
    return {
        "base_image": CUDA_BASE_IMAGE,
        "rust_toolchain": RUST_TOOLCHAIN_VERSION,
        "cuda_arch": "sm_100a",
        "ccbin": "g++-13",
        "cargo_command": "cargo build --release --features cuda --bin singularity",
        "binary_path": "/usr/local/bin/singularity",
        "workspace_paths": WORKSPACE_CRATES,
    }


def validate_workspace_sources(repo_root: Path = REPO_ROOT) -> Dict[str, object]:
    """Verify all required workspace manifests and crates exist before image build."""
    missing: List[str] = []
    for rel in WORKSPACE_CRATES:
        if not (repo_root / rel).exists():
            missing.append(rel)
    dockerfile = repo_root / "deploy" / "modal" / "Dockerfile"
    if not dockerfile.is_file():
        missing.append("deploy/modal/Dockerfile")
    return {
        "valid": len(missing) == 0,
        "repo_root": str(repo_root),
        "missing": missing,
        "spec": get_image_build_spec(),
    }


def build_singularity_image():
    """Construct the Modal Image for the SINGULARITY Rust + CUDA 13.0.2 runtime."""
    import modal

    image = (
        modal.Image.from_registry(CUDA_BASE_IMAGE, add_python="3.12")
        .apt_install(
            "build-essential",
            "ca-certificates",
            "curl",
            "g++-13",
            "gcc-13",
            "git",
            "libssl-dev",
            "pkg-config",
        )
        .pip_install(
            "fastapi[standard]>=0.115.0",
            "httpx>=0.27.0",
            "huggingface_hub[hf_transfer]>=0.26.0",
            "safetensors>=0.4.5",
        )
        .env(
            {
                "PATH": "/root/.cargo/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
                "CUDA_HOME": "/usr/local/cuda",
                "NVCC_CCBIN": "g++-13",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONPATH": "/workspace/singularity",
            }
        )
        .run_commands(
            f"curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain {RUST_TOOLCHAIN_VERSION} --profile minimal",
            "rustc --version && cargo --version && nvcc --version",
        )
    )

    for item in RUST_WORKSPACE_ITEMS:
        src = REPO_ROOT / item
        dst = f"/workspace/singularity/{item}"
        if src.is_dir():
            image = image.add_local_dir(
                str(src),
                remote_path=dst,
                copy=True,
                ignore=COPY_IGNORE_PATTERNS,
            )
        elif src.is_file():
            image = image.add_local_file(str(src), remote_path=dst, copy=True)

    image = image.run_commands(
        "cd /workspace/singularity && cargo build --release --features cuda --bin singularity",
        "cp /workspace/singularity/target/release/singularity /usr/local/bin/singularity",
        "/usr/local/bin/singularity --help",
    )

    deploy_src = REPO_ROOT / "deploy"
    if deploy_src.is_dir():
        image = image.add_local_dir(
            str(deploy_src),
            remote_path="/workspace/singularity/deploy",
            copy=True,
            ignore=COPY_IGNORE_PATTERNS,
        )
    return image


def build_deepinfer_image():
    """Construct the isolated Modal Image for the DeepInfer reference engine on 8x B200."""
    import modal

    image = (
        modal.Image.from_registry(CUDA_BASE_IMAGE, add_python="3.12")
        .apt_install(
            "build-essential",
            "ca-certificates",
            "curl",
            "git",
        )
        .pip_install(
            "torch>=2.6.0",
            "triton>=3.2.0",
            "transformers>=4.48.0",
            "safetensors>=0.4.5",
            "accelerate>=1.2.0",
            "fastapi[standard]>=0.115.0",
            "httpx>=0.27.0",
            "sentencepiece>=0.2.0",
        )
        .env(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "NCCL_DEBUG": "WARN",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "PYTHONPATH": "/workspace/singularity:/workspace/DeepInfer:/workspace/DeepInfer/examples",
            }
        )
        .run_commands(
            "git clone --depth 1 https://github.com/deepseek-ai/DeepInfer.git /workspace/DeepInfer || mkdir -p /workspace/DeepInfer",
            "if [ -f /workspace/DeepInfer/pyproject.toml ] || [ -f /workspace/DeepInfer/setup.py ]; then pip install --no-deps -e /workspace/DeepInfer; fi",
        )
    )
    deploy_src = REPO_ROOT / "deploy"
    if deploy_src.is_dir():
        image = image.add_local_dir(
            str(deploy_src),
            remote_path="/workspace/singularity/deploy",
            copy=True,
            ignore=COPY_IGNORE_PATTERNS,
        )
    return image

