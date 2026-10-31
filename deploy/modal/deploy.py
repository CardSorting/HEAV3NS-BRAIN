"""Complete Modal deployment & lifecycle management CLI for SINGULARITY
(singularity2.md Section 10).

Supports all 15 operational subcommands:
  build-image, validate-image, provision-storage, download-checkpoint,
  validate-checkpoint, deploy, inspect, logs, health, test-inference,
  benchmark, restart, update, stop, validate-gpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    from deploy.modal.app import STARTUP_SEQUENCE_STEPS, build_singularity_serve_command
    from deploy.modal.benchmark import run_benchmark_suite
    from deploy.modal.config import (
        APP_NAME,
        BENCHMARK_VOLUME_NAME,
        CHECKPOINT_DIR,
        CSF_MODEL_ID,
        CSF_PINNED_REVISION,
        DeploymentConfig,
        GPU_COUNT,
        KNOWN_APP_NAMES,
        MODEL_ID,
        VOLUME_NAME,
    )
    from deploy.modal.deepinfer_app import get_deepinfer_reference_spec
    from deploy.modal.health import (
        check_service_health,
        run_smoke_inference,
        verify_gpu_topology,
    )
    from deploy.modal.image import validate_workspace_sources
    from deploy.modal.model_storage import (
        download_pinned_checkpoint,
        validate_checkpoint_directory,
    )
    from deploy.modal.config import resolve_scaling_policy
    from deploy.modal.ops import ModalProvider, audit, measure_cold_start, status
else:
    from .app import STARTUP_SEQUENCE_STEPS, build_singularity_serve_command
    from .benchmark import run_benchmark_suite
    from .config import (
        APP_NAME,
        BENCHMARK_VOLUME_NAME,
        CHECKPOINT_DIR,
        CSF_MODEL_ID,
        CSF_PINNED_REVISION,
        DeploymentConfig,
        GPU_COUNT,
        KNOWN_APP_NAMES,
        MODEL_ID,
        VOLUME_NAME,
    )
    from .deepinfer_app import get_deepinfer_reference_spec
    from .health import check_service_health, run_smoke_inference, verify_gpu_topology
    from .image import validate_workspace_sources
    from .model_storage import download_pinned_checkpoint, validate_checkpoint_directory
    from .config import resolve_scaling_policy
    from .ops import ModalProvider, audit, measure_cold_start, status

ALL_LIFECYCLE_COMMANDS: List[str] = [
    "build-image",
    "validate-image",
    "provision-storage",
    "download-checkpoint",
    "validate-checkpoint",
    "deploy",
    "inspect",
    "logs",
    "health",
    "test-inference",
    "benchmark",
    "restart",
    "update",
    "stop",
    "validate-gpu",
]


# Scale-to-zero operations added by singularity3.md (kept separate: the 15-command set above
# is the singularity2.md contract and is asserted by tests).
SCALE_TO_ZERO_COMMANDS: List[str] = ["status", "audit", "measure-cold-start", "teardown"]


def _resolve_modal_bin() -> str:
    import shutil

    found = shutil.which("modal")
    if found:
        return found
    pyenv_modal = Path.home() / ".pyenv" / "shims" / "modal"
    if pyenv_modal.is_file():
        return str(pyenv_modal)
    return "modal"


def _run_modal_cli(args: List[str], *, dry_run: bool = False) -> Dict[str, object]:
    canonical_cmd = ["modal"] + args
    if dry_run:
        return {"status": "dry_run", "command": canonical_cmd}
    import os

    exec_cmd = [_resolve_modal_bin()] + args
    repo_root = _resolve_repo_root(Path(__file__))
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        exec_cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command {exec_cmd} failed (exit {proc.returncode}): "
            f"stderr={proc.stderr.strip()!r} stdout={proc.stdout.strip()[-500:]!r}"
        )
    return {
        "status": "ok",
        "command": canonical_cmd,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m deploy.modal.deploy",
        description="SINGULARITY DeepSeek-V4.1-Flash-1017 Modal Infrastructure Lifecycle CLI",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments and emit planned operations without mutating cloud state",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-image", help="Build and cache the SINGULARITY CUDA 13.0.2 + Rust 1.88.0 image")
    sub.add_parser("validate-image", help="Validate workspace sources, Dockerfile, and image toolchain spec")
    sub.add_parser("provision-storage", help="Provision persistent Modal Volumes for weights and benchmarks")

    dl = sub.add_parser("download-checkpoint", help="Download pinned DeepSeek-V4.1-Flash-1017-lossless-CSF into Modal Volume")
    dl.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    dl.add_argument("--repo-id", default=CSF_MODEL_ID)
    dl.add_argument("--revision", default=CSF_PINNED_REVISION)
    dl.add_argument("--hf-token", default=None)
    dl.add_argument("--local", action="store_true", help="Download into local path instead of remote Modal Volume")

    vc = sub.add_parser("validate-checkpoint", help="Validate pinned 48-shard Safetensors manifest & headers")
    vc.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    vc.add_argument("--non-strict", action="store_true", help="Allow synthetic/test checkpoints")

    dep = sub.add_parser("deploy", help="Deploy the persistent 8x B200 SINGULARITY service on Modal")
    dep.add_argument("--with-deepinfer", action="store_true", help="Also deploy isolated DeepInfer reference app")

    sub.add_parser("inspect", help="Inspect deployment configuration, startup sequence, and reference spec")

    lg = sub.add_parser("logs", help="Fetch container logs from the deployed Modal service")
    lg.add_argument("--app-name", default=APP_NAME)

    hl = sub.add_parser("health", help="Verify /health, /ready (all 9 checks), and /v1/models on endpoint")
    hl.add_argument("--url", default="http://127.0.0.1:8080")
    hl.add_argument("--api-key", default=None)

    ti = sub.add_parser("test-inference", help="Run smoke inference (chat, thinking, tool calls, SSE) on endpoint")
    ti.add_argument("--url", default="http://127.0.0.1:8080")
    ti.add_argument("--model", default=MODEL_ID)
    ti.add_argument("--api-key", default=None)

    bm = sub.add_parser("benchmark", help="Run controlled benchmark suite across configurations A/B/C/D")
    bm.add_argument("--url", default="http://127.0.0.1:8080")
    bm.add_argument("--deepinfer-url", default=None, help="Optional DeepInfer reference endpoint URL for head-to-head comparison")
    bm.add_argument("--model", default=MODEL_ID)
    bm.add_argument("--api-key", default=None)
    bm.add_argument("--output", default=None)

    sub.add_parser("restart", help="Restart the deployed Modal service container")
    sub.add_parser("update", help="Rebuild and roll out an updated SINGULARITY service deployment")
    sp = sub.add_parser("stop", help="Stop the deployed Modal service")
    sp.add_argument("--app-name", default=APP_NAME, help="Modal app name to stop (default: primary service)")
    sp.add_argument("--all", action="store_true", help="Stop all known SINGULARITY Modal apps")
    sub.add_parser("teardown", help="Stop all known SINGULARITY Modal apps and terminate any orphaned containers")
    sub.add_parser("validate-gpu", help="Run SM 10.0 nvcc + PTX validation on B200 via gpu_validation.py")

    st = sub.add_parser("status", help="Policy + provider-side zero-compute check (+ optional live container lifecycle)")
    st.add_argument("--url", default=None, help="Deployed endpoint; querying it counts as activity")

    au = sub.add_parser("audit", help="List provider containers, detect orphans; --terminate-orphans stops them")
    au.add_argument("--terminate-orphans", action="store_true", help="Actually stop orphaned containers (default: report only)")

    cs = sub.add_parser("measure-cold-start", help="Measure first-request (cold) vs warm TTFT on an endpoint")
    cs.add_argument("--url", required=True)
    cs.add_argument("--model", default=MODEL_ID)
    cs.add_argument("--api-key", default=None)
    cs.add_argument("--warm-requests", type=int, default=3)
    cs.add_argument("--timeout", type=float, default=300.0, help="Per-request client timeout in seconds")
    cs.add_argument("--gpu-count", type=int, default=GPU_COUNT, help="GPU count for report attribution")
    cs.add_argument("--output", default=None)

    return parser


def execute_command(
    argv: Optional[List[str]] = None, *, provider: Optional[ModalProvider] = None
) -> Dict[str, object]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    dry_run: bool = bool(args.dry_run)
    cmd: str = args.command

    if cmd in ("status", "audit"):
        policy = resolve_scaling_policy()
        prov = provider or ModalProvider()
        if cmd == "audit":
            if args.terminate_orphans and dry_run:
                return {"status": "dry_run", "note": "would terminate orphans after listing"}
            return audit(prov, policy, terminate_orphans=bool(args.terminate_orphans))
        return status(prov, policy, url=args.url)

    if cmd == "measure-cold-start":
        if dry_run:
            return {"status": "dry_run", "url": args.url}
        report = measure_cold_start(
            args.url,
            model=args.model,
            api_key=args.api_key,
            warm_requests=args.warm_requests,
            timeout=args.timeout,
            gpu_count=args.gpu_count,
        )
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2))
        return report

    if cmd == "validate-image":
        return validate_workspace_sources()

    if cmd == "build-image":
        src_check = validate_workspace_sources()
        if not src_check["valid"]:
            raise RuntimeError(f"Workspace source validation failed: {src_check['missing']}")
        return _run_modal_cli(["deploy", "deploy/modal/app.py"], dry_run=dry_run)

    if cmd == "provision-storage":
        vol1 = _run_modal_cli(["volume", "create", VOLUME_NAME], dry_run=dry_run)
        vol2 = _run_modal_cli(["volume", "create", BENCHMARK_VOLUME_NAME], dry_run=dry_run)
        return {"status": "provisioned" if not dry_run else "dry_run", "volumes": [vol1, vol2]}

    if cmd == "download-checkpoint":
        if args.local or dry_run:
            return download_pinned_checkpoint(
                args.checkpoint_dir,
                repo_id=args.repo_id,
                revision=args.revision,
                hf_token=args.hf_token,
                dry_run=dry_run,
            )
        modal_args = [
            "run",
            "deploy/modal/app.py::download_checkpoint_remote",
            "--repo-id",
            args.repo_id,
            "--revision",
            args.revision,
        ]
        if args.hf_token:
            modal_args.extend(["--hf-token", args.hf_token])
        return _run_modal_cli(modal_args, dry_run=dry_run)

    if cmd == "validate-checkpoint":
        if dry_run:
            return {
                "status": "dry_run",
                "checkpoint_dir": args.checkpoint_dir,
                "strict_pinned": not args.non_strict,
            }
        vc_result = validate_checkpoint_directory(
            args.checkpoint_dir,
            strict_pinned=not args.non_strict,
            verify_shard_headers=True,
        )
        if isinstance(vc_result, dict) and any(
            "does not exist" in str(e) for e in vc_result.get("errors", [])
        ):
            vc_result["hint"] = (
                "This command validates a LOCAL path only. The Modal Volume is not mounted "
                "here; use `modal volume ls <volume> <dir>` to inspect remote contents."
            )
        return vc_result

    if cmd == "deploy":
        primary = _run_modal_cli(["deploy", "deploy/modal/app.py"], dry_run=dry_run)
        result: Dict[str, object] = {"singularity": primary}
        if args.with_deepinfer:
            result["deepinfer"] = _run_modal_cli(
                ["deploy", "deploy/modal/deepinfer_app.py"], dry_run=dry_run
            )
        return result

    if cmd == "inspect":
        cfg = DeploymentConfig()
        return {
            "deployment": cfg.to_dict(),
            "serve_command": build_singularity_serve_command(),
            "startup_sequence": STARTUP_SEQUENCE_STEPS,
            "deepinfer_reference": get_deepinfer_reference_spec(),
        }

    if cmd == "logs":
        return _run_modal_cli(["app", "logs", args.app_name], dry_run=dry_run)

    if cmd == "health":
        if dry_run:
            return {"status": "dry_run", "url": args.url, "gpu_check": verify_gpu_topology()}
        return check_service_health(args.url, api_key=args.api_key)

    if cmd == "test-inference":
        if dry_run:
            return {"status": "dry_run", "url": args.url, "model": args.model}
        return run_smoke_inference(args.url, model=args.model, api_key=args.api_key)

    if cmd == "benchmark":
        if dry_run:
            return {
                "status": "dry_run",
                "url": args.url,
                "deepinfer_url": args.deepinfer_url,
                "model": args.model,
                "output": args.output,
            }
        return run_benchmark_suite(
            args.url,
            deepinfer_url=args.deepinfer_url,
            model=args.model,
            api_key=args.api_key,
            output_path=args.output,
        )

    if cmd == "restart":
        stop_res = _run_modal_cli(["app", "stop", "--yes", APP_NAME], dry_run=dry_run)
        deploy_res = _run_modal_cli(["deploy", "deploy/modal/app.py"], dry_run=dry_run)
        return {"status": "restarted" if not dry_run else "dry_run", "stop": stop_res, "deploy": deploy_res}

    if cmd == "update":
        return _run_modal_cli(["deploy", "deploy/modal/app.py"], dry_run=dry_run)

    if cmd == "stop":
        if getattr(args, "all", False):
            stopped = []
            for name in KNOWN_APP_NAMES:
                try:
                    stopped.append(_run_modal_cli(["app", "stop", "--yes", name], dry_run=dry_run))
                except RuntimeError as exc:
                    stopped.append({"app": name, "status": "skipped_or_not_deployed", "detail": str(exc)})
            return {"status": "stopped_all" if not dry_run else "dry_run", "apps": stopped}
        target_app = getattr(args, "app_name", APP_NAME) or APP_NAME
        return _run_modal_cli(["app", "stop", "--yes", target_app], dry_run=dry_run)

    if cmd == "teardown":
        if dry_run:
            return {
                "status": "dry_run",
                "apps": list(KNOWN_APP_NAMES),
                "terminate_orphans": True,
            }
        stopped_apps = []
        for name in KNOWN_APP_NAMES:
            try:
                stopped_apps.append(_run_modal_cli(["app", "stop", "--yes", name], dry_run=False))
            except Exception as exc:  # noqa: BLE001
                stopped_apps.append({"app": name, "status": "not_running", "detail": str(exc)})
        policy = resolve_scaling_policy()
        prov = provider or ModalProvider()
        audit_rep = audit(prov, policy, terminate_orphans=True)
        return {
            "status": "torn_down",
            "stopped_apps": stopped_apps,
            "audit": audit_rep,
        }

    if cmd == "validate-gpu":
        return _run_modal_cli(["run", "deploy/modal/gpu_validation.py"], dry_run=dry_run)

    raise ValueError(f"Unsupported command: {cmd}")


def main(argv: Optional[List[str]] = None) -> int:
    try:
        output = execute_command(argv)
        print(json.dumps(output, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
