"""Persistent 8x NVIDIA B200 Modal serving application for SINGULARITY DeepSeek-V4.1-Flash-1017."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
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
    from deploy.modal.config import (
        APP_NAME,
        BENCHMARK_MOUNT_PATH,
        BENCHMARK_VOLUME_NAME,
        CHECKPOINT_DIR,
        CPU_CORES,
        CSF_MODEL_ID,
        CSF_PINNED_REVISION,
        GPU_COUNT,
        GPU_SPEC,
        HF_SECRET_NAME,
        INTERNAL_SERVER_HOST,
        INTERNAL_SERVER_PORT,
        MAX_CONTAINERS,
        MEMORY_MB,
        MIN_CONTAINERS,
        REQUEST_TIMEOUT_SECS,
        SCALEDOWN_WINDOW_SECS,
        STARTUP_TIMEOUT_SECS,
        VOLUME_MOUNT_PATH,
        VOLUME_NAME,
    )
    from deploy.modal.health import check_service_health, verify_gpu_topology
    from deploy.modal.image import build_singularity_image
    from deploy.modal.model_storage import (
        download_pinned_checkpoint,
        prepare_csf_checkpoint_directory,
        validate_checkpoint_directory,
    )
else:
    from .config import (
        APP_NAME,
        BENCHMARK_MOUNT_PATH,
        BENCHMARK_VOLUME_NAME,
        CHECKPOINT_DIR,
        CPU_CORES,
        CSF_MODEL_ID,
        CSF_PINNED_REVISION,
        GPU_COUNT,
        GPU_SPEC,
        HF_SECRET_NAME,
        INTERNAL_SERVER_HOST,
        INTERNAL_SERVER_PORT,
        MAX_CONTAINERS,
        MEMORY_MB,
        MIN_CONTAINERS,
        REQUEST_TIMEOUT_SECS,
        SCALEDOWN_WINDOW_SECS,
        STARTUP_TIMEOUT_SECS,
        VOLUME_MOUNT_PATH,
        VOLUME_NAME,
    )
    from .health import check_service_health, verify_gpu_topology
    from .image import build_singularity_image
    from .model_storage import (
        download_pinned_checkpoint,
        prepare_csf_checkpoint_directory,
        validate_checkpoint_directory,
    )

STARTUP_SEQUENCE_STEPS: List[str] = [
    "1. Container booted & runtime environment initialized",
    "2. Verify 8x NVIDIA B200 GPUs and SM 10.0 compute capability",
    "3. Verify NVLink topology & NCCL environment variables",
    "4. Verify persistent Modal Volume mounted at /models",
    "5. Verify checkpoint directory /models/deepseek-v41-flash-1017",
    "6. Validate config.json & tokenizer.json",
    "7. Validate model.safetensors.index.json SHA-256 & 96,085 tensor manifest",
    "8. Validate 48 Safetensors shard headers & byte offsets",
    "9. Run `singularity plan-cluster` to verify 8-rank B200 HBM3e budget",
    "10. Run `singularity verify-checkpoint` against loaded Safetensors index",
    "11. Configure optional Bearer API key authentication",
    "12. Launch supervised `singularity serve` child process on 127.0.0.1:8080",
    "13. Load V4.1-Flash-1017 weights, sharded EP=8 experts, and Engram tables",
    "14. Initialize KV cache, compressed KV sources, and FP4 indexers",
    "15. Initialize DSpark 3-layer MTP draft engine",
    "16. Execute warm-up forward pass and verify finite logits",
    "17. Confirm HTTP 200 on GET /ready with all 9 readiness flags true",
]

HOP_BY_HOP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
    }
)


def build_singularity_serve_command(
    checkpoint_dir: str = CHECKPOINT_DIR,
    host: str = INTERNAL_SERVER_HOST,
    port: int = INTERNAL_SERVER_PORT,
    *,
    use_cuda: bool = True,
    dspark: bool = True,
    api_key: Optional[str] = None,
    binary_path: str = "/usr/local/bin/singularity",
) -> List[str]:
    """Build the exact command line used to start the supervised SINGULARITY server."""
    root = Path(checkpoint_dir)
    cmd = [
        binary_path,
        "serve",
        "--host",
        host,
        "--port",
        str(port),
        "--v41-config",
        str(root / "config.json"),
        "--model-path",
        str(root / "model.safetensors.index.json"),
        "--tokenizer-path",
        str(root / "tokenizer.json"),
    ]
    if use_cuda:
        cmd.append("--use-cuda")
    if dspark:
        cmd.append("--dspark")
    if api_key:
        cmd.extend(["--api-key", api_key])
    return cmd


class SingularityRuntimeSupervisor:
    """Manages the 17-step startup lifecycle and supervises the `singularity serve` process."""

    def __init__(
        self,
        checkpoint_dir: str = CHECKPOINT_DIR,
        host: str = INTERNAL_SERVER_HOST,
        port: int = INTERNAL_SERVER_PORT,
        binary_path: str = "/usr/local/bin/singularity",
        enable_dspark: bool = True,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.host = host
        self.port = port
        self.binary_path = binary_path
        self.enable_dspark = enable_dspark
        self.process: Optional[subprocess.Popen] = None
        self.startup_log: List[Dict[str, object]] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _record_step(self, step: int, description: str, detail: object = None) -> None:
        print(f"[SINGULARITY startup step {step}/17] {description}", flush=True)
        self.startup_log.append(
            {
                "step": step,
                "description": description,
                "timestamp": time.time(),
                "detail": detail,
            }
        )

    def start(
        self,
        *,
        require_gpus: bool = True,
        ready_timeout_secs: int = 900,
        skip_checkpoint_revalidation: bool = False,
    ) -> Dict[str, object]:
        """Execute the 17-step startup sequence and block until `/ready` returns 200 OK."""
        self._record_step(1, STARTUP_SEQUENCE_STEPS[0])

        if require_gpus:
            gpu_info = verify_gpu_topology(expected_gpus=GPU_COUNT)
            if not gpu_info["valid"]:
                raise RuntimeError(f"B200 GPU topology verification failed: {gpu_info['errors']}")
            self._record_step(2, STARTUP_SEQUENCE_STEPS[1], gpu_info)
        else:
            self._record_step(2, STARTUP_SEQUENCE_STEPS[1], {"skipped": True})

        os.environ.setdefault("NCCL_DEBUG", "WARN")
        os.environ.setdefault("NCCL_IB_DISABLE", "0")
        self._record_step(3, STARTUP_SEQUENCE_STEPS[2], {"NCCL_DEBUG": os.environ["NCCL_DEBUG"]})

        if not skip_checkpoint_revalidation:
            if Path(self.checkpoint_dir).is_dir():
                prepare_csf_checkpoint_directory(self.checkpoint_dir)

            ckpt_validation = validate_checkpoint_directory(
                self.checkpoint_dir,
                strict_pinned=True,
                verify_shard_headers=True,
            )
            if not ckpt_validation["valid"]:
                raise RuntimeError(f"Checkpoint validation failed: {ckpt_validation['errors']}")
        for step_no in (4, 5, 6, 7, 8):
            self._record_step(step_no, STARTUP_SEQUENCE_STEPS[step_no - 1])

        try:
            plan_proc = subprocess.run(
                [self.binary_path, "plan-cluster", "--world-size", str(GPU_COUNT)],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"`singularity plan-cluster` failed (exit {exc.returncode}): "
                f"stdout={exc.stdout!r} stderr={exc.stderr!r}"
            ) from exc
        self._record_step(9, STARTUP_SEQUENCE_STEPS[8], json.loads(plan_proc.stdout))

        if skip_checkpoint_revalidation:
            self._record_step(
                10,
                STARTUP_SEQUENCE_STEPS[9],
                {"verified_in_serve_startup": True},
            )
        else:
            try:
                verify_proc = subprocess.run(
                    [
                        self.binary_path,
                        "verify-checkpoint",
                        "--config-path",
                        str(Path(self.checkpoint_dir) / "config.json"),
                        "--model-path",
                        str(Path(self.checkpoint_dir) / "model.safetensors.index.json"),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"`singularity verify-checkpoint` failed (exit {exc.returncode}): "
                    f"stdout={exc.stdout!r} stderr={exc.stderr!r}"
                ) from exc
            self._record_step(10, STARTUP_SEQUENCE_STEPS[9], json.loads(verify_proc.stdout))

        api_key = os.environ.get("SINGULARITY_API_KEY")
        self._record_step(11, STARTUP_SEQUENCE_STEPS[10], {"auth_enabled": bool(api_key)})

        cmd = build_singularity_serve_command(
            checkpoint_dir=self.checkpoint_dir,
            host=self.host,
            port=self.port,
            use_cuda=require_gpus,
            dspark=self.enable_dspark,
            api_key=api_key,
            binary_path=self.binary_path,
        )
        self.process = subprocess.Popen(cmd)
        for step_no in (12, 13, 14, 15, 16):
            self._record_step(step_no, STARTUP_SEQUENCE_STEPS[step_no - 1])

        poll_start = time.time()
        last_log_time = poll_start
        deadline = poll_start + ready_timeout_secs
        last_health: Dict[str, object] = {}
        try:
            while time.time() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(
                        f"`singularity serve` exited prematurely with code {self.process.returncode}"
                    )
                last_health = check_service_health(self.base_url, api_key=api_key)
                if last_health.get("ready") is True:
                    self._record_step(17, STARTUP_SEQUENCE_STEPS[16], last_health)
                    return {
                        "status": "ready",
                        "pid": self.process.pid,
                        "base_url": self.base_url,
                        "health": last_health,
                        "startup_steps": len(self.startup_log),
                    }
                now = time.time()
                if now - last_log_time >= 10.0:
                    print(
                        f"[SINGULARITY startup] Waiting for /ready ({int(now - poll_start)}s elapsed): "
                        f"healthy={last_health.get('healthy')} errors={last_health.get('errors')}",
                        flush=True,
                    )
                    last_log_time = now
                time.sleep(1.0)
        except Exception:
            self.stop()
            raise

        self.stop()
        raise TimeoutError(
            f"Timed out waiting {ready_timeout_secs}s for SINGULARITY /ready: {last_health}"
        )

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()


def create_asgi_proxy_app(upstream_url: str, timeout_secs: float = REQUEST_TIMEOUT_SECS):
    """Create the FastAPI reverse proxy that forwards HTTP and SSE traffic to the local
    `singularity serve` child process without buffering `text/event-stream` responses.
    Uses a loop-scoped connection-pooled `httpx.AsyncClient` to eliminate per-request
    client/pool allocation and loopback TCP handshake churn under concurrent load.
    """
    import asyncio
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse
    import httpx

    # Ensure `from __future__ import annotations` resolves `Request` in module globals.
    globals()["Request"] = Request
    globals()["Response"] = Response

    upstream = upstream_url.rstrip("/")
    _client_holder: Dict[str, object] = {"client": None, "loop": None}

    def _get_shared_client() -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        existing = _client_holder.get("client")
        if (
            isinstance(existing, httpx.AsyncClient)
            and not existing.is_closed
            and _client_holder.get("loop") is loop
        ):
            return existing
        client = httpx.AsyncClient(
            timeout=timeout_secs,
            limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        )
        _client_holder["client"] = client
        _client_holder["loop"] = loop
        return client

    async def _close_shared_client() -> None:
        existing = _client_holder.get("client")
        _client_holder["client"] = None
        _client_holder["loop"] = None
        if isinstance(existing, httpx.AsyncClient) and not existing.is_closed:
            await existing.aclose()

    @asynccontextmanager
    async def _lifespan(_app: FastAPI):
        try:
            yield
        finally:
            await _close_shared_client()

    proxy = FastAPI(
        title="SINGULARITY DeepSeek-V4.1-Flash-1017 Gateway",
        lifespan=_lifespan,
    )
    proxy.state.get_upstream_client = _get_shared_client
    proxy.state.close_upstream_client = _close_shared_client

    @proxy.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    )
    async def forward(path: str, request: Request):
        clean_path = f"/{path.lstrip('/')}"
        method = request.method
        t0 = time.time()
        if method == "POST":
            print(f"[SINGULARITY proxy] --> {method} {clean_path}", flush=True)
        try:
            body = await request.body()
        except Exception as exc:
            elapsed_ms = (time.time() - t0) * 1000.0
            print(
                f"[SINGULARITY proxy] Client disconnected reading request body on "
                f"{method} {clean_path} after {elapsed_ms:.1f}ms: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return Response(
                content=json.dumps(
                    {
                        "error": f"Client disconnected before request body was received: {type(exc).__name__}"
                    }
                ),
                status_code=499,
                media_type="application/json",
            )

        target_url = f"{upstream}{clean_path}"
        req_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }
        client = _get_shared_client()
        try:
            upstream_req = client.build_request(
                method,
                target_url,
                content=body,
                headers=req_headers,
                params=dict(request.query_params),
            )
            resp = await client.send(upstream_req, stream=True)
        except Exception as exc:
            elapsed_ms = (time.time() - t0) * 1000.0
            print(
                f"[SINGULARITY proxy] ERROR on {method} {clean_path} after {elapsed_ms:.1f}ms: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return Response(
                content=json.dumps({"error": f"Upstream connection failed: {exc}"}),
                status_code=502,
                media_type="application/json",
            )

        resp_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower() not in HOP_BY_HOP_HEADERS
        }
        content_type = resp.headers.get("content-type", "")
        if "text/event-stream" in content_type.lower():
            async def stream_chunks():
                try:
                    async for chunk in resp.aiter_raw():
                        if chunk:
                            yield chunk
                except Exception as exc:
                    elapsed_ms = (time.time() - t0) * 1000.0
                    print(
                        f"[SINGULARITY proxy] Stream interrupted on {method} {clean_path} "
                        f"after {elapsed_ms:.1f}ms: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                finally:
                    await resp.aclose()
                    elapsed_ms = (time.time() - t0) * 1000.0
                    print(
                        f"[SINGULARITY proxy] <-- {method} {clean_path} {resp.status_code} "
                        f"(SSE stream closed in {elapsed_ms:.1f}ms)",
                        flush=True,
                    )

            return StreamingResponse(
                stream_chunks(),
                status_code=resp.status_code,
                headers=resp_headers,
                media_type="text/event-stream",
            )

        try:
            payload = await resp.aread()
        except Exception as exc:
            elapsed_ms = (time.time() - t0) * 1000.0
            print(
                f"[SINGULARITY proxy] Error reading upstream response on {method} {clean_path} "
                f"after {elapsed_ms:.1f}ms: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return Response(
                content=json.dumps({"error": f"Upstream response read failed: {exc}"}),
                status_code=502,
                media_type="application/json",
            )
        finally:
            await resp.aclose()

        elapsed_ms = (time.time() - t0) * 1000.0
        if method == "POST" or resp.status_code >= 400:
            print(
                f"[SINGULARITY proxy] <-- {method} {clean_path} {resp.status_code} ({elapsed_ms:.1f}ms)",
                flush=True,
            )
        return Response(
            content=payload,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=content_type or None,
        )

    return proxy


def ensure_checkpoint_ready(
    checkpoint_dir: str = CHECKPOINT_DIR,
    *,
    reload_callback: Optional[object] = None,
    commit_callback: Optional[object] = None,
    strict_pinned: bool = True,
) -> Dict[str, object]:
    """Ensure the pinned checkpoint is prepared and validated before starting `singularity serve`."""
    if callable(reload_callback):
        reload_callback()
    if Path(checkpoint_dir).is_dir():
        prepare_csf_checkpoint_directory(checkpoint_dir)
    precheck = validate_checkpoint_directory(
        checkpoint_dir,
        strict_pinned=strict_pinned,
        verify_shard_headers=False,
    )
    if not precheck["valid"]:
        print(
            f"[SINGULARITY container] Checkpoint precheck invalid ({precheck['errors']}); downloading checkpoint...",
            flush=True,
        )
        download_pinned_checkpoint(
            checkpoint_dir,
            repo_id=CSF_MODEL_ID,
            revision=CSF_PINNED_REVISION,
            commit_callback=commit_callback if callable(commit_callback) else None,
        )
        if callable(commit_callback):
            commit_callback()
        precheck = validate_checkpoint_directory(
            checkpoint_dir,
            strict_pinned=strict_pinned,
            verify_shard_headers=False,
        )
        if not precheck["valid"]:
            raise RuntimeError(
                f"Checkpoint validation failed after download: {precheck['errors']}"
            )
    shard_count = precheck.get("shard_count", precheck.get("present_shards", 0))
    total_bytes = precheck.get("total_bytes", precheck.get("total_shard_file_bytes", 0))
    print(
        f"[SINGULARITY container] Checkpoint inventory verified at {checkpoint_dir} "
        f"({shard_count} shards, {total_bytes} bytes).",
        flush=True,
    )
    return precheck


try:
    import modal

    app = modal.App(APP_NAME)
    singularity_image = build_singularity_image()
    weights_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    benchmark_volume = modal.Volume.from_name(BENCHMARK_VOLUME_NAME, create_if_missing=True)
    hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
    _effective_min_containers = int(os.environ.get("SINGULARITY_MIN_CONTAINERS", "0"))

    @app.function(
        image=singularity_image,
        volumes={VOLUME_MOUNT_PATH: weights_volume},
        secrets=[hf_secret],
        timeout=86_400,
        cpu=16.0,
        memory=65_536,
    )
    def download_checkpoint_remote(
        repo_id: str = CSF_MODEL_ID,
        revision: str = CSF_PINNED_REVISION,
        hf_token: str = "",
        dry_run: bool = False,
    ) -> Dict[str, object]:
        result = download_pinned_checkpoint(
            CHECKPOINT_DIR,
            repo_id=repo_id,
            revision=revision,
            hf_token=hf_token or None,
            dry_run=dry_run,
            commit_callback=weights_volume.commit if not dry_run else None,
        )
        if not dry_run:
            weights_volume.commit()
        return result

    @app.function(
        image=singularity_image,
        volumes={VOLUME_MOUNT_PATH: weights_volume},
        timeout=1800,
        cpu=8.0,
        memory=32_768,
    )
    def validate_checkpoint_remote() -> Dict[str, object]:
        weights_volume.reload()
        if Path(CHECKPOINT_DIR).is_dir():
            prepare_csf_checkpoint_directory(CHECKPOINT_DIR)
        return validate_checkpoint_directory(
            CHECKPOINT_DIR, strict_pinned=True, verify_shard_headers=True
        )

    @app.cls(
        image=singularity_image,
        gpu=GPU_SPEC,
        cpu=CPU_CORES,
        memory=MEMORY_MB,
        volumes={
            VOLUME_MOUNT_PATH: weights_volume,
            BENCHMARK_MOUNT_PATH: benchmark_volume,
        },
        secrets=[hf_secret],
        min_containers=_effective_min_containers,
        max_containers=MAX_CONTAINERS,
        scaledown_window=SCALEDOWN_WINDOW_SECS,
        startup_timeout=STARTUP_TIMEOUT_SECS,
        timeout=REQUEST_TIMEOUT_SECS,
    )
    class SingularityV41Service:
        @modal.enter()
        def startup(self) -> None:
            print("[SINGULARITY container] Entering SingularityV41Service.startup()...", flush=True)
            ensure_checkpoint_ready(
                CHECKPOINT_DIR,
                reload_callback=weights_volume.reload,
                commit_callback=weights_volume.commit,
                strict_pinned=True,
            )
            self.supervisor = SingularityRuntimeSupervisor(
                checkpoint_dir=CHECKPOINT_DIR,
                host=INTERNAL_SERVER_HOST,
                port=INTERNAL_SERVER_PORT,
            )
            self.supervisor.start(
                require_gpus=True,
                ready_timeout_secs=STARTUP_TIMEOUT_SECS,
                skip_checkpoint_revalidation=True,
            )
            print("[SINGULARITY container] SingularityV41Service.startup() complete and READY.", flush=True)

        @modal.exit()
        def shutdown(self) -> None:
            if hasattr(self, "supervisor"):
                self.supervisor.stop()

        @modal.method()
        def inspect_status(self) -> Dict[str, object]:
            return check_service_health(
                self.supervisor.base_url,
                api_key=os.environ.get("SINGULARITY_API_KEY"),
            )

        @modal.asgi_app()
        def serve(self):
            return create_asgi_proxy_app(
                self.supervisor.base_url,
                timeout_secs=REQUEST_TIMEOUT_SECS,
            )

except ImportError:
    app = None
    SingularityV41Service = None
