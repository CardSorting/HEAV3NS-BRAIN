"""Persistent 8x NVIDIA B200 Modal serving application for SINGULARITY DeepSeek-V4.1-Flash-1017."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional

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
        MEMORY_MB,
        REQUEST_TIMEOUT_SECS,
        STARTUP_RETRY_MAX_ATTEMPTS,
        STARTUP_TIMEOUT_SECS,
        VOLUME_MOUNT_PATH,
        VOLUME_NAME,
        policy_env_snapshot,
        resolve_scaling_policy,
    )
    from deploy.modal.health import check_service_health, verify_gpu_topology
    from deploy.modal.lifecycle import (
        AdmissionRejected,
        LifecycleState,
        LifecycleTracker,
        LifecycleWatchdog,
        retry_bounded,
        start_with_cleanup,
    )
    from deploy.modal.ops import gpu_snapshot
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
        MEMORY_MB,
        REQUEST_TIMEOUT_SECS,
        STARTUP_RETRY_MAX_ATTEMPTS,
        STARTUP_TIMEOUT_SECS,
        VOLUME_MOUNT_PATH,
        VOLUME_NAME,
        policy_env_snapshot,
        resolve_scaling_policy,
    )
    from .health import check_service_health, verify_gpu_topology
    from .lifecycle import (
        AdmissionRejected,
        LifecycleState,
        LifecycleTracker,
        LifecycleWatchdog,
        retry_bounded,
        start_with_cleanup,
    )
    from .ops import gpu_snapshot
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

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        """Idempotent; safe on partial init (process may be None or already dead)."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)


def create_asgi_proxy_app(
    upstream_url: str,
    timeout_secs: float = REQUEST_TIMEOUT_SECS,
    tracker: Optional[LifecycleTracker] = None,
    gpu_sampler: Optional[object] = None,
    liveness: Optional[object] = None,
):
    """Create the FastAPI reverse proxy that forwards HTTP and SSE traffic to the local
    `singularity serve` child process without buffering `text/event-stream` responses.
    Uses a loop-scoped connection-pooled `httpx.AsyncClient` to eliminate per-request
    client/pool allocation and loopback TCP handshake churn under concurrent load.

    With a `tracker`, every POST (generation work) is admitted/rejected by the lifecycle
    state machine (not-ready, draining, concurrency, lifetime, spend), counted in flight
    until the response -- including a streamed one -- fully closes, and bounded by a hard
    per-request wall-clock deadline. `GET /__lifecycle/status` is answered locally. The
    gateway runs NO background task: it never pings upstream on its own.
    """
    import asyncio
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import StreamingResponse
    import httpx

    # Ensure `from __future__ import annotations` resolves `Request` in module globals.
    globals()["Request"] = Request
    globals()["Response"] = Response

    class _ReleasingStreamingResponse(StreamingResponse):
        """StreamingResponse that explicitly closes `body_iterator` and runs `on_finish`
        when the ASGI call ends.

        Starlette cancels the body task when the client disconnects (including right after
        reading `data: [DONE]`). Explicitly closing `body_iterator` before `on_finish`
        guarantees deterministic cleanup order, and `on_finish` runs in normal async context
        so `await resp.aclose()` never executes inside a `GeneratorExit` handler.
        """

        def __init__(self, *args, on_finish, **kwargs):
            super().__init__(*args, **kwargs)
            self._on_finish = on_finish

        async def __call__(self, scope, receive, send):
            try:
                await super().__call__(scope, receive, send)
            finally:
                aclose = getattr(self.body_iterator, "aclose", None)
                if callable(aclose):
                    try:
                        await aclose()
                    except BaseException:  # noqa: BLE001
                        pass
                await self._on_finish()

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

    def _noop_close(*, ok: bool = True, tokens: int = 0) -> None:
        return None

    async def forward(path: str, request: Request):
        clean = f"/{path.lstrip('/')}"
        if tracker is not None and request.method == "GET" and clean == "/__lifecycle/status":
            snap = tracker.snapshot()
            if request.query_params.get("gpu") == "1" and callable(gpu_sampler):
                snap["gpu"] = gpu_sampler()  # on-demand only; never sampled in the background
            return Response(content=json.dumps(snap), media_type="application/json")
        if tracker is None or request.method != "POST":
            return await _forward_inner(path, request, _noop_close)

        tracker.reap_expired()  # pull-based stuck-job reaping; no background loop
        if callable(liveness) and not liveness():
            # Child `singularity serve` is gone: never keep a dead GPU worker admitting work.
            try:
                tracker.transition(LifecycleState.FAILED)
            except Exception:  # noqa: BLE001
                tracker.begin_drain()
            return Response(
                content=json.dumps({"error": "inference worker process exited", "code": "worker_dead"}),
                status_code=503,
                headers={"Retry-After": "30"},
                media_type="application/json",
            )
        # Server-generated key. A client-supplied x-request-id is NEVER the dedupe/accounting
        # key (a client could collide with or squat on another request's id); it is only
        # logged for correlation.
        rid = uuid.uuid4().hex
        client_rid = request.headers.get("x-request-id")
        if client_rid:
            print(f"[SINGULARITY proxy] request {rid} client x-request-id={client_rid[:64]!r}", flush=True)
        try:
            tracker.admit(rid)
        except AdmissionRejected as rej:
            return Response(
                content=json.dumps({"error": rej.detail, "code": rej.code}),
                status_code=rej.http_status,
                headers={"Retry-After": "30"},
                media_type="application/json",
            )

        def on_close(*, ok: bool = True, tokens: int = 0) -> None:
            tracker.release(rid, ok=ok, tokens=tokens)

        ctx: Dict[str, object] = {"tokens": 0}

        async def _close_upstream() -> None:
            up = ctx.pop("resp", None)
            if up is not None:
                try:
                    await asyncio.shield(up.aclose())
                except Exception:  # noqa: BLE001
                    pass

        try:
            result = await asyncio.wait_for(
                _forward_inner(path, request, on_close, ctx), timeout=timeout_secs
            )
        except asyncio.TimeoutError:
            await _close_upstream()  # never leak the upstream connection on the 504 path
            on_close(ok=False)
            return Response(
                content=json.dumps({"error": "per-request deadline exceeded", "code": "timeout"}),
                status_code=504,
                media_type="application/json",
            )
        except BaseException:
            await _close_upstream()
            on_close(ok=False)
            raise
        if not isinstance(result, StreamingResponse):
            on_close(ok=result.status_code < 500, tokens=int(ctx.get("tokens", 0) or 0))
        return result

    proxy.add_api_route(
        "/{path:path}",
        forward,
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"],
    )

    async def _forward_inner(path: str, request: Request, on_close, ctx: Optional[dict] = None):
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
            if ctx is not None:
                ctx["resp"] = resp
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
            stream_state: Dict[str, object] = {
                "finished": False,
                "started": False,
                "saw_done": False,
                "ok": False,
                "tokens": 0,
                "tail": b"",
            }

            def _scan_sse_bytes(raw: bytes, *, final: bool = False) -> None:
                buf = bytes(stream_state["tail"]) + raw
                lines = buf.split(b"\n")
                stream_state["tail"] = b"" if final else lines.pop()
                for raw_line in lines:
                    line = raw_line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        stream_state["saw_done"] = True
                        continue
                    try:
                        parsed = json.loads(data.decode("utf-8", errors="replace"))
                    except Exception:  # noqa: BLE001
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    choices = parsed.get("choices")
                    if isinstance(choices, list) and choices:
                        first = choices[0] if isinstance(choices[0], dict) else {}
                        delta = first.get("delta") if isinstance(first, dict) else None
                        if isinstance(delta, dict):
                            if (
                                delta.get("content")
                                or delta.get("reasoning_content")
                                or delta.get("tool_calls")
                            ):
                                stream_state["tokens"] = int(stream_state["tokens"]) + 1
                        elif first.get("text"):
                            stream_state["tokens"] = int(stream_state["tokens"]) + 1

            async def stream_chunks():
                stream_state["started"] = True
                try:
                    chunks = resp.aiter_raw().__aiter__()
                    while True:
                        remaining = (t0 + timeout_secs) - time.time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError("per-request deadline exceeded")
                        try:
                            chunk = await asyncio.wait_for(chunks.__anext__(), timeout=remaining)
                        except StopAsyncIteration:
                            _scan_sse_bytes(b"", final=True)
                            stream_state["ok"] = resp.status_code < 500
                            break
                        if chunk:
                            _scan_sse_bytes(chunk)
                            yield chunk
                except (GeneratorExit, asyncio.CancelledError) as exc:
                    _scan_sse_bytes(b"", final=True)
                    if stream_state["saw_done"]:
                        # Standard OpenAI SSE clients close the socket as soon as they read
                        # `data: [DONE]`; treat post-[DONE] disconnect as a clean completion.
                        stream_state["ok"] = resp.status_code < 500
                    else:
                        stream_state["ok"] = False
                        elapsed_ms = (time.time() - t0) * 1000.0
                        print(
                            f"[SINGULARITY proxy] Stream interrupted on {method} {clean_path} "
                            f"after {elapsed_ms:.1f}ms: {type(exc).__name__}: {exc}",
                            flush=True,
                        )
                    raise
                except BaseException as exc:
                    stream_state["ok"] = False
                    elapsed_ms = (time.time() - t0) * 1000.0
                    print(
                        f"[SINGULARITY proxy] Stream interrupted on {method} {clean_path} "
                        f"after {elapsed_ms:.1f}ms: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    if not isinstance(exc, Exception):
                        raise

            async def _finalize_stream() -> None:
                if stream_state["finished"]:
                    return
                stream_state["finished"] = True
                try:
                    await asyncio.shield(resp.aclose())
                except Exception:  # noqa: BLE001
                    pass
                ok = bool(stream_state["ok"])
                tokens = int(stream_state["tokens"])
                on_close(ok=ok, tokens=tokens)
                elapsed_ms = (time.time() - t0) * 1000.0
                if not stream_state["started"] or (not ok and not stream_state["saw_done"]):
                    print(
                        f"[SINGULARITY proxy] <-- {method} {clean_path} stream abandoned before "
                        f"completion ({elapsed_ms:.1f}ms); slot released",
                        flush=True,
                    )
                else:
                    print(
                        f"[SINGULARITY proxy] <-- {method} {clean_path} {resp.status_code} "
                        f"(SSE stream closed in {elapsed_ms:.1f}ms, tokens={tokens})",
                        flush=True,
                    )

            return _ReleasingStreamingResponse(
                stream_chunks(),
                on_finish=_finalize_stream,
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

        if ctx is not None and resp.status_code < 400 and "application/json" in content_type.lower():
            try:
                parsed_json = json.loads(payload.decode("utf-8", errors="replace"))
                if isinstance(parsed_json, dict):
                    usage = parsed_json.get("usage")
                    if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                        ctx["tokens"] = max(0, int(usage["completion_tokens"]))
            except Exception:  # noqa: BLE001
                pass

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


HARD_EXIT_CODE = 70  # EX_SOFTWARE: the graceful path failed; never report this as success
HARD_EXIT_CLEANUP_SECS = 10.0


def _hard_exit(cleanup: Optional[Callable[[], None]] = None) -> None:
    """Last-resort exit: bounded best-effort cleanup, flush logs, then exit non-zero.

    The cleanup (e.g. kill the 8-GPU child) runs in a helper thread and is abandoned after
    `HARD_EXIT_CLEANUP_SECS`, so a hung cleanup can never prevent the exit itself.
    """
    import threading

    print(f"[SINGULARITY container] graceful exit did not complete; hard-exit {HARD_EXIT_CODE}", flush=True)
    if cleanup is not None:
        def _run() -> None:
            try:
                cleanup()
            except BaseException as exc:  # noqa: BLE001 - best effort, must not block exit
                print(f"[SINGULARITY container] hard-exit cleanup failed: {type(exc).__name__}: {exc}", flush=True)

        worker = threading.Thread(target=_run, name="hard-exit-cleanup", daemon=True)
        worker.start()
        worker.join(HARD_EXIT_CLEANUP_SECS)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    os._exit(HARD_EXIT_CODE)


def _exit_container(grace_secs: float = 60.0, cleanup: Optional[Callable[[], None]] = None) -> None:
    """Ask the Modal runtime to shut this container down (graceful SIGINT -> `@modal.exit`).

    If the runtime has not killed us after `grace_secs`, run best-effort `cleanup` and
    hard-exit with a NON-ZERO code so a tripped guard can never leave the GPUs billing
    (and the failure is visible rather than looking like a clean exit).
    """
    import signal
    import threading

    print("[SINGULARITY container] guard tripped; exiting container", flush=True)
    timer = threading.Timer(grace_secs, lambda: _hard_exit(cleanup))
    timer.daemon = True
    timer.start()
    os.kill(os.getpid(), signal.SIGINT)


try:
    import modal

    if os.environ.get("SINGULARITY_SKIP_MODAL_APP") == "1":
        # Lets smoke_app.py reuse this module's gateway code without defining the 8x B200 app.
        raise ImportError("SINGULARITY_SKIP_MODAL_APP=1")

    app = modal.App(APP_NAME)
    singularity_image = build_singularity_image()
    _policy_env = policy_env_snapshot()
    if _policy_env:
        singularity_image = singularity_image.env(_policy_env)
    weights_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    benchmark_volume = modal.Volume.from_name(BENCHMARK_VOLUME_NAME, create_if_missing=True)
    hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
    _policy = resolve_scaling_policy()
    if _policy.pinned_warm:
        print(
            "[SINGULARITY WARNING] pinned warm capacity enabled: 8x B200 bill continuously "
            "until redeployed with SINGULARITY_MIN_CONTAINERS=0 or torn down.",
            flush=True,
        )

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
        min_containers=_policy.min_containers,
        max_containers=_policy.max_containers,
        scaledown_window=_policy.scaledown_window_secs,
        startup_timeout=_policy.startup_timeout_secs,
        timeout=_policy.request_timeout_secs,
    )
    class SingularityV41Service:
        @modal.enter()
        def startup(self) -> None:
            print("[SINGULARITY container] Entering SingularityV41Service.startup()...", flush=True)
            self.tracker = LifecycleTracker(
                max_concurrent=_policy.max_concurrent_requests,
                request_timeout_secs=_policy.request_timeout_secs,
                max_lifetime_secs=_policy.max_worker_lifetime_secs,
                gpu_count=GPU_COUNT,
                gpu_usd_per_gpu_hour=_policy.gpu_usd_per_gpu_hour,
                max_spend_usd=_policy.max_spend_usd,
            )
            self.supervisor = SingularityRuntimeSupervisor(
                checkpoint_dir=CHECKPOINT_DIR,
                host=INTERNAL_SERVER_HOST,
                port=INTERNAL_SERVER_PORT,
            )

            def _boot() -> None:
                # Bounded retry (network/volume flakiness only); GPU init is NOT retried
                # in-process because a failed 8xB200 start is expensive and usually deterministic.
                retry_bounded(
                    lambda: ensure_checkpoint_ready(
                        CHECKPOINT_DIR,
                        reload_callback=weights_volume.reload,
                        commit_callback=weights_volume.commit,
                        strict_pinned=True,
                    ),
                    attempts=STARTUP_RETRY_MAX_ATTEMPTS,
                )
                self.supervisor.start(
                    require_gpus=True,
                    ready_timeout_secs=STARTUP_TIMEOUT_SECS,
                    skip_checkpoint_revalidation=True,
                )

            result = start_with_cleanup(self.tracker, _boot, self.supervisor.stop, attempts=1)
            if not result.ok:
                # Re-raise so Modal fails the container instead of serving a half-initialised cluster.
                self.tracker.transition(LifecycleState.STOPPED)
                raise RuntimeError(f"startup failed and was cleaned up: {result.error}")
            # Enforce lifetime/spend guards by actually exiting (admission control alone
            # would leave the container billing). Polls local state only.
            self.watchdog = LifecycleWatchdog(
                self.tracker,
                drain_timeout_secs=_policy.drain_timeout_secs,
                stop_child=self.supervisor.stop,
                exit_fn=lambda: _exit_container(cleanup=self.supervisor.stop),
                liveness=self.supervisor.alive,
            )
            self.watchdog.start()
            print("[SINGULARITY container] SingularityV41Service.startup() complete and READY.", flush=True)

        @modal.exit()
        def shutdown(self) -> None:
            watchdog = getattr(self, "watchdog", None)
            if watchdog is not None:
                watchdog.halt()
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                # Stop admitting, wait (bounded) for in-flight generations, then kill the child.
                drained = tracker.drain(_policy.drain_timeout_secs)
                print(
                    f"[SINGULARITY container] drain complete={drained} snapshot={tracker.snapshot()}",
                    flush=True,
                )
            if hasattr(self, "supervisor"):
                self.supervisor.stop()
            if tracker is not None:
                try:
                    tracker.transition(LifecycleState.STOPPED)
                except Exception:  # noqa: BLE001
                    pass

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
                timeout_secs=_policy.request_timeout_secs,
                tracker=self.tracker,
                gpu_sampler=gpu_snapshot,
                liveness=self.supervisor.alive,
            )

except ImportError:
    app = None
    SingularityV41Service = None
