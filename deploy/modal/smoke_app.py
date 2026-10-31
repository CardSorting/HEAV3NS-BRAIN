"""Small-GPU smoke deployment of the SINGULARITY scale-to-zero stack.

Exercises the REAL gateway (`create_asgi_proxy_app`), `LifecycleTracker`, and
`LifecycleWatchdog` plus the Modal scale-to-zero policy, but fronts a trivial stub
inference server instead of the 8x B200 model. It proves provisioning, the lifetime
guard, and idle scale-down on provider infrastructure at a fraction of the cost. It does
NOT prove anything about B200 performance, NCCL, or the model.

    SINGULARITY_SMOKE_GPU=T4 SINGULARITY_MAX_WORKER_LIFETIME_SECS=120 \\
        modal deploy deploy/modal/smoke_app.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ["SINGULARITY_SKIP_MODAL_APP"] = "1"


def _resolve_repo_root(file_path: Path) -> Path:
    resolved = file_path.resolve()
    for parent in resolved.parents:
        if (parent / "deploy" / "modal").is_dir() or (parent / "Cargo.toml").is_file():
            return parent
    candidates = [
        Path("/root"),
        Path("/workspace/singularity"),
        Path("/opt/singularity"),
        Path.cwd(),
        *(Path(p) for p in sys.path if p),
    ]
    for candidate in candidates:
        if (candidate / "deploy" / "modal").is_dir() or (candidate / "Cargo.toml").is_file():
            return candidate
    return resolved.parent


_REPO_ROOT = _resolve_repo_root(Path(__file__))
if __package__ in (None, ""):
    _modal_dir = str(Path(__file__).resolve().parent)
    if _modal_dir in sys.path and (Path(_modal_dir) / "deploy.py").is_file():
        sys.path.remove(_modal_dir)
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from deploy.modal.config import SMOKE_APP_NAME
else:
    from .config import SMOKE_APP_NAME

SMOKE_GPU = os.environ.get("SINGULARITY_SMOKE_GPU", "T4")
SMOKE_PORT = 8080
STUB_FLAG = "--stub-server"


class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args: object) -> None:  # silence default stderr logging
        return

    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps({"status": "ok", "stub": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for tok in ("hello", " from", " stub"):
            chunk = {"choices": [{"delta": {"content": tok}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.02)
        done = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        self.wfile.write(f"data: {json.dumps(done)}\n\ndata: [DONE]\n\n".encode())
        self.wfile.flush()


def run_stub_server(port: int = SMOKE_PORT) -> None:
    ThreadingHTTPServer(("127.0.0.1", port), _StubHandler).serve_forever()


class StubSupervisor:
    """Same surface as `SingularityRuntimeSupervisor` (base_url/alive/stop), stub child."""

    def __init__(self, port: int = SMOKE_PORT) -> None:
        self.port = port
        self.process: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self, ready_timeout_secs: float = 30.0) -> None:
        import urllib.request

        self.process = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), STUB_FLAG, str(self.port)]
        )
        deadline = time.time() + ready_timeout_secs
        while time.time() < deadline:
            try:
                urllib.request.urlopen(self.base_url + "/ready", timeout=1).read()
                return
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
        raise RuntimeError("stub server did not become ready")

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)


try:
    import modal

    from deploy.modal.app import _exit_container, create_asgi_proxy_app
    from deploy.modal.config import policy_env_snapshot, resolve_scaling_policy
    from deploy.modal.lifecycle import LifecycleState, LifecycleTracker, LifecycleWatchdog

    _policy = resolve_scaling_policy()
    app = modal.App(SMOKE_APP_NAME)
    _image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("fastapi", "httpx")
        .env({"SINGULARITY_SKIP_MODAL_APP": "1", **policy_env_snapshot()})
        .add_local_dir(
            str(_REPO_ROOT / "deploy"),
            remote_path="/root/deploy",
            ignore=["__pycache__", "*.pyc"],
        )
    )

    @app.cls(
        image=_image,
        gpu=SMOKE_GPU,
        cpu=2.0,
        memory=4096,
        min_containers=0,
        max_containers=1,
        scaledown_window=_policy.scaledown_window_secs,
        startup_timeout=_policy.startup_timeout_secs,
        timeout=_policy.request_timeout_secs,
    )
    class SmokeService:
        @modal.enter()
        def startup(self) -> None:
            self.tracker = LifecycleTracker(
                max_concurrent=_policy.max_concurrent_requests,
                request_timeout_secs=_policy.request_timeout_secs,
                max_lifetime_secs=_policy.max_worker_lifetime_secs,
                gpu_count=1,
                gpu_usd_per_gpu_hour=_policy.gpu_usd_per_gpu_hour,
                max_spend_usd=_policy.max_spend_usd,
            )
            self.tracker.transition(LifecycleState.INITIALIZING)
            self.supervisor = StubSupervisor()
            self.supervisor.start()
            self.tracker.transition(LifecycleState.READY)
            self.watchdog = LifecycleWatchdog(
                self.tracker,
                drain_timeout_secs=_policy.drain_timeout_secs,
                stop_child=self.supervisor.stop,
                exit_fn=lambda: _exit_container(cleanup=self.supervisor.stop),
                liveness=self.supervisor.alive,
            )
            self.watchdog.start()
            print("[SMOKE] container READY", flush=True)

        @modal.exit()
        def shutdown(self) -> None:
            print("[SMOKE] container exiting (modal.exit)", flush=True)
            watchdog = getattr(self, "watchdog", None)
            if watchdog is not None:
                watchdog.halt()
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                tracker.drain(_policy.drain_timeout_secs)
            if hasattr(self, "supervisor"):
                self.supervisor.stop()

        @modal.asgi_app()
        def serve(self):
            return create_asgi_proxy_app(
                self.supervisor.base_url,
                timeout_secs=_policy.request_timeout_secs,
                tracker=self.tracker,
                liveness=self.supervisor.alive,
            )

except ImportError:
    app = None
    SmokeService = None


if __name__ == "__main__" and len(sys.argv) >= 3 and sys.argv[1] == STUB_FLAG:
    threading.current_thread().name = "stub-main"
    run_stub_server(int(sys.argv[2]))
