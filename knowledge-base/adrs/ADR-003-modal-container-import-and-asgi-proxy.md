# ADR-003: Modal Shallow-Mount Import Resolution & Connection-Pooled ASGI Streaming Proxy

- **Status**: Accepted & Deployed in Production (`im-R1TdH6JxBrlEvExHAIqaph`)
- **Date**: 2026-10-31
- **Components**: [`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py), `deploy/modal/*.py`

---

## 1. Context

Two container-boundary behaviors in Modal's serverless runtime required explicit architectural handling:
1. **Shallow-Mount Entrypoint Path (`PM-001`)**:
   - In the local repository and inside `/workspace/singularity/deploy/modal/app.py`, `Path(__file__).resolve()` has depth $\ge 3$, so `.parents[2]` resolves `/workspace/singularity`.
   - However, when Modal boots a container for an entrypoint script, it mounts the target file at `/root/app.py` (depth 2: `/root` and `/`). Calling `Path(__file__).resolve().parents[2]` raises `IndexError: 2` before container startup can even begin.
2. **Zero-Buffering HTTP & SSE Reverse Proxy to the Rust Engine**:
   - Modal's `@modal.asgi_app()` exposes a Python ASGI interface on the container ingress, while SINGULARITY's serving plane runs as a native Rust `axum` binary (`singularity serve` on `127.0.0.1:8080`).
   - Creating a new `httpx.AsyncClient` on every request introduced unnecessary object allocation and loopback TCP connection churn under high concurrency (`concurrency=64`).

## 2. Decision

1. **Dynamic Repository Root Discovery (`_resolve_repo_root`)**:
   - Every module in `deploy/modal/*.py` defines `_resolve_repo_root(file_path: Path) -> Path`, which walks `file_path.resolve().parents` and falls back to `/workspace/singularity`, `/opt/singularity`, `Path.cwd()`, and `sys.path` entries to locate `deploy/modal` or `Cargo.toml` without fixed parent-index assumptions.
2. **Loop-Scoped Connection-Pooled `httpx.AsyncClient` with Zero-Buffering SSE Forwarding**:
   - `create_asgi_proxy_app` maintains a loop-scoped `httpx.AsyncClient` configured with `httpx.Limits(max_connections=256, max_keepalive_connections=64)` and cleans it up via FastAPI's `lifespan` context manager.
   - `text/event-stream` responses stream raw chunks immediately via `resp.aiter_raw()` without buffering, and client disconnects while reading request bodies return structured `HTTP 499` responses.

## 3. Impact on the Optimal Runtime Experience (ORE)

- **Reliability**: Zero container import crashloops regardless of where Modal mounts the entrypoint (`/root/app.py` or `/app.py`).
- **Speed & Performance**: Eliminates per-request connection pool initialization and loopback socket churn under 64-way concurrent load while preserving unbuffered token-by-token SSE streaming.

