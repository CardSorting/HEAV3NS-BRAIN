# ADR-006: True Scale-to-Zero Container Lifecycle, Deploy-Time Policy Snapshot & Cost-Governance State Machine

- **Status**: Accepted & Verified Live on Modal (`T4`, `2×H100`) and Offline (`71` unit/gateway tests)
- **Date**: 2026-10-31
- **Components**: [`deploy/modal/config.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/config.py), [`deploy/modal/lifecycle.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/lifecycle.py), [`deploy/modal/ops.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/ops.py), [`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py), [`deploy/modal/deepinfer_app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/deepinfer_app.py), [`deploy/modal/smoke_app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/smoke_app.py), [`deploy/modal/deploy.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/deploy.py)

---

## 1. Context

Multi-GPU frontier inference nodes (`8× NVIDIA B200`) carry a high hourly burn rate when provisioned. An inference deployment that leaves warm containers idle, fails to bound worker lifetime or spend, leaks admission slots when streaming clients disconnect early, or leaves an unready container alive after a startup failure violates cost-governance invariants.

Specifically, our Pass 1–6 audit identified six architectural requirements:
1. **Default Zero Idle Compute**: `min_containers` must default to `0` across all Modal apps (`singularity-v41-flash-1017`, `singularity-deepinfer-v41`, `singularity-smoke`), and any attempt to pin warm containers (`SINGULARITY_MIN_CONTAINERS > 0`) must be rejected unless `SINGULARITY_ALLOW_PINNED_WARM=1` is explicitly set.
2. **Deploy-Time to Runtime Policy Propagation (`PM-004`)**: Modal containers re-import the entrypoint module on boot without inheriting the operator's local shell environment. Scaling and cost-guard environment variables must be baked into the container image at deploy time.
3. **Explicit Container Lifecycle & Separate Serving vs. Billing Clocks**: Model initialization on large checkpoints can take minutes. Counting `max_worker_lifetime_secs` from container start would prematurely kill containers before or shortly after they reach `READY`, whereas counting estimated GPU spend must start at container boot when cloud billing begins.
4. **Self-Enforcing Watchdog with Non-Zero Hard-Exit Fallback**: When a lifetime cap, spend cap, or child-process death occurs, the container must stop admitting requests, drain in-flight work within a bounded window, stop the child server, and trigger container shutdown (`SIGINT`), backed by a hard-exit timer (`os._exit(70)`) if the runtime does not terminate the container within 60 seconds.
5. **Streaming Slot & Token Accounting under Standard SSE Client Disconnects (`PM-005`)**: Both never-iterated streams (client disconnects before the first chunk) and standard OpenAI SSE clients (which close the socket immediately upon reading `data: [DONE]`) must deterministically release their admission slot, close the upstream `httpx.Response` in normal async context, and record accurate completion and `tokens_out` metrics.
6. **Provider-Side Zero-Compute & Orphan Verification**: Zero idle compute cannot be proven from inside a running container; it requires querying `modal container list --json` and `modal app list --json` outside the container and classifying any container exceeding `max_worker_lifetime_secs + startup_allowance + drain_timeout + 120s` across all known apps (`KNOWN_APP_NAMES`) as an orphan.

---

## 2. Decision

1. **Validated `ScalingPolicy` & `policy_env_snapshot()` (`deploy/modal/config.py`)**:
   - `resolve_scaling_policy()` validates and bounds all scaling parameters (`min_containers=0`, `max_containers <= 8`, `scaledown_window_secs <= 900`, `request_timeout_secs <= 3600`, `drain_timeout_secs <= 300`, `max_worker_lifetime_secs <= 86400`, `max_spend_usd` requires `gpu_usd_per_gpu_hour`).
   - `policy_env_snapshot()` captures all whitelisted `SINGULARITY_*` policy variables at deploy time and bakes them into the Modal image `.env(...)` for `app.py`, `deepinfer_app.py`, and `smoke_app.py`.
2. **6-State `LifecycleTracker` & `LifecycleWatchdog` (`deploy/modal/lifecycle.py`)**:
   - Tracks explicit transitions: `STARTING -> INITIALIZING -> READY -> DRAINING -> STOPPED` (and `-> FAILED` from any active state).
   - Starts the `max_worker_lifetime_secs` clock at `READY` (`serving_uptime_secs`) while measuring `estimated_spend_usd` from `started_at` (container boot).
   - Enforces server-generated request IDs (`uuid4().hex`), concurrency caps (`HTTP 429`), per-request deadlines (`HTTP 504` + `reap_expired`), and bounded graceful drain (`drain(timeout_secs)`).
   - `LifecycleWatchdog` polls every 5s (local memory + `Popen.poll()`, zero network/GPU traffic so it never resets Modal's idle timer) and triggers `_exit_container` on lifetime/spend/dead-child trips.
3. **Deterministic SSE Lifecycle in `_ReleasingStreamingResponse` (`deploy/modal/app.py`)**:
   - Overrides `__call__` to explicitly `await self.body_iterator.aclose()` before `await self._on_finish()`, ensuring both never-iterated streams and streams closed by the client immediately after `data: [DONE]` run `_finalize_stream()` in normal async context, release their admission slot, close `httpx.Response`, and record `ok=True` (`completed += 1`) and `tokens_out`.
4. **Provider-Side Audit & Unified Teardown (`deploy/modal/ops.py`, `deploy/modal/deploy.py`)**:
   - `ops.audit_zero_compute()` queries provider state across `KNOWN_APP_NAMES = (APP_NAME, DEEPINFER_APP_NAME, SMOKE_APP_NAME)`, flags orphans, and supports opt-in `--terminate-orphans`.
   - `deploy.py teardown` (`make teardown`) stops all known Singularity Modal apps with `--yes` and terminates any remaining containers.

---

## 3. Impact on the Optimal Runtime Experience (ORE)

- **Cost Efficiency**: Guarantees `$0.00/hr` idle GPU compute by default, verified live on Modal (`T4` and `2×H100` 0→1 cold start, idle scale-down to 0, and 60s lifetime-guard self-exit).
- **Reliability**: Eliminates false SSE failure counts on `[DONE]` disconnects, prevents leaked admission slots on abandoned streams, and prevents unready containers from billing after startup failures.
- **Verification**: Locked by 71 automated tests in [`deploy/modal/test_lifecycle.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/test_lifecycle.py), [`deploy/modal/test_scaling_policy.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/test_scaling_policy.py), and [`deploy/modal/test_modal_pipeline.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/test_modal_pipeline.py).

