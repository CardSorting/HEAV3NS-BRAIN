# Post-Mortem PM-004: Deploy-Time Scaling Policy Environment Drift in Modal Containers

- **Incident Date**: 2026-10-31
- **Severity**: SEV-2 (In-container `LifecycleTracker` and `LifecycleWatchdog` silently ran with default 6-hour lifetime and unlimited spend instead of operator-configured deploy-time bounds)
- **Status**: Resolved & Verified Live on Modal (`T4` and `2×H100`)
- **Affected Files**: [`deploy/modal/config.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/config.py), [`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py), [`deploy/modal/deepinfer_app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/deepinfer_app.py), [`deploy/modal/smoke_app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/smoke_app.py)

---

## 1. Executive Summary

During live scale-to-zero smoke validation on Modal (`2026-10-31`), `deploy/modal/smoke_app.py` was deployed with `SINGULARITY_MAX_WORKER_LIFETIME_SECS=60` and `SINGULARITY_SCALEDOWN_WINDOW_SECS=300` in the operator's shell environment. While Modal registered `scaledown_window=300` on the `@app.cls` decorator (evaluated locally at deploy time), the in-container `LifecycleWatchdog` failed to trip after 60 seconds of serving uptime and remained alive for over 2 minutes until manually stopped.

---

## 2. Root Cause Analysis

Modal's execution model evaluates `@app.cls(...)` decorator arguments (`min_containers`, `max_containers`, `scaledown_window`, `timeout`) in the **local deploy process**, where the operator's environment variables (`SINGULARITY_MAX_WORKER_LIFETIME_SECS=60`, `SINGULARITY_MAX_SPEND_USD`, etc.) are present in `os.environ`.

However, when a container is provisioned in the cloud, Modal starts a fresh Python interpreter inside the container image and re-imports the entrypoint script (`/root/app.py` or `/root/smoke_app.py`). Because the Modal image `.env({...})` dictionary only included static constants (`RUST_LOG`, `HF_HOME`, `SINGULARITY_MODEL_ID`), `os.environ` inside the container did **not** contain the deploy-time `SINGULARITY_*` policy variables. Consequently, `resolve_scaling_policy()` inside `startup()` silently fell back to default limits (`max_worker_lifetime_secs = 21600`, `max_spend_usd = None`).

---

## 3. Remediation

1. **Added `policy_env_snapshot()` in [`deploy/modal/config.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/config.py#L221-L234)**:
   - Captures all whitelisted `SINGULARITY_*` scaling and cost-guard environment variables (`_POLICY_ENV_KEYS`) present at deploy time so they can be merged directly into the Modal image `.env({...})` definition.
   - Intentionally excludes credentials/secrets (`HF_TOKEN`, `MODAL_TOKEN_ID`, etc.) so secrets are never baked into image layers.
2. **Wired `**policy_env_snapshot()` Across All Modal Images**:
   - Applied to `build_singularity_image()` in [`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py#L140), `build_deepinfer_image()` in [`deploy/modal/deepinfer_app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/deepinfer_app.py#L81), and `_image` in [`deploy/modal/smoke_app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/smoke_app.py#L46).
3. **Exposed Active `limits` in `/__lifecycle/status` (`deploy/modal/lifecycle.py`)**:
   - Added a `limits` block (`max_concurrent_requests`, `request_timeout_secs`, `drain_timeout_secs`, `max_worker_lifetime_secs`, `max_spend_usd`, `gpu_usd_per_gpu_hour`) to `LifecycleTracker.snapshot()` so operators can verify the exact bounds enforced inside a running container via `deploy.py status --url <endpoint>`.

---

## 4. Verification & Preventative Guardrails

- **Live Verification (`T4` and `2×H100`)**: Redeployed `singularity-smoke` with `SINGULARITY_MAX_WORKER_LIFETIME_SECS=60` and `SINGULARITY_SCALEDOWN_WINDOW_SECS=300`. Confirmed container reached `READY` at `12:46:22`, tripped `[SINGULARITY container] guard tripped; exiting container` at `12:47:22` (exactly 60s later, well before the 300s idle window), executed `@modal.exit`, and scaled to `0` containers (`modal container list --json` = `[]`). Repeated and verified on `2×H100` (`12:50:20` READY $\rightarrow$ `12:51:20` guard trip).
- **Automated Regression Tests**: Added `PolicyEnvSnapshotTests` (`test_policy_env_snapshot_captures_only_whitelisted_policy_vars`) and `SnapshotLimitsTests` in [`deploy/modal/test_lifecycle.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/test_lifecycle.py).

