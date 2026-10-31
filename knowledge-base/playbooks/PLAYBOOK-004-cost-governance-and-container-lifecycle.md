# PLAYBOOK-004: 8× NVIDIA B200 Cost Governance, Scale-to-Zero & Container Lifecycle Management

**Applies To**: [`deploy/modal/config.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/config.py), [`deploy/modal/lifecycle.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/lifecycle.py), [`deploy/modal/ops.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/ops.py), [`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py), [`deploy/modal/deepinfer_app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/deepinfer_app.py), [`deploy/modal/smoke_app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/smoke_app.py), [`deploy/modal/deploy.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/deploy.py)

---

## 1. Cost-Governance Invariants & Scale-to-Zero Architecture

> **Verification note (`singularity3.md` / [`docs/SCALE_TO_ZERO_REPORT.md`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/docs/SCALE_TO_ZERO_REPORT.md))**: Pre-existing legacy figures (`$49.99/hour`, `~40–60s` cold start, `TTFT P50 ~ 49ms`) predate the scale-to-zero commission and are treated as unverified legacy estimates. The code never assumes a hardcoded GPU price; spend estimation requires operator-supplied `SINGULARITY_GPU_USD_PER_GPU_HOUR`.

To prevent accidental idle GPU burn while preserving on-demand scale-up capability across all Singularity Modal apps (`singularity-v41-flash-1017`, `singularity-deepinfer-v41`, `singularity-smoke`):

1. **Zero Warm Containers by Default**:
   - `SINGULARITY_MIN_CONTAINERS` defaults to **`0`**.
   - Setting `SINGULARITY_MIN_CONTAINERS > 0` is rejected by `resolve_scaling_policy()` unless `SINGULARITY_ALLOW_PINNED_WARM=1` is explicitly set.
2. **Deploy-Time Policy Baking (`PM-004`)**:
   - `config.policy_env_snapshot()` bakes all whitelisted `SINGULARITY_*` policy variables into the Modal container image at deploy time so in-container `LifecycleTracker` and `LifecycleWatchdog` enforce the exact operator-configured bounds.
3. **Dual Clocks (Ready-Time Lifetime vs. Boot-Time Spend)**:
   - `SINGULARITY_MAX_WORKER_LIFETIME_SECS` (default `21600` = 6h, max `86400`) starts counting only when the container transitions to `READY` (`serving_uptime_secs`), so model initialization time is not subtracted from serving lifetime.
   - `SINGULARITY_MAX_SPEND_USD` counts from container start (`uptime_secs`) because cloud GPU billing starts when the container boots.
4. **Self-Enforcing Watchdog & Hard-Exit Fallback**:
   - `LifecycleWatchdog` polls every 5s (local state + `Popen.poll()` only; zero network/GPU activity so it never resets Modal's idle timer).
   - If the lifetime cap, spend cap, or child-process death trips: admission stops (`DRAINING`), in-flight requests drain up to `SINGULARITY_DRAIN_TIMEOUT_SECS` (default `60s`), the child server is stopped (`STOPPED`), and `SIGINT` is sent to PID 1. If Modal has not terminated the container within 60s, a daemon timer runs bounded cleanup and calls `os._exit(70)`.
5. **Distributed Rank Watchdog (`singularity-server`)**:
   - `spawn_rank_watchdog` monitors `ZosScheduler::watchdog_tick()` every 1s (`SINGULARITY_STEP_STALL_TIMEOUT_SECS`, default `600s`). On rank failure or hung step (including V4.1 `ExternalStepGuard`), it poisons the scheduler (`HTTP 503` on `/ready` and `/v1/chat/completions`), cancels subscribers, frees KV blocks, and exits the server process with code `70`.

---

## 2. Policy Environment Reference

| Variable | Default | Bounds / Rule | Purpose |
| :--- | :---: | :--- | :--- |
| `SINGULARITY_MIN_CONTAINERS` | `0` | `0..MAX_CONTAINERS` | Minimum warm containers (`>0` requires `SINGULARITY_ALLOW_PINNED_WARM=1`) |
| `SINGULARITY_ALLOW_PINNED_WARM` | `0` | `0` or `1` | Explicit opt-in required to allow `SINGULARITY_MIN_CONTAINERS > 0` |
| `SINGULARITY_MAX_CONTAINERS` | `1` | `1..8` | Upper bound on concurrent container instances |
| `SINGULARITY_SCALEDOWN_WINDOW_SECS` | `300` | `10..900` | Idle seconds before Modal scales container `1 -> 0` |
| `SINGULARITY_REQUEST_TIMEOUT_SECS` | `900` | `5..3600` | Per-request gateway & SSE wall-clock deadline (`HTTP 504` on expiry) |
| `SINGULARITY_MAX_CONCURRENT_REQUESTS` | `32` | `1..256` | Per-container in-flight request admission cap (`HTTP 429` at limit) |
| `SINGULARITY_DRAIN_TIMEOUT_SECS` | `60` | `5..300` | Max seconds `@modal.exit` or watchdog waits for in-flight requests to finish |
| `SINGULARITY_MAX_WORKER_LIFETIME_SECS` | `21600` | `60..86400` (`0` disables) | Max serving uptime from `READY` before self-exit |
| `SINGULARITY_GPU_USD_PER_GPU_HOUR` | unset | `> 0` | Operator-supplied per-GPU hourly rate for spend estimation |
| `SINGULARITY_MAX_SPEND_USD` | unset | `> 0` | Max estimated container spend before watchdog self-exit |
| `SINGULARITY_STEP_STALL_TIMEOUT_SECS` | `600` | `>= 0` (`0` disables) | Rust `RankHealthMonitor` step stall timeout before poison + exit `70` |

---

## 3. Operational Runbooks

### Mode A: Scale-to-Zero On-Demand Deployment (Default — `$0.00/hr` Idle)
```bash
export SINGULARITY_MIN_CONTAINERS=0
export SINGULARITY_SCALEDOWN_WINDOW_SECS=300
export SINGULARITY_MAX_WORKER_LIFETIME_SECS=3600
python3 -m deploy.modal.deploy deploy
```
- **Verify zero containers immediately after deploy**:
  ```bash
  make audit
  modal container list --json   # must be []
  ```

### Mode B: Small-GPU Live Lifecycle & Scale-to-Zero Smoke Verification (`T4` / `2×H100`)
To verify gateway admission, `policy_env_snapshot()`, SSE `[DONE]` accounting, idle scale-down (`1 -> 0`), and lifetime-guard self-exit without provisioning 8×B200:
```bash
SINGULARITY_SCALEDOWN_WINDOW_SECS=60 SINGULARITY_MAX_WORKER_LIFETIME_SECS=120 \
  modal deploy deploy/modal/smoke_app.py
python3 -m deploy.modal.deploy measure-cold-start --url "<smoke-endpoint-url>" --warm-requests 2 --gpu-count 1
python3 -m deploy.modal.deploy status --url "<smoke-endpoint-url>"
python3 -m deploy.modal.deploy teardown
```

### Mode C: Zero-Compute Audit & Orphan Remediation
```bash
# Read-only provider audit (never pings the container HTTP endpoint, so idle timer is untouched)
make audit

# Automatically terminate any container exceeding max_worker_lifetime + startup + drain ceiling
python3 -m deploy.modal.deploy audit --terminate-orphans
```

### Mode D: Immediate Full Teardown (Stop All Singularity Apps & Containers Instantly)
```bash
make teardown
# Equivalent to: python3 -m deploy.modal.deploy teardown
modal container list --json   # confirm []
```
- Stops `singularity-v41-flash-1017`, `singularity-deepinfer-v41`, and `singularity-smoke` with `--yes` and terminates any running containers while keeping the `singularity-v41-flash-1017-weights` Volume intact.
