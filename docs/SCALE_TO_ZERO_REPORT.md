# Scale-to-Zero Commission Report (singularity3.md)

## Current status (updated 2026-10-31 13:25 local — Pass 6 deep audit complete)

**Overall: lifecycle, scale-to-zero, gateway streaming accounting, distributed rank-health isolation, CI gates, and workspace ergonomics are implemented and verified (offline across 71 Python + 11 script + full Rust workspace tests, and live on small GPUs `T4` and `2×H100`). The 8×B200 model path remains `NEEDS-LIVE` by explicit owner decision.**

| Area | State | Evidence |
|---|---|---|
| Python tests | **71 OK** (`deploy/modal`) + **11 OK** (`scripts`), `ruff` clean | `.venv/bin/python -W ignore -m unittest discover -s deploy/modal -p "test_*.py"` |
| Rust workspace | `cargo fmt --check`, `cargo clippy -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc`, `cargo test --workspace` all green | `make check` (local, offline) |
| Modal image build (incl. `cargo build --features cuda`) | passes after dependency pruning | `modal deploy deploy/modal/app.py` at 12:57, 0 containers, then stopped |
| Scale 0 → 1 on request | VERIFIED-LIVE (T4 and 2×H100, stub) | §0b |
| Idle scale-down to 0 | VERIFIED-LIVE (T4, stub) | §0b |
| Lifetime guard stops the container | VERIFIED-LIVE (T4 and 2×H100, stub) | §0b |
| Deploy-time policy reaches the container | bug found live, fixed, re-verified | §0b, `policy_env_snapshot()` |
| SSE `[DONE]` client disconnect & `tokens_out` accounting | bug found in live smoke logs, fixed & unit-tested | `app.py` `_ReleasingStreamingResponse`, `test_lifecycle.py` |
| `deepinfer_app.py` scale-to-zero + startup failure propagation | wired to `LifecycleTracker`/`LifecycleWatchdog`; re-raises on startup failure | `deepinfer_app.py`, `test_modal_pipeline.py` |
| `singularity-smoke` orphan classification & `deploy.py teardown` | `SMOKE_APP_NAME` in `KNOWN_APP_NAMES`; `deploy.py teardown` stops all apps & containers | `config.py`, `ops.py`, `deploy.py`, `test_lifecycle.py` |
| Rust poison propagation to `/ready`, `/health`, V4.1 & KV pin rollback | `/ready` returns 503 when poisoned; `ExternalStepGuard` covers V4.1; `pin_block` rejects unallocated blocks & rolls back | `zos_pipeline.rs`, `handlers.rs`, `block_manager.rs`, `rank_health_serve.rs` (7 tests) |
| Teardown / nothing billing | VERIFIED-LIVE: every singularity app `stopped`, `modal container list` = `[]` | provider-side |
| 8×B200 allocation, real cold start, TTFT/TPOT, utilisation, cost, NCCL multi-rank | **NEEDS-LIVE** (owner chose to accept small-GPU verification and defer paid 8×B200 retry) | §0 |
| `audit` showing deployed settings | `audit` shows local policy; `status --url` exposes enforced `limits` in `/__lifecycle/status` | §0b |
| Workspace cleanup & ergonomics | 22 unused Rust deps removed; root `Makefile`, `AGENTS.md`, and CI `python-modal` job added | `Makefile`, `AGENTS.md`, `.github/workflows/ci.yml` |

**History note.** Earlier versions of this paragraph said nothing had run live. That was true before the 12:25 run and is superseded by the table above.

No benchmark number in this report is invented; every value comes from a command run on the date shown.

Legend: **VERIFIED-OFFLINE** = proven by an automated test or command in this repo, no cloud. **NEEDS-LIVE** = cannot be proven offline; exact command given. **DECISION** = needs an owner decision, not more code.

## 0. Live Modal validation (2026-10-31, capped, user-approved) — PARTIAL, no GPU ever allocated

Timeline (local, UTC-6): 12:25 `audit` → clean (0 containers, 0 orphans). 12:25:50 `deploy` with `SINGULARITY_MAX_WORKER_LIFETIME_SECS=120`, `SINGULARITY_GPU_USD_PER_GPU_HOUR=6.25`, `SINGULARITY_MAX_SPEND_USD=10` → app deployed in 78.2 s (image build incl. `cargo build --features cuda` and `singularity --help` passed on Modal). 12:27:29 `measure-cold-start` against the web URL → **no container was ever provisioned**: `modal container list --json` was `[]` at 12:29 and 12:31, the request hung with no response, and app logs showed nothing beyond the image build. Client aborted ~12:31:34 (≈4 min). 12:32:31 `modal app stop` → app `stopped`; `modal container list` = `[]`; final `audit` clean.

**Not measured (honest):** cold-start latency, TTFT/TPOT, init time, lifetime-guard trip, idle scale-down of a real container, VRAM, cost. No GPU container existed, so the guard and idle window are not evidenced. Likely cause (unconfirmed): the 8×B200 request stayed queued for capacity/quota; I did not investigate further or work around it.

**VERIFIED-LIVE (evidence above):** pre-deploy zero state; image builds on Modal; deploy succeeds with `min_containers=0` and creates **0 containers** with no traffic (1.3 provider side); post-teardown provider state = 0 containers, app `stopped` (teardown part of 1.crit). **Every other NEEDS-LIVE row below remains NEEDS-LIVE.**

**Defect found & fixed:** `deploy.py stop`/`restart` ran `modal app stop` without `--yes`; non-interactively it prints `[y/N]` and does not stop the app. Fixed in `deploy/modal/deploy.py`; not re-run live.

Also noted: `audit` prints policy from the local env (lifetime 21600, spend null), not deployed values, so it cannot confirm deployed guard settings.


## 0b. Live small-GPU smoke validation (2026-10-31, T4 x1, stub inference, under USD 0.05) — VERIFIED-LIVE

`deploy/modal/smoke_app.py` runs the REAL gateway, `LifecycleTracker`, `LifecycleWatchdog` and scale-to-zero policy on a T4 in front of a stub server (no model). It proves lifecycle/scale-to-zero behaviour on Modal. It proves **nothing** about B200 capacity, performance, NCCL or the model.

| Check | Evidence | Result |
|---|---|---|
| Container starts on request (0 → 1) | `modal container list` `[]` before; container created on first request | VERIFIED-LIVE |
| Cold start vs warm (stub, T4) | first request TTFT 3.57 s (12:38:30) and 11.5 s (12:46:22, fresh image); warm TTFT median ≈ 240 ms | VERIFIED-LIVE (stub only) |
| Idle scale-down to 0 | `SCALEDOWN_WINDOW=60`; last request 12:38:31, `[SMOKE] container exiting` 12:40:19, `container list` `[]` at 12:41:27 | VERIFIED-LIVE |
| Lifetime guard stops the container | lifetime 60 s, idle window 300 s; READY 12:46:22, `guard tripped; exiting container` + `modal.exit` 12:47:22, container list `[]` at 12:48:14 (stopped at lifetime, not idle window) | VERIFIED-LIVE |
| Teardown | `modal app stop singularity-smoke --yes` → `stopped`, 0 tasks, `container list` `[]` | VERIFIED-LIVE |

**Defect found live and fixed:** the container re-imports the app and re-resolves the scaling policy, but the deploy-time `SINGULARITY_*` values were not passed into it, so in-container lifetime/spend/concurrency/drain guards silently ran on defaults (the 60 s lifetime did not trip for >2 min). Fix: `config.policy_env_snapshot()` bakes the whitelisted policy variables into the image (`app.py`, `smoke_app.py`); unit-tested; the live guard test above passed after the fix. This also affected the 8×B200 app.

**Multi-GPU intermediate run (user-chosen, 2x H100, stub, 2026-10-31 12:50):** container allocated on first request (cold TTFT 4.1 s, warm median 329 ms); READY 12:50:20, lifetime guard tripped 12:51:20 (60 s), container list `[]`; app stopped, 0 containers. VERIFIED-LIVE for multi-GPU allocation and the guard; still says nothing about the model or 8x B200.

Other changes: `measure-cold-start` now has `--timeout` (default 300 s); `validate-checkpoint` hints that it validates a local path only. The checkpoint Volume itself is populated (48 shards confirmed with `modal volume ls`).

Still NEEDS-LIVE (B200 only): 8×B200 allocation (the earlier attempt never got a container; cause likely capacity/quota, unconfirmed), real cold start/TTFT/TPOT, GPU utilisation, real cost, NCCL multi-rank behaviour. `audit` still shows local policy; to see the limits a live container actually enforces, use `deploy.py status --url <endpoint>` (the `/__lifecycle/status` snapshot now includes a `limits` block; querying counts as activity). Offline-tested only.


## 1. Original architectural problems (§9.1)

- `@app.cls` risked an always-warm 8×B200 container (`min_containers` pin / long idle window) with no finite bound or opt-in guard.
- No container-side lifecycle state, in-flight accounting, admission limits, bounded drain on `@modal.exit`, or orphan/zero verification tooling.
- The child `singularity serve` could die after startup while the container kept billing and accepting requests.
- Proxy `timeout` used a constant, not the policy; SSE streams had no wall-clock bound.
- A hung/dead rank in the multi-GPU group had no detection: a stuck collective would block the serve loop forever while 8 GPUs billed.

## 2. Record of changes (§9.2)
| Area | Files | Evidence |
|---|---|---|
| Policy knobs (min=0 default, pin needs opt-in, bounded idle/request/concurrency/drain/lifetime/spend) | `deploy/modal/config.py` | `test_scaling_policy.py`, `test_lifecycle.py` |
| State machine, admission, in-flight, deadlines, drain, bounded retry, partial-init cleanup | `deploy/modal/lifecycle.py` | `test_lifecycle.py` |
| **Lifetime clock starts at READY** (new this round); spend still from container start | `lifecycle.py`, `ops.py` | `test_lifetime_clock_starts_at_ready_*`, `test_spend_guard_still_counts_*` |
| Provider audit, orphan detection/termination, status, cold-start measurement | `deploy/modal/ops.py`, `deploy.py status/audit/measure-cold-start` | mocked tests; `audit`/`status` also ran once against the real account (read-only) |
| Gateway: admission, `/__lifecycle/status`, SSE deadline, dead-child 503, release-on-close, server-generated request id | `deploy/modal/app.py` | `GatewayTests` against a real ASGI app + local fake upstream |
| **Never-iterated stream releases its slot immediately** (new this round) | `app.py` `_ReleasingStreamingResponse` | `test_never_iterated_stream_releases_slot_without_waiting_for_deadline` (negative-checked: fails with the fix disabled) |
| Container stop on lifetime/spend/dead child (`LifecycleWatchdog`) | `lifecycle.py`, `app.py` | fake-clock + thread tests |
| **Hard-exit fallback: bounded best-effort cleanup, then exit code 70 (was `os._exit(0)`)** (new this round) | `app.py` `_hard_exit`, `_exit_container` | 3 tests (order, failing cleanup, hanging cleanup, timer wiring) |
| **Rank health wired into the real serve loop** (new this round) | `singularity-scheduler/src/zos_pipeline.rs`, `rank_health.rs`, `dp_consensus.rs`; `singularity-server/src/app.rs` | `singularity-tests/tests/rank_health_serve.rs` (4 integration tests) |
| Rust dedup: 21 byte-identical `fn error()` copies → one `model_error` | `singularity-model/src/*.rs`, `lib.rs` | `cargo check/clippy/test --workspace` green |
| Untracked committed `scripts/__pycache__/*.pyc` | git index | `git ls-files` |
| Docs | this file, `CHANGELOG.md`, `deploy/modal/README.md`, `PLAYBOOK-004` | reviewed (see §10) |

### Rank health in the serve loop (how it works)
- `ZosScheduler::with_rank_health(monitor, local_rank, deadline)` and `with_dp_reduce(f)` attach a `RankHealthMonitor` and the NCCL-boundary all-reduce.
- Every DP vote runs through `run_collective_with_deadline` (bounded wait). A timeout/rank failure **poisons** the scheduler: sticky `fatal`, every subscriber receives `Cancelled`, all queued/running KV blocks are freed, `submit_sequence` and later steps fail fast (the dead communicator is never retried).
- `step()` first checks the monitor (sticky explicit failures, per-step stalls); the monitor's stall clock is refreshed at step start and while idle, so idle time is never mistaken for a stall.
- `ZosScheduler::watchdog_tick()` is called from an independent thread: it can poison the scheduler while the stepping thread is stuck inside a hung step (the stepper cannot report its own hang).
- `singularity-server/src/app.rs` attaches a monitor (world size 1, local rank 0) and spawns a `rank-watchdog` thread (1 s poll, `Weak` ref). On poison it logs, waits 2 s so clients receive their cancel, then `exit(70)` so the container stops billing. Stall timeout = `SINGULARITY_STEP_STALL_TIMEOUT_SECS` (default 600, `0` disables).
- **Limit:** the server runs a single-rank scheduler with a local-only reduce. The monitor therefore protects against a hung/failed *local* step today. Remote-rank heartbeats/failures (`report_progress` / `report_failure`) and a real NCCL all-reduce closure have no producer in this workspace (`CollectiveCoordinator` is an in-process simulation) — the hooks exist and are integration-tested with injected failures, not with real NCCL.

## 3. Final architecture and lifecycle (§9.3, §9.4)
*zero containers* → request arrives → Modal provisions 8×B200 → `@enter`: STARTING→INITIALIZING (checkpoint check ≤3 bounded retries, then supervisor 17-step start, no in-process GPU retry) → READY (only now does the ASGI app exist, so nothing is accepted before the cluster is ready; **serving-lifetime clock starts here**) → serve (POSTs admitted/counted until response/stream closes or is abandoned) → idle for `scaledown_window` (default 300 s, max 900 s) → Modal calls `@exit`: DRAINING (reject new), wait ≤ `drain_timeout` (60 s), kill child, STOPPED → container released → zero. Init failure → child stopped, FAILED, exception re-raised.

Guards: lifetime (serving time) / spend (from container start, because billing starts then) → `LifecycleWatchdog`: stop admission → bounded drain → stop child → STOPPED → SIGINT; if Modal has not killed the container after 60 s → bounded cleanup → `exit(70)`. Dead child → same path. Inside the process: rank-watchdog → `exit(70)` (above).

Design limits: the tracker governs one container. "Zero containers" is the absence of a process and cannot be observed from inside it. Pre-READY time is bounded only by Modal's `startup_timeout` (an orphan ceiling in `audit` accounts for it). Modal workspace spend caps remain a dashboard setting.

## Background-ping audit (requirement §1.6)
- Gateway: no lifespan tasks, timers or upstream pings; reaping, liveness and GPU sampling are pull-based.
- Python threads: `LifecycleWatchdog` (5 s, local counters + `Popen.poll()` only; no network/GPU).
- Rust: `rank-watchdog` thread (1 s) reads only in-process atomics/monitor state — no GPU, network or request activity, so it cannot count as Modal input.
- `check_service_health` runs only during startup polling and on operator request.
- **Residual risk:** any external monitor hitting the public endpoint resets Modal's idle window. Use `deploy.py audit` (provider API only).
- Not audited: other background threads inside `singularity-server` beyond the one added here.

## 4. Demonstrated cost-control behaviour (§9.5)
- **Measured (real account, read-only, earlier round):** `modal container list --json` → `[]`; `modal app list --json` showed `singularity-v41-flash-1017` deployed with `Tasks: 0`; `deploy.py audit` reported `zero_inference_compute: true`. This is a snapshot of the **previously deployed (old) code**, not of this implementation.
- **Offline-tested:** admit/reject, caps under 50-thread contention, stuck-job reaping, lifetime and spend limits, bounded drain, bounded retry, partial-init cleanup, orphan classification, termination only on explicit flag, missing credentials ⇒ "unverifiable" (never "zero").

## 5. Benchmarks vs original (§9.6)
**None measured.** Tooling exists (`measure-cold-start`, `/__lifecycle/status?gpu=1`, `benchmark`) but no B200 run was performed. Cold-start latency, init time, TTFT, TPOT, GPU utilisation, peak VRAM, provisioning/teardown latency and cost-per-execution are **unmeasured**. The code never assumes a GPU price (`SINGULARITY_GPU_USD_PER_GPU_HOUR` is user-supplied; spend is labelled an estimate). Older docs quote `~40–60 s` cold start, `TTFT P50 ~49 ms` and `$49.99/h`; those figures predate this commission and were **not re-measured here** (PLAYBOOK-004 now says so).

## 6. Testing and failure-injection results (§9.7)
Commands run this round (all offline):
- `.venv/bin/python -m unittest discover -s deploy/modal -p "test_*.py"` → **66 tests, OK** (includes gateway tests; requires `python3 -m venv .venv && .venv/bin/pip install fastapi httpx`).
- `cargo check --workspace --tests`, `cargo clippy --workspace --all-targets` (no warnings), `cargo test --workspace` → all green (scheduler 12 unit tests + 4 new integration tests; the real-inference suite also passes).
- `ruff check deploy/modal` → clean.
- Secrets scan: no scanner binary installed (gitleaks/trufflehog/detect-secrets absent). Ran a regex scan (HF/AWS/GitHub/Slack/Modal token shapes, private keys, `key|secret|token|password = <24+ chars>`) over all tracked + untracked files and `git log -G` over history → **no matches**. A regex scan is weaker than a real scanner.
- Injected failures: init failure, provider 503 (3 bounded attempts), garbage provider JSON, duplicate/concurrent admission, stuck request, drain with stuck request, upstream stalled before/after headers (504, slot released), spoofed `x-request-id`, lifetime/spend trip idle and busy, dead child, `stop_child` raising, **client vanishing before the first stream chunk (both Starlette disconnect paths)**, **hard-exit with failing/hanging cleanup**, **hung DP collective (bounded, subscribers cancelled, blocks freed, retry refused)**, **explicit dead rank detected by watchdog while idle**, **hung stepping thread released by watchdog**, idle > stall timeout not a false positive, panicking collective, sticky rank failure, silent rank.

## 7. Remaining limitations and known risks (§9.8)
- Modal wiring (`@enter`/`@exit`/watchdog/`_exit_container`) was never executed against Modal (`modal` not importable here → `app` is `None`). Redeploy is required for any of it to take effect.
- Provider CLI JSON field names for `container list` were only observed as an empty list.
- SIGINT self-exit assumes Modal treats it as graceful shutdown; behaviour with queued input is unobserved.
- Real NCCL hang/abort is not testable here; a blocked NCCL call is not cancellable from safe Rust (contract: detect → exit process). The `std::process::exit(70)` path in the Rust watchdog is **not covered by an automated test** (exiting would kill the test runner); only the detection that triggers it is.
- The Rust server's monitor is single-rank (see above). Default stall timeout of 600 s means a hung step is detected after ≤ ~10 min plus a ≤1 s poll; lower it for faster detection if single steps are known to be short.
- The slot-release for abandoned streams relies on the ASGI server finishing the response call; a server that never finishes the call at all would still fall back to `reap_expired` at the per-request deadline.
- Not done: checkpointing; dynamic batching/adaptive concurrency (fixed admission cap only).
- Legacy `singularity-model` modules (`attention.rs`, `engram.rs`, `moe.rs`) overlap in concept with `v41_*` but are used by `transformer.rs` and re-exported from the crate root; removing them is an API decision, not a safe mechanical cleanup (see §9).

## 8. Documentation and navigation (§9.9)
This report; `CHANGELOG.md` (Unreleased); `deploy/modal/README.md` (scale-to-zero operations); `knowledge-base/playbooks/PLAYBOOK-004`; `deploy/modal/deploy.py` subcommands `status | audit | measure-cold-start`.

## 9. Outstanding work needing decisions (§9.10)
1. Accept ≈cold-start on every first request (no warm pool) vs an opt-in time-boxed warm window.
2. Set `SINGULARITY_GPU_USD_PER_GPU_HOUR` + `SINGULARITY_MAX_SPEND_USD`; set the Modal workspace spend cap.
3. Approve redeploy and a live cold-start / idle / guard run (cost: several minutes of 8×B200 plus model load).
4. Decide the fate of the legacy `singularity-model` `attention/engram/moe` modules (remove vs keep as the host-development path).
5. Provide a real multi-rank transport (NCCL all-reduce closure + remote heartbeat producer) if multi-process DP is to be protected end to end.

---

# 10. Requirement → evidence checklist (singularity3.md, every section)

## §1 Primary mandate
| # | Requirement | Status | Evidence / exact command |
|---|---|---|---|
| 1.1 | No requests/workloads ⇒ complete scale-down | **NEEDS-LIVE** (policy default `min_containers=0`, window ≤900 s is VERIFIED-OFFLINE) | `python3 -m deploy.modal.deploy deploy && python3 -m deploy.modal.deploy audit` (expect zero) → send 1 request → `sleep 330 && python3 -m deploy.modal.deploy audit` → `modal container list --json` |
| 1.2 | Warm workers not provisioned indefinitely | VERIFIED-OFFLINE | `config.py` pin requires `SINGULARITY_ALLOW_PINNED_WARM=1`, bounded window; `test_scaling_policy.py` |
| 1.3 | Always-on infrastructure eliminated | VERIFIED-OFFLINE (config) / NEEDS-LIVE (provider) | `deploy.py status` (shows `min_containers`); then `modal app list --json` |
| 1.4 | Configurable idle shutdown | VERIFIED-OFFLINE (config); actual shutdown NEEDS-LIVE | `SINGULARITY_SCALEDOWN_WINDOW_SECS` tests; live: same as 1.1 |
| 1.5 | GPU allocations released on finish | **NEEDS-LIVE** | as 1.1; `modal container list --json` must be `[]` |
| 1.6 | No background process keeps compute busy | VERIFIED-OFFLINE (audit above, code review of both watchdogs); Rust server internals only partly reviewed | see "Background-ping audit" |
| 1.7 | Queued requests wake the system from zero | **NEEDS-LIVE** (Modal platform behaviour) | `python3 -m deploy.modal.deploy measure-cold-start --url "$SINGULARITY_MODAL_URL"` after `audit` shows zero |
| 1.8 | Distinguish active/queued/warm/idle | VERIFIED-OFFLINE | `LifecycleTracker.snapshot()` (`in_flight`, `idle_secs`, state); tests |
| 1.9 | Scale-down accounts for in-flight/prefill/KV/distributed | Process-level VERIFIED-OFFLINE (drain, abandoned-stream release); per-rank KV: scheduler poison frees blocks, VERIFIED-OFFLINE with mock backend; real KV/NCCL NEEDS-LIVE | `rank_health_serve.rs`, `test_lifecycle.py`; live: `modal app stop` mid-request |
| 1.10 | Graceful, deterministic, race-resistant shutdown | VERIFIED-OFFLINE (bounded drain, idempotent release, thread contention test, hard-exit with cleanup) / NEEDS-LIVE for Modal `@exit` | tests; live: guard test below |
| 1.c1 | Idle shutdown threshold | VERIFIED-OFFLINE | config tests |
| 1.c2 | Max worker lifetime | VERIFIED-OFFLINE (now measured from READY) | `test_lifetime_clock_starts_at_ready_not_at_container_start`; live container release NEEDS-LIVE: `SINGULARITY_MAX_WORKER_LIFETIME_SECS=120 python3 -m deploy.modal.deploy deploy`, send 1 request, wait ~(init + 200 s), `deploy.py audit`, expect container gone + log "guard tripped" |
| 1.c3 | Max provisioned GPU capacity | VERIFIED-OFFLINE (`max_containers` bound) | `test_scaling_policy.py` |
| 1.c4 | Max concurrent jobs | VERIFIED-OFFLINE | admission tests, 50-thread test |
| 1.c5 | Per-job timeouts | VERIFIED-OFFLINE | gateway 504 tests, `reap_expired` |
| 1.c6 | Request-level resource accounting | VERIFIED-OFFLINE | `busy_seconds`, `gpu_seconds_*` in snapshot |
| 1.c7 | Hard spend/resource limits | VERIFIED-OFFLINE (per-container estimate; needs user price) / NEEDS-LIVE for provider cap | set Modal workspace spend cap in dashboard (manual) |
| 1.c8 | Cleanup of abandoned workers | VERIFIED-OFFLINE (logic) / NEEDS-LIVE | `deploy.py audit --terminate-orphans` (only if audit reports orphans) |
| 1.c9 | Detect/terminate orphaned allocations | VERIFIED-OFFLINE (mocked provider); real `container list` schema unobserved | `python3 -m deploy.modal.deploy audit` against a running container |
| 1.c10 | No infinite retry loops | VERIFIED-OFFLINE | `retry_bounded` has no unbounded mode; tests |
| 1.c11 | No runaway provisioning | VERIFIED-OFFLINE (`max_containers`, init not retried in-process) | config tests |
| 1.c12 | Explicit lifecycle state tracking | VERIFIED-OFFLINE | state machine + illegal-transition tests |
| 1.c13 | Cost/utilisation observability | VERIFIED-OFFLINE (endpoint/fields) / NEEDS-LIVE (real values) | `curl -s "$SINGULARITY_MODAL_URL/__lifecycle/status?gpu=1"` |
| 1.crit | Zero verified via provider-side state | **NEEDS-LIVE** (tooling VERIFIED-OFFLINE; old-code snapshot only) | commands in 1.1 plus `modal container list --json; modal app list --json` |
| 1.alt | If something can't scale to zero: why, cost, alternatives | VERIFIED-OFFLINE (doc): Modal Volume storage is the only residual and is not GPU-billed; no pinned capacity by default. Storage cost not quantified (no price assumed) | §3, PLAYBOOK-004 |

## §2 Industry-standard architectures
| Requirement | Status | Evidence |
|---|---|---|
| Investigate/adopt scale-to-zero serverless, queue-driven provisioning, graceful draining, bounded retries, failure isolation, fault tolerance | VERIFIED-OFFLINE (adopted: Modal scale-to-zero + queue wake, drain, bounded retry, poison/isolation). Not adopted: dynamic batching, adaptive concurrency, resource-aware placement (documented in §7) | code + §3 |
| No partial cluster / incomplete init / inconsistent worker state | VERIFIED-OFFLINE (ASGI app built only after READY; `start_with_cleanup`) / real 8-rank init NEEDS-LIVE | `test_lifecycle.py`; live: `deploy`, then kill during `@enter` |
| No KV corruption / duplicate execution | Duplicate admission VERIFIED-OFFLINE (server-generated ids); KV corruption: scheduler KV logic unchanged, regression suites green; real KV NEEDS-LIVE | `cargo test --workspace` |
| No orphaned GPU allocations / shutdown deadlocks / stranded requests | VERIFIED-OFFLINE (bounded drain, hard-exit fallback, abandoned stream release) / NEEDS-LIVE | tests |

## §3 Cold-start / warm-state
| Requirement | Status | Evidence |
|---|---|---|
| Scale from zero on demand, provision, init & validate, accept only when ready | VERIFIED-OFFLINE (logic) / NEEDS-LIVE (real) | `start_with_cleanup`; `measure-cold-start` |
| Warm capacity configurable, finite, default cost-first, no hidden minimum | VERIFIED-OFFLINE | `config.py`, `test_scaling_policy.py` |
| Drain, terminate idle, return to zero | VERIFIED-OFFLINE (logic) / NEEDS-LIVE | as 1.1 |

## §4 Distributed reliability
| Requirement | Status | Evidence |
|---|---|---|
| Explicit lifecycle state machines | VERIFIED-OFFLINE | `lifecycle.py` |
| Idempotent provisioning/teardown | VERIFIED-OFFLINE (`stop()` idempotent, `release` idempotent, poison idempotent) | tests |
| Distributed health checks / failure detection | VERIFIED-OFFLINE with injected failures through the real scheduler loop; real NCCL/multi-process NEEDS-LIVE | `cargo test -p singularity-tests --test rank_health_serve` |
| Deadlock detection | VERIFIED-OFFLINE (bounded collective + in-step watchdog) | `hung_collective_*`, `watchdog_releases_clients_*` |
| Bounded retries with backoff | VERIFIED-OFFLINE | `retry_bounded` tests |
| Failure isolation | VERIFIED-OFFLINE (poison: cancel clients, free blocks, refuse work) | `rank_health_serve.rs` |
| Checkpointing | **Not done** (DECISION) | §7 |
| Recovery from partial init | VERIFIED-OFFLINE | `start_with_cleanup` tests |
| Deterministic cleanup | VERIFIED-OFFLINE | drain / hard-exit tests |
| Structured execution tracing | PARTIAL: lifecycle `history`, step log lines; no distributed tracing | `snapshot()["history"]` |
| No infinite loops / unbounded retries / uncontrolled acquisition | VERIFIED-OFFLINE | bounded everywhere; both watchdogs exit after tripping |

## §5 Workspace hardening
| Requirement | Status | Evidence |
|---|---|---|
| Identify/remove duplicated functionality | PARTIAL, VERIFIED-OFFLINE: 21 identical `error()` helpers consolidated; `__pycache__` untracked. Legacy model module overlap left deliberately (DECISION 4) | `cargo clippy/test --workspace` green; diff in `singularity-model/src` |
| Obsolete/abandoned experiments, conflicting configs | Reviewed; `deepinfer_app.py` (baseline comparison app) kept intentionally — `teardown` references it. No conflicting scaling configs found after `config.py` consolidation | grep of `min_containers`/`scaledown` |
| Improve error handling, logging, tests, docs, navigation | VERIFIED-OFFLINE for the touched areas (new tests, actionable errors, docs) | this report |
| Do not silently remove working implementations | VERIFIED-OFFLINE (nothing deleted but identical copies) | git diff |

## §6 Passes
| Pass | Status |
|---|---|
| 1 Discovery, 2 Architecture, 3 Implementation | Done (previous rounds) |
| 4 Failure-mode audit | Done offline; this round found and fixed: hard-exit returning success code, un-iterated stream slot leak, lifetime guard eating init time, unwired rank health |
| 5 Independent verification incl. provider-side teardown | **NEEDS-LIVE** (offline re-audit done; provider-side not) |
| 6 Further hardening | Diminishing returns offline; remaining items need live access or an architectural decision (§9) |

## §7 Testing and acceptance criteria
| Scenario | Status | Evidence / exact command |
|---|---|---|
| No requests → zero after idle timeout | **NEEDS-LIVE** | `sleep 330 && python3 -m deploy.modal.deploy audit` then `modal container list --json` |
| First request after zero → auto-provision | **NEEDS-LIVE** | `python3 -m deploy.modal.deploy measure-cold-start --url "$SINGULARITY_MODAL_URL"` |
| Model init → all workers ready | VERIFIED-OFFLINE (ordering logic) / NEEDS-LIVE (8 ranks) | `curl -s "$SINGULARITY_MODAL_URL/ready"` after cold start |
| Active generation keeps resources | VERIFIED-OFFLINE (in-flight blocks scale-down/drain) / NEEDS-LIVE | long request + `audit` mid-flight |
| Multiple concurrent requests safe | VERIFIED-OFFLINE (50-thread contention, 429 at cap) | `test_lifecycle.py` |
| Request completion → eligible for scale-down | VERIFIED-OFFLINE | tracker idle tests; live: `audit` after window |
| Queue empty → idle shutdown begins | VERIFIED-OFFLINE (state) / NEEDS-LIVE (Modal window) | as above |
| Worker failure detected & isolated | VERIFIED-OFFLINE (dead child → 503/FAILED/exit; dead/hung rank → poison, via injected failures) / real GPU fault NEEDS-LIVE | `rank_health_serve.rs`, `test_dead_worker_rejected`; live: `modal container exec <id> pkill -9 singularity` then `audit` |
| Partial provisioning cleaned up | VERIFIED-OFFLINE / NEEDS-LIVE | `start_with_cleanup`; live: kill during `@enter` |
| Infinite/stuck job bounded & terminated | VERIFIED-OFFLINE | gateway 504 tests, `reap_expired`, hung-step watchdog |
| Shutdown during activity safe | VERIFIED-OFFLINE (bounded drain/abandon) / NEEDS-LIVE | `modal app stop singularity-v41-flash-1017` mid-request |
| Repeated idle periods → no orphans | **NEEDS-LIVE** | loop: request → `sleep 330` → `audit`, repeat 3×; `audit` must stay `orphans: []` |
| Repeated cold starts reproducible | **NEEDS-LIVE** | same loop with `measure-cold-start` |

Measurements required by §7 — **all NEEDS-LIVE, none measured, none estimated here:** cold-start latency (`measure-cold-start`), model init time (`/__lifecycle/status` history: INITIALIZING→READY), TTFT/TPOT (`deploy.py benchmark`, `measure-cold-start`), GPU utilisation and peak VRAM (`/__lifecycle/status?gpu=1`), idle resource duration (`snapshot idle_secs` + `audit`), provisioning/teardown latency (Modal logs + `audit` timestamps), cost per execution and cost in zero-demand periods (needs `SINGULARITY_GPU_USD_PER_GPU_HOUR`; zero-demand cost should be `$0` GPU, to be confirmed in the Modal billing dashboard).

Consolidated live runbook (costs real 8×B200 time; **not run**):
```
export SINGULARITY_GPU_USD_PER_GPU_HOUR=<price> SINGULARITY_MAX_SPEND_USD=<cap> SINGULARITY_MAX_WORKER_LIFETIME_SECS=600
python3 -m deploy.modal.deploy deploy
python3 -m deploy.modal.deploy audit                       # expect zero_inference_compute: true before first request
python3 -m deploy.modal.deploy measure-cold-start --url "$SINGULARITY_MODAL_URL"
curl -s "$SINGULARITY_MODAL_URL/__lifecycle/status?gpu=1"  # state, in_flight, serving_uptime_secs, peak VRAM
sleep 330 && python3 -m deploy.modal.deploy audit          # idle window 300 s -> expect zero, no orphans
# guard test: redeploy with SINGULARITY_MAX_WORKER_LIFETIME_SECS=120, send one request, wait (init + ~200 s):
python3 -m deploy.modal.deploy audit                       # expect container gone; logs show "guard tripped"
modal container list --json; modal app list --json         # independent provider-side confirmation
python3 -m deploy.modal.deploy audit --terminate-orphans   # only if audit reports orphans
```

## §8 Implementation principles
| Principle | Status |
|---|---|
| Fixes not monkey patches; reliability; cost efficiency; reproducibility; clear architecture; operational control; established patterns | VERIFIED-OFFLINE by review: root causes patched (e.g. un-started async generator cleanup fixed in the response object rather than by waiting for a deadline) |
| Preserve existing functionality, no silent removals | VERIFIED-OFFLINE: full `cargo test --workspace` (including real-inference suites) and Python suite green; only byte-identical duplicate helpers removed |
| Incremental, with a record of decisions | This report + CHANGELOG |
| Document infrastructure limitations instead of claiming success | Done (§7, NEEDS-LIVE labels) |

## §9 Final deliverable
| Item | Where |
|---|---|
| 1 Original problems | §1 |
| 2 Record of changes | §2 |
| 3 Final architecture | §3 |
| 4 Provisioning & shutdown lifecycle | §3 |
| 5 Demonstrated cost-control behaviour | §4 (offline + one old-code provider snapshot) |
| 6 Benchmarks vs original | §5 — **none measured** |
| 7 Testing & failure-injection | §6 |
| 8 Limitations & risks | §7 |
| 9 Docs & navigation | §8 |
| 10 Decisions needed | §9 |

## Doc accuracy review (this round)
- `CHANGELOG.md`: updated to describe rank-health wiring, ready-time lifetime, hard-exit code 70, abandoned-stream release, `model_error` consolidation; the previous entry claimed rank health was only primitives — corrected.
- `deploy/modal/README.md`: added the `SINGULARITY_STEP_STALL_TIMEOUT_SECS` knob and ready-time semantics.
- `PLAYBOOK-004`: pre-existing cost/latency numbers (`$49.99/h`, `~40–60 s`, `49 ms`) are now flagged as unverified legacy figures; guard semantics added.
- Known inaccuracy fixed in this file: an earlier version said "unittest 60 tests" and "cargo test -p singularity-scheduler 12 pass"; current numbers are 66 Python tests and the full workspace.

## Appendix: unused Rust dependencies removed

A text scan found 22 dependencies with no source reference (cache: tracing, thiserror, serde; cli: singularity-grammar; core: tracing, uuid; cuda: thiserror, serde_json; grammar: serde, serde_json, tracing, thiserror; model: singularity-cache, parking_lot, tracing, thiserror; server: tower, tracing-subscriber, sha2, thiserror; tests: tracing, uuid). Each was removed one at a time and kept only if `cargo check --workspace --all-targets --offline` still passed. Afterwards: `cargo check --workspace --all-targets --all-features --offline` reports no errors and `cargo test --workspace --offline` passes (all suites ok). VERIFIED-OFFLINE. The CUDA-feature image build on Modal (`cargo build --features cuda`) was not re-run after the removal; run `deploy.py build-image` to confirm.
