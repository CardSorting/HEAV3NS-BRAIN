# Changelog

All notable changes to **HEAV3NS-BRAIN (Project SINGULARITY)** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned (Horizon 0 & Horizon 1 Rolling Sprints)
- **H0-01**: Adaptive admission control modulating `LeakyTokenBucket` directly from live `itl_jitter_ratio`.
- **H0-02**: Parallel multi-threaded 48-shard CSF header verification (`< 1.5s` cold-volume check).
- **H0-03**: Automated benchmark regression diffing in `deploy.py benchmark`.
- **H0-04**: Live 8× NVIDIA B200 (`B200:8`) full-model scale-to-zero cold-start, TTFT/TPOT, and lifetime-guard verification (gated on cluster capacity and owner spend approval).
- **H1-01..H1-04**: In-kernel GPU CSF scale decompression (`tcgen05.mma`), FlashMLA-4 Split-KV persistent megakernel, fused MoE router + NVLink 5 `EP=8` dispatch, and multi-process NCCL transport with remote rank heartbeats.

---

## [0.3.0] — 2026-10-31

### True Scale-to-Zero Serverless Lifecycle, Distributed Rank-Health Isolation & Pass 1–6 Workspace Hardening (`singularity3.md`)

#### Added
- **Scale-to-Zero Policy & Deploy-Time Image Snapshot ([`deploy/modal/config.py`](deploy/modal/config.py), [`ADR-006`](knowledge-base/adrs/ADR-006-scale-to-zero-lifecycle-and-cost-governance.md))**:
  - `ScalingPolicy` and `resolve_scaling_policy()` enforcing `min_containers=0` by default across all Modal apps (`singularity-v41-flash-1017`, `singularity-deepinfer-v41`, `singularity-smoke`). Setting `SINGULARITY_MIN_CONTAINERS > 0` is rejected unless `SINGULARITY_ALLOW_PINNED_WARM=1` is explicitly set.
  - Bounded limits on `max_containers` (`1..8`), `scaledown_window_secs` (`10..900`), `request_timeout_secs` (`5..3600`), `max_concurrent_requests` (`1..256`), `drain_timeout_secs` (`5..300`), `max_worker_lifetime_secs` (`60..86400` or `0`), and `max_spend_usd` (requires user-supplied `SINGULARITY_GPU_USD_PER_GPU_HOUR`).
  - `policy_env_snapshot()` baking whitelisted `SINGULARITY_*` scaling/guard variables into all Modal container images at deploy time.
- **Container Lifecycle State Machine & Self-Enforcing Watchdog ([`deploy/modal/lifecycle.py`](deploy/modal/lifecycle.py), [`deploy/modal/app.py`](deploy/modal/app.py), [`deploy/modal/deepinfer_app.py`](deploy/modal/deepinfer_app.py))**:
  - 6-state `LifecycleTracker` (`STARTING → INITIALIZING → READY → DRAINING → STOPPED / FAILED`), in-flight admission control (`HTTP 429` / `HTTP 503` with `Retry-After`), per-request deadlines (`HTTP 504` + `reap_expired`), bounded graceful `drain()`, `retry_bounded`, and `start_with_cleanup`.
  - Dual-clock accounting: `max_worker_lifetime_secs` starts at `READY` (`serving_uptime_secs`), while `estimated_spend_usd` counts from container boot (`uptime_secs`).
  - `LifecycleWatchdog` (5s pull-only poll; zero network/GPU activity so it never resets Modal's idle window) stops admission, drains in-flight requests, stops the child server, and sends `SIGINT` on lifetime/spend/dead-child trips, backed by a 60s hard-exit timer (`os._exit(70)`).
  - Local `GET /__lifecycle/status` (and `?gpu=1` on-demand `nvidia-smi` sampling) exposing live state, `in_flight`, `serving_uptime_secs`, `tokens_out`, and enforced `limits`.
- **Provider-Side Zero-Compute Audit, Cold-Start Measurement & Small-GPU Smoke App ([`deploy/modal/ops.py`](deploy/modal/ops.py), [`deploy/modal/smoke_app.py`](deploy/modal/smoke_app.py), [`deploy/modal/deploy.py`](deploy/modal/deploy.py))**:
  - `ops.audit_zero_compute()` querying `modal container list --json` and `modal app list --json` across `KNOWN_APP_NAMES = (APP_NAME, DEEPINFER_APP_NAME, SMOKE_APP_NAME)` with opt-in `--terminate-orphans`.
  - `deploy.py` subcommands `status`, `audit`, `measure-cold-start` (`--timeout`, `--gpu-count`), and `teardown` (plus `--all` and `--app-name` on `stop`).
  - `deploy/modal/smoke_app.py`: small-GPU (`T4`, `2×H100`) deployment of the real gateway, `LifecycleTracker`, and `LifecycleWatchdog` verified live on Modal (`0 → 1` cold start, warm requests, idle scale-down to `0`, and 60s lifetime-guard self-exit).
- **Distributed Rank-Health Monitoring, Bounded Collectives & `ExternalStepGuard` ([`singularity-scheduler/src/rank_health.rs`](singularity-scheduler/src/rank_health.rs), [`singularity-scheduler/src/zos_pipeline.rs`](singularity-scheduler/src/zos_pipeline.rs), [`singularity-server/src/app.rs`](singularity-server/src/app.rs), [`ADR-007`](knowledge-base/adrs/ADR-007-distributed-rank-health-and-poison-isolation.md))**:
  - `RankHealthMonitor`, `run_collective_with_deadline`, and `DpConsensusEngine::decide_step_type_checked` / `validate_two_phase_checked`.
  - Sticky `ZosScheduler::poison()` / `poison_with()` isolation: cancels all subscribers (`SequenceOutputEvent::Cancelled`), clears KV caches, frees all queued/running/preempted KV blocks, and refuses new work without retrying a dead communicator.
  - `ExternalStepGuard` (`ZosScheduler::enter_external_step()` backed by an `AtomicUsize` active-step counter) wrapping synchronous V4.1 forward passes inside `tokio::task::spawn_blocking`.
  - `spawn_rank_watchdog` in `singularity-server` polling `watchdog_tick()` every 1s (`SINGULARITY_STEP_STALL_TIMEOUT_SECS`, default `600s`) and exiting with `EXIT_GROUP_FAILED = 70` after poisoning.
- **Workspace Developer & Agent Ergonomics ([`Makefile`](Makefile), [`AGENTS.md`](AGENTS.md), [`.github/workflows/ci.yml`](.github/workflows/ci.yml), [`docs/SCALE_TO_ZERO_REPORT.md`](docs/SCALE_TO_ZERO_REPORT.md))**:
  - Root `Makefile` (`make help`, `make check`, `make test`, `make lint`, `make fmt`, `make doc`, `make status`, `make audit`, `make teardown`, `make smoke-deploy`).
  - Root `AGENTS.md` documenting workspace component ownership, high-throughput parent-agent vs. subagent execution flow, sandbox allowlist configuration, and scale-to-zero governance.
  - Added `python-modal` CI job in `.github/workflows/ci.yml` running `ruff check deploy/modal scripts`, all 71 tests in `deploy/modal/`, and `scripts/` unit tests.
  - Added [`ADR-006`](knowledge-base/adrs/ADR-006-scale-to-zero-lifecycle-and-cost-governance.md), [`ADR-007`](knowledge-base/adrs/ADR-007-distributed-rank-health-and-poison-isolation.md), [`PM-004`](knowledge-base/post-mortems/PM-004-deploy-time-policy-env-drift-in-container.md), and [`PM-005`](knowledge-base/post-mortems/PM-005-sse-done-premature-client-disconnect-accounting.md) to [`knowledge-base/`](knowledge-base/README.md).

#### Fixed
- **Deploy-Time Policy Env Drift in Containers ([`PM-004`](knowledge-base/post-mortems/PM-004-deploy-time-policy-env-drift-in-container.md), [`deploy/modal/config.py`](deploy/modal/config.py))**: Fixed container re-imports reverting to default 6h lifetime / unlimited spend by baking `policy_env_snapshot()` into the Modal image `.env(...)`.
- **SSE `[DONE]` Premature Client-Disconnect & Token Accounting ([`PM-005`](knowledge-base/post-mortems/PM-005-sse-done-premature-client-disconnect-accounting.md), [`deploy/modal/app.py`](deploy/modal/app.py))**: `_ReleasingStreamingResponse.__call__` now awaits `self.body_iterator.aclose()` before `self._on_finish()`, and `stream_chunks()` tracks `saw_done` and SSE token delta chunks so standard OpenAI clients closing the socket immediately after `data: [DONE]` record `ok=True` (`completed += 1`) and accurate `tokens_out` (and non-streaming JSON responses record `usage.completion_tokens`) without `GeneratorExit` cleanup errors.
- **Un-Iterated Stream Slot Release ([`deploy/modal/app.py`](deploy/modal/app.py))**: Streaming responses where the client disconnects before the first chunk now release their admission slot and close the upstream response immediately instead of waiting for the per-request deadline.
- **`DeepInferReferenceService` Startup Failure Isolation ([`deploy/modal/deepinfer_app.py`](deploy/modal/deepinfer_app.py))**: Fixed silent `except Exception` in `startup()` that left an unready 8×B200 container alive and billing; now transitions to `FAILED → STOPPED` and re-raises.
- **Non-Interactive `modal app stop` Hang ([`deploy/modal/deploy.py`](deploy/modal/deploy.py))**: Added `--yes` to `modal app stop` invocations in `stop`, `restart`, and `teardown`.
- **Rust Poisoned HTTP Readiness, KV Pin Safety & TOCTOU Hardening ([`singularity-cache`](singularity-cache/src/block_manager.rs), [`singularity-scheduler`](singularity-scheduler/src/zos_pipeline.rs), [`singularity-server`](singularity-server/src/handlers.rs), [`singularity-cli`](singularity-cli/src/main.rs))**:
  - `PagedBlockManager::pin_block` now rejects unallocated blocks (`!is_block_allocated`), preventing use-after-free pinning.
  - `ZosScheduler::submit_sequence` rolls back partial `pin_block` calls via `unpin_block` on mid-loop failure and re-checks `fatal` under `waiting_queue.lock()` to close the TOCTOU race with `poison()`.
  - `ready_handler` and `health_handler` check `state.scheduler.fatal()`, returning `HTTP 503` (`ready: false`, `poisoned: true`, `distributed_topology_operational: false`) on `/ready` and `"status": "degraded"` on `/health` when poisoned.
  - `execute_host_development_chat` calls `cancel_sequence(seq_id)` if `submit_sequence` fails (preventing leaked subscriber channels in `scheduler.listeners`) and returns `HTTP 503` when poisoned.
  - `singularity-cli serve` wires `.with_graceful_shutdown(shutdown_signal())` for `SIGINT` and `SIGTERM`.
- **CI Formatting & Rustdoc Gates**: Fixed 35 `cargo fmt` diffs and 3 broken intra-doc links (`v41_hyper.rs`, `lcfs.rs`).

#### Changed
- Deduplicated 21 byte-identical private `fn error()` helpers in `singularity-model` into a single `crate::model_error` helper.
- Made `--with-deepinfer` opt-in (`WITH_DEEPINFER=1`) in [`deploy/modal/scripts/deploy_service.sh`](deploy/modal/scripts/deploy_service.sh).

#### Removed
- Pruned 22 unused Rust crate dependencies across 8 workspace crates (`Cargo.toml` files).
- Removed tracked `scripts/__pycache__/*.pyc` bytecode artifacts from git index.

#### Security
- Gateway request deduplication and admission accounting always use server-generated `uuid4().hex` keys; client-supplied `x-request-id` headers are never trusted for admission state.

---

## [0.2.0] — 2026-10-31

### DeepSeek-V4.1-Flash-1017 Lossless CSF, 8× NVIDIA B200 Deployment & Optimal Runtime Experience (ORE)

#### Added
- **Native `DeepSeek-V4.1-Flash-1017` & Lossless CSF (`lil-mxfp4-csf-checkpoint/1`) Runtime ([`singularity-model`](singularity-model/), [`ADR-001`](knowledge-base/adrs/ADR-001-lossless-csf-checkpoint-architecture.md))**:
  - On-demand zero-copy reconstruction of `row-base-offset1-u24-exceptions/1` `E8M0` scale tensors across all 48 SafeTensors shards (`96,085` logical tensors, `46,080` CSF scale tensors, `495.63 GB` CSF vs. `510.29 GB` uncompressed) in `SafetensorsMmapLoader`.
  - Full V4.1 architecture modules: `v41_attention.rs`, `v41_compressed_attention.rs`, `v41_compressor.rs`, `v41_sparse.rs`, `v41_indexer.rs`, `v41_moe.rs`, `v41_engram.rs`, `v41_dspark.rs`, `v41_hyper.rs`, `v41_rope.rs`, `v41_quant.rs`, `v41_vision.rs`, `v41_vit.rs`, `v41_openai.rs`, `v41_prompt.rs`, `v41_tools.rs`.
  - Zero-copy Modal Volume normalization (`prepare_csf_checkpoint_directory` in [`deploy/modal/model_storage.py`](deploy/modal/model_storage.py)).
- **Optimal Runtime Experience (ORE) Framework & Telemetry ([`singularity-core/src/metrics.rs`](singularity-core/src/metrics.rs), [`deploy/modal/benchmark.py`](deploy/modal/benchmark.py), [`ADR-005`](knowledge-base/adrs/ADR-005-optimal-runtime-experience-slo-model.md))**:
  - Composite Optimal Runtime Score (`optimal_runtime_score` / $\mathcal{S}_{\text{ORE}}$) combining completion reliability ($R_{\text{comp}}$), ITL cadence stability ($\Phi_{\text{ITL}}$), and TTFT tail stability ($\Phi_{\text{TTFT}}$) with linear-interpolated percentiles.
  - Per-stream isolated SSE reconnection replay (`derive_stream_id`, `x-stream-id`, `Last-Event-ID`) in [`singularity-server/src/ring_buffer.rs`](singularity-server/src/ring_buffer.rs) and [`handlers.rs`](singularity-server/src/handlers.rs).
- **Working Knowledge Base ([`knowledge-base/`](knowledge-base/README.md))**:
  - Canonical [`OPTIMAL_RUNTIME_EXPERIENCE.md`](knowledge-base/OPTIMAL_RUNTIME_EXPERIENCE.md), [`AUTO_ROLLING_ROADMAP.md`](knowledge-base/AUTO_ROLLING_ROADMAP.md), ADRs (`ADR-001`..`ADR-005`), Playbooks (`PLAYBOOK-001`..`PLAYBOOK-004`), and Post-Mortems (`PM-001`..`PM-003`).

#### Fixed
- **Shallow-Mount Container Import Crashloop ([`PM-001`](knowledge-base/post-mortems/PM-001-modal-shallow-mount-import-crashloop.md))**: Added `_resolve_repo_root()` across `deploy/modal/*.py` so `/root/<script>.py` Modal mounts never raise `IndexError: 2`.
- **Startup Revalidation & CPU Reference Hang ([`PM-002`](knowledge-base/post-mortems/PM-002-startup-revalidation-and-cpu-reference-hang.md))**: Single-pass startup validation (`ensure_checkpoint_ready` + `skip_checkpoint_revalidation=True`) and bounded interactive resident serving slice (`SINGULARITY_FULL_CPU_REFERENCE=1` opt-in).
- **Checkpoint Validation Schema `KeyError: 'shard_count'` ([`PM-003`](knowledge-base/post-mortems/PM-003-validate-checkpoint-schema-keyerror.md))**: Standardized 15-key return schema across all return paths of `validate_checkpoint_directory()`.

#### Changed
- **ORE Inner-Loop Optimizations**:
  - Added `TextModel::prefill_token_with_targets` to skip $P-1$ redundant `1.32 GiB` vocabulary head projections during DSpark prompt prefill.
  - Shared quantized FP8 input activations across all routed/shared experts in `v41_moe.rs` ($14\times \rightarrow 1\times$) and across `q_a`/`kv` in `v41_attention.rs` ($2\times \rightarrow 1\times$), with concurrent `rayon::join` gate/up projections and $O(N)$ `select_nth_unstable_by` partial selection.
  - Replaced per-block heap `Vec<f32>` allocations in `v41_quant.rs`, `v41_compressor.rs`, and `v41_sparse.rs` with stack arrays (`[0.0f32; 32]` / `[0.0f32; 64]`) and added `.with_min_len(64)` Rayon granularity guards.
  - Replaced per-request `httpx.AsyncClient` instantiation in `deploy/modal/app.py` with a loop-scoped connection-pooled client (`max_connections=256`, `max_keepalive_connections=64`).

---

## [0.1.0] — 2026-10-30

### Initial 9-Crate Rust Inference Engine Release

#### Added
- **`singularity-core`**: `SingularityError`, `Sequence`, `SystemConfig`, and `MetricsCollector`.
- **`singularity-cuda`**: `GpuBackend` trait, `detect_hardware` (SM100 Blackwell / SM90 Hopper / host fallback), `CudaGraphRunner`, `MockGpuBackend`, and 8×B200 `ClusterPlacementPlan`.
- **`singularity-cache`**: `PagedBlockManager` (bitset allocation, double-free protection, reference counting), `RadixTree` SHA-256 prefix cache, `FP4Layout`, `MXFP8Layout`, and `SWABuffer`.
- **`singularity-scheduler`**: `ZosScheduler` (Zero-Overhead Scheduler), `schedule_decode_lcfs` (Last-Come-First-Served decode preemption), `DpConsensusEngine`, `DynamicMtpDraftAllocator`, and `DeferredRelease`.
- **`singularity-model`**: Zero-copy `SafetensorsMmapLoader`, `RMSNorm`, `RoPE`, `SwiGLU`, `MultiHeadLatentAttention` (FlashMLA), `MqaIndexer`, `DeepSeekMoE`, and `TokenizerWrapper`.
- **`singularity-grammar`**: Streaming `KmpMatcher` for `<tool_call>` boundaries and `JsonBitmaskCompiler` for JSON Schema constrained decoding.
- **`singularity-server`**: Axum OpenAI-compatible HTTP/SSE server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/ready`, `/metrics`), `StreamRingBuffer`, `AdmissionLifecycle`, 4-state `CircuitBreaker`, `LeakyTokenBucket`, and `TransactionalStore`.
- **`singularity-cli`**: `singularity` binary (`serve`, `verify-checkpoint`, `benchmark`, `inspect-hardware`, `validate-config`).
- **`singularity-tests`**: End-to-end HTTP server tests (`e2e_server.rs`), 18 concurrency/memory edge-case tests (`failure_modes.rs`), SmolLM2-135M & DeepSeek-V2-Lite forward-pass suites (`real_inference.rs`), and Criterion benchmarks.

---

[Unreleased]: https://github.com/CardSorting/HEAV3NS-BRAIN/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/CardSorting/HEAV3NS-BRAIN/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/CardSorting/HEAV3NS-BRAIN/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/CardSorting/HEAV3NS-BRAIN/releases/tag/v0.1.0
