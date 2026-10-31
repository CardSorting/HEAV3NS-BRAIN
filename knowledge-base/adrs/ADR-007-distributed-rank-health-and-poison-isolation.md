# ADR-007: Distributed Rank-Health Monitoring, Bounded Collectives, External Step Guards & Poison Isolation

- **Status**: Accepted & Verified across Workspace Integration Suites
- **Date**: 2026-10-31
- **Components**: [`singularity-scheduler/src/rank_health.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-scheduler/src/rank_health.rs), [`singularity-scheduler/src/zos_pipeline.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-scheduler/src/zos_pipeline.rs), [`singularity-scheduler/src/dp_consensus.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-scheduler/src/dp_consensus.rs), [`singularity-cache/src/block_manager.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-cache/src/block_manager.rs), [`singularity-server/src/app.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/app.rs), [`singularity-server/src/handlers.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/handlers.rs), [`singularity-cli/src/main.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-cli/src/main.rs)

---

## 1. Context

In multi-GPU distributed inference (`8× NVIDIA B200`), a single hung NCCL collective, stalled forward pass, or dead peer rank can wedge the stepping thread indefinitely while the container continues accepting HTTP requests or billing for 8 GPUs. Furthermore, our Pass 6 audit identified four subtle concurrency and resource-management failure modes in the Rust serving stack:
1. **Unbounded Collective Waits & Unwatched Synchronous Forward Passes**:
   - Data-parallel consensus voting (`DpConsensusEngine::decide_step_type`) and synchronous V4.1 forward passes (`V41ServingEngine::generate_*` inside `tokio::task::spawn_blocking`) needed bounded collective deadlines and independent watchdog stall detection that never mistakes idle periods for a hang.
2. **Silent `/ready` & `/health` Success When Poisoned**:
   - `health_handler` and `ready_handler` previously inspected only static `ReadinessState` flags and `CircuitBreaker`, returning `HTTP 200` (`ready: true`, `distributed_topology_operational: true`) even after `ZosScheduler` had been poisoned by a rank failure or collective timeout.
3. **`submit_sequence` TOCTOU Race with `poison()` & Subscriber Leak**:
   - `ZosScheduler::submit_sequence` checked `self.fatal` at entry, pinned prefix-cached KV blocks, and then locked `waiting_queue` to push the sequence. If `poison()` executed concurrently between the initial check and `waiting_queue.lock().push_back(seq)`, the sequence was enqueued *after* `poison()` drained `waiting_queue`, leaking its pinned KV blocks and stranding its subscriber channel.
   - In `execute_host_development_chat`, `state.scheduler.subscribe(seq_id)` was called before `submit_sequence(sequence)`; if `submit_sequence` failed, the receiver was dropped without removing the sender from `scheduler.listeners`.
4. **Unallocated Block Pinning & Partial-Pin Leak**:
   - `PagedBlockManager::pin_block` previously incremented `ref_count` without checking `self.is_block_allocated(block_id)`, allowing stale prefix references to pin unallocated blocks, and `submit_sequence` did not roll back earlier pins if a multi-block prefix pin failed mid-loop.

---

## 2. Decision

1. **`RankHealthMonitor` & `run_collective_with_deadline` (`singularity-scheduler/src/rank_health.rs`)**:
   - Tracks per-rank progress timestamps and sticky explicit rank failures (`report_failure`).
   - Wraps every DP consensus all-reduce in `run_collective_with_deadline(deadline, f)` so a hung collective times out deterministically and returns `CollectiveError::Timeout`.
2. **Sticky `poison()` Isolation & `ExternalStepGuard` (`singularity-scheduler/src/zos_pipeline.rs`)**:
   - When a collective fails/times out or `watchdog_tick()` detects a rank failure or active-step stall, `ZosScheduler::poison()` atomically records the sticky `CollectiveError`, sends `SequenceOutputEvent::Cancelled` to all active subscribers, clears host KV caches, drains `waiting_queue`, `running_sequences`, and `preempted_queue`, and frees all allocated KV blocks.
   - Active steps are tracked via `in_step: AtomicUsize` across both `ZosScheduler::step()` (`StepGuard`) and `ZosScheduler::enter_external_step()` (`ExternalStepGuard`), which wraps synchronous V4.1 generation inside `spawn_blocking`. Stall time is measured only when `in_step > 0`, so idle periods never trigger false-positive stalls.
   - `submit_sequence` re-checks `self.fatal.lock()` while holding `self.waiting_queue.lock()` (after rolling back any partial `pin_block` calls via `unpin_block`), closing the TOCTOU race with `poison()`.
3. **Unallocated Pin Rejection (`singularity-cache/src/block_manager.rs`)**:
   - `PagedBlockManager::pin_block` verifies `self.is_block_allocated(block_id)` before incrementing `ref_count`.
4. **HTTP 503 Degradation, Watchdog Process Exit & Graceful Signal Shutdown (`singularity-server`, `singularity-cli`)**:
   - `ready_handler` returns `HTTP 503` (`ready: false`, `poisoned: true`, `distributed_topology_operational: false`, `fatal_error`) and `health_handler` reports `"status": "degraded"` when `state.scheduler` is poisoned.
   - `completions_handler` and `chat_completions_handler` fail fast with `HTTP 503` when poisoned and clean up `scheduler.listeners` via `cancel_sequence(seq_id)` if `submit_sequence` fails.
   - `spawn_rank_watchdog` polls `watchdog_tick()` every 1s via a `Weak<ZosScheduler>` reference, drops its temporary strong `Arc` before the 2s client-cancel grace sleep, and exits the process with `EXIT_GROUP_FAILED = 70` if the scheduler is still alive.
   - `singularity-cli serve` wires `axum::serve(...).with_graceful_shutdown(shutdown_signal())` for `SIGINT` and `SIGTERM`.

---

## 3. Impact on the Optimal Runtime Experience (ORE)

- **Reliability**: Guarantees zero infinite collective hangs, zero leaked KV blocks on partial prefix pins or poison races, zero leaked subscriber channels, and immediate `HTTP 503` readiness withdrawal upon rank failure.
- **Verification**: Locked by 7 integration tests in [`singularity-tests/tests/rank_health_serve.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-tests/tests/rank_health_serve.rs) and unit tests in [`singularity-scheduler/src/rank_health.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-scheduler/src/rank_health.rs).

