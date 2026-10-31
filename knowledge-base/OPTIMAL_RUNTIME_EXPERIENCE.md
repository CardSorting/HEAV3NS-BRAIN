# Optimal Runtime Experience (ORE): Architectural Treatise & Mathematical Model

**Status**: Canonical Runtime Architecture Specification  
**Applies To**: `singularity-core`, `singularity-model`, `singularity-scheduler`, `singularity-server`, `deploy/modal/`  
**Target Hardware**: 8× NVIDIA B200 (`B200:8`, SM100 Blackwell) & Host Reference Runtime

---

## 1. Executive Summary

The **Optimal Runtime Experience (ORE)** is SINGULARITY's unifying systems-engineering doctrine. Rather than treating **Performance** (throughput/utilization), **Speed** (TTFT/ITL/streaming cadence), and **Reliability** (availability/determinism/bounded failure modes) as competing trade-offs managed by ad-hoc heuristics, ORE couples them into a closed-loop control system governed by real-time telemetry.

An inference runtime achieves the **Optimal Runtime Experience** when:
1. **Interactive streams never stall**: Inter-Token Latency jitter $J_{\text{ITL}} = \text{ITL}_{p99} / \text{ITL}_{p50} \le 1.35$, eliminating perceptual token stutter during long-context generation or concurrent prefill arrivals.
2. **Prefill admission never starves decode**: Chunked prefill and prefix-cached KV reuse keep $\text{TTFT}_{p99} / \text{TTFT}_{p50} \le 1.50$ without preempting active decode steps.
3. **Overload degrades gracefully and deterministically**: Excess load is shed in $< 50\,\mu\text{s}$ via the lock-free Token Bucket Rate Governor (`HTTP 429`) or Circuit Breaker (`HTTP 503`) before worker queues saturate or upstream gateways time out (`HTTP 499`/`502`).
4. **Transient network drops recover without re-computation**: Disconnected SSE clients resume mid-generation using `Last-Event-ID` backed by the per-stream `StreamRingBuffer`.

---

## 2. The ORE Triad: Formal Mathematical Formulation

### 2.1 Primary Telemetry Vector

At any observation window $W$, the runtime state in `singularity-core/src/metrics.rs` (`MetricsSnapshot`) and `deploy/modal/benchmark.py` (`summarize_samples`) is captured by the state vector:

$$\mathbf{x}(W) = \Big( \lambda_{\text{out}},\, \text{TTFT}_{p50},\, \text{TTFT}_{p99},\, \text{ITL}_{p50},\, \text{ITL}_{p99},\, N_{\text{comp}},\, N_{\text{abort}},\, N_{\text{reject}},\, \eta_{\text{cache}} \Big)$$

where:
- $\lambda_{\text{out}}$: Aggregate output token throughput ($\text{tokens/s}$).
- $\text{TTFT}_{p50}, \text{TTFT}_{p99}$: Linear-interpolated 50th and 99th percentile Time-To-First-Token ($\text{ms}$).
- $\text{ITL}_{p50}, \text{ITL}_{p99}$: Linear-interpolated 50th and 99th percentile Inter-Token Latency ($\text{ms}$).
- $N_{\text{comp}}, N_{\text{abort}}, N_{\text{reject}}$: Completed, aborted (panicked/errored/timed-out), and admission-rejected request counts.
- $\eta_{\text{cache}}$: Content-addressed prefix KV cache hit ratio $\in [0, 1]$.

### 2.2 Jitter Ratios & Stability Functions

Raw P50 latency hides tail spikes caused by garbage collection, heap allocations in inner quantization loops, Rayon thread-pool contention, or unchunked prefill interference. ORE therefore tracks two dimensionless tail-amplification ratios:

$$J_{\text{TTFT}} = \begin{cases} \dfrac{\text{TTFT}_{p99}}{\text{TTFT}_{p50}} & \text{if } \text{TTFT}_{p50} > 0 \\ 1.0 & \text{otherwise} \end{cases}, \qquad J_{\text{ITL}} = \begin{cases} \dfrac{\text{ITL}_{p99}}{\text{ITL}_{p50}} & \text{if } \text{ITL}_{p50} > 0 \\ 1.0 & \text{otherwise} \end{cases}$$

In `singularity-core/src/metrics.rs` (`compute_optimal_runtime_score`) and `deploy/modal/benchmark.py` (`summarize_samples`), the normalized damped stability components are:

$$\Phi_{\text{ITL}} = \text{clamp}\!\left(\frac{1.0}{1.0 + 0.25 \cdot \max(0.0,\, J_{\text{ITL}} - 1.0)},\, 0.0,\, 1.0\right), \qquad \Phi_{\text{TTFT}} = \text{clamp}\!\left(\frac{1.0}{1.0 + 0.20 \cdot \max(0.0,\, J_{\text{TTFT}} - 1.0)},\, 0.0,\, 1.0\right)$$

At the documented SLO boundary ($J_{\text{ITL}} = 1.35$, $J_{\text{TTFT}} = 1.50$, $R_{\text{comp}} = 1.0$), $\Phi_{\text{ITL}} \approx 0.9195$, $\Phi_{\text{TTFT}} \approx 0.9091$, and $\mathcal{S}_{\text{ORE}} \approx 0.9491 \ge 0.90$ (`optimal` tier).

And the terminal reliability ratio is:

$$R_{\text{comp}} = \begin{cases} \dfrac{N_{\text{comp}}}{N_{\text{comp}} + N_{\text{abort}} + N_{\text{reject}}} & \text{if } N_{\text{comp}} + N_{\text{abort}} + N_{\text{reject}} > 0 \\ 1.0 & \text{otherwise} \end{cases}$$

### 2.3 Composite Optimal Runtime Score ($\mathcal{S}_{\text{ORE}}$)

The composite score exported as `optimal_runtime_score` in `/metrics` and `optimal_runtime_experience.balance_score` in benchmark reports is:

$$\mathcal{S}_{\text{ORE}} = \text{clamp}\!\Big( 0.40 \cdot R_{\text{comp}} + 0.35 \cdot \Phi_{\text{ITL}} + 0.25 \cdot \Phi_{\text{TTFT}},\, 0.0,\, 1.0 \Big)$$

| Stability Tier | Score Range ($\mathcal{S}_{\text{ORE}}$) | Operational Interpretation | Automated Control Action |
| :--- | :--- | :--- | :--- |
| **`optimal`** | $[0.90,\, 1.00]$ | Zero aborts, smooth token cadence, tight prefill tail | Eligible for concurrency scale-up & roadmap promotion |
| **`balanced`** | $[0.75,\, 0.90)$ | Minor tail amplification under burst load; zero errors | Hold batch ceiling; prioritize prefix cache & DSpark verification |
| **`degraded`** | $< 0.75$ | High tail jitter or request rejections/aborts | Trigger admission throttling, reduce max batch tokens, halt rollout |

---

## 3. Deep Codebase Audit & ORE Engineering Optimizations

During our deep audit of the SINGULARITY workspace, we identified and eliminated concrete sources of latency jitter, redundant memory traffic, and reliability blind spots across the Rust engine and Python gateway:

### 3.1 Eliminating $N-1$ Redundant 1.3 GiB Output Head Projections in DSpark Prefill
- **Files**: [`singularity-model/src/v41_model.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_model.rs), [`singularity-model/src/v41_dspark.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_dspark.rs)
- **Audit Finding**: `generate_verified_greedy` previously called `model.forward_token_with_targets(token, cache)` on *every* prompt token `0..prompt.len()`. For a prompt of length $P$, the first $P-1$ calls computed `self.io.logits(&normed)`—a $129{,}280 \times 5{,}120$ BF16 matrix-vector projection ($1.32\text{ GiB}$ of weight reads per token)—and immediately discarded the resulting `logits` vector, keeping only `target_hidden` for the lightweight drafter!
- **ORE Fix**: Added `TextModel::prefill_token_with_targets` which runs the transformer layers, advances the MLA KV cache, and extracts `target_hidden` **without** invoking `self.io.logits(&normed)`. Only the final prompt token (`prompt.len() - 1`) executes `forward_token_with_targets`.
- **Impact**: Eliminates $(P - 1) \times 1.32\text{ GiB}$ of memory bandwidth during DSpark prompt prefill, directly slashing $\text{TTFT}_{p50}$ and $\text{TTFT}_{p99}$.

### 3.2 Zero-Heap-Allocation Inner Quantization & Online Softmax Loops + Rayon Granularity Guards
- **Files**: [`singularity-model/src/v41_quant.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_quant.rs), [`singularity-model/src/v41_compressor.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_compressor.rs), [`singularity-model/src/v41_sparse.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_sparse.rs)
- **Audit Finding**:
  1. `quantize_activation_fp8` and `fp4_activation_dequant` allocated a new `Vec<f32>` on the heap for every 32-element block (`values.chunks(32)`), and `sparse_attention_kernel_reference` allocated a heap `Vec<f32>` for every 64-position block of every attention head, causing thousands of allocator lock acquisitions per token and inflating $J_{\text{ITL}}$.
  2. Rayon `.par_iter()` across output rows lacked a minimum work granularity, causing work-stealing task-spawn overhead on small test/slice matrices to exceed the dot-product compute time.
  3. `candidate_blocks` and `compressed_topk` in `v41_sparse.rs` ran full $O(N \log N)$ sorts instead of $O(N)$ partial selection.
- **ORE Fix**:
  1. Replaced heap `Vec<f32>` allocations in `quantize_activation_fp8` and `fp4_activation_dequant` with fixed `[0.0f32; 32]` stack arrays, and in `sparse_attention_kernel_reference` with a `[0.0f32; 64]` stack array.
  2. Added `.with_min_len(64)` to Rayon parallel row iterators in `dense_fp8_matvec`, `expert_fp4_matvec`, `matvec_quantized_with_activation`, and `Bf16Matrix::project`, with a sequential fast path when `rows < 64`.
  3. Replaced full sorts in `candidate_blocks` and `compressed_topk` with $O(N)$ `select_nth_unstable_by` partial selection.
- **Impact**: Eliminates allocator contention and redundant sorting during decode steps, tightening $\text{ITL}_{p99}$.

### 3.3 Shared FP8 Input Activation Across MoE & Attention Projections + Parallel Gate/Up Projections
- **Files**: [`singularity-model/src/v41_moe.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_moe.rs), [`singularity-model/src/v41_attention.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_attention.rs)
- **Audit Finding**:
  1. In `forward_single_impl` (`v41_moe.rs`), each of the 6 selected routed experts (`num_experts_per_tok = 6` out of `n_routed_experts = 384`) plus the 1 shared expert independently called `quantize_activation_fp8(input)` twice (once for `w1`, once for `w3`)—quantizing the exact same `input` vector **14 times** per MoE layer (`18 times` when `top_k = 8`).
  2. In `LoadedWindowAttention::forward_with_compressed` (`v41_attention.rs`), `q_a` and `kv` independently called `quantize_activation_fp8(input)` on the same input vector.
  3. `route_logits` sorted all $E = 384$ expert scores with $O(E \log E)$ full sort when only top-$K$ ($K=6$) experts were needed.
- **ORE Fix**:
  1. Quantize `input` to `QuantizedActivation` **once** per MoE layer in `forward_single_impl` (`shared_act`) and pass it to all routed experts and the shared expert ($14\times \rightarrow 1\times$ per MoE layer), and quantize `input` **once** per attention layer in `LoadedWindowAttention::forward_with_compressed` for both `q_a` and `kv` ($2\times \rightarrow 1\times$ per attention layer).
  2. Execute `w1` (gate) and `w3` (up) projections concurrently via `rayon::join`.
  3. Use $O(E + K \log K)$ `select_nth_unstable_by` partial selection in `route_logits`.

### 3.4 $O(V)$ Greedy Fast-Path & Partial Selection in V4.1 Token Sampling
- **File**: [`singularity-server/src/v41_engine.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/v41_engine.rs)
- **Audit Finding**: `sample_v41_logits` unconditionally cloned the 129,280-element `raw_logits` vector (`517 KiB`) and performed full $O(V \log V)$ sorting across the entire vocabulary even when `temperature == 0` or `top_p == 1.0`.
- **ORE Fix**:
  1. When `temperature <= 1e-5` and no penalties or JSON bitmasks are active, scan `raw_logits` directly in-place without cloning `517 KiB` per token.
  2. When `top_k < V` is active in stochastic sampling, partition the top-$k$ candidates in $O(V)$ time via `select_nth_unstable_by`, and skip full vocabulary sorting completely when `top_p >= 1.0` and `top_k.is_none()`.

### 3.5 Exact Linear-Interpolated Percentiles, Complete Admission Accounting & Isolated SSE Reconnection
- **Files**: [`singularity-core/src/metrics.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-core/src/metrics.rs), [`singularity-server/src/handlers.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/handlers.rs), [`singularity-server/src/ring_buffer.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/ring_buffer.rs)
- **Audit Finding**:
  1. `calculate_stats` previously used integer-truncated array indexing (`sorted[(len * 95) / 100]`), which is discontinuous on small sample counts and inconsistent with `deploy/modal/benchmark.py::percentile`.
  2. Requests rejected by the Circuit Breaker (`503`) or Rate Governor (`429`) were not counted in `MetricsSnapshot`, `spawn_blocking` worker panics did not notify `circuit_breaker.record_failure()`, and `singularity-host-development` requests did not record TTFT/ITL or sequence aborts.
  3. V4.1 SSE streams (`execute_v41_chat` with `stream: true`) required deterministic per-stream isolation (`x-stream-id` or FNV-1a request hash) so `Last-Event-ID` reconnects replay only the missed tokens (`seq_no > last_id`) from `StreamRingBuffer` without leaking concurrent streams or re-executing the model.
- **ORE Fix**:
  1. Unified `MetricsCollector` and `benchmark.py` onto exact linear-interpolated percentiles (`interpolate_percentile`) and the damped `compute_optimal_runtime_score` formula.
  2. Wired `on_request_rejected()`, `on_request_aborted()`, `record_ttft()`, and `record_itl()` across both V4.1 and `singularity-host-development` handlers, and `circuit_breaker.record_failure()` into worker panic handlers.
  3. Implemented `derive_stream_id`, `x-stream-id` response headers, and `build_replay_sse_response` so `Last-Event-ID` reconnects replay missed chunks directly from `StreamRingBuffer` with zero duplicate model passes.

### 3.6 Connection-Pooled ASGI Reverse Proxy in Modal Gateway
- **File**: [`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py)
- **Audit Finding**: `create_asgi_proxy_app` instantiated and closed a brand-new `httpx.AsyncClient` on every incoming HTTP request, creating connection pool and TCP socket churn on loopback (`127.0.0.1:8080`) during concurrent benchmark sweeps (`concurrency=64`).
- **ORE Fix**: Replaced per-request client creation with a loop-scoped, connection-pooled `httpx.AsyncClient` (`max_connections=256`, `max_keepalive_connections=64`) managed by the FastAPI lifespan context and exposed via `proxy.state.get_upstream_client` / `proxy.state.close_upstream_client`.

### 3.7 True Scale-to-Zero Lifecycle, Deploy-Time Policy Snapshot & SSE `[DONE]` Accounting
- **Files**: [`deploy/modal/config.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/config.py), [`deploy/modal/lifecycle.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/lifecycle.py), [`deploy/modal/ops.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/ops.py), [`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py), [`deploy/modal/deepinfer_app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/deepinfer_app.py)
- **Audit Findings**:
  1. Deploy-time `SINGULARITY_*` policy variables were not propagated into the container image environment (`PM-004`), causing in-container `LifecycleTracker` and `LifecycleWatchdog` to silently fall back to defaults.
  2. Standard OpenAI/`urllib` SSE clients close the TCP socket immediately after reading `data: [DONE]` (`PM-005`), exiting Starlette's `StreamingResponse` before `stream_chunks()` resumed after its final `yield` and causing completed SSE streams to be misclassified as `"stream abandoned"` (`failed += 1`, `tokens_out = 0`).
  3. `DeepInferReferenceService.startup()` caught `Exception` without re-raising, risking an unready 8×B200 container staying alive and billing.
- **ORE Fixes**:
  1. Added `config.policy_env_snapshot()` to bake whitelisted `SINGULARITY_*` policy variables into all Modal images (`app.py`, `deepinfer_app.py`, `smoke_app.py`) and exposed enforced `limits` in `/__lifecycle/status`.
  2. Updated `_ReleasingStreamingResponse.__call__` to explicitly `await self.body_iterator.aclose()` before `await self._on_finish()`, tracked `saw_done` and SSE token delta events in `stream_chunks()`, and parsed `usage.completion_tokens` on JSON responses.
  3. Wired `LifecycleTracker`, `LifecycleWatchdog`, `@modal.exit()`, and startup exception propagation into `deepinfer_app.py`, and added `SMOKE_APP_NAME` to `KNOWN_APP_NAMES` for provider-side orphan auditing and `deploy.py teardown`.

### 3.8 Distributed Rank-Health Isolation, `ExternalStepGuard` & KV Block Pin Safety
- **Files**: [`singularity-scheduler/src/rank_health.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-scheduler/src/rank_health.rs), [`singularity-scheduler/src/zos_pipeline.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-scheduler/src/zos_pipeline.rs), [`singularity-cache/src/block_manager.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-cache/src/block_manager.rs), [`singularity-server/src/handlers.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/handlers.rs), [`singularity-cli/src/main.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-cli/src/main.rs)
- **Audit Findings**:
  1. Synchronous V4.1 forward passes (`spawn_blocking`) were not tracked as active steps on `ZosScheduler`, and `/ready` / `/health` did not inspect `state.scheduler.is_poisoned()`.
  2. `ZosScheduler::submit_sequence` had a TOCTOU window with `poison()` and did not roll back partial `pin_block` calls on mid-loop failure; `PagedBlockManager::pin_block` did not reject unallocated blocks; `execute_host_development_chat` leaked a subscriber channel in `scheduler.listeners` if `submit_sequence` failed.
- **ORE Fixes**:
  1. Added `ExternalStepGuard` (`enter_external_step` with `AtomicUsize` active-step tracking) around V4.1 generation, wired `/ready` and `/v1/chat/completions` to fail fast with `HTTP 503` when poisoned, and cleaned up subscriber listeners via `cancel_sequence` on submit error.
  2. Enforced `is_block_allocated` in `PagedBlockManager::pin_block`, added partial-pin rollback (`unpin_block`) and under-lock `fatal` verification in `submit_sequence`, and wired `SIGINT`/`SIGTERM` graceful shutdown into `singularity-cli serve`.


