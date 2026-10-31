# ADR-005: Optimal Runtime Experience (ORE) Composite SLO Model & Admission Control

- **Status**: Accepted
- **Date**: 2026-10-31
- **Components**: [`singularity-core/src/metrics.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-core/src/metrics.rs), [`singularity-server/src/handlers.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/handlers.rs), [`deploy/modal/benchmark.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/benchmark.py)

---

## 1. Context

Standard LLM serving metrics report aggregate tokens/sec alongside P50/P99 TTFT and TPOT in isolation. In practice, operators lacked a single unified signal that detects when:
1. High batch throughput causes severe token-streaming stutter (`p99_itl_ms >> p50_itl_ms`).
2. Admission rejections (`HTTP 429`/`503`) or worker panics (`HTTP 500`) silently erode effective reliability while successful requests still show low latency.
3. Discontinuous integer-index percentile estimation skews P95/P99 numbers on short benchmark runs.

## 2. Decision

1. **Unified Linear-Interpolated Percentiles**:
   - Both `singularity-core::metrics::interpolate_percentile` and `deploy/modal/benchmark.py::percentile` compute exact linear-interpolated percentiles over sorted sample windows.
2. **Complete Terminal Request Accounting**:
   - `MetricsSnapshot` tracks `completed_requests`, `aborted_requests`, and `rejected_requests` (`on_request_rejected()` wired into Circuit Breaker `503` and Rate Governor `429` paths; `circuit_breaker.record_failure()` wired into worker panic handlers).
3. **Composite Optimal Runtime Score ($\mathcal{S}_{\text{ORE}} \in [0.0, 1.0]$)**:
   - Exported in `/metrics` (`optimal_runtime_score`, `ttft_jitter_ratio`, `itl_jitter_ratio`, `completion_success_rate`) via `compute_optimal_runtime_score` and in `benchmark.py` (`optimal_runtime_experience.balance_score`, `stability_tier`):
     $$\Phi_{\text{ITL}} = \frac{1.0}{1.0 + 0.25 \cdot \max(0.0,\, J_{\text{ITL}} - 1.0)}, \qquad \Phi_{\text{TTFT}} = \frac{1.0}{1.0 + 0.20 \cdot \max(0.0,\, J_{\text{TTFT}} - 1.0)}$$
     $$\mathcal{S}_{\text{ORE}} = \text{clamp}\!\Big(0.40 \cdot R_{\text{completion}} + 0.35 \cdot \Phi_{\text{ITL}} + 0.25 \cdot \Phi_{\text{TTFT}},\, 0.0,\, 1.0\Big)$$

## 3. Impact on the Optimal Runtime Experience (ORE)

- Provides an immediate, machine-actionable scalar invariant ($\mathcal{S}_{\text{ORE}} \ge 0.90 \implies \text{optimal}$) for CI gates, canary rollouts, and the Auto-Rolling Roadmap.

