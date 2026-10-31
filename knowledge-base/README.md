# SINGULARITY Working Knowledge Base (`knowledge-base/`)

Welcome to the working engineering knowledge base for the **SINGULARITY DeepSeek-V4.1-Flash-1017 Engine & Modal 8× NVIDIA B200 Serving Platform**.

This directory serves as the canonical operational, architectural, and evolutionary record of the repository. It codifies our architectural decisions, live production post-mortems, operational runbooks, the **Optimal Runtime Experience (ORE)** engineering treatise, and our self-adjusting **Auto-Rolling Roadmap**.

---

## 1. Directory Taxonomy

```text
knowledge-base/
├── README.md                                              # Knowledge Base index, governance, and ORE charter
├── OPTIMAL_RUNTIME_EXPERIENCE.md                          # Deep architectural treatise on the Performance × Speed × Reliability triad
├── AUTO_ROLLING_ROADMAP.md                                # Self-adjusting H0–H3 roadmap with quantitative gates & feedback loops
├── adrs/                                                  # Architecture Decision Records (immutable design history)
│   ├── ADR-001-lossless-csf-checkpoint-architecture.md    # Zero-copy MXFP4 E8M0 Lossless CSF scale reconstruction
│   ├── ADR-002-single-pass-startup-and-bounded-slice.md   # Single-pass startup validation & bounded resident serving slice
│   ├── ADR-003-modal-container-import-and-asgi-proxy.md   # Shallow-mount import resolution & pooled ASGI streaming proxy
│   ├── ADR-004-zos-lcfs-flashmla-runtime-triad.md         # ZOS, LCFS, FlashMLA-4, and DSpark runtime integration
│   ├── ADR-005-optimal-runtime-experience-slo-model.md    # Formal ORE composite balance metric & admission control loop
│   ├── ADR-006-scale-to-zero-lifecycle-and-cost-governance.md # Scale-to-zero policy, LifecycleTracker/Watchdog, SSE [DONE] accounting
│   └── ADR-007-distributed-rank-health-and-poison-isolation.md # RankHealthMonitor, bounded collectives, ExternalStepGuard & 503 poison isolation
├── playbooks/                                             # Operational runbooks for deployment, triage, benchmarking & cost
│   ├── PLAYBOOK-001-modal-8xb200-deployment-and-hydration.md
│   ├── PLAYBOOK-002-startup-hang-and-latency-triage.md
│   ├── PLAYBOOK-003-benchmarking-and-ore-calibration.md
│   └── PLAYBOOK-004-cost-governance-and-container-lifecycle.md
└── post-mortems/                                          # Blameless root-cause analyses of live cloud & runtime incidents
    ├── PM-001-modal-shallow-mount-import-crashloop.md
    ├── PM-002-startup-revalidation-and-cpu-reference-hang.md
    ├── PM-003-validate-checkpoint-schema-keyerror.md
    ├── PM-004-deploy-time-policy-env-drift-in-container.md
    └── PM-005-sse-done-premature-client-disconnect-accounting.md
```

---

## 2. Core Charter: The Optimal Runtime Experience (ORE)

Modern large-scale LLM inference engines fail in production when they optimize a single dimension in isolation:
- **Throughput-myopic engines** maximize tokens/sec at batch size 128 while starving interactive requests (catastrophic TTFT spikes and jerky token streaming).
- **Latency-myopic engines** run batch size 1 with aggressive speculation that collapses under multi-tenant burst load or exhausts KV cache blocks.
- **Unbounded reference engines** prioritize bit-exact host emulation without bounding compute per HTTP request, resulting in gateway timeouts (HTTP 499/502) and unrecoverable cold-start loops.

**The Optimal Runtime Experience (ORE)** establishes a provable, continuously measured equilibrium across three orthogonal pillars:
1. **Performance (Throughput & Hardware Efficiency)**: Sustained aggregate output tokens/sec, high MoE active-expert utilization, zero-copy checkpoint hydration, and prefix-cached KV reuse.
2. **Speed (Interactive Latency & Streaming Fluidity)**: Low P50/P99 Time-To-First-Token (`TTFT`), low Time-Per-Output-Token (`TPOT`), and—critically—bounded Inter-Token Latency (`ITL`) jitter (`p99_itl / p50_itl`) so human and agentic consumers experience smooth, stutter-free streaming.
3. **Reliability (Resilience, Determinism & Tail Bounds)**: 99.9%+ completion rate, sub-millisecond admission shedding (Token Bucket + 3-State Circuit Breaker), SSE reconnection replay via `StreamRingBuffer` (`Last-Event-ID`), and bit-exact SHA-256 parity on reference tensors.

Both the Rust engine (`singularity-core::metrics::MetricsSnapshot`) and the Python benchmark suite (`deploy/modal/benchmark.py::summarize_samples`) compute the live **Optimal Runtime Score** ($\mathcal{S}_{\text{ORE}} \in [0.0, 1.0]$):

$$\mathcal{S}_{\text{ORE}} = 0.40 \cdot R_{\text{completion}} + 0.35 \cdot \Phi_{\text{ITL}} + 0.25 \cdot \Phi_{\text{TTFT}}$$

where:
- $R_{\text{completion}} = \frac{N_{\text{completed}}}{N_{\text{completed}} + N_{\text{aborted}} + N_{\text{rejected}}}$
- $\Phi_{\text{ITL}}$ penalizes token-cadence jitter ($J_{\text{ITL}} = \frac{\text{ITL}_{p99}}{\text{ITL}_{p50}}$)
- $\Phi_{\text{TTFT}}$ penalizes prefill tail amplification ($J_{\text{TTFT}} = \frac{\text{TTFT}_{p99}}{\text{TTFT}_{p50}}$)

---

## 3. Knowledge Base Governance Rules

1. **Every Production Incident Generates a Post-Mortem (`post-mortems/PM-XXX-*.md`)**:
   - Must include exact timeline, root cause in source code (file and line), blast radius, remediation diff, regression test added, and preventative architectural guardrails.
2. **Every Architectural Trade-Off Generates an ADR (`adrs/ADR-XXX-*.md`)**:
   - Must document Context, Considered Alternatives, Decision, Quantitative Impact on the ORE Triad, and Verification Tests.
3. **Every Operational Procedure Has a Copy-Pasteable Playbook (`playbooks/PLAYBOOK-XXX-*.md`)**:
   - All commands must be executable via `python3 deploy/modal/deploy.py` or `cargo` without manual guesswork.
4. **The Auto-Rolling Roadmap (`AUTO_ROLLING_ROADMAP.md`) Is Driven by Telemetry Gates**:
   - Items graduate from Horizon 0 (Active Hardening) $\rightarrow$ Baseline only when both `cargo test --workspace` and `deploy/modal/test_modal_pipeline.py` pass and $\mathcal{S}_{\text{ORE}} \ge 0.90$.

