# SINGULARITY Auto-Rolling Roadmap: Performance, Benchmarks & Optimal Runtime Experience

**Document Type**: Living Self-Adjusting Engineering Roadmap  
**Cadence**: Continuous Telemetry-Gated Rolling Horizons (`H0` $\rightarrow$ `H1` $\rightarrow$ `H2` $\rightarrow$ `H3`)  
**Governing Metric**: Composite Optimal Runtime Score ($\mathcal{S}_{\text{ORE}} \ge 0.90$)

---

## 1. How the Auto-Rolling Mechanism Works

Unlike static quarterly roadmaps that become obsolete after the first production deployment, this roadmap operates as a **state machine driven by benchmark and runtime telemetry**:

1. **Quantitative Promotion Gates**: An item in **Horizon 0 (Active Hardening)** cannot roll into the **Verified Production Baseline** until:
   - All workspace Rust tests (`cargo test --workspace`) and clippy lints (`--features cuda -D warnings`) pass with 0 warnings.
   - All Modal pipeline unit tests (`deploy/modal/test_modal_pipeline.py`) pass.
   - Section 12.1 Functional & Correctness Gate (`deploy/modal/benchmark.py::run_correctness_gate`) passes 100% deterministic greedy parity.
   - Live or simulated benchmark summary achieves `optimal_runtime_experience.stability_tier == "optimal"` ($\mathcal{S}_{\text{ORE}} \ge 0.90$, $J_{\text{ITL}} \le 1.35$, $J_{\text{TTFT}} \le 1.50$).
2. **Automatic Regression Rollback & Reprioritization**:
   - If any change in **Horizon 1 (Performance & Speed)** causes $J_{\text{ITL}} > 1.50$ or $R_{\text{comp}} < 0.995$, the roadmap automatically freezes throughput expansion and promotes the corresponding **Reliability / Tail-Latency Stabilization** item from Horizon 2 into Horizon 0.
   - Every production incident (`post-mortems/PM-XXX-*.md`) automatically injects a preventative guardrail task into Horizon 0 before new kernel work resumes.

```text
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                   TELEMETRY & BENCHMARK FEEDBACK LOOP                   │
  │  /metrics (optimal_runtime_score) + benchmark.py (stability_tier)       │
  └───────────────┬─────────────────────────────────────────▲───────────────┘
                  │                                         │
     Gate Passed  │ (S_ORE >= 0.90, J_ITL <= 1.35)          │ Regression Trigger
                  ▼                                         │ (S_ORE < 0.90 or PM-XXX)
  ┌───────────────────────────────┐         ┌───────────────┴───────────────┐
  │ VERIFIED PRODUCTION BASELINE  │◄────────┤   HORIZON 0: ACTIVE SPRINT    │
  │ (Locked by CI + E2E tests)    │         │   (Immediate Hardening/ORE)   │
  └───────────────────────────────┘         └───────────────▲───────────────┘
                                                            │ Auto-Pull Next
                                            ┌───────────────┴───────────────┐
                                            │   HORIZON 1: NEXT ROLLING     │
                                            │   (SM100 Kernel & KV Fusion)  │
                                            └───────────────▲───────────────┘
                                                            │ Auto-Pull Next
                                            ┌───────────────┴───────────────┐
                                            │   HORIZON 2 & 3: EXPANSION    │
                                            │   (Multi-Node EP & Speculative│
                                            │    Tree Verification)         │
                                            └───────────────────────────────┘
```

---

## 2. Verified Production Baseline (Completed & Locked)

Every item below has passed its promotion gate and is locked by automated regression tests:

| ID | Milestone | Verification Gate | Status |
| :--- | :--- | :--- | :--- |
| **BASE-01** | **Lossless CSF Checkpoint Integration (`lil-mxfp4-csf-checkpoint/1`)**: Zero-copy reconstruction of `row-base-offset1-u24-exceptions/1` E8M0 scales across 48 shards (`96,085` logical tensors, `46,080` CSF scale tensors, `142,165` raw header tensors, `495.63 GB` CSF vs `510.29 GB` uncompressed) | `singularity-model/src/safetensors.rs` + `test_csf_codec_reconstruct_and_checkpoint_preparation` | **LOCKED** |
| **BASE-02** | **Shallow-Mount Container Import Resilience (`PM-001`)**: `_resolve_repo_root` across all 9 `deploy/modal/*.py` modules for `/root/<script>.py` Modal mounts | `test_script_mode_and_modal_cli_import_refs` | **LOCKED** |
| **BASE-03** | **Single-Pass Startup Validation & Bounded Resident Slice (`PM-002`)**: Eliminated triple 48-shard startup scan (`ensure_checkpoint_ready` + `skip_checkpoint_revalidation=True`) and bounded resident serving slice (`SINGULARITY_FULL_CPU_REFERENCE=1` opt-in) | `test_e2e_v41_serving_readiness_and_auth` + live 8× B200 verification | **LOCKED** |
| **BASE-04** | **Standardized 15-Key Checkpoint Validation Contract (`PM-003`)**: Eliminated `KeyError: 'shard_count'` across all return paths of `validate_checkpoint_directory` | `test_safetensors_header_and_checkpoint_validation` | **LOCKED** |
| **BASE-05** | **ORE Inner-Loop & Telemetry Hardening**: DSpark `prefill_token_with_targets` (skipping $N-1$ 1.3 GiB logit projections), shared MoE/Attention FP8 activation quantization ($14\times \rightarrow 1\times$ MoE, $2\times \rightarrow 1\times$ Attention), stack-allocated FP8/FP4/sparse-softmax block buffers, $O(N)$ partial selection in MoE/sparse/sampling, pooled ASGI `httpx.AsyncClient`, isolated V4.1 SSE `StreamRingBuffer` replay, and unified $\mathcal{S}_{\text{ORE}}$ telemetry | `cargo test --workspace` + `deploy/modal/test_modal_pipeline.py` | **LOCKED** |
| **BASE-06** | **True Scale-to-Zero Lifecycle, Deploy-Time Policy Snapshot & SSE `[DONE]` Accounting (`ADR-006`, `PM-004`, `PM-005`)**: `min_containers=0` default with `SINGULARITY_ALLOW_PINNED_WARM=1` guard, `policy_env_snapshot()` baked into all Modal images, 6-state `LifecycleTracker` (serving lifetime from `READY`, spend from boot), `LifecycleWatchdog` self-exit (`SIGINT` $\rightarrow$ exit code `70`), `_ReleasingStreamingResponse` (`aclose()` before `_on_finish()`, `saw_done` + `tokens_out` accounting), `deepinfer_app.py` lifecycle wiring, and `ops.py` zero-compute/orphan audit (`71` Python tests + live `T4` & `2×H100` smoke verification) | `deploy/modal/test_lifecycle.py` + `deploy/modal/test_scaling_policy.py` + live Modal smoke runs (`docs/SCALE_TO_ZERO_REPORT.md` §0b) | **LOCKED** |
| **BASE-07** | **Distributed Rank-Health Isolation, `ExternalStepGuard` & KV Pin Safety (`ADR-007`)**: `RankHealthMonitor`, `run_collective_with_deadline`, sticky `ZosScheduler::poison()` isolation, `ExternalStepGuard` for V4.1 `spawn_blocking` forward passes, `HTTP 503` poisoned degradation on `/ready`, `/health`, and `/v1/chat/completions`, subscriber listener cleanup on submit error, `PagedBlockManager::pin_block` unallocated rejection + partial-pin rollback, and `SIGINT`/`SIGTERM` graceful server shutdown | `singularity-tests/tests/rank_health_serve.rs` (`7` integration tests) + `singularity-scheduler/src/rank_health.rs` | **LOCKED** |
| **BASE-08** | **Workspace Dependency Pruning, Unified CI Gates & High-Throughput Agent Ergonomics**: Pruned 22 unused Rust crate dependencies across 8 crates, fixed all `cargo fmt` and `RUSTDOCFLAGS="-D warnings" cargo doc` checks, added `python-modal` CI job in `.github/workflows/ci.yml`, and created root `Makefile` and `AGENTS.md` | `make check` (`cargo fmt`, `clippy`, `doc`, `test`, `ruff`, `71` modal + `11` script tests) | **LOCKED** |

---

## 3. Horizon 0: Active Hardening & ORE Calibration (Current Rolling Window)

**Objective**: Maintain $\mathcal{S}_{\text{ORE}} \ge 0.92$ across Configurations A (Base), B (Optimized), C (Serving), and D (Concurrency Scaling $1 \dots 64$) while keeping idle cloud spend at $\$0.00/\text{hr}$ (`min_containers=0`).

| ID | Initiative | Pillar | Target Metric / Gate | Promotion Trigger to Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **H0-01** | **Adaptive Admission Control Loop via Live $\mathcal{S}_{\text{ORE}}$**: Dynamically modulate `TokenBucketRateGovernor` burst capacity when `itl_jitter_ratio` exceeds `1.35` | Reliability × Speed | $J_{\text{ITL}} \le 1.35$ at `concurrency=32` | 100% pass on `failure_modes.rs` + zero HTTP 502s under 2× overload burst |
| **H0-02** | **Parallel Multi-Thread Shard Header Verification**: Parallelize `validate_checkpoint_directory` across all 48 CSF shards using `ThreadPoolExecutor` so cold-volume validation completes in $< 1.5\text{s}$ | Speed (Cold Start) | Startup Step 9 (`validate_checkpoint`) $< 1.5\text{s}$ on 48 shards | Cold-start time on 8× B200 reduced below $35\text{s}$ end-to-end |
| **H0-03** | **Automated Benchmark Regression Diffing**: Compare every new `benchmark.py` JSON artifact against the pinned baseline and fail CI if $\mathcal{S}_{\text{ORE}}$ drops by $> 0.02$ or $\text{TTFT}_{p95}$ regresses by $> 5\%$ | Benchmarks × Governance | $\Delta \mathcal{S}_{\text{ORE}} \ge -0.02$, $\Delta \text{TTFT}_{p95} \le +5\%$ | Integrated into `deploy.py benchmark` output report |
| **H0-04** | **Live 8× NVIDIA B200 (`B200:8`) Full-Model Scale-to-Zero & Cold-Start Verification**: Execute the capped live 8× B200 runbook (`docs/SCALE_TO_ZERO_REPORT.md` §7) once cluster quota/capacity is available and owner spend approval is granted | Cost Governance × Performance | Measure real 8× B200 `0 -> 1` cold start, TTFT/TPOT, peak VRAM (`/__lifecycle/status?gpu=1`), and `1 -> 0` idle teardown (`modal container list` = `[]`) | All `NEEDS-LIVE` rows in `docs/SCALE_TO_ZERO_REPORT.md` promoted to `VERIFIED-LIVE` |

---

## 4. Horizon 1: Next Rolling Window — SM100 Blackwell Kernel & Memory Bandwidth Saturation

**Entry Trigger**: Automatically pulled into Horizon 0 once all `H0-*` items reach Baseline or when scaling full 40-layer GPU resident execution across 8× B200.

| ID | Initiative | Pillar | Architectural Mechanism | Quantitative Exit Gate |
| :--- | :--- | :--- | :--- | :--- |
| **H1-01** | **In-Kernel GPU CSF Scale Decompression (`tcgen05.mma` Epilogue)**: Upload compact `mxfp4_csf_fixed` (`9/8` byte overhead) and `mxfp4_csf_exceptions` (`u24+u8`) directly to HBM3e and reconstruct E8M0 scales in shared memory (`SMEM`) inside the CUTLASS 3.x / PTX `tcgen05` GEMM prologue | Performance × Speed | Eliminates host-side scale expansion heap memory and saves `14.65 GB` of PCIe/HBM scale traffic (`510.29 GB -> 495.63 GB`) | Cold weight upload $< 8\text{s}$ across 8× B200; bit-exact parity with `decode_mxfp4_csf_scale` |
| **H1-02** | **FlashMLA-4 Split-KV Decode with Persistent Megakernel Dispatch**: Fuse MLA latent projection (`q_lora_rank=2048`, `head_dim=512`, `qk_rope_head_dim=64`), RoPE application, and Paged KV block gather into a persistent SM100 kernel with warp-specialized TMA producers | Speed × Performance | Eliminates intermediate HBM reads/writes between attention sub-steps; targets $1.4\times$ decode TPOT speedup at `long_32k` and `max_128k` tiers | $\text{TPOT}_{p50} \le 4.5\text{ms}$ at $32\text{K}$ context on 8× B200 |
| **H1-03** | **Fused MoE Router + NVLink 5 All-to-All Dispatch (`EP=8`)**: Combine 384-expert `sqrtsoftplus`-bias top-6 routing with asymmetric NVLink 5 (`1.8 TB/s` bidirectional per GPU) token dispatch/combine using CUDA IPC / NVSHMEM | Performance | Eliminates host synchronization barriers across all 40 MoE layers (`layers.0..39`) | $\ge 82\%$ linear scaling efficiency from `concurrency=1` to `concurrency=64` |
| **H1-04** | **Multi-Process NCCL Transport & Remote Rank Heartbeat Producer**: Connect `ZosScheduler::with_dp_reduce` and `RankHealthMonitor::report_progress` / `report_failure` to real multi-rank NCCL communicators and inter-rank heartbeat threads across all 8 B200 ranks | Reliability × Performance | Extends `ADR-007` rank-health poison isolation from single-process step monitoring to full 8-rank NCCL fault & hang isolation | Sub-second rank-failure isolation and container exit (`code 70`) under injected single-rank kill on 8× B200 |

---

## 5. Horizon 2: Medium-Term Rolling Window — Speculative & Prefix Intelligence

**Entry Trigger**: Pulled into Horizon 1 when `H1-01` and `H1-02` pass their exit gates.

| ID | Initiative | Pillar | Architectural Mechanism | Quantitative Exit Gate |
| :--- | :--- | :--- | :--- | :--- |
| **H2-01** | **Multi-Token DSpark Tree Verification ($K=4$ Draft Tree)**: Extend `DsparkModel` from linear $K$-step greedy drafting to tree-structured speculative verification in a single target forward pass using custom causal tree attention masks | Speed | Increases average accepted tokens per target verification step from $\sim 2.4$ to $\sim 3.3$ on structured code/JSON workloads | $\ge 2.1\times$ wall-clock decode speedup over Configuration A (`dspark=False`) with 100% lossless greedy equivalence |
| **H2-02** | **Cross-Request Prefix-Aware Continuous Batching Scheduler**: Upgrade `CooperativeScheduler` to group waiting sequences by shared BLAKE3 prefix hash (`PrefixCache`) and schedule shared-prefix micro-batches together | Performance × Speed | Maximizes L2/HBM cache locality for multi-turn agentic workflows and tool-calling loops | Prefix cache hit rate $\eta_{\text{cache}} \ge 0.70$ on multi-turn agent benchmarks; $\text{TTFT}_{p95}$ reduced by $\ge 40\%$ |
| **H2-03** | **Zero-Overhead Structured Output DFA Mask Caching**: Cache compiled JSON-schema token bitmasks in `singularity-grammar` keyed by schema SHA-256 so repeated `response_format: {"type": "json_schema"}` requests incur zero compilation latency | Speed × Reliability | Eliminates per-request grammar compilation overhead on structured tool/JSON calls | Grammar mask lookup $< 15\,\mu\text{s}$ per decode step |

---

## 6. Horizon 3: Long-Term Rolling Window — Autonomous Multi-Node Self-Healing Runtime

**Entry Trigger**: Pulled into Horizon 2 when single-node 8× B200 saturation exceeds $85\%$ HBM3e bandwidth utilization.

| ID | Initiative | Pillar | Architectural Mechanism | Quantitative Exit Gate |
| :--- | :--- | :--- | :--- | :--- |
| **H3-01** | **Disaggregated Prefill / Decode Clusters over RDMA**: Separate compute-bound prefill nodes (`FlashMLA-4` prefill pool) from memory-bandwidth-bound decode nodes (`EP=8` MoE decode pool) with asynchronous KV latent shipping ($576\text{ B/token/layer}$) | Speed × Performance | Eliminates prefill-induced decode ITL spikes entirely ($J_{\text{ITL}} \rightarrow 1.05$ even under 128K prefill bursts) | $\mathcal{S}_{\text{ORE}} \ge 0.96$ under mixed $128\text{K}$ prefill + interactive streaming load |
| **H3-02** | **Hot-Standby Expert Shadowing & Degraded-Rank Failover**: Replicate top-32 hottest routed experts across adjacent NVLink ranks so single-GPU ECC error or Xid fault triggers sub-second rank quarantine without dropping active SSE streams | Reliability | Zero-downtime recovery from single-rank hardware fault using `StreamRingBuffer` token replay | $99.99\%$ request completion rate across injected GPU fault chaos tests |

---

## 7. Rebalancing Matrix: Resolving Performance vs. Speed vs. Reliability Tensions

When runtime telemetry detects an imbalance, operators and automated controllers apply the following deterministic rebalancing rules:

| Observed Symptom in `/metrics` or `benchmark.py` | Imbalanced Pillar | Root Cause | Immediate Rebalancing Action |
| :--- | :--- | :--- | :--- |
| $J_{\text{ITL}} > 1.35$ (`p99_itl_ms >> p50_itl_ms`) while throughput $\lambda_{\text{out}}$ is high | **Performance** is starving **Speed** | Large prefill chunks or excessive concurrent batch size delaying decode steps | Reduce `chunked_prefill_size` (e.g., $2048 \rightarrow 512$) and cap `max_batch_sequences` |
| $J_{\text{TTFT}} > 1.50$ (`p99_ttft_ms >> p50_ttft_ms`) while $J_{\text{ITL}} \le 1.15$ | **Speed (Decode)** is starving **Speed (Prefill)** | Waiting queue head-of-line blocking or low prefix cache hit rate | Increase `PrefixCache` block allocation (`lcfs_kv_budget_ratio`) and enable `prefill_token_with_targets` |
| $N_{\text{abort}} > 0$ or HTTP 499/502 gateway errors under burst load | **Reliability** breached by unbounded admission | Requests admitted faster than service drain rate, causing upstream timeouts | Lower `TokenBucketRateGovernor` `max_rps` / `burst_capacity` so excess load fails fast with HTTP 429 in $< 50\,\mu\text{s}$ |
| DSpark acceptance rate $< 1.3\text{ tokens/step}$ on high-temperature creative prompts | **Speed (Speculation)** hurting **Performance** | Drafter verification overhead exceeds accepted token savings at high entropy | Automatically bypass DSpark (`dspark: false`) when `temperature > 0.6` |
| `/ready` returns `HTTP 503` with `poisoned: true` and `distributed_topology_operational: false` | **Reliability (Distributed Group Fault)** | Hung collective (`CollectiveError::Timeout`), stalled forward step (`RankStalled`), or dead rank (`RankFailed`) | `ZosScheduler::poison()` cancels all subscribers, frees KV blocks, and `spawn_rank_watchdog` exits with code `70` so Modal replaces or scales down the container |
| `/__lifecycle/status` transitions to `DRAINING` / `STOPPED` while idle window has not elapsed | **Cost Governance Guard Tripped** | Container reached `max_worker_lifetime_secs` (from `READY`), `max_spend_usd` (from boot), or child process exited | `LifecycleWatchdog` drains in-flight requests ($\le 60\text{s}$), stops child server, and sends `SIGINT` (with 60s hard-exit `70` fallback) to return compute to zero |

