# SINGULARITY Architecture State (`ARCH_STATE.md`)

**Status**: Phase III + V4.1-Flash-1017 Lossless CSF & Modal 8× NVIDIA B200 Verified (`v0.2.0-v41-csf`)  
**Knowledge Base**: [`knowledge-base/README.md`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/README.md)

---

## 1. System Topology

```text
SINGULARITY Workspace
├── singularity-core        # Config, Sequence, linear-interpolated TTFT/ITL & Optimal Runtime Score (S_ORE)
├── singularity-cuda        # Hardware Backend, MockGpuBackend, CudaBackend (cudarc), SM100 CuGraph & 8x B200 Layout
├── singularity-cache       # PagedBlockManager (64-bit bitset), BLAKE3 RadixTree PrefixCache, PinnedNodeGuard
├── singularity-grammar     # KMP Streaming Tool-Call Matcher, JSON Token Bitmask Engine
├── singularity-model       # DeepSeek-V4.1-Flash-1017 (MLA, 384-Expert MoE, Engram, DSpark) & Lossless CSF Loader
├── singularity-scheduler   # Zero-Overhead Scheduler (ZOS), LCFS Preemption, DP All/Any Consensus
├── singularity-server      # Axum OpenAI HTTP/SSE Server, V41ServingEngine, Circuit Breaker, Rate Governor, RingBuffer
├── singularity-cli         # CLI Entrypoint (serve, validate, benchmark)
├── singularity-tests       # Integration, 18 Failure-Mode Stress Tests, Real Inference, V4.1 E2E Readiness/Auth/CSF
├── deploy/modal            # 8x NVIDIA B200 Modal Orchestration, CSF Hydration, Pooled ASGI Proxy & ORE Benchmark Suite
└── knowledge-base          # ADRs, Playbooks, Post-Mortems (PM-001..003), ORE Treatise & Auto-Rolling Roadmap
```

---

## 2. Core Architectural Invariants

1. **Zero-Copy Lossless CSF Checkpoint Loading ([ADR-001](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/adrs/ADR-001-lossless-csf-checkpoint-architecture.md))**:
   - `SafetensorsMmapLoader` (`singularity-model/src/safetensors.rs`) natively parses both standard `deepseek-ai/DeepSeek-V4.1-Flash-1017` shards and `local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF` (`lil-mxfp4-csf-checkpoint/1`, `row-base-offset1-u24-exceptions/1`) shards, reconstructing `46,080` E8M0 scale tensors (`96,085` logical tensors total across `142,165` raw header entries) on demand with `OnceLock<Arc<[u8]>>` caching.
2. **Single-Pass Startup & Bounded Interactive Serving Slice ([ADR-002](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/adrs/ADR-002-single-pass-startup-and-bounded-slice.md))**:
   - `ensure_checkpoint_ready()` validates the 48 CSF shards once at container boot and passes `skip_checkpoint_revalidation=True` to `SingularityRuntimeSupervisor.start()`.
   - `PinnedV41Runner::new` bounds interactive serving to a 1-layer resident slice by default (`SINGULARITY_FULL_CPU_REFERENCE=1` enables full 40-layer offline CPU reference execution).
3. **Optimal Runtime Experience (ORE) Inner-Loop & Telemetry Equilibrium ([ADR-005](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/adrs/ADR-005-optimal-runtime-experience-slo-model.md))**:
   - `TextModel::prefill_token_with_targets` skips $N-1$ `1.32 GiB` vocabulary head projections during DSpark prompt prefill.
   - `v41_moe::forward_single_impl` quantizes the input activation to FP8 once per MoE layer (`14x -> 1x`) and runs `w1`/`w3` projections via `rayon::join`; `LoadedWindowAttention::forward_with_compressed` quantizes the input activation once per attention layer (`2x -> 1x`).
   - `MetricsCollector` and `deploy/modal/benchmark.py` compute exact linear-interpolated percentiles, jitter ratios ($J_{\text{TTFT}}, J_{\text{ITL}}$), and the unified composite Optimal Runtime Score ($\mathcal{S}_{\text{ORE}} \in [0.0, 1.0]$).
4. **Resilient Admission & Reconnectable Streaming**:
   - Lock-free `TokenBucketRateGovernor` (`HTTP 429`), 3-state `CircuitBreaker` (`HTTP 503`, including worker panic failure recording), and `StreamRingBuffer` (`Last-Event-ID` per-stream token replay via `derive_stream_id` and `x-stream-id` across both host-dev and V4.1 SSE streams).
