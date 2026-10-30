# PROJECT SINGULARITY (HEAV3NS-BRAIN)
## Next-Generation Rust GPU Inference Architecture for NVIDIA B200 (SM100) & H200 (SM90)

Project SINGULARITY is an independent, industrial-grade Rust inference system engineered for NVIDIA Blackwell (SM100 - B200) and Hopper (SM90 - H200) accelerators. Developed pursuant to the executive mandate in [`singularity.md`](singularity.md), SINGULARITY synthesizes architectural lessons from three reference teachers (`DeepInfer`, `NOVAV2`, and `MOSS-Studio`), refuting prior misconceptions and establishing a unified, zero-overhead, crash-resilient runtime.

---

## 1. System Architecture & Crate Topology

```text
/Users/bozoegg/Desktop/HEAV3NS-BRAIN/
├── Cargo.toml                          # Workspace root manifest
├── singularity-core/                   # Shared types, universal SingularityError, configuration, metrics
├── singularity-cuda/                   # Hardware detection, GpuBackend trait, Mock GPU backend, CudaGraph runner
├── singularity-cache/                  # Paged block manager, Radix prefix tree, FP4/MXFP8 layouts, SWA buffer
├── singularity-scheduler/              # Zero-Overhead Scheduler (ZOS), LCFS preemption, DP consensus, MTP
├── singularity-model/                  # Zero-copy Safetensors mmap loader, 2-stage MQA+FlashMLA, MoE EPLB
├── singularity-grammar/                # JSON Schema bitmask compiler, KMP tool call trigger matcher
├── singularity-server/                 # Axum 0.7 REST/SSE server, admission machine, token ring-buffer replay
├── singularity-cli/                    # Developer CLI (serve, benchmark, inspect-hardware, validate-config)
├── singularity-tests/                  # End-to-end integration test harness & Criterion microbenchmarks
├── benches/                            # Root symlink to microbenchmarks
└── tests/                              # Root symlink to 18-point failure mode integration tests
```

---

## 2. Key Architectural Subsystems & Verified Discoveries

### A. Zero-Overhead Scheduling (ZOS) & LCFS Decode Preemption
- **Verified Code Ground Truth (`csrc/engine/scheduler.hpp#L622-L646`):**
  DeepInfer sorts running sequences descending by output length (`a > b`), schedules shortest first (`running[r-1]`), and preempts longest first (`running[l]`, $l=0$) to protect incoming throughput. SINGULARITY implements this exact Last-Come-First-Served (LCFS) preemption model in `singularity-scheduler`.
- **ZOS CPU-GPU Concurrency:** Step N+1 forward pass is launched on the GPU compute stream immediately, running concurrently with Step N CPU post-processing (token sampling, argmax/top-k, logprob computation, and metadata updates).
- **Deferred Resource Release (`DeferredRelease`):** Finished sequence metadata, block unpins, and output handles are held across the forward boundary and only freed during `pre_sampler_sync`, eliminating CPU pauses during CUDA kernel launches.

### B. Two-Tier Data-Parallel (DP) Collective Consensus Voting
- **ALL Point (`decode_step % prefill_attempt_interval == 0`):** Requires 100% of DP ranks to have waiting prompts via `_dp_allreduce(want ? 1 : 0, MIN) == 1`.
- **ANY Point (`decode_step != 0 && decode_step % prefill_any_attempt_interval == 0`):** Triggers prefill if any rank has waiting prompts via `_dp_allreduce(want ? 0 : 1, MIN) == 0`, preventing prompt starvation on isolated ranks.
- **Two-Phase Validation:** If all ranks fail to schedule tokens during prefill (e.g. KV cache exhaustion), the entire cluster falls back immediately to decode, preventing infinite zero-work spin loops.

### C. True Radix Tree Prefix Cache & Multi-Tier KV Layout
- **True In-Memory Radix Tree:** While DeepInfer omitted its prefix cache from open-source release, SINGULARITY provides a pure-Rust Radix tree with reference counting, LRU eviction, and cryptographic SHA256 sequence hash verification to eliminate stale key bugs.
- **Physical Memory Layouts:**
  - **Blackwell SM100:** Microscaling FP4 (`NVFP4` E2M1) with token-interleaved data and scales (288 bytes/token), 32-byte memory bus aligned.
  - **Hopper SM90:** MXFP8 (E4M3) with token interleaving (528 bytes/token), 32-byte memory bus aligned.
- **StateBlock Disaggregation:** Splits memory into a Stored Region (persistent SWA window for PE $\to$ CE disaggregated network migration) and a Dropped Region (YOCO states recomputed on-the-fly to conserve network bandwidth).

### D. Two-Stage Sparse Attention & Shared Layer Topology
- **Stage 1 (MQAIndexer):** Quantized FP4 sparse indexing with DeepSelect Top-K token selection (e.g., Top 256 tokens).
- **Stage 2 (FlashMLA):** Dense low-rank projection ($A \to \text{norm} \to B$) computed strictly over the selected sparse tokens and SWA window.
- **SharedLayerTopology:** Maps non-adjacent layers to source projection layers (`layer_map[l] == l`), sharing projection weights across layers to minimize VRAM footprints.

### E. MoE Routing & Expert Load Balancing (EPLB)
- **Scoring:** `sqrtsoftplus` scoring with `noaux_tc` bias and text/vision correction biases (`e_score_correction_bias`).
- **Dynamic EPLB:** Tracks token arrivals per physical expert, detects hotspot imbalances, and plans redundant expert weight replication across NVLink using symmetric memory buffers.

### F. KMP Tool Call Trigger Matcher & Constrained Decoding
- **KMP Trigger State Machine:** Monitors streaming output for tool start triggers (e.g., `<tool_call>`), activating JSON grammar bitmask enforcement strictly inside the argument boundary, and restoring open vocabulary upon reaching the stop trigger (e.g., `</tool_call>`).
- **Bitmask Engine:** Fast vocabulary logit masking setting forbidden tokens to `-f32::INFINITY`.

### G. Resilient Axum 0.7 REST/SSE Serving & Token Ring-Buffer Replay
- **OpenAI Compatible:** Exposes `/v1/chat/completions`, `/v1/models`, `/health`, and `/metrics`.
- **Token Ring-Buffer Replay (`StreamRingBuffer`):** Bounded sliding window (`VecDeque<StreamToken>`) retaining tokens with monotonic sequence IDs (`seq_no`). Clients that drop connection can reconnect with `Last-Event-ID: <seq_no>` and resume streaming without re-running GPU inference.
- **Eager Disconnect Propagation:** Drops client TCP socket terminates orphaned GPU sequences immediately via `AdmissionLifecycle`.
- **Circuit Breaker & Rate Governor:** 4-state circuit breaker (`Healthy`, `Degraded`, `Exhausted`, `Disabled`) with canary slots and leaky token bucket rate limiting.
- **Transactional State Persistence (`TransactionalStore`):** MOSS-Studio-style atomic temporary files, `sync_all` (`fsync`), atomic POSIX `rename`, and startup sweep of stale files (`.tmp.*`).

---

## 3. Hardware Target Matrix & Fallback Policy

| Hardware Feature | NVIDIA Blackwell SM100 (B200) | NVIDIA Hopper SM90 (H200) | SINGULARITY Strategy |
| :--- | :--- | :--- | :--- |
| **Compute Capability** | SM 10.0 (`sm_100a`) | SM 9.0 (`sm_90a`) | Runtime capability interrogation |
| **HBM3e Capacity** | 180 GB | 141 GB | Dynamic KV reservation (90% fraction) |
| **Memory Bandwidth** | 8.0 TB/s | 4.8 TB/s | Bandwidth-aware batch sizing |
| **Native NVFP4** | **Supported (2nd Gen TC)** | **Not Supported** | FP4 on SM100; auto-fallback to MXFP8 on SM90 |
| **Dense FP8 TFLOPS** | ~4,500 TFLOPS | ~1,979 TFLOPS | Microscaling FP8 execution |
| **NVLink Generation**| Gen 5 (1,800 GB/s) | Gen 4 (900 GB/s) | NVLink symmetric memory EPLB |
| **TMA Version** | TMA v2 (Direct L2 bypass) | TMA v1 | TMA async descriptor issue |

---

## 4. Building and Running

### Prerequisites
- Rust 1.88.0+ (stable)
- Cargo package manager
- (Optional for real NVIDIA hardware): CUDA Toolkit 12.2+, Driver 535+, Hopper SM90 or Blackwell SM100

### Build Entire Workspace (CPU & Host Development)
```bash
cargo build --release
```

### Build with Real CUDA Driver & cuBLAS Backing (NVIDIA H200 / B200)
```bash
cargo build --release --features cuda
```

### Run All 61 Unit, Integration, and Reliability Tests
```bash
cargo test --all
```

### Run Transformer Forward & Scheduler Microbenchmark
```bash
cargo run --bin singularity -- benchmark --batch-size 8 --steps 5
```

### Inspect Hardware Capabilities (Truthful Discovery & Hopper/Blackwell Emulation)
```bash
# Auto-detect discovered accelerator on host (reports Host-CPU-Mock if no CUDA GPU)
cargo run --bin singularity -- inspect-hardware

# Inspect Blackwell B200 specifications (180 GB HBM3e, native FP4)
cargo run --bin singularity -- inspect-hardware --arch b200

# Inspect Hopper H200 specifications (141 GB HBM3e, graceful fallback to MXFP8)
cargo run --bin singularity -- inspect-hardware --arch h200
```

### Validate System Configuration
```bash
cargo run --bin singularity -- validate-config
```

### Start the development server

The current transformer runs on the host. This server is experimental and does not execute V4.1 or provide production GPU inference. The mock backend must be explicitly enabled:
```bash
cargo run --bin singularity -- serve --dev-mock --host 127.0.0.1 --port 8080 --storage-dir ./storage
```

---

## 5. Verification Record & Test Suite Results

All **61 unit and integration tests** pass cleanly with 0 warnings:

- **`singularity-cuda` (5 tests):** Allocation alignment, OOM rejection, hardware fallback, host DMA transfer (`copy_htod`, `copy_dtoh`), offset DMA (`copy_htod_offset`, `copy_dtoh_offset`), and mathematically exact `gemm_f32`.
- **`singularity-cache` (4 tests):** Bitset allocation, double-free rejection, Radix tree prefix matching, pinned node protection, and device-backed contiguous KV allocation.
- **`singularity-core` (4 tests):** Sequence lifecycle, TTFT capture, stop token detection, atomic metrics accounting, configuration serde.
- **`singularity-grammar` (3 tests):** KMP table compilation, tool call trigger and stop transitions, JSON bitmask filtering.
- **`singularity-model` (11 tests):** RMSNorm accumulation, RoPE 2D rotations, GQA attention, SwiGLU MLP, DeepSeekMoE routing and execution, Multi-Head Latent Attention (MLA) compressed latent KV cache, multi-shard Safetensors index parsing, HuggingFace `TokenizerWrapper` encode/decode fallback.
- **`singularity-scheduler` (4 tests):** LCFS decode preemption, DP consensus interval safety, two-phase consensus fallback preventing prefill starvation, model forward pass integration and event channel delivery.
- **`singularity-server` (5 tests):** Circuit breaker state machine, leaky token bucket rate governing, admission transitions, ring buffer stream isolation, transactional fsync store.
- **`tests/e2e_server.rs` (1 test):** Full HTTP server bring-up, `/health`, `/metrics`, `/v1/models`, non-streaming completions, and streaming SSE chunks.
- **`tests/failure_modes.rs` (18 tests):** All 18 identified edge cases and failure modes stress-tested across concurrency and memory boundaries.
- **`tests/real_inference.rs` (6 tests):** SmolLM2-135M full transformer forward pass and autoregressive decoding, physical KV-backed scheduler step execution, DeepSeek-V2-Lite MLA + MoE forward and compressed KV caching, genuine scheduler preemption and KV cache reconstruction, multi-shard Safetensors upload to GPU buffers, and paged block manager offset DMA.
