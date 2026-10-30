# ARCH_STATE: Project SINGULARITY
**Last Updated:** Phase II Extended Commission Complete (Pass 9 Complete)  
**Root Destination:** `/Users/bozoegg/Desktop/HEAV3NS-BRAIN`  
**Target Accelerators:** NVIDIA Hopper SM90 (H200 SXM5 141 GB HBM3e) & Blackwell SM100 (B200 180 GB HBM3e)  
**Host Environment:** Apple Silicon Mac (ARM64 Darwin 25.5.0) & Linux x86_64/aarch64 CUDA Targets  

---

## 1. Verified Code Facts (Ground Truth)

| Subsystem | Verified Behavior in Codebase | Source Reference |
| :--- | :--- | :--- |
| **GPU Backend & Transfers** | Dual backend: RAM-backed `MockGpuBackend` executing true host GEMM math, and `CudaBackend` via `cudarc 0.12` under `#[cfg(feature = "cuda")]` with cuBLAS GEMM, streams, and allocations. | `singularity-cuda/src/backend.rs`, `mock.rs`, `cuda_backend.rs` |
| **Hardware Detection** | Honest auto-detection: detects CUDA device if present; otherwise truthfully reports `Host-CPU-Mock (No CUDA Device Detected)` with 32 GB RAM, rather than unconditionally reporting B200. | `singularity-cuda/src/hardware.rs#L83-L160` |
| **Transformer Forward Pass** | Complete forward engine in `singularity-model`: Embedding lookup, RMSNorm, RoPE positional encoding, GQA scaled dot-product attention, SwiGLU MLP, MoE routing, LM head, and Sampler. | `singularity-model/src/transformer.rs#L1-L754` |
| **Tokenization Engine** | HuggingFace BPE tokenization via `tokenizers` crate (v0.21, `fancy-regex`) with deterministic fallback in `TokenizerWrapper`. | `singularity-model/src/tokenizer.rs#L1-L85` |
| **ZOS Scheduler Forward Pass** | Removed synthetic arithmetic (`first_token = 42` and `(seq.id * 31...) % 32000`). All prefill and decode passes invoke `model.forward()` and sample real logits. | `singularity-scheduler/src/zos_pipeline.rs#L315-L415` |
| **Physical KV Device Backing** | `PagedBlockManager::with_device_backing` allocates contiguous GPU memory blocks via `GpuBackend`, providing `write_block_data` and `read_block_data` with RAII reclamation on Drop. | `singularity-cache/src/block_manager.rs#L35-L140` |
| **Multi-Shard Safetensors** | `SafetensorsMmapLoader` parses single files, directory bundles, and `model.safetensors.index.json` manifests with typed F32/BF16/F16 extraction. | `singularity-model/src/safetensors.rs#L1-L320` |
| **DeepSeek Architectural Bridge** | Implemented Multi-Head Latent Attention (FlashMLA + MQAIndexer), compressed latent KV cache, and DeepSeekMoE (shared and routed expert execution). | `singularity-model/src/attention.rs`, `moe.rs`, `transformer.rs` |
| **E2E Serving & Streaming** | Axum 0.7 REST/SSE server using `TokenizerWrapper` for real prompt encoding and completion decoding; token ring-buffer replay with `Last-Event-ID`. | `singularity-server/src/handlers.rs`, `app.rs`, `ring_buffer.rs` |

---

## 2. Workspace Crate Boundaries (Verified Clean Build)

```text
/Users/bozoegg/Desktop/HEAV3NS-BRAIN/
├── Cargo.toml                          # Workspace root with cudarc, tokenizers, half
├── singularity-core/                   # Shared types, error handling, configuration, metrics
├── singularity-cuda/                   # GpuBackend trait, Mock RAM-backed GPU, CudaBackend (cudarc), cuGraph
├── singularity-cache/                  # Paged block manager with device backing, Radix tree, layout
├── singularity-scheduler/              # ZOS scheduler wired to TransformerModel forward & logits sampling
├── singularity-model/                  # Multi-shard Safetensors, SmolLM2-135M, DeepSeekMoE, TokenizerWrapper
├── singularity-grammar/                # JSON Schema bitmask compiler, KMP trigger matcher
├── singularity-server/                 # Axum 0.7 REST/SSE server, token decoding, admission machine
├── singularity-cli/                    # Developer CLI (serve, benchmark, inspect-hardware, validate-config)
├── singularity-tests/                  # 18-point failure modes, E2E server, real_inference suite, benchmarks
├── benches/                            # Root symlink to Criterion microbenchmarks
└── tests/                              # Root symlink to failure modes and real inference tests
```

---

## 3. Execution Roadmap Status

- [x] **Pass 1 — Discovery & Comparative Audit:** Audited all 3 teacher repositories, evaluated SM100 vs SM90, refuted errors, and established architectural baseline (`RESEARCH_REPORT.md`).
- [x] **Pass 2 — Architecture & Scaffolding:** Initialized Cargo workspace, crate definitions, universal error types (`SingularityError`), and core traits.
- [x] **Pass 3 — Implementation:** Implemented pure-Rust core modules with host-mockable GPU backends for deterministic compilation.
- [x] **Pass 4 — Performance Profiling:** Microbenchmarking TTFT, ITL, scheduling latency, and cache eviction speed (`engine_benchmarks.rs`, `singularity benchmark`).
- [x] **Pass 5 — Reliability & Hardening:** Exercised all 18 identified failure modes with multi-threaded stress and failure recovery integration tests (`tests/failure_modes.rs`).
- [x] **Pass 6 — Ergonomics & DX:** Workspace documentation (`README.md`), CLI tools (`singularity`), configuration validation, diagnostic output.
- [x] **Pass 7 — Independent Review:** Adversarial check of memory bounds, lock safety, integer overflow, memory leaks, and concurrency races.
- [x] **Pass 8 — Initial Verification:** Verified clean baseline tests.
- [x] **Pass 9 — Phase II Extended Commission (Real Implementations & Hardening):**
  - Expanded `GpuBackend` trait with `copy_htod`, `copy_dtoh`, `copy_htod_offset`, `copy_dtoh_offset`, and `gemm_f32`.
  - Implemented real RAM backing and true GEMM matrix multiply on CPU in `MockGpuBackend`.
  - Implemented `CudaBackend` using `cudarc 0.12` and cuBLAS with zero-copy `DeviceF32Ptr` and dynamic driver loading under `#[cfg(feature = "cuda")]`.
  - Implemented `TransformerModel` in `singularity-model` (RMSNorm, RoPE, GQA, SwiGLU, LM Head, and Sampler) for SmolLM2-135M and DeepSeek-V2-Lite.
  - Implemented Multi-Head Latent Attention (MLA) and compressed latent KV caching ($c_t^{KV}, k_t^R$) for DeepSeek-V2-Lite architectural bridge.
  - Integrated `tokenizers` crate with `TokenizerWrapper` for HuggingFace encoding/decoding and ChatML tokens.
  - Replaced synthetic token generation in `ZosScheduler` with actual model forward pass and logits sampling.
  - Implemented physical device allocation in `PagedBlockManager::with_device_backing` with direct offset DMA.
  - Fixed prefill starvation deadlock via two-phase consensus fallback (`validate_two_phase(0, ...)`).
  - Fixed LCFS preemption memory leaks and preserved batch overflow sequences in `running_sequences`.
  - Implemented multi-shard index parsing in `SafetensorsMmapLoader`.
  - Upgraded hardware interrogation to truthfully report host status when CUDA is unavailable.
  - Verified 0 warnings across all targets with `cargo clippy --all-targets -- -D warnings` and `cargo clippy --all-targets --features cuda -- -D warnings`.
  - Verified all 61 tests pass in `cargo test --all`.
