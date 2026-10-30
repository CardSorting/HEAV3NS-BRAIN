# Changelog

All notable changes to HEAV3NS-BRAIN (Project SINGULARITY) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Initial planning for multi-node NVLink EPLB weight migration.
- Placeholder for CUDA Graph capture on SM100 (Blackwell).

---

## [0.1.0] — 2026-10-30

### Added

#### Core (`singularity-core`)
- `SingularityError` unified error type with `thiserror` derivation.
- `SequenceGroup` lifecycle state machine: `Waiting → Running → Finished/Preempted`.
- `AtomicMetrics` counters for TTFT, tokens/s, throughput, and preemption rate.
- TOML/JSON configuration serde with validation (context length, KV fraction, DP ranks).

#### CUDA Backend (`singularity-cuda`)
- `GpuBackend` trait abstracting over real CUDA and host-CPU mock backends.
- `HardwareDetector` with truthful SM capability interrogation (SM100 / SM90 / host fallback).
- `CudaGraph` runner for capturing and replaying decode steps.
- `MockGpuBackend` for CPU-only development and CI without NVIDIA hardware.
- `gemm_f32`, `copy_htod`, `copy_dtoh`, `copy_htod_offset`, `copy_dtoh_offset` primitives.
- Bounds-checked allocation and OOM rejection.
- `cuGraph` dependency via `cudarc 0.12` with `cuda-12050` feature.

#### KV Cache (`singularity-cache`)
- Paged `BlockManager` with bitset allocation, double-free protection, and reference counting.
- Pure-Rust Radix tree prefix cache with SHA256 sequence hash verification and LRU eviction.
- `FP4Layout` — NVFP4 E2M1 token-interleaved layout (288 bytes/token, 32-byte aligned) for SM100.
- `MXFP8Layout` — MXFP8 E4M3 token-interleaved layout (528 bytes/token, 32-byte aligned) for SM90.
- `SWABuffer` — Sliding window attention buffer supporting PE→CE state disaggregation.
- Device-backed contiguous KV block allocation.

#### Scheduler (`singularity-scheduler`)
- `ZoSPipeline` — Zero-Overhead Scheduler with CPU/GPU concurrency across forward steps.
- `LcfsScheduler` — Verified Last-Come-First-Served decode preemption (sort descending by output length, schedule shortest, preempt longest).
- `DpConsensus` — Two-tier ALL/ANY collective voting for DP-parallel prefill coordination.
- `MtpEstimator` — Multi-Token Prediction arrival rate estimator.
- `DeferredRelease` — Deferred sequence cleanup across the forward boundary.

#### Model (`singularity-model`)
- Zero-copy Safetensors `memmap2` loader with multi-shard index support.
- `RMSNorm`, `RoPE` (2D rotations), `SwiGLU` MLP primitives.
- `GqaAttention` — Grouped Query Attention with configurable heads.
- `MultiHeadLatentAttention` (FlashMLA) — Stage 2 dense low-rank projection over sparse tokens.
- `MqaIndexer` — Stage 1 FP4 sparse indexer with DeepSelect Top-K token selection.
- `SharedLayerTopology` — Non-adjacent layer projection weight sharing.
- `DeepSeekMoE` — `sqrtsoftplus` scoring, `noaux_tc` + correction biases, dynamic EPLB.
- `HuggingFaceTokenizerWrapper` with encode/decode fallback.
- DeepSeek-V4.1 Flash config loader (`deepseek-v41-flash-1017-config.json`).
- V4.1 test fixture generators: FP8 rounding, RoPE, MoE, engram, activation, sparse vectors.

#### Grammar (`singularity-grammar`)
- `KmpMatcher` — Knuth-Morris-Pratt state machine for streaming tool-call trigger/stop detection.
- `JsonBitmaskCompiler` — JSON Schema vocabulary bitmask with `-f32::INFINITY` logit masking.
- Constrained decoding activation inside `<tool_call>` / `</tool_call>` boundaries.

#### Server (`singularity-server`)
- Axum 0.7 REST/SSE server with OpenAI-compatible endpoints:
  - `POST /v1/chat/completions` (streaming + non-streaming)
  - `GET /v1/models`
  - `GET /health`
  - `GET /metrics`
- `StreamRingBuffer` — Bounded sliding window token replay with `Last-Event-ID` reconnect.
- `AdmissionLifecycle` — Eager TCP disconnect propagation to GPU sequence teardown.
- 4-state `CircuitBreaker`: `Healthy → Degraded → Exhausted → Disabled` with canary slots.
- `RateGovernor` — Leaky token bucket rate limiter.
- `TransactionalStore` — Atomic fsync + POSIX rename persistence with stale `.tmp.*` sweep.

#### CLI (`singularity-cli`)
- `singularity serve` — Start the HTTP server (real or dev-mock mode).
- `singularity benchmark` — Run transformer forward + scheduler microbenchmark.
- `singularity inspect-hardware` — Truthful GPU capability discovery with `--arch b200|h200` emulation.
- `singularity validate-config` — Configuration file validation.

#### Tests & Benchmarks (`singularity-tests`)
- 61 passing unit and integration tests across all crates.
- `e2e_server.rs` — Full HTTP server lifecycle test.
- `failure_modes.rs` — 18 concurrency and memory edge-case tests.
- `real_inference.rs` — SmolLM2-135M and DeepSeek-V2-Lite forward pass tests.
- `engine_benchmarks.rs` — Criterion microbenchmarks for scheduler and model paths.

#### Infrastructure
- Rust workspace (`Cargo.toml`) with resolver `"2"` and shared dependency table.
- `.gitignore` — Excludes `target/` (build artifacts) and editor/OS ephemera.
- Modal GPU deployment scripts (`deploy/modal/`) with NVIDIA GPU validation.
- Architecture state document (`ARCH_STATE.md`), phase audit reports (`PHASE_II_EXECUTION_AUDIT.md`, `PHASE_III_REPORT.md`), and research report (`RESEARCH_REPORT.md`).

---

[Unreleased]: https://github.com/CardSorting/HEAV3NS-BRAIN/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/CardSorting/HEAV3NS-BRAIN/releases/tag/v0.1.0
