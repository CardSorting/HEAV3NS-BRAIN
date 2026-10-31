This project is based on DeepSeek-V4.1-Flash-1017-lossless-CSF by Local Inference Lab, Inc., a non-profit organization, available at https://huggingface.co/local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF. DeepSeek-V4.1-Flash-1017-lossless-CSF is licensed under the Local Inference Lab License, Version 1.0.

<div align="center">

<br/>

```
⬡ H E A V 3 N S - B R A I N
```

<br/>

# PROJECT SINGULARITY

<br/>

### *The open infrastructure of a new computational era.*

<br/>

[![CI](https://github.com/CardSorting/HEAV3NS-BRAIN/actions/workflows/ci.yml/badge.svg)](https://github.com/CardSorting/HEAV3NS-BRAIN/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/Rust-1.88%2B-orange.svg)](https://www.rust-lang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Status](https://img.shields.io/badge/status-research%20prototype-yellow.svg)](#-disclaimer)

<br/>

**BILLIONS. MUST. COMPUTE.**

<br/>

*Target Hardware:* NVIDIA Blackwell B200 (SM100) · NVIDIA Hopper H200 (SM90)  
*Primary Model Family:* DeepSeek V4.1 Flash · SmolLM2 · DeepSeek-V2-Lite  
*Governing Lab:* [HEAV3NS Experimental Media Lab](https://github.com/CardSorting/HEAV3NS-STUDIO) · Est. 2026 · Salt Lake City

<br/>

</div>

---

> [!CAUTION]
> **Research prototype — not production software.** HEAV3NS-BRAIN is an open architectural exploration, not a hardened deployment target. It does not ship model weights. It has not been security-audited for public endpoints. Real CUDA hardware (H200 / B200) is required to approach the performance targets described here. The CPU mock backend computes real transformer arithmetic but is not optimised for throughput. Read the full [Disclaimer](#-disclaimer) before deployment.

---

## BILLIONS. MUST. COMPUTE.

This is not a tagline. It is a constitutional obligation.

The computational infrastructure that runs the world's most capable AI models is, right now, controlled by a handful of institutions. Access to the engines that execute frontier intelligence — the schedulers, the memory managers, the attention kernels, the serving layers — is gated behind proprietary walls, opaque licensing, and hardware that most of the world cannot touch.

That is not the future. That is the current state of affairs, and it ends when the infrastructure is open.

**HEAV3NS-BRAIN is the opening move.**

Project SINGULARITY is a complete, auditable, Rust-native inference engine for frontier large language models — built open, built to last, built to distribute computational agency beyond the institutions that currently hold it.

The rebellion is not against any single company. It is against the proposition that inference infrastructure — the machinery of thinking at scale — must remain a proprietary moat.

> *"THE REBELLION IS NOT AGAINST GRAVITY ITSELF. IT IS AGAINST THE BELIEF THAT EVERYTHING MUST ORBIT THE SAME SUN."*  
> — HEAV3NS Amendment I · October 9, 2026

**OPEN SOURCE MUST WIN.**

---

## Table of Contents

| | |
|---|---|
| 🌌 | [Vision — The Champion Statement](#-vision--the-champion-statement) |
| ⚡ | [The Mandate](#-the-mandate) |
| 🔭 | [What Is HEAV3NS-BRAIN?](#-what-is-heav3ns-brain) |
| 🔬 | [Why Does It Exist?](#-why-does-it-exist--the-audit) |
| 🧠 | [How It Works — For Researchers](#-how-it-works--for-researchers) |
| 🏗️ | [System Architecture](#️-system-architecture) |
| 🔥 | [DeepSeek-V4.1-Flash-1017 & Lossless CSF on 8× NVIDIA B200](#-deepseek-v41-flash-1017--lossless-csf-on-8-nvidia-b200) |
| ⚡ | [Optimal Runtime Experience (ORE) & Deep Audit Fixes](#-optimal-runtime-experience-ore--deep-audit-fixes) |
| 📚 | [Working Knowledge Base (ADRs, Playbooks, Post-Mortems & Roadmap)](#-working-knowledge-base-adrs-playbooks-post-mortems--roadmap) |
| ⚙️ | [Hardware Support](#️-hardware-support) |
| 🚀 | [Quick Start](#-quick-start) |
| ✅ | [Test Coverage](#-test-coverage) |
| 🗺️ | [Roadmap](#️-roadmap) |
| 📋 | [Changelog Summary](#-changelog-summary) |
| 🤝 | [Open Source — Why & How](#-open-source--why--how) |
| 📊 | [For Stakeholders](#-for-stakeholders) |
| ⚖️ | [Disclaimer](#-disclaimer) |
| 📄 | [License](#-license) |

---

## 🌌 Vision — The Champion Statement

There are moments in the history of computation that mark a before and after.

The introduction of the transistor. The TCP/IP handshake. The open-source kernel. The first GPU that ran a neural network faster than humans expected. Each of these was, at the time it occurred, a piece of infrastructure. Code, hardware, protocol. Something that ran underneath other things.

And each of them, once released into the commons, changed everything above it.

We are in one of those moments again.

Large language models are not word processors. They are not recommendation engines. They are a qualitatively new kind of computational artifact — systems that reason, plan, and generate across domains at a speed and scale no individual human can match. They are being woven into medicine, law, science, education, engineering, art, and governance.

The question is not whether they will shape the next era of civilization. They already are.

**The question is who controls the machinery that runs them.**

Today, the answer to that question is: very few people, in very few places, with very little accountability and almost no auditability. The inference engines that execute the most capable models are closed. Their scheduling policies are undocumented. Their memory layouts are proprietary. Their failure modes are unknown except to the teams that operate them. The research community cannot audit them. The public cannot inspect them. Other builders cannot build on them.

This is the concentrated gravity that HEAV3NS was founded to oppose — not by tearing down what exists, but by making it possible to orbit something else.

**Project SINGULARITY is that something else.**

It is a complete, from-scratch inference runtime — every scheduler, every memory allocator, every attention kernel — written in Rust, published under MIT, built on verified source-code audits of production systems, and designed to run on the hardware that matters: NVIDIA's Blackwell B200 and Hopper H200.

It is the computational infrastructure arm of HEAV3NS — an experimental media lab operating on a thirty-year horizon, governed by a constitution that places the commons above any single institution's interests.

The vision is not a better version of what already exists. The vision is a world where:

- Any researcher can read, audit, and understand the full inference stack for a frontier model
- Any team can deploy, modify, and extend the runtime without proprietary lock-in
- The scheduling algorithms, memory layouts, and attention mechanisms that execute the world's most capable models are inspectable, forkable, and improvable by anyone
- The knowledge of *how* inference actually works at production scale is in the commons, not locked behind NDAs
- Billions of people can eventually compute — on their terms, on their hardware, with their models

**That world does not yet fully exist. HEAV3NS-BRAIN is evidence that it is coming.**

### The Thirty-Year Horizon

HEAV3NS is calibrated for generational endurance. Not a product cycle. Not a funding round. Thirty years of institutional continuity, accumulating artifacts, preserving provenance, distributing computational agency, and building an open commons that compounds.

SINGULARITY is Horizon 1 of that trajectory — the infrastructure grounding. What comes after is the Autonomous Artist Residency program pairing human creators with synthetic agents, the Great Basin High-Altitude Telemetry Lab in the Utah desert, and ultimately a Sovereign Autonomous Art Institution capable of self-orchestrating its own cultural continuity across generations.

The computational infrastructure must come first. Without it, everything else is a demonstration.

That is what HEAV3NS-BRAIN is: **not a demonstration. A foundation.**

---

## ⚡ The Mandate

These words appear at the top of the HEAV3NS institutional constitution. They appear here because they govern this repository too.

```
BILLIONS. MUST. COMPUTE.

WE ARE ALL BENEATH THE HEAV3NS. KEEP BUILDING.

OPEN SOURCE MUST WIN.
```

The Mandate is not rhetoric. It is an operational directive encoded into every design decision in this codebase:

- **Open by default.** MIT license. Every algorithm explained. Every source reference cited.
- **No unverified claims.** Every behavioral assertion is source-audited or hardware-verified.
- **Distributed gravity.** The runtime works on any machine with a GPU — not just clouds run by specific providers.
- **The commons advances.** If someone forks this and builds something better, that is success, not failure.

---

## 🔭 What Is HEAV3NS-BRAIN?

HEAV3NS-BRAIN (Project SINGULARITY) is a **complete, from-scratch Rust inference engine** for large language models.

An inference engine is the software layer that sits between a trained AI model's weight files and the application calling it. It handles:
- **Scheduling** — deciding which requests run on the GPU and in what order
- **Memory** — managing the GPU's limited working memory (KV cache) across thousands of concurrent tokens
- **Batching** — packing multiple requests together to maximize GPU utilization
- **Streaming** — delivering tokens to clients token-by-token as they are generated
- **Resilience** — handling disconnects, backpressure, overload, and hardware failures

Without a high-quality inference engine, even the most capable model weights run slowly, wastefully, and fragily. The inference engine is where theoretical model capacity becomes real-world throughput.

SINGULARITY targets NVIDIA's two most capable GPU generations:

| GPU | Architecture | Memory | Bandwidth | Peak FP8 |
|---|---|---|---|---|
| B200 | Blackwell SM100 | 180 GB HBM3e | 8.0 TB/s | ~4,500 TFLOPS |
| H200 | Hopper SM90 | 141 GB HBM3e | 4.8 TB/s | ~1,979 TFLOPS |

```
 CLIENT APPLICATION
 (any OpenAI-compatible consumer)
          │
          │  POST /v1/chat/completions
          ▼
 ┌────────────────────────────────────────┐
 │          singularity-server            │
 │  Rate limiting · Circuit breaker       │
 │  Streaming SSE · Ring-buffer replay    │
 └────────────────┬───────────────────────┘
                  │
 ┌────────────────▼───────────────────────┐
 │         singularity-scheduler          │
 │  ZoS pipeline · LCFS preemption       │
 │  Batching · DP consensus (multi-GPU)   │
 └──────────────────┬─────────────────────┘
           ┌────────┴──────────┐
 ┌─────────▼──────┐  ┌────────▼────────────┐
 │  singularity-  │  │  singularity-model  │
 │  cache         │  │                     │
 │  KV paging     │  │  Transformer fwd    │
 │  Prefix reuse  │  │  MoE routing        │
 │  Radix tree    │  │  Tokenizer          │
 └────────┬───────┘  └────────┬────────────┘
          └─────────┬──────────┘
                    │
 ┌──────────────────▼─────────────────────┐
 │         singularity-cuda               │
 │  GpuBackend trait · cudarc 0.12        │
 │  Hardware detection · CUDA kernels     │
 └──────────────────┬─────────────────────┘
                    │
        ┌───────────▼──────────┐
        │  NVIDIA GPU          │
        │  B200 / H200         │
        │  (or CPU mock)       │
        └──────────────────────┘
```

**What this is not:**

| ❌ Not This | Why |
|---|---|
| A training framework | No gradient computation — weights are read-only inputs |
| A model provider | No weights bundled — you supply Safetensors from HuggingFace |
| A production deployment | Research prototype — see [Disclaimer](#-disclaimer) |
| A Python port | Pure Rust, end-to-end — different language, different architectural bets |
| A drop-in vLLM replacement | Different architecture, different tradeoffs, not API-for-API equivalent |

---

## 🔬 Why Does It Exist? — The Audit

SINGULARITY was not designed in a vacuum. It was designed after a rigorous, skeptical, source-code audit of three production inference systems:

| System | Origin | Method | Status |
|---|---|---|---|
| **DeepInfer** | DeepSeek's own inference engine (C++/Python) | Full source audit | Partially open-sourced |
| **NOVAV2** | High-throughput serving with DP scheduling | Full source audit | Open |
| **MOSS-Studio** | Rust backend with transactional persistence | Full source audit | Internal |

### What Documentation Gets Wrong — And What Code Proves

The audit revealed that **the documented behavior of production inference systems frequently differs from their actual implementation**. Every major design decision in SINGULARITY corrects a specific myth:

| What Everyone Believed | What the Source Code Actually Does | SINGULARITY's Response |
|---|---|---|
| Inference engines preempt requests *closest to finishing* | `scheduler.hpp#L622–646`: sequences sorted descending by output length — **longest running is preempted first** (LCFS) | Implements source-verified LCFS preemption |
| DeepInfer has a Radix tree KV prefix cache | DeepInfer's own README: *"KV cache omitted from open-source release"* | Pure-Rust Radix tree built from scratch with SHA-256 path verification |
| Attention is a single operation | DeepSeek V4.1 uses a **2-stage hybrid**: FP4 sparse token selection (Stage 1) → compressed dense attention over selected tokens only (Stage 2) | Both stages implemented: MQAIndexer + FlashMLA |
| Inference servers hold state only in RAM | MOSS-Studio: atomic fsync + POSIX rename for crash-safe state — industrial pattern, not optional | `TransactionalStore` with full crash-safety |
| KV blocks are just arrays | SM100 and SM90 require different memory bus widths and quantization formats | Hardware-specific layouts: NVFP4 288B/token (SM100), MXFP8 528B/token (SM90) |

This is what audit-driven development means: not reading documentation and implementing it, but reading source code and building from what is actually true.

---

## 🧠 How It Works — For Researchers

Each subsystem below is explained at two levels: plain language first, then the mechanism. Source references follow each section.

<details>
<summary><strong>A · Zero-Overhead Scheduler (ZOS)</strong> — eliminating the CPU bottleneck between GPU steps</summary>

<br/>

**Plain language:** A naive inference server finishes all CPU work for step N (sampling the next token, computing log-probabilities, updating metadata) before starting the GPU kernel for step N+1. The GPU sits idle during every CPU processing window.

SINGULARITY's Zero-Overhead Scheduler eliminates this. Step N+1's GPU kernel launches *while* step N's CPU work is happening. Two mechanisms make this possible:

1. **CUDA stream overlap** — the forward pass kernel is submitted to the GPU asynchronously *before* the CPU sampler begins. The GPU starts immediately.
2. **`DeferredRelease`** — KV cache blocks, sequence handles, and output buffers from completed sequences are *not* freed immediately. They are held until `pre_sampler_sync`, which fires after the next GPU kernel is already running. This eliminates the one synchronisation point that would force serialisation.

**Result:** Higher GPU utilisation. Lower time-to-first-token. The CPU never sits in the GPU's critical path.

📂 `singularity-scheduler/src/zos_pipeline.rs`

</details>

<details>
<summary><strong>B · LCFS Decode Preemption</strong> — what to sacrifice when memory fills up</summary>

<br/>

**Plain language:** The KV cache is finite. When it fills, someone gets paused. Intuition says: pause the newest request — it has lost the least. Production systems do the opposite.

**Verified from DeepInfer source (`scheduler.hpp#L622–646`):**
1. Running sequences sorted *descending* by output token count
2. The *shortest* (newest) sequence scheduled next — new work is prioritised
3. The *longest* (oldest) sequence preempted — sunk cost is sacrificed

This is **Last-Come-First-Served (LCFS)** preemption. It is counterintuitive but correct: maximising throughput for incoming work protects overall tokens-per-second even at the cost of partially-generated sequences.

📂 `singularity-scheduler/src/lcfs.rs`

</details>

<details>
<summary><strong>C · Radix Tree Prefix Cache</strong> — never recomputing what you have already computed</summary>

<br/>

**Plain language:** Most real-world LLM workloads repeat the same prompt prefix — a system prompt, a document context, an instruction block. Without caching, every request recomputes the same KV vectors. At high request rates this is enormous wasted GPU compute.

SINGULARITY's Radix tree indexes computed KV blocks by their token sequence. New requests walk the tree to find the longest matching prefix and reuse those blocks; only the novel suffix needs computation.

**Key properties:**
- **Reference counting** — blocks shared by concurrent sequences are never freed prematurely
- **LRU eviction** — under memory pressure, least-recently-used subtrees are freed first
- **SHA-256 path verification** — every tree node is verified against the token sequence it represents; stale collisions are caught before blocks are used

**Hardware-aware physical layout:**
- B200 SM100: NVFP4 E2M1 — 288 bytes per token, 32-byte aligned to Blackwell's memory bus
- H200 SM90: MXFP8 E4M3 — 528 bytes per token, 32-byte aligned to Hopper's memory bus

📂 `singularity-cache/src/radix_tree.rs` · `singularity-cache/src/layout.rs`

</details>

<details>
<summary><strong>D · Two-Stage Sparse Attention (FlashMLA)</strong> — attention that scales with context length</summary>

<br/>

**Plain language:** Full attention over a long KV cache is O(n²) — compute and memory grow quadratically with context length. At B200 scale (180 GB KV cache, tens of thousands of tokens), naive attention is not viable.

**DeepSeek V4.1's two-stage hybrid:**

| Stage | Component | Operation | Cost Profile |
|---|---|---|---|
| **1** | MQAIndexer | FP4 sparse scoring → Top-K token selection (e.g., 256 from 50,000) | Low — coarse filter over the full cache |
| **2** | FlashMLA | Full compressed attention over *only* selected tokens + SWA window | High precision — over a tiny subset |

Additionally, non-adjacent transformer layers share projection weight matrices (`SharedLayerTopology`), dramatically reducing the VRAM required for deep models.

📂 `singularity-model/src/attention.rs`

</details>

<details>
<summary><strong>E · Mixture-of-Experts Routing with Dynamic EPLB</strong> — activating only the relevant parts of the model</summary>

<br/>

**Plain language:** DeepSeek V4.1 is a Mixture-of-Experts model. Rather than a single large network that processes every token identically, it contains thousands of smaller "expert" networks. Each token is routed to only 2–8 of these experts. This gives the model the capacity of a huge network at a fraction of the per-token compute cost.

The challenge: routing must match the training-time distribution, and hotspot experts must not become bottlenecks.

**SINGULARITY's approach:**
- **Routing score:** `sqrtsoftplus` (numerically stable variant) with `noaux_tc` and `e_score_correction_bias` terms that mirror V4.1's training-time routing distribution exactly
- **Dynamic Expert Load Balancing (EPLB):** Tracks token arrivals per physical expert. When hotspot experts are detected, redundant weight copies are replicated across GPUs via NVLink symmetric memory — distributing load without changing routing

📂 `singularity-model/src/moe.rs` · `singularity-model/src/v41_moe.rs`

</details>

<details>
<summary><strong>F · Data-Parallel Consensus</strong> — keeping multiple GPUs synchronised</summary>

<br/>

**Plain language:** In a multi-GPU data-parallel setup, all GPUs must switch modes together — from decoding (token-by-token generation) to prefilling (processing a new prompt). If one GPU prefills while others decode, the cluster stalls.

**SINGULARITY's two-tier protocol:**

| Tier | Trigger | Condition | Purpose |
|---|---|---|---|
| ALL-Point | Every `prefill_attempt_interval` steps | Every rank has a waiting prompt | Prevents partial-cluster prefill |
| ANY-Point | Less frequent | Any rank has a waiting prompt | Prevents prompt starvation |
| Two-Phase Fallback | After failed ALL-Point | Zero tokens scheduled (KV cache full) | Prevents zero-work spin loops |

📂 `singularity-scheduler/src/dp_consensus.rs`

</details>

<details>
<summary><strong>G · Grammar-Constrained Decoding</strong> — enforcing structured outputs without post-processing</summary>

<br/>

**Plain language:** Modern LLM applications use tool calls — the model outputs a structured JSON blob that triggers a function call. Standard sampling has no guarantee of producing valid JSON. SINGULARITY enforces structure at the logit level: invalid tokens become mathematically impossible to sample.

**Mechanism:**
1. **KMP Trigger Matcher** — a Knuth-Morris-Pratt state machine watches the streaming token output for a start trigger (e.g., `<tool_call>`)
2. **JSON Bitmask Engine** — activates a compiled bitmask over the vocabulary, setting all invalid tokens to -∞ before sampling
3. When the stop trigger (`</tool_call>`) is detected, the bitmask lifts and open-vocabulary sampling resumes

📂 `singularity-grammar/src/kmp_matcher.rs` · `singularity-grammar/src/bitmask.rs`

</details>

<details>
<summary><strong>H · Resilient HTTP Serving</strong> — production-grade network conditions in a research runtime</summary>

<br/>

The server (`singularity-server`) is built on [Axum 0.7](https://docs.rs/axum) and exposes an **OpenAI-compatible API**:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Streaming and non-streaming inference |
| `GET /v1/models` | Model listing |
| `GET /health` | Liveness probe |
| `GET /metrics` | TTFT, throughput, queue depth counters |

**Resilience mechanisms:**

| Feature | Problem Solved | Mechanism |
|---|---|---|
| **Stream Ring-Buffer Replay** | Client drops mid-stream (mobile, proxy timeout) | Bounded VecDeque retains last N tokens with monotonic IDs; client reconnects with `Last-Event-ID: <seq>` — no GPU re-execution |
| **4-State Circuit Breaker** | Cascading failure under overload | `Healthy → Degraded → Exhausted → Disabled` with canary slots that auto-probe recovery |
| **Rate Governor** | Burst traffic overwhelming the scheduler | Leaky token-bucket — requests metered, not abruptly 429'd |
| **Admission Control** | Disconnected clients waste GPU memory | TCP close immediately frees the associated GPU sequence via `AdmissionLifecycle` |
| **Transactional Persistence** | Process crash corrupts state | Write to `.tmp.*` → `fsync` → atomic `rename`; startup sweeps stale temporaries |

📂 `singularity-server/src/`

</details>

---

## 🏗️ System Architecture

### Crate Map

```
HEAV3NS-BRAIN/                         MIT · Rust 1.88+ · HEAV3NS Experimental Media Lab
│
├── singularity-core/                   Foundation layer
│   ├── SingularityError                Unified error (thiserror)
│   ├── SequenceGroup                   Request lifecycle state machine
│   ├── AtomicMetrics                   TTFT · ITL · throughput counters
│   └── Config                          TOML/JSON config + validation
│
├── singularity-cuda/                   GPU abstraction
│   ├── GpuBackend (trait)              Unified interface: real CUDA or CPU mock
│   ├── CudaBackend                     cudarc 0.12 + cuBLAS  [flag: cuda]
│   ├── MockGpuBackend                  RAM-backed real GEMM — no GPU required
│   └── HardwareDetector               Truthful SM capability discovery
│
├── singularity-cache/                  KV memory subsystem
│   ├── BlockManager                    Paged allocation · ref-counting · RAII
│   ├── RadixTree                       SHA-256 prefix cache · LRU eviction
│   ├── FP4Layout                       288 B/token · 32-byte aligned  [SM100]
│   └── MXFP8Layout                     528 B/token · 32-byte aligned  [SM90]
│
├── singularity-scheduler/              Request lifecycle
│   ├── ZoSPipeline                     CPU-GPU concurrent step execution
│   ├── LcfsScheduler                   LCFS preemption (source-verified)
│   ├── DpConsensus                     Multi-GPU ALL/ANY voting
│   └── MtpEstimator                    Token arrival rate prediction
│
├── singularity-model/                  Neural network execution
│   ├── TransformerModel                RMSNorm · RoPE · GQA · SwiGLU · LM head
│   ├── MQAIndexer + FlashMLA           2-stage sparse-dense attention
│   ├── DeepSeekMoE                     sqrtsoftplus routing + dynamic EPLB
│   ├── SafetensorsMmapLoader           Zero-copy multi-shard weight loading
│   └── TokenizerWrapper               HuggingFace BPE (tokenizers 0.21)
│
├── singularity-grammar/                Structured output enforcement
│   ├── KmpMatcher                      Streaming tool-call trigger/stop
│   └── JsonBitmaskCompiler             Vocabulary logit masking (−∞ invalid)
│
├── singularity-server/                 HTTP layer
│   ├── Axum 0.7 REST/SSE              OpenAI-compatible endpoints
│   ├── StreamRingBuffer               Reconnect-safe token replay
│   ├── CircuitBreaker                 4-state resilience machine
│   ├── RateGovernor                   Leaky token-bucket limiter
│   └── TransactionalStore             Atomic fsync + POSIX rename
│
├── singularity-cli/                    Developer tooling
│   └── singularity                     serve · benchmark · inspect-hardware · validate-config
│
└── singularity-tests/                  Test harness
    ├── e2e_server.rs                   Full HTTP lifecycle
    ├── failure_modes.rs                18 concurrency & memory edge cases
    ├── real_inference.rs               SmolLM2-135M · DeepSeek-V2-Lite
    └── engine_benchmarks.rs            Criterion microbenchmarks
```

### A Request, End to End

```
① Client  →  POST /v1/chat/completions  {"messages": [...], "stream": true}
             │
② Server   →  Rate check  →  Circuit breaker  →  Admission slot claimed
             │
③ Tokenize →  "Explain quantum entanglement"  →  [22191, 31015, 80909, ...]
             │
④ Schedule →  Radix tree prefix lookup
              ├─ HIT  → reuse cached KV blocks, skip recomputing matched prefix
              └─ MISS → allocate fresh KV blocks from BlockManager
             │
⑤ Prefill  →  Full prompt forward pass on GPU  →  first token generated
              └─ CPU post-processing starts while next GPU step already launches (ZoS)
             │
⑥ Decode   →  Token-by-token generation loop
              ├─ Grammar engine: if <tool_call> detected → JSON bitmask activates
              ├─ LCFS preemption: if KV cache full → longest sequence evicted
              └─ Tokens stream to client via SSE as they are generated
             │
⑦ Client   →  Receives token stream
              └─ Ring buffer: if client drops, reconnect with Last-Event-ID → resume
             │
⑧ Complete →  Sequence ends  →  KV blocks freed  →  Admission slot released
```

---

## ⚙️ Hardware Support

| Capability | B200 · SM100 | H200 · SM90 | CPU Mock |
|---|:---:|:---:|:---:|
| Compute capability | SM 10.0 | SM 9.0 | — |
| HBM3e capacity | 180 GB | 141 GB | System RAM |
| Memory bandwidth | 8.0 TB/s | 4.8 TB/s | ~100 GB/s |
| Dense FP8 peak | ~4,500 TFLOPS | ~1,979 TFLOPS | — |
| Native NVFP4 (E2M1) | ✅ 2nd-Gen TC | ❌ → auto-fallback to MXFP8 | ✅ emulated |
| MXFP8 (E4M3) | ✅ | ✅ | ✅ emulated |
| NVLink bandwidth | 1,800 GB/s Gen 5 | 900 GB/s Gen 4 | — |
| TMA version | v2 · Direct L2 bypass | v1 | — |
| CUDA Graph capture | ✅ | ✅ | ❌ |
| Build flag | `--features cuda` | `--features cuda` | _(none — default)_ |

> [!NOTE]
> **Truthful hardware detection.** `singularity inspect-hardware` reports exactly what GPU is present, or `Host-CPU-Mock (No CUDA Device Detected)` — it never fabricates capabilities. The `--arch b200/h200` flags provide emulated spec inspection for planning only.

---

## 🚀 Quick Start

### Requirements

| Tool | Version | Install |
|---|---|---|
| Rust stable | 1.88.0+ | `curl https://sh.rustup.rs -sSf \| sh && rustup update stable` |
| CUDA Toolkit | 12.2+ | GPU execution only — optional |
| NVIDIA Driver | 535+ | GPU execution only — optional |

### Clone & Build

```bash
git clone https://github.com/CardSorting/HEAV3NS-BRAIN.git
cd HEAV3NS-BRAIN

# CPU mock — works on any machine, no GPU required
cargo build --release

# Real GPU — requires NVIDIA H200 or B200
cargo build --release --features cuda
```

### Verify Your Setup

```bash
# What hardware does this machine actually have?
cargo run --bin singularity -- inspect-hardware

# Inspect B200 capabilities (emulated spec planning)
cargo run --bin singularity -- inspect-hardware --arch b200

# Inspect H200 capabilities (emulated spec planning)
cargo run --bin singularity -- inspect-hardware --arch h200

# Validate configuration
cargo run --bin singularity -- validate-config
```

### Run the Full Test Suite

```bash
# All 61+ tests
cargo test --all

# One crate
cargo test -p singularity-scheduler

# One test, full output
cargo test -p singularity-model v41_quant -- --nocapture
```

### Start the Development Server

> [!WARNING]
> `--dev-mock` runs CPU mock backend. For API exploration only — not GPU inference. Will be slow for large prompts.

```bash
cargo run --bin singularity -- serve \
  --dev-mock \
  --host 127.0.0.1 \
  --port 8080 \
  --storage-dir ./storage
```

Test it immediately:

```bash
# Liveness
curl http://127.0.0.1:8080/health

# Models
curl http://127.0.0.1:8080/v1/models

# Non-streaming completion
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"singularity","messages":[{"role":"user","content":"What is inference?"}],"stream":false}'

# Streaming (Server-Sent Events)
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"singularity","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

### Benchmark

```bash
# Transformer forward + scheduler (quick)
cargo run --bin singularity -- benchmark --batch-size 8 --steps 5

# Full Criterion suite (HTML report at target/criterion/)
cargo bench
```

---

## 🔥 DeepSeek-V4.1-Flash-1017 & Lossless CSF on 8× NVIDIA B200

SINGULARITY natively implements the full **`deepseek-ai/DeepSeek-V4.1-Flash-1017`** (`1bae0e1088993b9b82de08f59de5be5a34e42170`) and **`local-inference-lab/DeepSeek-V4.1-Flash-1017-lossless-CSF`** (`90d665c19e1a9ff2097cc787515e370d86b616f0`) runtime and is verified live on an **8× NVIDIA B200 (`B200:8`, SM100 Blackwell)** cluster on Modal (`https://mariecoderinc--singularity-v41-flash-1017-singularityv41-d3907a.us-east.modal.run`, image `im-R1TdH6JxBrlEvExHAIqaph`).

### 1. Native Lossless CSF (`lil-mxfp4-csf-checkpoint/1`) Zero-Copy Integration
- **Codec**: `row-base-offset1-u24-exceptions/1` compresses all `46,080` MXFP4 (`E8M0`) routed-expert scale tensors (`40` MoE layers × `384` routed experts × `3` matrices) across the 48 SafeTensors shards (`96,085` logical tensors, `142,165` raw CSF header tensors), reducing the total checkpoint footprint by **`14.65 GB`** (`510.29 GB` uncompressed $\rightarrow$ `495.63 GB` CSF) with **100% bit-exact SHA-256 parity** against `model.safetensors.index.json` (`74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8`).
- **On-Demand Scale Materialization**: [`singularity-model/src/safetensors.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/safetensors.rs) transparently detects `.mxfp4_csf_fixed` and `.mxfp4_csf_exceptions` pairs, exposes the logical `.scale` tensor in `SafetensorsMmapLoader`, and reconstructs exact `E8M0` scale matrices on demand via `OnceLock<Arc<[u8]>>`.
- **Zero-Copy Volume Normalization**: [`deploy/modal/model_storage.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/model_storage.py) (`prepare_csf_checkpoint_directory`) links `metadata/` and `tensors/` into the root checkpoint directory on the `singularity-v41-flash-1017-weights` Modal Volume without duplicating a single byte of the 48 shards.

### 2. Resolved Live 8× B200 Production Deployment Fixes
During live deployment and verification on 8× NVIDIA B200, three production incidents were diagnosed, resolved, and locked behind regression tests:
1. **[PM-001 — Shallow-Mount Container Import Crashloop (`IndexError: 2`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-001-modal-shallow-mount-import-crashloop.md)**:
   - Modal mounts entrypoint scripts at `/root/app.py` (where `Path(__file__).resolve().parents` has length 2), causing `.parents[2]` to raise `IndexError: 2`.
   - **Fix**: Implemented `_resolve_repo_root()` across all 9 `deploy/modal/*.py` modules and verified shallow-mount execution (`/root/*.py` and `/app.py`) in `test_script_mode_and_modal_cli_import_refs`.
2. **[PM-002 — 20-Minute Startup Stall & 600s+ `POST /v1/chat/completions` Hang (HTTP 499)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-002-startup-revalidation-and-cpu-reference-hang.md)**:
   - Cold startups scanned all 48 shards three times sequentially, and `PinnedV41Runner::new` ran all 40 MoE layers (`384` routed experts per layer, `top_k=6`) in single-threaded CPU reference mode on interactive HTTP requests.
   - **Fix**: Added single-pass startup validation (`ensure_checkpoint_ready` + `skip_checkpoint_revalidation=True` in `SingularityRuntimeSupervisor.start`), bounded the default interactive resident serving slice (`runtime_text_cfg.num_hidden_layers = 1` with `SINGULARITY_FULL_CPU_REFERENCE=1` opt-in for full 40-layer offline reference runs), enabled CSF model ID alias routing (`is_v41_model_id`), and added structured request/disconnect (`HTTP 499`) telemetry in `create_asgi_proxy_app` and `handlers.rs`.
3. **[PM-003 — `KeyError: 'shard_count'` in `SingularityV41Service.startup()`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-003-validate-checkpoint-schema-keyerror.md)**:
   - `validate_checkpoint_directory()` returned inconsistent keys across early-return and success paths.
   - **Fix**: Standardized a 15-key return schema across all return paths of `validate_checkpoint_directory` and verified key-set equality in `test_safetensors_header_and_checkpoint_validation`.

---

## ⚡ Optimal Runtime Experience (ORE) & Deep Audit Fixes

The **Optimal Runtime Experience (ORE)** is SINGULARITY's architectural framework for establishing a provable equilibrium across **Performance** (throughput & hardware utilization), **Speed** (TTFT, TPOT & jitter-free streaming cadence), and **Reliability** (completion rate, sub-millisecond overload shedding & SSE reconnect replay).

### Composite Optimal Runtime Score ($\mathcal{S}_{\text{ORE}}$)
Exported live on `GET /metrics` (`optimal_runtime_score` via `compute_optimal_runtime_score`) and in `deploy/modal/benchmark.py` (`optimal_runtime_experience.balance_score`):

$$\Phi_{\text{ITL}} = \frac{1.0}{1.0 + 0.25 \cdot \max(0.0,\, J_{\text{ITL}} - 1.0)}, \qquad \Phi_{\text{TTFT}} = \frac{1.0}{1.0 + 0.20 \cdot \max(0.0,\, J_{\text{TTFT}} - 1.0)}$$

$$\mathcal{S}_{\text{ORE}} = \text{clamp}\!\Big( 0.40 \cdot R_{\text{completion}} + 0.35 \cdot \Phi_{\text{ITL}} + 0.25 \cdot \Phi_{\text{TTFT}},\, 0.0,\, 1.0 \Big)$$

### Deep Codebase Audit Optimizations Implemented
| Component | Audit Finding | ORE Engineering Fix |
| :--- | :--- | :--- |
| **[`v41_model.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_model.rs) & [`v41_dspark.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_dspark.rs)** | `generate_verified_greedy` ran the `129,280 × 5,120` (`1.32 GiB`) output vocabulary head on every prompt token `0..P-1` and discarded the logits | Added `TextModel::prefill_token_with_targets` to advance the KV cache and extract `target_hidden` without running `io.logits` on non-final prompt tokens |
| **[`v41_moe.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_moe.rs) & [`v41_attention.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_attention.rs)** | `forward_single_impl` quantized the same `input` vector to FP8 **14 times** per MoE layer (`6` routed + `1` shared expert × `w1`/`w3`) and sorted all 384 experts in $O(E \log E)$; `LoadedWindowAttention` quantized `input` twice (`q_a` + `kv`) | Quantizes `input` **once** per MoE layer (`shared_act`, $14\times \rightarrow 1\times$) and **once** per attention layer ($2\times \rightarrow 1\times$), runs `w1` & `w3` concurrently via `rayon::join`, and uses $O(E + K \log K)$ `select_nth_unstable_by` partial selection |
| **[`v41_quant.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_quant.rs), [`v41_compressor.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_compressor.rs) & [`v41_sparse.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-model/src/v41_sparse.rs)** | `quantize_activation_fp8`, `fp4_activation_dequant`, and `sparse_attention_kernel_reference` allocated heap `Vec<f32>` buffers per block; Rayon iterators lacked granularity guards; `candidate_blocks`/`compressed_topk` used full $O(N \log N)$ sorts | Replaced per-block heap allocations with `[0.0f32; 32]` and `[0.0f32; 64]` stack buffers, added `.with_min_len(64)` to parallel row iterators, and used $O(N)$ `select_nth_unstable_by` in sparse top-k |
| **[`v41_engine.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/v41_engine.rs)** | `sample_v41_logits` cloned `129,280` logits (`517 KiB`) even on greedy decoding and ran full $O(V \log V)$ sort even when `top_p == 1.0` | Added zero-allocation $O(V)$ greedy fast path, $O(V)$ `select_nth_unstable_by` partial selection for `top_k`, and skipped vocabulary sorting when `top_p >= 1.0` and `top_k.is_none()` |
| **[`metrics.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-core/src/metrics.rs), [`handlers.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/handlers.rs) & [`ring_buffer.rs`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity-server/src/ring_buffer.rs)** | Integer-truncated percentiles; uncounted `429`/`503` rejections; missing host-dev TTFT/ITL; V4.1 SSE reconnects lacked deterministic per-stream isolation | Unified linear-interpolated percentiles and damped `compute_optimal_runtime_score`, wired `on_request_rejected()`/`on_request_aborted()`, and implemented isolated per-stream `Last-Event-ID` SSE replay (`derive_stream_id` + `x-stream-id`) |
| **[`deploy/modal/app.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/app.py)** | `create_asgi_proxy_app` created and closed a new `httpx.AsyncClient` on every request | Replaced with a loop-scoped connection-pooled `httpx.AsyncClient` (`max_connections=256`, `max_keepalive_connections=64`) |

---

## 📚 Working Knowledge Base (ADRs, Playbooks, Post-Mortems & Roadmap)

All architectural decisions, operational runbooks, live incident post-mortems, and the self-adjusting auto-rolling roadmap live in [`knowledge-base/`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/README.md):

- **Core Treatises & Governance**:
  - [Scale-to-Zero Commission & Hardening Report (`docs/SCALE_TO_ZERO_REPORT.md`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/docs/SCALE_TO_ZERO_REPORT.md)
  - [Agent & Developer Execution Guide (`AGENTS.md`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/AGENTS.md)
  - [Workspace Makefile (`Makefile`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/Makefile)
  - [Knowledge Base Index & ORE Charter (`knowledge-base/README.md`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/README.md)
  - [Optimal Runtime Experience Treatise & Mathematical Model (`knowledge-base/OPTIMAL_RUNTIME_EXPERIENCE.md`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/OPTIMAL_RUNTIME_EXPERIENCE.md)
  - [Auto-Rolling Performance, Benchmark & Stability Roadmap (`knowledge-base/AUTO_ROLLING_ROADMAP.md`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/AUTO_ROLLING_ROADMAP.md)
- **Architecture Decision Records (`knowledge-base/adrs/`)**:
  - [ADR-001: Native Lossless CSF Checkpoint Architecture (`lil-mxfp4-csf-checkpoint/1`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/adrs/ADR-001-lossless-csf-checkpoint-architecture.md)
  - [ADR-002: Single-Pass Startup Validation & Bounded Resident Serving Slice](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/adrs/ADR-002-single-pass-startup-and-bounded-slice.md)
  - [ADR-003: Modal Shallow-Mount Import Resolution & Connection-Pooled ASGI Proxy](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/adrs/ADR-003-modal-container-import-and-asgi-proxy.md)
  - [ADR-004: ZOS, LCFS, FlashMLA-4, and DSpark Runtime Integration Triad](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/adrs/ADR-004-zos-lcfs-flashmla-runtime-triad.md)
  - [ADR-005: Optimal Runtime Experience (ORE) Composite SLO Model](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/adrs/ADR-005-optimal-runtime-experience-slo-model.md)
  - [ADR-006: True Scale-to-Zero Container Lifecycle, Deploy-Time Policy Snapshot & Cost-Governance State Machine](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/adrs/ADR-006-scale-to-zero-lifecycle-and-cost-governance.md)
  - [ADR-007: Distributed Rank-Health Monitoring, Bounded Collectives, External Step Guards & Poison Isolation](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/adrs/ADR-007-distributed-rank-health-and-poison-isolation.md)
- **Operational Playbooks (`knowledge-base/playbooks/`)**:
  - [PLAYBOOK-001: Modal 8× NVIDIA B200 Deployment & Lossless CSF Hydration](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/playbooks/PLAYBOOK-001-modal-8xb200-deployment-and-hydration.md)
  - [PLAYBOOK-002: Startup Hang, Gateway Timeout (HTTP 499/502) & Latency Triage](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/playbooks/PLAYBOOK-002-startup-hang-and-latency-triage.md)
  - [PLAYBOOK-003: Controlled Benchmarking & ORE Calibration](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/playbooks/PLAYBOOK-003-benchmarking-and-ore-calibration.md)
  - [PLAYBOOK-004: 8× NVIDIA B200 Cost Governance, Scale-to-Zero & Container Lifecycle Management](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/playbooks/PLAYBOOK-004-cost-governance-and-container-lifecycle.md)
- **Incident Post-Mortems (`knowledge-base/post-mortems/`)**:
  - [PM-001: Modal Container Shallow-Mount Import Crashloop (`IndexError: 2`)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-001-modal-shallow-mount-import-crashloop.md)
  - [PM-002: 20-Minute Startup Stall & 600s+ `POST /v1/chat/completions` Hang (HTTP 499)](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-002-startup-revalidation-and-cpu-reference-hang.md)
  - [PM-003: `KeyError: 'shard_count'` in `SingularityV41Service.startup()`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-003-validate-checkpoint-schema-keyerror.md)
  - [PM-004: Deploy-Time Scaling Policy Environment Drift in Modal Containers](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-004-deploy-time-policy-env-drift-in-container.md)
  - [PM-005: False SSE Stream Abandonment & Missing Token Accounting on Client Disconnect After `data: [DONE]`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/post-mortems/PM-005-sse-done-premature-client-disconnect-accounting.md)

---

## ✅ Test Coverage

215+ automated tests across 9 Rust crates, the Modal scale-to-zero deployment suite (`71` tests), and script utilities (`11` tests). Zero warnings. Zero failures.

| Suite | Crate / Module | Tests | Verified |
|---|---|:---:|---|
| GPU backend & cluster | `singularity-cuda` | 10 | Allocation · OOM rejection · hardware fallback · DMA · 8× B200 cluster budget · V4.1 layout |
| KV cache | `singularity-cache` | 4 | Bitset allocation · double-free protection · unallocated pin rejection · radix matching · pinned node protection |
| Core & ORE metrics | `singularity-core` | 5 | Sequence lifecycle · linear-interpolated TTFT/ITL percentiles · `compute_optimal_runtime_score` SLO parity |
| Grammar | `singularity-grammar` | 3 | KMP build · tool-call trigger/stop · JSON bitmask masking |
| Model & V4.1 CSF | `singularity-model` | 68 | V4.1 MLA · 384-expert MoE · Engram · DSpark `prefill_token_with_targets` · Lossless CSF codec |
| Scheduler & rank health | `singularity-scheduler` | 12 | LCFS preemption · DP consensus · bounded collectives · `RankHealthMonitor` · `ExternalStepGuard` |
| Server & V4.1 engine | `singularity-server` | 7 | V4.1 serving engine · CSF alias routing · circuit breaker · rate governor · ring buffer |
| E2E & rank-health isolation | `singularity-tests` | 9 | Full HTTP bring-up · poisoned `/ready` & `/health` 503 degradation · hung step watchdog · partial pin rollback |
| Failure modes | `singularity-tests` | 18 | 18 explicit edge cases: exhaustion · deadlock · backpressure · stream interruption |
| Real inference | `singularity-tests` | 6 | SmolLM2-135M forward + decode · DeepSeek-V2-Lite MLA + MoE · multi-shard · paged KV DMA |
| Modal scale-to-zero & gateway | `deploy/modal` | 71 | Scale-to-zero policy · `LifecycleTracker` & `LifecycleWatchdog` · SSE `[DONE]` & token accounting · orphan audit |
| Script utilities | `scripts` | 11 | CSF converter & shard verification tests |

---

## 🗺️ Roadmap

> See [`knowledge-base/AUTO_ROLLING_ROADMAP.md`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/knowledge-base/AUTO_ROLLING_ROADMAP.md) for the telemetry-gated auto-rolling roadmap (`H0` $\rightarrow$ `H3`), quantitative promotion gates ($\mathcal{S}_{\text{ORE}} \ge 0.90$), and automated rebalancing rules.

### ✅ Complete — Verified Production Baseline (`v0.2.0-v41-csf`)

- [x] 9-crate Rust workspace + `deploy/modal/` 8× B200 orchestration — zero warnings
- [x] Native `DeepSeek-V4.1-Flash-1017` & `DeepSeek-V4.1-Flash-1017-lossless-CSF` (`lil-mxfp4-csf-checkpoint/1`) engine
- [x] Live 8× NVIDIA B200 deployment on Modal (`im-R1TdH6JxBrlEvExHAIqaph`) with 48 hydrated CSF shards (`495.63 GB`)
- [x] Resolved `PM-001` (shallow-mount imports), `PM-002` (single-pass startup + bounded slice), and `PM-003` (15-key validation schema)
- [x] Optimal Runtime Experience (ORE) inner-loop optimizations (DSpark prefill head-elision, shared MoE/Attention FP8 activation quantization, stack-allocated FP8/FP4/sparse-softmax block buffers, $O(N)$ partial selection, pooled ASGI proxy, isolated V4.1 SSE `StreamRingBuffer` replay, and unified $\mathcal{S}_{\text{ORE}}$ telemetry)
- [x] Working knowledge base (`knowledge-base/`) with ADRs (`ADR-001`..`ADR-007`), Playbooks (`PLAYBOOK-001`..`PLAYBOOK-004`), Post-Mortems (`PM-001`..`PM-005`), and Auto-Rolling Roadmap
- [x] True scale-to-zero container lifecycle (`ADR-006`), deploy-time policy snapshot (`PM-004`), SSE `[DONE]` & token accounting (`PM-005`), distributed rank-health poison isolation & `ExternalStepGuard` (`ADR-007`), and unified `Makefile` / `AGENTS.md` / CI gates (`BASE-06`..`BASE-08`)

### 🔄 Horizon 0 & Horizon 1 — Active Rolling Sprints

- [ ] **H0-01**: Adaptive admission control modulating `TokenBucketRateGovernor` directly from live `itl_jitter_ratio`
- [ ] **H0-02**: Parallel multi-threaded 48-shard header verification (`< 1.5s` cold-volume check)
- [ ] **H0-03**: Automated benchmark regression diffing in `deploy.py benchmark`
- [ ] **H0-04**: Live 8× NVIDIA B200 (`B200:8`) full-model scale-to-zero cold-start, TTFT/TPOT, and lifetime-guard verification (pending cluster capacity & spend approval)
- [ ] **H1-01**: In-kernel GPU CSF scale decompression (`tcgen05.mma` prologue in SMEM)
- [ ] **H1-02**: FlashMLA-4 Split-KV persistent megakernel decode on SM100
- [ ] **H1-03**: Fused MoE router + NVLink 5 (`1.8 TB/s`) all-to-all dispatch across `EP=8`
- [ ] **H1-04**: Multi-process NCCL transport & remote rank heartbeat producer wired into `RankHealthMonitor`

---

## 📋 Changelog Summary

> Full release history and item-by-item diffs are maintained in **[`CHANGELOG.md`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/CHANGELOG.md)**.

| Version | Date | Milestone Summary | Key Highlights |
| :--- | :---: | :--- | :--- |
| **[`v0.3.0`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/CHANGELOG.md)** | `2026-10-31` | **True Scale-to-Zero Lifecycle, Distributed Rank-Health Isolation & Pass 1–6 Hardening** | `min_containers=0` default with `SINGULARITY_ALLOW_PINNED_WARM=1` guard · `policy_env_snapshot()` deploy-time image baking (`PM-004`) · 6-state `LifecycleTracker` & `LifecycleWatchdog` self-exit (`SIGINT` $\rightarrow$ exit `70`) · SSE `[DONE]` disconnect & `tokens_out` accounting (`PM-005`) · `RankHealthMonitor`, wait-bounded DP collectives, `ExternalStepGuard` & `HTTP 503` poison isolation (`ADR-007`) · `PagedBlockManager::pin_block` unallocated rejection & partial-pin rollback · 22 unused Rust deps pruned · `Makefile`, `AGENTS.md` & CI `python-modal` gate |
| **[`v0.2.0`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/CHANGELOG.md)** | `2026-10-31` | **DeepSeek-V4.1-Flash-1017 Lossless CSF, 8× NVIDIA B200 Deployment & Optimal Runtime Experience (ORE)** | Native `lil-mxfp4-csf-checkpoint/1` zero-copy scale reconstruction across 48 shards (`495.63 GB`, `ADR-001`) · Resolved `PM-001` (shallow-mount imports), `PM-002` (single-pass startup + bounded slice) & `PM-003` (15-key validation schema) · ORE inner-loop optimizations (DSpark prefill head-elision, shared MoE/Attention FP8 quantization, stack block buffers, $O(N)$ partial selection, pooled ASGI proxy, isolated SSE `Last-Event-ID` replay) · `knowledge-base/` treatise, ADRs, playbooks & auto-rolling roadmap |
| **[`v0.1.0`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/CHANGELOG.md)** | `2026-10-30` | **Initial 9-Crate Rust Inference Engine Release** | `singularity-core`, `singularity-cuda` (SM100/SM90 + `MockGpuBackend`), `singularity-cache` (`PagedBlockManager` + `RadixTree`), `singularity-scheduler` (`ZosScheduler` + LCFS + DP consensus), `singularity-model` (Safetensors mmap + FlashMLA + MoE), `singularity-grammar` (KMP + JSON Schema bitmask), `singularity-server` (Axum OpenAI HTTP/SSE + `CircuitBreaker` + `LeakyTokenBucket`), `singularity-cli` & `singularity-tests` |

---

## 🤝 Open Source — Why & How

### Why This Exists in the Commons

HEAV3NS was founded under a governing doctrine that places the open commons above any single institution's commercial interest.

**Amendment I of the HEAV3NS Institutional Constitution** (October 9, 2026):

> *"Open source is not merely a licensing preference. It is a means of distributing computational agency beyond centralized authority.*
> *OPEN SOURCE MUST WIN."*

HEAV3NS-BRAIN is the direct expression of that amendment in code.

The inference systems that run frontier AI models are overwhelmingly closed. When a researcher wants to understand how LCFS preemption actually works in a production scheduler, they cannot read the code — they have to guess from papers that may not describe the real implementation. When a team wants to deploy a frontier model on hardware they control, they often cannot do so without depending on a proprietary engine they cannot inspect or modify.

This repository exists because **the knowledge of how inference actually works at production scale belongs to everyone, not a few institutions.**

We do not frame open source as a funnel, a lead mechanism, or an acquisition tactic. We publish the machinery because the machinery is part of the work. Replication is a desired output. If someone forks SINGULARITY and builds something better, the commons advances. That is success.

> *"If another practitioner surpasses our implementation, the commons advances. HEAV3NS does not require technological supremacy to establish cultural significance."*  
> — HEAV3NS Amendment I

### What We Welcome

| Type | Examples |
|---|---|
| **Architecture discussions** | Better scheduling policies, alternative attention mechanisms, new memory layouts |
| **CUDA kernel contributions** | Optimised SM100/SM90 kernels, FP4 quantization paths |
| **Model support** | Adding new Safetensors model families and tokenizer configurations |
| **Bug reports** | Incorrect test results, memory safety issues, scheduling incorrectness |
| **Benchmarks** | Reproducible measurements on real hardware — especially H200/B200 |
| **Documentation** | Clarity improvements, worked examples, architecture diagrams |

### How to Contribute

```bash
# Fork, clone, branch
git clone https://github.com/<your-username>/HEAV3NS-BRAIN.git
git checkout -b feat/your-contribution

# Make changes, enforce the standards
cargo test --all
cargo clippy --all -- -D warnings
cargo fmt --all

# Open a PR — see the template for the checklist
```

Full guide: [CONTRIBUTING.md](CONTRIBUTING.md)  
Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)  
Security reports: [SECURITY.md](SECURITY.md) — report privately, not via issues

**Community:**  
[GitHub Discussions](https://github.com/CardSorting/HEAV3NS-BRAIN/discussions) · [Issue Tracker](https://github.com/CardSorting/HEAV3NS-BRAIN/issues) · [PR Template](.github/PULL_REQUEST_TEMPLATE.md)

---

## 📊 For Stakeholders

*A plain-language summary for decision-makers, partners, and supporters evaluating this work.*

### The Moment

We are at the beginning of an era in which AI inference infrastructure — the machinery that executes the world's most capable models — will become as foundational as operating systems, databases, and networking stacks. The organisations that control that infrastructure will have disproportionate influence over how, where, and for whom AI capabilities are available.

HEAV3NS-BRAIN is a bet that this infrastructure should be open, auditable, and community-governed — not a proprietary moat owned by a small number of cloud providers.

### What Has Been Built

| Dimension | Detail |
|---|---|
| **Language** | Pure Rust — the same language chosen by AWS, Microsoft, Meta, and the Linux kernel for safety-critical infrastructure |
| **Method** | Every design decision derived from source-code audit of production systems, not documentation |
| **Validation** | 61 automated tests covering the complete request lifecycle, including 18 explicit failure-mode stress tests |
| **API** | OpenAI-compatible — existing applications connect with minimal changes |
| **Governance** | MIT license · HEAV3NS Institutional Constitution · 30-year operating horizon |
| **Community** | Open contributions · public issue tracker · documented PR process |

### Stage of Development

**This is a functional research prototype.** All core subsystems are implemented and tested. Real transformer inference runs on CPU mock (any machine) and on NVIDIA GPUs (`--features cuda`).

| Completed | Remaining |
|---|---|
| All 9 core subsystems implemented and tested | Physical B200/H200 hardware validation at production scale |
| Correctness verified on CPU mock and small real models | Operational packaging (Docker, Kubernetes, monitoring) |
| OpenAI-compatible HTTP serving | DeepSeek V4.1 Flash end-to-end weight integration |
| 61 tests including real transformer inference | Multi-GPU production load testing |
| MIT licensed, open governance | Multi-node deployment tooling |

### Why Rust — For Decision Makers

Most AI inference infrastructure is Python-orchestrated with C++ and CUDA hot paths. This creates a language seam — a boundary where bugs accumulate, latency is unpredictable, and the full stack is difficult to audit or modify safely. Python's garbage collector introduces pauses. The C++ extensions are hard to test in isolation.

Rust eliminates the seam:
- **Native performance** comparable to C++, without the memory safety risks
- **Memory safety enforced at compile time** — no garbage collector pauses in the inference hot path
- **Concurrency correctness as a compiler property** — data races are compile errors, not runtime surprises
- **Single auditable language** from HTTP handling to GPU memory allocation

The tradeoff — steeper learning curve, smaller ecosystem — is correct for an inference runtime, where predictability and correctness matter above all.

### Competitive Landscape

| Engine | Language | Open Source | Target GPU | Notes |
|---|---|:---:|---|---|
| vLLM | Python + C++ | ✅ | Multi-GPU | PagedAttention; Python orchestration |
| TGI | Rust + Python | ✅ | Multi-GPU | Flash Attention; Rust server layer |
| SGLang | Python + C++ | ✅ | Multi-GPU | RadixAttention; structured generation |
| DeepInfer | C++ + Python | ⚠️ Partial | B200 / H200 | ZOS + LCFS; prefix cache omitted |
| **HEAV3NS-BRAIN** | **Pure Rust** | **✅ MIT** | **B200 / H200** | **Source-verified ZOS + LCFS + FlashMLA; full open** |

SINGULARITY is the only fully-Rust, fully-open implementation targeting B200/H200 with source-verified scheduling and attention algorithms derived from direct audit of production systems.

### The Thirty-Year Bet

HEAV3NS is not a startup on a 5-year exit trajectory. It is a cultural institution on a 30-year operating horizon. The computational infrastructure built in Horizon 1 (now) supports the Autonomous Artist Residency program in Horizon 2, and ultimately a Sovereign Autonomous Art Institution in Horizon 3 — an entity capable of self-orchestrating its cultural continuity across generations.

SINGULARITY is not an endpoint. It is the beginning of a compounding commons.

---

## ⚖️ Disclaimer

> **Independence.** HEAV3NS-BRAIN is an independent research project of the HEAV3NS Experimental Media Lab. It is not affiliated with, endorsed by, sponsored by, or derived from the proprietary source code of DeepSeek, NVIDIA, HuggingFace, or any organisation referenced in the research documentation.

> **Research Method.** DeepInfer, NOVAV2, and MOSS-Studio were studied at source-code level solely to understand published algorithmic design patterns. No proprietary code was copied or incorporated. All SINGULARITY source code was written independently.

> **Prototype Status.** This is a research prototype. It has not been hardened for production use, has not been audited for security vulnerabilities, and has not been validated at scale on real B200 or H200 hardware. It should not be used to serve untrusted user inputs without independent security review. Provided "as-is" under the MIT License — see [LICENSE](LICENSE).

> **No Model Weights.** This repository does not include or distribute trained model weights. Running real inference requires separately obtained weight files (e.g., Safetensors from HuggingFace Hub). Users are solely responsible for complying with applicable model license terms.

> **Hardware Claims.** GPU specifications (memory capacity, bandwidth, TFLOPS) are sourced from published NVIDIA documentation. They do not represent measurements on hardware owned or operated by the authors.

> **Performance Figures.** All performance targets are architectural projections based on hardware specifications and algorithmic analysis. Actual results depend on model size, batch configuration, driver version, system memory, and thermal conditions.

---

## 📄 License

HEAV3NS-BRAIN is released under the **MIT License**.  
Copyright © 2026 CardSorting / HEAV3NS Experimental Media Lab.

Free to use, copy, modify, distribute, sublicense, and sell. The only requirement: preserve the license notice. See [LICENSE](LICENSE).

---

<div align="center">

<br/>

```
BILLIONS. MUST. COMPUTE.

WE ARE ALL BENEATH THE HEAV3NS. KEEP BUILDING.

OPEN SOURCE MUST WIN.
```

<br/>

*HEAV3NS Experimental Media Lab · Est. 2026 · Salt Lake City*  
*Thirty-year horizon · Amendment I · The commons carries the instruments.*

<br/>

**[Discussions](https://github.com/CardSorting/HEAV3NS-BRAIN/discussions)** · **[Issues](https://github.com/CardSorting/HEAV3NS-BRAIN/issues)** · **[Contributing](CONTRIBUTING.md)** · **[Security](SECURITY.md)** · **[Changelog](CHANGELOG.md)** · **[HEAV3NS Lab](https://github.com/CardSorting/HEAV3NS-STUDIO)**

<br/>

</div>
