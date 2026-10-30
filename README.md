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
| ⚙️ | [Hardware Support](#️-hardware-support) |
| 🚀 | [Quick Start](#-quick-start) |
| ✅ | [Test Coverage](#-test-coverage) |
| 🗺️ | [Roadmap](#️-roadmap) |
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

## ✅ Test Coverage

61 tests. Zero warnings. Zero failures.

| Suite | Crate | Tests | Verified |
|---|---|:---:|---|
| GPU backend | `singularity-cuda` | 5 | Allocation · OOM rejection · hardware fallback · DMA transfers · exact `gemm_f32` |
| KV cache | `singularity-cache` | 4 | Bitset allocation · double-free protection · radix matching · pinned node protection |
| Core | `singularity-core` | 4 | Sequence lifecycle · TTFT · stop-token detection · metrics · config serde |
| Grammar | `singularity-grammar` | 3 | KMP build · tool-call trigger/stop · JSON bitmask masking |
| Model | `singularity-model` | 11 | RMSNorm · RoPE · GQA · SwiGLU · MoE · MLA compressed KV · multi-shard Safetensors · tokenizer |
| Scheduler | `singularity-scheduler` | 4 | LCFS preemption · DP consensus · two-phase fallback · forward pass integration |
| Server | `singularity-server` | 5 | Circuit breaker · rate governor · admission · ring buffer isolation · fsync store |
| E2E server | `tests/` | 1 | Full HTTP bring-up · all endpoints · streaming + non-streaming |
| Failure modes | `tests/` | 18 | 18 explicit edge cases: exhaustion · deadlock · backpressure · stream interruption |
| Real inference | `tests/` | 6 | SmolLM2-135M forward + decode · DeepSeek-V2-Lite MLA + MoE · multi-shard · paged KV DMA |

---

## 🗺️ Roadmap

### ✅ Complete — v0.1.0 (Phase I–III)

The foundation is complete. Everything below is implemented, tested, and passing.

- [x] 9-crate Rust workspace — clean build, zero warnings
- [x] Verified LCFS preemption (source-audited against DeepInfer `scheduler.hpp#L622–646`)
- [x] Pure-Rust SHA-256 Radix tree prefix cache with LRU eviction
- [x] Two-stage FlashMLA — MQAIndexer (Stage 1) + FlashMLA (Stage 2)
- [x] DeepSeekMoE with `sqrtsoftplus` routing and dynamic EPLB
- [x] ZoS CPU-GPU concurrent pipeline with `DeferredRelease`
- [x] Two-tier DP consensus — ALL/ANY voting + two-phase starvation fallback
- [x] OpenAI-compatible Axum 0.7 REST/SSE server
- [x] Circuit breaker · rate governor · ring-buffer replay · `TransactionalStore`
- [x] `CudaBackend` via cudarc 0.12 + cuBLAS (feature: `cuda`)
- [x] `MockGpuBackend` with real host GEMM (no GPU required)
- [x] SmolLM2-135M and DeepSeek-V2-Lite real forward-pass test coverage
- [x] 61 tests · 0 warnings · MIT licensed · GitHub CI pipeline

### 🔄 In Progress

- [ ] CUDA Graph capture for zero-overhead decode-phase kernel replay (SM100/SM90)
- [ ] FP4 custom CUDA kernel integration (`singularity-cuda/src/kernels/v41_quant.cu`)
- [ ] DeepSeek V4.1 Flash full end-to-end inference validation

### 📋 Planned — Horizon 1 (2026–2027)

- [ ] Multi-node NVLink EPLB weight migration for multi-server deployments
- [ ] Continuous batching at production scale
- [ ] Prometheus metrics + Grafana dashboard
- [ ] Docker container + Kubernetes Helm chart
- [ ] Model weight validation tooling (checksum, format verification)
- [ ] Tensor parallelism (TP) within a node
- [ ] NCCL integration for multi-node collective operations

### 🌟 Horizon 2 — Scaled Propagation (2027–2028)

- [ ] Community plugin interface for custom attention kernels
- [ ] Hardware-in-the-loop CI on physical B200/H200 hardware
- [ ] Quantization-aware weight conversion tooling (BF16 → FP8 → FP4)
- [ ] Inference-time compute scaling (speculative decoding, best-of-N)
- [ ] Integration with the HEAV3NS Autonomous Artist Residency platform

### ∞ Horizon 3 — Thirty-Year Commons (2028–2056)

- [ ] A provenance-preserving universal registry of open inference artifacts
- [ ] Decentralised compute distribution: inference beyond data-centre boundaries
- [ ] Deep-time archival of computational art artifacts in the HEAV3NS open commons
- [ ] Billions computing — on their terms, with their models

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
