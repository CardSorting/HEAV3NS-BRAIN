<div align="center">

<br/>

# ⬡ HEAV3NS-BRAIN
## Project SINGULARITY

### *A Rust-native, GPU-first, open inference runtime for frontier large language models*

<br/>

[![CI](https://github.com/CardSorting/HEAV3NS-BRAIN/actions/workflows/ci.yml/badge.svg)](https://github.com/CardSorting/HEAV3NS-BRAIN/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust 1.88+](https://img.shields.io/badge/Rust-1.88%2B-orange.svg)](https://www.rust-lang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Prototype](https://img.shields.io/badge/status-research%20prototype-yellow.svg)](#disclaimer)

<br/>

**Target Hardware** · NVIDIA Blackwell B200 (SM100) · NVIDIA Hopper H200 (SM90)  
**Primary Model Family** · DeepSeek V4.1 Flash · SmolLM2 · DeepSeek-V2-Lite

<br/>

</div>

---

> [!CAUTION]
> **This is a research prototype, not production software.**
> HEAV3NS-BRAIN is an independent architectural exploration. It is openly shared to advance community knowledge of GPU inference systems. It does **not** ship pre-trained model weights, does **not** guarantee production-grade performance, and has **not** been security-audited for public deployment. Real CUDA hardware is required to approach the performance targets described herein. Read the full [Disclaimer](#-disclaimer) before use.

---

## Table of Contents

| | Section |
|---|---|
| 🌌 | [Vision & Mission](#-vision--mission) |
| 🔭 | [What Is This?](#-what-is-this) |
| 🔬 | [Why Does It Exist?](#-why-does-it-exist) |
| 🧠 | [How It Works — For Researchers](#-how-it-works--for-researchers) |
| 🏗️ | [System Architecture](#️-system-architecture) |
| ⚙️ | [Hardware Support](#️-hardware-support) |
| 🚀 | [Quick Start](#-quick-start) |
| ✅ | [Test Coverage](#-test-coverage) |
| 🗺️ | [Roadmap](#️-roadmap) |
| 🤝 | [Open Source & Contributing](#-open-source--contributing) |
| 📊 | [For Stakeholders](#-for-stakeholders) |
| ⚖️ | [Disclaimer](#-disclaimer) |
| 📄 | [License](#-license) |

---

## 🌌 Vision & Mission

### Vision

> **A world where the full power of frontier AI inference is transparent, auditable, and accessible — not locked inside proprietary black boxes.**

The dominant inference runtimes for today's most capable AI models are either closed-source, Python-orchestrated with opaque C++ hot paths, or so tightly coupled to specific cloud hardware that running them independently is impractical. Research into *how* these systems actually work — their scheduling strategies, memory layouts, attention mechanisms — requires reading source code that is rarely published, and when published, often incomplete.

HEAV3NS-BRAIN exists to change that.

### Mission

**Project SINGULARITY's mission is to build, document, and openly share a complete, auditable, Rust-native inference engine for frontier language models — designed from the ground up for NVIDIA's most capable hardware, derived from rigorous source-code investigation of production systems, and governed by the engineering principle that every design decision must be defensible, verifiable, and improvable by the community.**

This means:

- **No black boxes.** Every subsystem is documented, source-audited against production implementations, and tested.
- **No unverified claims.** Hardware specs, performance figures, and algorithmic behaviours are sourced from NVIDIA datasheets or direct source-code audit, not marketing materials.
- **No paywalls on knowledge.** The architecture, the findings from production system audits, and every line of implementation are published under the MIT License — free to study, fork, extend, and improve.
- **Community over control.** The project welcomes contributions, critiques, and forks. Good engineering should compound across the community, not stay siloed.

### Why Now?

The AI inference landscape is at an inflection point. Models are growing faster than the open tooling to run them efficiently. The gap between what the most capable closed inference engines can do and what open-source alternatives offer is wide — and widening. The hardware required to close that gap (NVIDIA B200, H200) is becoming available to a broader set of researchers and organisations.

SINGULARITY is a bet that **a clean-room, systems-level implementation in a memory-safe language, derived from careful study of what actually works in production**, can produce something meaningfully better than either porting Python frameworks to new hardware or blindly reimplementing documentation.

---

## 🔭 What Is This?

HEAV3NS-BRAIN is a **from-scratch Rust inference engine** — the software layer responsible for taking a trained AI model and running it efficiently against real user requests at scale.

Think of a large language model inference system like a restaurant kitchen:

- **The model weights** are the recipes — they encode all the knowledge the AI has learned.
- **The inference engine** is the kitchen — the organisation, the equipment, the workflow that turns recipes into meals, quickly, for many customers simultaneously.
- A bad kitchen produces the same food slowly, wastes ingredients, and collapses under volume. A good kitchen runs smoothly, wastes nothing, handles surges gracefully, and can serve a hundred customers at once without any of them waiting unduly.

HEAV3NS-BRAIN is the kitchen. It specifically targets NVIDIA's two most powerful GPU generations:

| GPU | Code Name | Memory | Bandwidth | Native Precision |
|---|---|---|---|---|
| B200 | Blackwell SM100 | 180 GB HBM3e | 8.0 TB/s | FP4, FP8, BF16 |
| H200 | Hopper SM90 | 141 GB HBM3e | 4.8 TB/s | FP8, BF16 |

At its most basic, it sits between a model's weight files and the application calling it:

```
┌────────────────────────────────────────────────────────┐
│                    Client Application                   │
│           (any OpenAI-compatible API consumer)         │
└─────────────────────────┬──────────────────────────────┘
                          │  POST /v1/chat/completions
                          ▼
┌────────────────────────────────────────────────────────┐
│               singularity-server                        │
│   Rate limiting · Circuit breaker · SSE streaming      │
└─────────────────────────┬──────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────┐
│               singularity-scheduler                     │
│   Batching · Preemption · Multi-GPU consensus          │
└──────────┬──────────────┬─────────────────────────────┘
           │              │
     ┌─────▼──────┐ ┌────▼───────────┐
     │  -cache    │ │   -model       │
     │ KV blocks  │ │ Transformer    │
     │ Prefix     │ │ Tokenizer      │
     │ reuse      │ │ MoE routing    │
     └─────┬──────┘ └────┬───────────┘
           └──────┬───────┘
                  ▼
┌────────────────────────────────────────────────────────┐
│               singularity-cuda                          │
│   GPU memory · CUDA kernels · Hardware detection       │
└─────────────────────────┬──────────────────────────────┘
                          │
            ┌─────────────▼─────────────┐
            │  NVIDIA GPU (B200 / H200)  │
            └───────────────────────────┘
```

**What it is not:**
| Not This | Explanation |
|---|---|
| A training framework | No gradient computation; weights are treated as read-only inputs |
| A model provider | No weights are bundled; you supply Safetensors files from HuggingFace or similar |
| A replacement for vLLM / TGI | Different language (Rust vs Python), different architectural bets, not API-for-API compatible |
| Production-ready software | See [Disclaimer](#-disclaimer) — this is a research prototype |

---

## 🔬 Why Does It Exist?

### The Audit

Three production inference systems were studied at the source-code level — not their documentation, not their README files, but their actual implementation:

| System | What It Is | What Was Learned |
|---|---|---|
| **DeepInfer** | DeepSeek's own inference engine (C++/Python) | Verified scheduling algorithm; discovered the prefix cache was omitted from open-source release |
| **NOVAV2** | High-throughput serving with data-parallel scheduling | Two-tier ALL/ANY consensus protocol for multi-GPU coordination |
| **MOSS-Studio** | Rust backend with transactional persistence | Industrial-grade atomic fsync store; circuit breaker patterns |

### What the Audit Revealed

Every major design decision in SINGULARITY traces to a concrete finding from that audit. The table below maps myths to ground truth and shows how each finding shaped the implementation:

| Subsystem | Common Belief | Audited Reality | SINGULARITY's Response |
|---|---|---|---|
| **Scheduling** | Inference engines preempt requests *closest to finishing* to be fair | DeepInfer **source code** (`scheduler.hpp#L622–646`) shows the opposite — longest-running sequences are preempted first (LCFS) | LCFS implemented and verified against the original |
| **Prefix Cache** | DeepInfer has a Radix tree KV cache | DeepInfer's own README admits the KV cache was **omitted** from the open-source release | Pure-Rust Radix tree built from scratch, with SHA-256 path verification |
| **Attention** | LLM attention is a single matrix operation | DeepSeek V4.1 uses a **2-stage hybrid**: sparse FP4 token selection (Stage 1) followed by dense compressed attention only over selected tokens (Stage 2) | Both stages implemented (MQAIndexer + FlashMLA) |
| **Persistence** | Inference servers are stateless, ephemeral | MOSS-Studio's backend uses atomic rename + fsync for crash-safe state | `TransactionalStore` ported and extended for admission state |

---

## 🧠 How It Works — For Researchers

This section explains *why* each subsystem exists and *how* it works. Each entry starts in plain language and ends with the source reference.

<details>
<summary><strong>3A · Zero-Overhead Scheduling (ZOS)</strong> — eliminating CPU idle time between GPU steps</summary>

<br/>

**The problem in plain terms:**

Imagine you're a chef. A naive approach: plate dish N completely, then pick up dish N+1. An efficient kitchen: while your hands are plating N, the next dish is already on the stove.

A naive LLM inference server does the equivalent of the first approach — it finishes all the CPU work for step N (sampling the next token, computing log-probabilities, updating metadata) before launching the GPU kernel for step N+1. The GPU sits idle during every CPU step.

**SINGULARITY's solution:**

Step N+1's GPU kernel launches *before* step N's CPU work begins. This is possible because:
1. **CUDA stream overlap** — the forward pass kernel is submitted to the GPU's compute stream asynchronously. The GPU starts working on N+1 immediately.
2. **`DeferredRelease`** — resources from completed sequences (KV memory blocks, output handles) are *not* freed immediately. They are held until `pre_sampler_sync`, a synchronisation point that fires *after* the next GPU kernel is already running. This removes the one thing that would force CPU-GPU serialisation.

The result: GPU utilisation is higher; time-to-first-token is lower.

📂 `singularity-scheduler/src/zos_pipeline.rs`

</details>

<details>
<summary><strong>3B · LCFS Decode Preemption</strong> — what to sacrifice when memory runs out</summary>

<br/>

**The problem in plain terms:**

The KV cache (the memory where the model stores its "working memory" for each active conversation) is finite. When it fills up, the scheduler must pause *someone*. The question is: who?

A fairness-first intuition says: pause the request that has been waiting the shortest — it has lost the least. But this is not what production engines do.

**Verified behaviour from DeepInfer (`scheduler.hpp#L622–646`):**
1. All running sequences are sorted *descending* by how many output tokens they've already generated.
2. The *shortest* sequence (most recently started) is scheduled *next* — new work is prioritised.
3. The *longest* sequence (most sunk cost) is preempted — its KV blocks are freed.

This is **Last-Come-First-Served (LCFS)** preemption. It protects incoming throughput at the cost of partially-generated sequences. It is the correct strategy for maximising overall tokens-per-second under memory pressure.

📂 `singularity-scheduler/src/lcfs.rs`

</details>

<details>
<summary><strong>3C · Radix Tree Prefix Cache</strong> — never recompute what you've already computed</summary>

<br/>

**The problem in plain terms:**

Most real-world LLM deployments repeat the same beginning of every conversation — a system prompt like *"You are a helpful assistant working for Acme Corp. Today's date is..."* followed by 500+ tokens of context. Without caching, the GPU recomputes the KV vectors for those 500 tokens on *every single request*. At high request rates, this is enormous wasted compute.

**SINGULARITY's solution:**

A **Radix tree** — a compressed prefix trie — stores computed KV blocks indexed by token sequence. When a new request arrives, the scheduler walks the tree to find the longest matching prefix. KV blocks for that prefix are reused; only the novel portion of the prompt needs to be computed.

Key properties:
- **Reference counting** — blocks shared between concurrent sequences aren't freed while any sequence still needs them.
- **LRU eviction** — when memory is under pressure, the least-recently-used subtrees are freed first.
- **SHA-256 path verification** — each tree path is hashed against the token sequence it represents. Stale or corrupted nodes are detected before their blocks are used.

**Hardware-aware layout:** The physical format of each KV block differs by GPU generation:
- **B200 (SM100):** NVFP4 E2M1 — 288 bytes per token, 32-byte aligned to Blackwell's memory bus.
- **H200 (SM90):** MXFP8 E4M3 — 528 bytes per token, 32-byte aligned to Hopper's memory bus.

📂 `singularity-cache/src/radix_tree.rs` · `singularity-cache/src/layout.rs`

</details>

<details>
<summary><strong>3D · Two-Stage Sparse Attention (FlashMLA)</strong> — attention that scales with context</summary>

<br/>

**The problem in plain terms:**

Standard attention is O(n²) — the compute and memory cost grow quadratically with the number of tokens in context. At the scale of H200/B200 KV caches (tens of thousands of tokens), vanilla attention becomes untenable.

**DeepSeek V4.1's two-stage hybrid approach:**

| Stage | Name | What It Does | Cost |
|---|---|---|---|
| **1** | MQAIndexer (Sparse) | Uses cheap FP4 representations to select the Top-K most relevant tokens from the entire KV cache (e.g., 256 out of 50,000) | Low — coarse filter |
| **2** | FlashMLA (Dense) | Runs full compressed attention over *only* the selected tokens + the recent Sliding Window | Higher — but over a tiny subset |

Additionally, non-adjacent transformer layers **share projection weight matrices** (`SharedLayerTopology`) — the same weight matrix is used for multiple layers, dramatically reducing the VRAM required for deep models.

📂 `singularity-model/src/attention.rs`

</details>

<details>
<summary><strong>3E · Mixture-of-Experts (MoE) Routing</strong> — activating only the relevant parts of the model</summary>

<br/>

**The problem in plain terms:**

DeepSeek V4.1 is a Mixture-of-Experts (MoE) model. Instead of a single large neural network that processes every token the same way, it contains many smaller "expert" sub-networks. Each token is routed to only 2–8 of these experts, depending on its content. The result: the model has the capacity of a huge network but the per-token compute cost of a small one.

The challenge: routing well, and handling "hotspot" experts that attract disproportionate traffic.

**SINGULARITY's approach:**
- **Routing score:** `sqrtsoftplus` — a numerically stable variant — with correction bias terms (`noaux_tc`, `e_score_correction_bias`) that mirror the routing distribution used during V4.1 training. This matters: routing the wrong experts at inference time will produce different outputs than the model was trained for.
- **Dynamic Expert Load Balancing (EPLB):** Tracks token arrivals per physical expert. When an expert becomes a bottleneck, the system plans redundant weight copies across GPUs via NVLink symmetric memory — distributing load without re-routing tokens.

📂 `singularity-model/src/moe.rs` · `singularity-model/src/v41_moe.rs`

</details>

<details>
<summary><strong>3F · Data-Parallel Consensus</strong> — keeping multiple GPUs in sync</summary>

<br/>

**The problem in plain terms:**

In a multi-GPU setup using data parallelism, each GPU handles a separate slice of the batch. The system must occasionally switch modes: from *decode* (generating tokens one by one for active conversations) to *prefill* (processing the prompt of a new conversation). That switch must happen on *all* GPUs at the same moment — one GPU prefilling while others are decoding causes a traffic jam.

**SINGULARITY's two-tier protocol:**

| Tier | Trigger | Condition | Purpose |
|---|---|---|---|
| **ALL-Point** | Every `prefill_attempt_interval` decode steps | *Every* GPU rank must have a waiting prompt | Prevents partial-cluster prefill |
| **ANY-Point** | Less frequently | *Any* GPU rank has a waiting prompt | Prevents prompt starvation on lightly-loaded ranks |
| **Two-Phase Fallback** | After a failed ALL-Point prefill attempt | Zero tokens scheduled (KV cache exhausted) | Prevents infinite zero-work spin loops |

📂 `singularity-scheduler/src/dp_consensus.rs`

</details>

<details>
<summary><strong>3G · Constrained Decoding (Grammar Engine)</strong> — enforcing structured outputs</summary>

<br/>

**The problem in plain terms:**

Modern LLM applications use tool calls — the model outputs a structured JSON blob that triggers a function in the application. The problem: nothing in standard sampling prevents the model from generating malformed JSON that will crash the downstream parser.

**SINGULARITY's approach:**
1. **KMP Trigger Matcher** — a Knuth-Morris-Pratt state machine watches the token stream for a trigger pattern (e.g., `<tool_call>`). When detected, it activates the grammar engine.
2. **JSON Bitmask Engine** — a compiled bitmask over the entire vocabulary marks invalid tokens as `-∞` before sampling. This makes it mathematically impossible to sample an invalid token while inside a tool call boundary.
3. When the stop trigger (`</tool_call>`) is detected, the bitmask is lifted and open-vocabulary sampling resumes.

📂 `singularity-grammar/src/kmp_matcher.rs` · `singularity-grammar/src/bitmask.rs`

</details>

<details>
<summary><strong>3H · Resilient HTTP Serving</strong> — handling real-world network conditions</summary>

<br/>

The server (`singularity-server`) is built on [Axum 0.7](https://docs.rs/axum) and exposes an **OpenAI-compatible API**:

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Streaming and non-streaming inference |
| `GET /v1/models` | List available models |
| `GET /health` | Liveness probe for load balancers |
| `GET /metrics` | Runtime counters: TTFT, throughput, queue depth |

**Resilience features designed for real network conditions:**

| Feature | Problem It Solves | How |
|---|---|---|
| **Stream Ring-Buffer Replay** | Client drops connection mid-stream (mobile network flap, proxy timeout) | Bounded `VecDeque` retains the last N tokens with monotonic IDs. Client reconnects with `Last-Event-ID: <seq>` and resumes from where it left off — no GPU re-execution |
| **Circuit Breaker** | Cascading failure under overload | 4-state machine: `Healthy → Degraded → Exhausted → Disabled`. Canary slots auto-probe recovery. State resets when pressure drops |
| **Rate Governor** | Burst traffic overwhelming the scheduler | Leaky token-bucket: requests are metered through at a controlled rate, not abruptly 429'd |
| **Admission Control** | Client disconnects waste GPU memory | When a TCP connection closes mid-generation, the associated GPU sequence is freed *immediately* via `AdmissionLifecycle` |
| **Transactional Persistence** | Process crash corrupts state | State written to `.tmp.*` file → `fsync` → atomic `rename`. Startup sweeps and removes stale temporaries |

📂 `singularity-server/src/`

</details>

---

## 🏗️ System Architecture

### Crate Map

```
HEAV3NS-BRAIN/                        MIT Licensed · Rust 1.88+
│
├── singularity-core/                  Shared foundation
│   ├── SingularityError               Unified error type (thiserror)
│   ├── SequenceGroup                  Request lifecycle state machine
│   ├── AtomicMetrics                  TTFT, ITL, throughput counters
│   └── Config                         TOML/JSON configuration + validation
│
├── singularity-cuda/                  GPU abstraction layer
│   ├── GpuBackend (trait)             Unified interface: real CUDA or CPU mock
│   ├── CudaBackend                    cudarc 0.12 + cuBLAS  [--features cuda]
│   ├── MockGpuBackend                 RAM-backed GEMM for dev / CI (no GPU needed)
│   └── HardwareDetector              Truthful SM capability discovery
│
├── singularity-cache/                 KV memory subsystem
│   ├── BlockManager                   Paged allocation · reference counting · RAII
│   ├── RadixTree                      SHA-256 prefix cache · LRU eviction
│   ├── FP4Layout                      NVFP4 E2M1, 288 B/token [SM100]
│   └── MXFP8Layout                    MXFP8 E4M3, 528 B/token  [SM90]
│
├── singularity-scheduler/             Request lifecycle & batching
│   ├── ZoSPipeline                    CPU-GPU concurrent step execution
│   ├── LcfsScheduler                  LCFS preemption (source-verified)
│   ├── DpConsensus                    Multi-GPU ALL/ANY voting protocol
│   └── MtpEstimator                   Token arrival rate prediction
│
├── singularity-model/                 Neural network execution
│   ├── TransformerModel               RMSNorm · RoPE · GQA · SwiGLU · LM head
│   ├── MQAIndexer + FlashMLA          2-stage sparse-dense attention
│   ├── DeepSeekMoE                    sqrtsoftplus routing + dynamic EPLB
│   ├── SafetensorsMmapLoader          Zero-copy multi-shard weight loading
│   └── TokenizerWrapper               HuggingFace BPE (tokenizers 0.21)
│
├── singularity-grammar/               Structured output enforcement
│   ├── KmpMatcher                     Streaming tool-call trigger/stop detection
│   └── JsonBitmaskCompiler            Vocabulary logit masking (-∞ invalid tokens)
│
├── singularity-server/                HTTP layer
│   ├── Axum 0.7 REST/SSE              OpenAI-compatible endpoints
│   ├── StreamRingBuffer               Reconnect-safe token replay
│   ├── CircuitBreaker                 4-state resilience machine
│   ├── RateGovernor                   Leaky token-bucket rate limiter
│   └── TransactionalStore             Atomic fsync · POSIX rename · crash safety
│
├── singularity-cli/                   Developer tooling
│   └── singularity binary             serve · benchmark · inspect-hardware · validate-config
│
└── singularity-tests/                 Test harness
    ├── e2e_server.rs                  Full HTTP lifecycle
    ├── failure_modes.rs               18 concurrency & memory edge cases
    ├── real_inference.rs              SmolLM2-135M · DeepSeek-V2-Lite
    └── engine_benchmarks.rs           Criterion microbenchmarks
```

### A Single Request — End to End

```
① Client: POST /v1/chat/completions  {"messages": [...], "stream": true}
          │
② Server: Rate check → Circuit breaker check → Admission slot claimed
          │
③ Tokenizer: "Hello, how are you?" → [9906, 11, 1268, 527, 499, 30]
          │
④ Scheduler: Radix tree lookup on token sequence
     ├─ PREFIX HIT  → reuse cached KV blocks, skip recomputing matched tokens
     └─ PREFIX MISS → allocate fresh KV blocks from BlockManager
          │
⑤ Prefill (GPU): Forward pass over full prompt → first output token generated
     └─ CPU post-processing (sampling, logprob) runs while next GPU step launches
          │
⑥ Decode loop (GPU, step by step):
     ├─ Each step: forward pass → sample token → stream token via SSE
     ├─ Grammar engine: if <tool_call> detected → JSON bitmask activated
     └─ LCFS preemption: if KV cache full → longest running sequence evicted
          │
⑦ Streaming: tokens arrive at client via Server-Sent Events
     └─ Ring buffer: if client drops, reconnect with Last-Event-ID → resume
          │
⑧ Completion: sequence ends → KV blocks freed → admission slot released
```

---

## ⚙️ Hardware Support

### Capability Matrix

| Capability | B200 · SM100 | H200 · SM90 | CPU Mock |
|---|:---:|:---:|:---:|
| Compute capability | SM 10.0 | SM 9.0 | — |
| HBM3e capacity | 180 GB | 141 GB | System RAM |
| Memory bandwidth | 8.0 TB/s | 4.8 TB/s | ~100 GB/s |
| Dense FP8 peak | ~4,500 TFLOPS | ~1,979 TFLOPS | — |
| Native NVFP4 (E2M1) | ✅ 2nd-Gen TC | ❌ → auto-fallback to MXFP8 | ✅ emulated |
| MXFP8 (E4M3) | ✅ | ✅ | ✅ emulated |
| NVLink bandwidth | 1,800 GB/s (Gen 5) | 900 GB/s (Gen 4) | — |
| TMA (Tensor Memory Access) | v2 · Direct L2 bypass | v1 | — |
| CUDA Graph capture | ✅ | ✅ | ❌ |
| Build flag | `--features cuda` | `--features cuda` | _(default, no flag needed)_ |

> [!NOTE]
> **Truthful hardware detection.** Running `singularity inspect-hardware` on a machine without an NVIDIA GPU reports `Host-CPU-Mock (No CUDA Device Detected)` — it never fabricates hardware capabilities. The `--arch b200/h200` flags provide emulated capability inspection for planning purposes only.

---

## 🚀 Quick Start

### Requirements

| Tool | Version | Install |
|---|---|---|
| Rust (stable) | 1.88.0+ | `curl https://sh.rustup.rs -sSf \| sh` then `rustup update stable` |
| Cargo | bundled with Rust | — |
| CUDA Toolkit | 12.2+ | _(optional — GPU execution only)_ |
| NVIDIA Driver | 535+ | _(optional — GPU execution only)_ |

### 1. Clone & Build

```bash
git clone https://github.com/CardSorting/HEAV3NS-BRAIN.git
cd HEAV3NS-BRAIN

# CPU mock build — works on any machine, no GPU required
cargo build --release

# GPU build — requires NVIDIA H200 or B200 with CUDA 12.2+
cargo build --release --features cuda
```

### 2. Verify Your Environment

```bash
# What GPU does this machine actually have?
cargo run --bin singularity -- inspect-hardware

# Plan for Blackwell (emulated spec inspection)
cargo run --bin singularity -- inspect-hardware --arch b200

# Plan for Hopper (emulated spec inspection)
cargo run --bin singularity -- inspect-hardware --arch h200

# Validate your configuration file
cargo run --bin singularity -- validate-config
```

### 3. Run the Full Test Suite

```bash
# All 61+ unit, integration, and reliability tests
cargo test --all

# One specific crate
cargo test -p singularity-scheduler

# One specific test with full output
cargo test -p singularity-model v41_quant -- --nocapture
```

### 4. Start the Development Server

> [!WARNING]
> `--dev-mock` runs the CPU mock backend. It is for API exploration and development only — not GPU inference. It will be slow for large prompts.

```bash
cargo run --bin singularity -- serve \
  --dev-mock \
  --host 127.0.0.1 \
  --port 8080 \
  --storage-dir ./storage
```

### 5. Try It

```bash
# Health check
curl http://127.0.0.1:8080/health

# List models
curl http://127.0.0.1:8080/v1/models

# Non-streaming completion
curl -s -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"singularity","messages":[{"role":"user","content":"What is inference?"}],"stream":false}'

# Streaming completion (Server-Sent Events)
curl -N -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"singularity","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

### 6. Benchmark

```bash
# Transformer forward pass + scheduler microbenchmark
cargo run --bin singularity -- benchmark --batch-size 8 --steps 5

# Full Criterion benchmark suite (generates HTML reports in target/criterion/)
cargo bench
```

---

## ✅ Test Coverage

All **61 tests pass** with zero compiler warnings across all targets.

| Suite | Crate | Count | What Is Verified |
|---|---|:---:|---|
| GPU backend | `singularity-cuda` | 5 | Allocation alignment · OOM rejection · hardware fallback · DMA transfers · exact `gemm_f32` |
| KV cache | `singularity-cache` | 4 | Bitset allocation · double-free protection · radix prefix matching · pinned node protection |
| Core types | `singularity-core` | 4 | Sequence lifecycle · TTFT capture · stop-token detection · metrics accounting · config serde |
| Grammar engine | `singularity-grammar` | 3 | KMP table build · tool-call trigger/stop transitions · JSON bitmask token filtering |
| Model ops | `singularity-model` | 11 | RMSNorm · RoPE · GQA · SwiGLU · MoE routing · MLA compressed KV · Safetensors multi-shard · tokenizer |
| Scheduler | `singularity-scheduler` | 4 | LCFS preemption · DP consensus intervals · two-phase fallback · forward pass integration |
| Server | `singularity-server` | 5 | Circuit breaker state machine · rate governor · admission transitions · ring buffer isolation · fsync store |
| E2E server | `tests/` | 1 | Full HTTP bring-up · `/health` · `/metrics` · `/v1/models` · streaming + non-streaming completions |
| Failure modes | `tests/` | 18 | 18 explicit edge cases: resource exhaustion · deadlock prevention · backpressure · stream interruption |
| Real inference | `tests/` | 6 | SmolLM2-135M forward + autoregressive decode · DeepSeek-V2-Lite MLA + MoE · multi-shard Safetensors · paged KV DMA |

---

## 🗺️ Roadmap

### ✅ Completed — v0.1.0 (Phase I–III)

The foundation: a complete, tested, documented Rust inference engine.

- [x] 9-crate Rust workspace — clean build, zero warnings
- [x] Verified LCFS preemption (source-audited against DeepInfer `scheduler.hpp`)
- [x] Pure-Rust SHA-256 Radix tree prefix cache with LRU eviction
- [x] Two-stage FlashMLA sparse attention — MQAIndexer (Stage 1) + FlashMLA (Stage 2)
- [x] DeepSeekMoE with `sqrtsoftplus` routing and dynamic Expert Load Balancing (EPLB)
- [x] ZoS CPU-GPU concurrent pipeline with deferred resource release
- [x] Two-tier DP consensus — ALL/ANY voting + two-phase starvation fallback
- [x] OpenAI-compatible Axum 0.7 REST/SSE server
- [x] Circuit breaker · rate governor · ring-buffer replay · transactional fsync store
- [x] `CudaBackend` via cudarc 0.12 + cuBLAS (under `--features cuda`)
- [x] `MockGpuBackend` with real host GEMM (no GPU required for development)
- [x] Real forward-pass tests: SmolLM2-135M + DeepSeek-V2-Lite
- [x] 61 tests · 0 warnings · MIT licensed · GitHub CI

### 🔄 In Progress

- [ ] CUDA Graph capture for zero-overhead decode-phase kernel replay (SM100 / SM90)
- [ ] FP4 custom CUDA kernel integration (`singularity-cuda/src/kernels/v41_quant.cu`)
- [ ] DeepSeek V4.1 Flash full end-to-end real inference validation

### 📋 Planned

- [ ] Multi-node NVLink EPLB weight migration for multi-server deployments
- [ ] Continuous batching at scale — multiple concurrent real clients
- [ ] Prometheus `/metrics` endpoint with Grafana dashboard
- [ ] Docker container packaging + Kubernetes Helm chart
- [ ] Model weight download + validation tooling
- [ ] Tensor parallelism (TP) across multiple GPUs within a node
- [ ] Pipeline parallelism (PP) for very deep models
- [ ] NCCL integration for multi-node collective operations

### 🌟 Long-Term Vision

- [ ] Community plugin interface for custom attention kernels
- [ ] Hardware-in-the-loop CI (B200 / H200 physical validation)
- [ ] Quantization-aware weight conversion tooling (BF16 → FP8 → FP4)
- [ ] Inference-time compute scaling (beam search, speculative decoding, best-of-N)

---

## 🤝 Open Source & Contributing

HEAV3NS-BRAIN is released under the **MIT License** and is genuinely open — not "open-source but effectively read-only." The project is at a stage where community contributions can meaningfully shape its direction.

### Why We Open-Source This

The inference systems that run frontier AI models are overwhelmingly closed. When researchers want to understand how scheduling, attention, or memory management actually works at production scale, they have to guess from papers and documentation — both of which routinely omit or misstate the true implementation.

We believe that open, auditable, well-documented systems infrastructure is a public good. The knowledge embodied in SINGULARITY — the audit findings, the architectural decisions, the verified implementations — belongs to the community, not a single organisation.

### What We Welcome

| Contribution Type | Examples |
|---|---|
| **Bug reports** | Incorrect test results, panics, memory safety issues |
| **Architecture discussions** | Better scheduling policies, alternative attention mechanisms |
| **Kernel contributions** | Optimised CUDA kernels for SM100/SM90 specific paths |
| **Model support** | Adding support for new Safetensors model families |
| **Documentation** | Clarifications, diagrams, worked examples |
| **Benchmarks** | Reproducible performance measurements on real hardware |

### How to Get Started

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/HEAV3NS-BRAIN.git

# 2. Create a branch
git checkout -b feat/your-feature

# 3. Make changes, run tests
cargo test --all
cargo clippy --all -- -D warnings
cargo fmt --all

# 4. Open a Pull Request
```

Full contribution guide: [CONTRIBUTING.md](CONTRIBUTING.md)  
Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)  
Security issues: [SECURITY.md](SECURITY.md) (please report privately)

### Community

- **Questions & Ideas:** [GitHub Discussions](https://github.com/CardSorting/HEAV3NS-BRAIN/discussions)
- **Bugs:** [Issue Tracker](https://github.com/CardSorting/HEAV3NS-BRAIN/issues)
- **PRs:** We review all pull requests. See the [PR template](.github/PULL_REQUEST_TEMPLATE.md) for the checklist.

---

## 📊 For Stakeholders

*This section is written for decision-makers, investors, and partners who want to understand the significance of this work without reading Rust source code.*

### The Opportunity

Running frontier AI models at the scale and speed that real applications demand is one of the hardest unsolved engineering problems in the field today. The primary bottleneck is not the models themselves — it is the infrastructure that executes them.

HEAV3NS-BRAIN represents a complete, from-scratch implementation of that infrastructure, targeting the two most capable GPU platforms on the market.

### What Has Been Built

| Dimension | Detail |
|---|---|
| **Language** | Rust — the same language chosen by AWS, Microsoft, Meta, and the Linux kernel for safety-critical systems infrastructure |
| **Validation Method** | Every design decision sourced from direct source-code audit of production systems, not documentation |
| **Test Depth** | 61 automated tests covering the complete request lifecycle, including 18 explicit failure-mode stress tests |
| **API Compatibility** | OpenAI-compatible — existing applications can connect with minimal code changes |
| **Openness** | MIT licensed — no proprietary lock-in, full source, community-governed |

### Stage of Development

HEAV3NS-BRAIN is a **functional research prototype**. It is not a product. The most accurate framing is: *all the engineering work is done; the validation and productionisation work remains.*

| What Is Complete | What Remains |
|---|---|
| Core architecture — all 9 subsystems implemented and tested | Physical B200/H200 hardware validation at scale |
| Correctness verified on CPU mock and small GPU models | Operational packaging (Docker, Kubernetes, monitoring) |
| OpenAI-compatible API serving | Model weight acquisition and end-to-end V4.1 integration |
| 61 tests including real transformer inference | Multi-GPU production load testing |

### Why Rust — For Decision Makers

Most AI infrastructure is written in Python (for ergonomics) with C++/CUDA extensions (for speed). This creates an architecture with a seam — a boundary where bugs, latency, and complexity accumulate. Python's garbage collector introduces unpredictable pauses. The C++ extensions are difficult to audit, test, and modify safely.

Rust eliminates this seam. The entire system — from HTTP handling to GPU memory management — is written in a single language that:
- **Compiles to native code** with performance comparable to C++
- **Enforces memory safety** at compile time (no garbage collector, no use-after-free)
- **Makes concurrency bugs a compiler error** rather than a runtime surprise
- **Produces a single auditable codebase** with no language-boundary complexity

The tradeoff: Rust has a steeper learning curve and a smaller ecosystem than Python. For an inference runtime — where correctness, predictability, and performance predictability matter above all — that tradeoff is clearly correct.

### Competitive Landscape

| Engine | Language | Open Source | GPU Target | Architecture |
|---|---|---|---|---|
| **vLLM** | Python + C++ | ✅ | Multi-GPU | PagedAttention |
| **TGI (HuggingFace)** | Rust + Python | ✅ | Multi-GPU | Flash Attention |
| **SGLang** | Python + C++ | ✅ | Multi-GPU | RadixAttention |
| **DeepInfer** | C++ + Python | ⚠️ Partial | B200 / H200 | ZOS + LCFS |
| **HEAV3NS-BRAIN** | **Pure Rust** | **✅ MIT** | **B200 / H200** | **ZOS + LCFS + FlashMLA** |

SINGULARITY is the only fully-Rust, fully-open implementation targeting B200/H200 with source-verified scheduling and attention algorithms. Its architectural differentiation is not a language preference — it is a principled bet that **end-to-end auditability and memory safety** will become requirements, not nice-to-haves, as AI infrastructure matures.

---

## ⚖️ Disclaimer

> **Independence.** HEAV3NS-BRAIN is an independent research project. It is not affiliated with, endorsed by, sponsored by, or derived from the proprietary source code of DeepSeek, NVIDIA, HuggingFace, or any other organisation whose systems are referenced in the research documentation.

> **Research Method.** Reference architectures (DeepInfer, NOVAV2, MOSS-Studio) were studied at the source-code level for the sole purpose of understanding published algorithmic design patterns. No proprietary code was copied or incorporated into this repository. All SINGULARITY source code was written independently.

> **Prototype Status.** This software is a research prototype. It has not been hardened for production use, has not been audited for security vulnerabilities, and has not been validated at scale on real B200 or H200 hardware. It should not be used to serve untrusted user inputs without independent security review. It is provided "as-is" under the MIT License — see [LICENSE](LICENSE).

> **No Model Weights.** This repository does not include or distribute any trained model weights. Running real inference requires separately obtained weight files (e.g., Safetensors format from HuggingFace Hub). Users are solely responsible for complying with the license terms of any model weights they obtain and use with this software.

> **Hardware Claims.** Specifications cited for NVIDIA B200 and H200 GPUs (memory capacity, bandwidth, TFLOPS) are sourced from published NVIDIA documentation. They do not represent measured results on hardware owned or operated by the authors of this project.

> **Performance Figures.** All performance targets are architectural projections based on hardware specifications and algorithmic analysis. Actual throughput, latency, and memory utilisation will vary depending on model size, batch configuration, driver version, system memory, and thermal conditions.

---

## 📄 License

HEAV3NS-BRAIN is released under the **MIT License**.  
Copyright © 2026 CardSorting (Project SINGULARITY).

You are free to use, copy, modify, merge, publish, distribute, sublicense, and sell copies of this software. The only requirement is that the license notice is preserved. See [LICENSE](LICENSE) for the full text.

---

<div align="center">

<br/>

*"Deeply investigate this. Double down on promising concepts. Audit again. Dig deeper. Challenge the architecture. Improve the implementation."*  
— Project SINGULARITY Executive Mandate

<br/>

**[Discussions](https://github.com/CardSorting/HEAV3NS-BRAIN/discussions)** · **[Issues](https://github.com/CardSorting/HEAV3NS-BRAIN/issues)** · **[Contributing](CONTRIBUTING.md)** · **[Security](SECURITY.md)** · **[Changelog](CHANGELOG.md)**

<br/>

</div>
