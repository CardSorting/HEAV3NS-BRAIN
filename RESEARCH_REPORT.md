# COMPREHENSIVE ARCHITECTURAL AUDIT & CRITIQUE REPORT
## Project SINGULARITY: Next-Generation Rust GPU Inference Architecture
**Target Hardware:** NVIDIA B200 (SM100) / H200 (SM90)  
**Destination:** `/Users/bozoegg/Desktop/HEAV3NS-BRAIN`  
**Teacher Repositories Audited:**  
1. `/Users/bozoegg/Downloads/DeepInfer-main`  
2. `/Users/bozoegg/Downloads/NOVAV2-main`  
3. `/Users/bozoegg/Desktop/MOSS-Studio` (including `/Users/bozoegg/Desktop/MOSS-Studio/modal-studio/moss-backend`)  
**Mandate Reference:** [`/Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity.md`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/singularity.md)

---

### Executive Summary

Pursuant to the executive commission defined in `singularity.md`, this investigation performs an independent, skeptical deep-code audit across all three reference teachers, evaluates the target accelerator architectures (NVIDIA B200 SM100 and H200 SM90), directly challenges and corrects the errors in prior assumptions, and specifies the production engineering architecture for **Project SINGULARITY**.

Our independent audit reveals several critical findings:
1. **Inverted LCFS Preemption Finding in DeepInfer:** The prior belief claimed DeepInfer's scheduler preempts requests with the *shortest* output to finish requests closest to completion. Source inspection of `csrc/engine/scheduler.hpp#L622-L646` proves the exact opposite: the scheduler sorts running sequences descending by output length, schedules the shortest output sequences first (`running[r-1]`), and preempts `running[l]` ($l=0$, the *longest* running sequences). It is a textbook Last-Come-First-Served (LCFS) preemption model.
2. **Fabricated Prefix Caching in DeepInfer:** `BlockManager` in `csrc/engine/block_manager.hpp` contains no Radix Tree, no hash table, and no prefix matching. Its reference counting is used strictly for parallel sequence forking (`scheduler.hpp#L1110`) and deferred unpinning in ZOS (`llm_engine.py#L116`). DeepInfer's own `README.md#L38-L43` explicitly admits that a complete KV-cache system was omitted from the open-source release. SINGULARITY must build its own pure-Rust Radix tree.
3. **Discovered Industrial Rust Subsystem in MOSS-Studio:** Teacher 3 contains `MOSS-Studio/modal-studio/moss-backend`, a standalone, production-grade Rust backend (`Cargo.toml`, Axum 0.7, Tokio, Rusqlite 0.32, Parking Lot) containing a 64KB transactional persistence engine (`store.rs`), background worker pools (`worker.rs`), and circuit breakers that directly inform SINGULARITY's persistence and recovery design.
4. **Omission of Two-Stage Attention & Shared Layer Topology:** DeepSeek-V4.1 uses a 2-stage hybrid sparse-compressed architecture: Stage 1 uses an FP4 sparse indexer (`MQAIndexer` in `deepinfer/model_executor/models/v4_modules/mla_attn.py#L100-L140`) and Top-K token selection (`DeepSelect`), followed by Stage 2 dense latent attention (`FlashMLA`). Furthermore, non-adjacent layers share projection weights via `SharedLayerTopology` (`v4_modules/cross_layer.py`).
5. **Mandate Section 6 Implementation:** An end-to-end operational framework for parent-agent task decomposition, non-blocking delegation, and skill invocation authority.

---

### Part 1: Deep Code Audit of Teacher 1 — DeepInfer

DeepInfer (`DeepInfer-main`) is DeepSeek's high-performance inference engine for DeepSeek-V4.1-Flash on SM100 and Ascend 950 NPUs.

#### 1. Zero-Overhead Scheduling (ZOS) Execution Lifecycle
- **Implementation:** `deepinfer/worker/worker.py#L88-L190`, `csrc/engine/zos.hpp#L16-L30`, `csrc/engine/zos.cpp#L142-L196`.
- **Traced Execution Pipeline:**
  ```text
  Step N (GPU Stream):    [Forward Step N] --------> [Sample Step N] --------> [CUDA Event: sample_end_evt]
  Step N (CPU Thread):    [Schedule Step N+1] -----> [Post-Process Step N-1] -> [Update CPU Metadata]
  Step N+1 (GPU Stream):  [Forward Step N+1 (Overlapped with Step N Post-Process)]
  ```
- **On-Device Correction Kernel:** In `worker.py#L191-L240` (`_correct_seqlens_qk`), CPU sequence lengths lag GPU execution by one step. DeepInfer runs `prepare_zos_inputs` in C++ (`csrc/engine/zos.cpp#L142-L196`) to compute a `remap` index between Step N and Step N-1. It then launches GPU TileLang kernels:
  - `compute_seqlens_k_zos` (`deepinfer/model_executor/layers/tilelang_ops/compute_seqlens_k_zos.py`): Corrects KV sequence lengths directly on GPU memory based on `num_accepted_tokens_gpu`.
  - `compute_seqlens_q_by_confidence_rates`: Adjusts query lengths according to MTP confidence scores.
  - Copies `seqlens_q_cpu` back to host asynchronously via pinned memory with non-blocking event synchronization (`d2h_evt.synchronize()` in `worker.py#L149`).
- **Deferred Resource Release:** `deepinfer/engine/llm_engine.py#L67-L120` implements `DeferredRelease`. Finished sequence metadata, output pointers, and block unpins (`collect_clone_unpins`) are held across the forward boundary and only freed during `pre_sampler_sync` (`release()` in `llm_engine.py#L100-L119`), preventing CPU deallocation pauses while GPU kernels are launching.

#### 2. Continuous Batching, Collective DP Voting & Preemption
- **Implementation:** `csrc/engine/scheduler.hpp#L290-L342`, `scheduler.hpp#L622-L646`.
- **Two-Tier DP Collective Consensus:** In `EngineType::NORMAL` mode (lines 295-342), all ranks in a Data-Parallel (DP) group must execute either prefill or decode together to prevent communication deadlocks:
  - *ALL Point* (`decode_step % prefill_attempt_interval == 0`): Triggers prefill only if **every** DP rank has waiting prompts (`_dp_allreduce(want ? 1 : 0, MIN) == 1`).
  - *ANY Point* (`decode_step % prefill_any_attempt_interval == 0`): Triggers prefill if **any** rank has waiting prompts (`_dp_allreduce(want ? 0 : 1, MIN) == 0`), preventing prompt starvation on isolated ranks.
  - *Two-Phase Validation:* A secondary all-reduce validates that at least one rank actually scheduled tokens. If all ranks failed (e.g., KV exhaustion), the entire DP group switches to decode on line 334 to avoid infinite zero-work loops.
- **LCFS Preemption Mechanics (Verified):** In `scheduler.hpp#L624-L643`:
  ```cpp
  std::sort(running.begin(), running.end(),
      [](const SequencePtr &a, const SequencePtr &b) { return a->get_output_len() > b->get_output_len(); });
  size_t l = 0, r = running.size();
  while (l < r && scheduled.size() < max_seqs) {
      auto &seq = running[r - 1]; // Shortest output length scheduled first
      while (!block_manager->can_allocate_single_nolock() && r - l > 1) {
          auto victim_seq = running[l]; // Longest output length preempted first
          _preempt(victim_seq, ignored_seqs);
          l += 1;
      }
      ...
  }
  ```
  `running[l]` ($l=0$) is the sequence with the *longest* generated output. DeepInfer sacrifices the oldest, longest running request to preserve throughput for newer, shorter jobs.

#### 3. Multi-Tier KV-Cache & StateBlock Memory Architecture
- **Implementation:** `deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L24-L140`, `deepinfer/worker/cache_engine.py#L205-L241`.
- **Physical Memory Layouts:**
  - `ca_cache` (Compressed Attention Cache): Shape `[num_blocks, block_size // ratio, 1, bytes_per_token]`.
    - **FP4 (`V41_FP4_FORMAT`):** 512 elements per token at 4 bits (256 bytes) + 32 E4M3 scale factors (32 bytes) = **288 bytes/token**.
    - **MXFP8 (`V41_FORMAT`):** 512 elements at 8 bits (512 bytes) + 16 UE8M0 scale factors (16 bytes) = **528 bytes/token**.
    - **Token Interleaving:** Data and scales are interleaved per token (`token_interleaved=True`) to maintain 32-byte memory bus alignment (`gcd(32, bytes_per_token)`).
  - `ica_cache` & `ica_state`: Index attention cache for sparse token selection (68 bytes/token) and recurrent state slots.
  - `swa_cache`: Sliding Window Attention cache (`[num_state_seqs, swa_block_size, 1, 528]`).
  - `StateBlock` Disaggregation Partitioning: In `cache_engine.py#L221-L241`, memory is explicitly split into:
    - `Stored Region`: SWA window required for persistence and Prefill Engine (PE) $\to$ Completion Engine (CE) disaggregated network transfer.
    - `Dropped Region`: YOCO consumer SWA and compression states, which are discarded and recomputed on the fly to save interconnect bandwidth.

#### 4. Speculative Decoding & Multi-Token Prediction (MTP) Varlen
- **Implementation:** `csrc/engine/mtp_varlen.hpp#L20-L276`, `deepinfer/worker/dflash_cuda_graph_runner.py#L31-L70`.
- **Statistical Estimators:**
  - `SamplerRuntimeEstimator`: Rolling 9-step median of sampler GPU execution time.
  - `ForwardRuntimeEstimator`: Dual-multiset median tracker (`MedianTracker` using two balanced `std::multiset` containers) maintaining a 5,000-step sliding window to fit the context-dependent slope of forward execution time: $\text{time} = \text{baseline} + \text{slope} \cdot \text{graph\_size} + \text{sampler\_ms}$.
  - Dynamic draft allocation: In `mtp_varlen.hpp#L235-L267`, it computes expected tokens per millisecond for each candidate batch size, selects `best_bsz`, and dynamically distributes draft validation slots ($K \in [1, 5]$) to sequences with the highest confidence scores.

#### 5. Bucketed CUDA Graphs & Shared Memory Pooling
- **Implementation:** `deepinfer/worker/cuda_graph_runner.py#L41-L148`.
- **Reverse Capture Strategy:** Graph capture iterates over `reversed(self.batch_sizes)` (lines 102-140). By capturing the *largest* batch size first, PyTorch initializes the memory pool at its maximum allocation envelope. Line 140 sets `self.graph_memory_pool = graph.pool()`, and all subsequent smaller batch graphs share this exact pool, eliminating both activation memory multiplication and CPU kernel launch overhead.

#### 6. Multi-Head Latent Attention (MLA), MQA Sparse Indexer & EPLB
- **Implementation:** `deepinfer/model_executor/models/v4_modules/mla_attn.py#L26-L140`, `deepinfer/model_executor/models/v4_modules/moe.py#L40-L100`, `deepinfer/model_executor/eplb/__init__.py#L21-L60`.
- **Two-Stage Sparse Attention:**
  - `MQAIndexer` (lines 100-140): Implements FP4 quantized sparse index attention (`self.use_fp4 = True`) with DeepSelect top-k index selection (`cfg.index_topk = 256`).
  - `FlashMLA` (lines 26-62): Low-rank projection ($A \to \text{norm} \to B$) computed strictly over the selected sparse tokens and SWA window.
  - `SharedLayerTopology` (`v4_modules/cross_layer.py`): Layers with `layer_map[l] == l` act as source projection layers, and subsequent layers reuse projection weights to minimize VRAM footprints.
- **MoE Routing & EPLB:** Top-k routing uses `sqrtsoftplus` scoring with `noaux_tc` bias and dedicated text/vision correction biases (`e_score_correction_bias` and `e_score_correction_bias_vl` in `moe.py#L62-L64`). `ExpertLoadBalancer` tracks token arrivals per physical expert, recomputes optimal expert placement, and dynamically shuffles redundant expert weights across NVLink using CUDA symmetric memory handles (`_symmetric_memory_handles` in `eplb/__init__.py#L58`).

#### 7. Structured Output Constrained Decoding
- **Implementation:** `deepinfer/structure_output/matcher.py#L24-L188`.
- **Trigger-Bound Matching:** Uses `xgrammar` to compile grammars into vocabulary bitmasks. A Knuth-Morris-Pratt matcher (`_kmp_failure` on line 24) tracks prefix progress for trigger sequences (e.g., tool call start tokens). When `bound_token_id` is matched, it activates constrained mask evaluation strictly within the argument window and restores open vocabulary upon reaching the stop token.

#### 8. DeepInfer Weaknesses & Failure Modes
1. **CPU Decoding Bottleneck:** `deepinfer/engine/llm_engine.py#L931` contains an explicit developer comment: `# TODO(CPU): Eliminate O(num_reqs) python operations in decoding step`. Under high batch sizes ($>256$), Python object overhead, GIL contention, and list comprehensions limit decode throughput.
2. **Raw Pointer Passing:** Pointers to `RequestOutput` objects are cast to `np.uint64` raw addresses (`llm_engine.py#L864`, `worker.py#L95`). If an upstream engine reads a pointer after `DeferredRelease` recycles it, it triggers undefined memory access or segmentation faults.
3. **Missing Production Serving Layer:** No HTTP server, rate governor, admission queue, client socket disconnection monitoring, or request cancellation propagation. Disconnected clients continue consuming GPU compute until sequence completion.

---

### Part 2: Deep Code Audit of Teacher 2 — NOVAV2

NOVAV2 (`NOVAV2-main`) is an industrial-strength, pure-Rust routing, relay, and fleet management engine built with Tokio and Axum.

#### 1. Zero-Allocation Tokio Actor Model & Clean Subsystem Isolation
- **Implementation:** `src/relay/server.rs#L36-L100`, `src/relay/handlers.rs#L38-L60`.
- **Subsystem Architecture:** All operational state is encapsulated in an `Arc<AppState>` container:
  - `fleet`: Shard registry with dynamic health probing.
  - `rate_governor`: Two-tier leaky token bucket and sliding window quota enforcer.
  - `circuit_breaker`: Four-state node circuit breaker with canary slots.
  - `job_store`: In-memory atomic job registry.
  - `idempotency`: Atomic SHA256 request deduplication.
  - `admission_receipts`: Crash-resilient request receipt store with `fsync`.
  - Zero GIL contention, strict Rust type safety, and clean shutdown draining via `is_draining: Arc<AtomicBool>`.

#### 2. Resilient Streaming & Sliding Ring-Buffer Replay
- **Implementation:** `src/relay/job.rs#L44-L100`.
- **Mechanics:** Each inference job is represented by a `JobRecord`:
  - `broadcast_tx: broadcast::Sender<StreamEvent>`: Tokio broadcast channel supporting fan-out delivery to multiple SSE/WebSocket client connections.
  - `ring_buffer: VecDeque<StreamToken>`: Bounded sliding window of recent tokens tagged with monotonically increasing sequence IDs (`seq_no`).
  - *Network Replay:* When a mobile or flaky client drops connection and reconnects with `Last-Event-ID: <seq_no>`, the server scans `ring_buffer` and immediately replays missed tokens without interrupting or duplicating GPU inference.

#### 3. Deterministic Admission State Machine & Backpressure
- **Implementation:** `src/relay/admission.rs#L17-L80`.
- **Formal State Transitions:**
  $$\text{Received} \longrightarrow \text{Queued} \longrightarrow \text{Admitted} \longrightarrow \text{Dispatched} \longrightarrow \{\text{Completed} \mid \text{Cancelled} \mid \text{TimedOut} \mid \text{Rejected} \mid \text{Failed}\}$$
  - *Disconnect Propagation:* Client sockets are monitored on every stream poll. If a client drops, `client_disconnected` is immediately set to `true`, transitioning the admission machine to `Cancelled` and sending cancellation signals upstream.
  - *Atomic Journaling:* `AdmissionReceiptsStore` writes admission receipts to disk with atomic temp-file renaming and `fsync`, guaranteeing idempotency and crash recovery.

#### 4. Smooth Weighted Round Robin (SWRR) & Circuit Breakers
- **Implementation:** `src/pool/swrr.rs#L23-L60`, `src/pool/circuit_breaker.rs#L9-L70`, `src/pool/rate_governor.rs#L8-L70`.
- **Fleet Balancing:**
  - `SwrrBalancer`: Interleaves requests across heterogeneous shards according to effective weights, eliminating burst hotspots on high-capacity nodes.
  - `CircuitBreaker`: Four-state machine (`Healthy`, `Degraded`, `Exhausted`, `Disabled`). On consecutive failures, trips to `Exhausted` with an exponential cooldown timer. Recovery requires acquiring a single-caller canary slot (`acquire_canary_slot`) to probe node health before restoring full traffic.
  - `RateGovernor`: Combines a continuous leaky token bucket (`TokenBucket`) for instantaneous concurrency with a sliding-window duration limiter (`SlidingWindow`) for daily quotas.

#### 5. NOVAV2 Weaknesses & Omissions
1. **No Embedded GPU Execution Engine:** Operates entirely as a network relay over WebSockets/SSE upstreams (`src/relay/upstream.rs#L36-L80`). It contains no CUDA driver calls, tensor memory managers, or kernel dispatchers.
2. **Missing Local KV-Cache & Attention Logic:** Assumes all memory management and KV caching is handled remotely.

---

### Part 3: Deep Code Audit of Teacher 3 — MOSS-Studio

MOSS-Studio (`MOSS-Studio`) contains two distinct layers: a high-level creative video/image orchestration workflow, and a standalone production Rust backend: `MOSS-Studio/modal-studio/moss-backend`.

#### 1. The Discovered Industrial Rust Subsystem (`moss-backend`)
- **Implementation:** `moss-backend/Cargo.toml`, `moss-backend/src/store.rs`, `moss-backend/src/worker.rs`.
- Verified compilation: Successfully compiled via `cargo check` in 1.36 seconds on macOS arm64.
- **Crash-Resilient State Persistence (`store.rs`):**
  - Implements `StudioStore`, managing transactional persistence for jobs, images, videos, and settings.
  - Employs atomic write semantics: writes JSON state to temporary files, flushes via `fsync`, and executes atomic POSIX renames (`rename`) with startup stale-file sweeping (`sweep_stale_temporary_files` in `store.rs#L70`).
  - Provides dual-engine migration between SQLite (`studio.sqlite3`) and serialized JSON manifests (`workspace.json`).
- **Dedicated Background Worker Queue (`worker.rs`):**
  - Implements an asynchronous worker thread consuming from a job queue, synchronizing via `tokio::sync::Notify` and semaphores (`Arc<Semaphore>`), decoupling submission latency from GPU rendering time.
- **Subsystem Circuit Breaker (`pool/circuit_breaker.rs`):**
  - Adapts node fault tolerance to remote Modal GPU worker invocations with canary probing.

---

### Part 4: Critique and Correction of Prior Investigation

| Subsystem / Topic | Prior Attempt Claim | Evidence Cited | Verified Code Reality | Corrected Architectural Finding |
| :--- | :--- | :--- | :--- | :--- |
| **Decode Preemption Direction** | Preempts sequences with the *shortest* output (`running[l]`) to prioritize requests closest to completion. | `csrc/engine/scheduler.hpp#L622-L646` | `std::sort` orders descending by output length (`a > b`). `running[0]` has the longest output; `running[r-1]` is shortest. Scheduled loop picks `r-1` and preempts `running[l]` ($l=0$). | **Inverted Finding:** DeepInfer implements Last-Come-First-Served (LCFS), preempting the *longest* running requests to prevent newly arrived requests from starving. |
| **KV-Cache Prefix Caching** | `BlockManager` implements reference counting for Copy-On-Write (COW) prefix caching. | `csrc/engine/block_manager.hpp#L27-L187` | No Radix Tree, hash table, or prefix lookup exists in `block_manager.hpp`. `pin_block` is used solely for sequence forking (`scheduler.hpp#L1110`) and deferred ZOS unpinning (`llm_engine.py#L116`). | **Fabricated Feature:** DeepInfer omitted its KV cache / prefix-cache system from open-source release (as stated in `README.md#L38-L43`). SINGULARITY must build its own Rust Radix tree. |
| **Teacher 3 Scope (MOSS-Studio)** | Dismissed as a ComfyUI video looping script and prompt template repository. | `howitworks.md` | Contains `modal-studio/moss-backend`, an industrial pure-Rust backend (`Cargo.toml`, Axum 0.7, Rusqlite, `store.rs`, `worker.rs`). | **Critical Miss:** Overlooked a 64KB Rust transactional persistence engine that directly informs SINGULARITY's crash recovery and job lifecycle architecture. |
| **CUDA Graph Pool Capture** | Captures power-of-two graphs sharing a pool, without explaining allocation order. | `deepinfer/worker/cuda_graph_runner.py#L41-L100` | Loop executes over `reversed(self.batch_sizes)` (lines 102-140). The *largest* batch size is captured first to size the pool; smaller graphs reuse it. | **Mechanics Gap:** Captured graphs must be recorded in descending size order so the memory pool is pre-sized to the maximum envelope without runtime reallocation. |
| **Attention Architecture** | Simple low-rank projection FlashMLA ($A \to \text{norm} \to B$). | `deepinfer/model_executor/models/v4_modules/mla_attn.py#L26-L98` | DeepSeek-V4.1 uses a 2-stage hybrid attention: Stage 1 FP4 `MQAIndexer` + DeepSelect top-k indexer; Stage 2 dense FlashMLA. | **Incomplete Attention:** SINGULARITY must support 2-stage sparse indexing (top-k FP4 indexer) and `SharedLayerTopology` to match DeepSeek-V4.1 efficiency. |
| **KV-Cache Physical Layout** | Listed cache names (`ca_cache`, `swa_cache`) without memory alignment details. | `deepinfer/worker/cache_engine.py#L205-L300` | `cache_common.py#L24-L140` proves data and scales are interleaved per token (`token_interleaved=True`) to maintain 32-byte alignment. | **Memory Stride Gap:** Rust cache manager must implement token-interleaving and `StateBlock` PE $\to$ CE disaggregated memory partitioning. |
| **Mandate Section 6 Compliance** | Completely omitted Section 6 of `singularity.md` (Parent Agent & Skill Execution). | None | Section 6 is an explicit mandate requirement covering agent workflows, task decomposition, and execution authority. | **Scope Omission:** Fully restored and developed in Part 8 of this report. |

---

### Part 5: Hardware Target Analysis — NVIDIA B200 (SM100) vs H200 (SM90)

| Architectural Metric / Feature | NVIDIA H200 SXM (Hopper SM90) | NVIDIA B200 SXM (Blackwell SM100) | SINGULARITY Microarchitectural Strategy |
| :--- | :--- | :--- | :--- |
| **Compute Capability** | SM 9.0 (`sm_90a`) | SM 10.0 (`sm_100a`) | Runtime capability query via `cuDeviceGetAttribute`. |
| **HBM Technology & Capacity** | 141 GB HBM3e | 180 GB HBM3e (or 192 GB NVL) | Dynamic KV cache sizing reserving 90% of free HBM3e. |
| **Memory Bandwidth** | 4.8 TB/s | 8.0 TB/s ($1.67\times$ speedup) | Auto-tune batch sizes and SWA window sizes to bandwidth limits. |
| **Native Microscaling FP4 (NVFP4)**| **Not Supported** (requires emulated dequant) | **Native 2nd Gen Tensor Cores (NVFP4)** | Compile SM100 CUBINs for NVFP4 (E2M1); fallback to FP8 on SM90. |
| **Dense FP8 Tensor Cores** | ~1,979 TFLOPS (E4M3 / E5M2) | ~4,500 TFLOPS ($2.27\times$ speedup) | Hopper uses FP8 (E4M3); Blackwell uses FP8 or NVFP4. |
| **Tensor Memory Accelerator (TMA)** | TMA v1 (Async global-to-shared copy) | TMA v2 (Enhanced concurrency, L2 direct) | Issue TMA descriptors for FlashMLA block loading. |
| **NVLink Interconnect** | 4th Gen NVLink (900 GB/s bidirectional) | 5th Gen NVLink (1,800 GB/s bidirectional) | DeepEP token dispatch across NVLink; 50% lower MoE latency on B200. |
| **Hardware Decompression Engine** | None (CPU or CUDA kernel decompress) | Dedicated Hardware Decompression Engine | Stream compressed Safetensors directly to GPU memory on B200. |

#### Hardware Capability Interrogation & Graceful Fallback Matrix
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceArchitecture {
    BlackwellSm100 { hbm_bytes: usize, nvlink_gen: u32, has_native_fp4: bool },
    HopperSm90 { hbm_bytes: usize, nvlink_gen: u32, has_native_fp4: bool },
    Unsupported { major: i32, minor: i32 },
}

pub fn detect_hardware(device_id: i32) -> Result<DeviceArchitecture, SingularityError> {
    let major = get_device_attribute(device_id, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
    let minor = get_device_attribute(device_id, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
    let hbm_bytes = get_total_device_memory(device_id)?;

    match (major, minor) {
        (10, 0) => Ok(DeviceArchitecture::BlackwellSm100 { hbm_bytes, nvlink_gen: 5, has_native_fp4: true }),
        (9, 0) => Ok(DeviceArchitecture::HopperSm90 { hbm_bytes, nvlink_gen: 4, has_native_fp4: false }),
        _ => Err(SingularityError::UnsupportedHardware(major, minor)),
    }
}
```
**Fallback Logic:** If an FP4 model checkpoint is loaded on an H200 (SM90) accelerator, SINGULARITY emits an explicit startup warning, loads the FP4 weights, and performs an on-the-fly dequantization pass to FP8 (E4M3) in pinned host memory or via a fast conversion kernel before populating HBM3e, preventing illegal PTX instruction traps.

---

### Part 6: Student Architecture Specification — Project SINGULARITY

The destination for SINGULARITY is `/Users/bozoegg/Desktop/HEAV3NS-BRAIN`. It is structured as an independent, high-performance Rust workspace composed of 7 cleanly bounded crates:

```text
/Users/bozoegg/Desktop/HEAV3NS-BRAIN/
├── Cargo.toml                          # Workspace root definition
├── singularity-core/                   # Shared types, error handling, config, metrics, traits
├── singularity-cuda/                   # CUDA 12/13 Driver FFI, stream/event management, cuGraph
├── singularity-cache/                  # Virtual memory manager, FP4/FP8/SWA cache, Radix prefix tree
├── singularity-scheduler/              # Pure-Rust ZOS engine, continuous batching, DP voting, LCFS
├── singularity-model/                  # Safetensors mmap loader, FlashMLA + MQA indexer, MoE EPLB
├── singularity-grammar/                # JSON Schema/regex bitmask engine & KMP trigger matcher
├── singularity-server/                 # Axum 0.7 REST/SSE server, admission machine, ring buffer replay
├── benches/                            # Criterion microbenchmarks (TTFT, ITL, scheduling overhead)
└── tests/                              # Integration, multi-threaded stress, and recovery tests
```

#### Detailed Subsystem Responsibilities:

1. **`singularity-core` (Foundational Substrate):**
   - Universal error types (`SingularityError`) with source backtraces.
   - Request and sequence representations (`Sequence`, `SequenceMetadata`, `SamplingParams`).
   - Telemetry metrics using `metrics` and `tracing` (p95/p99 TTFT, ITL, KV block utilization).

2. **`singularity-cuda` (CUDA Driver API & CudaGraph Runner):**
   - Direct FFI bindings to CUDA Driver API (`libcuda.so` / `nvcuda.dll`) via `cudarc`.
   - Asynchronous stream and event management (`CudaStream`, `CudaEvent`).
   - **Bucketed CudaGraph Runner:** Pre-captures execution graphs across batch sizes $[1, 2, 4, 8, \dots, 256]$ iterating in **reverse order** to establish a shared `cuMemAlloc` memory pool (`cuGraphInstantiateWithFlags`), eliminating activation re-allocation.
   - Low-level CUDA Virtual Memory Management (VMM) via `cuMemAddressReserve`, `cuMemCreate`, and `cuMemMap`.

3. **`singularity-cache` (Virtual Memory Manager & Radix Prefix Cache):**
   - High-throughput `PagedBlockManager` using atomic bitsets for $O(1)$ allocation and zero lock contention.
   - Implements the multi-tier token-interleaved memory layout:
     - FP4 E2M1 (288 bytes/token) on B200; MXFP8 E4M3 (528 bytes/token) on H200.
     - SWA ring-buffered window with backtracking protection.
   - **True Radix Tree Prefix Cache:** Stores prompt token prefixes in an in-memory Radix tree with reference counting and LRU eviction, enabling zero-compute prefix reuse across requests.
   - `StateBlock` partitioning: Separates persistent SWA regions from dropped YOCO state for efficient network disaggregation.

4. **`singularity-scheduler` (Pure-Rust Zero-Overhead Scheduler):**
   - **Rust ZOS Engine:** Step N+1's forward pass is submitted to the GPU compute stream immediately, executing concurrently with CPU post-processing of Step N's outputs.
   - Prefill/Decode disaggregation (`EngineType::Prompt`, `EngineType::Completion`, `EngineType::Normal`).
   - Multi-GPU DP consensus voting (ALL point vs ANY point) to synchronize step types across data-parallel ranks.
   - LCFS decode prioritization with victim preemption of longest-running sequences when blocks are exhausted.
   - Multi-Token Prediction (MTP) dynamic draft scheduling using the dual-multiset median runtime estimator.

5. **`singularity-model` (Model Execution, FlashMLA & MoE EPLB):**
   - Zero-copy Safetensors loader using memory-mapped I/O (`memmap2`) and direct DMA streaming to HBM3e.
   - Direct FFI dispatch to `FlashMLA` (compressed KV attention) and `FlashAttention-3`.
   - Two-stage attention: `MQAIndexer` FP4 sparse filtering followed by dense FlashMLA attention.
   - MoE router with `sqrtsoftplus` scoring, `noaux_tc` bias, and DeepEP all-to-all NVLink communication.
   - Dynamic EPLB: Tracks expert load, recomputes placement, and executes zero-downtime weight shuffling across GPUs using symmetric memory buffers.

6. **`singularity-grammar` (Constrained Decoding & Tool Calling):**
   - High-speed grammar compiler producing 32-bit vocabulary bitmasks from JSON Schemas and regex.
   - Knuth-Morris-Pratt (KMP) trigger matcher monitoring token streams for tool-invocation patterns, engaging grammar validation only inside argument boundaries and freeing constraints upon exit tokens.

7. **`singularity-server` (High-Throughput OpenAI REST & SSE Server):**
   - Axum 0.7 HTTP server exposing `/v1/chat/completions` and `/v1/models`.
   - Adapted from NOVAV2: Deterministic admission state machine (`AdmissionLifecycle`), two-tier token bucket rate governor, and four-state circuit breaker with canary slots.
   - Sliding ring buffer (`VecDeque<StreamToken>`) with broadcast channels: Allows dropped client connections to reconnect with `Last-Event-ID` and resume streaming without re-executing GPU inference.
   - Eager client disconnect detection: Polling the TCP stream terminates orphaned GPU jobs within microseconds.

---

### Part 7: Hardening & Failure Remediation Plan (All 18 Failure Modes)

1. **Resource Exhaustion & GPU OOM:** Strict virtual memory reservation via CUDA 12 VMM (`cuMemAddressReserve`). Deterministic LCFS preemption evicts victims before memory pressure triggers kernel aborts.
2. **CPU-GPU Execution Bubbles:** Pure-Rust worker thread pinned to isolated CPU cores communicating with Tokio frontend via lock-free ring buffers (`rtrb`). ZOS keeps GPU execution streams continuously saturated.
3. **Flaky Client Drops & Orphaned GPU Work:** NOVAV2-style socket drop detection flags `client_disconnected: true`, removing aborted sequences from the scheduler's running queue on the very next step.
4. **Multi-GPU Communication Deadlocks:** Synchronized DP consensus voting (ALL vs ANY) and explicit timeout guards on all NCCL operations with deterministic recovery fallbacks.
5. **Memory Leaks in KV Block Allocators:** RAII guards (`Drop` trait) wrap block handles and state IDs. Destruction of a request handle automatically returns its blocks to `PagedBlockManager`.
6. **Speculative Decoding Miss Penalties:** Online `ForwardRuntimeEstimator` and `SamplerRuntimeEstimator` track throughput dynamically; if confidence falls below threshold, draft length $K$ collapses to 0.
7. **Process Crashes & State Corruption:** MOSS-Studio-style transactional persistence (`store.rs`): atomic temporary files, `fsync`, and POSIX renames ensure clean recovery upon restart.
8. **CUDA Graph Instantiation Failures:** Static memory envelopes and pinned staging buffers; capture graphs in reverse order to establish a single immutable memory pool.
9. **Backpressure Breakdown Under Bursts:** Two-tier rate governors with continuous leaky token buckets and strict admission bounds (`max_waiting_tokens`), returning immediate HTTP 429 when capacity is saturated.
10. **Partial Initialization Hangs:** Coordinated initialization handshake with explicit timeout deadlines across all worker threads.
11. **Serialization Inconsistencies:** Direct byte-level grammar bitmask verification in `singularity-grammar`, guaranteeing valid JSON emissions before tokens reach the client.
12. **Network Connection Drops During SSE Streaming:** Sliding ring buffer replay (`VecDeque<StreamToken>`) retaining the last $N$ tokens for seamless reconnect via `Last-Event-ID`.
13. **Invalid Tool Call JSON Syntax:** KMP trigger matcher locks decoding into the compiled JSON Schema bitmask the instant tool call syntax is detected.
14. **Unmatched Hardware Capabilities:** Explicit hardware interrogation at boot via `cuDeviceGetAttribute` with graceful on-the-fly dequantization fallback to FP8.
15. **MoE Expert Imbalance & Hotspots:** Dynamic EPLB weight shuffling across NVLink using symmetric memory buffers, duplicating hot experts to redundant slots.
16. **Scheduling Starvation Under Heavy Prefill:** Chunked prefill scheduling and ANY-point DP voting intervals guaranteeing decode slots every $M$ steps.
17. **Model Loading OOM on Host:** Memory-mapped Safetensors (`memmap2`) streaming weights directly from disk into GPU HBM3e via zero-copy DMA.
18. **Radix Tree Cache Stale Key Invalidation:** Cryptographic SHA256 hashing of token sequences combined with strict cache slot generation counters.

---

### Part 8: Parent Agent & Skill Execution Workflow Improvements

1. **Parent-Agent Concurrency & Flow Optimization:**
   - **Parallel Decomposition:** Subsystems with orthogonal boundaries (`singularity-cache`, `singularity-grammar`, `singularity-server`) can be developed and tested concurrently using mock GPU drivers.
   - **Non-Blocking Delegation:** The parent orchestrator maintains an asynchronous coordinator role: delegating well-scoped, self-contained subtasks to worker agents with explicit input/output interfaces rather than issuing serial step-by-step commands.
   - **Shared Architectural Context:** Maintain a clean workspace manifest (`ARCH_STATE.md`) tracking verified code facts, eliminating duplicate codebase re-investigations across subagents.

2. **Skill Execution Audit & Overhead Reduction:**
   - **Targeted Tool Invocation:** Eliminate exploratory grep/find sweeps. Use targeted line-range reads and symbol greps with specific path scopes to preserve context window tokens.
   - **Direct Deterministic Execution:** Execute non-destructive commands (builds, tests, checks) directly without inserting unnecessary interactive approval pauses.
   - **Output Integration:** Subagent outputs must be critically verified against code reality before being accepted into the architectural state.

