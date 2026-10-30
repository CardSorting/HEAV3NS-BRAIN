Okay. I think the previous briefs were still treating this too much like a model-porting exercise. The actual objective is considerably bigger.

SINGULARITY is the independent inference subsystem of HEAV3NS Code. We are building the engine, not wrapping DeepInfer, not deploying somebody else's engine under our name, and not producing a demonstration that happens to generate a few tokens.

The complete objective is to take the existing Rust workspace, turn its claimed Phase II capabilities into verified execution, implement the actual DeepSeek V4.1-Flash architecture, build the deployment infrastructure, and operate the resulting system as a real inference service. Then measure it against DeepInfer on identical B200 hardware.

There are two important corrections the commission needs to preserve:

- The `1017` checkpoint is a weight refresh of the V4.1-Flash architecture, not an invitation to implement an approximate generic transformer. Its native mixed checkpoint uses MXFP4 experts and MXFP8 dense weights. The architecture includes Engram, sparse attention, Hyper-Connections, and DSpark. The earlier Phase II report's MTP description must not be treated as authoritative.&#x20;

  [image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

  GitHub+1

- DeepInfer is the reference engine, but its public release is an offline inference engine with some pieces of a complete online serving system explicitly absent. SINGULARITY's objective includes the operational serving infrastructure around its engine, not merely reproducing DeepInfer's offline execution.&#x20;

  [image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

  GitHub+1



Here is the full commission I'd use.

# PROJECT SINGULARITY

## NOVA v4 — HEAV3NS-BRAIN

### Phase III: Independent Inference Engine, DeepSeek V4.1 Execution, Modal Infrastructure, Deployment & Operational Validation

**Commission type:** Full-stack inference system implementation and deployment\
**Primary language:** Rust\
**Infrastructure orchestration:** Python / Modal\
**Target accelerator:** 8 × NVIDIA B200 (Blackwell SM100)\
**Flagship model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017`\
**Reference implementation:** DeepSeek-AI DeepInfer\
**Student workspace:** `/Users/bozoegg/Desktop/HEAV3NS-BRAIN`\
**Teacher repositories:**

- `/Users/bozoegg/Downloads/DeepInfer-main`
- `/Users/bozoegg/Downloads/NOVAV2-main`

---

# 0. Executive mandate

Build SINGULARITY into a genuinely operational, independently implemented inference system capable of running and serving DeepSeek V4.1-Flash-1017 on eight NVIDIA B200 GPUs.

The assignment encompasses the complete system:

1. Inspect and understand the existing SINGULARITY implementation.
2. Study both teacher repositories and extract the relevant architectural and implementation knowledge.
3. Implement and verify the actual model architecture.
4. Build a real GPU execution backend optimized for B200.
5. Implement the model loading, memory management, scheduling, inference, sampling, and serving paths.
6. Build a reproducible Linux container.
7. Create the Modal deployment pipeline.
8. Provision eight B200 GPUs.
9. Download and validate the actual model checkpoint.
10. Launch SINGULARITY as a persistent inference service.
11. Exercise it through its remotely accessible API.
12. Run sustained inference and operational validation.
13. Deploy DeepInfer independently on equivalent hardware.
14. Conduct reproducible, controlled performance comparisons.
15. Deliver a functioning system, its deployment infrastructure, and its operational evidence.

**The required outcome is a running system, not a report describing a system that could run.**

A successful compilation is not completion.

A successful test suite is not completion.

A container that starts is not completion.

A model that loads without generating verified output is not completion.

The assignment ends when SINGULARITY is deployed, performing real inference on B200 hardware, and remains operational.

---

# 1. Architectural identity and project boundaries

SINGULARITY is the Rust inference engine being developed within NOVA v4 / HEAV3NS-BRAIN.

It is an independent implementation informed by two teacher repositories:

- DeepInfer
- NOVAV2

These repositories are references for architectural research, implementation patterns, optimizations, and operational behavior.

They are not runtime dependencies that replace SINGULARITY.

They are not substitutes for implementing the student system.

They must not be used to disguise a wrapper around another inference engine as an independently implemented Rust runtime.

## 1.1 System ownership

SINGULARITY must own its own:

- Model architecture and forward execution
- Tensor representations
- GPU execution backend
- Memory allocation
- KV cache
- Model weight loading
- Quantization handling
- Attention implementation
- MoE routing and expert execution
- Scheduling
- Sampling and decoding
- Speculative decoding integration
- Grammar-constrained generation
- Tool-call handling
- Request lifecycle
- Streaming generation
- API execution path
- Runtime telemetry
- Benchmark instrumentation

Modal provides infrastructure and lifecycle management.

Python provides deployment orchestration.

Neither replaces the Rust inference runtime.

## 1.2 System architecture

The intended architecture is:

```
                   HEAV3NS CODE
                        |
                        |
                   SINGULARITY
                        |
          +-------------+-------------+
          |             |             |
       MODEL         RUNTIME        SERVING
          |             |             |
     Architecture   Scheduler       REST API
     Tensor Loader   KV Cache        SSE
     Quantization    GPU Backend     Tool Calls
     Attention       Batching        Grammar
     MoE Execution   Sampling        Request State
     Engram          Speculation     Telemetry
          |             |             |
          +-------------+-------------+
                        |
                  CUDA / NCCL
                        |
                  8 × B200 GPUs
                        |
                 MODAL CONTAINER
                        |
                PERSISTENT SERVICE
                        |
                 REMOTE ENDPOINT
```

The system must retain a clean distinction between model execution, runtime orchestration, and infrastructure.

---

# 2. Establish the actual implementation baseline

Before modifying anything, inspect the complete existing workspace.

The Phase II completion report claims:

- Nine-crate Rust workspace
- CUDA backend using `cudarc`
- Transformer forward execution
- GQA and MLA
- DeepSeekMoE
- Multi-shard Safetensors
- Tokenizer integration
- Device-backed KV cache
- Scheduler integration
- Axum REST/SSE server
- 61 passing tests

These are implementation claims to verify, not assumptions of completion.

The previous reality audit demonstrated why this distinction matters.

## 2.1 Existing workspace

Preserve and inspect the existing crate boundaries:

| Crate                   | Responsibility                                                          |
| ----------------------- | ----------------------------------------------------------------------- |
| `singularity-core`      | Shared types, configuration, errors, metrics                            |
| `singularity-cuda`      | Hardware discovery, CUDA backend, GPU execution                         |
| `singularity-cache`     | Paged KV cache, radix prefix cache, memory layouts                      |
| `singularity-scheduler` | ZOS, request scheduling, preemption, distributed coordination           |
| `singularity-model`     | Model loading, tensors, architecture, forward execution                 |
| `singularity-grammar`   | Structured output, JSON Schema, grammar constraints, tool-call matching |
| `singularity-server`    | REST, SSE, request lifecycle, streaming                                 |
| `singularity-cli`       | Runtime commands, inspection, benchmarking                              |
| `singularity-tests`     | Functional, integration, correctness, and failure tests                 |

Do not discard existing functional infrastructure unnecessarily.

Do not preserve broken abstractions merely because they already exist.

Refactor where required to support actual execution.

## 2.2 Baseline deliverables

Produce:

- Repository structure report
- Current build results
- Existing test results
- CUDA feature build results
- Current subsystem implementation matrix
- Dependency inventory
- Existing technical debt inventory
- Git revision and working-tree state

Every subsystem must be classified:

- Operational and verified
- Implemented but not independently verified
- Partial
- Mocked
- Specification only
- Missing

Explicitly locate and eliminate any production execution paths that still generate synthetic tokens, synthetic benchmark numbers, fake GPU addresses, or placeholder model outputs.

Keep the CPU mock backend for local development and CI. Never allow it to masquerade as real accelerator execution.

---

# 3. Teacher research and independent implementation

Study both teacher repositories before committing to architecture changes.

## 3.1 DeepInfer

Study the actual implementation of:

- Model execution
- V4.1-specific architecture
- FlashMLA
- DeepGEMM
- DeepEP
- TileKernels
- DeepSelect
- TileLang kernels
- Expert parallelism
- Expert load balancing
- Attention data parallelism
- Context parallelism
- Zero-overhead scheduling
- Asynchronous execution
- Batch-invariant inference
- Deterministic sampling
- DSpark
- Confidence-scheduled verification
- SWA bounded replay
- Structured output

Extract the relevant execution algorithms, memory layouts, tensor operations, kernel strategies, and scheduling behavior.

## 3.2 NOVAV2

Inspect NOVAV2 for reusable architectural ideas, runtime structures, abstractions, and implementation patterns relevant to SINGULARITY.

Determine what should be adapted, what should be redesigned, and what should remain independent.

## 3.3 Research output

Create a technical design record mapping:

- Teacher implementation
- Relevant source location
- Underlying algorithm
- SINGULARITY destination
- Required adaptation
- Correctness test
- Performance validation

Preserve source attribution and respect repository and dependency licenses.

Do not blindly port the entire teacher implementation.

Do not introduce Python or another inference framework into the critical model execution path simply because a reference implementation uses it.

The Rust implementation remains the primary engine.

---

# 4. Implement DeepSeek V4.1-Flash-1017

## 4.1 Exact model target

Hugging Face repository:

`deepseek-ai/DeepSeek-V4.1-Flash-1017`

Use the exact checkpoint revision and record its immutable revision identifier.

The implementation must be driven by the actual model configuration, tokenizer assets, tensor manifest, and quantization metadata.

Do not infer architecture from the model name.

Do not use a generic transformer implementation and claim compatibility.

## 4.2 Architecture requirements

The implementation must account for the model's actual architecture, including:

### Model structure

- Approximately 552B backbone parameters
- Approximately 196B Engram memory parameters
- 8B active parameters per prompt token
- 16B active parameters per generated token
- 40 transformer layers
- Hidden dimension 5120
- Causal encoder-decoder architecture
- 20-layer causal encoder
- 20-layer decoder
- Vision Transformer with 32 layers and associated aligner

### Sparse attention

Implement the actual two-tier sparse attention mechanism.

This includes:

- Sliding-window attention
- Compressed KV representations
- Learned compression pooling
- Sparse indexer
- Candidate selection
- Top-k sparse retrieval
- Attention score calculation
- Correct attention output reconstruction
- KV storage and retrieval

The implementation must follow the actual checkpoint configuration rather than relying on generic attention abstractions.

### Engram

Implement Engram memory access and integration.

Account for:

- Model-specific hash tables
- Hash lookup
- Embedding retrieval
- Memory placement
- GPU memory requirements
- Synchronization
- Integration with the model's execution path

Engram is not an optional metadata feature. It is part of the model.

### Hyper-Connections

Implement the model's actual multi-stream residual architecture.

Support:

- Parallel residual streams
- Learned mixing coefficients
- Pre-sublayer and post-sublayer transformations
- Stream combination
- Required normalization behavior

A conventional single residual stream must not silently replace this architecture.

### Mixture of Experts

Implement actual expert execution.

Support:

- Expert routing
- Routing bias
- Top-k selection
- Shared experts
- Routed experts
- Expert weight loading
- Expert placement
- Expert dispatch
- GPU GEMM execution
- Expert result aggregation
- Load balancing

Host-side expert scoring without actual expert execution does not count.

### Quantization

The exact checkpoint uses mixed native precision.

Support:

- MXFP4 expert weights
- MXFP8 dense weights
- UE8M0 scaling metadata
- Required scale layouts
- Quantized tensor loading
- Correct dequantization or native quantized execution
- B200-appropriate execution kernels

Do not silently convert the entire model into an unrelated format and describe that as native checkpoint support.

Record any conversions, their numerical effects, memory implications, and performance costs.

### DSpark

Implement the actual DSpark draft-and-verification architecture.

Do not substitute a generic MTP implementation.

Support the checkpoint's draft stages, expert routing, draft generation, confidence handling, and verification behavior.

Speculative decoding must be independently switchable for correctness testing and benchmarking.

### Vision

Implement the actual vision path, including:

- Image preprocessing
- Vision encoder
- Vision-language alignment
- Image token representation
- Integration into the language model

The first operational milestone may prioritize text inference, but the architecture must explicitly track vision support.

Do not claim full V4.1 multimodal compatibility while silently dropping image inputs.

## 4.3 Tokenizer and prompt protocol

Integrate the exact tokenizer assets and the model-specific prompt encoding protocol.

The V4.1-Flash-1017 repository must be inspected directly for its tokenizer mode and encoding requirements.

Do not assume a conventional Hugging Face chat template exists.

Verify:

- Prompt encoding
- Special tokens
- Conversation serialization
- Role boundaries
- Assistant generation boundaries
- Tool-call serialization
- End-of-turn behavior
- Token decoding
- Streaming text reconstruction

Use fixed test vectors to verify token IDs against the reference implementation.

## 4.4 Model loading

Implement full checkpoint loading.

Requirements:

- Parse the model configuration.
- Parse the Safetensors index.
- Discover every shard.
- Validate all expected tensors.
- Validate shapes and data types.
- Validate quantization metadata.
- Map tensor names to model components.
- Load tensors without corrupting their original representation.
- Support efficient parallel reads.
- Transfer tensors to the correct GPUs.
- Record loaded tensor inventory.
- Detect missing, duplicate, unexpected, and malformed tensors.

The checkpoint contains approximately 96,085 tensors and 510.3 GB of weights according to the current V4.1-Flash recipe.

Model loading must be treated as a distributed memory-placement problem, not a single-file mmap operation.

## 4.5 Model correctness

Before performance optimization, validate actual execution.

Required:

1. Checkpoint integrity.
2. Tensor inventory completeness.
3. Tokenizer equivalence.
4. Embedding correctness.
5. Individual layer correctness.
6. Attention correctness.
7. Engram correctness.
8. MoE routing correctness.
9. Quantization correctness.
10. Logit equivalence.
11. Greedy decoding equivalence.
12. Multi-token generation equivalence.

Compare against an independent supported reference implementation.

Use numerical tolerances appropriate to the quantization and execution precision.

Document differences rather than hiding them.

---

# 5. B200 GPU execution architecture

Target:

**Eight NVIDIA B200 GPUs, SM100.**

The implementation must execute on physical B200 devices.

## 5.1 CUDA backend

Implement and verify:

- CUDA device discovery
- Device properties
- Context management
- Streams
- Events
- Device memory allocation
- Device memory release
- Host-to-device transfer
- Device-to-host transfer
- Device-to-device transfer
- Kernel dispatch
- Synchronization
- Error propagation
- CUDA graph support where applicable

Use a CUDA 13-compatible environment for the target platform.

Use the actual CUDA and GPU execution stack required by the selected kernels.

Do not treat `cudarc` integration or successful linking as proof of GPU computation.

## 5.2 Compute kernels

Implement or integrate the required B200 execution kernels.

Prioritize:

- Quantized matrix multiplication
- Dense GEMM
- Expert GEMM
- RMSNorm and required normalization operations
- RoPE and positional operations
- Sparse attention
- Compressed KV operations
- Engram lookup
- Expert routing
- Sampling
- DSpark verification

Use optimized vendor or teacher kernels where appropriate, while retaining SINGULARITY's independent runtime and execution orchestration.

## 5.3 Multi-GPU execution

Implement actual distributed execution across all eight devices.

The design must define:

- Tensor placement
- Expert placement
- Tensor parallelism
- Expert parallelism
- Data parallelism where appropriate
- Communication topology
- Collective operations
- NCCL integration
- GPU synchronization
- Cross-device memory movement
- Failure propagation

Do not equate eight visible devices with an operational eight-GPU model.

The final execution topology must be selected based on the model's architecture and memory requirements.

Verify the actual placement of backbone weights, Engram memory, expert weights, and KV state.

## 5.4 Memory architecture

The checkpoint's weight footprint is approximately 510.3 GB, with a published minimum VRAM estimate of approximately 614 GB.

Eight B200 GPUs provide sufficient aggregate nominal HBM capacity for the intended target, but aggregate capacity does not automatically imply a valid memory layout.

Implement and measure:

- Weight placement
- Engram placement
- KV cache allocation
- Activation buffers
- Temporary workspace
- Communication buffers
- CUDA graph memory
- Fragmentation
- Memory reclamation

Ensure the system can load and execute the model without exhausting memory during real inference.

---

# 6. Inference runtime and scheduling

SINGULARITY must function as a complete inference runtime rather than a model-forward demonstration.

## 6.1 Request lifecycle

Implement the complete request lifecycle:

```
Request
  |
Validation
  |
Prompt Serialization
  |
Tokenization
  |
Admission
  |
Scheduling
  |
Prefill
  |
KV Cache
  |
Decode
  |
Sampling
  |
Streaming
  |
Completion
```

Every transition must be connected to actual model execution.

## 6.2 Zero-Overhead Scheduler

Retain and advance the ZOS scheduler.

Implement:

- Request admission
- Dynamic batching
- Prefill scheduling
- Decode scheduling
- Continuous batching
- Sequence lifecycle management
- Resource-aware scheduling
- LCFS preemption where appropriate
- Fairness and starvation prevention
- KV allocation coordination
- Cancellation
- Request deadlines
- Backpressure

Scheduler decisions must reflect real GPU and KV resource consumption.

## 6.3 KV cache

Implement a genuinely device-backed paged KV cache.

Support:

- Physical GPU memory backing
- Block allocation
- Block reclamation
- Sequence-to-block mapping
- Prefix reuse
- Radix prefix caching
- KV format selection
- Sparse and compressed KV requirements
- Sliding-window behavior
- Preemption and restoration
- Multi-request isolation

The KV cache must integrate with the actual V4.1 attention path.

Logical block bookkeeping without backing model state is insufficient.

## 6.4 Prefill and decode

Implement and optimize prefill and decode as distinct execution phases.

Measure them independently.

Support:

- Prompt ingestion
- Prefix reuse
- Chunked prefill
- Decode scheduling
- KV reuse
- Request interleaving
- Context management
- Long-context operation

The initial deployment is a single eight-B200 node.

Design interfaces so the runtime can evolve toward separate prefill/decode resources and multi-node execution without requiring a wholesale rewrite.

## 6.5 Sampling and speculative decoding

Implement:

- Greedy sampling
- Temperature
- Top-p
- Top-k where supported
- Repetition controls
- Stop sequences
- Deterministic seeded sampling
- Structured output constraints
- DSpark draft generation
- Verification
- Fallback to ordinary decoding

Verify that sampling operates on actual model logits.

---

# 7. Grammar, tools, and HEAV3NS Code integration

SINGULARITY is intended to be useful as an inference subsystem, not simply a standalone text generator.

Preserve and advance its structured generation capabilities.

Implement:

- JSON Schema constrained generation
- Grammar-constrained generation
- Token-level logit masking
- Tool-call trigger detection
- Structured tool-call serialization
- Tool-call streaming
- Stop-condition handling
- Correct conversation continuation after tool execution

Ensure grammar constraints are compatible with the actual tokenizer and vocabulary.

Provide a stable interface through which HEAV3NS Code can submit inference requests, receive generated output, and process tool calls.

The runtime must not assume that all requests are ordinary chat completions.

Support the request types required by the broader agent and coding workflows while maintaining a clear separation between inference and external tool execution.

---

# 8. Complete Modal infrastructure pipeline

Create a dedicated deployment project within the SINGULARITY workspace.

Suggested structure:

```
HEAV3NS-BRAIN/
|
+-- Cargo.toml
+-- crates/
|   +-- singularity-core/
|   +-- singularity-cuda/
|   +-- singularity-cache/
|   +-- singularity-scheduler/
|   +-- singularity-model/
|   +-- singularity-grammar/
|   +-- singularity-server/
|   +-- singularity-cli/
|   +-- singularity-tests/
|
+-- deploy/
|   +-- modal/
|       +-- app.py
|       +-- image.py
|       +-- Dockerfile
|       +-- config.py
|       +-- model_storage.py
|       +-- deploy.py
|       +-- health.py
|       +-- benchmark.py
|       +-- scripts/
|       +-- README.md
|
+-- benchmarks/
+-- tests/
+-- docs/
+-- RESEARCH_REPORT.md
+-- ARCH_STATE.md
+-- PHASE_III_REPORT.md
```

Adapt this structure to the existing repository rather than creating duplicate systems.

## 8.1 Infrastructure ownership

Modal must orchestrate the deployment.

The actual inference process must be the SINGULARITY Rust runtime.

Python must not become a replacement inference engine.

The Modal project must be version-controlled and reproducible.

## 8.2 Container image

Build a production Linux container containing:

- SINGULARITY release binary
- Required Rust runtime libraries
- CUDA-compatible execution dependencies
- CUDA 13-compatible environment
- NVIDIA communication libraries
- NCCL
- Required system libraries
- HTTP server dependencies
- Runtime configuration
- Health-check tooling
- Telemetry tooling

Use compatible compiler and build-tool versions, including GCC 13 or newer where required by the CUDA and reference-kernel build.

Build the Rust application for the actual Linux target environment.

Do not assume a macOS binary can run inside the Modal container.

Pin dependency versions and record the final image configuration.

The image must be reproducibly rebuildable from the repository.

## 8.3 Container verification

Before model provisioning, verify:

- Container starts.
- SINGULARITY binary executes.
- CUDA runtime initializes.
- Eight B200 devices are visible.
- Device allocation works.
- A real GPU kernel executes.
- NCCL initializes.
- Inter-GPU communication succeeds.
- Runtime configuration loads.
- Health instrumentation works.

A container build passing is only the container milestone, not model execution.

## 8.4 Modal application

Create the Modal application and define its resource requirements.

Target configuration:

- GPU: `B200`
- GPU count: `8`
- One colocated eight-GPU container
- Appropriate CPU resources
- Sufficient host memory
- Adequate ephemeral disk
- Long model initialization timeout
- Persistent checkpoint storage
- Controlled concurrency
- Explicit scaling configuration
- Authenticated API access

Use Modal's current deployment APIs and supported eight-GPU B200 configuration.

Do not assume that a multi-container Modal cluster is necessary for the initial single-node eight-GPU deployment.

If a single eight-GPU allocation cannot be provisioned, report the actual infrastructure limitation rather than silently changing hardware.

## 8.5 Model storage

Create a persistent Modal Volume for the model checkpoint and required runtime artifacts.

Implement:

- Checkpoint download
- Revision pinning
- Shard inventory
- Integrity validation
- Download resumption
- Persistent caching
- Parallel reads
- Storage health checks

Do not download 510 GB of model data every time a container restarts.

Keep model storage independent of ephemeral container filesystems.

Store Hugging Face credentials and other secrets through Modal Secrets.

Never embed credentials in code, image layers, logs, or artifacts.

## 8.6 Deployment lifecycle

Create commands or scripts for:

- Build image
- Validate image
- Provision model storage
- Download checkpoint
- Validate checkpoint
- Deploy application
- Inspect deployment
- Inspect logs
- Check health
- Run inference tests
- Run benchmarks
- Restart service
- Update deployment
- Stop service

The deployment must be independently reproducible.

---

# 9. Persistent inference service

Deploy SINGULARITY as a persistent Modal application.

The service must load the model and remain available after successful initialization.

Use a warm replica configuration appropriate to the intended operating state.

The model must not be reloaded for every request.

## 9.1 Startup sequence

The service must perform:

1. Container initialization.
2. Runtime configuration loading.
3. CUDA initialization.
4. B200 discovery.
5. Eight-GPU validation.
6. NCCL initialization.
7. Distributed topology initialization.
8. Model storage validation.
9. Checkpoint tensor inventory validation.
10. Model weight loading.
11. Engram initialization.
12. KV cache initialization.
13. Tokenizer initialization.
14. Model-specific execution initialization.
15. Warm-up inference.
16. Readiness validation.
17. API availability.

The service must not become ready before all required initialization has succeeded.

## 9.2 API

Expose:

- `GET /health`
- `GET /ready`
- `GET /v1/models`
- `POST /v1/chat/completions`
- Streaming responses using SSE
- Metrics endpoint

Maintain OpenAI-compatible request and response behavior where practical.

Support:

- Streaming
- Non-streaming
- Sampling parameters
- Context limits
- Stop sequences
- Tool calls
- Structured outputs
- Request cancellation
- Error reporting

The API must expose the actual SINGULARITY model and execution state.

## 9.3 Health and observability

Health must distinguish:

- Container alive
- Runtime initialized
- CUDA operational
- GPUs available
- Distributed topology operational
- Checkpoint present
- Model loaded
- Model ready
- Inference successfully executed

Record:

- Request counts
- Token counts
- TTFT
- Inter-token latency
- Prefill latency
- Decode latency
- GPU memory
- KV utilization
- GPU utilization
- Power
- Errors
- Queue depth
- Scheduling behavior

Expose logs and metrics through the deployment environment.

## 9.4 Persistent availability

After successful validation, retain a deployed service rather than terminating the entire application after a benchmark.

Keep the model warm as intended.

The service must support restart and recovery without requiring the full checkpoint to be downloaded again.

The deployment must be controllable and independently operable.

---

# 10. End-to-end functional validation

All tests must use actual model weights and actual GPU execution.

## 10.1 Basic validation

Test:

- Health
- Readiness
- Model discovery
- Request parsing
- Tokenization
- Prompt encoding
- Model forward execution
- Logit generation
- Greedy decoding
- Token decoding
- Non-streaming response
- Streaming response

## 10.2 Model validation

Test:

- Short prompts
- Long prompts
- Multi-turn conversations
- Different sampling configurations
- Stop sequences
- Structured output
- JSON Schema
- Tool calls
- Context boundaries
- Long-context retrieval
- Repeated deterministic requests

Test vision inputs when the vision implementation is ready.

## 10.3 Concurrency

Test:

- Single request
- Multiple concurrent requests
- Continuous batching
- Different prompt lengths
- Different output lengths
- Mixed prefill/decode workloads
- Request cancellation
- Request timeouts
- Backpressure
- KV exhaustion and recovery

## 10.4 Sustained execution

Run a bounded sustained inference workload.

Verify:

- No synthetic execution.
- No CPU fallback.
- No silent model substitution.
- No memory corruption.
- No unexplained GPU loss.
- No scheduler deadlocks.
- No request cross-contamination.
- No unbounded memory growth.
- No persistent streaming failures.

Preserve the raw execution evidence.

---

# 11. DeepInfer reference deployment

Deploy DeepInfer separately as the reference implementation.

Repository:

`https://github.com/deepseek-ai/DeepInfer`

Use the same exact V4.1-Flash-1017 checkpoint revision.

DeepInfer must run in its own isolated container and Modal application.

Do not import DeepInfer into SINGULARITY's runtime.

Do not use DeepInfer to answer SINGULARITY inference requests.

The two systems must be independently executable.

## 11.1 Reference environment

Use the supported DeepInfer environment:

- Linux
- GCC 13+
- Python 3.12
- CUDA 13
- NVIDIA SM100
- B200 hardware

Build the reference implementation from its source repository and record the exact Git revision and dependency versions.

Verify the reference can load the same checkpoint and produce valid output before using it in benchmark comparisons.

## 11.2 Understand the reference boundary

DeepInfer's public release is an offline inference engine.

Its release documentation identifies several components not included in the complete online serving system, including KV cache infrastructure, prefill/decode disaggregation, and load balancing across data-parallel replicas.

Do not confuse the reference engine's offline capabilities with the complete online service being built for SINGULARITY.

For the engine benchmark, use DeepInfer's native inference entrypoints.

For the online service comparison, build an equivalent thin benchmark adapter around the reference engine where necessary.

Clearly distinguish engine performance from serving infrastructure performance.

---

# 12. Controlled performance benchmarking

The objective is to establish how SINGULARITY performs against DeepInfer under equivalent conditions.

Both systems must run on the same B200 hardware.

The initial comparison target is:

**8 × B200, one node, same checkpoint revision.**

Do not compare an eight-B200 SINGULARITY deployment against a different GPU configuration and call the result a head-to-head benchmark.

## 12.1 Functional gate

Performance testing cannot substitute for correctness.

Before benchmarking, both systems must:

- Load the same checkpoint.
- Use the same tokenizer and prompt protocol.
- Accept equivalent inputs.
- Produce valid model outputs.
- Respect the same context and output constraints.

Where deterministic equivalence is possible, compare token IDs and logits.

Where implementation-specific numerical differences are expected, report them and establish appropriate tolerances.

## 12.2 Benchmark categories

### A. Base inference comparison

Disable DSpark speculative decoding in DeepInfer.

Disable equivalent speculative acceleration in SINGULARITY.

Compare the underlying model execution path.

### B. Optimized inference comparison

Enable supported DeepInfer optimizations.

Enable corresponding SINGULARITY optimizations.

Document differences in the optimization strategies.

### C. Serving comparison

Compare end-to-end API performance under equivalent request loads.

Include tokenization, request admission, scheduling, prefill, decode, streaming, and response delivery.

### D. Scaling comparison

Measure performance as request concurrency increases.

Measure the impact of batching, KV utilization, and different context lengths.

## 12.3 Workload matrix

Run workloads across:

- Short context
- 4K context
- 32K context
- 128K context
- Longer contexts where supported

Vary:

- Input token count
- Output token count
- Batch size
- Request concurrency
- Sampling configuration
- Prefix reuse
- Speculative decoding

Keep prompt and output lengths controlled across both engines.

## 12.4 Metrics

Record:

| Metric             | Purpose                               |
| ------------------ | ------------------------------------- |
| Prefill tokens/sec | Prompt-processing throughput          |
| Decode tokens/sec  | Generation throughput                 |
| TTFT               | Time to first token                   |
| ITL                | Inter-token latency                   |
| TPOT               | Time per output token                 |
| Requests/sec       | Serving throughput                    |
| P50 latency        | Typical request behavior              |
| P95 latency        | High-percentile latency               |
| P99 latency        | Tail latency                          |
| Peak HBM           | Maximum GPU memory use                |
| Steady-state HBM   | Persistent model and cache footprint  |
| GPU utilization    | Accelerator activity                  |
| GPU power          | Energy and utilization context        |
| KV utilization     | Cache efficiency                      |
| Inter-GPU traffic  | Communication overhead                |
| Scaling efficiency | Benefit from concurrency and batching |

Record raw measurements, test parameters, environment versions, and benchmark commands.

## 12.5 Official benchmark reproduction

Reproduce DeepInfer's published benchmark workload shapes where hardware permits.

Its published charts use four SM100 GPUs for prefill and 64 SM100 GPUs for decode.

The eight-B200 head-to-head comparison is a separate benchmark configuration.

Do not claim to have reproduced the 64-GPU official decode configuration using eight GPUs.

Present official workload reproduction and matched eight-GPU comparisons separately.

---

# 13. Performance optimization

Once correctness and deployment are established, optimize SINGULARITY.

Priorities:

1. GPU kernel efficiency
2. Weight placement
3. Quantized GEMM
4. Expert execution
5. Expert placement and communication
6. Sparse attention
7. Compressed KV operations
8. Engram access
9. Prefill throughput
10. Decode latency
11. Scheduler overhead
12. Continuous batching
13. KV allocation and reclamation
14. Prefix reuse
15. CUDA graph execution
16. Host/device synchronization
17. Network and collective communication

Use measured bottlenecks to determine optimization order.

Do not optimize an unmeasured subsystem based on assumptions.

Do not sacrifice correctness for benchmark numbers.

Keep a record of each optimization, its baseline, its measured impact, and any numerical or operational tradeoffs.

---

# 14. Reliability, security, and operating controls

The deployed system must be operable independently.

Implement:

- Authenticated API access
- Secret management
- Request limits
- Request cancellation
- Timeout handling
- GPU error handling
- Graceful shutdown
- Restart behavior
- Model reload recovery
- Logging
- Metrics
- Health checks
- Failure reporting
- Resource monitoring

The API must not expose model storage credentials, internal secrets, or administrative controls.

## Cost and provisioning discipline

B200 provisioning and sustained operation incur real infrastructure costs.

Before the first long-running deployment:

- Inspect the active Modal workspace.
- Verify deployment credentials.
- Determine actual GPU availability.
- Estimate provisioning and sustained runtime cost.
- Configure available spend controls.
- Run bounded validation workloads first.

Once the system is verified, maintain the intended live deployment.

Do not leave failed, duplicated, or abandoned GPU containers running.

Do not silently change the hardware target if B200 capacity is unavailable.

Report the actual blocker and preserve the work completed.

---

# 15. Execution order and milestone gates

The work must progress through verifiable gates.

A later milestone cannot be declared complete by assuming an earlier one.

## Gate 1 — Repository reality

- Full workspace inspected.
- Teacher repositories studied.
- Existing implementation verified.
- Architecture gaps documented.
- Baseline established.

## Gate 2 — Model architecture

- Exact checkpoint analyzed.
- Architecture mapped.
- Tensor manifest understood.
- Quantization understood.
- Required operations implemented.
- Reference correctness tests established.

## Gate 3 — Real GPU execution

- B200 backend operational.
- CUDA kernels executing.
- Multi-GPU topology operational.
- Model weights placed on GPUs.
- Actual forward execution verified.

## Gate 4 — End-to-end inference

- Tokenizer verified.
- Prompt processing works.
- Model produces real logits.
- Sampling works.
- KV cache works.
- Scheduler executes actual model inference.
- Real output matches reference within established tolerance.

## Gate 5 — Modal infrastructure

- Container built.
- Container verified.
- Persistent storage configured.
- Model downloaded and validated.
- Eight B200s provisioned.
- Distributed execution initialized.

## Gate 6 — Live deployment

- Modal application deployed.
- Checkpoint loaded.
- Warm-up completed.
- Readiness passes.
- Remote API responds.
- Streaming works.
- Sustained inference succeeds.

## Gate 7 — DeepInfer comparison

- Reference environment built.
- Reference checkpoint loaded.
- Functional comparison passes.
- Matched benchmarks completed.
- Raw results preserved.

## Gate 8 — Operational completion

- Persistent service remains deployed.
- Restart and recovery validated.
- Monitoring operational.
- Documentation complete.
- Final results independently reproducible.

---

# 16. Required deliverables

The completed assignment must produce actual artifacts.

## Source and architecture

- Updated Rust workspace
- Model architecture implementation
- CUDA execution backend
- Multi-GPU execution
- KV cache implementation
- Scheduler integration
- Tokenizer integration
- Grammar and tool-call integration
- Architecture documentation
- Teacher research report

## Infrastructure

- Modal application
- Container image definition
- Reproducible image build
- Deployment configuration
- Persistent model storage
- Model download and verification pipeline
- Deployment scripts
- Health checks
- Operational controls

## Deployed system

- Live SINGULARITY Modal application
- Eight-B200 execution
- Exact V4.1-Flash-1017 checkpoint loaded
- Remote inference endpoint
- Working streaming responses
- Sustained execution evidence

## Benchmarking

- DeepInfer reference deployment
- Matched hardware configuration
- Functional correctness report
- Prefill benchmark
- Decode benchmark
- Serving benchmark
- Concurrency benchmark
- Memory and GPU telemetry
- Raw benchmark outputs
- Reproduction scripts

## Documentation

- `RESEARCH_REPORT.md`
- `ARCH_STATE.md`
- `PHASE_III_REPORT.md`
- Deployment README
- Operational runbook
- Benchmark methodology
- Known limitations

---

# 17. Reporting and verification requirements

Maintain an accurate implementation state throughout the commission.

Every reported feature must identify whether it is:

- Specified
- Implemented
- Compiled
- Unit tested
- Integration tested
- Executed on physical B200 hardware
- Verified against a reference
- Deployed
- Running
- Benchmarked

Never collapse these into a single completion status.

For every important milestone, preserve:

- Git revision
- Commands executed
- Environment configuration
- Test output
- Logs
- GPU telemetry
- Model revision
- Relevant output samples
- Known limitations

If an operation fails, report the actual failure and the next technical blocker.

Do not fabricate results, skip failing tests, replace model output with synthetic data, or report CPU execution as GPU execution.

---

# 18. Final definition of done

SINGULARITY Phase III is complete when:

1. The independent Rust inference engine is implemented and operational.
2. The exact DeepSeek V4.1-Flash-1017 architecture is supported to the declared completeness level.
3. The exact checkpoint loads successfully.
4. The native model weights and quantization are correctly handled.
5. Actual inference executes across eight B200 GPUs.
6. Generated outputs pass reference correctness validation.
7. The scheduler, KV cache, tokenizer, and sampling are connected to real execution.
8. The system exposes a functional inference API.
9. The Modal container is reproducibly built and deployed.
10. The Modal application remains available as a live service.
11. Real requests are successfully processed through the remote endpoint.
12. Streaming and concurrent inference work.
13. Sustained execution succeeds.
14. DeepInfer is independently deployed and tested.
15. A controlled, reproducible performance comparison is complete.
16. The implementation and operational documentation are delivered.

## The final objective

**SINGULARITY must exist as an independently implemented inference engine inside HEAV3NS Code, capable of loading and running DeepSeek V4.1-Flash-1017 on eight NVIDIA B200 GPUs, deployed as a persistent Modal service, accessible through its own inference API, and validated against DeepInfer through actual execution and reproducible benchmarks.**

The deliverable is the running system.

Not the architecture diagram.

Not the Cargo workspace.

Not the test count.

Not the container image.

Not the deployment report.

The running system.

**Build it. Deploy it. Load the model. Run it. Keep it running. Measure it.**

## Why this version is materially different

Three things are now explicit that were previously under-specified:

1. SINGULARITY is the product of the implementation, not merely the model runner. Its scheduler, cache, grammar, tool calls, API, telemetry, and HEAV3NS Code integration all remain part of the assignment.

2. DeepInfer is both a teacher and a competitor, not a dependency. We study its algorithms, deploy it independently, and compare actual execution. Its published release has its own online-serving gaps, which we must not accidentally inherit.&#x20;

   [image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

   GitHub+1

3. The deployment is part of the engineering deliverable. Modal's documented eight-B200 configuration is a concrete route to the target, with persistent storage and a warm service.&#x20;

   [image](https://www.google.com/s2/favicons?domain=https://modal.com\&sz=32)

   Modal Docs+2



And one subtle but important point: the eight-B200 configuration is the initial single-node target. It is not a claim that the whole system has already solved arbitrary multi-node inference, nor that eight GPUs automatically make every model component fit and execute. The implementation has to demonstrate that.

That's the complete objective as I understand it: make SINGULARITY a real HEAV3NS inference system, then let it prove itself in production-like conditions.