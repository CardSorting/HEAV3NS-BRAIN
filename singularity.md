# EXECUTIVE COMMISSION
## NOVA v4 — HEAV3NS-BRAIN
### Project SINGULARITY
**Commissioning authority:** MAJO, Executive Chair  
**Execution authority:** Deputy Gemini
**Implementation language:** Rust  
**Target hardware:** NVIDIA B200 / H200  
**Destination:** `/Users/bozoegg/Desktop/HEAV3NS-BRAIN`

---

## 1. THE MANDATE

You are hereby commissioned to undertake a deep research, architectural investigation, and independent engineering implementation of a next-generation inference system.

The objective is to produce **SINGULARITY**, a new Rust-based inference system developed through intensive examination of two existing reference implementations.

### Teacher repositories

1. `/Users/bozoegg/Downloads/DeepInfer-main`
2. `/Users/bozoegg/Downloads/NOVAV2-main`
3. `/Users/bozoegg/Desktop/MOSS-Studio`

These repositories are your architectural teachers.

Study their implementations in depth. Understand their design decisions, inference execution strategies, scheduling, memory management, concurrency, model execution, communication pathways, and operational characteristics.

Do not simply reproduce their interfaces or wrap their existing implementations.

Understand how they work, identify what makes them effective, discover where they are limited, and use those findings to develop a substantially more capable implementation.

### Student project

`/Users/bozoegg/Desktop/HEAV3NS-BRAIN`

This is the authoritative destination for the new implementation.

Rust is the primary implementation language.

The resulting system must be independently structured, maintainable, extensible, and engineered for serious GPU inference workloads.

**The objective is not to make another demonstration. The objective is to build an actual inference system.**

---

## 2. RESEARCH DIRECTIVE: DEEP INVESTIGATION

Do not begin by immediately writing code.

First, deeply investigate both teacher repositories.

Perform multiple independent research and audit passes.

Investigate:

- Overall architecture and component boundaries.
- Inference execution lifecycle.
- Model loading and initialization.
- GPU memory allocation and management.
- Tensor operations and execution scheduling.
- Attention implementations and KV-cache strategies.
- Continuous batching and request scheduling.
- Tensor, pipeline, and expert parallelism.
- Multi-GPU communication.
- Asynchronous execution and concurrency.
- Token generation and streaming.
- Prefill/decode separation.
- Tool calling and structured output.
- API compatibility.
- Error handling and recovery.
- Observability and performance instrumentation.
- Hardware-specific optimizations.
- Failure modes under sustained workloads.

Examine source code, not just documentation.

Trace important execution paths end to end.

Identify the actual implementation behind every major architectural claim.

### Comparative research

Produce a comparative architectural assessment of DeepInfer and NOVAV2.

For every significant subsystem:

1. Explain how each teacher implements it.
2. Identify the underlying engineering strategy.
3. Identify strengths and weaknesses.
4. Identify scalability constraints.
5. Investigate alternative industry approaches.
6. Determine which ideas should be adopted, redesigned, combined, or rejected.
7. Document the reasoning behind the student implementation.

Do not assume that the newest, most complicated, or most heavily optimized implementation is automatically the best.

Prioritize measurable engineering quality.

---

## 3. BUILD SINGULARITY

Develop an independent Rust inference architecture.

The system should be designed around a clean separation of responsibilities, with well-defined subsystem boundaries.

Investigate and implement the following capabilities where appropriate:

- Model execution runtime.
- Model loading and weight management.
- GPU backend abstraction.
- Memory and KV-cache management.
- Request scheduling.
- Continuous batching.
- Asynchronous execution.
- Token generation.
- Streaming response delivery.
- Tool-call preservation.
- Structured output handling.
- OpenAI-compatible API surface.
- Multi-GPU execution.
- Telemetry and observability.
- Configuration and deployment management.

Investigate suitable Rust GPU integration strategies, CUDA interoperability, NCCL integration, and existing ecosystem libraries before committing to implementation choices.

Do not unnecessarily reinvent mature infrastructure.

However, do not introduce dependencies simply because they are popular. Evaluate their suitability, maturity, performance, maintenance, and architectural implications.

### Student architecture principle

SINGULARITY must be an independent implementation informed by the teachers, not a superficial fork or a collection of disconnected adapters.

Every major architectural decision must have a defensible reason.

---

## 4. HARDWARE OBJECTIVE: B200 / H200

Treat NVIDIA B200 and H200 as first-class deployment targets.

Investigate the differences between the architectures and develop an implementation strategy that can exploit their respective capabilities.

Prioritize:

- GPU utilization.
- Memory bandwidth.
- Memory capacity.
- Low-precision execution.
- Efficient attention.
- KV-cache efficiency.
- Multi-GPU communication.
- Scheduling efficiency.
- Request concurrency.
- Sustained throughput.
- Latency under mixed workloads.

Investigate FP8, FP4, BF16, and other appropriate execution formats.

Do not assume all formats or kernels are interchangeable between GPU generations.

Develop explicit hardware capability detection and graceful fallback behavior.

The system must not silently select unsupported execution paths.

### Performance philosophy

Optimize the complete inference lifecycle rather than maximizing one isolated benchmark.

Measure:

- Time to first token.
- Inter-token latency.
- Prefill throughput.
- Decode throughput.
- Aggregate tokens per second.
- Concurrent request capacity.
- GPU utilization.
- Memory utilization.
- Recovery after workload spikes.

Establish reproducible benchmarks before making performance claims.

---

## 5. ARCHITECTURAL ERGONOMICS AND WORKSPACE QUALITY

Treat the entire HEAV3NS-BRAIN workspace as a first-class engineering product.

Investigate and improve the ergonomics of the development environment throughout implementation.

Establish familiar, predictable navigation patterns inspired by successful developer tools and established engineering projects.

Improve:

- Project organization.
- Module boundaries.
- Documentation navigation.
- Configuration discovery.
- Build and execution workflows.
- Test organization.
- Benchmark accessibility.
- Diagnostic output.
- Error reporting.
- Developer onboarding.
- Operational visibility.

Make the workspace understandable without requiring someone to reconstruct its architecture from source code.

Implement useful documentation and examples alongside the actual implementation.

Avoid unnecessary complexity and duplicated functionality.

Keep the project clean, navigable, and maintainable as it grows.

---

## 6. PARENT AGENT AND SKILL EXECUTION IMPROVEMENTS

Investigate the existing parent-agent and skill-execution workflows.

Improve throughput without creating unnecessary coordination bottlenecks.

### Parent-agent flow

- Reduce unnecessary sequential execution.
- Identify independent tasks suitable for parallel execution.
- Improve task decomposition.
- Minimize blocking dependencies.
- Improve delegation and result aggregation.
- Prevent duplicated investigation.
- Establish clear ownership of implementation areas.
- Improve failure recovery.
- Preserve shared architectural understanding.
- Keep the parent agent focused on coordination and verification rather than becoming a bottleneck.

### Skill utilization

- Audit available skills before implementation.
- Identify skills that can materially improve research or execution.
- Improve skill discovery and invocation.
- Reduce unnecessary skill-loading overhead.
- Improve context preservation between delegated tasks.
- Avoid redundant work across agents.
- Ensure outputs are reviewed and integrated rather than blindly accepted.

### Execution authority

Use available I/O and execution capabilities effectively.

Favor direct, deterministic execution when safe.

Do not introduce unnecessary approval gates for routine, reversible development operations.

Maintain appropriate safeguards for destructive operations, credential handling, deployment, and irreversible changes.

The objective is a faster, more balanced, less obstructive engineering workflow.

---

## 7. HARDENING AND FAILURE REMEDIATION

This project must undergo repeated hardening passes.

Do not stop after the first implementation or successful build.

Investigate and remediate:

- Resource exhaustion.
- Memory leaks.
- GPU allocation failures.
- Deadlocks.
- Race conditions.
- Scheduling starvation.
- Request cancellation failures.
- Backpressure failures.
- Partial initialization.
- Model-loading failures.
- Interrupted streams.
- Invalid tool-call responses.
- Serialization inconsistencies.
- Network interruptions.
- Multi-GPU synchronization problems.
- Unexpected hardware capabilities.
- Configuration errors.
- Process crashes.
- Recovery and restart behavior.

For every discovered weakness:

1. Identify the root cause.
2. Determine the correct architectural remedy.
3. Implement the actual fix.
4. Add regression coverage.
5. Verify that the fix does not introduce new failures.

Do not settle for monkey patches, superficial workarounds, or swallowed exceptions.

Prefer systemic fixes.

Design graceful degradation where full recovery is impossible.

---

## 8. MULTIPLE INDEPENDENT AUDIT PASSES

This is an explicit requirement, not optional polish.

Conduct multiple investigation and implementation reviews.

**Pass 1 — Discovery:** Understand the teachers and establish the architectural baseline.

**Pass 2 — Architecture:** Develop the independent student architecture and validate its feasibility.

**Pass 3 — Implementation:** Build the actual Rust subsystems and integrate them.

**Pass 4 — Performance:** Profile execution, identify bottlenecks, and optimize.

**Pass 5 — Reliability:** Exercise failure modes, concurrency, resource limits, and recovery.

**Pass 6 — Ergonomics:** Audit the entire workspace, developer experience, navigation, documentation, and operational workflows.

**Pass 7 — Independent review:** Re-examine earlier decisions and challenge assumptions. Look for problems that the original implementation process overlooked.

**Pass 8 — Final verification:** Run regression tests, integration tests, benchmarks, and deployment checks.

Each pass must produce actionable findings and corresponding implementation improvements.

Do not manufacture work merely to satisfy a pass. When a subsystem is already correct, document the evidence.

---

## 9. ENGINEERING STANDARDS

Maintain the following standards throughout the project:

- Strong Rust typing and explicit ownership boundaries.
- Clear error types and meaningful diagnostics.
- Modular, testable architecture.
- Reproducible builds.
- Deterministic configuration.
- Automated regression tests.
- Integration testing.
- Performance benchmarking.
- Documented hardware requirements.
- Versioned interfaces.
- Dependency auditing.
- Security-conscious defaults.
- Explicit capability detection.
- No undocumented critical behavior.

Do not claim support for hardware, models, precision formats, or execution features that have not actually been verified.

Distinguish between implemented, experimental, and planned functionality.

---

## 10. EXECUTION EXPECTATIONS

You have broad implementation discretion within this mandate.

Do not wait for the commissioning authority to make routine architectural decisions.

Investigate, decide, implement, test, audit, and improve.

When encountering uncertainty:

- Research the underlying issue.
- Compare available approaches.
- Make a defensible engineering decision.
- Document significant trade-offs.
- Continue execution.

When encountering blockers:

- Determine their actual cause.
- Investigate alternatives.
- Resolve the blocker at its source.
- Avoid allowing one blocked subsystem to halt independent work.

Use parallel investigation where appropriate, but maintain architectural coherence and integration discipline.

Preserve the teacher repositories. Do not modify or destructively reorganize them.

Keep all new project artifacts within the designated student workspace unless a dependency or build system requires otherwise.

---

## 11. DEFINITION OF SUCCESS

SINGULARITY is not complete merely because it compiles.

The objective is an independently implemented Rust inference system that:

1. Has a documented and defensible architecture.
2. Has learned from both teacher implementations.
3. Can load and execute supported models.
4. Supports GPU execution on its verified target hardware.
5. Provides a robust request and response lifecycle.
6. Preserves structured tool calls and agent interactions.
7. Handles concurrent inference workloads.
8. Has measurable performance characteristics.
9. Has been tested against realistic failure conditions.
10. Has an approachable, maintainable engineering workspace.
11. Has repeatable build, deployment, and benchmarking procedures.

If full functionality cannot be completed, deliver the strongest verified implementation possible, with explicit limitations and a clear continuation plan.

Never substitute a demonstration or mock implementation for a functioning subsystem without clearly identifying it.

---

## FINAL EXECUTIVE DIRECTIVE

**Deeply investigate this. Double down on promising concepts. Audit again. Dig deeper. Challenge the architecture. Improve the implementation.**

Study both teachers with genuine technical rigor.

Build something better where evidence supports doing so.

Prioritize engineering substance over appearances, systemic solutions over patches, and measurable improvements over unsupported claims.

Maintain high execution throughput without sacrificing correctness.

Make the workspace resilient, understandable, and pleasant to operate.

Do not mistake a successful prototype for a completed system.

Do not stop at the first working implementation.

**The mission is SINGULARITY.**

Build it properly.