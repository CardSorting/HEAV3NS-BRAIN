# COMMISSION: ZERO-IDLE INFERENCE INFRASTRUCTURE, AUTOSCALING & WORKSPACE HARDENING

## Executive Objective

Conduct a comprehensive engineering investigation, implementation, and hardening initiative across the entire existing workspace.

The primary objective is to transform the current distributed inference implementation into a robust, cost-efficient, production-grade system that automatically scales down to ZERO when there is no demand.

The system currently maintains always-on and warm execution states that continue generating infrastructure costs even when no requests are being processed.

This is unacceptable.

**ZERO DEMAND MUST EVENTUALLY MEAN ZERO PROVISIONED COMPUTE.**

Do not sacrifice the fundamental cost objective simply to optimize cold-start latency.

Warm capacity is an optional performance optimization that must always remain subordinate to cost control.

This is an implementation commission, not a proposal or an audit-only exercise. Investigate, implement, test, verify, document, and continue improving the actual system.

---

# 1. PRIMARY MANDATE: TRUE SCALE-TO-ZERO

Deeply investigate the complete inference lifecycle and implement genuine scale-to-zero behavior.

### Required behavior

1. No incoming requests and no active workloads must result in complete compute scale-down.
2. Warm workers must not remain provisioned indefinitely.
3. Always-on infrastructure must be eliminated wherever it is not strictly necessary.
4. Idle workers must automatically terminate after an appropriately configurable idle period.
5. All GPU allocations must be released when workloads finish.
6. No background process, heartbeat, scheduler, health check, or orchestration loop should inadvertently maintain expensive compute resources.
7. Queued requests must be capable of waking the system from zero.
8. The system must correctly distinguish active workloads, queued workloads, warm capacity, and genuinely idle infrastructure.
9. Scale-down must account for in-flight requests, active generations, prefill operations, KV-cache dependencies, and distributed worker coordination.
10. Shutdown must be graceful, deterministic, and resistant to race conditions.

### Cost-control requirements

Implement:

- Configurable idle shutdown thresholds.
- Maximum worker lifetime where appropriate.
- Maximum provisioned GPU capacity.
- Maximum concurrent jobs.
- Per-job execution timeouts.
- Request-level resource accounting.
- Hard spending and resource consumption limits wherever supported.
- Automatic cleanup of abandoned workers.
- Detection and termination of orphaned allocations.
- Protection against infinite retry loops.
- Protection against runaway provisioning.
- Explicit lifecycle state tracking.
- Cost and utilization observability.

Do not mistake an empty request queue for an idle system if workloads are still executing.

Likewise, do not mistake a warm worker for productive utilization.

### Critical acceptance criterion

When the system has no requests, no queued work, and no in-flight operations, it must eventually release all billable inference compute.

Verify this through actual provider-side resource state and billing semantics, not merely an application-level status indicator.

If any resource cannot scale to zero, identify precisely why, quantify its cost, and investigate alternatives.

---

# 2. INVESTIGATE INDUSTRY-STANDARD ARCHITECTURES

Deeply investigate established autoscaling, distributed inference, and production serving strategies.

Do not reinvent established solutions unnecessarily.

Examine applicable patterns from:

- Scale-to-zero serverless architectures.
- Queue-driven worker provisioning.
- Demand-based autoscaling.
- Distributed GPU scheduling.
- Graceful worker draining.
- Dynamic batching.
- Request-aware scheduling.
- Adaptive concurrency control.
- Cold-start mitigation.
- Resource-aware placement.
- Fault-tolerant distributed execution.

Investigate the actual capabilities and limitations of the infrastructure providers and orchestration systems currently used in this workspace.

Identify which strategies are appropriate for this particular workload.

Do not blindly copy architectures designed for conventional web services when they conflict with large-model distributed inference requirements.

### Important architectural consideration

The inference system requires coordinated GPU resources and a large model footprint.

Scale-to-zero must not accidentally introduce:

- Partial cluster provisioning.
- Incomplete model initialization.
- Inconsistent distributed worker states.
- KV-cache corruption.
- Duplicate request execution.
- Orphaned GPU allocations.
- Deadlocks during worker shutdown.
- Requests stranded during cold starts.

Preserve the existing working inference capabilities while improving the infrastructure surrounding them.

---

# 3. COLD-START AND WARM-STATE ERGONOMICS

Investigate the complete tradeoff between cold-start latency and idle compute expenditure.

The objective is not to eliminate cold starts at any cost.

The objective is to provide excellent execution ergonomics without permanently paying for unused infrastructure.

Implement an intelligent lifecycle strategy:

- Scale from zero when demand arrives.
- Provision the required GPU resources.
- Initialize and validate the distributed inference workers.
- Accept work only when the cluster is ready.
- Maintain warm capacity while justified by actual demand.
- Drain workers safely when workloads finish.
- Automatically terminate idle infrastructure.
- Return to zero.

Warm-state retention must be configurable and must have a finite upper bound.

The default operating policy should prioritize cost efficiency over speculative future demand.

Do not introduce hidden minimum replica counts or permanent warm pools.

---

# 4. DISTRIBUTED INFERENCE RELIABILITY

Conduct a deeper investigation into the existing multi-GPU inference architecture.

Pay particular attention to:

- GPU allocation and scheduling.
- Distributed worker initialization.
- Model loading and placement.
- Prefill execution.
- KV-cache lifecycle and coordination.
- Decode execution.
- Inter-worker communication.
- Resource synchronization.
- Worker health and failure detection.
- Cleanup and termination behavior.

Preserve the functional inference pipeline.

Do not replace working components merely to make the architecture look cleaner.

Identify actual weaknesses and remediate their underlying causes.

Where possible, implement:

- Explicit lifecycle state machines.
- Idempotent provisioning and teardown.
- Distributed health checks.
- Deadlock detection.
- Bounded retries with backoff.
- Failure isolation.
- Checkpointing where applicable.
- Recovery from partial initialization.
- Deterministic cleanup.
- Structured execution tracing.

No infinite loops, unbounded retries, or uncontrolled resource acquisition.

---

# 5. WORKSPACE-WIDE HARDENING AND QUALITY OF LIFE

Continue the investigation across the ENTIRE workspace, not just the immediate inference entry point.

Identify opportunities to improve architecture, developer experience, navigation, maintainability, and operational reliability.

Conduct comprehensive passes to:

- Identify duplicated functionality.
- Identify obsolete implementations.
- Identify abandoned experiments.
- Identify conflicting configurations.
- Consolidate unnecessary abstractions.
- Improve module boundaries.
- Improve configuration management.
- Improve error handling.
- Improve logging and observability.
- Improve test coverage.
- Improve documentation.
- Improve onboarding and navigation.
- Remove unnecessary complexity.
- Establish consistent naming and organizational conventions.

Curate and clean the workspace using established strategies for robust, resilient agent-operated development environments.

### Developer experience

Improve the entire workspace so that future agents and developers can understand and operate it without relying on undocumented knowledge held by its original creator.

Implement familiar, approachable patterns inspired by mature engineering tools and popular developer platforms.

Prioritize:

- Clear navigation.
- Predictable directory organization.
- Consistent configuration.
- Discoverable commands.
- Clear operational status.
- Actionable errors.
- Reproducible workflows.
- Useful documentation.
- Explicit ownership of system components.
- Straightforward debugging.

Do not introduce unnecessary dependencies or elaborate abstractions.

---

# 6. MULTIPLE AUDIT AND IMPLEMENTATION PASSES

This commission explicitly requires repeated investigation.

Do not perform one superficial review and declare completion.

### Pass 1 — Discovery

Map the entire workspace, trace execution paths, identify dependencies, understand infrastructure costs, and establish the current operational baseline.

### Pass 2 — Architectural investigation

Identify structural weaknesses, expensive design decisions, failure modes, and opportunities to adopt mature industry patterns.

### Pass 3 — Implementation

Implement the actual scale-to-zero lifecycle, cost controls, reliability improvements, and workspace-wide quality-of-life improvements.

### Pass 4 — Failure-mode audit

Attempt to identify how the new implementation could fail.

Investigate:

- Idle resource leakage.
- Failed shutdown.
- Partial provisioning.
- Distributed worker failure.
- Lost requests.
- Duplicate execution.
- Infinite retries.
- Stuck jobs.
- Unexpected background activity.
- Unbounded costs.
- Resource contention.
- Concurrent lifecycle transitions.

Patch underlying causes, not symptoms.

### Pass 5 — Independent verification

Conduct another audit after implementation.

Do not assume that successful tests prove the system is production-ready.

Verify actual behavior, including provider-side compute allocation and teardown.

### Pass 6 — Further hardening

Repeat the investigation to identify remaining weaknesses, architectural limitations, and opportunities for further improvement.

Continue until additional changes have diminishing returns or require an explicit architectural decision.

---

# 7. TESTING AND ACCEPTANCE CRITERIA

The work is not complete until the following behaviors have been demonstrated.

| Scenario | Expected behavior |
|---|---|
| No requests | Zero provisioned inference compute after idle timeout |
| First request after zero | Infrastructure provisions automatically |
| Model initialization | All required workers become ready |
| Active generation | Required resources remain allocated |
| Multiple concurrent requests | Capacity is managed safely |
| Request completion | Resources become eligible for scale-down |
| Queue becomes empty | Idle shutdown begins |
| Worker failure | Failure is detected and isolated |
| Partial provisioning | Resources are cleaned up |
| Infinite or stuck job | Execution is bounded and terminated |
| Shutdown during activity | In-flight work is handled safely |
| Repeated idle periods | No accumulating orphaned resources |
| Repeated cold starts | System remains reproducible and reliable |

Measure and report:

- Cold-start latency.
- Model initialization time.
- Time to first token.
- Time per output token.
- GPU utilization.
- Peak VRAM consumption.
- Idle resource duration.
- Provisioning and teardown latency.
- Cost per execution.
- Cost during zero-demand periods.

Do not invent benchmark results. Distinguish measured results from estimates.

---

# 8. IMPLEMENTATION PRINCIPLES

These principles govern the entire commission.

**Fixes, not monkey patches.**

**Reliability, not merely successful execution.**

**Cost efficiency, not speculative utilization.**

**Reproducibility, not undocumented tribal knowledge.**

**Clear architecture, not unnecessary abstraction.**

**Operational control, not opaque infrastructure behavior.**

**Established engineering patterns, not reinvented conventions.**

Preserve existing functionality and avoid destructive refactoring.

Do not silently remove working implementations or discard useful experimental work.

Make changes incrementally, verify them, and maintain a clear record of architectural decisions.

If an infrastructure limitation prevents a requirement from being achieved, document the actual limitation and investigate viable alternatives rather than claiming success.

---

# 9. FINAL DELIVERABLE

At completion, provide:

1. A concise overview of the original architectural problems.
2. A record of changes implemented across the workspace.
3. The final scale-to-zero architecture.
4. An explanation of the provisioning and shutdown lifecycle.
5. Demonstrated cost-control behavior.
6. Benchmark comparisons against the original implementation.
7. Testing and failure-injection results.
8. Remaining limitations and known risks.
9. Documentation and navigation improvements.
10. A list of outstanding work requiring architectural decisions.

The final implementation should be understandable, maintainable, resilient, and economical to operate.

## FINAL DIRECTIVE

The system exists to perform inference for actual creative and production workloads.

Infrastructure is a means to that end.

Do not optimize the system into an expensive always-on demonstration of its own capabilities.

**When there is work, provision the compute required to do it. When the work is finished, release the compute. When nobody is using the system, stop paying for it.**

Make the infrastructure serve the production workflow, not the other way around.