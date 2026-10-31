# PLAYBOOK-003: Controlled Benchmarking & Optimal Runtime Experience (ORE) Calibration

**Benchmark Harness**: [`deploy/modal/benchmark.py`](file:///Users/bozoegg/Desktop/HEAV3NS-BRAIN/deploy/modal/benchmark.py)  
**Specification Reference**: `singularity2.md` Section 12

---

## 1. Benchmark Configurations

The benchmark suite evaluates four controlled configurations only after the **Section 12.1 Functional & Correctness Gate** (`run_correctness_gate`) passes 100% deterministic greedy equivalence:

1. **`A_base` (Base V4.1 Engine)**: `dspark=False`, concurrency `1` across context tiers (`short_128`, `medium_4k`, `long_32k`, `max_128k`).
2. **`B_optimized` (DSpark Speculative Verification)**: `dspark=True`, concurrency `1`, verifying speculative speedup and zero quality degradation.
3. **`C_serving` (Multi-Request Serving)**: `dspark=True`, concurrency `8`, measuring continuous batching and prefix cache reuse.
4. **`D_scaling` (Concurrency Scaling Sweep)**: `dspark=True`, concurrency sweep `[1, 2, 4, 8, 16, 32, 64]`.

---

## 2. Running the Benchmark Suite

```bash
export SINGULARITY_MODAL_URL="https://mariecoderinc--singularity-v41-flash-1017-singularityv41-d3907a.us-east.modal.run"

python3 deploy/modal/deploy.py benchmark \
  --url "$SINGULARITY_MODAL_URL" \
  --max-tokens 32 \
  --output /tmp/singularity_b200_benchmark.json
```

---

## 3. Interpreting ORE Telemetry in the Output Artifact

Each entry in `benchmarks[]` contains an `optimal_runtime_experience` block:

```json
{
  "configuration": "B_optimized",
  "context_tier": "short_128",
  "concurrency": 1,
  "ttft_jitter_ratio": 1.0512,
  "itl_jitter_ratio": 1.0844,
  "optimal_runtime_experience": {
    "balance_score": 0.9862,
    "itl_jitter_ratio": 1.0844,
    "ttft_jitter_ratio": 1.0512,
    "completion_rate": 1.0,
    "stability_tier": "optimal"
  }
}
```

### Calibration Targets:
- **`stability_tier == "optimal"`** ($\mathcal{S}_{\text{ORE}} \ge 0.90$): Required for all `concurrency <= 16` tiers before promoting any kernel or scheduler change.
- **`itl_jitter_ratio <= 1.35`**: Ensures human-imperceptible streaming cadence variance.
- **`ttft_jitter_ratio <= 1.50`**: Ensures predictable time-to-first-token across concurrent arrivals.

