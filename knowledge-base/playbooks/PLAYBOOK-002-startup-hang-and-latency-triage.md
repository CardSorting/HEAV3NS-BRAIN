# PLAYBOOK-002: Startup Hang, Gateway Timeout (HTTP 499/502) & Latency Triage

**Applies To**: `SingularityV41Service` (`deploy/modal/app.py`) and `singularity serve` (`singularity-cli`, `singularity-server`)

---

## 1. Symptom Classification Matrix

| Symptom | HTTP Status | Likely Layer | Primary Diagnostic Check |
| :--- | :--- | :--- | :--- |
| Container crashes before `SingularityV41Service.startup()` logs appear | `502` / CrashLoop | Python container import (`PM-001`) | Check `modal app logs singularity-v41-flash-1017` for `IndexError` or `ModuleNotFoundError` |
| `SingularityV41Service.startup()` fails after checkpoint check | `502` / CrashLoop | Checkpoint schema mismatch (`PM-003`) | Run `python3 deploy/modal/deploy.py validate-checkpoint` |
| `POST /v1/chat/completions` hangs $> 60\text{s}$ and logs `ClientDisconnect` | `499` | Full 40-layer CPU reference accidentally enabled (`PM-002`) | Ensure `SINGULARITY_FULL_CPU_REFERENCE` is unset (`0`) in serving containers |
| Immediate rejection under burst traffic | `429` or `503` | Admission protection working as designed | Inspect `/metrics` (`rejected_requests`, `circuit_breaker` state in `/health`) |

---

## 2. Triage Procedure

### Step 1: Inspect `/health` and `/ready`
```bash
curl -sS "$SINGULARITY_MODAL_URL/health" | jq .
curl -sS "$SINGULARITY_MODAL_URL/ready" | jq .
```
Verify all 9 readiness booleans are `true`:
- `container_alive`, `runtime_initialized`, `cuda_operational`, `gpus_available`, `distributed_topology_operational`, `checkpoint_present`, `model_loaded`, `model_ready`, `inference_executed`.

### Step 2: Inspect `/metrics` for ORE Stability & Jitter
```bash
curl -sS "$SINGULARITY_MODAL_URL/metrics" | jq '{
  optimal_runtime_score,
  completion_success_rate,
  ttft_jitter_ratio,
  itl_jitter_ratio,
  p50_ttft_ms,
  p99_ttft_ms,
  p50_itl_ms,
  p99_itl_ms,
  completed_requests,
  aborted_requests,
  rejected_requests
}'
```

### Step 3: Verify Single-Pass Startup Supervision
Confirm `SingularityV41Service.startup()` passes `skip_checkpoint_revalidation=True` to `SingularityRuntimeSupervisor.start()` so the supervisor does not re-scan all 48 SafeTensors shards after `ensure_checkpoint_ready()` has already verified them.

